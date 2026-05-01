#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDPG 版本：在网络攻击环境下训练后车 CAV 的示例脚本

说明：该脚本保持原有环境接口（CyberAttackEnv）的使用方式，仅将训练算法从 PPO 替换为 DDPG。
核心特性：
- Actor (policy) 输出确定性动作，经过 tanh 缩放到动作范围
- Critic 估计 Q(s,a)
- 经验回放缓冲区（ReplayBuffer）用于采样批次训练
- 高斯噪声用于探索（训练阶段），并支持噪声衰减
- 软更新 (soft target update) 用于目标网络
- 支持保存/加载模型

注意：真实工程中建议对 reward 进行标准化、添加归一化器、使用层归一化、以及更完善的超参搜索。

默认保留原来的 env.step 接口（动作格式为列表，前车为 None，后车为数值）
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import pandas as pd  # 添加pandas用于Excel操作
from flexible_rl_test import generate_deceleration_acceleration_speed_sequence,generate_random_oscillating_speed_sequence

max_steps = 600
dt = 0.1
speed_sequence2 = generate_deceleration_acceleration_speed_sequence(max_steps, dt)

# 添加项目根路径（按原脚本风格）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sim_env.envs.cyber_attack_env import CyberAttackEnv

Use_cbf=False


# ----------------------------- 网络定义 -----------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_limit=3.0):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.action_limit = action_limit

    def forward(self, state):
        # state: (batch, state_dim)
        a = self.net(state)
        return a * self.action_limit


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q(s,a) 网络：state 和 action 先各自通过线性层再拼接
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
        )
        self.action_net = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.ELU(),
        )
        self.out_net = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        s = self.state_net(state)
        a = self.action_net(action)
        x = torch.cat([s, a], dim=-1)
        q = self.out_net(x)
        return q


# ----------------------------- Replay Buffer -----------------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # convert to tensors
        states = torch.FloatTensor(np.array([b.state for b in batch]))
        actions = torch.FloatTensor(np.array([b.action for b in batch]))
        rewards = torch.FloatTensor(np.array([b.reward for b in batch])).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([b.next_state for b in batch]))
        dones = torch.FloatTensor(np.array([b.done for b in batch]).astype(float)).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ----------------------------- DDPG Agent -----------------------------
class DDPGAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_limit=3.0,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 gamma=0.99,
                 tau=1e-3,
                 buffer_capacity=100000,
                 batch_size=128,
                 device=None):

        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        print(f"🔧 DDPG Agent 使用设备: {self.device}")  # 添加设备信息打印

        self.actor = Actor(state_dim, action_dim, action_limit).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, action_limit).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)

        # copy params
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.batch_size = batch_size

        # exploration noise (Gaussian)
        self.noise_std = 0.6
        self.min_noise_std = 0.05
        # 基于episode的噪声衰减将在训练中设置

    def get_action(self, state, noise=True):
        # state: 1D numpy array
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, state_dim)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().flatten()
        self.actor.train()
        if noise:
            action = action + np.random.normal(0, self.noise_std, size=action.shape)
        # clip to action limits
        action = np.clip(action, -self.actor.action_limit, self.actor.action_limit)
        if action.size == 1:
            return float(action[0])
        return action

    def push_transition(self, state, action, reward, next_state, done):
        # ensure numpy arrays
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = np.array([action], dtype=np.float32) if np.isscalar(action) else np.array(action, dtype=np.float32)
        self.replay_buffer.push(state, action, float(reward), next_state, float(done))

    def soft_update(self, source_net, target_net):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_next = self.critic_target(next_states, next_actions)
            q_target = rewards + (1.0 - dones) * self.gamma * q_next

        q_current = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_current, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Actor update (policy gradient)
        pred_actions = self.actor(states)
        actor_loss = -self.critic(states, pred_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # soft update targets
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item()}

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, 'ddpg_actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(path, 'ddpg_critic.pth'))

    def load(self, path):
        actor_p = os.path.join(path, 'ddpg_actor.pth')
        critic_p = os.path.join(path, 'ddpg_critic.pth')
        if os.path.isfile(actor_p):
            self.actor.load_state_dict(torch.load(actor_p, map_location=self.device))
            self.actor_target.load_state_dict(self.actor.state_dict())
        if os.path.isfile(critic_p):
            self.critic.load_state_dict(torch.load(critic_p, map_location=self.device))
            self.critic_target.load_state_dict(self.critic.state_dict())


# ----------------------------- 环境创建函数 -----------------------------
def create_env(enable_cyber_attack=True, attack_type="data_tampering"):
    env = CyberAttackEnv(
        num_vehicles=6,
        cav_indices=[1, 5],
        dt=0.1,
        enable_cyber_attack=enable_cyber_attack,
        attack_frequency=0.1,
        attack_type=attack_type,
        attack_targets=["speed", "acceleration"],
        attack_variances={"speed": 5.0, "acceleration": 2.0},
        use_cbf=Use_cbf,
        filter_alpha=1,
    )
    return env

# ----------------------------- 测试函数 -----------------------------
def test_agent(agent, device, use_cbf=True, test_episodes=5):
    """测试智能体性能"""
    print("🧪 正在测试智能体性能...")
    test_env = create_env(enable_cyber_attack=True, attack_type="data_tampering")
    
    test_scores = []
    for episode in range(test_episodes):
        test=random.random()
        if test<0.2:
            speed_sequence1 = generate_random_oscillating_speed_sequence(max_steps, dt)
            state, _ = test_env.reset(lead_speed_sequence=random.choice([speed_sequence1, speed_sequence2]))
        else:
             state, _ = test_env.reset()
        
        episode_reward = 0.0
        rear_obs = state[1]  # 后车 CAV 的观测
        
        max_steps = 500
        for step in range(max_steps):
            # 测试阶段：不添加噪声
            action_rear = agent.get_action(rear_obs, noise=False)
            action = [None, action_rear]
            next_state, reward, terminated, truncated, info = test_env.step(action)
            
            next_rear_obs = next_state[1]
            rear_obs = next_rear_obs
            episode_reward += float(reward)
            
            if terminated or truncated:
                break
        
        test_scores.append(episode_reward)
        print(f"   测试回合 {episode+1}/{test_episodes}, 得分: {episode_reward:.2f}")
    
    avg_score = np.mean(test_scores)
    std_score = np.std(test_scores)
    print(f"   平均得分: {avg_score:.2f} ± {std_score:.2f}")
    return avg_score, std_score

# ----------------------------- 保存scores到CSV -----------------------------
def save_scores_to_csv(scores, filename):
    """将scores结果保存到CSV文件"""
    # 创建DataFrame
    df = pd.DataFrame({
        'episode': range(1, len(scores) + 1),
        'score': scores
    })
    
    # 确保输出目录存在
    output_dir = os.path.join(_PROJECT_ROOT, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存到CSV文件
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"💾 scores结果已保存到: {filepath}")
    return filepath

# ----------------------------- 保存测试结果到Excel -----------------------------
def save_test_results_to_excel(results, filename):
    """将测试结果保存到Excel文件"""
    df = pd.DataFrame(results)
    
    # 确保输出目录存在
    output_dir = os.path.join(_PROJECT_ROOT, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存到Excel文件
    filepath = os.path.join(output_dir, filename)
    df.to_excel(filepath, index=False)
    print(f"💾 测试结果已保存到: {filepath}")
    return filepath

# ----------------------------- 主训练循环 -----------------------------
def main():
    print("🚀 使用 DDPG 训练后车 CAV（混合控制）...")
    env = create_env(enable_cyber_attack=True, attack_type="data_tampering")

    state, _ = env.reset()
    state_dim = state.shape[1]
    action_dim = 1

    agent = DDPGAgent(state_dim=state_dim,
                      action_dim=action_dim,
                      action_limit=3.0,
                      actor_lr=1e-4,
                      critic_lr=1e-3,
                      gamma=0.99,
                      tau=5e-3,
                      buffer_capacity=50000,
                      batch_size=128,
                      )

    # 设置基于回合的训练参数
    max_episodes = 300  # 总训练回合数
    max_steps_per_episode = 500  # 每个episode的最大步数
    
    # 基于回合数计算噪声衰减率
    agent.noise_decay_episodes = 100  # 在前100个回合中衰减噪声
    agent.noise_decay_per_episode = (agent.noise_std - agent.min_noise_std) / agent.noise_decay_episodes

    scores = []
    stats_log = []
    best_score = -np.inf

    # 初始化测试结果列表
    test_results = []

    for episode in range(max_episodes):
        a=random.randint(3,6)
        reset_options = {
        "cav_indices": [1,a] ,
        'num_vehicles': a+1,
    }

        state, _ = env.reset(options=reset_options)
        episode_reward = 0.0

        #前车观测
        # 后车 CAV 的观测
        rear_obs = state[1]


        for step in range(max_steps_per_episode):
            # 训练阶段：带噪声探索
            # action_front = agent.get_action(front_obs, noise=True)
            action_rear = agent.get_action(rear_obs, noise=True)


            action = [None, action_rear]
            next_state, reward, terminated, truncated, info = env.step(action)

            next_rear_obs = next_state[1]

            done = bool(terminated or truncated)

            # 将 transition 放入回放池（用后车的观测作为 state）
            agent.push_transition(rear_obs, action_rear, reward[1], next_rear_obs, done)
            # agent.push_transition(front_obs, action_front, reward[0], next_front_obs, done)
            reward = reward[1]

            # 学习
            train_info = agent.train_step()

            rear_obs = next_rear_obs
            episode_reward += float(reward)

            if done:
                break

        scores.append(episode_reward)
        env.plot_timeseries()
        
        # 打印回合的jerk均值（如果环境有此功能）

        # 基于回合的噪声衰减（在前noise_decay_episodes个回合中衰减）
        if episode < agent.noise_decay_episodes:
            agent.noise_std = max(agent.min_noise_std, agent.noise_std - agent.noise_decay_per_episode)

        # 打印与日志（每完成5个episode时打印）
        if (episode + 1) % 5 == 0:
            avg_score = float(np.mean(scores[-5:])) if len(scores) >= 5 else float(np.mean(scores))
            print(f"Episode {episode+1}/{max_episodes} | AvgScore(5): {avg_score:.2f} | ReplaySize: {len(agent.replay_buffer)} | NoiseStd: {agent.noise_std:.3f}")
            if env.enable_cyber_attack:
                attack_stats = env.get_cyber_attack_stats()
                if attack_stats.get("enabled", False):
                    stats = attack_stats["statistics"]
                    print(f"   攻击统计: 总攻击次数={stats['total_attacks']}, 实际攻击率={stats['actual_attack_rate']:.2f}")

        # 保存中间模型（每完成50个episode时保存）
        if (episode + 1) % 50 == 0:
            model_dir = os.path.join(_PROJECT_ROOT, 'models', f'ddpg_episode{episode+1}')
            agent.save(model_dir)

        # 保存最佳模型
        if len(scores) >= 5:
            avg_score = float(np.mean(scores[-5:]))
            if avg_score > best_score:
                best_score = avg_score
                model_dir = os.path.join(_PROJECT_ROOT, "models")
                os.makedirs(model_dir, exist_ok=True)
                best_model_path = os.path.join(model_dir, "ddpg_mixed_cav_agent_best.pth")
                agent.save(best_model_path)
                print(f"   🏆 新的最佳模型已保存，平均得分: {avg_score:.2f}")

    # 训练结束，保存该配置的测试结果到Excel
    if Use_cbf:
        suffix='cbf'
    else:
        suffix='no_cbf'
    
    if test_results:
        save_test_results_to_excel(test_results, f"ddpg_training_test_results_{suffix}.xlsx")
        
    # 保存scores结果到CSV
    if scores:
        save_scores_to_csv(scores, f"ddpg_training_scores_{suffix}.csv")

    # 保存最终模型
    final_dir = os.path.join(_PROJECT_ROOT, 'models', f'ddpg_final_{suffix}')
    agent.save(final_dir)
    print(f"💾 最终模型已保存到: {final_dir}")
    print("🎉 训练完成!")


if __name__ == '__main__':
    main()