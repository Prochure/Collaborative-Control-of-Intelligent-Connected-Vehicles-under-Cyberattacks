#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本文件
该脚本允许在相同的环境配置下测试不同的强化学习模型。
环境配置保持不变，只需要切换算法即可。
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import csv
import pandas as pd

# 使用英文字体和数学字符
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 全部英文字体
plt.rcParams['font.family'] = 'serif'


# 添加项目路径
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sim_env.envs.cyber_attack_env import CyberAttackEnv

# 定义算法相关类（用于评估模式）

class Actor(nn.Module):
    """DDPG Actor网络"""
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

def create_env(enable_cyber_attack=True, attack_type="data_tampering", attack_frequency=0.2):
    """
    创建环境（环境配置保持不变）
    
    参数:
        enable_cyber_attack: 是否启用网络攻击
        attack_type: 攻击类型 ("data_tampering", "packet_drop", "delay")
        attack_frequency: 攻击频率
    """
    env = CyberAttackEnv(
        num_vehicles=7,  # 总共6辆车
        cav_indices=[1,3,6],  # 第二辆和第六辆是CAV（索引从0开始）
        dt=0.1,
        enable_cyber_attack=enable_cyber_attack,
        attack_frequency=attack_frequency,
        attack_type='data_tampering',
        attack_targets=["speed", "acceleration",'position'],
        attack_variances={"speed": 3.0, "acceleration": 1.0, "position": 200.0},  # 独立方差配置
        use_cbf=False,
        filter_alpha=0.1,
        force_lead_cav_p_one=True,
        attack_start_time=5,
        attack_means={"speed": 6.0, "acceleration": 1.50, "position": 5.0},
    )
    return env

def load_model(algorithm, model_path, state_dim, action_dim):
    """
    根据算法类型加载训练好的模型
    
    参数:
        algorithm: 算法类型 ("ddpg", "ppo", "sac", "td3")
        model_path: 模型文件路径
        state_dim: 状态维度
        action_dim: 动作维度
    
    返回:
        model: 加载的模型
        device: 设备
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if algorithm == "ddpg":
        model = Actor(state_dim, action_dim, action_limit=3.0).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
    elif algorithm == "ppo":
        model = ActorCritic(state_dim, action_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
    elif algorithm == "sac":
        model = GaussianPolicyForEval(state_dim, action_dim, action_limit=3.0).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
    elif algorithm == "td3":
        model = TD3Actor(state_dim, action_dim, action_limit=2.0).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
    else:
        raise ValueError(f"不支持的算法类型: {algorithm}")
    
    return model, device

def select_action(algorithm, model, device, state):
    """
    根据算法类型选择动作
    
    参数:
        algorithm: 算法类型
        model: 模型
        device: 设备
        state: 状态
    
    返回:
        action: 选择的动作
    """
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    if algorithm == "ddpg":
        action = model(state)
        action = action.detach().cpu().numpy().flatten()
        return action[0]  # 只返回动作
    elif algorithm == "ppo":
        action_mean, _ = model(state)
        action = action_mean.detach().cpu().numpy().flatten()
        return action[0]  # 只返回动作
    elif algorithm == "sac":
        with torch.no_grad():
            action_t = model.act_deterministic(state)
        action = action_t.detach().cpu().numpy().flatten()
        # 裁剪以防止数值溢出
        action = np.clip(action, -model.action_limit, model.action_limit)
        return float(action[0])
    elif algorithm == "td3":
        with torch.no_grad():
            action = model(state)
        action = action.detach().cpu().numpy().flatten()
        # 裁剪以防止数值溢出
        action = np.clip(action, -model.action_limit, model.action_limit)
        return float(action[0])
    else:
        raise ValueError(f"不支持的算法类型: {algorithm}")

def save_trajectory_to_csv(trajectory_data, dt, algorithm="rl", scenario_name="default", output_dir=None):
    """
    保存轨迹数据到CSV文件，包括位置、速度、加速度
    
    参数:
        trajectory_data: 轨迹数据列表，每个元素包含时间步的状态信息
        dt: 时间步长
        algorithm: 算法名称
        scenario_name: 场景名称
        output_dir: 输出目录，默认为examples/outputs
    
    返回:
        csv_file_path: 保存的CSV文件路径
    """
    if not trajectory_data:
        print("❌ 没有轨迹数据可保存")
        return None
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(_THIS_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 按车辆ID组织数据
    vehicle_data = {}
    
    # 首先收集所有车辆ID
    sample_state = trajectory_data[0]["state"]
    for vehicle_state in sample_state:
        vid = vehicle_state["id"]
        vehicle_data[vid] = {
            "time": [],
            "step": [],
            "position": [],
            "speed": [],
            "acceleration": [],
            "vehicle_type": vehicle_state["type"]
        }
    
    # 填充轨迹数据
    for data in trajectory_data:
        time = data["time"]
        step = time / dt
        for vehicle_state in data["state"]:
            vid = vehicle_state["id"]
            vehicle_data[vid]["time"].append(time)
            vehicle_data[vid]["step"].append(step)
            vehicle_data[vid]["position"].append(vehicle_state["x"])
            vehicle_data[vid]["speed"].append(vehicle_state["v"])
            vehicle_data[vid]["acceleration"].append(vehicle_state["a"])
    
    # 同时保存一个汇总的CSV文件，包含所有车辆
    all_data = []
    for vid, data in vehicle_data.items():
        vehicle_type = data["vehicle_type"]
        for i in range(len(data["time"])):
            all_data.append({
                "VehicleID": vid,
                "VehicleType": vehicle_type,
                "Time(s)": data["time"][i],
                "Step": data["step"][i],
                "Position(m)": data["position"][i],
                "Speed(m/s)": data["speed"][i],
                "Acceleration(m/s2)": data["acceleration"][i]
            })
    
    # 创建汇总DataFrame
    df_all = pd.DataFrame(all_data)
    
    # 生成汇总文件名
    summary_csv_filename = f"{algorithm}_{scenario_name}_all_vehicles_trajectory.csv"
    summary_csv_path = os.path.join(output_dir, summary_csv_filename)
    
    # 保存汇总CSV
    df_all.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存所有车辆汇总轨迹数据到: {summary_csv_filename}")
    
    return summary_csv_path


def test_model(algorithm, model_path, enable_attack=True, attack_type="data_tampering"):
    """
    测试指定算法的模型
    
    参数:
        algorithm: 算法类型 ("ddpg", "ppo", "sac")
        model_path: 模型文件路径
        enable_attack: 是否启用网络攻击
        attack_type: 攻击类型
    """
    print(f"🚀 开始 {algorithm.upper()} 模型测试...")
    print("=" * 50)
    
    # 生成三种随机工况的加速度序列
    max_steps = 600
    dt = 0.1

    # 存储所有工况的轨迹数据
    all_trajectory_data = []
    all_predicted_hv_data = []
    
    # 对每种工况分别进行测试
    for i, acc_sequence in enumerate([1,2,3]):
        print(f"📊 测试工况 {i+1}...")
        
        # 创建环境（环境配置保持不变）
        env = create_env(enable_cyber_attack=enable_attack, attack_type=attack_type)
        
        # 重置环境，传入加速度序列

        state, _ = env.reset()
        
        # 环境参数
        state_dim = state.shape[1]  # 每个CAV的观测维度（3维浓缩观测）
        action_dim = 1  # 每个CAV的动作维度（加速度）
        
        if i == 0:  # 只在第一次打印环境配置
            print(f"🚗 环境配置:")
            print(f"   - 车辆总数: {env.num_vehicles}")
            print(f"   - CAV数量: {len(env.cav_ids)}")
            print(f"   - CAV索引: {env.cav_indices}")
            print(f"   - 状态维度: {state_dim}")
            print(f"   - 动作维度: {action_dim}")
            print(f"   - 网络攻击: {'启用' if enable_attack else '禁用'}")
            if enable_attack:
                print(f"   - 攻击类型: {attack_type}")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            print(f"请确保模型文件路径正确")
            return
        
        # 加载模型（只在第一次加载）
        if i == 0:
            try:
                model, device = load_model(algorithm, model_path, state_dim, action_dim)
                print(f"✅ {algorithm.upper()} 模型加载成功: {model_path}")
            except Exception as e:
                print(f"❌ {algorithm.upper()} 模型加载失败: {str(e)}")
                return
        # 运行仿真并收集轨迹数据
        trajectory_data = []
        attack_data = []  # 存储攻击数据
        predicted_hv_data = []  # 存储预测的HV数据
        rewards = []  # 存储奖励数据
        
        for step in range(max_steps):
            # 获取所有CAV的观测值
            cav_observations = state  # shape: (num_cavs, state_dim)
            
            # 初始化动作数组，全部设为None
            action = [None] * len(env.cav_ids)
            
            # 第一辆CAV（索引为0）使用IDM模型（动作设为None，已经在初始化时设定）
            # 其他CAV使用强化学习选择动作
            for j in range(1, len(env.cav_ids)):  # 从第二个CAV开始
                cav_obs = cav_observations[j]
                rl_action = select_action(algorithm, model, device, cav_obs)
                action[j] = rl_action
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # 保存奖励（使用第一个CAV的奖励）
            rewards.append(reward[0] if isinstance(reward, (list, tuple)) else reward)
            
            # 保存当前状态数据
            step_data = {
                "time": env.sim.t,
                "state": env.sim.get_state(),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": info
            }
            trajectory_data.append(step_data)
            
            # 保存攻击数据
            if env.enable_cyber_attack and "cyber_attack" in info:
                attack_info = info["cyber_attack"]
                if attack_info.get("current_step_attacks"):
                    attack_data.append({
                        "time": env.sim.t,
                        "attacks": attack_info["current_step_attacks"]
                    })
            
            # 保存预测的HV数据
            predicted_hv_states = env.get_predicted_hv_states()
            predicted_hv_data.append({
                "time": env.sim.t,
                "predicted_hv_states": predicted_hv_states.copy() if predicted_hv_states else {}
            })
            
            # 更新状态
            state = next_state
            
            # 检查是否终止
            if terminated or truncated:
                print(f"⚠️  仿真在第 {step} 步结束")
                break
        
        print(f"🏁 {algorithm.upper()} 工况 {i+1} 仿真完成，运行了 {len(trajectory_data)} 步")
        try:
            current_jerk_mean = env.get_current_episode_jerk_mean()
            print(f"   回合 的平均jerk值: {current_jerk_mean:.6f}")
        except AttributeError:
            # 如果环境没有这个方法，则跳过
            pass
        # 保存轨迹数据
        all_trajectory_data.append(trajectory_data)
        all_predicted_hv_data.append(predicted_hv_data)
    
    # 绘制轨迹图（3x3网格图）
    plot_two_condition_trajectories(all_trajectory_data, dt, algorithm)
    
    # 计算跟驰性能指标（使用第一种工况）
    calculate_following_performance(all_trajectory_data[0], algorithm)
    
    # 显示攻击统计信息
    # 注意：这里使用最后一次测试的环境来获取攻击统计信息
    if env.enable_cyber_attack:
        attack_stats = env.get_cyber_attack_stats()
        if attack_stats["enabled"]:
            stats = attack_stats["statistics"]
            print(f"\n🛡️  {algorithm.upper()} 网络攻击统计:")
            print(f"   - 总攻击次数: {stats['total_attacks']}")
            print(f"   - 实际攻击率: {stats['actual_attack_rate']:.2f}")
            print(f"   - 攻击目标分布: {stats['target_distribution']}")
    
    # 显示奖励统计信息（使用第一种工况）
    if rewards:
        total_reward = sum(rewards)
        avg_reward = np.mean(rewards)
        max_reward = max(rewards)
        min_reward = min(rewards)
        print(f"\n💰 {algorithm.upper()} 奖励统计:")
        print(f"   - 总奖励: {total_reward:.2f}")
        print(f"   - 平均奖励: {avg_reward:.2f}")
        print(f"   - 最大奖励: {max_reward:.2f}")
        print(f"   - 最小奖励: {min_reward:.2f}")
    
    # 计算并保存无量纲性能指标
    calculate_and_save_dimensionless_metrics(all_trajectory_data, algorithm, dt)
    
    # 保存轨迹数据到CSV文件
    print(f"\n📊 保存轨迹数据到CSV文件...")
    for i, trajectory_data in enumerate(all_trajectory_data):
        scenario_name = f"scenario_{i+1}"
        save_trajectory_to_csv(trajectory_data, dt, algorithm, scenario_name)
    
    print(f"\n🎉 {algorithm.upper()} 测试完成!")

def generate_random_oscillating_speed_sequence(total_steps, dt):
    """
    生成随机震荡速度序列，具有规则性且加速度变化平滑
    
    参数:
        total_steps: 总步数
        dt: 时间步长
    
    返回:
        speed_sequence: 随机震荡速度序列 (m/s)
    """
    t = np.arange(total_steps) * dt
    
    # 基础速度 (15 m/s) + 多频率振荡分量（确保加速度平滑）
    speed_sequence = 15.0 + (
        2.0 * np.sin(0.3 * t) +      # 低频振荡（周期约21秒）
        1.2 * np.sin(0.8 * t) +      # 中频振荡（周期约8秒）
        0.6 * np.sin(1.5 * t) +      # 较高频振荡（周期约4秒）
        0.3 * np.sin(3.0 * t)        # 高频振荡（周期约2秒）
    )
    
    # 添加平滑的随机变化分量
    # 生成低频随机噪声，确保加速度变化平滑
    np.random.seed(42)  # 固定种子确保可重现性
    random_component = np.zeros(total_steps)
    
    # 每隔50步生成一个随机值，然后线性插值实现平滑过渡
    segment_length = 50
    num_segments = total_steps // segment_length + 1
    
    random_values = np.random.normal(0, 0.3, num_segments)  # 降低随机噪声幅度
    
    for i in range(num_segments - 1):
        start_idx = i * segment_length
        end_idx = min((i + 1) * segment_length, total_steps)
        
        # 线性插值实现平滑过渡
        interpolation = np.linspace(random_values[i], random_values[i + 1], end_idx - start_idx)
        random_component[start_idx:end_idx] = interpolation
    
    speed_sequence += random_component
    
    # 限制速度范围在合理区间内 (8-22 m/s)
    speed_sequence = np.clip(speed_sequence, 8.0, 22.0)
    
    return speed_sequence

def generate_deceleration_acceleration_speed_sequence(total_steps, dt):
    """
    生成先减速然后匀速，然后加速的速度序列
    
    参数:
        total_steps: 总步数
        dt: 时间步长
    
    返回:
        speed_sequence: 速度序列 (m/s)
    """
    speed_sequence = np.full(total_steps, 18.0)  # 默认速度 18 m/s
    
    # 第一阶段：减速 (前100步)
    deceleration_steps = 60
    # 从18 m/s减速到8 m/s
    speed_sequence[:deceleration_steps] = np.linspace(18.0, 5.0, deceleration_steps)
    
    # 第二阶段：匀速 (接下来的200步)
    constant_speed_steps = 200
    start_constant = deceleration_steps
    # 保持8 m/s
    speed_sequence[start_constant:start_constant+constant_speed_steps] = 5.0
    
    # 第三阶段：加速 (接下来的100步)
    acceleration_steps = 60
    start_acceleration = start_constant + constant_speed_steps
    # 从8 m/s加速到18 m/s
    speed_sequence[start_acceleration:start_acceleration+acceleration_steps] = np.linspace(5.0, 18.0, acceleration_steps)
    
    # 剩余时间保持18 m/s
    speed_sequence[start_acceleration+acceleration_steps:] = 18.0
    
    return speed_sequence

def test_model_two_conditions(algorithm, model_path, enable_attack=True, attack_type="data_tampering"):
    """
    测试指定算法的模型，使用两种特定工况
    
    参数:
        algorithm: 算法类型 ("ddpg", "ppo", "sac")
        model_path: 模型文件路径
        enable_attack: 是否启用网络攻击
        attack_type: 攻击类型
    """
    print(f"🚀 开始 {algorithm.upper()} 模型测试（两种工况）...")
    print("=" * 50)
    
    # 生成两种特定工况的速度序列
    max_steps = 600
    dt = 0.1
    
    # 第一种情况：随机震荡速度序列
    speed_sequence1 = generate_random_oscillating_speed_sequence(max_steps, dt)
    
    # 第二种情况：先减速然后匀速，然后加速的速度序列
    speed_sequence2 = generate_deceleration_acceleration_speed_sequence(max_steps, dt)
    
    # 存储所有工况的轨迹数据
    all_trajectory_data = []
    all_predicted_hv_data = []
    speed_sequences = [speed_sequence1, speed_sequence2]
    
    # 对每种工况分别进行测试
    for i, speed_sequence in enumerate(speed_sequences):
        print(f"📊 测试工况 {i+1}...")
        
        # 创建环境（环境配置保持不变）
        env = create_env(enable_cyber_attack=enable_attack, attack_type=attack_type)
        
        # 重置环境，传入速度序列
        reset_options = {
            "lead_speed_sequence": speed_sequence  # 传入当前工况的速度序列
        }
        state, _ = env.reset(options=reset_options)
        
        # 环境参数
        state_dim = state.shape[1]  # 每个CAV的观测维度（3维浓缩观测）
        action_dim = 1  # 每个CAV的动作维度（加速度）
        
        if i == 0:  # 只在第一次打印环境配置
            print(f"🚗 环境配置:")
            print(f"   - 车辆总数: {env.num_vehicles}")
            print(f"   - CAV数量: {len(env.cav_ids)}")
            print(f"   - CAV索引: {env.cav_indices}")
            print(f"   - 状态维度: {state_dim}")
            print(f"   - 动作维度: {action_dim}")
            print(f"   - 序列长度: {len(speed_sequence)}")
            print(f"   - 网络攻击: {'启用' if enable_attack else '禁用'}")
            if enable_attack:
                print(f"   - 攻击类型: {attack_type}")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            print(f"请确保模型文件路径正确")
            return
        
        # 加载模型（只在第一次加载）
        if i == 0:
            try:
                model, device = load_model(algorithm, model_path, state_dim, action_dim)
                print(f"✅ {algorithm.upper()} 模型加载成功: {model_path}")
            except Exception as e:
                print(f"❌ {algorithm.upper()} 模型加载失败: {str(e)}")
                return
        
        # 运行仿真并收集轨迹数据
        trajectory_data = []
        attack_data = []  # 存储攻击数据
        predicted_hv_data = []  # 存储预测的HV数据
        rewards = []  # 存储奖励数据
        
        for step in range(max_steps):
            # 获取所有CAV的观测值
            cav_observations = state  # shape: (num_cavs, state_dim)
            
            # 初始化动作数组，全部设为None
            action = [None] * len(env.cav_ids)
            
            # 第一辆CAV（索引为0）使用IDM模型（动作设为None，已经在初始化时设定）
            # 其他CAV使用强化学习选择动作
            for j in range(1, len(env.cav_ids)):  # 从第二个CAV开始
                cav_obs = cav_observations[j]
                rl_action = select_action(algorithm, model, device, cav_obs)
                action[j] = rl_action
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # 保存奖励（使用第一个CAV的奖励）
            rewards.append(reward[0] if isinstance(reward, (list, tuple)) else reward)
            
            # 保存当前状态数据
            step_data = {
                "time": env.sim.t,
                "state": env.sim.get_state(),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": info
            }
            trajectory_data.append(step_data)
            
            # 保存攻击数据
            if env.enable_cyber_attack and "cyber_attack" in info:
                attack_info = info["cyber_attack"]
                if attack_info.get("current_step_attacks"):
                    attack_data.append({
                        "time": env.sim.t,
                        "attacks": attack_info["current_step_attacks"]
                    })
            
            # 保存预测的HV数据
            predicted_hv_states = env.get_predicted_hv_states()
            predicted_hv_data.append({
                "time": env.sim.t,
                "predicted_hv_states": predicted_hv_states.copy() if predicted_hv_states else {}
            })
            
            # 更新状态
            state = next_state
            
            # 检查是否终止
            if terminated or truncated:
                print(f"⚠️  仿真在第 {step} 步结束")
                break
        
        print(f"🏁 {algorithm.upper()} 工况 {i+1} 仿真完成，运行了 {len(trajectory_data)} 步")
        
        # 保存轨迹数据
        all_trajectory_data.append(trajectory_data)
        all_predicted_hv_data.append(predicted_hv_data)
    
    # 绘制轨迹图（2x3网格图）
    plot_two_condition_trajectories(all_trajectory_data, dt,  algorithm )
    
    # 计算跟驰性能指标（使用第一种工况）
    calculate_following_performance(all_trajectory_data[0], algorithm)
    
    # 显示攻击统计信息
    # 注意：这里使用最后一次测试的环境来获取攻击统计信息
    if env.enable_cyber_attack:
        attack_stats = env.get_cyber_attack_stats()
        if attack_stats["enabled"]:
            stats = attack_stats["statistics"]
            print(f"\n🛡️  {algorithm.upper()} 网络攻击统计:")
            print(f"   - 总攻击次数: {stats['total_attacks']}")
            print(f"   - 实际攻击率: {stats['actual_attack_rate']:.2f}")
            print(f"   - 攻击目标分布: {stats['target_distribution']}")
    
    # 显示奖励统计信息（使用第一种工况）
    if rewards:
        total_reward = sum(rewards)
        avg_reward = np.mean(rewards)
        max_reward = max(rewards)
        min_reward = min(rewards)
        print(f"\n💰 {algorithm.upper()} 奖励统计:")
        print(f"   - 总奖励: {total_reward:.2f}")
        print(f"   - 平均奖励: {avg_reward:.2f}")
        print(f"   - 最大奖励: {max_reward:.2f}")
        print(f"   - 最小奖励: {min_reward:.2f}")
    
    # 计算并保存无量纲性能指标
    calculate_and_save_dimensionless_metrics(all_trajectory_data, algorithm, dt)
    
    # 保存轨迹数据到CSV文件
    print(f"\n📊 保存轨迹数据到CSV文件...")
    scenario_names = ["random_oscillating", "decel_accel"]
    for i, trajectory_data in enumerate(all_trajectory_data):
        scenario_name = scenario_names[i] if i < len(scenario_names) else f"scenario_{i+1}"
        save_trajectory_to_csv(trajectory_data, dt, algorithm, scenario_name)
    
    print(f"\n🎉 {algorithm.upper()} 测试完成!")

def test_model_attack_frequencies(algorithm, model_path, attack_type="data_tampering"):
    """
    测试指定算法的模型在不同网络攻击频率下的表现
    
    参数:
        algorithm: 算法类型 ("ddpg", "ppo", "sac")
        model_path: 模型文件路径
        attack_type: 攻击类型
    """
    print(f"🚀 开始 {algorithm.upper()} 模型测试（不同攻击频率）...")
    print("=" * 50)
    
    max_steps = 600
    dt = 0.1
    
    # 使用随机震荡速度序列
    speed_sequence = generate_random_oscillating_speed_sequence(max_steps, dt)
    
    # 测试三种不同的攻击频率
    attack_frequencies = [0.1, 0.3, 0.5]
    all_trajectory_data = []
    
    for i, freq in enumerate(attack_frequencies):
        print(f"📊 测试攻击频率 {freq}...")
        
        # 创建环境，使用不同的攻击频率
        env = create_env(enable_cyber_attack=True, attack_type=attack_type, attack_frequency=freq)
        
        # 重置环境，传入速度序列
        reset_options = {
            "lead_speed_sequence": speed_sequence
        }
        state, _ = env.reset(options=reset_options)
        
        # 环境参数
        state_dim = state.shape[1]
        action_dim = 1
        
        if i == 0:
            print(f"🚗 环境配置:")
            print(f"   - 车辆总数: {env.num_vehicles}")
            print(f"   - CAV数量: {len(env.cav_ids)}")
            print(f"   - CAV索引: {env.cav_indices}")
            print(f"   - 状态维度: {state_dim}")
            print(f"   - 动作维度: {action_dim}")
            print(f"   - 序列长度: {len(speed_sequence)}")
            print(f"   - 攻击类型: {attack_type}")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            print(f"请确保模型文件路径正确")
            return
        
        # 加载模型（只在第一次加载）
        if i == 0:
            try:
                model, device = load_model(algorithm, model_path, state_dim, action_dim)
                print(f"✅ {algorithm.upper()} 模型加载成功: {model_path}")
            except Exception as e:
                print(f"❌ {algorithm.upper()} 模型加载失败: {str(e)}")
                return
        
        # 运行仿真并收集轨迹数据
        trajectory_data = []
        
        for step in range(max_steps):
            cav_observations = state
            action = [None] * len(env.cav_ids)
            
            for j in range(1, len(env.cav_ids)):
                cav_obs = cav_observations[j]
                rl_action = select_action(algorithm, model, device, cav_obs)
                action[j] = rl_action
            
            next_state, reward, terminated, truncated, info = env.step(action)
            
            step_data = {
                "time": env.sim.t,
                "state": env.sim.get_state(),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": info
            }
            trajectory_data.append(step_data)
            
            state = next_state
            
            if terminated or truncated:
                print(f"⚠️  仿真在第 {step} 步结束")
                break
        
        print(f"🏁 攻击频率 {freq} 仿真完成，运行了 {len(trajectory_data)} 步")
        a=env.get_cav_safety_metrics()
        # 显示攻击统计信息
        if env.enable_cyber_attack:
            attack_stats = env.get_cyber_attack_stats()
            if attack_stats["enabled"]:
                stats = attack_stats["statistics"]
                print(f"   - 总攻击次数: {stats['total_attacks']}")
                print(f"   - 实际攻击率: {stats['actual_attack_rate']:.2f}")
        
        all_trajectory_data.append(trajectory_data)
    
    # 计算并保存无量纲性能指标
    scenario_names = [f"Attack Freq {f}" for f in attack_frequencies]
    calculate_and_save_dimensionless_metrics(all_trajectory_data, algorithm, dt, scenario_names=scenario_names)
    
    # 保存轨迹数据到CSV文件
    print(f"\n📊 保存轨迹数据到CSV文件...")
    for i, trajectory_data in enumerate(all_trajectory_data):
        scenario_name = f"attack_freq_{attack_frequencies[i]}"
        save_trajectory_to_csv(trajectory_data, dt, algorithm, scenario_name)
    
    print(f"\n🎉 {algorithm.upper()} 攻击频率测试完成!")

def plot_two_condition_trajectories(all_trajectory_data, dt, algorithm="rl"):
    """
    绘制2x3网格图：每行位置/速度/加速度，对比两种工况
    """

    def _apply_publication_style():
        """匹配 test_attack_frequency_effect.py 中的出版级绘图风格"""
        plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })

    _apply_publication_style()

    if not all_trajectory_data or len(all_trajectory_data) < 2:
        print("❌ 没有足够的轨迹数据可绘制")
        return

    # 创建2x3子图，整体更紧凑
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.4))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.05,
                        hspace=0.30, wspace=0.18)

    # 定义颜色，与 test_attack_frequency_effect.py 保持一致
    hv_color = '#d62728'
    cav_color = '#1f77b4'
    lead_color = '#2ca02c'

    # 遍历两种工况
    for condition_idx in range(2):

        trajectory_data = all_trajectory_data[condition_idx]

        # 时间步（Step）
        time_series = [data["time"] for data in trajectory_data]
        step_series = np.array(time_series) / dt   # 将秒→step

        # 按车辆ID组织数据
        vehicle_data = {}
        sample_state = trajectory_data[0]["state"]

        for vehicle_state in sample_state:
            vid = vehicle_state["id"]
            vehicle_data[vid] = {
                "step": [],
                "position": [],
                "speed": [],
                "acceleration": [],
                "is_cav": vehicle_state["type"] == "CAV"
            }

        # 填充轨迹数据
        for data in trajectory_data:
            step = data["time"] / dt
            for vehicle_state in data["state"]:
                vid = vehicle_state["id"]
                vehicle_data[vid]["step"].append(step)
                vehicle_data[vid]["position"].append(vehicle_state["x"])
                vehicle_data[vid]["speed"].append(vehicle_state["v"])
                vehicle_data[vid]["acceleration"].append(vehicle_state["a"])

        # 计算加加速度（jerk）
        for vid, data in vehicle_data.items():
            accelerations = np.array(data["acceleration"])
            if len(accelerations) > 1:
                # 计算加加速度，使用差分方法
                jerk = np.diff(accelerations) / dt  # 加速度变化率
                # 为了保持数组长度一致，我们在开头添加一个0
                jerk = np.concatenate([[0], jerk])
                data["jerk"] = jerk.tolist()
            else:
                data["jerk"] = [0] * len(accelerations)

        # 获取对应的子图
        ax_pos = axes[condition_idx, 0]
        ax_vel = axes[condition_idx, 1]
        ax_acc = axes[condition_idx, 2]
        # ax_jerk = axes[condition_idx, 3]  # 新增加加速度子图

        # 绘制车辆曲线
        for vid, data in vehicle_data.items():
            color = cav_color if data['is_cav'] else hv_color
            line_style = '--' if data['is_cav'] else '-'

            # 位置
            ax_pos.plot(data["step"], data["position"],
                        color=color, linestyle=line_style, linewidth=1.3)

            # 速度
            ax_vel.plot(data["step"], data["speed"],
                        color=color, linestyle=line_style, linewidth=1.3)

            # 加速度
            ax_acc.plot(data["step"], data["acceleration"],
                        color=color, linestyle=line_style, linewidth=1.3)

            # # 加加速度（仅对CAV车辆绘制）
            # if data['is_cav']:
            #     ax_jerk.plot(data["step"], data["jerk"],
            #                 color=color, linestyle=line_style, linewidth=1.3)

        # 设置速度 y-lim
        all_speeds = []
        for data in vehicle_data.values():
            all_speeds += data["speed"]
        if all_speeds:
            ax_vel.set_ylim(min(all_speeds)-2, max(all_speeds)+2)

        # 加速度范围
        ax_acc.set_ylim(-3, 3)

        # 加加速度范围
        # all_jerks = []
        # for data in vehicle_data.values():
        #     if 'jerk' in data and data['is_cav']:
        #         all_jerks += data["jerk"]
        # if all_jerks:
        #     ax_jerk.set_ylim(min(all_jerks)-1, max(all_jerks)+1)

        # 轴标签
        ax_pos.set_ylabel("Position (m)")
        ax_vel.set_ylabel("Speed (m/s)")
        ax_acc.set_ylabel("Acceleration (m/s$^2$)")
        #ax_jerk.set_ylabel("Jerk (m/s$^3$)")

        #ax_jerk.set_xlabel("Time Step [0.1 s]")
        ax_acc.set_xlabel("Time Step [0.1 s]")
        ax_vel.set_xlabel("Time Step [0.1 s]")
        ax_pos.set_xlabel("Time Step [0.1 s]")

        # 网格
        for ax in (ax_pos, ax_vel, ax_acc):
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

        # 设置图例（仅第1行）
        if condition_idx == 0:
            legend_elements = [
                plt.Line2D([0], [0], color=cav_color, linestyle='--', linewidth=1.5),
                plt.Line2D([0], [0], color=hv_color, linestyle='-', linewidth=1.5)
            ]
            legend_labels = ['CAV', 'HV']

            ax_pos.legend(legend_elements, legend_labels, loc='upper right')

    # 行级标注 (a)、(b)，位于各行最下方中央
    for row_idx, label in enumerate(['(a)', '(b)']):
        row_axes = axes[row_idx]
        row_bottom = min(ax.get_position().y0 for ax in row_axes)
        fig.text(0.5, row_bottom - 0.08, label,
                 fontsize=16, fontweight='bold', ha='center', va='top')

    # 保存
    output_dir = os.path.join(_THIS_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{algorithm}_two_conditions_trajectories.png")

    plt.savefig(plot_path, dpi=350, bbox_inches='tight')
    print(f"📊 {algorithm.upper()} 两种工况轨迹图已保存到: {plot_path}")

    plt.show()

def calculate_following_performance(trajectory_data, algorithm="rl"):
    """
    计算跟驰性能指标
    
    参数:
        trajectory_data: 轨迹数据
        algorithm: 算法类型
    """
    if not trajectory_data:
        print("❌ 没有轨迹数据可分析")
        return
    
    # 提取车辆数据
    vehicle_data = {}
    sample_state = trajectory_data[0]["state"]
    for vehicle_state in sample_state:
        vid = vehicle_state["id"]
        vehicle_data[vid] = {
            "position": [],
            "speed": [],
            "acceleration": []
        }
    
    # 填充数据
    for data in trajectory_data:
        for vehicle_state in data["state"]:
            vid = vehicle_state["id"]
            if vid in vehicle_data:
                vehicle_data[vid]["position"].append(vehicle_state["x"])
                vehicle_data[vid]["speed"].append(vehicle_state["v"])
                vehicle_data[vid]["acceleration"].append(vehicle_state["a"])
    
    # 自适应确定CAV和HV车辆
    # 找到所有CAV车辆，按位置排序
    cav_vehicles = [vid for vid, data in vehicle_data.items() if data.get("is_cav", False)]
    # 按位置排序CAV（位置越大的越靠前）
    sorted_cav_vehicles = sorted(cav_vehicles, key=lambda vid: vehicle_data[vid]["position"][0] if vehicle_data[vid]["position"] else 0, reverse=True)
    
    # 确定第二辆CAV（索引为1的CAV）
    second_cav = sorted_cav_vehicles[1] if len(sorted_cav_vehicles) > 1 else None
    
    # 确定两个CAV之间的HV车辆
    hv_vehicles_between_cavs = []
    if len(sorted_cav_vehicles) >= 2:
        # 获取第一辆CAV和第二辆CAV
        first_cav = sorted_cav_vehicles[0]
        second_cav = sorted_cav_vehicles[1]
        
        # 获取所有HV车辆
        hv_vehicles = [vid for vid, data in vehicle_data.items() if not data.get("is_cav", False)]
        
        # 找到位于两个CAV之间的HV车辆
        first_cav_pos = vehicle_data[first_cav]["position"][0] if vehicle_data[first_cav]["position"] else 0
        second_cav_pos = vehicle_data[second_cav]["position"][0] if vehicle_data[second_cav]["position"] else 0
        
        # 确保first_cav_pos > second_cav_pos（因为位置越大越靠前）
        for hv_id in hv_vehicles:
            hv_pos = vehicle_data[hv_id]["position"][0] if vehicle_data[hv_id]["position"] else 0
            # HV车辆位置在两个CAV之间
            if second_cav_pos < hv_pos < first_cav_pos:
                hv_vehicles_between_cavs.append(hv_id)
    
    # 计算第二辆CAV和其后方第一个HV之间的跟驰误差
    if second_cav and hv_vehicles_between_cavs:
        # 选择最靠近第二辆CAV的HV车辆（位置最小的HV）
        following_hv = min(hv_vehicles_between_cavs, key=lambda vid: vehicle_data[vid]["position"][0] if vehicle_data[vid]["position"] else float('inf'))
        
        cav_pos = np.array(vehicle_data[second_cav]["position"])
        lead_pos = np.array(vehicle_data[following_hv]["position"])
        
        # 计算间距
        gap = lead_pos - cav_pos - 5.0  # 减去车辆长度
        
        # 计算平均间距误差
        target_gap = 20.0  # 目标间距
        gap_error = np.abs(gap - target_gap)
        mean_gap_error = np.mean(gap_error)
        
        # 计算速度差
        cav_speed = np.array(vehicle_data[second_cav]["speed"])
        lead_speed = np.array(vehicle_data[following_hv]["speed"])
        speed_diff = np.abs(cav_speed - lead_speed)
        mean_speed_diff = np.mean(speed_diff)
        
        print(f"\n📈 {algorithm.upper()} 跟驰性能指标:")
        print(f"   - CAV车辆: {second_cav}")
        print(f"   - 跟驰HV车辆: {following_hv}")
        print(f"   - 平均间距误差: {mean_gap_error:.2f} m")
        print(f"   - 平均速度差: {mean_speed_diff:.2f} m/s")
        print(f"   - 最大间距误差: {np.max(gap_error):.2f} m")
        print(f"   - 最大速度差: {np.max(speed_diff):.2f} m/s")

def calculate_and_save_dimensionless_metrics(all_trajectory_data, algorithm="rl", dt=0.1, scenario_names=None):
    """
    计算并保存无量纲性能指标到txt文件
    
    参数:
        all_trajectory_data: 所有工况的轨迹数据列表
        algorithm: 算法类型
        dt: 时间步长
        scenario_names: 工况名称列表（可选）
    """
    if not all_trajectory_data:
        print("❌ 没有轨迹数据可分析")
        return
    
    output_dir = os.path.join(_THIS_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{algorithm}_dimensionless_metrics.txt")

    def calculate_fci(velocity_list, acceleration_list):
        """
        输入:
        velocity_list: 速度列表 [v1, v2, ...] (单位 m/s)
        acceleration_list: 加速度列表 [a1, a2, ...] (单位 m/s^2)
        
        输出:
        FCI: 平均燃油消耗指数 (g/s)
        """
        # 参数定义
        rho1 = 0.365
        rho2 = 0.00114
        rho3 = 9.65e-07
        rho4 = 0.943
        rho5 = 0.299
        
        # 物理阻力参数
        rho_r = 0.1326      # 滚动阻力
        rho_s = 2.7384e-03  # 速度修正
        rho_a = 1.0843e-03  # 风阻
        rho_m = 1325.0      # 车辆质量 (kg)
        
        fr_sum = 0
        n = len(velocity_list)
        
        for v, a in zip(velocity_list, acceleration_list):
            # 1. 计算牵引功率需求 P_tract
            # 注意：坡度设为0，sin(0)=0，故省略坡度项
            p_tract = (rho_r * v) + (rho_s * v**2) + (rho_a * v**3) + (rho_m * v * a)
            
            # 2. 计算瞬时油耗 FR
            if p_tract > 0:
                fr = rho1 + (rho2 * v) + (rho3 * v**3) + (rho4 * v * a)
            else:
                fr = rho5
                
            fr_sum += fr

        # 3. 计算平均值 FCI
        fci = fr_sum / n if n > 0 else 0.0
        return fci
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"Evaluation Metrics Report - {algorithm.upper()}\n")
        f.write(f"{'='*80}\n\n")
        # 对每个工况分别计算指标
        for condition_idx, trajectory_data in enumerate(all_trajectory_data):
            # f.write(f"\n{'='*80}\n")
            # f.write(f"Condition {condition_idx + 1}\n")
            # f.write(f"{'='*80}\n\n")
            
            # 提取车辆数据
            vehicle_data = {}
            sample_state = trajectory_data[0]["state"]
            for vehicle_state in sample_state:
                vid = vehicle_state["id"]
                vehicle_data[vid] = {
                    "position": [],
                    "speed": [],
                    "acceleration": [],
                    "is_cav": vehicle_state["type"] == "CAV"
                }
            
            # 填充数据
            for data in trajectory_data:
                for vehicle_state in data["state"]:
                    vid = vehicle_state["id"]
                    if vid in vehicle_data:
                        vehicle_data[vid]["position"].append(vehicle_state["x"])
                        vehicle_data[vid]["speed"].append(vehicle_state["v"])
                        vehicle_data[vid]["acceleration"].append(vehicle_state["a"])
            
            # 计算加加速度（jerk）
            for vid, data in vehicle_data.items():
                accelerations = np.array(data["acceleration"])
                if len(accelerations) > 1:
                    jerk = np.diff(accelerations) / dt
                    jerk = np.concatenate([[0], jerk])
                    data["jerk"] = jerk
                else:
                    data["jerk"] = np.array([0] * len(accelerations))
            
            # 分离CAV和HV车辆
            cav_ids = [vid for vid, data in vehicle_data.items() if data["is_cav"]]
            hv_ids = [vid for vid, data in vehicle_data.items() if not data["is_cav"]]
            all_vids = list(vehicle_data.keys())
            
            # 1. Fleet Average Speed (Sum of average speed of each vehicle / Total number of vehicles)
            avg_speeds = []
            for vid in all_vids:
                avg_speed_vehicle = np.mean(vehicle_data[vid]["speed"])
                avg_speeds.append(avg_speed_vehicle)
            
            fleet_avg_speed = np.sum(avg_speeds) / len(all_vids) if all_vids else 0.0
            
            # f.write(f"1. Fleet Average Speed\n")
            # f.write(f"   Value: {fleet_avg_speed:.6f} m/s\n")
            # f.write(f"   Description: Sum of average speed of each vehicle / Total number of vehicles\n\n")
            
            # 2. String Stability Metrics
            sorted_vehicles = sorted(vehicle_data.keys(), 
                                   key=lambda vid: vehicle_data[vid]["position"][0], 
                                   reverse=True)
            
            cav_acc_sq_ratios = []
            cav_speed_ratios = []
            for i in range(len(sorted_vehicles) - 1):
                lead_vid = sorted_vehicles[i]
                follow_vid = sorted_vehicles[i + 1]
                
                # Only consider if follower is CAV
                if vehicle_data[follow_vid]["is_cav"]:
                    lead_acc = np.array(vehicle_data[lead_vid]["acceleration"])
                    follow_acc = np.array(vehicle_data[follow_vid]["acceleration"])
                    
                    sum_lead_acc_sq = np.sqrt(np.sum(abs(lead_acc**2)))
                    sum_follow_acc_sq = np.sqrt(np.sum(abs(follow_acc**2)))
                    
                    if sum_follow_acc_sq > 1e-6:
                        ratio = sum_lead_acc_sq/sum_follow_acc_sq
                        cav_acc_sq_ratios.append(ratio)
                    
                    # New Speed Stability Metric
                    lead_speed = np.array(vehicle_data[lead_vid]["speed"])
                    follow_speed = np.array(vehicle_data[follow_vid]["speed"])
                    
                    sum_lead_speed = np.sum(lead_speed)
                    sum_follow_speed = np.sum(follow_speed)
                    
                    if sum_lead_speed > 1e-6:
                        ratio = sum_lead_speed/sum_follow_speed
                        cav_speed_ratios.append(ratio)
            #去掉第一个值，因为第一个值是0
            # cav_acc_sq_ratios = cav_acc_sq_ratios[1:]
            # cav_speed_ratios = cav_speed_ratios[1:]
            max_cav_acc_sq_ratio = np.max(cav_acc_sq_ratios) if cav_acc_sq_ratios else 0.0
            avg_cav_acc_sq_ratio = np.mean(cav_acc_sq_ratios) if cav_acc_sq_ratios else 0.0
            
            max_cav_speed_ratio = np.max(cav_speed_ratios) if cav_speed_ratios else 0.0
            avg_cav_speed_ratio = np.mean(cav_speed_ratios) if cav_speed_ratios else 0.0
            
            # f.write(f"2. String Stability Metrics (CAV Follower)\n")
            # f.write(f"   Max Ratio: {max_cav_acc_sq_ratio:.6f}\n")
            # f.write(f"   Avg Ratio: {avg_cav_acc_sq_ratio:.6f}\n")
            # f.write(f"   Description: Ratio of (Sum of Preceding Acc^2 / Sum of Following Acc^2)\n")
            # f.write(f"   Note: > 1 implies attenuation (stable), < 1 implies amplification (unstable)\n\n")
            
            # 3. Jerk Metrics
            total_steps = len(trajectory_data)
            
            # Average Total Absolute Jerk (per step)
            sum_abs_jerk_cav = 0.0
            max_abs_jerk_cav = 0.0
            for vid in cav_ids:
                abs_jerk = np.abs(vehicle_data[vid]["jerk"])
                sum_abs_jerk_cav += np.sum(abs_jerk)
                max_abs_jerk_cav = max(max_abs_jerk_cav, np.max(abs_jerk))
            
            avg_abs_jerk_cav = sum_abs_jerk_cav / (len(cav_ids) * total_steps) if cav_ids and total_steps > 0 else 0.0
            
            sum_abs_jerk_hv = 0.0
            max_abs_jerk_hv = 0.0
            for vid in hv_ids:
                abs_jerk = np.abs(vehicle_data[vid]["jerk"])
                sum_abs_jerk_hv += np.sum(abs_jerk)
                max_abs_jerk_hv = max(max_abs_jerk_hv, np.max(abs_jerk))
            
            avg_abs_jerk_hv = sum_abs_jerk_hv / (len(hv_ids) * total_steps) if hv_ids and total_steps > 0 else 0.0
            
            # f.write(f"3. Jerk Metrics\n")
            # f.write(f"   Avg Abs Jerk (CAV): {avg_abs_jerk_cav:.6f} m/s^3 (per step)\n")
            # f.write(f"   Avg Abs Jerk (HV):  {avg_abs_jerk_hv:.6f} m/s^3 (per step)\n")
            # f.write(f"   Max Abs Jerk (CAV): {max_abs_jerk_cav:.6f} m/s^3\n")
            # f.write(f"   Max Abs Jerk (HV):  {max_abs_jerk_hv:.6f} m/s^3\n\n")
            
            # 4. Speed Tracking Error Coefficient
            # Definition: Mean(|v_lead - v_follow|) / Mean(v_fleet)
            speed_diff_sum = 0.0
            pair_count = 0
            total_points = 0
            
            for i in range(len(sorted_vehicles) - 1):
                lead_vid = sorted_vehicles[i]
                follow_vid = sorted_vehicles[i + 1]
                
                lead_speed = np.array(vehicle_data[lead_vid]["speed"])
                follow_speed = np.array(vehicle_data[follow_vid]["speed"])
                
                speed_diff_sum += np.sum(np.abs(lead_speed - follow_speed))
                total_points += len(lead_speed)
                pair_count += 1
                
            avg_speed_diff = speed_diff_sum / total_points if total_points > 0 else 0.0
            speed_tracking_error_coeff = avg_speed_diff / fleet_avg_speed if fleet_avg_speed > 1e-3 else 0.0
            
            # f.write(f"4. Speed Tracking Error Coefficient\n")
            # f.write(f"   Value: {speed_tracking_error_coeff:.6f}\n")
            # f.write(f"   Description: Mean(|v_lead - v_follow|) / Fleet Avg Speed\n\n")
            
            # 5. Fuel Consumption (FCI)
            fci_values = []
            for vid in all_vids:
                speed = np.array(vehicle_data[vid]["speed"])
                acc = np.array(vehicle_data[vid]["acceleration"])
                fci = calculate_fci(speed, acc)
                fci_values.append(fci)
            
            avg_fci = np.mean(fci_values) if fci_values else 0.0
            
            # f.write(f"5. Fuel Consumption (Average FCI)\n")
            # f.write(f"   Value: {avg_fci:.6f} g/s\n")
            # f.write(f"   Description: Average Fuel Consumption Index across all vehicles\n\n")
            
            # 6. Minimum TTC
            min_ttc_values = []
            for i in range(len(sorted_vehicles) - 1):
                lead_vid = sorted_vehicles[i]
                follow_vid = sorted_vehicles[i + 1]
                
                lead_pos = np.array(vehicle_data[lead_vid]["position"])
                follow_pos = np.array(vehicle_data[follow_vid]["position"])
                lead_speed = np.array(vehicle_data[lead_vid]["speed"])
                follow_speed = np.array(vehicle_data[follow_vid]["speed"])
                
                # Calculate gap and relative speed
                gap = lead_pos - follow_pos - 5.0
                rel_speed = follow_speed - lead_speed
                
                # Calculate TTC only where rel_speed > 0
                closing_mask = rel_speed > 1e-3
                
                if np.any(closing_mask):
                    ttc = gap[closing_mask] / rel_speed[closing_mask]
                    min_ttc_values.append(np.min(ttc))
            
            min_ttc = np.min(min_ttc_values) if min_ttc_values else float('inf')
            
            # f.write(f"6. Minimum TTC (minttc)\n")
            # f.write(f"   Value: {min_ttc:.6f} s\n")
            # f.write(f"   Description: Minimum Time-To-Collision across all pairs\n\n")
            
            # 7. CAV Tracking Error Metric: Average of (|deltad/10| + |deltav|) for CAV only
            # Calculate deltad (gap error) and deltav (speed error) from vehicle states
            cav_tracking_errors = []
            
            # IDM parameters for desired gap calculation
            s0 = 2.0
            T = 1.6
            
            for i in range(len(sorted_vehicles) - 1):
                lead_vid = sorted_vehicles[i]
                follow_vid = sorted_vehicles[i + 1]
                
                # Only consider if follower is CAV
                if vehicle_data[follow_vid]["is_cav"]:
                    lead_pos = np.array(vehicle_data[lead_vid]["position"])
                    follow_pos = np.array(vehicle_data[follow_vid]["position"])
                    lead_speed = np.array(vehicle_data[lead_vid]["speed"])
                    follow_speed = np.array(vehicle_data[follow_vid]["speed"])
                    
                    # Calculate actual gap (bumper to bumper)
                    gap = lead_pos - follow_pos - 5.0  # 5.0 is vehicle length
                    
                    # Calculate desired gap: s0 + v * T
                    desired_gap = s0 + follow_speed * T
                    
                    # Calculate deltad (gap error): actual gap - desired gap
                    deltad = gap - desired_gap
                    
                    # Calculate deltav (speed error): lead speed - own speed
                    deltav = lead_speed - follow_speed
                    
                    # Calculate tracking error: |deltad/10| + |deltav|
                    tracking_error = np.abs(deltad / 10.0) + np.abs(deltav)
                    cav_tracking_errors.extend(tracking_error.tolist())
            
            # Calculate average tracking error
            avg_cav_tracking_error = np.mean(cav_tracking_errors) if cav_tracking_errors else 0.0
            
            # Summary Table
            f.write(f"\n{'='*80}\n")
            if scenario_names and condition_idx < len(scenario_names):
                f.write(f"{scenario_names[condition_idx]} Summary\n")
            else:
                f.write(f"Condition {condition_idx + 1} Summary\n")
            f.write(f"{'='*80}\n")
            f.write(f"Fleet Avg Speed:      {fleet_avg_speed:.6f}\n")
            f.write(f"String Stability Max: {max_cav_acc_sq_ratio:.6f}\n")
            f.write(f"String Stability Avg: {avg_cav_acc_sq_ratio:.6f}\n")
            f.write(f"Speed Stability Max:  {max_cav_speed_ratio:.6f}\n")
            f.write(f"Speed Stability Avg:  {avg_cav_speed_ratio:.6f}\n")
            f.write(f"Avg Abs Jerk (CAV):   {avg_abs_jerk_cav:.6f}\n")
            f.write(f"Avg Abs Jerk (HV):    {avg_abs_jerk_hv:.6f}\n")
            f.write(f"Max Abs Jerk (CAV):   {max_abs_jerk_cav:.6f}\n")
            f.write(f"Max Abs Jerk (HV):    {max_abs_jerk_hv:.6f}\n")
            f.write(f"Speed Tracking Err:   {speed_tracking_error_coeff:.6f}\n")
            f.write(f"Fuel Consumption:     {avg_fci:.6f}\n")
            f.write(f"Min TTC:              {min_ttc:.6f}\n")
            f.write(f"CAV Tracking Error:   {avg_cav_tracking_error:.6f}\n")


    print(f"Evaluation metrics saved to: {output_file}")

def main():
    """主函数 - 使用示例"""
    print("🚗 灵活的强化学习算法测试脚本")
    print("=" * 50)
    print("该脚本允许在相同的环境配置下测试不同的强化学习算法模型")
    print()
    
    # 获取项目根目录
    model_dir = os.path.join(_PROJECT_ROOT, "models")
    ddpg_model_path = os.path.join(model_dir, "ddpg_mixed_cav_agent_best.pth", "ddpg_actor.pth")
    
    # 测试不同攻击频率下的表现
    test_model_attack_frequencies("ddpg", ddpg_model_path)
    
    # 示例：测试DDPG模型（使用两种特定工况）
    # test_model_two_conditions("ddpg", ddpg_model_path)


if __name__ == "__main__":
    main()