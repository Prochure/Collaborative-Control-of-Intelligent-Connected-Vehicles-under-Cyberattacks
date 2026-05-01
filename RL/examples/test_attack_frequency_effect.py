#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试统一领车速度序列下，网络攻击频率对车头时距、平均速度、加速度震荡、加速度绝对值平均的影响
使用强化学习控制CAV车辆
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# 添加项目路径
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)



plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 全部英文字体
plt.rcParams['font.family'] = 'serif'
from sim_env.envs.cyber_attack_env import CyberAttackEnv

# 修正导入路径
from examples.flexible_rl_test import generate_deceleration_acceleration_speed_sequence, generate_random_oscillating_speed_sequence, Actor

# DDPG Actor网络（用于加载预训练模型）
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

def generate_unified_lead_speed_sequence(total_steps=600, dt=0.1):
    """
    生成统一的领车速度序列
    
    参数:
        total_steps: 总步数
        dt: 时间步长
    
    返回:
        speed_sequence: 速度序列 (m/s)
    """
    t = np.arange(total_steps) * dt
    
    # 生成一个平滑的速度序列，包含多种驾驶行为
    speed_sequence = 15.0 + (
        2.0 * np.sin(0.2 * t) +      # 低频变化（周期约31秒）
        1.0 * np.sin(0.5 * t) +      # 中频变化（周期约13秒）
        0.5 * np.sin(1.0 * t)        # 高频变化（周期约6秒）
    )
    
    # 添加一些特定的驾驶行为
    # 前100步：匀速行驶
    speed_sequence[:100] = 15.0
    
    # 100-200步：缓慢减速
    speed_sequence[100:200] = np.linspace(15.0, 12.0, 100)
    
    # 200-300步：匀速行驶
    speed_sequence[200:300] = 12.0
    
    # 300-400步：加速
    speed_sequence[300:400] = np.linspace(12.0, 18.0, 100)
    
    # 400-500步：匀速行驶
    speed_sequence[400:500] = 18.0
    
    # 500-600步：减速
    speed_sequence[500:600] = np.linspace(18.0, 15.0, 100)
    
    # 限制速度范围在合理区间内 (8-25 m/s)
    speed_sequence = np.clip(speed_sequence, 8.0, 25.0)
    
    return speed_sequence

def calculate_time_headways(trajectory_data, vehicle_id):
    """
    计算车头时距（支持任意车辆，包括CAV和HV）
    
    参数:
        trajectory_data: 轨迹数据
        vehicle_id: 车辆ID（可以是CAV或HV）
    
    返回:
        time_headways: 车头时距序列
    """
    time_headways = []
    
    for data in trajectory_data:
        state = data["state"]
        
        # 找到目标车辆
        target_vehicle = None
        front_vehicle = None
        
        for i, vehicle_state in enumerate(state):
            if vehicle_state["id"] == vehicle_id:
                target_vehicle = vehicle_state
                # 前车是列表中的下一个车辆（位置更大的车辆）
                if i > 0:
                    front_vehicle = state[i-1]
                break
        
        if target_vehicle and front_vehicle:
            # 计算车头时距 = (前车位置 - 前车长度 - 后车位置) / 后车速度
            gap = front_vehicle["x"] - 5.0 - target_vehicle["x"]  # 5.0是车辆长度
            speed = max(0.1, target_vehicle["v"])  # 避免除以0
            time_headway = gap / speed
            time_headways.append(time_headway)
        else:
            time_headways.append(0.0)
    
    return np.array(time_headways)

def calculate_jerk(accelerations, dt=0.1):
    """
    计算急动度（Jerk）：加速度的变化率
    
    参数:
        accelerations: 加速度序列
        dt: 时间步长
    
    返回:
        jerk: 急动度序列
        avg_abs_jerk: 平均绝对急动度
        max_abs_jerk: 最大绝对急动度
    """
    if len(accelerations) < 2:
        return np.array([]), 0.0, 0.0
    
    jerk = np.diff(accelerations) / dt
    avg_abs_jerk = np.mean(np.abs(jerk))
    max_abs_jerk = np.max(np.abs(jerk))
    
    return jerk, avg_abs_jerk, max_abs_jerk

def calculate_oscillation_metrics(values):
    """
    计算震荡相关指标
    
    参数:
        values: 数值序列（速度、加速度等）
    
    返回:
        dict: 包含震荡指标的字典
    """
    if len(values) == 0:
        return {
            "range": 0.0,
            "cv": 0.0,  # 变异系数
            "max": 0.0,
            "min": 0.0,
            "variation_rate_std": 0.0  # 变化率的标准差
        }
    
    values = np.array(values)
    mean_val = np.mean(values)
    
    # 避免除以0
    cv = np.std(values) / mean_val if abs(mean_val) > 1e-6 else 0.0
    
    # 变化率（一阶差分）的标准差
    if len(values) > 1:
        variation_rate = np.diff(values)
        variation_rate_std = np.std(variation_rate)
    else:
        variation_rate_std = 0.0
    
    return {
        "range": np.max(values) - np.min(values),  # 震荡幅度
        "cv": cv,  # 变异系数（Coefficient of Variation）
        "max": np.max(values),
        "min": np.min(values),
        "variation_rate_std": variation_rate_std
    }

def calculate_metrics(trajectory_data, cav_ids):
    """
    计算性能指标（同时包括CAV和普通车辆HV）
    
    参数:
        trajectory_data: 轨迹数据
        cav_ids: CAV车辆ID列表
    
    返回:
        metrics: 性能指标字典（包含CAV和HV的分别统计以及总体统计）
    """
    # 从第一次状态中获取所有车辆ID和类型
    if len(trajectory_data) == 0:
        return {}
    
    first_state = trajectory_data[0]["state"]
    all_vehicle_ids = [v["id"] for v in first_state]
    # HV车辆：不在CAV列表中的所有车辆
    cav_id_set = set(cav_ids)
    hv_ids = [v["id"] for v in first_state if v["id"] not in cav_id_set]
    
    # 存储CAV的数据
    all_cav_speeds = []
    all_cav_accelerations = []
    all_cav_time_headways = []
    
    # 存储HV的数据
    all_hv_speeds = []
    all_hv_accelerations = []
    all_hv_time_headways = []
    
    # 存储所有车辆的数据（CAV + HV）
    all_vehicle_speeds = []
    all_vehicle_accelerations = []
    all_vehicle_time_headways = []
    
    # 计算CAV指标
    for cav_id in cav_ids:
        cav_speeds = []
        cav_accelerations = []
        
        for data in trajectory_data:
            state = data["state"]
            for vehicle_state in state:
                if vehicle_state["id"] == cav_id:
                    cav_speeds.append(vehicle_state["v"])
                    cav_accelerations.append(vehicle_state["a"])
                    break
        
        cav_speeds = np.array(cav_speeds)
        cav_accelerations = np.array(cav_accelerations)
        time_headways = calculate_time_headways(trajectory_data, cav_id)
        
        all_cav_speeds.extend(cav_speeds)
        all_cav_accelerations.extend(cav_accelerations)
        all_cav_time_headways.extend(time_headways)
    
    # 计算HV指标
    for hv_id in hv_ids:
        hv_speeds = []
        hv_accelerations = []
        
        for data in trajectory_data:
            state = data["state"]
            for vehicle_state in state:
                if vehicle_state["id"] == hv_id:
                    hv_speeds.append(vehicle_state["v"])
                    hv_accelerations.append(vehicle_state["a"])
                    break
        
        hv_speeds = np.array(hv_speeds)
        hv_accelerations = np.array(hv_accelerations)
        time_headways = calculate_time_headways(trajectory_data, hv_id)
        
        all_hv_speeds.extend(hv_speeds)
        all_hv_accelerations.extend(hv_accelerations)
        all_hv_time_headways.extend(time_headways)
    
    # 转换为numpy数组
    all_cav_speeds = np.array(all_cav_speeds)
    all_cav_accelerations = np.array(all_cav_accelerations)
    all_cav_time_headways = np.array(all_cav_time_headways)
    
    all_hv_speeds = np.array(all_hv_speeds)
    all_hv_accelerations = np.array(all_hv_accelerations)
    all_hv_time_headways = np.array(all_hv_time_headways)
    
    # 合并所有车辆数据
    all_vehicle_speeds = np.concatenate([all_cav_speeds, all_hv_speeds])
    all_vehicle_accelerations = np.concatenate([all_cav_accelerations, all_hv_accelerations])
    all_vehicle_time_headways = np.concatenate([all_cav_time_headways, all_hv_time_headways])
    
    # 计算时间步长（从轨迹数据中获取）
    dt = 0.1  # 默认值
    if len(trajectory_data) > 1:
        dt = trajectory_data[1].get("time", 0.1) - trajectory_data[0].get("time", 0.0)
        if dt <= 0:
            dt = 0.1
    
    # 计算震荡指标
    # CAV震荡指标
    cav_speed_osc = calculate_oscillation_metrics(all_cav_speeds)
    cav_acc_osc = calculate_oscillation_metrics(all_cav_accelerations)
    cav_th_osc = calculate_oscillation_metrics(all_cav_time_headways)
    _, cav_avg_abs_jerk, cav_max_abs_jerk = calculate_jerk(all_cav_accelerations, dt)
    
    # HV震荡指标
    hv_speed_osc = calculate_oscillation_metrics(all_hv_speeds)
    hv_acc_osc = calculate_oscillation_metrics(all_hv_accelerations)
    hv_th_osc = calculate_oscillation_metrics(all_hv_time_headways)
    _, hv_avg_abs_jerk, hv_max_abs_jerk = calculate_jerk(all_hv_accelerations, dt)
    
    # 总体震荡指标
    overall_speed_osc = calculate_oscillation_metrics(all_vehicle_speeds)
    overall_acc_osc = calculate_oscillation_metrics(all_vehicle_accelerations)
    overall_th_osc = calculate_oscillation_metrics(all_vehicle_time_headways)
    _, overall_avg_abs_jerk, overall_max_abs_jerk = calculate_jerk(all_vehicle_accelerations, dt)
    
    # 计算综合指标（包括CAV、HV和总体）
    metrics = {
        # CAV基本指标
        "cav_avg_time_headway": np.mean(all_cav_time_headways) if len(all_cav_time_headways) > 0 else 0.0,
        "cav_time_headway_std": np.std(all_cav_time_headways) if len(all_cav_time_headways) > 0 else 0.0,
        "cav_avg_speed": np.mean(all_cav_speeds) if len(all_cav_speeds) > 0 else 0.0,
        "cav_speed_std": np.std(all_cav_speeds) if len(all_cav_speeds) > 0 else 0.0,
        "cav_acceleration_oscillation": np.std(all_cav_accelerations) if len(all_cav_accelerations) > 0 else 0.0,
        "cav_avg_abs_acceleration": np.mean(np.abs(all_cav_accelerations)) if len(all_cav_accelerations) > 0 else 0.0,
        "cav_max_abs_acceleration": np.max(np.abs(all_cav_accelerations)) if len(all_cav_accelerations) > 0 else 0.0,
        
        # CAV震荡指标
        "cav_speed_range": cav_speed_osc["range"],
        "cav_speed_cv": cav_speed_osc["cv"],  # 速度变异系数
        "cav_speed_variation_rate_std": cav_speed_osc["variation_rate_std"],  # 速度变化率标准差
        "cav_acceleration_range": cav_acc_osc["range"],
        "cav_acceleration_cv": cav_acc_osc["cv"],  # 加速度变异系数
        "cav_time_headway_cv": cav_th_osc["cv"],  # 车头时距变异系数
        "cav_avg_abs_jerk": cav_avg_abs_jerk,  # 平均绝对急动度
        "cav_max_abs_jerk": cav_max_abs_jerk,  # 最大绝对急动度
        
        # HV基本指标
        "hv_avg_time_headway": np.mean(all_hv_time_headways) if len(all_hv_time_headways) > 0 else 0.0,
        "hv_time_headway_std": np.std(all_hv_time_headways) if len(all_hv_time_headways) > 0 else 0.0,
        "hv_avg_speed": np.mean(all_hv_speeds) if len(all_hv_speeds) > 0 else 0.0,
        "hv_speed_std": np.std(all_hv_speeds) if len(all_hv_speeds) > 0 else 0.0,
        "hv_acceleration_oscillation": np.std(all_hv_accelerations) if len(all_hv_accelerations) > 0 else 0.0,
        "hv_avg_abs_acceleration": np.mean(np.abs(all_hv_accelerations)) if len(all_hv_accelerations) > 0 else 0.0,
        "hv_max_abs_acceleration": np.max(np.abs(all_hv_accelerations)) if len(all_hv_accelerations) > 0 else 0.0,
        
        # HV震荡指标
        "hv_speed_range": hv_speed_osc["range"],
        "hv_speed_cv": hv_speed_osc["cv"],
        "hv_speed_variation_rate_std": hv_speed_osc["variation_rate_std"],
        "hv_acceleration_range": hv_acc_osc["range"],
        "hv_acceleration_cv": hv_acc_osc["cv"],
        "hv_time_headway_cv": hv_th_osc["cv"],
        "hv_avg_abs_jerk": hv_avg_abs_jerk,
        "hv_max_abs_jerk": hv_max_abs_jerk,
        
        # 总体基本指标（所有车辆）
        "avg_time_headway": np.mean(all_vehicle_time_headways) if len(all_vehicle_time_headways) > 0 else 0.0,
        "time_headway_std": np.std(all_vehicle_time_headways) if len(all_vehicle_time_headways) > 0 else 0.0,
        "avg_speed": np.mean(all_vehicle_speeds) if len(all_vehicle_speeds) > 0 else 0.0,
        "speed_std": np.std(all_vehicle_speeds) if len(all_vehicle_speeds) > 0 else 0.0,
        "acceleration_oscillation": np.std(all_vehicle_accelerations) if len(all_vehicle_accelerations) > 0 else 0.0,
        "avg_abs_acceleration": np.mean(np.abs(all_vehicle_accelerations)) if len(all_vehicle_accelerations) > 0 else 0.0,
        "max_abs_acceleration": np.max(np.abs(all_vehicle_accelerations)) if len(all_vehicle_accelerations) > 0 else 0.0,
        
        # 总体震荡指标
        "speed_range": overall_speed_osc["range"],
        "speed_cv": overall_speed_osc["cv"],
        "speed_variation_rate_std": overall_speed_osc["variation_rate_std"],
        "acceleration_range": overall_acc_osc["range"],
        "acceleration_cv": overall_acc_osc["cv"],
        "time_headway_cv": overall_th_osc["cv"],
        "avg_abs_jerk": overall_avg_abs_jerk,
        "max_abs_jerk": overall_max_abs_jerk,
    }
    
    return metrics

def load_rl_model(model_path, state_dim, action_dim):
    """
    加载预训练的强化学习模型
    
    参数:
        model_path: 模型路径
        state_dim: 状态维度
        action_dim: 动作维度
    
    返回:
        model: 加载的模型
        device: 设备
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"⚠️  模型文件不存在: {model_path}")
        return None, None
    
    try:
        # 创建模型
        model = Actor(state_dim, action_dim, action_limit=3.0).to(device)
        
        # 加载模型权重
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        print(f"✅ 成功加载预训练模型: {model_path}")
        return model, device
    except PermissionError:
        print(f"❌ 权限错误: 无法访问模型文件 {model_path}")
        print("请检查文件权限或以管理员身份运行程序")
        return None, None
    except Exception as e:
        print(f"❌ 加载模型时出错: {str(e)}")
        return None, None

def select_action(model, device, state):
    """
    使用强化学习模型选择动作
    
    参数:
        model: 模型
        device: 设备
        state: 状态
    
    返回:
        action: 选择的动作
    """
    if model is None or device is None:
        return None
        
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    action = model(state)
    action = action.detach().cpu().numpy().flatten()
    return action[0]  # 只返回动作

def run_experiment(attack_frequency, lead_speed_sequence, attack_type='packet_drop', rl_model=None, device=None):
    """
    运行单次实验
    
    参数:
        attack_frequency: 攻击频率
        lead_speed_sequence: 领车速度序列
        attack_type: 攻击类型 ('packet_drop', 'data_tampering', 'delay')
        rl_model: 强化学习模型（可选）
        device: 设备（可选）
    
    返回:
        metrics: 性能指标
        trajectory_data: 轨迹数据
        all_vehicle_states: 全部车辆状态历史
    """
    # 创建环境
    env = CyberAttackEnv(
        num_vehicles=10,
        cav_indices=[1,3,5,7,9],  # 多辆CAV
        dt=0.1,
        enable_cyber_attack=True,
        attack_type=attack_type,
        attack_frequency=attack_frequency,
        attack_targets=["speed", "acceleration",'position'],
        attack_variances={"speed": 20.0, "acceleration": 10, "position": 20},
        attack_start_time=0.0
    )
    
    # 重置环境，传入速度序列
    reset_options = {
        "lead_speed_sequence": lead_speed_sequence
    }
    state, _ = env.reset(options=reset_options)
    
    # 获取环境参数
    state_dim = state.shape[1]  # 每个CAV的观测维度
    action_dim = 1  # 每个CAV的动作维度
    
    # 如果没有提供模型，则尝试加载预训练模型
    if rl_model is None:
        model_dir = os.path.join(_PROJECT_ROOT, "models")
        model_path = os.path.join(model_dir, "ddpg_mixed_cav_agent_best.pth", "ddpg_actor.pth")
        rl_model, device = load_rl_model(model_path, state_dim, action_dim)
    
    # 运行仿真
    trajectory_data = []
    all_vehicle_states = []  # 记录全部车辆状态历史
    max_steps = len(lead_speed_sequence)
    
    for step in range(max_steps):
        # 初始化动作数组
        action = [None] * len(env.cav_ids)
        
        # 如果有RL模型，则使用它来控制CAV
        if rl_model is not None:
            # 获取所有CAV的观测值
            cav_observations = state  # shape: (num_cavs, state_dim)
            
            # 为每个CAV选择动作（除了第一辆CAV，它使用IDM模型）
            for j in range(1, len(env.cav_ids)):  # 从第二个CAV开始
                cav_obs = cav_observations[j]
                rl_action = select_action(rl_model, device, cav_obs)
                if rl_action is not None:
                    action[j] = rl_action
                # 如果RL动作为空，则默认使用IDM控制
        
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # 保存轨迹数据
        step_data = {
            "time": env.sim.t,
            "state": env.sim.get_state(),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        }
        trajectory_data.append(step_data)
        
        # 记录全部车辆状态（包括HV和CAV）
        all_vehicles_state = env.sim.get_state()
        all_vehicle_states.append({
            "time": env.sim.t,
            "vehicles": all_vehicles_state
        })
        
        state = next_state
        
        if terminated or truncated:
            break
    
    # 计算性能指标
    cav_ids = env.cav_ids  # 获取所有CAV ID
    metrics = calculate_metrics(trajectory_data, cav_ids)
    
    return metrics, trajectory_data, all_vehicle_states

def main():
    """主函数"""
    print("🧪 网络攻击频率对车辆性能影响测试（使用强化学习控制）")
    print("=" * 50)
    
    # 生成统一的领车速度序列（只使用一种序列）
    lead_speed_sequence = generate_unified_lead_speed_sequence(600, 0.1)
    print(f"✅ 生成统一领车速度序列，长度: {len(lead_speed_sequence)} 步")
    
    # 定义不同的攻击频率进行测试
    attack_frequencies = [i * 0.05 for i in range(21)]  # 从0.0到1.0，步长为0.05
    
    # 定义三种攻击类型
    attack_types = ['packet_drop', 'data_tampering', 'delay']
    attack_type_labels = ['Packet Drop', 'Data Tampering', 'Delay']
    
    # 存储结果（按攻击类型组织）
    all_results = {}
    for attack_type in attack_types:
        all_results[attack_type] = {
            "attack_frequency": [],
            # 总体基本指标
            "avg_time_headway": [],
            "time_headway_std": [],
            "avg_speed": [],
            "speed_std": [],
            "acceleration_oscillation": [],
            "avg_abs_acceleration": [],
            "max_abs_acceleration": [],
            # 总体震荡指标
            "speed_range": [],
            "speed_cv": [],
            "speed_variation_rate_std": [],
            "acceleration_range": [],
            "acceleration_cv": [],
            "time_headway_cv": [],
            "avg_abs_jerk": [],
            "max_abs_jerk": []
        }
    
    # 对每种攻击类型和每种攻击频率运行实验
    for attack_type, attack_label in zip(attack_types, attack_type_labels):
        print(f"\n{'='*60}")
        print(f"🔍 测试攻击类型: {attack_label}")
        print(f"{'='*60}")
        
        # for freq in attack_frequencies:
        #     print(f"\n🔄 测试攻击频率: {freq*100:.0f}% ({attack_label})")
        #     metrics, trajectory_data, all_vehicle_states = run_experiment(freq, lead_speed_sequence, attack_type=attack_type)
        
        #     # 保存结果（总体基本指标）
        #     all_results[attack_type]["attack_frequency"].append(freq)
        #     all_results[attack_type]["avg_time_headway"].append(metrics["avg_time_headway"])
        #     all_results[attack_type]["time_headway_std"].append(metrics["time_headway_std"])
        #     all_results[attack_type]["avg_speed"].append(metrics["avg_speed"])
        #     all_results[attack_type]["speed_std"].append(metrics["speed_std"])
        #     all_results[attack_type]["acceleration_oscillation"].append(metrics["acceleration_oscillation"])
        #     all_results[attack_type]["avg_abs_acceleration"].append(metrics["avg_abs_acceleration"])
        #     all_results[attack_type]["max_abs_acceleration"].append(metrics["max_abs_acceleration"])
            
        #     # 保存总体震荡指标
        #     all_results[attack_type]["speed_range"].append(metrics["speed_range"])
        #     all_results[attack_type]["speed_cv"].append(metrics["speed_cv"])
        #     all_results[attack_type]["speed_variation_rate_std"].append(metrics["speed_variation_rate_std"])
        #     all_results[attack_type]["acceleration_range"].append(metrics["acceleration_range"])
        #     all_results[attack_type]["acceleration_cv"].append(metrics["acceleration_cv"])
        #     all_results[attack_type]["time_headway_cv"].append(metrics["time_headway_cv"])
        #     all_results[attack_type]["avg_abs_jerk"].append(metrics["avg_abs_jerk"])
        #     all_results[attack_type]["max_abs_jerk"].append(metrics["max_abs_jerk"])
            
        #     # 打印结果
        #     print(f"   平均车头时距: {metrics['avg_time_headway']:.2f} s")
        #     print(f"   平均速度: {metrics['avg_speed']:.2f} m/s")
        #     print(f"   加速度震荡: {metrics['acceleration_oscillation']:.4f} m/s²")
        #     print(f"   速度震荡幅度: {metrics['speed_range']:.2f} m/s")
        #     print(f"   平均绝对急动度: {metrics['avg_abs_jerk']:.4f} m/s³")
    
    # # 保存用于绘图的数据并绘制
    plot_data_file = 'attack_frequency_plot_data.csv'
    # save_plot_data(all_results, attack_types, attack_type_labels, plot_data_file)
    plot_results(plot_data_file, attack_types, attack_type_labels)
    
    # 保存结果到文件
    save_results(all_results, attack_types)
    
    print(f"\n🎉 所有测试完成!")

def save_vehicle_states(all_vehicle_states, attack_frequency):
    """保存全部车辆状态到文件"""
    import csv
    
    # 创建文件名
    filename = f'vehicle_states_attack_freq_{attack_frequency:.1f}.csv'
    
    # 保存为CSV文件
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        # 写入表头
        fieldnames = ['time', 'vehicle_id', 'vehicle_type', 'position', 'speed', 'acceleration']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        # 写入数据
        for step_data in all_vehicle_states:
            time = step_data['time']
            vehicles = step_data['vehicles']
            
            for vehicle in vehicles:
                writer.writerow({
                    'time': time,
                    'vehicle_id': vehicle['id'],
                    'vehicle_type': vehicle['type'],
                    'position': vehicle['x'],
                    'speed': vehicle['v'],
                    'acceleration': vehicle['a']
                })
    
    print(f"💾 全部车辆状态已保存到: {filename}")

def save_plot_data(all_results, attack_types, attack_type_labels, filename):
    """将用于多指标绘图的数据保存到CSV"""
    import csv
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'attack_type', 'attack_label', 'attack_frequency', 'attack_frequency_percent',
            'avg_time_headway', 'time_headway_std', 'time_headway_cv',
            'avg_speed', 'speed_std', 'speed_range', 'speed_cv',
            'acceleration_oscillation', 'acceleration_range', 'acceleration_cv',
            'avg_abs_acceleration', 'max_abs_acceleration',
            'avg_abs_jerk', 'max_abs_jerk'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for attack_type, attack_label in zip(attack_types, attack_type_labels):
            freqs = all_results[attack_type]["attack_frequency"]
            for values in zip(
                freqs,
                all_results[attack_type]["avg_time_headway"],
                all_results[attack_type]["time_headway_std"],
                all_results[attack_type]["time_headway_cv"],
                all_results[attack_type]["avg_speed"],
                all_results[attack_type]["speed_std"],
                all_results[attack_type]["speed_range"],
                all_results[attack_type]["speed_cv"],
                all_results[attack_type]["acceleration_oscillation"],
                all_results[attack_type]["acceleration_range"],
                all_results[attack_type]["acceleration_cv"],
                all_results[attack_type]["avg_abs_acceleration"],
                all_results[attack_type]["max_abs_acceleration"],
                all_results[attack_type]["avg_abs_jerk"],
                all_results[attack_type]["max_abs_jerk"]
            ):
                (freq, avg_th, th_std, th_cv, avg_speed, speed_std, speed_range,
                 speed_cv, acc_osc, acc_range, acc_cv, avg_abs_acc, max_abs_acc,
                 avg_abs_jerk, max_abs_jerk) = values
                
                writer.writerow({
                    'attack_type': attack_type,
                    'attack_label': attack_label,
                    'attack_frequency': freq,
                    'attack_frequency_percent': freq * 100.0,
                    'avg_time_headway': avg_th,
                    'time_headway_std': th_std,
                    'time_headway_cv': th_cv,
                    'avg_speed': avg_speed,
                    'speed_std': speed_std,
                    'speed_range': speed_range,
                    'speed_cv': speed_cv,
                    'acceleration_oscillation': acc_osc,
                    'acceleration_range': acc_range,
                    'acceleration_cv': acc_cv,
                    'avg_abs_acceleration': avg_abs_acc,
                    'max_abs_acceleration': max_abs_acc,
                    'avg_abs_jerk': avg_abs_jerk,
                    'max_abs_jerk': max_abs_jerk
                })
    
    print(f"💾 绘图数据已保存到: {filename}")


def plot_results(plot_data_path, attack_types, attack_type_labels):
    """从CSV读取数据绘图（三种攻击类型对比，多项指标）"""
    import csv
    import math
    
    metric_configs = [
        ("avg_time_headway", "Average Time Headway (s)", "(a)"),
        ("avg_speed", "Average Speed (m/s)", "(d)"),
        ("avg_abs_jerk", "Avg |Jerk| (m/s³)", "(j)"),
        ("avg_abs_acceleration", "Avg |Acceleration| (m/s²)", "(i)")
    ]
    
    # 从CSV读取数据
    plot_data = {
        attack_type: {
            "attack_frequency_percent": [],
            **{metric: [] for metric, _, _ in metric_configs}
        }
        for attack_type in attack_types
    }
    
    with open(plot_data_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            attack_type = row['attack_type']
            if attack_type in plot_data:
                plot_data[attack_type]["attack_frequency_percent"].append(float(row['attack_frequency_percent']))
                for metric, _, _ in metric_configs:
                    plot_data[attack_type][metric].append(float(row[metric]))
    
    # 设置英文字体
    plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 增大字体大小
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    # 确保每种攻击类型都有数据
    if not plot_data_path:
        print("⚠️ 未找到绘图数据文件")
        return
    
    # 定义颜色和标记样式
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']
    
    # 创建图表（3列布局，多指标）
    cols = 2
    rows = math.ceil(len(metric_configs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    fig.suptitle('Impact of Cyber Attack Frequency on Vehicle Performance', fontsize=18)
    axes = axes.flatten()
    
    for ax, (metric_key, ylabel, tag) in zip(axes, metric_configs):
        for i, (attack_type, label) in enumerate(zip(attack_types, attack_type_labels)):
            ax.plot(plot_data[attack_type]["attack_frequency_percent"],
                    plot_data[attack_type][metric_key],
                    marker=markers[i],
                    linestyle='-',
                    label=label,
                    linewidth=2,
                    markersize=6,
                    color=colors[i])
        ax.set_xlabel('Attack Frequency (%)')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.text(0.5, -0.25, tag, transform=ax.transAxes,
                fontsize=16, fontweight='bold', ha='center')
    
    # 隐藏多余子图
    for j in range(len(metric_configs), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig('attack_frequency_effect_rl_control.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Results chart saved to: attack_frequency_effect_rl_control.png")

def save_results(all_results, attack_types):
    """保存结果到文件（三种攻击类型）"""
    import csv
    
    # 为每种攻击类型保存一个CSV文件
    for attack_type in attack_types:
        results = all_results[attack_type]
        filename = f'attack_frequency_effect_results_{attack_type}.csv'
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'attack_frequency', 'attack_frequency_percent',
                # 总体基本指标
                'avg_time_headway', 'time_headway_std', 
                'avg_speed', 'speed_std', 'acceleration_oscillation', 
                'avg_abs_acceleration', 'max_abs_acceleration',
                # 总体震荡指标
                'speed_range', 'speed_cv', 'speed_variation_rate_std',
                'acceleration_range', 'acceleration_cv', 'time_headway_cv',
                'avg_abs_jerk', 'max_abs_jerk'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for i in range(len(results["attack_frequency"])):
                writer.writerow({
                    'attack_frequency': results["attack_frequency"][i],
                    'attack_frequency_percent': results["attack_frequency"][i] * 100,
                    # 总体基本指标
                    'avg_time_headway': results["avg_time_headway"][i],
                    'time_headway_std': results["time_headway_std"][i],
                    'avg_speed': results["avg_speed"][i],
                    'speed_std': results["speed_std"][i],
                    'acceleration_oscillation': results["acceleration_oscillation"][i],
                    'avg_abs_acceleration': results["avg_abs_acceleration"][i],
                    'max_abs_acceleration': results["max_abs_acceleration"][i],
                    # 总体震荡指标
                    'speed_range': results["speed_range"][i],
                    'speed_cv': results["speed_cv"][i],
                    'speed_variation_rate_std': results["speed_variation_rate_std"][i],
                    'acceleration_range': results["acceleration_range"][i],
                    'acceleration_cv': results["acceleration_cv"][i],
                    'time_headway_cv': results["time_headway_cv"][i],
                    'avg_abs_jerk': results["avg_abs_jerk"][i],
                    'max_abs_jerk': results["max_abs_jerk"][i]
                })
        
        print(f"💾 {attack_type} 结果数据已保存到: {filename}")

if __name__ == "__main__":
    main()