#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成完整的网络攻击训练数据集

该脚本生成一个完整的网络攻击训练数据集，包含多种车辆配置、
固定IDM参数和不同类型的网络攻击。
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

# 添加项目路径
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sim_env.envs.cyber_attack_env import CyberAttackEnv

def generate_topology_configurations():
    """生成不同的拓扑结构配置"""
    configurations = []
    
    # 固定6个车辆
    num_vehicles = 6
    
    # CAV在队列的2，3，4，5位置（索引1，2，3，4）
    cav_positions_list = [
        [1],  # CAV在位置2
        [2],  # CAV在位置3
        [3],  # CAV在位置4
        [4],  # CAV在位置5
    ]

    
    for cav_positions in cav_positions_list:
        config = {
            "num_vehicles": num_vehicles,
            "cav_positions": cav_positions,
            "topology_id": f"6car_cav{'+'.join([str(p+1) for p in cav_positions])}",
            "description": f"6辆车，CAV在第{'+'.join([str(p+1) for p in cav_positions])}位置"
        }
        configurations.append(config)

    return configurations


def create_fixed_idm_parameters():
    """创建固定的IDM参数组合（不使用自定义参数，使用环境默认值）"""
    
    # 不使用自定义IDM参数，使用环境默认值
    fixed_parameters = {
        "default": {
            "name": "默认型",
            "idm_params": None  # 使用环境默认参数
        }
    }
    
    return fixed_parameters


def setup_vehicles_with_fixed_parameters(topology_config, fixed_params, param_key, base_seed=42):
    """设置使用默认IDM参数的车辆配置"""
    np.random.seed(base_seed)
    
    num_vehicles = topology_config["num_vehicles"]
    cav_positions = topology_config["cav_positions"]
    
    manual_vehicles = []
    base_gap = 20.0
    
    # 生成车辆ID
    vehicle_ids = [f"hv{i}" if i not in cav_positions else f"cav{cav_positions.index(i)+1}" for i in range(num_vehicles)]
    
    # 为每辆车生成配置
    for i in range(num_vehicles):
        vehicle_id = vehicle_ids[i]
        is_cav = (i in cav_positions)
        x_position = (num_vehicles - 1 - i) * (base_gap + 5.0) + 100.0
        initial_speed = np.random.uniform(16.0, 20.0)
        
        vehicle_config = {
            "vehicle_id": vehicle_id,
            "is_cav": is_cav,
            "x_front": x_position,
            "speed": initial_speed,
            "length": 5.0
        }
        
        # 不再为HV设置自定义IDM参数，使用环境默认参数
        # 环境会自动为未指定IDM参数的HV车辆使用默认参数
        
        manual_vehicles.append(vehicle_config)
    
    return manual_vehicles


def create_cyber_attack_configurations():
    """创建网络攻击配置（考虑不同的攻击频率）"""
    
    attack_configs = [
        {
            "attack_type": "none",
            "name": "无攻击",
            "enable_cyber_attack": False
        }
    ]
    
    # 数据篡改攻击 - 不同频率
    for freq in [0.1, 0.2, 0.3]:
        attack_configs.append({
            "attack_type": "data_tampering",
            "name": f"数据篡改攻击(频率{freq})",
            "enable_cyber_attack": True,
            "attack_type_param": "data_tampering",
            "attack_frequency": freq,
            "attack_variances": {"speed": 1.5, "acceleration": 0.8},
            "attack_targets": ["speed", "acceleration"]
        })
    
    # 丢包攻击 - 不同概率
    for prob in [0.1, 0.2, 0.3]:
        attack_configs.append({
            "attack_type": "packet_drop",
            "name": f"丢包攻击(概率{prob})",
            "enable_cyber_attack": True,
            "attack_type_param": "packet_drop",
            "attack_frequency": prob,
            "attack_targets": ["speed", "acceleration"]
        })
    
    # 延迟攻击 - 不同频率
    for freq in [0.1, 0.2, 0.3]:
        attack_configs.append({
            "attack_type": "delay",
            "name": f"延迟攻击(频率{freq})",
            "enable_cyber_attack": True,
            "attack_type_param": "delay",
            "attack_frequency": freq,
            "delay_steps": 2,
            "attack_targets": ["speed", "acceleration"]
        })
    
    return attack_configs


class AdaptiveCAVController:
    """自适应CAV控制器"""
    
    def __init__(self):
        self.desired_time_headway = 1.7
        self.min_gap = 2.0
        self.max_decel = -2.2
        self.max_accel = 1.5
        self.comfort_decel = -1.6
        self.desired_speed = 18.0
        
    def get_action(self, cav_vehicle, front_vehicle):
        if front_vehicle is None:
            speed_error = self.desired_speed - cav_vehicle.speed
            return np.clip(0.4 * speed_error, self.max_decel, self.max_accel)
        
        gap = front_vehicle.x_front - front_vehicle.length - cav_vehicle.x_front
        relative_speed = cav_vehicle.speed - front_vehicle.speed
        desired_gap = self.min_gap + self.desired_time_headway * cav_vehicle.speed
        gap_error = gap - desired_gap
        
        if gap < self.min_gap:
            action = self.max_decel
        elif gap_error < 0:
            action = self.comfort_decel * (1 - gap_error / desired_gap) - 0.1 * relative_speed
        else:
            speed_error = self.desired_speed - cav_vehicle.speed
            action = 0.3 * speed_error - 0.06 * relative_speed
        
        return np.clip(action, self.max_decel, self.max_accel)


def load_lead_speed_data():
    """加载前车速度序列数据"""
    try:
        data = np.load("data/scaled_data.npy", allow_pickle=True)
        speed_sequences = (data + 1) * 8 + 5
        return speed_sequences
    except FileNotFoundError:
        print("❌ 未找到速度序列数据文件")
        return None


def run_cyber_attack_experiment(topology_config, fixed_params_key, fixed_params, attack_config, lead_sequence, experiment_id, base_seed=42):
    """运行网络攻击实验"""
    # 设置车辆（使用默认参数）
    manual_vehicles = setup_vehicles_with_fixed_parameters(topology_config, fixed_params, fixed_params_key, base_seed + experiment_id)
    
    # 创建环境
    num_vehicles = topology_config["num_vehicles"]
    cav_positions = topology_config["cav_positions"]
    cav_indices = cav_positions  # CAV的索引
    
    # 配置网络攻击参数
    env_kwargs = {
        "seed": base_seed + experiment_id,
        "num_vehicles": num_vehicles,
        "cav_indices": cav_indices,
        "dt": 0.1,
        "v_target": 18.0,
        "collision_penalty": 200.0,
        "enable_cyber_attack": attack_config["enable_cyber_attack"]
    }
    
    # 根据攻击类型添加特定参数
    if attack_config["enable_cyber_attack"]:
        env_kwargs["attack_type"] = attack_config["attack_type_param"]
        env_kwargs["attack_targets"] = attack_config["attack_targets"]
        
        if attack_config["attack_type_param"] == "data_tampering":
            env_kwargs["attack_frequency"] = attack_config["attack_frequency"]
            env_kwargs["attack_variances"] = attack_config["attack_variances"]
        elif attack_config["attack_type_param"] == "packet_drop":
            env_kwargs["attack_frequency"] = attack_config["attack_frequency"]
        elif attack_config["attack_type_param"] == "delay":
            env_kwargs["attack_frequency"] = attack_config["attack_frequency"]
            env_kwargs["delay_steps"] = attack_config["delay_steps"]
    
    env = CyberAttackEnv(**env_kwargs)
    
    # 重置环境
    reset_options = {
        "manual_vehicles": manual_vehicles,
        "lead_speed_sequence": lead_sequence
    }
    obs, info = env.reset()
    
    # 初始化CAV控制器
    cav_controller = AdaptiveCAVController()
    
    # 记录实验数据
    experiment_data = {
        "experiment_id": experiment_id,
        "topology_config": topology_config,
        "fixed_params_key": fixed_params_key,
        "fixed_params": fixed_params[fixed_params_key],
        "attack_config": attack_config,
        "vehicle_configs": manual_vehicles,
        "time_series": {},
        "simulation_info": {"collision_occurred": False, "final_step": 0, "duration": 0.0},
        "attack_statistics": {}
    }
    
    # 初始化车辆时间序列记录
    vehicle_names = [v["vehicle_id"] for v in manual_vehicles]
    for i, vehicle in enumerate(env.sim.vehicles):
        experiment_data["time_series"][vehicle_names[i]] = {
            "time": [], "position": [], "speed": [], "acceleration": [],
            "is_cav": vehicle.is_cav,
            "driver_profile": "默认型" if not vehicle.is_cav else "CAV",
            "profile_key": "default" if not vehicle.is_cav else "cav"
        }
        # 不再记录IDM参数，因为使用默认参数
        # 为CAV车辆添加网络传输状态记录
        if vehicle.is_cav:
            experiment_data["time_series"][vehicle_names[i]]["network_speed"] = []
            experiment_data["time_series"][vehicle_names[i]]["network_acceleration"] = []
            experiment_data["time_series"][vehicle_names[i]]["network_position"] = []
    
    # 运行仿真
    max_steps = min(len(lead_sequence), 600)  # 限制仿真步数以加快运行速度
    
    for step in range(max_steps):
        # CAV控制 - 控制所有CAV
        actions = []
        for cav_idx in cav_positions:
            cav_vehicle = env.sim.vehicles[cav_idx]
            front_vehicle = env.sim.vehicles[cav_idx - 1] if cav_idx > 0 else None
            
            if front_vehicle:
                cav_action = cav_controller.get_action(cav_vehicle, front_vehicle)
            else:
                speed_error = 18.0 - cav_vehicle.speed
                cav_action = 0.4 * speed_error
            
            cav_action = np.clip(cav_action, -2.0, 2.0)
            actions.append(cav_action)
        
        # 执行仿真步骤
        obs, reward, terminated, truncated, info = env.step(np.array(actions))
        
        # 记录车辆数据
        current_time = step * 0.1
        for i, vehicle in enumerate(env.sim.vehicles):
            vid = vehicle_names[i]
            experiment_data["time_series"][vid]["time"].append(current_time)
            experiment_data["time_series"][vid]["position"].append(vehicle.x_front)
            experiment_data["time_series"][vid]["speed"].append(vehicle.speed)
            experiment_data["time_series"][vid]["acceleration"].append(vehicle.acceleration)
            
            # 如果是CAV车辆，记录网络传输状态
            if vehicle.is_cav:
                cav_id = vehicle.vehicle_id
                if cav_id in env._cav_network_states:
                    network_state = env._cav_network_states[cav_id]
                    experiment_data["time_series"][vid]["network_speed"].append(network_state.get("speed", vehicle.speed))
                    experiment_data["time_series"][vid]["network_acceleration"].append(network_state.get("acceleration", vehicle.acceleration))
                    experiment_data["time_series"][vid]["network_position"].append(network_state.get("position", vehicle.x_front))
                else:
                    # 如果没有网络状态，使用真实状态
                    experiment_data["time_series"][vid]["network_speed"].append(vehicle.speed)
                    experiment_data["time_series"][vid]["network_acceleration"].append(vehicle.acceleration)
                    experiment_data["time_series"][vid]["network_position"].append(vehicle.x_front)
        
        if terminated:
            experiment_data["simulation_info"]["collision_occurred"] = True
            break
    
    final_step = step
    experiment_data["simulation_info"]["final_step"] = final_step
    experiment_data["simulation_info"]["duration"] = final_step * 0.1
    experiment_data["simulation_info"]["total_steps"] = max_steps  # 添加总步数信息
    
    # 获取攻击统计信息
    if attack_config["enable_cyber_attack"]:
        attack_stats = env.get_cyber_attack_stats()
        experiment_data["attack_statistics"] = attack_stats
    
    # 输出实验的step数量
    print(f"    实验 {experiment_id + 1}: 完成 {final_step + 1} 步仿真 (总步数: {max_steps})")
    
    return experiment_data


def create_unified_neural_network_dataset(experiments_data, window_size=5.0, dt=0.1):
    """创建统一的神经网络训练数据集
    输入使用CAV遭遇网络攻击的数据（网络传输状态）
    输出使用CAV未遭遇网络攻击的真实数据（真实状态）
    """
    window_steps = int(window_size / dt)  # 5.0秒 / 0.1秒 = 50步
    nn_dataset = []
    
    for exp in experiments_data:
        exp_id = exp["experiment_id"]
        topology_config = exp["topology_config"]
        attack_config = exp["attack_config"]
        time_series_data = exp["time_series"]
        
        num_vehicles = topology_config["num_vehicles"]
        cav_positions = topology_config["cav_positions"]
        topology_id = topology_config["topology_id"]
        
        vehicle_names = list(time_series_data.keys())
        cav_names = [vehicle_names[pos] for pos in cav_positions]
        last_vehicle_name = vehicle_names[-1]  # 最后一辆车作为参考车辆
        
        # 确定目标车辆（CAV后方的所有HV）
        # 找到CAV的位置
        cav_pos = cav_positions[0]  # 只有一个CAV
        # 目标车辆是CAV之后的所有HV车辆
        target_vehicles = [name for i, name in enumerate(vehicle_names) 
                          if i > cav_pos and not time_series_data[name]["is_cav"]]
        
        if len(target_vehicles) < 1:
            continue
        
        # 检查时间序列长度
        time_lengths = {name: len(time_series_data[name]["time"]) for name in vehicle_names}
        min_length = min(time_lengths.values())
        
        if min_length < window_steps:
            continue
        
        # 滑动窗口提取数据
        # 步长为窗口大小的一半，即25步，实现50%重叠
        for start_idx in range(0, min_length - window_steps + 1, window_steps // 2):
            end_idx = start_idx + window_steps
            
            try:
                # 获取参考车辆（最后一辆车）的时间序列（使用真实数据）
                ref_speeds = time_series_data[last_vehicle_name]["speed"][start_idx:end_idx]
                ref_accels = time_series_data[last_vehicle_name]["acceleration"][start_idx:end_idx]
                ref_positions = time_series_data[last_vehicle_name]["position"][start_idx:end_idx]
                
                # 获取所有CAV的时间序列
                all_cav_speeds = []
                all_cav_accels = []
                all_cav_positions = []
                all_cav_network_speeds = []
                all_cav_network_accels = []
                all_cav_network_positions = []
                
                for cav_name in cav_names:
                    cav_speeds = time_series_data[cav_name]["speed"][start_idx:end_idx]
                    cav_accels = time_series_data[cav_name]["acceleration"][start_idx:end_idx]
                    cav_positions = time_series_data[cav_name]["position"][start_idx:end_idx]
                    
                    all_cav_speeds.extend(cav_speeds)
                    all_cav_accels.extend(cav_accels)
                    all_cav_positions.extend(cav_positions)
                    
                    # 获取CAV的网络传输状态时间序列（用于输入）
                    if "network_speed" in time_series_data[cav_name]:
                        cav_network_speeds = time_series_data[cav_name]["network_speed"][start_idx:end_idx]
                        cav_network_accels = time_series_data[cav_name]["network_acceleration"][start_idx:end_idx]
                        cav_network_positions = time_series_data[cav_name]["network_position"][start_idx:end_idx]
                    else:
                        # 如果没有网络状态记录，使用真实状态
                        cav_network_speeds = cav_speeds
                        cav_network_accels = cav_accels
                        cav_network_positions = cav_positions
                    
                    all_cav_network_speeds.extend(cav_network_speeds)
                    all_cav_network_accels.extend(cav_network_accels)
                    all_cav_network_positions.extend(cav_network_positions)
                
                # 计算CAV相对于参考车辆的差值（使用网络传输状态作为输入）
                cav_speed_diff = np.array(all_cav_network_speeds) - np.tile(np.array(ref_speeds), len(cav_names))
                cav_accel_diff = np.array(all_cav_network_accels) - np.tile(np.array(ref_accels), len(cav_names))
                cav_position_diff = np.array(all_cav_network_positions) - np.tile(np.array(ref_positions), len(cav_names))
                
                # 构建输入特征：每辆CAV有250维特征（相对速度差50+相对加速度差50+相对位置差50+CAV速度50+CAV加速度50）
                input_features = []
                input_features.extend(cav_speed_diff.tolist())      # 相对速度差（50维）
                input_features.extend(cav_accel_diff.tolist())     # 相对加速度差（50维）
                input_features.extend(cav_position_diff.tolist())  # 相对位置差（50维）
                input_features.extend(all_cav_network_speeds)      # CAV网络传输速度（50维）
                input_features.extend(all_cav_network_accels)      # CAV网络传输加速度（50维）
                
                # 计算目标车辆的平均值（使用真实状态作为输出标签）
                target_speeds = []
                target_accels = []
                
                for vehicle_name in target_vehicles:
                    target_speeds.append(time_series_data[vehicle_name]["speed"][end_idx - 1])
                    target_accels.append(time_series_data[vehicle_name]["acceleration"][end_idx - 1])
                
                avg_target_speed = np.mean(target_speeds)
                avg_target_accel = np.mean(target_accels)
                
                # 创建数据样本
                sample = {
                    "experiment_id": exp_id,
                    "topology_id": topology_id,
                    "num_vehicles": num_vehicles,
                    "cav_positions": ",".join(map(str, cav_positions)),  # 修复：正确存储CAV位置
                    "window_start_time": start_idx * dt,
                    "window_end_time": end_idx * dt,
                    "window_size_seconds": window_size,
                    "num_input_features": len(input_features),
                    "input_feature_structure": f"CAV特征(相对速度差50+相对加速度差50+相对位置差50+CAV速度50+CAV加速度50=250维)",
                    "num_target_vehicles": len(target_vehicles),
                    "target_vehicles": ",".join(target_vehicles),
                    "output_target_avg_speed": avg_target_speed,
                    "output_target_avg_acceleration": avg_target_accel,
                    "idm_strategy": exp["fixed_params"]["name"],
                    "attack_type": attack_config["attack_type"],
                    "attack_name": attack_config["name"]
                }
                
                # 添加攻击统计信息（如果有）
                if "attack_statistics" in exp and exp["attack_statistics"]:
                    stats = exp["attack_statistics"]
                    if "statistics" in stats:
                        sample["total_attacks"] = stats["statistics"].get("total_attacks", 0)
                        sample["actual_attack_rate"] = stats["statistics"].get("actual_attack_rate", 0.0)
                        
                        # 添加各攻击类型统计
                        attack_type_dist = stats["statistics"].get("attack_type_distribution", {})
                        for attack_type, dist in attack_type_dist.items():
                            sample[f"{attack_type}_count"] = dist.get("count", 0)
                            sample[f"{attack_type}_percentage"] = dist.get("percentage", 0.0)
                
                # 添加输入特征
                for i, feature_val in enumerate(input_features):
                    sample[f"input_feature_{i:03d}"] = feature_val
                
                nn_dataset.append(sample)
                
            except Exception as e:
                continue
    
    return pd.DataFrame(nn_dataset)


def main():
    """主函数：生成完整的网络攻击训练数据集"""
    print("🚀 生成完整的网络攻击训练数据集")
    print("=" * 50)
    
    # 创建输出目录
    os.makedirs("training_data", exist_ok=True)
    
    # 加载速度序列数据
    speed_sequences = load_lead_speed_data()
    if speed_sequences is None:
        return
    
    # 生成配置
    topology_configs = generate_topology_configurations()
    fixed_idm_params = create_fixed_idm_parameters()
    attack_configs = create_cyber_attack_configurations()
    
    # 实验配置
    experiments_per_config = 30 #每种配置运行30次实验（增加样本量）
    total_experiments = len(topology_configs) * len(fixed_idm_params) * len(attack_configs) * experiments_per_config
    
    print(f"📊 实验配置:")
    print(f"   - 拓扑配置数: {len(topology_configs)}")
    print(f"   - 固定参数组数: {len(fixed_idm_params)}")
    print(f"   - 攻击类型数: {len(attack_configs)}")
    print(f"   - 每配置实验数: {experiments_per_config}")
    print(f"   - 总实验数: {total_experiments}")
    
    # 随机选择速度序列
    np.random.seed(500)
    selected_indices = np.random.choice(speed_sequences.shape[0], size=total_experiments, replace=False)
    
    # 运行所有实验
    all_experiments = []
    exp_counter = 0
    
    print(f"\n🔬 开始运行实验...")
    
    for topo_config in topology_configs:
        for param_key in fixed_idm_params:
            for attack_config in attack_configs:
                for rep in range(experiments_per_config):
                    lead_sequence = speed_sequences[selected_indices[exp_counter]].tolist()
                    
                    try:
                        exp_data = run_cyber_attack_experiment(
                            topo_config, param_key, fixed_idm_params, attack_config, lead_sequence, exp_counter, base_seed=500
                        )
                        all_experiments.append(exp_data)
                        # print(f"    ✅ 实验 {exp_counter + 1}/{total_experiments} 完成")
                    except Exception as e:
                        print(f"    ❌ 实验 {exp_counter + 1}/{total_experiments} 失败: {e}")
                    
                    exp_counter += 1
    
    print(f"\n📈 实验完成统计:")
    print(f"   成功实验数: {len(all_experiments)}/{total_experiments}")
    print(f"   成功率: {len(all_experiments)/total_experiments*100:.1f}%")
    
    # 输出每个实验的step数量统计
    if all_experiments:
        step_counts = [exp["simulation_info"]["final_step"] + 1 for exp in all_experiments]
        print(f"\n📊 仿真步数统计:")
        print(f"   平均步数: {np.mean(step_counts):.1f}")
        print(f"   最小步数: {np.min(step_counts)}")
        print(f"   最大步数: {np.max(step_counts)}")
        print(f"   步数标准差: {np.std(step_counts):.1f}")
    
    # 生成神经网络数据集
    print(f"\n🧠 生成神经网络训练数据集...")
    nn_dataset = create_unified_neural_network_dataset(all_experiments)
    
    # 保存数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存完整实验数据
    with open(f"training_data/cyber_attack_experiments_complete_{timestamp}.pkl", 'wb') as f:
        pickle.dump(all_experiments, f)
    
    # 保存神经网络数据集
    nn_dataset.to_csv(f"training_data/cyber_attack_neural_network_dataset_complete_{timestamp}.csv", index=False)
    with open(f"training_data/cyber_attack_neural_network_dataset_complete_{timestamp}.pkl", 'wb') as f:
        pickle.dump(nn_dataset, f)
    
    print(f"\n🎉 网络攻击训练数据生成完成！")
    print(f"   - 实验数据: cyber_attack_experiments_complete_{timestamp}.pkl")
    print(f"   - 神经网络数据集: cyber_attack_neural_network_dataset_complete_{timestamp}.csv")
    print(f"   - 样本数量: {len(nn_dataset)}")
    print(f"   - 说明：通过滑动窗口技术，{len(all_experiments)}个实验生成了{len(nn_dataset)}个训练样本")


if __name__ == "__main__":
    main()