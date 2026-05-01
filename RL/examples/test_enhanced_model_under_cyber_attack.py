#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强模型网络攻击测试脚本
在多种网络攻击环境下测试已训练的增强模型效果

功能：
1. 加载已训练的增强模型
2. 在多种攻击类型下进行测试（无攻击、数据篡改、丢包、延迟）
3. 评估模型在不同攻击环境下的鲁棒性
4. 生成详细的性能报告和可视化结果
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目路径
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sim_env.envs.cyber_attack_env import CyberAttackEnv
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 导入增强模型类（与训练脚本相同）
class TopologyAwareLSTM(nn.Module):
    """拓扑感知LSTM模型（与训练脚本相同）"""
    
    def __init__(self, input_size=250, topology_size=2, hidden_size=64, num_layers=2, 
                 output_size=2, dropout=0.2, use_topology=True):
        super(TopologyAwareLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_topology = use_topology
        self.sequence_length = 50
        self.feature_dim = 5  # 固定为5维：速度差、加速度差、位置差、CAV速度、CAV加速度
        
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        if self.use_topology:
            self.topology_embedding = nn.Sequential(
                nn.Linear(topology_size, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            final_hidden_size = hidden_size + hidden_size // 4
        else:
            final_hidden_size = hidden_size
        
        self.fc_layers = nn.Sequential(
            nn.Linear(final_hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, output_size)
        )
        
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)
                
        if self.use_topology:
            for layer in self.topology_embedding:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    layer.bias.data.fill_(0)
    
    def forward(self, x, topology_info=None):
        batch_size = x.size(0)
        
        # 250维输入：5特征 x 50时间步
        speed_diff = x[:, 0:50]
        accel_diff = x[:, 50:100]
        position_diff = x[:, 100:150]
        cav_speed = x[:, 150:200]
        cav_accel = x[:, 200:250]
        x_reshaped = torch.stack([speed_diff, accel_diff, position_diff, cav_speed, cav_accel], dim=2)
        
        lstm_out, (hidden, cell) = self.lstm(x_reshaped)
        last_output = lstm_out[:, -1, :]
        
        if self.use_topology and topology_info is not None:
            topology_embedded = self.topology_embedding(topology_info)
            combined_features = torch.cat([last_output, topology_embedded], dim=1)
        else:
            combined_features = last_output
        
        output = self.fc_layers(combined_features)
        
        return output


def load_enhanced_model():
    """加载增强模型"""
    print("📥 加载增强模型...")
    
    # 查找最新的增强模型文件
    model_files = [f for f in os.listdir("models") if f.startswith("enhanced_traffic_lstm_model_")]
    if not model_files:
        print("❌ 未找到增强模型文件")
        return None, None, None, None
    
    latest_model = sorted(model_files)[-1]
    timestamp = latest_model.replace("enhanced_traffic_lstm_model_", "").replace(".pth", "")
    
    model_path = f"models/enhanced_traffic_lstm_model_{timestamp}.pth"
    scaler_path = f"models/enhanced_scalers_{timestamp}.pkl"
    
    print(f"   使用模型: {latest_model}")
    
    # 加载标准化器信息
    try:
        with open(scaler_path, 'rb') as f:
            scalers_info = pickle.load(f)
            use_topology = scalers_info.get('use_topology', True)
    except Exception as e:
        print(f"   警告: 无法读取模型信息，使用默认设置: {e}")
        use_topology = True
    
    # 创建增强模型
    print("   创建增强模型结构")
    model = TopologyAwareLSTM(
        input_size=250,
        topology_size=2,
        hidden_size=96,
        num_layers=3,
        output_size=2,
        dropout=0.3,
        use_topology=use_topology
    )
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # 加载标准化器
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
        feature_scaler = scalers['feature_scaler']
        target_scaler = scalers['target_scaler']
        topology_scaler = scalers.get('topology_scaler', None)
    
    print("   ✅ 增强模型加载成功")
    return model, feature_scaler, target_scaler, topology_scaler


def load_lead_speed_data():
    """加载前车速度序列数据"""
    try:
        data = np.load("data/scaled_data.npy", allow_pickle=True)
        speed_sequences = (data + 1) * 8 + 5
        print(f"✅ 成功加载速度序列数据，形状: {speed_sequences.shape}")
        return speed_sequences
    except FileNotFoundError:
        print("❌ 未找到速度序列数据文件")
        return None


def create_test_vehicle_configs():
    """创建测试用的车辆配置"""
    # 使用更合理的IDM参数配置，避免碰撞，增加车辆间距
    base_configs = [
        {
            "vehicle_id": "hv0",
            "is_cav": False,
            "x_front": 150.0,  # 增加初始间距
            "speed": 18.0,
            "length": 5.0,
            "idm_params": {
                "desired_speed": 18.0,
                "minimum_spacing": 2.0,
                "desired_time_headway": 1.5,
                "max_acceleration": 1.0,
                "comfortable_deceleration": 1.5,
                "delta": 4.0
            }
        },
        {
            "vehicle_id": "cav1",
            "is_cav": True,
            "x_front": 120.0,  # 增加初始间距
            "speed": 18.0,
            "length": 5.0
        },
        {
            "vehicle_id": "hv2",
            "is_cav": False,
            "x_front": 90.0,   # 增加初始间距
            "speed": 18.0,
            "length": 5.0,
            "idm_params": {
                "desired_speed": 18.0,
                "minimum_spacing": 2.0,
                "desired_time_headway": 1.5,
                "max_acceleration": 1.0,
                "comfortable_deceleration": 1.5,
                "delta": 4.0
            }
        },
        {
            "vehicle_id": "hv3",
            "is_cav": False,
            "x_front": 60.0,   # 增加初始间距
            "speed": 18.0,
            "length": 5.0,
            "idm_params": {
                "desired_speed": 18.0,
                "minimum_spacing": 2.0,
                "desired_time_headway": 1.5,
                "max_acceleration": 1.0,
                "comfortable_deceleration": 1.5,
                "delta": 4.0
            }
        }
    ]
    
    return base_configs


def prepare_input_features(time_series_data, cav_position, last_vehicle_name, window_size=50, use_attacked_data=False, env=None):
    """准备250维输入特征"""
    vehicle_names = list(time_series_data.keys())
    cav_name = vehicle_names[cav_position]
    
    # 确保有足够的数据点
    if len(time_series_data[last_vehicle_name]["speed"]) < window_size:
        return None
    
    # 获取最后window_size个时间步的数据（参考车辆数据始终使用真实数据）
    ref_speeds = time_series_data[last_vehicle_name]["speed"][-window_size:]
    ref_accels = time_series_data[last_vehicle_name]["acceleration"][-window_size:]
    ref_positions = time_series_data[last_vehicle_name]["position"][-window_size:]
    
    # 根据是否使用攻击数据来获取CAV数据
    if use_attacked_data and env is not None:
        # 使用被网络攻击污染的数据
        cav_id = cav_name  # 假设cav_name就是cav_id
        
        # 尝试从环境获取网络传输状态历史数据
        if hasattr(env, 'get_cav_network_states_history'):
            try:
                network_history_dict = env.get_cav_network_states_history(cav_id)
                if cav_id in network_history_dict and len(network_history_dict[cav_id]) >= window_size:
                    network_history = network_history_dict[cav_id]
                    # 从历史数据中获取最后window_size个时间步的数据
                    # 处理可能为None的值
                    cav_speeds = []
                    cav_accels = []
                    cav_positions = []
                    
                    for state in network_history[-window_size:]:
                        # 对于None值，使用时间序列中的对应值作为替代
                        if state['speed'] is not None:
                            cav_speeds.append(state['speed'])
                        else:
                            # 使用时间序列数据中的对应值
                            idx = len(cav_speeds) - window_size
                            cav_speeds.append(time_series_data[cav_name]["speed"][idx])
                            
                        if state['acceleration'] is not None:
                            cav_accels.append(state['acceleration'])
                        else:
                            idx = len(cav_accels) - window_size
                            cav_accels.append(time_series_data[cav_name]["acceleration"][idx])
                            
                        if state['position'] is not None:
                            cav_positions.append(state['position'])
                        else:
                            idx = len(cav_positions) - window_size
                            cav_positions.append(time_series_data[cav_name]["position"][idx])
                else:
                    # 如果历史数据不足，回退到使用时间序列数据
                    cav_speeds = time_series_data[cav_name]["speed"][-window_size:]
                    cav_accels = time_series_data[cav_name]["acceleration"][-window_size:]
                    cav_positions = time_series_data[cav_name]["position"][-window_size:]
            except Exception as e:
                print(f"获取网络状态历史数据时出错: {e}")
                # 回退到使用时间序列数据
                cav_speeds = time_series_data[cav_name]["speed"][-window_size:]
                cav_accels = time_series_data[cav_name]["acceleration"][-window_size:]
                cav_positions = time_series_data[cav_name]["position"][-window_size:]
        elif hasattr(env, '_cav_network_states_history') and cav_id in env._cav_network_states_history:
            network_history = env._cav_network_states_history[cav_id]
            
            # 确保有足够的历史数据
            if len(network_history) >= window_size:
                # 从历史数据中获取最后window_size个时间步的数据
                # 处理可能为None的值
                cav_speeds = []
                cav_accels = []
                cav_positions = []
                
                for state in network_history[-window_size:]:
                    # 对于None值，使用时间序列中的对应值作为替代
                    if state['speed'] is not None:
                        cav_speeds.append(state['speed'])
                    else:
                        # 使用时间序列数据中的对应值
                        idx = len(cav_speeds) - window_size
                        cav_speeds.append(time_series_data[cav_name]["speed"][idx] if abs(idx) < len(time_series_data[cav_name]["speed"]) else time_series_data[cav_name]["speed"][-1])
                        
                    if state['acceleration'] is not None:
                        cav_accels.append(state['acceleration'])
                    else:
                        idx = len(cav_accels) - window_size
                        cav_accels.append(time_series_data[cav_name]["acceleration"][idx] if abs(idx) < len(time_series_data[cav_name]["acceleration"]) else time_series_data[cav_name]["acceleration"][-1])
                        
                    if state['position'] is not None:
                        cav_positions.append(state['position'])
                    else:
                        idx = len(cav_positions) - window_size
                        cav_positions.append(time_series_data[cav_name]["position"][idx] if abs(idx) < len(time_series_data[cav_name]["position"]) else time_series_data[cav_name]["position"][-1])
            else:
                # 如果历史数据不足，回退到使用时间序列数据
                cav_speeds = time_series_data[cav_name]["speed"][-window_size:]
                cav_accels = time_series_data[cav_name]["acceleration"][-window_size:]
                cav_positions = time_series_data[cav_name]["position"][-window_size:]
        else:
            # 如果没有网络状态历史数据，尝试从当前状态获取
            if hasattr(env, '_cav_network_states') and cav_id in env._cav_network_states:
                # 从网络传输状态获取被攻击的数据
                network_states = env._cav_network_states[cav_id]
                # 处理可能为None的值
                speed_val = network_states['speed'] if network_states['speed'] is not None else time_series_data[cav_name]["speed"][-1]
                accel_val = network_states['acceleration'] if network_states['acceleration'] is not None else time_series_data[cav_name]["acceleration"][-1]
                position_val = network_states['position'] if network_states['position'] is not None else time_series_data[cav_name]["position"][-1]
                
                cav_speeds = [speed_val] * window_size  # 简化处理
                cav_accels = [accel_val] * window_size
                cav_positions = [position_val] * window_size
            else:
                # 如果没有网络状态数据，回退到使用时间序列数据
                cav_speeds = time_series_data[cav_name]["speed"][-window_size:]
                cav_accels = time_series_data[cav_name]["acceleration"][-window_size:]
                cav_positions = time_series_data[cav_name]["position"][-window_size:]
    else:
        # 使用真实数据
        cav_speeds = time_series_data[cav_name]["speed"][-window_size:]
        cav_accels = time_series_data[cav_name]["acceleration"][-window_size:]
        cav_positions = time_series_data[cav_name]["position"][-window_size:]
    
    # 确保所有数组长度一致
    min_len = min(len(ref_speeds), len(ref_accels), len(ref_positions), 
                  len(cav_speeds), len(cav_accels), len(cav_positions))
    
    if min_len < window_size:
        # 如果长度不足，回退到使用时间序列数据
        cav_speeds = time_series_data[cav_name]["speed"][-window_size:]
        cav_accels = time_series_data[cav_name]["acceleration"][-window_size:]
        cav_positions = time_series_data[cav_name]["position"][-window_size:]
    
    # 计算相对差值
    cav_speed_diff = np.array(cav_speeds[-window_size:]) - np.array(ref_speeds[-window_size:])
    cav_accel_diff = np.array(cav_accels[-window_size:]) - np.array(ref_accels[-window_size:])
    cav_position_diff = np.array(cav_positions[-window_size:]) - np.array(ref_positions[-window_size:])
    
    # 构建250维输入特征
    input_features = []
    input_features.extend(cav_speed_diff.tolist())      # 50维：CAV相对速度差
    input_features.extend(cav_accel_diff.tolist())     # 50维：CAV相对加速度差
    input_features.extend(cav_position_diff.tolist())  # 50维：CAV相对位置差
    input_features.extend(cav_speeds[-window_size:])   # 50维：CAV绝对速度
    input_features.extend(cav_accels[-window_size:])   # 50维：CAV绝对加速度
    
    return np.array(input_features)


def run_attack_test(model, feature_scaler, target_scaler, topology_scaler, 
                   attack_type="none", attack_params=None, use_attacked_data=False):
    """运行攻击测试"""
    print(f"\n🛡️  攻击测试: {attack_type}")
    print(f"   使用被攻击数据: {use_attacked_data}")
    
    # 加载速度序列数据
    speed_sequences = load_lead_speed_data()
    if speed_sequences is None:
        return None
    
    # 选择一个速度序列用于测试，使用较平稳的部分
    lead_sequence = speed_sequences[100:600].tolist()  # 使用中间较平稳的部分
    
    # 创建车辆配置
    manual_vehicles = create_test_vehicle_configs()
    num_vehicles = len(manual_vehicles)
    cav_position = 1  # CAV在第2位
    cav_indices = [cav_position]
    
    # 创建网络攻击环境
    env = CyberAttackEnv(
        seed=42,
        num_vehicles=num_vehicles,
        cav_indices=cav_indices,
        dt=0.1,
        v_target=18.0,
        collision_penalty=200.0
    )
    
    # 设置攻击配置
    if attack_type != "none":
        attack_config = {
            "enable_cyber_attack": True,
            "attack_type": attack_type,
            "attack_frequency": 0.15,  # 15%的攻击频率
            "attack_mean": 0.0,
            "attack_variances": {"speed": 0.1, "position": 0.1, "acceleration": 0.1}
        }
        
        if attack_type == "packet_drop":
            attack_config["attack_frequency"] = 0.3  # 统一使用attack_frequency
        elif attack_type == "delay":
            attack_config["delay_steps"] = 3
            
        env.set_cyber_attack_config(**attack_config)
    
    # 重置环境
    reset_options = {
        "manual_vehicles": manual_vehicles,
        "lead_speed_sequence": lead_sequence
    }
    obs, info = env.reset()
    
    # 记录测试数据
    test_data = {
        "time_series": {},
        "predictions": [],
        "ground_truth": [],
        "attack_info": {
            "type": attack_type,
            "params": attack_params
        },
        "use_attacked_data": use_attacked_data  # 记录是否使用了攻击数据
    }
    
    # 初始化车辆时间序列记录
    vehicle_names = [v["vehicle_id"] for v in manual_vehicles]
    last_vehicle_name = vehicle_names[-1]
    
    for i, vehicle in enumerate(env.sim.vehicles):
        test_data["time_series"][vehicle_names[i]] = {
            "time": [], "position": [], "speed": [], "acceleration": [],
            "is_cav": vehicle.is_cav
        }
    
    # 运行仿真
    max_steps = min(len(lead_sequence), 500)  # 限制测试步数
    
    for step in range(max_steps):
        # 执行仿真步骤（使用简单的跟驰控制）
        cav_vehicle = env.sim.vehicles[cav_position]
        front_vehicle = env.sim.vehicles[cav_position - 1] if cav_position > 0 else None
        
        if front_vehicle:
            # 简单的跟驰控制，使用更温和的控制策略
            gap = front_vehicle.x_front - front_vehicle.length - cav_vehicle.x_front
            desired_gap = 2.0 + 1.5 * cav_vehicle.speed
            gap_error = gap - desired_gap
            speed_error = 18.0 - cav_vehicle.speed
            
            # 使用更小的控制增益以避免剧烈动作
            action = 0.2 * speed_error - 0.05 * gap_error
        else:
            speed_error = 18.0 - cav_vehicle.speed
            action = 0.3 * speed_error
        
        action = float(np.clip(action, -1.0, 1.0))  # 限制动作范围
        
        obs, reward, terminated, truncated, info = env.step({})  # 修改为传递action而不是空字典
        
        # 记录车辆数据
        current_time = step * 0.1
        for i, vehicle in enumerate(env.sim.vehicles):
            vid = vehicle_names[i]
            test_data["time_series"][vid]["time"].append(current_time)
            test_data["time_series"][vid]["position"].append(vehicle.x_front)
            test_data["time_series"][vid]["speed"].append(vehicle.speed)
            test_data["time_series"][vid]["acceleration"].append(vehicle.acceleration)
        
        # 每50个时间步进行一次预测（模拟滑动窗口）
        # 只有在有足够的数据时才进行预测
        if step >= 50 and step % 10 == 0:
            # 准备输入特征，根据参数决定是否使用被攻击的数据
            input_features = prepare_input_features(
                test_data["time_series"], cav_position, last_vehicle_name,
                use_attacked_data=use_attacked_data, env=env if use_attacked_data else None
            )
            
            # 检查是否有足够的数据
            if input_features is not None and len(test_data["time_series"][last_vehicle_name]["speed"]) >= 50:
                # 标准化输入特征
                input_scaled = feature_scaler.transform(input_features.reshape(1, -1))
                
                # 准备拓扑信息
                topology_info = np.array([[num_vehicles, cav_position]])
                if topology_scaler:
                    topology_scaled = topology_scaler.transform(topology_info)
                else:
                    topology_scaled = topology_info
                
                # 模型预测
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(input_scaled)
                    topology_tensor = torch.FloatTensor(topology_scaled)
                    
                    prediction = model(input_tensor, topology_tensor)
                    prediction_np = target_scaler.inverse_transform(prediction.numpy())
                    
                    # 记录预测结果
                    test_data["predictions"].append({
                        "time": current_time,
                        "predicted_speed": prediction_np[0, 0],
                        "predicted_accel": prediction_np[0, 1]
                    })
                    
                    # 记录真实值（目标车辆的平均速度和加速度）
                    # 这里我们使用CAV后面和参考车辆前面的车辆作为目标车辆
                    target_vehicles = vehicle_names[cav_position + 1:-1]  # CAV和参考车辆之间的车辆
                    if len(target_vehicles) > 0:
                        target_speeds = [test_data["time_series"][name]["speed"][-1] for name in target_vehicles]
                        target_accels = [test_data["time_series"][name]["acceleration"][-1] for name in target_vehicles]
                        avg_target_speed = np.mean(target_speeds)
                        avg_target_accel = np.mean(target_accels)
                        
                        test_data["ground_truth"].append({
                            "time": current_time,
                            "true_speed": avg_target_speed,
                            "true_accel": avg_target_accel
                        })
        
        if terminated:
            print(f"    ⚠️  测试在第 {step} 步发生碰撞")
            break
    
    print(f"    ✅ 攻击测试完成，生成 {len(test_data['predictions'])} 个预测")
    
    return test_data, env  # 返回env以便获取攻击信息


def analyze_attack_performance(test_results):
    """分析攻击环境下的模型性能"""
    print("\n📊 攻击环境模型性能分析")
    print("=" * 60)
    
    all_metrics = {}
    
    for attack_type, test_data in test_results.items():
        if not test_data or len(test_data["predictions"]) == 0:
            continue
            
        # 提取预测值和真实值
        predicted_speeds = [p["predicted_speed"] for p in test_data["predictions"]]
        predicted_accels = [p["predicted_accel"] for p in test_data["predictions"]]
        
        true_speeds = [g["true_speed"] for g in test_data["ground_truth"]]
        true_accels = [g["true_accel"] for g in test_data["ground_truth"]]
        
        # 确保长度一致
        min_len = min(len(predicted_speeds), len(true_speeds))
        if min_len == 0:
            continue
            
        predicted_speeds = predicted_speeds[:min_len]
        predicted_accels = predicted_accels[:min_len]
        true_speeds = true_speeds[:min_len]
        true_accels = true_accels[:min_len]
        
        # 计算评估指标
        speed_mae = mean_absolute_error(true_speeds, predicted_speeds)
        speed_rmse = np.sqrt(mean_squared_error(true_speeds, predicted_speeds))
        speed_r2 = r2_score(true_speeds, predicted_speeds)
        
        accel_mae = mean_absolute_error(true_accels, predicted_accels)
        accel_rmse = np.sqrt(mean_squared_error(true_accels, predicted_accels))
        accel_r2 = r2_score(true_accels, predicted_accels)
        
        # 存储指标
        all_metrics[attack_type] = {
            "speed_mae": speed_mae,
            "speed_rmse": speed_rmse,
            "speed_r2": speed_r2,
            "accel_mae": accel_mae,
            "accel_rmse": accel_rmse,
            "accel_r2": accel_r2,
            "sample_count": min_len
        }
        
        print(f"\n🛡️  {attack_type}攻击:")
        print(f"   样本数量: {min_len}")
        print(f"   速度预测 - MAE: {speed_mae:.4f}, RMSE: {speed_rmse:.4f}, R²: {speed_r2:.4f}")
        print(f"   加速度预测 - MAE: {accel_mae:.4f}, RMSE: {accel_rmse:.4f}, R²: {accel_r2:.4f}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"results/cyber_attack_enhanced_model_test_{timestamp}.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(all_metrics, f)
    
    print(f"\n💾 攻击测试结果已保存: {results_path}")
    
    return all_metrics


def plot_attack_performance(all_metrics):
    """绘制攻击环境下的性能对比图"""
    if not all_metrics:
        return
    
    # 创建性能对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('增强模型在不同网络攻击下的性能对比', fontsize=16)
    
    attack_types = list(all_metrics.keys())
    x_pos = np.arange(len(attack_types))
    
    # 速度预测指标
    speed_mae = [all_metrics[at]["speed_mae"] for at in attack_types]
    speed_r2 = [all_metrics[at]["speed_r2"] for at in attack_types]
    
    # 加速度预测指标
    accel_mae = [all_metrics[at]["accel_mae"] for at in attack_types]
    accel_r2 = [all_metrics[at]["accel_r2"] for at in attack_types]
    
    # 绘制速度MAE
    axes[0, 0].bar(x_pos, speed_mae, color='skyblue')
    axes[0, 0].set_title('速度预测MAE')
    axes[0, 0].set_ylabel('MAE (m/s)')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(attack_types, rotation=45)
    
    # 绘制速度R²
    axes[0, 1].bar(x_pos, speed_r2, color='lightgreen')
    axes[0, 1].set_title('速度预测R²')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(attack_types, rotation=45)
    
    # 绘制加速度MAE
    axes[1, 0].bar(x_pos, accel_mae, color='orange')
    axes[1, 0].set_title('加速度预测MAE')
    axes[1, 0].set_ylabel('MAE (m/s²)')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(attack_types, rotation=45)
    
    # 绘制加速度R²
    axes[1, 1].bar(x_pos, accel_r2, color='pink')
    axes[1, 1].set_title('加速度预测R²')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(attack_types, rotation=45)
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"results/cyber_attack_enhanced_model_performance_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 性能对比图已保存: {plot_path}")


def plot_prediction_vs_ground_truth(test_results):
    """绘制预测值与真实值的对比曲线图"""
    if not test_results:
        return
    
    # 为每种攻击类型创建预测vs真实值的对比图
    for attack_type, test_data in test_results.items():
        if not test_data or len(test_data["predictions"]) == 0:
            continue
            
        # 提取预测值和真实值
        predicted_speeds = [p["predicted_speed"] for p in test_data["predictions"]]
        predicted_accels = [p["predicted_accel"] for p in test_data["predictions"]]
        times = [p["time"] for p in test_data["predictions"]]
        
        true_speeds = [g["true_speed"] for g in test_data["ground_truth"]]
        true_accels = [g["true_accel"] for g in test_data["ground_truth"]]
        
        # 确保长度一致
        min_len = min(len(predicted_speeds), len(true_speeds))
        if min_len == 0:
            continue
            
        predicted_speeds = predicted_speeds[:min_len]
        predicted_accels = predicted_accels[:min_len]
        true_speeds = true_speeds[:min_len]
        true_accels = true_accels[:min_len]
        times = times[:min_len]
        
        # 创建对比图
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'增强模型预测值与真实值对比 - {attack_type}攻击', fontsize=16)
        
        # 速度对比
        axes[0].plot(times, true_speeds, label='真实速度', color='blue', linewidth=2)
        axes[0].plot(times, predicted_speeds, label='预测速度', color='red', linestyle='--', linewidth=2)
        axes[0].set_title('速度预测对比')
        axes[0].set_xlabel('时间 (s)')
        axes[0].set_ylabel('速度 (m/s)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 加速度对比
        axes[1].plot(times, true_accels, label='真实加速度', color='blue', linewidth=2)
        axes[1].plot(times, predicted_accels, label='预测加速度', color='red', linestyle='--', linewidth=2)
        axes[1].set_title('加速度预测对比')
        axes[1].set_xlabel('时间 (s)')
        axes[1].set_ylabel('加速度 (m/s²)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"results/prediction_vs_ground_truth_{attack_type}_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 {attack_type}攻击预测对比图已保存: {plot_path}")


def plot_detailed_comparison(test_results):
    """绘制详细对比图：真实值、网络攻击值、预测值"""
    if not test_results:
        return
    
    # 为每种攻击类型创建详细对比图
    for attack_type, test_data in test_results.items():
        if not test_data or len(test_data["predictions"]) == 0:
            continue
            
        # 提取时间序列数据
        time_series = test_data["time_series"]
        vehicle_names = list(time_series.keys())
        
        # 找到CAV和目标车辆
        cav_name = None
        target_vehicles = []
        for name, data in time_series.items():
            if data["is_cav"]:
                cav_name = name
            else:
                # 非CAV车辆都可能是目标车辆
                target_vehicles.append(name)
        
        if not cav_name or not target_vehicles:
            continue
            
        # 获取CAV数据
        cav_times = time_series[cav_name]["time"]
        cav_speeds = time_series[cav_name]["speed"]
        cav_accels = time_series[cav_name]["acceleration"]
        
        # 获取目标车辆数据（这里我们选择第一个目标车辆作为示例）
        target_name = target_vehicles[0] if target_vehicles else None
        if not target_name or target_name not in time_series:
            continue
            
        target_times = time_series[target_name]["time"]
        target_speeds = time_series[target_name]["speed"]
        target_accels = time_series[target_name]["acceleration"]
        
        # 获取预测数据
        predicted_speeds = [p["predicted_speed"] for p in test_data["predictions"]]
        predicted_accels = [p["predicted_accel"] for p in test_data["predictions"]]
        pred_times = [p["time"] for p in test_data["predictions"]]
        
        # 获取真实值数据
        true_speeds = [g["true_speed"] for g in test_data["ground_truth"]]
        true_accels = [g["true_accel"] for g in test_data["ground_truth"]]
        true_times = [g["time"] for g in test_data["ground_truth"]]
        
        # 确保长度一致
        min_len = min(len(pred_times), len(true_times))
        if min_len == 0:
            continue
            
        pred_times = pred_times[:min_len]
        predicted_speeds = predicted_speeds[:min_len]
        predicted_accels = predicted_accels[:min_len]
        true_times = true_times[:min_len]
        true_speeds = true_speeds[:min_len]
        true_accels = true_accels[:min_len]
        
        # 创建详细对比图
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f'详细对比图 - {attack_type}攻击\nCAV: {cav_name}, 目标车辆: {target_name}', fontsize=16)
        
        # 速度对比
        axes[0].plot(cav_times, cav_speeds, label=f'CAV ({cav_name}) 速度', color='blue', linewidth=1, alpha=0.7)
        axes[0].plot(target_times, target_speeds, label=f'目标车辆 ({target_name}) 速度', color='green', linewidth=1, alpha=0.7)
        axes[0].plot(true_times, true_speeds, label='真实平均速度', color='black', linewidth=2)
        axes[0].plot(pred_times, predicted_speeds, label='预测速度', color='red', linestyle='--', linewidth=2)
        axes[0].set_title('速度对比')
        axes[0].set_xlabel('时间 (s)')
        axes[0].set_ylabel('速度 (m/s)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 加速度对比
        axes[1].plot(cav_times, cav_accels, label=f'CAV ({cav_name}) 加速度', color='blue', linewidth=1, alpha=0.7)
        axes[1].plot(target_times, target_accels, label=f'目标车辆 ({target_name}) 加速度', color='green', linewidth=1, alpha=0.7)
        axes[1].plot(true_times, true_accels, label='真实平均加速度', color='black', linewidth=2)
        axes[1].plot(pred_times, predicted_accels, label='预测加速度', color='red', linestyle='--', linewidth=2)
        axes[1].set_title('加速度对比')
        axes[1].set_xlabel('时间 (s)')
        axes[1].set_ylabel('加速度 (m/s²)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"results/detailed_comparison_{attack_type}_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 {attack_type}攻击详细对比图已保存: {plot_path}")


def main():
    """主函数"""
    print("🛡️  增强模型网络攻击测试")
    print("=" * 60)
    
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    # 1. 加载增强模型
    model, feature_scaler, target_scaler, topology_scaler = load_enhanced_model()
    if model is None:
        return
    
    # 2. 定义攻击类型和参数
    attack_scenarios = {
        "none": None,
        "data_tampering": {
            "attack_frequency": 0.15,
            "attack_mean": 0.0,
            "attack_variances": {"speed": 0.1, "position": 0.1, "acceleration": 0.1}
        },
        "packet_drop": {
            "attack_frequency": 0.3  # 统一使用attack_frequency作为攻击概率
        },
        "delay": {
            "attack_frequency": 0.15,
            "delay_steps": 3
        }
    }
    
    # 3. 运行各种攻击测试（使用真实数据）
    print("\n📊 运行使用真实数据的测试...")
    test_results = {}
    
    for attack_type, attack_params in attack_scenarios.items():
        try:
            test_data, env = run_attack_test(
                model, feature_scaler, target_scaler, topology_scaler,
                attack_type, attack_params, use_attacked_data=False
            )
            test_results[attack_type] = test_data
        except Exception as e:
            print(f"❌ {attack_type}攻击测试失败: {e}")
            import traceback
            traceback.print_exc()
            test_results[attack_type] = None
    
    # 4. 运行各种攻击测试（使用被攻击数据）
    print("\n📊 运行使用被攻击数据的测试...")
    test_results_with_attack = {}
    
    for attack_type, attack_params in attack_scenarios.items():
        if attack_type == "none":
            # 对于无攻击情况，不需要重复测试
            test_results_with_attack[attack_type] = test_results[attack_type]
            continue
            
        try:
            test_data, env = run_attack_test(
                model, feature_scaler, target_scaler, topology_scaler,
                attack_type, attack_params, use_attacked_data=True
            )
            test_results_with_attack[attack_type] = test_data
        except Exception as e:
            print(f"❌ {attack_type}攻击测试（使用被攻击数据）失败: {e}")
            import traceback
            traceback.print_exc()
            test_results_with_attack[attack_type] = None
    
    # 5. 分析性能
    print("\n📊 分析使用真实数据的测试性能...")
    all_metrics = analyze_attack_performance(test_results)
    
    print("\n📊 分析使用被攻击数据的测试性能...")
    all_metrics_with_attack = analyze_attack_performance(test_results_with_attack)
    
    # 6. 绘制性能对比图
    plot_attack_performance(all_metrics)
    
    # 7. 绘制预测值与真实值对比图
    plot_prediction_vs_ground_truth(test_results)
    plot_prediction_vs_ground_truth(test_results_with_attack)
    
    # 8. 绘制详细对比图（真实值、网络攻击值、预测值）
    plot_detailed_comparison(test_results)
    plot_detailed_comparison(test_results_with_attack)
    
    print(f"\n🎉 增强模型网络攻击测试完成！")


if __name__ == "__main__":
    main()