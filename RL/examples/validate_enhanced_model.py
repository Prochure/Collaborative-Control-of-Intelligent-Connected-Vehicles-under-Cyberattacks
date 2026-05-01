#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强模型验证脚本
测试拓扑感知LSTM模型在不同车辆数量和拓扑结构下的性能

功能：
1. 加载增强的拓扑感知模型
2. 在多种拓扑结构下进行验证
3. 对比增强模型与原始模型的泛化能力
4. 生成详细的性能分析报告
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
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目路径
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sim_env.envs.single_lane_env import SingleLaneFollowingEnv
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


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
    """加载250维增强模型"""
    print("📥 加载250维增强模型...")
    
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
    
    # 创建250维模型
    print("   创建250维模型结构")
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
    
    print("   ✅ 250维模型加载成功")
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
    profiles = {
        "aggressive": {
            "name": "激进驾驶",
            "idm_params": {
                "desired_speed": 20.5, "minimum_spacing": 1.4, "desired_time_headway": 1.3,
                "max_acceleration": 2.0, "comfortable_deceleration": 2.5, "delta": 3.0
            }
        },
        "normal": {
            "name": "正常驾驶",
            "idm_params": {
                "desired_speed": 18.0, "minimum_spacing": 2.0, "desired_time_headway": 1.6,
                "max_acceleration": 1.2, "comfortable_deceleration": 1.8, "delta": 4.0
            }
        },
        "conservative": {
            "name": "保守驾驶",
            "idm_params": {
                "desired_speed": 16.8, "minimum_spacing": 2.4, "desired_time_headway": 1.9,
                "max_acceleration": 1.0, "comfortable_deceleration": 1.5, "delta": 4.5
            }
        }
    }
    return profiles


def setup_test_vehicles(num_vehicles, cav_position, profiles, base_seed=1000):
    """设置测试车辆配置"""
    np.random.seed(base_seed)
    
    manual_vehicles = []
    base_gap = 20.0
    
    vehicle_ids = [f"hv{i}" if i != cav_position else "cav1" for i in range(num_vehicles)]
    profile_keys = list(profiles.keys())
    
    for i in range(num_vehicles):
        vehicle_id = vehicle_ids[i]
        is_cav = (i == cav_position)
        x_position = (num_vehicles - 1 - i) * (base_gap + 5.0) + 100.0
        initial_speed = np.random.uniform(10.0, 14.0)
        
        vehicle_config = {
            "vehicle_id": vehicle_id,
            "is_cav": is_cav,
            "x_front": x_position,
            "speed": initial_speed,
            "length": 5.0
        }
        
        if not is_cav:
            profile_key = np.random.choice(profile_keys)
            profile = profiles[profile_key]
            vehicle_config["idm_params"] = profile["idm_params"]
            vehicle_config["driver_profile"] = profile["name"]
        
        manual_vehicles.append(vehicle_config)
    
    return manual_vehicles


class SimpleCAVController:
    """简单CAV控制器"""
    
    def __init__(self):
        self.desired_time_headway = 1.7
        self.min_gap = 2.0
        self.max_decel = -2.2
        self.max_accel = 1.5
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
            action = -1.6 * (1 - gap_error / desired_gap) - 0.1 * relative_speed
        else:
            speed_error = self.desired_speed - cav_vehicle.speed
            action = 0.3 * speed_error - 0.06 * relative_speed
        
        return np.clip(action, self.max_decel, self.max_accel)


def smooth_time_series(data, method='savgol', **kwargs):
    """
    对时间序列数据进行平滑处理，减少IDM模型中随机误差的影响
    
    参数:
    - data: 一维时间序列数据（numpy数组）
    - method: 平滑方法，可选值:
        - 'savgol': Savitzky-Golay滤波器（推荐用于速度、加速度）
        - 'gaussian': 高斯滤波器（推荐用于位置）
        - 'moving_avg': 移动平均（简单但有效）
        - 'exponential': 指数加权移动平均（对最新数据权重更大）
    - kwargs: 各方法的特定参数
    
    返回:
    - smoothed_data: 平滑后的时间序列数据
    """
    data = np.array(data)
    
    if len(data) < 5:  # 数据点太少，不进行平滑
        return data
    
    if method == 'savgol':
        # Savitzky-Golay滤波器：保持信号形状的同时减少噪声
        window_length = kwargs.get('window_length', min(11, len(data) if len(data) % 2 == 1 else len(data) - 1))
        if window_length < 3:
            window_length = 3
        if window_length % 2 == 0:
            window_length += 1
        if window_length > len(data):
            window_length = len(data) if len(data) % 2 == 1 else len(data) - 1
        
        polyorder = kwargs.get('polyorder', min(3, window_length - 1))
        return savgol_filter(data, window_length, polyorder)
    
    elif method == 'gaussian':
        # 高斯滤波器：温和的平滑效果
        sigma = kwargs.get('sigma', 1.0)
        return gaussian_filter1d(data, sigma=sigma)
    
    elif method == 'moving_avg':
        # 移动平均：简单有效的平滑方法
        window_size = kwargs.get('window_size', min(5, len(data)))
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')
    
    elif method == 'exponential':
        # 指数加权移动平均：对最新数据给予更高权重
        alpha = kwargs.get('alpha', 0.3)
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    else:
        raise ValueError(f"不支持的平滑方法: {method}")


def apply_comprehensive_smoothing(speeds, accels, positions, smooth_config=None):
    """
    对速度、加速度、位置序列应用综合平滑处理
    
    参数:
    - speeds: 速度序列
    - accels: 加速度序列  
    - positions: 位置序列
    - smooth_config: 平滑配置字典，包含各类型数据的平滑参数
    
    返回:
    - 平滑后的 (speeds, accels, positions)
    """
    if smooth_config is None:
        smooth_config = {
            'speed': {'method': 'savgol', 'window_length': 9, 'polyorder': 3},
            'accel': {'method': 'savgol', 'window_length': 7, 'polyorder': 2},
            'position': {'method': 'gaussian', 'sigma': 1.2}
        }
    
    # 平滑速度序列
    speed_config = smooth_config.get('speed', {})
    smoothed_speeds = smooth_time_series(speeds, **speed_config)
    
    # 平滑加速度序列
    accel_config = smooth_config.get('accel', {})
    smoothed_accels = smooth_time_series(accels, **accel_config)
    
    # 平滑位置序列
    position_config = smooth_config.get('position', {})
    smoothed_positions = smooth_time_series(positions, **position_config)
    
    return smoothed_speeds, smoothed_accels, smoothed_positions


def run_enhanced_validation_experiment(model, feature_scaler, target_scaler, topology_scaler,
                                     num_vehicles, cav_position, lead_sequence, experiment_id=0):
    """运行增强模型验证实验"""
    print(f"\n🧪 验证实验 {experiment_id + 1}: {num_vehicles}车拓扑，CAV在第{cav_position+1}位")
    
    profiles = create_test_vehicle_configs()
    manual_vehicles = setup_test_vehicles(num_vehicles, cav_position, profiles, base_seed=1000 + experiment_id)
    
    env = SingleLaneFollowingEnv(
        seed=1000 + experiment_id,
        num_vehicles=num_vehicles,
        cav_indices=[cav_position],
        dt=0.1,
        v_target=18.0,
        collision_penalty=200.0
    )
    
    reset_options = {
        "manual_vehicles": manual_vehicles,
        "lead_speed_sequence": lead_sequence
    }
    obs, info = env.reset(options=reset_options)
    
    cav_controller = SimpleCAVController()
    
    experiment_data = {
        "experiment_id": experiment_id,
        "num_vehicles": num_vehicles,
        "cav_position": cav_position,
        "topology_id": f"{num_vehicles}car_cav{cav_position+1}",
        "time_series": {},
        "predictions": []
    }
    
    # 初始化车辆时间序列记录
    vehicle_names = [v["vehicle_id"] for v in manual_vehicles]
    for i, vehicle in enumerate(env.sim.vehicles):
        experiment_data["time_series"][vehicle_names[i]] = {
            "time": [], "position": [], "speed": [], "acceleration": []
        }
    
    # 运行仿真
    max_steps = min(len(lead_sequence), 800)
    window_size = 50
    
    for step in range(max_steps):
        # CAV控制
        cav_vehicle = env.sim.vehicles[cav_position]
        front_vehicle = env.sim.vehicles[cav_position - 1] if cav_position > 0 else None
        
        if front_vehicle:
            cav_action = cav_controller.get_action(cav_vehicle, front_vehicle)
        else:
            speed_error = 18.0 - cav_vehicle.speed
            cav_action = 0.4 * speed_error
        
        cav_action = np.clip(cav_action, -2.0, 2.0)
        
        # 执行仿真步骤
        obs, reward, terminated, truncated, info = env.step({})
        
        # 记录车辆数据
        current_time = step * 0.1
        for i, vehicle in enumerate(env.sim.vehicles):
            vid = vehicle_names[i]
            experiment_data["time_series"][vid]["time"].append(current_time)
            experiment_data["time_series"][vid]["position"].append(vehicle.x_front)
            experiment_data["time_series"][vid]["speed"].append(vehicle.speed)
            experiment_data["time_series"][vid]["acceleration"].append(vehicle.acceleration)
        
        # 每1步进行一次预测（提高预测密度）
        if step >= window_size:
            try:
                start_idx = step - window_size
                end_idx = step
                
                # 使用CAV相对于最后一辆车的特征
                last_vehicle_name = vehicle_names[-1]
                cav_name = vehicle_names[cav_position]
                
                # 获取参考车辆和CAV的原始时间序列
                ref_speeds = experiment_data["time_series"][last_vehicle_name]["speed"][start_idx:end_idx]
                ref_accels = experiment_data["time_series"][last_vehicle_name]["acceleration"][start_idx:end_idx]
                ref_positions = experiment_data["time_series"][last_vehicle_name]["position"][start_idx:end_idx]
                
                cav_speeds = experiment_data["time_series"][cav_name]["speed"][start_idx:end_idx]
                cav_accels = experiment_data["time_series"][cav_name]["acceleration"][start_idx:end_idx]
                cav_positions = experiment_data["time_series"][cav_name]["position"][start_idx:end_idx]
                
                # # 应用平滑处理以减少IDM随机误差的影响
                # print(f"        🔧 对第{step}步的时间窗口数据进行平滑处理") if step % 100 == 0 else None
                
                # # 为参考车辆应用平滑
                # ref_speeds_smooth, ref_accels_smooth, ref_positions_smooth = apply_comprehensive_smoothing(
                #     ref_speeds, ref_accels, ref_positions
                # )
                
                # # 为CAV应用平滑
                # cav_speeds_smooth, cav_accels_smooth, cav_positions_smooth = apply_comprehensive_smoothing(
                #     cav_speeds, cav_accels, cav_positions
                # )
                
                # 计算平滑后的相对差值
                cav_speed_diff = np.array(cav_speeds) - np.array(ref_speeds)
                cav_accel_diff = np.array(cav_accels) - np.array(ref_accels)
                cav_position_diff = np.array(cav_positions) - np.array(ref_positions)
                
                # 构建250维输入特征（使用平滑后的数据）
                input_features = []
                input_features.extend(cav_speed_diff.tolist())     # 50维：CAV相对速度差（平滑后）
                input_features.extend(cav_accel_diff.tolist())    # 50维：CAV相对加速度差（平滑后）
                input_features.extend(cav_position_diff.tolist()) # 50维：CAV相对位置差（平滑后）
                input_features.extend(cav_speeds)          # 50维：CAV绝对速度（平滑后）
                input_features.extend(cav_accels)          # 50维：CAV绝对加速度（平滑后）
                
                # 标准化输入特征
                input_features = np.array(input_features).reshape(1, -1)
                input_features_scaled = feature_scaler.transform(input_features)
                
                # 准备拓扑信息
                topology_info = np.array([[num_vehicles, cav_position]]).reshape(1, -1)
                if topology_scaler is not None:
                    topology_info_scaled = topology_scaler.transform(topology_info)
                else:
                    topology_info_scaled = topology_info
                
                # 模型预测
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(input_features_scaled)
                    topology_tensor = torch.FloatTensor(topology_info_scaled)
                    prediction_scaled = model(input_tensor, topology_tensor).numpy()
                    prediction = target_scaler.inverse_transform(prediction_scaled)[0]
                
                # 计算实际值（除了第一辆车和CAV的其他车辆平均值）
                target_vehicles = [name for i, name in enumerate(vehicle_names) 
                                 if i != 0 and i != cav_position]
                
                if len(target_vehicles) > 0:
                    target_speeds = []
                    target_accels = []
                    
                    for vehicle_name in target_vehicles:
                        target_speeds.append(experiment_data["time_series"][vehicle_name]["speed"][-1])
                        target_accels.append(experiment_data["time_series"][vehicle_name]["acceleration"][-1])
                    
                    actual_avg_speed = np.mean(target_speeds)
                    actual_avg_accel = np.mean(target_accels)
                    
                    # 记录预测结果
                    prediction_data = {
                        "step": step,
                        "time": current_time,
                        "predicted_speed": prediction[0],
                        "predicted_accel": prediction[1],
                        "actual_speed": actual_avg_speed,
                        "actual_accel": actual_avg_accel,
                        "speed_error": abs(prediction[0] - actual_avg_speed),
                        "accel_error": abs(prediction[1] - actual_avg_accel),
                        "num_target_vehicles": len(target_vehicles)
                    }
                    
                    experiment_data["predictions"].append(prediction_data)
                
            except Exception as e:
                continue
        
        if terminated:
            break
    
    print(f"    ✅ 验证实验完成，生成 {len(experiment_data['predictions'])} 个预测")
    return experiment_data


def analyze_enhanced_model_performance(experiments_data):
    """分析增强模型性能"""
    print(f"\n📊 增强模型性能分析")
    print("=" * 60)
    
    # 按拓扑结构分组
    topology_results = {}
    
    for exp in experiments_data:
        topology_id = exp["topology_id"]
        predictions = exp["predictions"]
        
        if len(predictions) == 0:
            continue
        
        if topology_id not in topology_results:
            topology_results[topology_id] = {
                "predictions": [],
                "num_experiments": 0
            }
        
        topology_results[topology_id]["predictions"].extend(predictions)
        topology_results[topology_id]["num_experiments"] += 1
    
    # 分析每种拓扑的性能
    overall_results = {}
    
    for topology_id, data in topology_results.items():
        predictions = data["predictions"]
        
        if len(predictions) == 0:
            continue
        
        pred_speeds = [p["predicted_speed"] for p in predictions]
        actual_speeds = [p["actual_speed"] for p in predictions]
        pred_accels = [p["predicted_accel"] for p in predictions]
        actual_accels = [p["actual_accel"] for p in predictions]
        speed_errors = [p["speed_error"] for p in predictions]
        accel_errors = [p["accel_error"] for p in predictions]
        
        # 计算评估指标
        speed_mae = mean_absolute_error(actual_speeds, pred_speeds)
        speed_rmse = np.sqrt(mean_squared_error(actual_speeds, pred_speeds))
        speed_r2 = r2_score(actual_speeds, pred_speeds)
        
        accel_mae = mean_absolute_error(actual_accels, pred_accels)
        accel_rmse = np.sqrt(mean_squared_error(actual_accels, pred_accels))
        accel_r2 = r2_score(actual_accels, pred_accels)
        
        overall_results[topology_id] = {
            "num_predictions": len(predictions),
            "num_experiments": data["num_experiments"],
            "speed_mae": speed_mae,
            "speed_rmse": speed_rmse,
            "speed_r2": speed_r2,
            "accel_mae": accel_mae,
            "accel_rmse": accel_rmse,
            "accel_r2": accel_r2,
            "speed_error_mean": np.mean(speed_errors),
            "speed_error_std": np.std(speed_errors),
            "accel_error_mean": np.mean(accel_errors),
            "accel_error_std": np.std(accel_errors)
        }
        
        print(f"\n🚗 {topology_id}:")
        print(f"   实验数量: {data['num_experiments']}, 预测数量: {len(predictions)}")
        print(f"   速度预测 - MAE: {speed_mae:.4f}, RMSE: {speed_rmse:.4f}, R²: {speed_r2:.4f}")
        print(f"   加速度预测 - MAE: {accel_mae:.5f}, RMSE: {accel_rmse:.5f}, R²: {accel_r2:.4f}")
    
    # 计算总体性能
    if overall_results:
        all_speed_r2 = [results["speed_r2"] for results in overall_results.values()]
        all_accel_r2 = [results["accel_r2"] for results in overall_results.values()]
        all_speed_mae = [results["speed_mae"] for results in overall_results.values()]
        all_accel_mae = [results["accel_mae"] for results in overall_results.values()]
        
        print(f"\n📈 总体性能:")
        print(f"   平均速度R²: {np.mean(all_speed_r2):.4f} ± {np.std(all_speed_r2):.4f}")
        print(f"   平均加速度R²: {np.mean(all_accel_r2):.4f} ± {np.std(all_accel_r2):.4f}")
        print(f"   平均速度MAE: {np.mean(all_speed_mae):.4f} ± {np.std(all_speed_mae):.4f} m/s")
        print(f"   平均加速度MAE: {np.mean(all_accel_mae):.5f} ± {np.std(all_accel_mae):.5f} m/s²")
    
    return overall_results


def main():
    """主验证函数"""
    print("🧪 增强模型多拓扑验证实验")
    print("=" * 60)
    
    # 加载增强模型
    model_info = load_enhanced_model()
    if model_info[0] is None:
        print("❌ 无法加载增强模型")
        return
    
    model, feature_scaler, target_scaler, topology_scaler = model_info
    
    # 加载测试数据
    speed_sequences = load_lead_speed_data()
    if speed_sequences is None:
        return
    
    # 选择测试序列
    np.random.seed(2000)
    test_indices = np.random.choice(speed_sequences.shape[0], size=12, replace=False)
    
    print(f"🧪 开始多拓扑验证，使用 {len(test_indices)} 个测试序列")
    
    # 测试不同的拓扑配置
    test_topologies = [
        (4, 1), (4, 2),  # 4车拓扑
        (5, 1), (5, 2),  # 5车拓扑  
        (6, 1), (6, 2),  # 6车拓扑
        (7, 1), (7, 2),  # 7车拓扑
    ]
    
    all_experiments = []
    exp_counter = 0
    
    for i, (num_vehicles, cav_position) in enumerate(test_topologies):
        if i < len(test_indices):
            lead_sequence = speed_sequences[test_indices[i]].tolist()
            
            try:
                exp_data = run_enhanced_validation_experiment(
                    model, feature_scaler, target_scaler, topology_scaler,
                    num_vehicles, cav_position, lead_sequence, exp_counter
                )
                all_experiments.append(exp_data)
            except Exception as e:
                print(f"    ❌ 实验失败: {e}")
            
            exp_counter += 1
    
    # 分析结果
    results = analyze_enhanced_model_performance(all_experiments)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"results/enhanced_model_validation_{timestamp}.pkl", 'wb') as f:
        pickle.dump(all_experiments, f)
    
    print(f"\n🎉 增强模型验证完成！")
    print(f"   结果已保存: enhanced_model_validation_{timestamp}.pkl")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main()