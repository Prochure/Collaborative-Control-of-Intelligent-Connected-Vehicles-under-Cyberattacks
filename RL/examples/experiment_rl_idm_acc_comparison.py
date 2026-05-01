#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验脚本：对比RL控制、IDM控制和ACC控制
参考 experiment_lead_cav_p.py 的结构

只修改CAV的控制模式，从RL到IDM和到ACC，其他设置保持不变。


!!!!!!!(ACC效果不行待修改)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add project root to path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sim_env.envs.cyber_attack_env import CyberAttackEnv
from sim_env.models.acc import compute_acc_acceleration, ACCParameters
from examples.flexible_rl_test import (
    load_model, 
    select_action,
    generate_random_oscillating_speed_sequence,
    generate_deceleration_acceleration_speed_sequence
)

def run_experiment(control_mode, model_path, speed_sequence, algorithm="ddpg"):
    """
    运行实验，根据控制模式选择不同的控制方法
    
    参数:
        control_mode: 控制模式，'rl', 'idm', 或 'acc'
        model_path: RL模型路径（仅RL模式需要）
        speed_sequence: 前车速度序列
        algorithm: RL算法类型（仅RL模式需要）
    
    返回:
        trajectory_data: 轨迹数据列表
    """
    print(f"运行实验: {control_mode.upper()}控制模式")
    
    # 创建环境（所有控制模式使用相同的环境配置）
    env = CyberAttackEnv(
        num_vehicles=7,
        cav_indices=[1,3,6],
        dt=0.1,
        enable_cyber_attack=True,
        attack_frequency=0.9,
        attack_type='data_tampering',
        attack_targets=["speed", "acceleration"],
        attack_variances={"speed": 5.0, "acceleration": 2.0},
        use_cbf=False,
        force_lead_cav_p_one=False,
        filter_alpha=0.1    
    )
    
    # Reset with specific speed sequence for reproducibility
    max_steps = len(speed_sequence)
    dt = 0.1
    
    reset_options = {
        "lead_speed_sequence": speed_sequence
    }
    state, _ = env.reset(options=reset_options)
    
    # 根据控制模式进行不同的初始化
    if control_mode == 'rl':
        state_dim = state.shape[1]
        action_dim = 1
        
        if not os.path.exists(model_path):
            print(f"错误: 模型路径不存在: {model_path}")
            return []
        
        model, device = load_model(algorithm, model_path, state_dim, action_dim)
    elif control_mode == 'acc':
        # 为每个CAV创建与场景匹配的ACC参数
        # 这里将期望速度设置为环境目标速度(若无则退化为18m/s)，并调低控制增益，避免过冲
        base_acc_params = ACCParameters(
            desired_speed=getattr(env, "v_target", 18.0),
            minimum_gap=5.0,
            desired_time_gap=1.4,
            kp_gap=0.8,
            kd_gap=0.35,
            kp_speed=0.5,
            max_acceleration=2.0,
            max_deceleration=3.0,
        )
        acc_params_list = [base_acc_params.copy() for _ in env.cav_ids]
    
    trajectory_data = []
    
    for step in range(max_steps):
        action = [None] * len(env.cav_ids)
        
        if control_mode == 'rl':
            # RL控制：使用RL模型选择动作
            # 第一辆CAV（索引为0）使用IDM模型（action[0]保持为None）
            # 其他CAV使用强化学习选择动作
            cav_observations = state
            for j in range(1, len(env.cav_ids)):  # 从第二个CAV开始
                cav_obs = cav_observations[j]
                rl_action = select_action(algorithm, model, device, cav_obs)
                action[j] = rl_action
                
        elif control_mode == 'idm':
            # IDM控制：action保持为None，让CAV使用IDM模型
            pass  # action已经是[None] * len(env.cav_ids)
            
        elif control_mode == 'acc':
            # ACC控制：显式计算ACC加速度
            for j in range(len(env.cav_ids)):
                cav_id = env.cav_ids[j]
                # 找到CAV在车辆列表中的索引
                cav_idx = next((i for i, v in enumerate(env.sim.vehicles) if v.vehicle_id == cav_id), None)
                
                if cav_idx is None:
                    continue
                
                # 获取CAV状态
                cav_vehicle = env.sim.vehicles[cav_idx]
                v_ego = cav_vehicle.speed
                
                # 获取前车信息
                leader = env.sim.get_leader(cav_idx)
                if leader is not None:
                    gap = env.sim.gap_to_leader(cav_vehicle, leader)
                    v_lead = leader.speed
                else:
                    gap = np.inf
                    v_lead = 0.0
                
                rel_speed = v_ego - v_lead
                
                # 使用ACC计算加速度
                acc_action = compute_acc_acceleration(
                    v=v_ego,
                    gap=gap,
                    rel_speed=rel_speed,
                    params=acc_params_list[j]
                )
                
                action[j] = acc_action
        
        next_state, reward, terminated, truncated, info = env.step(action)
        
        step_data = {
            "time": env.sim.t,
            "state": env.sim.get_state(),
            "reward": reward
        }
        trajectory_data.append(step_data)
        
        state = next_state
        if terminated or truncated:
            break
            
    return trajectory_data

def plot_comparison(data_rl_list, data_idm_list, data_acc_list, dt, 
                    output_file="rl_idm_acc_comparison.png"):
    """
    绘制2x3网格图：展示加速度
    Row 1: Condition 1 (RL, IDM, ACC)
    Row 2: Condition 2 (RL, IDM, ACC)
    """
    rows_data = [data_rl_list, data_idm_list, data_acc_list]
    row_labels = ['RL Control', 'IDM Control', 'ACC Control']

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

    # 创建2x3子图 (2个工况 x 3种控制方法)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.1,
                        hspace=0.3, wspace=0.2)

    # 定义颜色
    hv_color = '#d62728'
    cav_color = '#1f77b4'

    # 遍历控制方法 (列)
    for col_idx in range(3):
        current_method_data = rows_data[col_idx]  # [cond1, cond2]
        
        # 遍历工况 (行)
        for row_idx in range(2):
            trajectory_data = current_method_data[row_idx]
            ax = axes[row_idx, col_idx]
            
            if not trajectory_data:
                continue

            # 时间步（Step）
            time_series = [data["time"] for data in trajectory_data]
            step_series = np.array(time_series) / dt

            # 按车辆ID组织数据
            vehicle_data = {}
            sample_state = trajectory_data[0]["state"]

            for vehicle_state in sample_state:
                vid = vehicle_state["id"]
                vehicle_data[vid] = {
                    "step": [],
                    "acceleration": [],
                    "is_cav": vehicle_state["type"] == "CAV"
                }

            # 填充轨迹数据
            for data in trajectory_data:
                step = data["time"] / dt
                for vehicle_state in data["state"]:
                    vid = vehicle_state["id"]
                    vehicle_data[vid]["step"].append(step)
                    vehicle_data[vid]["acceleration"].append(vehicle_state["a"])

            # 绘制车辆加速度曲线
            for vid, data in vehicle_data.items():
                color = cav_color if data['is_cav'] else hv_color
                line_style = '--' if data['is_cav'] else '-'

                ax.plot(data["step"], data["acceleration"],
                        color=color, linestyle=line_style, linewidth=1.3)

            # 设置轴标签和范围
            ax.set_ylim(-3, 3)
            ax.set_xlabel("Time Step [0.1 s]")
            # 每一幅图都显示纵坐标标签
            if col_idx == 0:
                ax.set_ylabel("Acceleration (m/s$^2$)")
            
            # 设置列标题（仅第一行）
            if row_idx == 0:
                ax.set_title(row_labels[col_idx], fontsize=14, fontweight='bold')

            # 网格
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

            # 设置图例（仅(0,0)）
            if row_idx == 0 and col_idx == 0:
                legend_elements = [
                    plt.Line2D([0], [0], color=cav_color, linestyle='--', linewidth=1.5),
                    plt.Line2D([0], [0], color=hv_color, linestyle='-', linewidth=1.5)
                ]
                legend_labels = ['CAV', 'HV']
                ax.legend(legend_elements, legend_labels, loc='upper right')

    # 行级标注 (a), (b)
    labels = ['(a) Condition 1', '(b) Condition 2']
    for row_idx, label in enumerate(labels):
        # 在左侧添加行标签
        fig.text(0.02, 0.75 - row_idx * 0.45, label,
                 fontsize=14, fontweight='bold', rotation=90,
                 ha='center', va='center')

    # 保存
    plt.savefig(output_file, dpi=350, bbox_inches='tight')
    print(f"对比图已保存到: {output_file}")

def save_quantitative_analysis(data_rl_list, data_idm_list, data_acc_list, dt, 
                               output_file="rl_idm_acc_analysis.txt"):
    """
    保存量化分析结果到txt文件
    
    参数:
        data_rl_list: RL控制数据列表 [condition1, condition2]
        data_idm_list: IDM控制数据列表 [condition1, condition2]
        data_acc_list: ACC控制数据列表 [condition1, condition2]
        dt: 时间步长
        output_file: 输出文件名
    """
    
    def calculate_metrics(trajectory_data, dt):
        """计算单个工况的所有指标"""
        if not trajectory_data:
            return {}
        
        # 按车辆ID组织数据
        vehicle_data = {}
        for step_data in trajectory_data:
            for v_state in step_data["state"]:
                vid = v_state["id"]
                if vid not in vehicle_data:
                    vehicle_data[vid] = {
                        "position": [],
                        "speed": [],
                        "acceleration": [],
                        "is_cav": v_state["type"] == "CAV"
                    }
                vehicle_data[vid]["position"].append(v_state["x"])
                vehicle_data[vid]["speed"].append(v_state["v"])
                vehicle_data[vid]["acceleration"].append(v_state["a"])
        
        # 计算Jerk（加加速度）
        for vid, data in vehicle_data.items():
            acc = np.array(data["acceleration"])
            if len(acc) > 1:
                jerk = np.diff(acc) / dt
                data["jerk"] = np.concatenate([[0], jerk])
            else:
                data["jerk"] = np.array([0])
        
        # 分离CAV和HV数据
        cav_ids = [vid for vid, data in vehicle_data.items() if data["is_cav"]]
        hv_ids = [vid for vid, data in vehicle_data.items() if not data["is_cav"]]
        
        metrics = {}
        
        # 全体车辆指标
        all_acc = np.concatenate([vehicle_data[vid]["acceleration"] for vid in vehicle_data.keys()])
        all_jerk = np.concatenate([vehicle_data[vid]["jerk"] for vid in vehicle_data.keys()])
        all_speed = np.concatenate([vehicle_data[vid]["speed"] for vid in vehicle_data.keys()])
        
        metrics["all"] = {
            "acc_mean": np.mean(np.abs(all_acc)),
            "acc_std": np.std(all_acc),
            "acc_max": np.max(all_acc),
            "acc_min": np.min(all_acc),
            "jerk_mean": np.mean(np.abs(all_jerk)),
            "jerk_std": np.std(all_jerk),
            "jerk_max": np.max(np.abs(all_jerk)),
            "speed_mean": np.mean(all_speed),
            "speed_std": np.std(all_speed),
            "speed_max": np.max(all_speed),
            "speed_min": np.min(all_speed)
        }
        
        # CAV指标
        if cav_ids:
            cav_acc = np.concatenate([vehicle_data[vid]["acceleration"] for vid in cav_ids])
            cav_jerk = np.concatenate([vehicle_data[vid]["jerk"] for vid in cav_ids])
            cav_speed = np.concatenate([vehicle_data[vid]["speed"] for vid in cav_ids])
            
            metrics["cav"] = {
                "acc_mean": np.mean(np.abs(cav_acc)),
                "acc_std": np.std(cav_acc),
                "acc_max": np.max(cav_acc),
                "acc_min": np.min(cav_acc),
                "jerk_mean": np.mean(np.abs(cav_jerk)),
                "jerk_std": np.std(cav_jerk),
                "jerk_max": np.max(np.abs(cav_jerk)),
                "speed_mean": np.mean(cav_speed),
                "speed_std": np.std(cav_speed),
                "speed_max": np.max(cav_speed),
                "speed_min": np.min(cav_speed)
            }
        
        # HV指标
        if hv_ids:
            hv_acc = np.concatenate([vehicle_data[vid]["acceleration"] for vid in hv_ids])
            hv_jerk = np.concatenate([vehicle_data[vid]["jerk"] for vid in hv_ids])
            hv_speed = np.concatenate([vehicle_data[vid]["speed"] for vid in hv_ids])
            
            metrics["hv"] = {
                "acc_mean": np.mean(np.abs(hv_acc)),
                "acc_std": np.std(hv_acc),
                "acc_max": np.max(hv_acc),
                "acc_min": np.min(hv_acc),
                "jerk_mean": np.mean(np.abs(hv_jerk)),
                "jerk_std": np.std(hv_jerk),
                "jerk_max": np.max(np.abs(hv_jerk)),
                "speed_mean": np.mean(hv_speed),
                "speed_std": np.std(hv_speed),
                "speed_max": np.max(hv_speed),
                "speed_min": np.min(hv_speed)
            }
        
        # 车队稳定性指标
        speed_std_over_time = []
        for step_data in trajectory_data:
            speeds = [v["v"] for v in step_data["state"]]
            speed_std_over_time.append(np.std(speeds))
        metrics["fleet_stability"] = np.mean(speed_std_over_time) / (metrics["all"]["speed_mean"] + 1e-6)
        
        # 弦稳定性（加速度波动放大率）
        sorted_vids = sorted(vehicle_data.keys(), 
                           key=lambda vid: vehicle_data[vid]["position"][0], 
                           reverse=True)
        acc_ratios = []
        for i in range(len(sorted_vids) - 1):
            lead_acc_std = np.std(vehicle_data[sorted_vids[i]]["acceleration"])
            follow_acc_std = np.std(vehicle_data[sorted_vids[i+1]]["acceleration"])
            if lead_acc_std > 1e-6:
                acc_ratios.append(follow_acc_std / lead_acc_std)
        metrics["string_stability"] = np.mean(acc_ratios) if acc_ratios else 0.0
        
        # 速度振幅衰减指标：末车速度振幅 / 头车速度振幅
        speed_amplitudes = []
        for vid in sorted_vids:
            speeds = np.asarray(vehicle_data[vid]["speed"])
            if speeds.size == 0:
                continue
            speed_amplitudes.append(np.max(speeds) - np.min(speeds))
        if speed_amplitudes:
            speed_osc_ratio = speed_amplitudes[-1] / (speed_amplitudes[0] + 1e-6)
        else:
            speed_osc_ratio = 0.0
        metrics["speed_osc_ratio"] = speed_osc_ratio
        
        # 速度梯度指标：相邻车辆平均速度差的归一化值
        speed_diff_values = []
        for i in range(1, len(sorted_vids)):
            lead_speeds = np.asarray(vehicle_data[sorted_vids[i - 1]]["speed"])
            follow_speeds = np.asarray(vehicle_data[sorted_vids[i]]["speed"])
            seq_len = min(len(lead_speeds), len(follow_speeds))
            if seq_len == 0:
                continue
            diff = np.mean(np.abs(lead_speeds[:seq_len] - follow_speeds[:seq_len]))
            speed_diff_values.append(diff)
        if speed_diff_values:
            avg_speed_diff = np.mean(speed_diff_values)
            metrics["speed_diff_index"] = avg_speed_diff / (metrics["all"]["speed_mean"] + 1e-6)
        else:
            metrics["speed_diff_index"] = 0.0
        
        return metrics
    
    # 计算所有工况的指标
    conditions = ["Condition 1 (Oscillating)", "Condition 2 (Decel-Accel)"]
    rl_metrics = [calculate_metrics(data, dt) for data in data_rl_list]
    idm_metrics = [calculate_metrics(data, dt) for data in data_idm_list]
    acc_metrics = [calculate_metrics(data, dt) for data in data_acc_list]
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("控制方法对比分析: RL vs IDM vs ACC\n")
        f.write("="*100 + "\n\n")
        
        for cond_idx, cond_name in enumerate(conditions):
            f.write(f"\n{'='*100}\n")
            f.write(f"{cond_name}\n")
            f.write(f"{'='*100}\n\n")
            
            rl_m = rl_metrics[cond_idx]
            idm_m = idm_metrics[cond_idx]
            acc_m = acc_metrics[cond_idx]
            
            # 全体车辆指标对比
            f.write("全体车辆指标:\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Metric':<30} {'RL Control':>20} {'IDM Control':>20} {'ACC Control':>20}\n")
            f.write("-" * 100 + "\n")
            
            all_metrics_keys = [
                ("Acceleration Mean (m/s²)", "acc_mean"),
                ("Acceleration Std (m/s²)", "acc_std"),
                ("Acceleration Max (m/s²)", "acc_max"),
                ("Acceleration Min (m/s²)", "acc_min"),
                ("Jerk Mean (m/s³)", "jerk_mean"),
                ("Jerk Std (m/s³)", "jerk_std"),
                ("Jerk Max (m/s³)", "jerk_max"),
                ("Speed Mean (m/s)", "speed_mean"),
                ("Speed Std (m/s)", "speed_std"),
                ("Speed Max (m/s)", "speed_max"),
                ("Speed Min (m/s)", "speed_min")
            ]
            
            for name, key in all_metrics_keys:
                rl_val = rl_m["all"][key]
                idm_val = idm_m["all"][key]
                acc_val = acc_m["all"][key]
                f.write(f"{name:<30} {rl_val:>20.6f} {idm_val:>20.6f} {acc_val:>20.6f}\n")
            
            f.write("\n")
            
            # CAV指标对比
            if "cav" in rl_m and "cav" in idm_m and "cav" in acc_m:
                f.write("CAV指标:\n")
                f.write("-" * 100 + "\n")
                f.write(f"{'Metric':<30} {'RL Control':>20} {'IDM Control':>20} {'ACC Control':>20}\n")
                f.write("-" * 100 + "\n")
                
                for name, key in all_metrics_keys:
                    rl_val = rl_m["cav"][key]
                    idm_val = idm_m["cav"][key]
                    acc_val = acc_m["cav"][key]
                    f.write(f"{name:<30} {rl_val:>20.6f} {idm_val:>20.6f} {acc_val:>20.6f}\n")
                
                f.write("\n")
            
            # HV指标对比
            if "hv" in rl_m and "hv" in idm_m and "hv" in acc_m:
                f.write("HV指标:\n")
                f.write("-" * 100 + "\n")
                f.write(f"{'Metric':<30} {'RL Control':>20} {'IDM Control':>20} {'ACC Control':>20}\n")
                f.write("-" * 100 + "\n")
                
                for name, key in all_metrics_keys:
                    rl_val = rl_m["hv"][key]
                    idm_val = idm_m["hv"][key]
                    acc_val = acc_m["hv"][key]
                    f.write(f"{name:<30} {rl_val:>20.6f} {idm_val:>20.6f} {acc_val:>20.6f}\n")
                
                f.write("\n")
            
            # 稳定性指标
            f.write("稳定性指标:\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Metric':<30} {'RL Control':>20} {'IDM Control':>20} {'ACC Control':>20}\n")
            f.write("-" * 100 + "\n")
            
            rl_fleet = rl_m["fleet_stability"]
            idm_fleet = idm_m["fleet_stability"]
            acc_fleet = acc_m["fleet_stability"]
            f.write(f"{'Fleet Stability Index':<30} {rl_fleet:>20.6f} {idm_fleet:>20.6f} {acc_fleet:>20.6f}\n")
            
            rl_string = rl_m["string_stability"]
            idm_string = idm_m["string_stability"]
            acc_string = acc_m["string_stability"]
            f.write(f"{'String Stability Index':<30} {rl_string:>20.6f} {idm_string:>20.6f} {acc_string:>20.6f}\n")
            
            rl_osc = rl_m.get("speed_osc_ratio", 0.0)
            idm_osc = idm_m.get("speed_osc_ratio", 0.0)
            acc_osc = acc_m.get("speed_osc_ratio", 0.0)
            f.write(f"{'Speed Oscillation Ratio':<30} {rl_osc:>20.6f} {idm_osc:>20.6f} {acc_osc:>20.6f}\n")
            
            rl_grad = rl_m.get("speed_diff_index", 0.0)
            idm_grad = idm_m.get("speed_diff_index", 0.0)
            acc_grad = acc_m.get("speed_diff_index", 0.0)
            f.write(f"{'Relative Speed Gradient':<30} {rl_grad:>20.6f} {idm_grad:>20.6f} {acc_grad:>20.6f}\n")
            
            f.write("\n")
        
        # 总结
        f.write(f"\n{'='*100}\n")
        f.write("总结 (两种工况平均)\n")
        f.write(f"{'='*100}\n\n")
        
        avg_rl_acc = np.mean([m["all"]["acc_mean"] for m in rl_metrics])
        avg_idm_acc = np.mean([m["all"]["acc_mean"] for m in idm_metrics])
        avg_acc_acc = np.mean([m["all"]["acc_mean"] for m in acc_metrics])
        
        avg_rl_jerk = np.mean([m["all"]["jerk_mean"] for m in rl_metrics])
        avg_idm_jerk = np.mean([m["all"]["jerk_mean"] for m in idm_metrics])
        avg_acc_jerk = np.mean([m["all"]["jerk_mean"] for m in acc_metrics])
        
        avg_rl_fleet = np.mean([m["fleet_stability"] for m in rl_metrics])
        avg_idm_fleet = np.mean([m["fleet_stability"] for m in idm_metrics])
        avg_acc_fleet = np.mean([m["fleet_stability"] for m in acc_metrics])
        
        avg_rl_osc = np.mean([m.get("speed_osc_ratio", 0.0) for m in rl_metrics])
        avg_idm_osc = np.mean([m.get("speed_osc_ratio", 0.0) for m in idm_metrics])
        avg_acc_osc = np.mean([m.get("speed_osc_ratio", 0.0) for m in acc_metrics])
        
        avg_rl_grad = np.mean([m.get("speed_diff_index", 0.0) for m in rl_metrics])
        avg_idm_grad = np.mean([m.get("speed_diff_index", 0.0) for m in idm_metrics])
        avg_acc_grad = np.mean([m.get("speed_diff_index", 0.0) for m in acc_metrics])
        
        f.write(f"{'指标':<30} {'RL Control':>20} {'IDM Control':>20} {'ACC Control':>20}\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'平均加速度(绝对值)':<30} {avg_rl_acc:>20.6f} {avg_idm_acc:>20.6f} {avg_acc_acc:>20.6f}\n")
        f.write(f"{'平均Jerk(绝对值)':<30} {avg_rl_jerk:>20.6f} {avg_idm_jerk:>20.6f} {avg_acc_jerk:>20.6f}\n")
        f.write(f"{'车队稳定性':<30} {avg_rl_fleet:>20.6f} {avg_idm_fleet:>20.6f} {avg_acc_fleet:>20.6f}\n")
        f.write(f"{'速度振幅比':<30} {avg_rl_osc:>20.6f} {avg_idm_osc:>20.6f} {avg_acc_osc:>20.6f}\n")
        f.write(f"{'相对速度梯度':<30} {avg_rl_grad:>20.6f} {avg_idm_grad:>20.6f} {avg_acc_grad:>20.6f}\n")
        
    print(f"量化分析已保存到: {output_file}")

if __name__ == "__main__":
    # Use a valid model path
    model_path = os.path.join(_PROJECT_ROOT, "models", "ddpg_mixed_cav_agent_best.pth", "ddpg_actor.pth")
    
    max_steps = 600
    dt = 0.1
    
    # Generate speed sequences
    seq1 = generate_random_oscillating_speed_sequence(max_steps, dt)
    seq2 = generate_deceleration_acceleration_speed_sequence(max_steps, dt)
    
    print("="*80)
    print("运行RL控制实验")
    print("="*80)
    print("  - Condition 1...")
    rl_c1 = run_experiment('rl', model_path, seq1)
    print("  - Condition 2...")
    rl_c2 = run_experiment('rl', model_path, seq2)
    
    print("\n" + "="*80)
    print("运行IDM控制实验")
    print("="*80)
    print("  - Condition 1...")
    idm_c1 = run_experiment('idm', model_path, seq1)
    print("  - Condition 2...")
    idm_c2 = run_experiment('idm', model_path, seq2)
    
    print("\n" + "="*80)
    print("运行ACC控制实验")
    print("="*80)
    print("  - Condition 1...")
    acc_c1 = run_experiment('acc', model_path, seq1)
    print("  - Condition 2...")
    acc_c2 = run_experiment('acc', model_path, seq2)
    
    if all([rl_c1, rl_c2, idm_c1, idm_c2, acc_c1, acc_c2]):
        plot_comparison([rl_c1, rl_c2], [idm_c1, idm_c2], [acc_c1, acc_c2], dt)
        save_quantitative_analysis([rl_c1, rl_c2], [idm_c1, idm_c2], [acc_c1, acc_c2], dt)
        
        print("\n" + "="*50)
        print("实验完成!")
        print("="*50)
        print("生成的文件:")
        print("  - rl_idm_acc_comparison.png (对比图)")
        print("  - rl_idm_acc_analysis.txt (量化分析)")
    else:
        print("\n错误: 部分实验未能完成")

