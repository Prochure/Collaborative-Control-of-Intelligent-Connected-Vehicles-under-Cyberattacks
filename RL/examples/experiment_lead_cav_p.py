#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
force_lead_cav_p_one=False vs True.(shi)
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
from examples.flexible_rl_test import load_model, select_action, generate_random_oscillating_speed_sequence, generate_deceleration_acceleration_speed_sequence

def run_experiment(force_lead_cav_p_one, model_path, speed_sequence, algorithm="ddpg"):
    print(f"Running experiment with force_lead_cav_p_one={force_lead_cav_p_one}")
    
    # Create environment with the new flag
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
        filter_alpha=0.1,
        force_lead_cav_p_one=force_lead_cav_p_one  # Pass the flag
    )
    
    # Reset with specific speed sequence for reproducibility
    max_steps = 600
    dt = 0.1
    
    reset_options = {
        "lead_speed_sequence": speed_sequence
    }
    state, _ = env.reset(options=reset_options)
    
    state_dim = state.shape[1]
    action_dim = 1
    
    # Load model
    if not os.path.exists(model_path):
        print(f"Error: Model path not found: {model_path}")
        return []

    model, device = load_model(algorithm, model_path, state_dim, action_dim)
    
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
            "reward": reward
        }
        trajectory_data.append(step_data)
        
        state = next_state
        if terminated or truncated:
            break
            
    return trajectory_data

def plot_comparison(data_ori_list, data_mod_list, dt, output_file="acceleration_comparison.png"):
    """
    绘制2x2网格图：只展示加速度
    Row 1: Original (Condition 1, Condition 2)
    Row 2: Modified (Condition 1, Condition 2)
    """
    rows_data = [data_ori_list, data_mod_list]

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

    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1,
                        hspace=0.3, wspace=0.15)

    # 定义颜色
    hv_color = '#d62728'
    cav_color = '#1f77b4'

    # 遍历行 (Original, Modified)
    for row_idx in range(2):
        current_row_data = rows_data[row_idx] # [cond1, cond2]
        
        # 遍历列 (Condition 1, Condition 2)
        for col_idx in range(2):
            trajectory_data = current_row_data[col_idx]
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
            ax.set_ylabel("Acceleration (m/s$^2$)")
            
            # 去掉标题
            # if row_idx == 0:
            #     ax.set_title(f"Condition {col_idx + 1}")

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
    labels = ['(a)', '(b)']
    for row_idx, label in enumerate(labels):
        row_axes = axes[row_idx]
        # 获取该行第一个子图的位置
        bbox = row_axes[0].get_position()
        # 在横坐标标签下方显示 (a)/(b)
        # bbox.y0 是子图轴的底部，减去一个较大的偏移量以避开x轴标签
        text_y = bbox.y0 - 0.05
        fig.text(0.5, text_y, label,
                 fontsize=16, fontweight='bold', ha='center', va='top')

    # 保存
    plt.savefig(output_file, dpi=350, bbox_inches='tight')
    print(f"Comparison plot saved to {output_file}")

def save_quantitative_analysis(data_ori_list, data_mod_list, dt, output_file="experiment_analysis.txt"):
    """
    保存量化分析结果到txt文件
    
    参数:
        data_ori_list: Original工况数据列表 [condition1, condition2]
        data_mod_list: Modified工况数据列表 [condition1, condition2]
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
        
        return metrics
    
    # 计算所有工况的指标
    conditions = ["Condition 1 (Oscillating)", "Condition 2 (Decel-Accel)"]
    ori_metrics = [calculate_metrics(data, dt) for data in data_ori_list]
    mod_metrics = [calculate_metrics(data, dt) for data in data_mod_list]
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Quantitative Analysis: Original vs Modified (lead_cav_p=1)\n")
        f.write("="*80 + "\n\n")
        
        for cond_idx, cond_name in enumerate(conditions):
            f.write(f"\n{'='*80}\n")
            f.write(f"{cond_name}\n")
            f.write(f"{'='*80}\n\n")
            
            ori_m = ori_metrics[cond_idx]
            mod_m = mod_metrics[cond_idx]
            
            # 全体车辆指标对比
            f.write("All Vehicles Metrics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Metric':<30} {'Original':>15} {'Modified':>15} {'Difference':>15}\n")
            f.write("-" * 80 + "\n")
            
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
                ori_val = ori_m["all"][key]
                mod_val = mod_m["all"][key]
                diff = mod_val - ori_val
                f.write(f"{name:<30} {ori_val:>15.6f} {mod_val:>15.6f} {diff:>15.6f}\n")
            
            f.write("\n")
            
            # CAV指标对比
            if "cav" in ori_m and "cav" in mod_m:
                f.write("CAV Metrics:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Metric':<30} {'Original':>15} {'Modified':>15} {'Difference':>15}\n")
                f.write("-" * 80 + "\n")
                
                for name, key in all_metrics_keys:
                    ori_val = ori_m["cav"][key]
                    mod_val = mod_m["cav"][key]
                    diff = mod_val - ori_val
                    f.write(f"{name:<30} {ori_val:>15.6f} {mod_val:>15.6f} {diff:>15.6f}\n")
                
                f.write("\n")
            
            # HV指标对比
            if "hv" in ori_m and "hv" in mod_m:
                f.write("HV Metrics:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Metric':<30} {'Original':>15} {'Modified':>15} {'Difference':>15}\n")
                f.write("-" * 80 + "\n")
                
                for name, key in all_metrics_keys:
                    ori_val = ori_m["hv"][key]
                    mod_val = mod_m["hv"][key]
                    diff = mod_val - ori_val
                    f.write(f"{name:<30} {ori_val:>15.6f} {mod_val:>15.6f} {diff:>15.6f}\n")
                
                f.write("\n")
            
            # 稳定性指标
            f.write("Stability Metrics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Metric':<30} {'Original':>15} {'Modified':>15} {'Difference':>15}\n")
            f.write("-" * 80 + "\n")
            
            ori_fleet = ori_m["fleet_stability"]
            mod_fleet = mod_m["fleet_stability"]
            f.write(f"{'Fleet Stability Index':<30} {ori_fleet:>15.6f} {mod_fleet:>15.6f} {(mod_fleet-ori_fleet):>15.6f}\n")
            
            ori_string = ori_m["string_stability"]
            mod_string = mod_m["string_stability"]
            f.write(f"{'String Stability Index':<30} {ori_string:>15.6f} {mod_string:>15.6f} {(mod_string-ori_string):>15.6f}\n")
            
            f.write("\n")
        
        # 总结
        f.write(f"\n{'='*80}\n")
        f.write("Summary\n")
        f.write(f"{'='*80}\n\n")
        
        f.write("Overall Comparison (Average across both conditions):\n\n")
        
        avg_ori_acc = np.mean([m["all"]["acc_mean"] for m in ori_metrics])
        avg_mod_acc = np.mean([m["all"]["acc_mean"] for m in mod_metrics])
        avg_ori_jerk = np.mean([m["all"]["jerk_mean"] for m in ori_metrics])
        avg_mod_jerk = np.mean([m["all"]["jerk_mean"] for m in mod_metrics])
        avg_ori_fleet = np.mean([m["fleet_stability"] for m in ori_metrics])
        avg_mod_fleet = np.mean([m["fleet_stability"] for m in mod_metrics])
        
        f.write(f"Average Acceleration (abs): Original={avg_ori_acc:.6f}, Modified={avg_mod_acc:.6f}, Diff={avg_mod_acc-avg_ori_acc:.6f}\n")
        f.write(f"Average Jerk (abs): Original={avg_ori_jerk:.6f}, Modified={avg_mod_jerk:.6f}, Diff={avg_mod_jerk-avg_ori_jerk:.6f}\n")
        f.write(f"Fleet Stability: Original={avg_ori_fleet:.6f}, Modified={avg_mod_fleet:.6f}, Diff={avg_mod_fleet-avg_ori_fleet:.6f}\n")
        
    print(f"Quantitative analysis saved to {output_file}")

if __name__ == "__main__":
    # Use a valid model path
    model_path = os.path.join(_PROJECT_ROOT, "models", "ddpg_mixed_cav_agent_best.pth", "ddpg_actor.pth")
    
    max_steps = 600
    dt = 0.1
    
    # Generate speed sequences
    seq1 = generate_random_oscillating_speed_sequence(max_steps, dt)
    seq2 = generate_deceleration_acceleration_speed_sequence(max_steps, dt)
    
    print("Running Case 1: Original (force_lead_cav_p_one=False)")
    print("  - Condition 1...")
    ori_c1 = run_experiment(False, model_path, seq1)
    print("  - Condition 2...")
    ori_c2 = run_experiment(False, model_path, seq2)
    
    print("Running Case 2: Modified (force_lead_cav_p_one=True)")
    print("  - Condition 1...")
    mod_c1 = run_experiment(True, model_path, seq1)
    print("  - Condition 2...")
    mod_c2 = run_experiment(True, model_path, seq2)
    
    if all([ori_c1, ori_c2, mod_c1, mod_c2]):
        plot_comparison([ori_c1, ori_c2], [mod_c1, mod_c2], dt)
        save_quantitative_analysis([ori_c1, ori_c2], [mod_c1, mod_c2], dt)
        
        print("\n" + "="*50)
        print("Experiment completed successfully!")
        print("="*50)
