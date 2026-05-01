#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feedback Control Test using CyberAttackEnvErrorObs
Comparing two traffic conditions (Oscillating vs Decel-Accel)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sim_env.envs.cyber_attack_env_error_obs import CyberAttackEnvErrorObs

def feedback_controller(obs, k_gap=0.5, k_v=1.0, k_a=0.5):
    """
    Simple linear feedback controller:
    u = k_gap * e_gap + k_v * e_v + k_a * e_a
    
    Args:
        obs: Observation array (N_cav, 3) containing [gap_error, speed_error, accel_error]
        k_gap: Gain for gap error
        k_v: Gain for speed error
        k_a: Gain for acceleration error
        
    Returns:
        actions: Array of accelerations (N_cav,)
    """
    # Calculate control input
    # obs[:, 0] is gap error (actual - desired)
    # obs[:, 1] is speed error (lead - own)
    # obs[:, 2] is accel error (lead - own)
    
    # We want to accelerate if gap is too large (error > 0)
    # We want to accelerate if lead is faster (error > 0)
    # We want to accelerate if lead is accelerating (error > 0)
    
    actions = k_gap * obs[:, 0] + k_v * obs[:, 1] + k_a * obs[:, 2]
    
    # Clip actions to reasonable limits
    actions = np.clip(actions, -3.0, 3.0)
    
    return actions

def generate_random_oscillating_speed_sequence(total_steps, dt):
    """
    Generate random oscillating speed sequence with regularity and smooth acceleration changes
    """
    t = np.arange(total_steps) * dt
    
    # Base speed (15 m/s) + Multi-frequency oscillation
    speed_sequence = 15.0 + (
        2.0 * np.sin(0.3 * t) +      # Low frequency (period ~21s)
        1.2 * np.sin(0.8 * t) +      # Medium frequency (period ~8s)
        0.6 * np.sin(1.5 * t) +      # Higher frequency (period ~4s)
        0.3 * np.sin(3.0 * t)        # High frequency (period ~2s)
    )
    
    # Add smooth random component
    np.random.seed(42)
    random_component = np.zeros(total_steps)
    
    segment_length = 50
    num_segments = total_steps // segment_length + 1
    
    random_values = np.random.normal(0, 0.3, num_segments)
    
    for i in range(num_segments - 1):
        start_idx = i * segment_length
        end_idx = min((i + 1) * segment_length, total_steps)
        
        # Linear interpolation
        interpolation = np.linspace(random_values[i], random_values[i + 1], end_idx - start_idx)
        random_component[start_idx:end_idx] = interpolation
    
    speed_sequence += random_component
    
    # Clip speed range (8-22 m/s)
    speed_sequence = np.clip(speed_sequence, 8.0, 22.0)
    
    return speed_sequence

def generate_deceleration_acceleration_speed_sequence(total_steps, dt):
    """
    Generate speed sequence: Decelerate -> Constant -> Accelerate
    """
    speed_sequence = np.full(total_steps, 18.0)  # Default 18 m/s
    
    # Phase 1: Deceleration (First 60 steps)
    deceleration_steps = 60
    # Decelerate from 18 m/s to 5 m/s
    speed_sequence[:deceleration_steps] = np.linspace(18.0, 5.0, deceleration_steps)
    
    # Phase 2: Constant (Next 200 steps)
    constant_speed_steps = 200
    start_constant = deceleration_steps
    speed_sequence[start_constant:start_constant+constant_speed_steps] = 5.0
    
    # Phase 3: Acceleration (Next 60 steps)
    acceleration_steps = 60
    start_acceleration = start_constant + constant_speed_steps
    # Accelerate from 5 m/s to 18 m/s
    speed_sequence[start_acceleration:start_acceleration+acceleration_steps] = np.linspace(5.0, 18.0, acceleration_steps)
    
    # Remaining time: 18 m/s
    speed_sequence[start_acceleration+acceleration_steps:] = 18.0
    
    return speed_sequence

def plot_two_condition_trajectories(all_trajectory_data, dt, algorithm="Feedback Control"):
    """
    Plot 2x3 grid: Position/Speed/Acceleration for two conditions
    """

    def _apply_publication_style():
        """Match publication quality style"""
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
        print("Not enough trajectory data to plot")
        return

    # Create 2x2 subplots (Speed, Acceleration)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08,
                        hspace=0.30, wspace=0.25)

    # Define colors
    hv_color = '#d62728'
    cav_color = '#1f77b4'

    # Loop over two conditions
    for condition_idx in range(2):

        trajectory_data = all_trajectory_data[condition_idx]

        # Time steps
        time_series = [data["time"] for data in trajectory_data]
        step_series = np.array(time_series) / dt   # Seconds -> steps

        # Organize data by Vehicle ID
        vehicle_data = {}
        sample_state = trajectory_data[0]["state"]

        for vehicle_state in sample_state:
            vid = vehicle_state["id"]
            vehicle_data[vid] = {
                "step": [],
                "speed": [],
                "acceleration": [],
                "is_cav": vehicle_state["type"] == "CAV"
            }

        # Fill trajectory data
        for data in trajectory_data:
            step = data["time"] / dt
            for vehicle_state in data["state"]:
                vid = vehicle_state["id"]
                vehicle_data[vid]["step"].append(step)
                vehicle_data[vid]["speed"].append(vehicle_state["v"])
                vehicle_data[vid]["acceleration"].append(vehicle_state["a"])

        # Get corresponding axes
        ax_vel = axes[condition_idx, 0]
        ax_acc = axes[condition_idx, 1]

        # Plot curves
        for vid, data in vehicle_data.items():
            color = cav_color if data['is_cav'] else hv_color
            line_style = '--' if data['is_cav'] else '-'

            # Speed
            ax_vel.plot(data["step"], data["speed"],
                        color=color, linestyle=line_style, linewidth=1.3)

            # Acceleration
            ax_acc.plot(data["step"], data["acceleration"],
                        color=color, linestyle=line_style, linewidth=1.3)

        # Set Speed Y-limits
        all_speeds = []
        for data in vehicle_data.values():
            all_speeds += data["speed"]
        if all_speeds:
            ax_vel.set_ylim(min(all_speeds)-2, max(all_speeds)+2)

        # Set Acceleration Y-limits
        ax_acc.set_ylim(-3, 3)

        # Axis Labels
        ax_vel.set_ylabel("Speed (m/s)")
        ax_acc.set_ylabel("Acceleration (m/s$^2$)")

        ax_acc.set_xlabel("Time Step [0.1 s]")
        ax_vel.set_xlabel("Time Step [0.1 s]")

        # Grid
        for ax in (ax_vel, ax_acc):
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

        # Legend (only for first row, first column)
        if condition_idx == 0:
            legend_elements = [
                plt.Line2D([0], [0], color=cav_color, linestyle='--', linewidth=1.5),
                plt.Line2D([0], [0], color=hv_color, linestyle='-', linewidth=1.5)
            ]
            legend_labels = ['CAV', 'HV']

            ax_vel.legend(legend_elements, legend_labels, loc='upper right')

    # Row labels (a), (b)
    for row_idx, label in enumerate(['(a)', '(b)']):
        row_axes = axes[row_idx]
        row_bottom = min(ax.get_position().y0 for ax in row_axes)
        fig.text(0.5, row_bottom - 0.04, label,
                 fontsize=16, fontweight='bold', ha='center', va='top')

    # Save
    output_path = os.path.join(_THIS_DIR, f"{algorithm.replace(' ', '_')}_two_conditions_trajectories_2x2.png")
    plt.savefig(output_path, dpi=350, bbox_inches='tight')
    print(f"{algorithm} two conditions trajectory plot (2x2) saved to: {output_path}")

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
        print("No trajectory data to save")
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
    summary_csv_filename = f"{algorithm}_{scenario_name}_all_vehicles_trajectory_shi.csv"
    summary_csv_path = os.path.join(output_dir, summary_csv_filename)
    
    # 保存汇总CSV
    df_all.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    print(f"Saved all vehicles summary trajectory data to: {summary_csv_filename}")
    
    return summary_csv_path

def calculate_and_save_dimensionless_metrics(all_trajectory_data, algorithm="Feedback_Control", dt=0.1, scenario_names=None):
    """
    Calculate and save dimensionless performance metrics to a txt file.
    Adapted from flexible_rl_test.py
    
    Args:
        all_trajectory_data: List of trajectory data for all conditions
        algorithm: Algorithm name
        dt: Time step
        scenario_names: Optional list of scenario names for each condition
    """
    if not all_trajectory_data:
        print("No trajectory data to analyze")
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
        f.write(f"Evaluation Metrics Report - {algorithm}\n")
        f.write(f"{'='*80}\n\n")
        
        # Calculate metrics for each condition
        for condition_idx, trajectory_data in enumerate(all_trajectory_data):
            # f.write(f"\n{'='*80}\n")
            # if scenario_names and condition_idx < len(scenario_names):
            #     f.write(f"{scenario_names[condition_idx]}\n")
            # else:
            #     f.write(f"Condition {condition_idx + 1}\n")
            # f.write(f"{'='*80}\n\n")
            
            # Extract vehicle data
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
            
            # Fill data
            for data in trajectory_data:
                for vehicle_state in data["state"]:
                    vid = vehicle_state["id"]
                    if vid in vehicle_data:
                        vehicle_data[vid]["position"].append(vehicle_state["x"])
                        vehicle_data[vid]["speed"].append(vehicle_state["v"])
                        vehicle_data[vid]["acceleration"].append(vehicle_state["a"])
            
            # Calculate jerk
            for vid, data in vehicle_data.items():
                accelerations = np.array(data["acceleration"])
                if len(accelerations) > 1:
                    jerk = np.diff(accelerations) / dt
                    jerk = np.concatenate([[0], jerk])
                    data["jerk"] = jerk
                else:
                    data["jerk"] = np.array([0] * len(accelerations))
            
            # Separate CAV and HV
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
                    
                    sum_lead_acc_sq = np.sqrt(np.sum(lead_acc ** 2))    
                    sum_follow_acc_sq = np.sqrt(np.sum(follow_acc ** 2))
                    
                    if sum_follow_acc_sq > 1e-6:
                        ratio = sum_lead_acc_sq/sum_follow_acc_sq
                        cav_acc_sq_ratios.append(ratio)
                    
                    # New Speed Stability Metric
                    lead_speed = np.array(vehicle_data[lead_vid]["speed"])
                    follow_speed = np.array(vehicle_data[follow_vid]["speed"])
                    
                    sum_lead_speed = np.sum(lead_speed)
                    sum_follow_speed = np.sum(follow_speed)
                    
                    if sum_lead_speed > 1e-6:
                        ratio = sum_follow_speed / sum_lead_speed
                        cav_speed_ratios.append(ratio)
            
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
            # deltad = headway distance error (gap error)
            # deltav = speed error
            cav_tracking_errors = []
            
            for data in trajectory_data:
                if "obs" in data:
                    obs = data["obs"]  # Shape: (num_cav, 3) where columns are [gap_error, speed_error, accel_error]
                    
                    # For each CAV, calculate |deltad/10| + |deltav|
                    for cav_obs in obs:
                        deltad = cav_obs[0]  # Gap error (headway distance error)
                        deltav = cav_obs[1]  # Speed error
                        
                        # Calculate tracking error: |deltad/10| + |deltav|
                        tracking_error = np.abs(deltad / 10.0) + np.abs(deltav)
                        cav_tracking_errors.append(tracking_error)
            
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

def run_simulation():
    print("Starting Feedback Control Simulation...")
    print("=" * 50)
    
    max_steps = 600
    dt = 0.1
    
    # Use random oscillating speed sequence (same as flexible_rl_test.py)
    speed_sequence = generate_random_oscillating_speed_sequence(max_steps, dt)
    
    print(f"Testing Feedback Control...")
    
    # Initialize environment
    env = CyberAttackEnvErrorObs(
        num_vehicles=7,  # 总共6辆车
        cav_indices=[1,3,6],  # 第二辆和第六辆是CAV（索引从0开始）
        dt=0.1,
        enable_cyber_attack=True,
        attack_frequency=0.4,
        attack_type='data_tampering',
        attack_targets=["speed", "acceleration",'position'],
          # 独立方差配置
        use_cbf=False,
        filter_alpha=0.1,
        force_lead_cav_p_one=False,
        attack_start_time=5,
        attack_variances={"speed": 20.0, "acceleration": 1.50, "position": 10.0},
        attack_means={"speed": 8.0, "acceleration": 1.50, "position": 20.0},
    )
    
    # Reset with speed sequence
    reset_options = {
        "lead_speed_sequence": speed_sequence
    }
    obs, _ = env.reset(seed=42, options=reset_options)
    
    trajectory_data = []
    
    for t in range(max_steps):
        # Compute actions using feedback controller
        actions = feedback_controller(obs, k_gap=3, k_v=0.4, k_a=2.5)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(actions)
        
        # Store state data
        step_data = {
            "time": env.sim.t,
            "state": env.sim.get_state(),
            "info": info,
            "obs": obs.copy()  # Store observation data for tracking error metrics
        }
        trajectory_data.append(step_data)
        
        if terminated or truncated:
            print(f"Simulation ended at step {t}")
            break
            
    env.close()
    print(f"Simulation finished, {len(trajectory_data)} steps.")
    
    # Calculate and save metrics (single condition)
    calculate_and_save_dimensionless_metrics([trajectory_data], "Feedback_Control", dt)
    
    # Save trajectory data to CSV
    save_trajectory_to_csv(trajectory_data, dt, "Feedback_Control", "single_run")

def run_attack_frequency_simulation():
    """
    Test feedback control under different attack frequencies (0.1, 0.3, 0.5)
    """
    print("Starting Feedback Control Simulation (Attack Frequency Test)...")
    print("=" * 50)
    
    max_steps = 600
    dt = 0.1
    
    # Use random oscillating speed sequence for all tests
    speed_sequence = generate_random_oscillating_speed_sequence(max_steps, dt)
    
    # Test three different attack frequencies
    attack_frequencies = [0.1, 0.3, 0.5]
    all_trajectory_data = []
    
    for i, freq in enumerate(attack_frequencies):
        print(f"Testing Attack Frequency {freq}...")
        
        # Initialize environment with specific attack frequency
        env = CyberAttackEnvErrorObs(
            num_vehicles=7,
            cav_indices=[1,3,6],
            dt=0.1,
            enable_cyber_attack=True,
            attack_frequency=freq,
            attack_type='data_tampering',
            attack_targets=["speed", "acceleration", 'position'],
            use_cbf=False,
            filter_alpha=0.1,
            force_lead_cav_p_one=False,
            attack_start_time=5,
            attack_variances={"speed": 2.0, "acceleration": 0.50, "position": 3.0},
            attack_means={"speed": 8.0, "acceleration": 1.50, "position": 15.0},
        )
        
        # Reset with speed sequence
        reset_options = {
            "lead_speed_sequence": speed_sequence
        }
        obs, _ = env.reset(seed=42, options=reset_options)
        
        trajectory_data = []
        
        for t in range(max_steps):
            # Compute actions using feedback controller
            actions = feedback_controller(obs, k_gap=3, k_v=0.4, k_a=2.5)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(actions)
            
            # Store state data
            step_data = {
                "time": env.sim.t,
                "state": env.sim.get_state(),
                "info": info,
                "obs": obs.copy()  # Store observation data for tracking error metrics
            }
            trajectory_data.append(step_data)
            
            if terminated or truncated:
                print(f"Simulation ended at step {t}")
                break
        
        # Display attack statistics
        if env.enable_cyber_attack:
            attack_stats = env.get_cyber_attack_stats()
            if attack_stats["enabled"]:
                stats = attack_stats["statistics"]
                print(f"   - Total attacks: {stats['total_attacks']}")
                print(f"   - Actual attack rate: {stats['actual_attack_rate']:.2f}")
                
        env.close()
        all_trajectory_data.append(trajectory_data)
        print(f"Attack frequency {freq} finished, {len(trajectory_data)} steps.")
    
    # Calculate and save metrics with scenario names
    scenario_names = [f"Attack Freq {f}" for f in attack_frequencies]
    calculate_and_save_dimensionless_metrics(all_trajectory_data, "Feedback_Control", dt, scenario_names=scenario_names)
    
    # Save trajectory data to CSV
    print(f"\nSaving trajectory data to CSV...")
    for i, trajectory_data in enumerate(all_trajectory_data):
        scenario_name = f"attack_freq_{attack_frequencies[i]}"
        save_trajectory_to_csv(trajectory_data, dt, "Feedback_Control", scenario_name)

    print("\nAttack frequency test completed!")

if __name__ == "__main__":
    # Test different attack frequencies
    run_attack_frequency_simulation()
    
    # Test two different traffic conditions
    # run_simulation()
