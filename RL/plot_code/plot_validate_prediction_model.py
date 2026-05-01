#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测模型准确性验证脚本

该脚本用于验证在不同网络攻击环境下，增强模型对HV车辆状态预测的准确性。
参考 flexible_rl_test.py 中的方法来获取预测值和真实值。
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim_env.envs.cyber_attack_env import CyberAttackEnv


def _apply_publication_style():
    """统一的出版级绘图风格"""
    plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 15,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 15
    })

def validate_prediction_model():
    """
    验证预测模型的准确性，比较预测值和真实值
    """
    print("🧪 预测模型准确性验证")
    print("=" * 50)
    
    # 定义不同的攻击配置进行测试
    attack_configs = [
        {
            "name": "No Attack",
            "enable_cyber_attack": False,
        },
        {
            "name": "Data Tampering Attack",
            "enable_cyber_attack": True,
            "attack_type": "data_tampering",
            "attack_frequency": 0.2,
            "attack_variances": {"speed": 1.5, "position": 1.0, "acceleration": 0.5},
        },
        {
            "name": "Delay Attack",
            "enable_cyber_attack": True,
            "attack_type": "delay",
            "attack_frequency": 0.2,
            "delay_steps": 2,
        },
        {
            "name": "Packet Drop Attack",
            "enable_cyber_attack": True,
            "attack_type": "packet_drop",
            "attack_frequency": 0.3,  # 统一使用attack_frequency作为攻击概率
        }
    ]
    
    # 存储所有测试结果
    all_results = {}
    
    for config in attack_configs:
        print(f"\n测试配置: {config['name']}")
        print("-" * 30)
        
        # 创建环境
        env_kwargs = {
            "num_vehicles": 6,
            "cav_indices": [1, 4],  # 第二辆和第五辆是CAV
            "enable_cyber_attack": config["enable_cyber_attack"],
            "attack_start_time": 5.0,
            "dt": 0.1
        }
        
        # 如果启用攻击，添加攻击相关参数
        if config["enable_cyber_attack"]:
            env_kwargs.update({
                "attack_type": config["attack_type"],
                "attack_frequency": config["attack_frequency"],
                "attack_targets": ["speed", "position", "acceleration"],
            })
            
            # 根据攻击类型添加特定参数
            if config["attack_type"] == "data_tampering":
                env_kwargs["attack_variances"] = config["attack_variances"]
            elif config["attack_type"] == "delay":
                env_kwargs["delay_steps"] = config["delay_steps"]
            elif config["attack_type"] == "packet_drop":
                pass  # 使用attack_frequency作为攻击概率
        
        env = CyberAttackEnv(**env_kwargs)
        
        # 重置环境
        obs, info = env.reset()
        
        # 获取CAV ID（根据索引生成，格式为"V{index}"）
        cav_ids = [f"V{idx}" for idx in env_kwargs["cav_indices"]]
        print(f"CAV IDs: {cav_ids}")
        
        # 数据记录
        time_steps = []
        predicted_data = []  # 存储预测数据
        real_data = []      # 存储真实数据
        cav_p_values = []   # 存储CAV的p值
        
        max_steps = 500  # 运行足够长时间以确保预测功能启动
        
        print("开始仿真...")
        successful_predictions = 0
        
        for step in range(max_steps):
            action = [None, None]  # IDM 控制
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            predicted_states = env.get_predicted_hv_states()
            real_states = env.get_real_hv_states()
            p_values = env.get_cav_p_values()
            
            time_steps.append(env.sim.t)
            
            predicted_data.append(predicted_states if predicted_states else {})
            real_data.append(real_states if real_states else {})
            cav_p_values.append(p_values.copy() if p_values else {})
            
            if predicted_states:
                successful_predictions += 1
            
            if terminated or truncated:
                print(f"⚠️  仿真在第 {step} 步结束")
                break
        
        print(f"测试完成，成功预测次数: {successful_predictions}")
        
        all_results[config["name"]] = {
            "time_steps": time_steps,
            "predicted_data": predicted_data,
            "real_data": real_data,
            "cav_p_values": cav_p_values,
            "cav_ids": cav_ids
        }
        
        env.close()
    
    plot_prediction_results(all_results)
    calculate_prediction_error_statistics(all_results)
    
    return all_results


# ===============================================================
# 🔧 修改过的 plot_prediction_results（加速度轴改为黑色 + m/s²）
# ===============================================================
def plot_prediction_results(all_results):
    """
    绘制预测值与真实值的对比图
    """
    _apply_publication_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 8.8))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.08,
                        hspace=0.252, wspace=0.22)
    
    all_lines = []
    all_labels = []
    
    for idx, (config_name, data) in enumerate(all_results.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        time_steps = data["time_steps"]
        predicted_data = data["predicted_data"]
        real_data = data["real_data"]
        cav_ids = data["cav_ids"]
        
        pred_speeds = []
        real_speeds = []
        pred_accels = []
        real_accels = []
        plot_times = []
        
        for time_step, pred_step, real_step in zip(time_steps, predicted_data, real_data):
            cav_id = cav_ids[1]  # 后车 CAV
            
            if cav_id in pred_step and cav_id in real_step:
                pred_speeds.append(pred_step[cav_id]["speed"])
                real_speeds.append(real_step[cav_id]["speed"])
                pred_accels.append(pred_step[cav_id]["acceleration"])
                real_accels.append(real_step[cav_id]["acceleration"])
                plot_times.append(time_step * 10)  # 将时间步放大十倍
        
        ax_twin = ax.twinx()
        
        line_real_speed = None
        line_pred_speed = None
        if len(real_speeds) > 0:
            line_real_speed, = ax.plot(plot_times, real_speeds, color='black', linewidth=2, label='Real Speed')
            line_pred_speed, = ax.plot(plot_times, pred_speeds, color='blue', linewidth=2, linestyle='--', label='Predicted Speed')
            
            # Y 左轴标签字体放大
            ax.set_ylabel('Speed (m/s)', color='black', fontsize=16)
            ax.tick_params(axis='y', labelcolor='black', labelsize=14)   # ←刻度字体放大
        
        line_real_accel = None
        line_pred_accel = None
        if len(real_accels) > 0:
            line_real_accel, = ax_twin.plot(plot_times, real_accels, color='red', linewidth=2, label='Real Acceleration')
            line_pred_accel, = ax_twin.plot(plot_times, pred_accels, color='green', linewidth=2, linestyle='--', label='Predicted Acceleration')

            # 右轴字体放大
            ax_twin.set_ylabel('Acceleration (m/s²)', color='black', fontsize=16)
            ax_twin.tick_params(axis='y', labelcolor='black', labelsize=14)
        
        # X 轴字体放大，使用新的标签
        ax.set_xlabel('Time Step [0.1 s]', fontsize=16)

        # 取消子图标题，改用子图下方标识
        # ax.set_title(config_name, fontsize=18, pad=6)
        # 在子图下方添加标识
        ax.tick_params(axis='both', labelsize=14)  # 主坐标系刻度字体放大
        ax.grid(True, alpha=0.3)
        
        # Legend
        if idx == 0:
            if line_real_speed: all_lines.append(line_real_speed); all_labels.append('Real Speed')
            if line_pred_speed: all_lines.append(line_pred_speed); all_labels.append('Predicted Speed')
            if line_real_accel: all_lines.append(line_real_accel); all_labels.append('Real Acceleration')
            if line_pred_accel: all_lines.append(line_pred_accel); all_labels.append('Predicted Acceleration')
    
    fig.legend(all_lines, all_labels, 
              loc='upper center', 
              bbox_to_anchor=(0.5, 0.97),
              ncol=4,
              frameon=True,
              fancybox=True,
              shadow=False,
              fontsize=12)
    
    # 每个子图添加 (a)-(d) 标注，字体更大
    subplot_labels = {
        (0, 0): '(a)',
        (0, 1): '(b)',
        (1, 0): '(c)',
        (1, 1): '(d)'
    }
    for (row, col), label in subplot_labels.items():
        ax = axes[row, col]
        ax.text(0.5, -0.18, label, transform=ax.transAxes,
                fontsize=18, fontweight='bold', ha='center', va='top')
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "prediction_model_validation.png")
    plt.savefig(plot_path, dpi=800, bbox_inches='tight')
    print(f"\n📊 Prediction model validation chart saved to: {plot_path}")
    
    plt.show()


def calculate_prediction_error_statistics(all_results):
    print("\n" + "=" * 60)
    print("预测模型误差统计")
    print("=" * 60)
    
    for config_name, data in all_results.items():
        predicted_data = data["predicted_data"]
        real_data = data["real_data"]
        cav_ids = data["cav_ids"]
        
        pred_speeds = []
        real_speeds = []
        pred_accels = []
        real_accels = []
        
        for pred_step, real_step in zip(predicted_data, real_data):
            cav_id = cav_ids[1]
            
            if cav_id in pred_step and cav_id in real_step:
                pred_speeds.append(pred_step[cav_id]["speed"])
                real_speeds.append(real_step[cav_id]["speed"])
                pred_accels.append(pred_step[cav_id]["acceleration"])
                real_accels.append(real_step[cav_id]["acceleration"])
        
        if len(pred_speeds) == 0:
            print(f"{config_name}: 没有有效的数据点用于计算误差")
            continue
        
        speed_errors = np.array(pred_speeds) - np.array(real_speeds)
        speed_mse = mean_squared_error(real_speeds, pred_speeds)
        speed_rmse = np.sqrt(speed_mse)
        speed_mae = mean_absolute_error(real_speeds, pred_speeds)
        
        non_zero_real = np.array(real_speeds) != 0
        if np.sum(non_zero_real) > 0:
            speed_mape = np.mean(np.abs(speed_errors[non_zero_real] / np.array(real_speeds)[non_zero_real])) * 100
            speed_accuracy = 100 - speed_mape
        else:
            speed_mape = np.nan
            speed_accuracy = np.nan
            
        if len(pred_speeds) > 1:
            speed_correlation = np.corrcoef(real_speeds, pred_speeds)[0, 1]
        else:
            speed_correlation = np.nan
        
        accel_errors = np.array(pred_accels) - np.array(real_accels)
        accel_mse = mean_squared_error(real_accels, pred_accels)
        accel_rmse = np.sqrt(accel_mse)
        accel_mae = mean_absolute_error(real_accels, pred_accels)
        
        non_zero_real_accel = np.array(real_accels) != 0
        if np.sum(non_zero_real_accel) > 0:
            accel_mape = np.mean(np.abs(accel_errors[non_zero_real_accel] / np.array(real_accels)[non_zero_real_accel])) * 100
            accel_accuracy = 100 - accel_mape
        else:
            accel_mape = np.nan
            accel_accuracy = np.nan
            
        if len(pred_accels) > 1:
            accel_correlation = np.corrcoef(real_accels, pred_accels)[0, 1]
        else:
            accel_correlation = np.nan
        
        print(f"\n{config_name}:")
        print(f"   - 有效数据点数量: {len(pred_speeds)}")
        print(f"   - 速度预测:")
        print(f"     * 均方误差(MSE): {speed_mse:.4f}")
        print(f"     * 均方根误差(RMSE): {speed_rmse:.4f}")
        print(f"     * 平均绝对误差(MAE): {speed_mae:.4f}")
        if not np.isnan(speed_mape):
            print(f"     * 平均绝对百分比误差(MAPE): {speed_mape:.2f}%")
            print(f"     * 预测精度: {speed_accuracy:.2f}%")
        if not np.isnan(speed_correlation):
            print(f"     * 预测相关系数: {speed_correlation:.4f}")
            
        print(f"   - 加速度预测:")
        print(f"     * 均方误差(MSE): {accel_mse:.6f}")
        print(f"     * 均方根误差(RMSE): {accel_rmse:.6f}")
        print(f"     * 平均绝对误差(MAE): {accel_mae:.6f}")
        if not np.isnan(accel_mape):
            print(f"     * 平均绝对百分比误差(MAPE): {accel_mape:.2f}%")
            print(f"     * 预测精度: {accel_accuracy:.2f}%")
        if not np.isnan(accel_correlation):
            print(f"     * 预测相关系数: {accel_correlation:.4f}")


def main():
    validate_prediction_model()


if __name__ == "__main__":
    main()
