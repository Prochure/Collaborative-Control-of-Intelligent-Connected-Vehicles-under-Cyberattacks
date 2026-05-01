#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整测试HV车辆状态预测效果
该脚本用于评估在不同网络攻击类型下，模型对HV车辆速度预测的准确性
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim_env.envs.cyber_attack_env import CyberAttackEnv

def test_hv_prediction_under_different_attacks():
    """
    测试不同网络攻击环境下HV车辆状态预测效果
    """
    print("测试不同网络攻击环境下HV车辆状态预测效果")
    print("=" * 50)
    
    # 定义不同的攻击配置进行测试
    attack_configs = [
        {
            "name": "无攻击",
            "enable_cyber_attack": False,
        },
        {
            "name": "轻度数据篡改",
            "enable_cyber_attack": True,
            "attack_type": "data_tampering",
            "attack_frequency": 0.1,
            "attack_variances": {"speed": 1.0, "position": 0.5, "acceleration": 0.2},
        },
        {
            "name": "中度数据篡改",
            "enable_cyber_attack": True,
            "attack_type": "data_tampering",
            "attack_frequency": 0.3,
            "attack_variances": {"speed": 2.0, "position": 1.0, "acceleration": 0.5},
        },
        {
            "name": "重度数据篡改",
            "enable_cyber_attack": True,
            "attack_type": "data_tampering",
            "attack_frequency": 0.5,
            "attack_variances": {"speed": 3.0, "position": 1.5, "acceleration": 1.0},
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
            "cav_indices": [1, 4],
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
                "attack_variances": config["attack_variances"],
            })
        
        env = CyberAttackEnv(**env_kwargs)
        
        # 重置环境
        obs, info = env.reset()
        
        # 数据记录
        time_steps = []
        predicted_speeds = []
        actual_speeds = []
        cav_p_values = []
        
        # 仿真运行
        max_steps = 300  # 运行足够长时间以确保预测功能启动
        
        print("开始仿真...")
        successful_predictions = 0
        
        for step in range(max_steps):
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(np.array([0.5, -0.2]))
            
            # 获取预测的HV状态
            predicted_states = env.get_predicted_hv_states()
            
            # 获取CAV的p值
            p_values = env.get_cav_p_values()
            
            # 获取实际的HV状态（获取两个CAV之间的HV车辆）
            actual_hv_speeds = []
            if len(env.sim.vehicles) > 4:
                # 获取位置2和3的HV车辆速度
                for i in [2, 3]:
                    if i < len(env.sim.vehicles) and not env.sim.vehicles[i].is_cav:
                        actual_hv_speeds.append(env.sim.vehicles[i].speed)
            
            # 记录数据
            time_steps.append(env.sim.t)
            
            # 获取后车CAV(cav4)的预测结果
            if 'V4' in predicted_states:
                predicted_speeds.append(predicted_states['V4']['speed'])
                successful_predictions += 1
            else:
                predicted_speeds.append(np.nan)
                
            if actual_hv_speeds:
                actual_speeds.append(np.mean(actual_hv_speeds))
            else:
                actual_speeds.append(np.nan)
                
            # 记录前车CAV的p值
            if 'V1' in p_values:
                cav_p_values.append(p_values['V1'])
            else:
                cav_p_values.append(1.0)
            
            # 输出信息
            if step % 100 == 0 and step > 0:
                print(f"时间 {env.sim.t:4.1f}s:")
                print(f"  CAV p值: {p_values}")
                if actual_hv_speeds:
                    print(f"  实际HV平均速度: {np.mean(actual_hv_speeds):.2f}")
                if 'V4' in predicted_states:
                    print(f"  预测HV状态: 速度={predicted_states['V4']['speed']:.2f}, 加速度={predicted_states['V4']['acceleration']:.4f}")
                print()
            
            if terminated or truncated:
                break
        
        print(f"测试完成，成功预测次数: {successful_predictions}")
        
        # 存储结果
        all_results[config["name"]] = {
            "time_steps": time_steps,
            "predicted_speeds": predicted_speeds,
            "actual_speeds": actual_speeds,
            "cav_p_values": cav_p_values,
        }
        
        # 清理环境
        env.close()
    
    # 绘制对比图
    plot_comparison_results(all_results)
    
    # 计算并显示误差统计
    calculate_error_statistics(all_results)
    
    return all_results

def plot_comparison_results(all_results):
    """
    绘制不同攻击配置下的预测效果对比图
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('不同网络攻击环境下HV车辆速度预测效果对比', fontsize=16)
    
    # 绘制每种攻击配置的结果
    colors = ['blue', 'green', 'orange', 'red']
    
    for idx, (config_name, data) in enumerate(all_results.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # 转换为numpy数组并过滤掉nan值
        time_steps = np.array(data["time_steps"])
        predicted_speeds = np.array(data["predicted_speeds"])
        actual_speeds = np.array(data["actual_speeds"])
        
        # 过滤掉nan值
        valid_indices = ~np.isnan(predicted_speeds) & ~np.isnan(actual_speeds)
        valid_time_steps = time_steps[valid_indices]
        valid_predicted = predicted_speeds[valid_indices]
        valid_actual = actual_speeds[valid_indices]
        
        if len(valid_predicted) > 0:
            # 绘制实际速度和预测速度
            ax.plot(valid_time_steps, valid_actual, color='black', linewidth=2, label='实际HV速度')
            ax.plot(valid_time_steps, valid_predicted, color=colors[idx], linewidth=2, 
                    linestyle='--', label=f'预测HV速度')
            
            # 计算并显示相关系数
            if len(valid_predicted) > 1:
                correlation = np.corrcoef(valid_actual, valid_predicted)[0, 1]
                ax.text(0.05, 0.95, f'相关系数: {correlation:.3f}', 
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('速度 (m/s)')
        ax.set_title(config_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "hv_prediction_attack_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 预测效果对比图已保存到: {plot_path}")
    
    # 显示图像
    plt.show()

def calculate_error_statistics(all_results):
    """
    计算并显示不同攻击配置下的预测误差统计
    """
    print("\n" + "=" * 60)
    print("预测误差统计")
    print("=" * 60)
    
    for config_name, data in all_results.items():
        # 转换为numpy数组并过滤掉nan值
        predicted_speeds = np.array(data["predicted_speeds"])
        actual_speeds = np.array(data["actual_speeds"])
        
        # 过滤掉nan值
        valid_indices = ~np.isnan(predicted_speeds) & ~np.isnan(actual_speeds)
        valid_predicted = predicted_speeds[valid_indices]
        valid_actual = actual_speeds[valid_indices]
        
        if len(valid_predicted) == 0:
            print(f"{config_name}: 没有有效的数据点用于计算误差")
            continue
        
        # 计算各种误差指标
        errors = valid_predicted - valid_actual
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(errors))
        
        # 计算MAPE，避免除以0
        non_zero_actual = valid_actual != 0
        if np.sum(non_zero_actual) > 0:
            mape = np.mean(np.abs(errors[non_zero_actual] / valid_actual[non_zero_actual])) * 100
            accuracy = 100 - mape
        else:
            mape = np.nan
            accuracy = np.nan
            
        # 计算相关系数
        if len(valid_predicted) > 1:
            correlation = np.corrcoef(valid_actual, valid_predicted)[0, 1]
        else:
            correlation = np.nan
        
        print(f"\n{config_name}:")
        print(f"   - 有效数据点数量: {len(valid_predicted)}")
        print(f"   - 均方误差(MSE): {mse:.4f}")
        print(f"   - 均方根误差(RMSE): {rmse:.4f}")
        print(f"   - 平均绝对误差(MAE): {mae:.4f}")
        if not np.isnan(mape):
            print(f"   - 平均绝对百分比误差(MAPE): {mape:.2f}%")
            print(f"   - 预测精度: {accuracy:.2f}%")
        if not np.isnan(correlation):
            print(f"   - 预测相关系数: {correlation:.4f}")

if __name__ == "__main__":
    test_hv_prediction_under_different_attacks()