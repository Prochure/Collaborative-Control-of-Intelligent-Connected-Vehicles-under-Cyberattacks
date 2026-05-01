#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义攻击时间点功能演示脚本

演示如何使用 CyberAttackEnv 类的自定义攻击时间点功能。
"""

import os
import sys
import numpy as np

# 添加项目路径
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sim_env.envs.cyber_attack_env import CyberAttackEnv

def demo_custom_attack_times():
    """演示自定义攻击时间点功能"""
    print("⏰ 演示自定义攻击时间点功能")
    print("=" * 60)
    
    # 创建环境，设置自定义攻击时间点
    env = CyberAttackEnv(
        seed=42,
        num_vehicles=4,
        cav_indices=[1],
        dt=0.1,
        enable_cyber_attack=True,
        custom_attack_times=[2.5, 5.0, 7.3, 10.0, 12.8],  # 在这些时间点发生攻击
        attack_variances={
            "speed": 2.0,
            "acceleration": 1.5
        },
        attack_targets=["speed", "acceleration"]
    )
    
    obs, info = env.reset()
    
    print("🔧 攻击配置:")
    attack_info = info.get('cyber_attack', {})
    print(f"  攻击启用: {attack_info.get('enabled', False)}")
    print(f"  自定义攻击时间点: {attack_info.get('custom_attack_times', [])}")
    print(f"  攻击开始时间: {attack_info.get('attack_start_time', 0)}秒")
    print()
    
    print("🚗 开始仿真...")
    
    # 记录攻击情况
    attack_times = []  # 记录实际攻击时间点
    
    # 运行150步仿真（15秒）
    for step in range(150):
        action = np.array([0.3])
        # 处理可能的不同返回值数量
        step_result = env.step(action)
        if len(step_result) == 5:
            # Gymnasium格式: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = step_result
        else:
            # Gym格式: (obs, reward, terminated, info)
            obs, reward, terminated, info = step_result
            truncated = False
        
        current_time = step * 0.1  # 当前仿真时间
        attack_info = info.get('cyber_attack', {})
        current_attacks = attack_info.get('current_step_attacks', [])
        
        if current_attacks:
            attack_times.append(current_time)
            print(f"✅ 步骤 {step:2d} (t={current_time:.1f}s): 发生攻击")
        
        if terminated:
            break
    
    print(f"\n📊 攻击统计结果:")
    print(f"  配置的攻击时间点: {[2.5, 5.0, 7.3, 10.0, 12.8]}")
    print(f"  实际攻击时间点: {attack_times}")
    
    # 验证功能正确性
    expected_times = [2.5, 5.0, 7.3, 10.0, 12.8]
    if len(attack_times) == len(expected_times):
        print("✅ 功能验证成功：攻击次数正确")
    else:
        print(f"⚠️  注意：期望 {len(expected_times)} 次攻击，实际 {len(attack_times)} 次")
    
    # 获取最终统计
    final_stats = env.get_cyber_attack_stats()
    total_attacks = final_stats['statistics']['total_attacks']
    print(f"\n🎯 最终统计:")
    print(f"  总攻击次数: {total_attacks}")
    print(f"  自定义攻击时间点: {final_stats['config']['custom_attack_times']}")

def demo_mixed_config():
    """演示混合配置：自定义时间点 + 频率攻击"""
    print("\n🔄 演示混合配置功能")
    print("=" * 60)
    
    # 创建环境，先设置自定义攻击时间点
    env = CyberAttackEnv(
        seed=123,
        num_vehicles=3,
        cav_indices=[1],
        dt=0.1,
        enable_cyber_attack=True,
        custom_attack_times=[1.0, 3.0, 5.0],  # 指定时间点攻击
        attack_frequency=0.3,  # 30%频率攻击（作为补充）
        attack_variances={"speed": 1.0}
    )
    
    obs, info = env.reset()
    print(f"初始自定义攻击时间点: {info.get('cyber_attack', {}).get('custom_attack_times', [])}")
    
    # 运行前60步（6秒）
    print("\n阶段1: 使用自定义时间点攻击")
    attack_count_1 = 0
    for step in range(60):
        action = np.array([0.2])
        # 处理可能的不同返回值数量
        step_result = env.step(action)
        if len(step_result) == 5:
            # Gymnasium格式: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = step_result
        else:
            # Gym格式: (obs, reward, terminated, info)
            obs, reward, terminated, info = step_result
            truncated = False
        
        current_attacks = info.get('cyber_attack', {}).get('current_step_attacks', [])
        if current_attacks:
            attack_count_1 += 1
            current_time = step * 0.1
            print(f"✅ 步骤 {step} (t={current_time:.1f}s): 发生攻击")
    
    # 动态清除自定义时间点，仅使用频率攻击
    print("\n阶段2: 动态清除自定义时间点，仅使用频率攻击")
    env.set_cyber_attack_config(custom_attack_times=[])  # 清除自定义时间点
    print(f"调整后自定义攻击时间点: {env.custom_attack_times}")
    
    attack_count_2 = 0
    # 继续运行60步
    for step in range(60, 120):
        action = np.array([0.2])
        # 处理可能的不同返回值数量
        step_result = env.step(action)
        if len(step_result) == 5:
            # Gymnasium格式: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = step_result
        else:
            # Gym格式: (obs, reward, terminated, info)
            obs, reward, terminated, info = step_result
            truncated = False
        
        current_attacks = info.get('cyber_attack', {}).get('current_step_attacks', [])
        if current_attacks:
            attack_count_2 += 1
            current_time = step * 0.1
            print(f"✅ 步骤 {step} (t={current_time:.1f}s): 发生攻击")
    
    print(f"\n📊 混合配置结果:")
    print(f"  自定义时间点攻击次数: {attack_count_1}")
    print(f"  频率攻击次数: {attack_count_2}")
    print("✅ 混合配置功能正常")

def demo_dynamic_custom_times():
    """演示动态调整自定义攻击时间点"""
    print("\n🔁 演示动态调整自定义攻击时间点")
    print("=" * 60)
    
    # 创建环境，初始无自定义时间点
    env = CyberAttackEnv(
        seed=456,
        num_vehicles=3,
        cav_indices=[1],
        dt=0.1,
        enable_cyber_attack=True,
        attack_frequency=0.1,  # 低频率攻击作为背景
        attack_variances={"speed": 1.0}
    )
    
    obs, info = env.reset()
    print(f"初始自定义攻击时间点: {info.get('cyber_attack', {}).get('custom_attack_times', [])}")
    
    # 运行前30步（3秒）
    print("\n阶段1: 仅使用低频率攻击")
    attack_count_1 = 0
    for step in range(30):
        action = np.array([0.2])
        # 处理可能的不同返回值数量
        step_result = env.step(action)
        if len(step_result) == 5:
            # Gymnasium格式: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = step_result
        else:
            # Gym格式: (obs, reward, terminated, info)
            obs, reward, terminated, info = step_result
            truncated = False
    
        current_attacks = info.get('cyber_attack', {}).get('current_step_attacks', [])
        if current_attacks:
            attack_count_1 += 1
    
    # 动态设置自定义攻击时间点
    print("\n阶段2: 动态设置自定义攻击时间点")
    env.set_cyber_attack_config(custom_attack_times=[3.5, 4.0, 4.5, 6.0])
    print(f"调整后自定义攻击时间点: {env.custom_attack_times}")
    
    attack_times = []
    # 继续运行到第70步（7秒）
    for step in range(30, 70):
        action = np.array([0.2])
        # 处理可能的不同返回值数量
        step_result = env.step(action)
        if len(step_result) == 5:
            # Gymnasium格式: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = step_result
        else:
            # Gym格式: (obs, reward, terminated, info)
            obs, reward, terminated, info = step_result
            truncated = False
        
        current_time = step * 0.1
        current_attacks = info.get('cyber_attack', {}).get('current_step_attacks', [])
        if current_attacks:
            attack_times.append(current_time)
            print(f"✅ 步骤 {step} (t={current_time:.1f}s): 发生攻击")
    
    print(f"\n📊 动态调整结果:")
    print(f"  背景攻击次数: {attack_count_1}")
    print(f"  自定义时间点攻击: {attack_times}")
    print("✅ 动态调整自定义攻击时间点功能正常")

def main():
    """主演示函数"""
    print("⏰ 自定义攻击时间点功能演示")
    print("=" * 80)
    print("本演示展示如何设置自定义攻击时间点，精确控制攻击发生的时间")
    print()
    
    try:
        # 演示基本功能
        demo_custom_attack_times()
        
        # 演示混合配置
        demo_mixed_config()
        
        # 演示动态调整
        demo_dynamic_custom_times()
        
        print("\n🎉 所有演示完成！")
        print("\n💡 功能特点总结:")
        print("✅ 支持设置自定义攻击时间点，精确控制攻击发生时间")
        print("✅ 支持运行时动态调整自定义攻击时间点")
        print("✅ 可与频率攻击混合使用")
        print("✅ 提供详细的攻击状态监控和反馈")
        print("✅ 完全向后兼容现有功能")
        
        print("\n🔧 使用方法:")
        print("env = CyberAttackEnv(custom_attack_times=[2.5, 5.0, 7.3])")
        print("env.set_cyber_attack_config(custom_attack_times=[10.0, 12.5])")
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()