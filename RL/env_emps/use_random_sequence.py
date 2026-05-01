"""
使用随机生成的加速度序列运行仿真的简单示例

此脚本展示如何：
1. 加载预生成的随机加速度序列
2. 在仿真中应用此序列到第一辆HV
3. 观察仿真效果
"""

import os
import sys
import numpy as np

# 允许直接运行本文件：将项目根目录加入 sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sim_env.envs.single_lane_env import SingleLaneFollowingEnv


def load_acceleration_sequence(filename="examples/random_lead_acceleration.txt"):
    """从文件加载加速度序列"""
    try:
        sequence = np.loadtxt(filename)
        print(f"✓ 成功加载序列文件: {filename}")
        print(f"  序列长度: {len(sequence)} 步")
        print(f"  仿真时长: {len(sequence) * 0.1:.1f} 秒")
        print(f"  加速度范围: [{sequence.min():.2f}, {sequence.max():.2f}] m/s²")
        print(f"  平均加速度: {sequence.mean():.3f} m/s²")
        return sequence
    except FileNotFoundError:
        print(f"✗ 未找到序列文件: {filename}")
        print("  正在生成新的随机序列...")
        # 生成新序列
        np.random.seed(42)
        sequence = np.random.uniform(-2.0, 2.0, 300)
        np.savetxt(filename, sequence, fmt="%.3f")
        print(f"✓ 已生成并保存新序列到: {filename}")
        return sequence


def main():
    """主函数"""
    print("🚗 使用随机加速度序列控制第一辆HV的仿真演示\n")
    
    # 加载加速度序列
    sequence = load_acceleration_sequence()
    
    print(f"\n📈 序列预览（前10个值）:")
    print(f"   {sequence[:10]}")
    
    print(f"\n🚗 创建仿真环境...")
    # 创建环境：4辆车，第2辆为CAV
    env = SingleLaneFollowingEnv(
        seed=42,
        num_vehicles=4,
        cav_indices=[1],  # 第1辆车为CAV（索引从0开始）
        dt=0.1
    )
    
    print(f"✓ 环境已创建")
    print(f"  车辆总数: 4 辆")
    print(f"  CAV 索引: [1] (第2辆车)")
    print(f"  第一辆车将使用随机加速度序列控制")
    
    # 重置环境并传入序列
    obs, info = env.reset(options={
        "base_gap": 25.0,
        "v0": 10.0,
        "lead_acc_sequence": sequence  # 关键：传入加速度序列
    })
    
    print(f"\n🎬 开始仿真...")
    print(f"CAV 车辆: {env.cav_ids}")
    
    print(f"\n初始状态:")
    print(env.render())
    
    # 仿真循环
    collision_occurred = False
    max_steps = min(100, len(sequence))  # 最多运行100步或序列长度
    
    for step in range(max_steps):
        # CAV 动作：保持稳定（加速度为0）
        cav_action = [5]
        
        # 执行仿真步骤
        obs, reward, terminated, truncated, info = env.step({})
        
        # 每20步显示状态
        if step % 20 == 0 or step < 3:
            print(f"\n--- 步骤 {step} (t={step*0.1:.1f}s) ---")
            print(env.render())
            if step < len(sequence):
                print(f"当前领车加速度: {sequence[step]:.2f} m/s²")
        
        if terminated:
            collision_occurred = True
            print(f"\n⚠️ 仿真在第 {step} 步终止")
            if "collision" in info and info["collision"]:
                follower, leader = info["collision"]
                print(f"   碰撞发生: {follower} 撞上了 {leader}")
            break
    
    # 仿真完成
    print(f"\n🏁 仿真完成!")
    final_step = step if collision_occurred else max_steps - 1
    print(f"   总步数: {final_step + 1}")
    print(f"   仿真时长: {(final_step + 1) * 0.1:.1f} 秒")
    
    if not collision_occurred:
        print(f"   ✓ 无碰撞发生")
    
    print(f"\n📊 序列使用情况:")
    print(f"   序列总长度: {len(sequence)} 步")
    print(f"   已使用: {final_step + 1} 步 ({((final_step + 1)/len(sequence)*100):.1f}%)")
    
    # 绘制时间序列图
    try:
        print(f"\n📈 正在生成时间序列图...")
        env.plot_timeseries()
        print(f"✓ 图表已生成")
    except Exception as e:
        print(f"✗ 绘图失败: {e}")
    
    print(f"\n💡 使用说明:")
    print(f"   1. 加速度序列文件: examples/random_lead_acceleration.txt")
    print(f"   2. 第一辆车 (V0) 按照序列控制加速度")
    print(f"   3. CAV 车辆 (V1) 使用 IDM 模型跟驰")
    print(f"   4. 其他 HV 车辆也使用 IDM 模型")
    
    return env, sequence


if __name__ == "__main__":
    try:
        env, sequence = main()
        print(f"\n🎉 演示完成! 您可以修改序列文件来测试不同的驾驶行为。")
    except KeyboardInterrupt:
        print(f"\n⏹️ 用户中断了仿真")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()