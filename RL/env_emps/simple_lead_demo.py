import os
import sys
import numpy as np

# 允许直接运行本文件：将项目根目录加入 sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sim_env.envs.single_lane_env import SingleLaneFollowingEnv


def generate_simple_lead_sequence():
    """生成一个简单的领车加速度序列"""
    np.random.seed(42)  # 固定种子确保可重现
    
    # 生成50秒的序列，步长0.1秒
    duration = 50.0
    dt = 0.1
    n_steps = int(duration / dt)  # 500步
    
    # 方法1：随机噪声 + 周期性变化
    time_steps = np.arange(n_steps) * dt
    
    # 基础周期性变化（模拟交通流的周期性）
    periodic_component = 0.8 * np.sin(0.2 * time_steps) + 0.4 * np.cos(0.15 * time_steps)
    
    # 随机噪声（模拟驾驶员的随机行为）
    noise_component = np.random.normal(0, 0.5, n_steps)
    
    # 突发事件（急刹车/急加速）
    event_component = np.zeros(n_steps)
    for i in range(5):  # 添加5个随机事件
        event_time = np.random.randint(50, n_steps-50)
        event_duration = np.random.randint(10, 30)
        event_intensity = np.random.choice([-2.5, 2.0])  # 急刹车或急加速
        event_component[event_time:event_time+event_duration] = event_intensity
    
    # 组合所有成分
    sequence = periodic_component + noise_component + 0.3 * event_component
    
    # 限制在合理范围内
    sequence = np.clip(sequence, -3.0, 2.5)
    
    return sequence


def run_demo():
    """运行演示"""
    print("=== 随机生成领车加速度序列演示 ===\n")
    
    # 生成序列
    lead_sequence = generate_simple_lead_sequence()
    
    print(f"生成的序列信息：")
    print(f"  序列长度: {len(lead_sequence)} 步")
    print(f"  仿真时长: {len(lead_sequence) * 0.1:.1f} 秒")
    print(f"  加速度范围: [{lead_sequence.min():.2f}, {lead_sequence.max():.2f}] m/s²")
    print(f"  平均加速度: {lead_sequence.mean():.3f} m/s²")
    print(f"  标准差: {lead_sequence.std():.3f} m/s²")
    
    print(f"\n前20个加速度值：")
    for i in range(0, 20, 5):
        values = [f"{val:6.2f}" for val in lead_sequence[i:i+5]]
        print(f"  {i:2d}-{i+4:2d}: {' '.join(values)}")
    
    # 保存序列到文件
    filename = "examples/generated_lead_sequence.txt"
    np.savetxt(filename, lead_sequence, fmt="%.3f", 
               header=f"Generated lead vehicle acceleration sequence\n"
                      f"Duration: {len(lead_sequence) * 0.1:.1f}s, Steps: {len(lead_sequence)}, dt=0.1s\n"
                      f"Range: [{lead_sequence.min():.2f}, {lead_sequence.max():.2f}] m/s²"
    print(f"\n序列已保存到: {filename}")
    
    print(f"\n=== 开始仿真测试 ===")
    
    # 创建环境
    env = SingleLaneFollowingEnv(
        seed=42,
        num_vehicles=6,
        cav_indices=[2,5],  # 第2辆车为CAV
        dt=0.1
    )
    
    # 重置环境并传入序列
    reset_options = {
        "base_gap": 25.0,
        "v0": 10.0,
        "lead_acc_sequence": lead_sequence
    }
    
    obs, info = env.reset(options=reset_options)
    
    print(f"CAV 车辆: {env.cav_ids}")
    print(f"初始状态:")
    print(env.render())
    
    # 运行前面100步来演示效果
    for step in range(min(100, len(lead_sequence))):
        # CAV动作：保持稳定
        action = {}
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 每20步显示一次
        if step % 20 == 0:
            print(f"\n--- 步骤 {step} (t={step*0.1:.1f}s) ---")
            print(env.render())
            print(f"当前领车加速度: {lead_sequence[step]:.2f} m/s²")
        
        if terminated:
            print(f"\n仿真在第 {step} 步终止")
            break
    

    env.plot_timeseries()
    print(f"\n=== 演示完成 ===")
    
    return env, lead_sequence


if __name__ == "__main__":
    env, sequence = run_demo()
    
    print(f"\n如何使用生成的序列：")
    print(f"1. 序列文件: examples/generated_lead_sequence.txt")
    print(f"2. 在环境重置时传入: reset(options={{'lead_acc_sequence': sequence}})")
    print(f"3. 第一辆车将按照这个序列执行加速度控制")
    
    # 额外输出一些统计信息
    print(f"\n序列统计分析：")
    print(f"  正加速度步数: {np.sum(sequence > 0)} ({np.sum(sequence > 0)/len(sequence)*100:.1f}%)")
    print(f"  负加速度步数: {np.sum(sequence < 0)} ({np.sum(sequence < 0)/len(sequence)*100:.1f}%)")
    print(f"  零加速度步数: {np.sum(sequence == 0)} ({np.sum(sequence == 0)/len(sequence)*100:.1f}%)")
    print(f"  最大正加速度: {sequence.max():.2f} m/s²")
    print(f"  最大负加速度: {sequence.min():.2f} m/s²")