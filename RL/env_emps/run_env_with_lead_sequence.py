import os
import sys
import numpy as np

# 允许直接运行本文件：将项目根目录加入 sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sim_env.envs.single_lane_env import SingleLaneFollowingEnv


def generate_lead_acceleration_sequence(duration=50.0, dt=0.1, scenario="mixed"):
    """
    生成领车加速度序列
    
    Args:
        duration: 持续时间（秒）
        dt: 时间步长（秒）
        scenario: 场景类型 ("smooth", "aggressive", "mixed", "custom")
    
    Returns:
        加速度序列数组
    """
    n_steps = int(duration / dt)
    np.random.seed(42)  # 固定种子确保可重现
    
    if scenario == "smooth":
        # 平稳驾驶：小幅度变化
        base_noise = np.random.normal(0, 0.3, n_steps)
        trend = 0.5 * np.sin(np.linspace(0, 4*np.pi, n_steps))
        sequence = base_noise + trend
        sequence = np.clip(sequence, -1.0, 1.0)
        
    elif scenario == "aggressive":
        # 激进驾驶：大幅度随机变化
        sequence = np.random.uniform(-2.0, 2.0, n_steps)
        # 添加一些平滑处理
        for i in range(1, n_steps):
            if abs(sequence[i] - sequence[i-1]) > 1.5:
                sequence[i] = sequence[i-1] + np.random.uniform(-0.8, 0.8)
                
    elif scenario == "mixed":
        # 混合场景：包含加速、减速、稳定等阶段
        sequence = np.zeros(n_steps)
        phase_length = n_steps // 4
        
        # 第一阶段：稳定
        sequence[:phase_length] = np.random.normal(0, 0.2, phase_length)
        
        # 第二阶段：加速
        sequence[phase_length:2*phase_length] = np.random.uniform(0.5, 1.5, phase_length)
        
        # 第三阶段：急刹车
        sequence[2*phase_length:3*phase_length] = np.random.uniform(-2.0, -0.5, phase_length)
        
        # 第四阶段：恢复
        sequence[3*phase_length:] = np.random.normal(0.3, 0.4, n_steps - 3*phase_length)
        
        # 限制范围
        sequence = np.clip(sequence, -3.0, 2.0)
        
    elif scenario == "custom":
        # 自定义场景：模拟真实交通情况
        sequence = []
        current_accel = 0.0
        
        for i in range(n_steps):
            # 80% 概率保持当前加速度附近的小变化
            if np.random.random() < 0.8:
                change = np.random.normal(0, 0.1)
            else:
                # 20% 概率大幅改变
                change = np.random.uniform(-1.0, 1.0)
            
            current_accel += change
            current_accel = np.clip(current_accel, -2.5, 2.0)
            sequence.append(current_accel)
        
        sequence = np.array(sequence)
    
    else:
        raise ValueError(f"未知的场景类型: {scenario}")
    
    return sequence


def run_simulation_with_lead_sequence():
    """运行带有领车加速度序列的仿真"""
    
    print("=== 生成领车加速度序列 ===")
    # 生成加速度序列
    lead_sequence = generate_lead_acceleration_sequence(
        duration=30.0,  # 30秒的仿真
        dt=0.1,
        scenario="mixed"  # 使用混合场景
    )
    
    print(f"生成序列长度: {len(lead_sequence)} 步")
    print(f"序列统计信息:")
    print(f"  最小值: {lead_sequence.min():.2f} m/s²")
    print(f"  最大值: {lead_sequence.max():.2f} m/s²")
    print(f"  平均值: {lead_sequence.mean():.2f} m/s²")
    print(f"  标准差: {lead_sequence.std():.2f} m/s²")
    
    print(f"\n前10个加速度值: {lead_sequence[:10]}")
    
    # 创建环境
    env = SingleLaneFollowingEnv(
        seed=42,
        num_vehicles=6,
        cav_indices=[2, 4],  # 第2和第4辆车为CAV
        dt=0.1
    )
    
    # 重置环境，传入领车加速度序列
    reset_options = {
        "base_gap": 20.0,
        "v0": 8.0,
        "lead_acc_sequence": lead_sequence  # 传入生成的序列
    }
    
    obs, info = env.reset(options=reset_options)
    
    print(f"\n=== 开始仿真 ===")
    print(f"CAV 车辆 ID: {env.cav_ids}")
    print(f"初始状态:")
    print(env.render())
    
    # 仿真循环
    step_count = 0
    max_steps = len(lead_sequence)
    
    for step in range(max_steps):
        # CAV 动作：第一辆CAV保持稳定，第二辆CAV轻微制动
        actions = {}  # 对应两辆CAV的动作
        
        # 执行仿真步骤
        step_result = env.step(actions)
        obs, reward, terminated, truncated, info = step_result
        
        step_count += 1
        
        # 每10步显示一次状态
        if step % 10 == 0 or step < 5:
            print(f"\n步骤 {step}:")
            print(env.render())
            
            # 显示当前领车加速度
            if step < len(lead_sequence):
                print(f"领车当前加速度: {lead_sequence[step]:.2f} m/s²")
        
        if terminated:
            print(f"\n仿真在第 {step} 步终止")
            if "collision" in info and info["collision"]:
                follower, leader = info["collision"]
                print(f"检测到碰撞: {follower} -> {leader}")
            break
    
    print(f"\n=== 仿真完成 ===")
    print(f"总步数: {step_count}")
    
    # 绘制结果
    try:
        print("正在生成时间序列图...")
        env.plot_timeseries()
    except Exception as e:
        print(f"绘图失败: {e}")
    
    return env, lead_sequence


if __name__ == "__main__":
    env, sequence = run_simulation_with_lead_sequence()
    
    # 保存序列到文件
    np.savetxt("examples/used_lead_sequence.txt", sequence, fmt="%.3f")
    print(f"\n使用的加速度序列已保存到: examples/used_lead_sequence.txt")