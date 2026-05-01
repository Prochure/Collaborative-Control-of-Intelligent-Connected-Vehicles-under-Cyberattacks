import numpy as np

def generate_deceleration_acceleration_speed_sequence(total_steps, dt):
    """
    生成先减速然后匀速，然后加速的速度序列
    
    参数:
        total_steps: 总步数
        dt: 时间步长
    
    返回:
        speed_sequence: 速度序列 (m/s)
    """
    speed_sequence = np.full(total_steps, 18.0)  # 默认速度 18 m/s
    
    # 第一阶段：减速 (前100步)
    deceleration_steps = 60
    # 从18 m/s减速到8 m/s
    speed_sequence[:deceleration_steps] = np.linspace(18.0, 5.0, deceleration_steps)
    
    # 第二阶段：匀速 (接下来的200步)
    constant_speed_steps = 200
    start_constant = deceleration_steps
    # 保持8 m/s
    speed_sequence[start_constant:start_constant+constant_speed_steps] = 5.0
    
    # 第三阶段：加速 (接下来的100步)
    acceleration_steps = 60
    start_acceleration = start_constant + constant_speed_steps
    # 从8 m/s加速到18 m/s
    speed_sequence[start_acceleration:start_acceleration+acceleration_steps] = np.linspace(5.0, 18.0, acceleration_steps)
    
    # 剩余时间保持18 m/s
    speed_sequence[start_acceleration+acceleration_steps:] = 18.0
    
    return speed_sequence


def generate_random_oscillating_speed_sequence(total_steps, dt):
    """
    生成随机震荡速度序列，具有规则性且加速度变化平滑
    
    参数:
        total_steps: 总步数
        dt: 时间步长
    
    返回:
        speed_sequence: 随机震荡速度序列 (m/s)
    """
    t = np.arange(total_steps) * dt
    
    # 基础速度 (15 m/s) + 多频率振荡分量（确保加速度平滑）
    speed_sequence = 15.0 + (
        2.0 * np.sin(0.3 * t) +      # 低频振荡（周期约21秒）
        1.2 * np.sin(0.8 * t) +      # 中频振荡（周期约8秒）
        0.6 * np.sin(1.5 * t) +      # 较高频振荡（周期约4秒）
        0.3 * np.sin(3.0 * t)        # 高频振荡（周期约2秒）
    )
    
    # 添加平滑的随机变化分量
    # 生成低频随机噪声，确保加速度变化平滑
    np.random.seed(42)  # 固定种子确保可重现性
    random_component = np.zeros(total_steps)
    
    # 每隔50步生成一个随机值，然后线性插值实现平滑过渡
    segment_length = 50
    num_segments = total_steps // segment_length + 1
    
    random_values = np.random.normal(0, 0.3, num_segments)  # 降低随机噪声幅度
    
    for i in range(num_segments - 1):
        start_idx = i * segment_length
        end_idx = min((i + 1) * segment_length, total_steps)
        
        # 线性插值实现平滑过渡
        interpolation = np.linspace(random_values[i], random_values[i + 1], end_idx - start_idx)
        random_component[start_idx:end_idx] = interpolation
    
    speed_sequence += random_component
    
    # 限制速度范围在合理区间内 (8-22 m/s)
    speed_sequence = np.clip(speed_sequence, 8.0, 22.0)
    
    return speed_sequence
