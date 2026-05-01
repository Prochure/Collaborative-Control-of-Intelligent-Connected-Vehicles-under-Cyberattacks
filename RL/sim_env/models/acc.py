"""
Adaptive Cruise Control (ACC) Model

ACC是一种基于距离和速度控制的车辆跟驰模型，
通过PID控制器来维持期望的车间距离和速度。
"""

import numpy as np
from typing import Optional


class ACCParameters:
    """ACC控制器参数"""
    def __init__(
        self,
        desired_time_gap: float = 1.5,  # 期望时间车间距离(s)
        minimum_gap: float = 2.0,  # 最小车间距离(m)
        desired_speed: float = 33.0,  # 期望速度(m/s)
        # PID控制器参数
        kp_gap: float = 0.5,  # 距离误差比例增益
        kd_gap: float = 0.2,  # 距离误差微分增益
        kp_speed: float = 1.0,  # 速度误差比例增益
        # 舒适性参数
        max_acceleration: float = 2.0,  # 最大加速度(m/s²)
        max_deceleration: float = 3.0,  # 最大减速度(m/s²)
    ) -> None:
        self.desired_time_gap = float(desired_time_gap)
        self.s_min = float(minimum_gap)
        self.v_desired = float(desired_speed)
        
        self.kp_gap = float(kp_gap)
        self.kd_gap = float(kd_gap)
        self.kp_speed = float(kp_speed)
        
        self.a_max = float(max_acceleration)
        self.b_max = float(max_deceleration)

    def __repr__(self) -> str:
        return (f"ACCParameters(desired_time_gap={self.desired_time_gap:.1f}, "
                f"s_min={self.s_min:.1f}, v_desired={self.v_desired:.1f}, "
                f"a_max={self.a_max:.1f}, b_max={self.b_max:.1f})")

    def copy(self) -> 'ACCParameters':
        """创建参数的深拷贝"""
        return ACCParameters(
            desired_time_gap=self.desired_time_gap,
            minimum_gap=self.s_min,
            desired_speed=self.v_desired,
            kp_gap=self.kp_gap,
            kd_gap=self.kd_gap,
            kp_speed=self.kp_speed,
            max_acceleration=self.a_max,
            max_deceleration=self.b_max,
        )


def compute_acc_acceleration(
    v: float,
    gap: float,
    rel_speed: float,
    params: ACCParameters,
) -> float:
    """
    计算ACC加速度
    
    ACC控制策略:
    1. 计算期望的间距 (基于时间间隔)
    2. 使用PD控制器计算基于间距误差的控制输入
    3. 考虑相对速度进行调整
    4. 限制在舒适性范围内
    
    Args:
        v: 本车速度 (m/s)
        gap: 到前车的净间距 (m), gap <= 0 表示有碰撞风险
        rel_speed: v - v_lead (m/s). 正值表示本车比前车快
        params: ACC参数
    
    Returns:
        加速度 (m/s²)
    """
    v = float(max(0.0, v))
    
    # 如果没有前车，使用巡航控制
    if np.isinf(gap):
        # 巡航控制: 维持期望速度
        speed_error = params.v_desired - v
        a = params.kp_speed * speed_error
        return float(np.clip(a, -params.b_max, params.a_max))
    
    # 计算期望间距 (时间间隔 + 最小间距)
    desired_gap = params.s_min + params.desired_time_gap * v
    
    # 间距误差
    gap_error = gap - desired_gap
    
    # PD控制: 比例项 + 微分项
    # 微分项使用相对速度 (gap的变化率的负值)
    a_gap = params.kp_gap * gap_error - params.kd_gap * rel_speed
    
    # 速度控制项: 当间距足够大时，考虑速度控制
    if gap > desired_gap:
        speed_error = params.v_desired - v
        a_speed = params.kp_speed * speed_error
        
        # 混合控制: 间距控制和速度控制
        # 当间距误差较小时，更多考虑速度控制
        weight = min(1.0, gap_error / (params.desired_time_gap * params.v_desired + 1e-6))
        a = weight * a_speed + (1.0 - weight) * a_gap
    else:
        # 间距不足，只使用间距控制
        a = a_gap
    
    # 限制在舒适范围内
    return float(np.clip(a, -params.b_max, params.a_max))


def create_acc_parameters_list(
    n_vehicles: int,
    base_params: Optional[ACCParameters] = None,
) -> list:
    """
    为多辆车创建ACC参数列表
    
    Args:
        n_vehicles: 车辆数量
        base_params: 基础参数，如果为None则使用默认参数
    
    Returns:
        包含ACC参数的列表
    """
    if base_params is None:
        base_params = ACCParameters()
    
    # 暂时所有车使用相同参数
    return [base_params.copy() for _ in range(n_vehicles)]
