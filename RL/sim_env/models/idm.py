from __future__ import annotations

import numpy as np
from typing import List, Dict, Optional, Union


class IDMParameters:
	def __init__(
		self,
		desired_speed: float = 33.0,
		minimum_spacing: float = 2.0,
		desired_time_headway: float = 1.6,
		max_acceleration: float = 3.0,
		comfortable_deceleration: float = 3,
		delta: float = 4.0,
	) -> None:
		self.v0 = float(desired_speed)
		self.s0 = float(minimum_spacing)
		self.T = float(desired_time_headway)
		self.a_max = float(max_acceleration)
		self.b_max = float(comfortable_deceleration)
		self.delta = float(delta)

	def __repr__(self) -> str:
		return (f"IDMParameters(v0={self.v0:.1f}, s0={self.s0:.1f}, T={self.T:.1f}, "
				f"a_max={self.a_max:.1f}, b_max={self.b_max:.1f}, delta={self.delta:.1f})")

	def copy(self) -> 'IDMParameters':
		"""创建参数的深拷贝"""
		return IDMParameters(
			desired_speed=self.v0,
			minimum_spacing=self.s0,
			desired_time_headway=self.T,
			max_acceleration=self.a_max,
			comfortable_deceleration=self.b_max,
			delta=self.delta
		)


def compute_idm_acceleration(
	v: float,
	gap: float,
	rel_speed: float,
	params: IDMParameters,
) -> float:
	"""
	Compute IDM acceleration for an HV.

	Args:
		v: ego speed (m/s)
		gap: net gap to leader (m), gap <= 0 implies collision risk
		rel_speed: v - v_lead (m/s). Positive means ego is faster than leader.
		params: IDMParameters

	Returns:
		acceleration (m/s²)
	"""
	v = float(max(0.0, v))
	# Free-road term
	free_term = params.a_max * (1.0 - (v / max(1e-6, params.v0)) ** params.delta)

	if np.isinf(gap):
		interaction_term = 0.0
	else:
		# Desired dynamic gap s* (can be < s0 if closing with negative rel_speed)
		denom = 2.0 * np.sqrt(max(1e-12, params.a_max * params.b_max))
		s_star = params.s0 + v * params.T + (v * rel_speed) / denom
		s_star = max(params.s0, s_star)
		if gap <= 0.0:
			interaction_term = params.a_max * ((s_star / max(1e-6, 0.01)) ** 2)
		else:
			interaction_term = params.a_max * ((s_star / gap) ** 2)

	a = free_term - interaction_term
	#添加一个随机扰动
	# a = a + np.random.normal(0, 0.01)
	# Clamp to reasonable physical limits
	return float(np.clip(a, -10.0, 3.0))


def create_heterogeneous_idm_parameters(
	n_vehicles: int,
	base_params: Optional[IDMParameters] = None,
	variation_config: Optional[Dict[str, Dict[str, float]]] = None,
	seed: Optional[int] = None
) -> List[IDMParameters]:
	"""
	为多辆车创建异质的IDM参数（基础版本）。
	
	注意：此函数用于环境初始化的简单异质化需求。
	对于训练数据生成，建议使用 examples/generate_diverse_training_data.py 
	中的 generate_continuous_idm_parameters() 函数，该函数提供更复杂的参数相关性。
	
	Args:
		n_vehicles: 车辆数量
		base_params: 基础参数，如果为None则使用默认参数
		variation_config: 参数变化配置，格式为 {param_name: {"std": value, "min": value, "max": value}}
		seed: 随机种子
		
	Returns:
		包含异质IDM参数的列表
	"""
	if seed is not None:
		np.random.seed(seed)
	
	if base_params is None:
		base_params = IDMParameters()
	
	# 默认变化配置：每个参数的标准差和范围
	default_variation = {
		"desired_speed": {"std": 2.0, "min": 8.0, "max": 20.0},
		"minimum_spacing": {"std": 0.5, "min": 1.0, "max": 4.0},
		"desired_time_headway": {"std": 0.3, "min": 0.8, "max": 2.5},
		"max_acceleration": {"std": 0.2, "min": 0.5, "max": 2.0},
		"comfortable_deceleration": {"std": 0.3, "min": 1.0, "max": 3.0},
		"delta": {"std": 0.5, "min": 2.0, "max": 6.0}
	}
	
	if variation_config is not None:
		# 更新默认配置
		for param, config in variation_config.items():
			if param in default_variation:
				default_variation[param].update(config)
			else:
				default_variation[param] = config
	
	params_list = []
	
	for i in range(n_vehicles):
		# 为每个参数生成随机值
		v0 = np.clip(
			np.random.normal(base_params.v0, default_variation["desired_speed"]["std"]),
			default_variation["desired_speed"]["min"],
			default_variation["desired_speed"]["max"]
		)
		
		s0 = np.clip(
			np.random.normal(base_params.s0, default_variation["minimum_spacing"]["std"]),
			default_variation["minimum_spacing"]["min"],
			default_variation["minimum_spacing"]["max"]
		)
		
		T = np.clip(
			np.random.normal(base_params.T, default_variation["desired_time_headway"]["std"]),
			default_variation["desired_time_headway"]["min"],
			default_variation["desired_time_headway"]["max"]
		)
		
		a_max = np.clip(
			np.random.normal(base_params.a_max, default_variation["max_acceleration"]["std"]),
			default_variation["max_acceleration"]["min"],
			default_variation["max_acceleration"]["max"]
		)
		
		b_max = np.clip(
			np.random.normal(base_params.b_max, default_variation["comfortable_deceleration"]["std"]),
			default_variation["comfortable_deceleration"]["min"],
			default_variation["comfortable_deceleration"]["max"]
		)
		
		delta = np.clip(
			np.random.normal(base_params.delta, default_variation["delta"]["std"]),
			default_variation["delta"]["min"],
			default_variation["delta"]["max"]
		)
		
		params_list.append(IDMParameters(
			desired_speed=v0,
			minimum_spacing=s0,
			desired_time_headway=T,
			max_acceleration=a_max,
			comfortable_deceleration=b_max,
			delta=delta
		))
	
	return params_list


def create_driver_type_parameters(driver_types: List[str]) -> List[IDMParameters]:
	"""
	根据驾驶员类型创建预定义的IDM参数。
	
	Args:
		driver_types: 驾驶员类型列表，可选值: "aggressive", "normal", "conservative", "cautious"
		
	Returns:
		对应的IDM参数列表
	"""
	type_configs = {
		"aggressive": IDMParameters(
			desired_speed=18.0,  # 高期望速度
			minimum_spacing=1.5,  # 小最小距离
			desired_time_headway=1.0,  # 短时间车头时距
			max_acceleration=1.5,  # 高加速度
			comfortable_deceleration=2.0,  # 高减速度
			delta=4.0
		),
		"normal": IDMParameters(
			desired_speed=15.0,
			minimum_spacing=2.0,
			desired_time_headway=1.5,
			max_acceleration=1.0,
			comfortable_deceleration=1.5,
			delta=4.0
		),
		"conservative": IDMParameters(
			desired_speed=12.0,  # 低期望速度
			minimum_spacing=2.5,  # 大最小距离
			desired_time_headway=2.0,  # 长时间车头时距
			max_acceleration=0.8,  # 低加速度
			comfortable_deceleration=1.2,  # 低减速度
			delta=4.0
		),
		"cautious": IDMParameters(
			desired_speed=10.0,  # 很低的期望速度
			minimum_spacing=3.0,  # 很大的最小距离
			desired_time_headway=2.5,  # 很长的时间车头时距
			max_acceleration=0.6,  # 很低的加速度
			comfortable_deceleration=1.0,  # 很低的减速度
			delta=4.0
		)
	}
	
	params_list = []
	for driver_type in driver_types:
		if driver_type in type_configs:
			params_list.append(type_configs[driver_type].copy())
		else:
			# 如果未知类型，使用正常类型
			params_list.append(type_configs["normal"].copy())
	
	return params_list
