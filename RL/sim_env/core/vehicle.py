from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Vehicle:
	vehicle_id: str
	is_cav: bool
	length: float = 5.0
	x_front: float = 0.0
	speed: float = 0.0
	acceleration: float = 0.0
	idm_params: Optional['IDMParameters'] = None  # 每个车辆的个性化IDM参数

	def update_kinematics(self, acceleration: float, dt: float, v_min: float = 0.0, v_max: float | None = None) -> None:
		# 保存当前速度用于计算真实加速度
		v_old = self.speed
		a_old = self.acceleration  # 保存旧加速度
		
		# 计算理论新速度
		v_new = v_old + a_old * dt
		
		# 应用速度限制
		if v_max is None:
			v_new = max(v_min, v_new)
		else:
			v_new = float(np.clip(v_new, v_min, v_max))
		
		# 计算acc(t)真实加速度（考虑速度限制后的实际加速度）
		real_acceleration = (v_new - v_old) / dt if dt > 0 else 0.0
		
		# 使用旧速度和加速度更新位置
		# x = x0 + v0*t + 0.5*a0*t^2
		self.x_front = self.x_front + v_old * dt + 0.5 * real_acceleration * dt * dt
		
		# 更新车辆状态
		self.acceleration = acceleration  # 存储acc(t+1)
		self.speed = v_new

	@property
	def type_label(self) -> str:
		return "CAV" if self.is_cav else "HV"