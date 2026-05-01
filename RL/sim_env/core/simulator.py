from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

from sim_env.core.vehicle import Vehicle
from sim_env.models.idm import IDMParameters, compute_idm_acceleration


@dataclass
class SimulationConfig:
	dt: float = 0.1
	v_max: Optional[float] = None
	idm_params: IDMParameters = field(default_factory=IDMParameters)
	# 三阶动力学模型参数
	epsilon: float = 0.1  # 时间常数参数
	kappa: float = 1.0    # 增益参数


class SingleLaneSimulator:
	def __init__(self, vehicles: List[Vehicle], config: SimulationConfig | None = None) -> None:
		# 按位置从大到小排序：索引小的更靠前（前车），保证 leader=vehicles[i-1]
		self.vehicles: List[Vehicle] = sorted(vehicles, key=lambda v: v.x_front, reverse=True)
		self.config = config or SimulationConfig()
		self.t: float = 0.0
		self.terminated: bool = False
		self.collision_info: Optional[Tuple[str, str]] = None
		# 外部每步强制加速度覆盖（优先级最高，按车辆ID）
		self._external_overrides: Dict[str, float] = {}
		# 历史记录：每辆车记录 t, x, v, a, gap
		self.history: Dict[str, Dict[str, List[float]]] = {
			v.vehicle_id: {"t": [], "x": [], "v": [], "a": [], "gap": []}
			for v in self.vehicles
		}
		# 记录初始状态（t=0）
		self._record_current_state()

	def set_external_overrides(self, overrides: Dict[str, float]) -> None:
		"""设置当前步外部强制加速度覆盖。

		Args:
			overrides: 车辆ID到加速度的映射。只对提供的车辆生效，不区分CAV/HV。
		"""
		self._external_overrides = dict(overrides)

	def get_leader(self, idx: int) -> Optional[Vehicle]:
		if idx <= 0:
			return None
		return self.vehicles[idx - 1]

	@staticmethod
	def gap_to_leader(ego: Vehicle, leader: Optional[Vehicle]) -> float:
		if leader is None:
			return float(np.inf)
		# gap = leader rear bumper - ego front bumper
		return float(leader.x_front - leader.length - ego.x_front)

	def compute_accelerations(self, cav_actions: Dict[str, float]) -> List[float]:
		"""计算所有车辆的加速度，支持多 CAV 配置。
		
    该方法遍历所有车辆，根据车辆类型(CAV或HV)以及是否有外部控制输入，
    计算每辆车的加速度。CAV可以接受外部加速度输入，否则使用IDM模型；
    HV则始终使用IDM模型计算加速度。
		Args:
			cav_actions: CAV 车辆 ID 到加速度的映射字典。如果为空字典，则所有车辆都使用IDM模型。
    Returns:
        List[float]: 包含所有车辆加速度的列表，顺序与车辆列表一致。
		"""
    # 初始化加速度列表
		accels: List[float] = []
    # 遍历所有车辆
		for i, veh in enumerate(self.vehicles):
        # 获取前导车辆
			leader = self.get_leader(i)
        # 计算与前导车辆的车距
			gap = self.gap_to_leader(veh, leader)
        # 获取前导车辆的速度，如果没有前导车辆则为0
			v_lead = 0.0 if leader is None else leader.speed
        # 计算相对速度
			rel_v = veh.speed - v_lead
			# 外部强制覆盖：最高优先级
			if veh.vehicle_id in self._external_overrides:
            # 如果有外部强制覆盖，使用外部提供的加速度
				a = float(self._external_overrides[veh.vehicle_id])
				accels.append(a)
				continue  # 跳过后续计算
			
			if veh.is_cav:
				# CAV 使用外部提供的加速度，如果没有提供则使用IDM模型
				if veh.vehicle_id in cav_actions:
                # 如果CAV有外部动作输入，使用外部提供的加速度
					u = float(cav_actions[veh.vehicle_id])  # 控制输入
					# 使用三阶动力学模型计算加速度:
					# ai(tk+1) = (1 - T/εi)ai(tk) + (Tκi/εi)ui(tk)
					a = (1 - self.config.dt / self.config.epsilon) * veh.acceleration + \
					    (self.config.dt * self.config.kappa / self.config.epsilon) * u
				else:
					# CAV 没有外部动作时使用 IDM 模型
					# 使用车辆自己的IDM参数，如果没有则使用默认参数
					idm_params = veh.idm_params if veh.idm_params is not None else self.config.idm_params
					idm_accel = compute_idm_acceleration(veh.speed, gap, rel_v, idm_params)
					# 对于IDM，同样使用三阶动力学模型
					# ai(tk+1) = (1 - T/εi)ai(tk) + (Tκi/εi)ui(tk)
					a = (1 - self.config.dt / self.config.epsilon) * veh.acceleration + \
					    (self.config.dt * self.config.kappa / self.config.epsilon) * idm_accel
			else:
				# HV 使用 IDM 模型（优先使用自己的参数）
            # 使用车辆自己的IDM参数，如果没有则使用默认参数
				idm_params = veh.idm_params if veh.idm_params is not None else self.config.idm_params
				idm_accel = compute_idm_acceleration(veh.speed, gap, rel_v, idm_params)
				# 对于HV，同样使用三阶动力学模型
				# ai(tk+1) = (1 - T/εi)ai(tk) + (Tκi/εi)ui(tk)
				a = (1 - self.config.dt / self.config.epsilon) * veh.acceleration + \
				    (self.config.dt * self.config.kappa / self.config.epsilon) * idm_accel
        # 将计算得到的加速度添加到列表中
			accels.append(a)
    # 返回所有车辆的加速度列表
		return accels

	def step(self, cav_actions: Dict[str, float]) -> None:
		"""执行一步仿真，支持多 CAV 配置。
		
		Args:
			cav_actions: CAV 车辆 ID 到加速度的映射字典
		"""
		if self.terminated:
			return
		# Compute accelerations based on current state
		accels = self.compute_accelerations(cav_actions)
		# Update kinematics simultaneously
		for veh, a in zip(self.vehicles, accels):
			veh.update_kinematics(a, self.config.dt, v_min=0.0, v_max=self.config.v_max)
		
		# 先检测碰撞（基于更新后但未重新排序的车辆状态）
		self._check_collisions()
		
		self.t += self.config.dt
		# 本步结束后清除外部覆盖，仅对当前步生效
		self._external_overrides = {}
		# 记录当前状态（确保终止也有该时刻数据）
		self._record_current_state()
		
		# 最后重新排序（前车在前），为下一步计算做准备
		self.vehicles.sort(key=lambda v: v.x_front, reverse=True)

	def _check_collisions(self) -> None:
		for i in range(1, len(self.vehicles)):
			leader = self.vehicles[i - 1]
			follower = self.vehicles[i]
			gap = self.gap_to_leader(follower, leader)
			if gap <= 0.0:
				self.terminated = True
				self.collision_info = (follower.vehicle_id, leader.vehicle_id)
				break

	def _record_current_state(self) -> None:
		# 根据当前排序计算每辆车与前车净距，并记录时间序列
		# 净距列表与车辆顺序对齐
		gaps: List[float] = []
		for i, veh in enumerate(self.vehicles):
			leader = self.get_leader(i)
			gaps.append(self.gap_to_leader(veh, leader))
		for veh, gap in zip(self.vehicles, gaps):
			rec = self.history[veh.vehicle_id]
			rec["t"].append(float(self.t))
			rec["x"].append(float(veh.x_front))
			rec["v"].append(float(veh.speed))
			rec["a"].append(float(veh.acceleration))
			rec["gap"].append(float(gap if np.isfinite(gap) else np.inf))

	def get_state(self) -> List[Dict[str, float | str]]:
		return [
			{
				"id": v.vehicle_id,
				"x": float(v.x_front),
				"v": float(v.speed),
				"a": float(v.acceleration),
				"type": v.type_label,
			}
			for v in self.vehicles
		]

	def reset_time(self) -> None:
		self.t = 0.0
		self.terminated = False
		self.collision_info = None
