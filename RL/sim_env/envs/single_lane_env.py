"""
单车道车辆跟驰环境（Gym/Gymnasium 兼容）。

- 车辆类型：HV 使用 IDM 模型，CAV 由外部动作加速度控制（连续 [-3, 3] m/s^2）。
- 观测空间：单 CAV 为 [v, gap_lead, rel_v_lead, gap_foll, rel_v_foll]；多 CAV 为 K×5。
- 重置支持：随机初始化或通过 reset(options={"manual_vehicles": [...]}) 手动指定车辆列表。
- 多 CAV：通过 cav_indices 指定多个 CAV；动作为长度 K 的连续向量。
- 碰撞规则：相邻车辆净距 gap<=0 立即终止。
"""
from __future__ import annotations

from typing import Dict, Tuple, Any, List, Optional
import numpy as np

try:
	import gymnasium as gym
	supports_gymnasium = True
except Exception:  # pragma: no cover
	try:
		import gym  # type: ignore
		supports_gymnasium = False
	except Exception:  # pragma: no cover
		gym = None  # type: ignore
		supports_gymnasium = False

from sim_env.core.vehicle import Vehicle
from sim_env.core.simulator import SingleLaneSimulator, SimulationConfig
from sim_env.models.idm import IDMParameters, create_heterogeneous_idm_parameters, create_driver_type_parameters
import os


class SingleLaneFollowingEnv(gym.Env):  # type: ignore
	metadata = {"render_modes": ["ansi"], "render_fps": 10}

	def __init__(
		self,
		seed: Optional[int] = None,                    # 随机种子，用于确保仿真结果可重现
		num_vehicles: int = 5,                        # 车辆总数（必须>=2）
		cav_index: int = 1,                           # 单CAV模式下的CAV索引（向后兼容）
		cav_indices: Optional[List[int]] = None,      # 多CAV模式下的CAV索引列表
		idm_cfg: Optional[dict] = None,               # IDM配置参数（预留扩展）
		dt: float = 0.1,                              # 仿真时间步长（秒）
		v_max: Optional[float] = None,                # 最大速度限制（None表示无限制）
		v_target: float = 15.0,                       # CAV目标速度，用于奖励计算
		jerk_coeff: float = 0.1,                      # 加速度变化惩罚系数（抑制急加急减）
		collision_penalty: float = 100.0,             # 碰撞惩罚值
		reward_aggregate: str = "mean",               # 多CAV奖励聚合方式（"mean"或"sum"）
		heterogeneous_idm: bool = False,              # 是否启用异质IDM参数
		idm_variation_config: Optional[dict] = None,  # IDM参数变化配置
		driver_types: Optional[List[str]] = None,     # 预定义驾驶员类型列表
		# 三阶动力学模型参数
		epsilon: float = 0.1,                         # 时间常数参数
		kappa: float = 1.0,                           # 增益参数
	):
		"""创建环境实例。

		Args:
			seed: 随机种子。
			num_vehicles: 车辆总数（>=2）。
			cav_index: 向后兼容的单 CAV 下标（若提供 cav_indices 则忽略）。
			cav_indices: 多 CAV 下标列表（0 基）。
			idm_cfg: 预留参数（如需自定义 IDM，可扩展）。
			dt: 仿真步长（秒）。
			v_max: 速度上限（None 表示无限制）。
			v_target: 奖励中希望的目标速度。
			jerk_coeff: 奖励中加速度惩罚系数（越大越惩罚频繁加速/减速）。
			collision_penalty: 碰撞惩罚绝对值。
			reward_aggregate: 多 CAV 奖励聚合方式（"mean" 或 "sum"）。
			heterogeneous_idm: 是否启用异质IDM参数。
			idm_variation_config: IDM参数变化配置，用于随机生成参数。
			driver_types: 驾驶员类型列表，用于生成预定义的IDM参数。
			epsilon: 三阶动力学模型的时间常数参数
			kappa: 三阶动力学模型的增益参数
		"""
		# 参数有效性检查
		assert num_vehicles >= 2, "至少需要2辆车"
		if cav_indices is not None:
			assert len(cav_indices) >= 1, "至少需要1辆 CAV"
			for idx in cav_indices:
				assert 0 <= idx < num_vehicles, "CAV 下标越界"
		else:
			assert 0 <= cav_index < num_vehicles, "CAV 下标越界"

		# 初始化基本参数
		self.rng = np.random.default_rng(seed)  # 随机数生成器
		self.num_vehicles = num_vehicles
		self.cav_index = cav_index
		self.cav_indices: List[int] = list(cav_indices) if cav_indices is not None else [cav_index]
		self.dt = dt  # 仿真时间步长
		self.v_max = v_max  # 速度上限
		self.idm_cfg = idm_cfg or {}  # IDM配置参数

		# 奖励相关配置
		self.v_target = float(v_target)                # CAV目标速度
		self.jerk_coeff = float(jerk_coeff)             # 加速度变化惩罚系数
		self.collision_penalty = float(collision_penalty) # 碰撞惩罚值
		self.reward_aggregate = str(reward_aggregate)   # 多CAV奖励聚合方式
		
		# IDM异质性配置
		self.heterogeneous_idm = bool(heterogeneous_idm)  # 是否启用异质IDM
		self.idm_variation_config = idm_variation_config  # IDM参数变化配置
		self.driver_types = driver_types                  # 驾驶员类型列表

		# 仿真相关对象
		self.sim: Optional[SingleLaneSimulator] = None    # 仿真器实例
		self.cav_ids: List[str] = []                      # CAV车辆ID列表
			
		# 领车速度数据缓存
		self._cached_lead_speed_data: Optional[np.ndarray] = None  # 缓存的领车速度数据

		# 初始化动作和观测空间（在reset时会根据实际CAV数重新调整）
		k = max(1, len(self.cav_indices))  # CAV数量
		# 动作空间：每个CAV的加速度在[-3, 3] m/s²范围内
		low = np.full((k,), -3.0, dtype=np.float32)
		high = np.full((k,), 3.0, dtype=np.float32)
		self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
		
		# 观测空间：每个CAV观测5个维度 [v, gap_lead, rel_v_lead, gap_foll, rel_v_foll]
		obs_low = np.tile(np.array([0.0, 0.0, -50.0, 0.0, -50.0], dtype=np.float32), (k, 1))
		obs_high = np.tile(np.array([50.0, 1e4, 50.0, 1e4, 50.0], dtype=np.float32), (k, 1))
		self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)
		
		# 保存三阶动力学模型参数
		self.epsilon = epsilon
		self.kappa = kappa
		
	def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):  # type: ignore
		"""重置环境。

		- 若 options["manual_vehicles"] 提供车辆列表，则按手动配置初始化；
		- 否则按给定参数随机初始化。
		- 多 CAV：可通过 options["cav_indices"] 指定；若手动列表含多辆 is_cav=True，也会自动识别。

		Gymnasium 返回 (obs, info)，Gym 返回 obs。
		"""
		if seed is not None:
			self.rng = np.random.default_rng(seed)
		vehicles = self._init_from_options(options)
		# 传递三阶动力学模型参数给SimulationConfig
		cfg = SimulationConfig(dt=self.dt, v_max=self.v_max, epsilon=self.epsilon, kappa=self.kappa)
		self.sim = SingleLaneSimulator(vehicles, cfg)
		# 处理领车序列（加速度或速度）
		opts = options or {}
		# 支持加速度序列（旧）和速度序列（新）
		lead_acc_seq = opts.get("lead_acc_sequence", None)  # 领车加速度序列
		lead_speed_seq = opts.get("lead_speed_sequence", None)  # 领车速度序列
		
		# 如果没有提供任何领车序列，尝试从数据文件中随机选择一个
		if lead_acc_seq is None and lead_speed_seq is None:
			random_speed_seq = self._get_random_lead_speed_sequence()
			if random_speed_seq is not None:
				lead_speed_seq = random_speed_seq
		
		# 初始化领车控制参数
		self._lead_acc_sequence = None
		self._lead_speed_sequence = None
		self._lead_acc_index = 0
		self._lead_speed_index = 0
		self._last_lead_speed = None  # 用于速度控制时计算加速度
		
		if lead_acc_seq is not None:
			# 使用加速度序列（传统方式）
			arr = np.asarray(lead_acc_seq, dtype=float).reshape(-1)
			self._lead_acc_sequence = arr
			self._lead_acc_index = 0
		elif lead_speed_seq is not None:
			# 使用速度序列（新方式）
			arr = np.asarray(lead_speed_seq, dtype=float).reshape(-1)
			self._lead_speed_sequence = arr
			self._lead_speed_index = 0
			# 记录领车初始速度以便计算加速度
			if len(self.sim.vehicles) > 0:
				self._last_lead_speed = self.sim.vehicles[0].speed
			
		# 更新CAV ID列表
		self.cav_ids = [vehicles[i].vehicle_id for i in self.cav_indices]
		
		# 根据实际CAV数量更新动作和观测空间
		self._update_spaces(len(self.cav_indices))
		
		# 获取初始观测和状态信息
		obs = self._build_observation()
		info = {"state": self.sim.get_state()}
		
		# 根据Gymnasium/Gym兼容性返回结果
		return obs, info if supports_gymnasium else obs

	def step(self, action):  # type: ignore
		"""执行一步仿真。

		执行流程：
		1. 处理CAV动作输入（或IDM模式）
		2. 处理领车加速度序列（如果有）
		3. 调用仿真器执行一步
		4. 计算奖励和终止条件
		5. 返回观测、奖励、终止标志和信息

		Args:
			action: CAV加速度数组或空字典：
				- 数组：长度K的CAV加速度，范围被裁剪到[-3, 3] m/s²
				- 空字典{}：所有车辆（包括CAV）都使用IDM模型
				
		Returns:
			Gymnasium: (obs, reward, terminated, truncated, info)
			Gym: (obs, reward, terminated, info)
		"""
		assert self.sim is not None, "环境未重置，请先调用reset()"
		
		# 步骤1：处理CAV动作输入
		if isinstance(action, dict) and len(action) == 0:
			# 空字典表示所有车辆都使用IDM模型
			actions = {}
		elif isinstance(action, (list, tuple, np.ndarray)):
			# 处理混合动作输入：[None, 1.0] 表示第一辆CAV使用IDM，第二辆CAV使用加速度1.0
			assert len(action) == len(self.cav_ids), "action长度与CAV数量不一致"
			actions = {}
			for i, a in enumerate(action):
				if a is not None:
					# 裁剪加速度到安全范围
					clipped_a = float(np.clip(a, -3.0, 3.0))
					actions[self.cav_ids[i]] = clipped_a
				# 如果a是None，则该CAV使用IDM模型（不添加到actions字典中）
		else:
			# 处理CAV加速度输入
			assert len(self.cav_ids) >= 1, "CAV数量不能为0"
			a_arr = np.asarray(action, dtype=np.float32).reshape(-1)
			# 如果是单个值但有多CAV，则复制到所有CAV
			if a_arr.size == 1 and len(self.cav_ids) > 1:
				a_arr = np.repeat(a_arr, len(self.cav_ids))
			assert a_arr.size == len(self.cav_ids), "action长度与CAV数量不一致"
			# 裁剪加速度到安全范围
			a_arr = np.clip(a_arr, -3.0, 3.0)
			# 构建动作字典：CAV ID -> 加速度
			actions = {vid: float(a) for vid, a in zip(self.cav_ids, a_arr.tolist())}
		
		# 步骤2：处理领车序列控制（加速度或速度）
		# 领车是位置最靠前的车辆，即当前排序后的索引0
		lead_override: Dict[str, float] = {}
		
		# 处理加速度序列控制
		if getattr(self, "_lead_acc_sequence", None) is not None and not self.sim.terminated:
			seq = self._lead_acc_sequence  # type: ignore[attr-defined]
			idx = int(getattr(self, "_lead_acc_index", 0))
			if idx < len(seq):
				# 获取领车并设置其加速度
				lead_vehicle = self.sim.vehicles[0]
				lead_override[lead_vehicle.vehicle_id] = float(np.clip(seq[idx], -3.0, 3.0))
				self._lead_acc_index = idx + 1  # 更新序列索引
				
		# 处理速度序列控制（新功能）
		elif getattr(self, "_lead_speed_sequence", None) is not None and not self.sim.terminated:
			seq = self._lead_speed_sequence  # type: ignore[attr-defined]
			idx = int(getattr(self, "_lead_speed_index", 0))
			if idx < len(seq):
				# 获取领车和目标速度
				lead_vehicle = self.sim.vehicles[0]
				target_speed = float(np.clip(seq[idx], 0.0, 50.0))  # 限制速度范围
				
				# 根据当前速度和目标速度计算需要的加速度
				current_speed = lead_vehicle.speed
				# 使用改进的PD控制器来计算加速度
				speed_error = target_speed - current_speed
				# 比例控制：加速度与速度误差成正比
				desired_acceleration = 2.0 * speed_error  # 可调整的增益系数
				# 限制加速度在合理范围内
				desired_acceleration = float(np.clip(desired_acceleration, -3.0, 3.0))
				
				lead_override[lead_vehicle.vehicle_id] = desired_acceleration
				self._lead_speed_index = idx + 1  # 更新序列索引
				
		# 步骤3：设置外部覆盖并执行一步仿真
		if hasattr(self.sim, "set_external_overrides"):
			self.sim.set_external_overrides(lead_override)



		
		self.sim.step(actions)  # 执行一步仿真
		
		# 步骤4：计算观测、奖励和终止条件
		obs = self._build_observation()      # 构建观测
		reward = self._reward_multi()        # 计算奖励
		terminated = bool(self.sim.terminated) # 检查是否终止（如碰撞）
		truncated = False                    # Gymnasium的truncated标志
		
		# 步骤5：构建信息字典
		info = {
			"state": self.sim.get_state(),      # 仿真器状态
			"collision": self.sim.collision_info # 碰撞信息
		}
		
		# 步骤6：根据Gymnasium/Gym兼容性返回结果
		if supports_gymnasium:
			return obs, reward, terminated, truncated, info
		else:
			return obs, reward, terminated, info

	def render(self):  # type: ignore
		"""渲染环境状态为文本格式。
		
		返回包含以下信息的字符串：
		- 当前仿真时间和终止状态
		- 碰撞信息（如果发生）
		- 每辆车的详细状态（ID、类型、位置、速度、加速度）
		
		Returns:
			格式化的状态字符串
		"""
		assert self.sim is not None, "环境未重置，请先调用reset()"
		
		# 构建状态信息列表
		lines = [
			f"t={self.sim.t:.1f}s" + (" (TERMINATED)" if self.sim.terminated else "")
		]
		
		# 添加碰撞信息（如果发生）
		if self.sim.collision_info is not None:
			follower, leader = self.sim.collision_info
			lines.append(f"Collision: {follower} -> {leader}")
			
		# 添加每辆车的状态信息
		for s in self.sim.get_state():
			lines.append(
				f"{s['id']:>4} {s['type']}  x={s['x']:7.2f}  v={s['v']:5.2f}  a={s['a']:5.2f}"
			)
			
		return "\n".join(lines)

	def plot_timeseries(self) -> None:
		"""绘制每辆车的时间序列曲线。
		
		绘制四个子图：
		1. 加速度 (m/s²) 随时间变化
		2. 速度 (m/s) 随时间变化
		3. 时间车头时距 (s) 随时间变化
		4. 位置 (m) 随时间变化
		
		CAV和HV使用不同颜色区分：
		- CAV: 蓝色
		- HV: 红色
		
		Note:
			需要在仿真结束后调用，以获取完整的历史数据。
		"""
		import matplotlib.pyplot as plt
		assert self.sim is not None, "环境未重置或仿真未运行"
		
		H = self.sim.history  # 获取仿真历史数据
		
		# 创建四个子图：加速度、速度、时间车头时距、位置
		fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
		ax_a, ax_v, ax_th, ax_x = axes
		
		# 定义车辆类型颜色：CAV用蓝色，HV用红色
		cav_color = 'blue'
		hv_color = 'red'
		
		# 遍历每辆车的历史数据
		for vid, rec in H.items():
			# 获取车辆类型（CAV或HV）
			veh = next((v for v in self.sim.vehicles if v.vehicle_id == vid), None)
			is_cav = veh.is_cav if veh else False
			color = cav_color if is_cav else hv_color
			line_style = '-'  # 所有车辆都使用实线
			vehicle_label = f"{vid} ({'CAV' if is_cav else 'HV'})"  # 为每辆车创建标签
			
			# 计算时间车头时距 = gap / max(v, 1e-3)
			v_arr = np.asarray(rec["v"], dtype=float)
			g_arr = np.asarray(rec["gap"], dtype=float)
			th_arr = g_arr / np.maximum(v_arr, 1e-3)  # 避免除零
			
			# 绘制四条曲线，使用不同颜色区分CAV和HV，并添加标签
			ax_a.plot(rec["t"], rec["a"], color=color, linestyle=line_style, label=vehicle_label)
			ax_v.plot(rec["t"], rec["v"], color=color, linestyle=line_style, label=vehicle_label)
			ax_th.plot(rec["t"], th_arr, color=color, linestyle=line_style, label=vehicle_label)
			ax_x.plot(rec["t"], rec["x"], color=color, linestyle=line_style, label=vehicle_label)
		
		# 设置各子图的标签
		ax_a.set_ylabel("acc (m/s^2)")       # 加速度
		ax_v.set_ylabel("vel (m/s)")        # 速度
		ax_th.set_ylabel("time_headway (s)") # 时间车头时距
		ax_x.set_ylabel("x (m)")            # 位置
		ax_x.set_xlabel("t (s)")            # 时间轴
		
		# 为每个子图添加网格和图例（仅第一个子图显示图例以避免重复）
		for i, ax in enumerate(axes):
			ax.grid(True, linestyle=":", alpha=0.6)  # 添加网格
			if i == 0:  # 只在第一个子图显示图例
				ax.legend(loc="best", fontsize=8)
				
		# 调整布局并显示
		plt.savefig("timeseries.png")
		plt.close()

	def _load_lead_speed_data(self) -> Optional[np.ndarray]:
		"""
		加载领车速度序列数据
		参考 generate_diverse_training_data.py 的实现（第297-306行）
		
		Returns:
		  成功加载返回速度数据数组，失败返回None
		"""
		if self._cached_lead_speed_data is not None:
			return self._cached_lead_speed_data
			
		try:
			# 尝试加载数据文件
			data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "scaled_data.npy")
			data = np.load(data_path, allow_pickle=True)
			
			# 处理对象数组格式的数据
			if data.dtype == 'object' and len(data) > 0:
				# 取所有可用的序列，每个都进行数据转换
				speed_sequences = []
				for i in range(len(data)):  # 最多取100个序列
					if isinstance(data[i], np.ndarray):
						# 应用数据转换：(data + 1) * 8 + 5
						speed_seq = (data[i] + 1) * 8 + 5
						# 限制在合理范围内
						speed_seq = np.clip(speed_seq, 5, 25)
						# 截取前600个数据点（避免序列过长）
						if len(speed_seq) > 600:
							speed_seq = speed_seq[:600]
						speed_sequences.append(speed_seq)
						
				if speed_sequences:
					self._cached_lead_speed_data = np.array(speed_sequences, dtype=object)
					return self._cached_lead_speed_data
			else:
				# 普通数组格式
				speed_sequences = (data + 1) * 8 + 5
				speed_sequences = np.clip(speed_sequences, 5, 25)
				self._cached_lead_speed_data = speed_sequences
				return self._cached_lead_speed_data
				
		except Exception as e:
			# 数据加载失败，返回None
			return None
	
	def _get_random_lead_speed_sequence(self) -> Optional[np.ndarray]:
		"""
		从加载的数据中随机选择一个领车速度序列
		
		Returns:
		  随机选择的速度序列，没有数据时返回None
		"""
		speed_data = self._load_lead_speed_data()
		if speed_data is None:
			return None
			
		# 如果是对象数组（多个序列）
		if speed_data.dtype == 'object' and len(speed_data) > 0:
			# 随机选择一个序列
			selected_idx = self.rng.integers(0, len(speed_data))
			return speed_data[selected_idx]
		else:
			# 单个序列，直接返回
			return speed_data
	def _update_spaces(self, k: int) -> None:
		"""根据CAV数量更新动作和观测空间。
		
		Args:
			k: CAV数量
		"""
		# 动作空间：K维，每个CAV的加速度在[-3, 3] m/s²范围内
		low = np.full((k,), -3.0, dtype=np.float32)
		high = np.full((k,), 3.0, dtype=np.float32)
		self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
		
		# 观测空间：K×5，每个CAV观测5个维度
		# [v, gap_lead, rel_v_lead, gap_foll, rel_v_foll]
		obs_low = np.tile(np.array([0.0, 0.0, -50.0, 0.0, -50.0], dtype=np.float32), (k, 1))
		obs_high = np.tile(np.array([50.0, 1e4, 50.0, 1e4, 50.0], dtype=np.float32), (k, 1))
		self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)

	def _init_from_options(self, options: Optional[dict]) -> List[Vehicle]:
		"""根据选项初始化车辆列表。
		
		支持两种初始化方式：
		1. 手动指定：通过options["manual_vehicles"]提供车辆列表
		2. 随机生成：根据参数随机生成车辆
		
		Args:
			options: 初始化选项
			
		Returns:
			按位置排序的车辆列表（领车在前）
		"""
		opts = options or {}
		
		# 优先读取显式cav_indices
		explicit_cav_indices = opts.get("cav_indices", None)
		if explicit_cav_indices is not None:
			self.cav_indices = [int(i) for i in explicit_cav_indices]

		if "manual_vehicles" in opts and opts["manual_vehicles"]:
			# 手动指定车辆模式
			vehicles: List[Vehicle] = []
			manual_list = opts["manual_vehicles"]
			
			# 遍历手动指定的车辆配置
			for i, spec in enumerate(manual_list):
				# 获取车辆基本信息
				vid = str(spec.get("vehicle_id", f"V{i}"))
				is_cav = bool(spec.get("is_cav", False))
				length = float(spec.get("length", 5.0))
				
				# 处理车辆位置：如果未指定，则按领车在前的方式生成
				if "x_front" not in spec:
					x_front = (len(manual_list) - 1 - i) * (15.0 + length) + 50.0
				else:
					x_front = float(spec.get("x_front"))
					
				v = float(spec.get("speed", 0.0))  # 初始速度
				
				# 处理IDM参数（如果指定）
				idm_params = None
				if "idm_params" in spec:
					idm_spec = spec["idm_params"]
					idm_params = IDMParameters(**idm_spec) if isinstance(idm_spec, dict) else idm_spec
					
				# 创建车辆对象
				vehicles.append(Vehicle(
					vehicle_id=vid, is_cav=is_cav, length=length, 
					x_front=x_front, speed=v, idm_params=idm_params
				))
				
			# 如果未显式指定CAV索引，则从手动车辆中自动推断
			if explicit_cav_indices is None:
				cav_idx_auto = [i for i, v in enumerate(vehicles) if v.is_cav]
				self.cav_indices = cav_idx_auto if cav_idx_auto else [self.cav_index]
				
			# 按位置排序（领车在前）
			return sorted(vehicles, key=lambda v: v.x_front, reverse=True)
		else:
			# 随机生成模式
			return self._random_init(opts)

	def _random_init(self, options: Optional[dict]) -> List[Vehicle]:
		"""随机初始化车辆列表。
		
		根据环境参数和选项随机生成车辆，支持异质IDM参数。
		
		Args:
			options: 初始化选项，包括车辆数量、间距等
			
		Returns:
			随机生成的车辆列表
		"""
		opts = options or {}
		n = int(opts.get("num_vehicles", self.num_vehicles))  # 车辆数量
		
		# 处理CAV索引：cav_indices优先于cav_index
		if "cav_indices" in opts:
			self.cav_indices = [int(i) for i in opts["cav_indices"]]
			assert len(self.cav_indices) >= 1, "CAV数量不能为0"
		else:
			# 如果没有在options中提供cav_indices，使用初始化时的cav_indices
			if not self.cav_indices:
				self.cav_index = int(opts.get("cav_index", self.cav_index))
				self.cav_indices = [self.cav_index]
				
		# 获取初始化参数
		base_gap = float(opts.get("base_gap", 15.0))    # 基础车间距
		v0 = float(opts.get("v0", 15.0))              # 初始速度平均值
		length = float(opts.get("length", 5.0))       # 车辆长度

		vehicles: List[Vehicle] = []
		
		# 生成异质IDM参数（如果启用）
		idm_params_list = None
		if self.heterogeneous_idm:
			if self.driver_types is not None:
				# 使用指定的驾驶员类型
				if len(self.driver_types) != n:
					# 如果类型数量不匹配，重复或截断
					extended_types = (self.driver_types * ((n // len(self.driver_types)) + 1))[:n]
					idm_params_list = create_driver_type_parameters(extended_types)
				else:
					idm_params_list = create_driver_type_parameters(self.driver_types)
			else:
				# 随机生成异质参数
				base_idm = IDMParameters()
				idm_params_list = create_heterogeneous_idm_parameters(
					n_vehicles=n,
					base_params=base_idm,
					variation_config=self.idm_variation_config,
					seed=opts.get("idm_seed", None)
				)
		
		# 创建车辆对象
		for i in range(n):
			is_cav = i in self.cav_indices    # 判断是否为CAV
			vid = f"V{i}"                    # 车辆ID
			
			# 计算车辆位置：领车在前，x坐标递减
			# 索引0的车辆x最大，索引1次之，以此类推
			x_front = (n - 1 - i) * (base_gap + length) + 50.0
			
			# 生成初始速度（正态分布 + 下限限制）
			v = max(0.0, self.rng.normal(v0, 1.0))
			
			# 分配IDM参数（如果有）
			idm_params = idm_params_list[i] if idm_params_list is not None else None
			
			# 创建车辆对象并添加到列表
			vehicles.append(
				Vehicle(
					vehicle_id=vid, is_cav=is_cav, length=length, 
					x_front=x_front, speed=v, idm_params=idm_params
				)
			)
			
		return vehicles

	def _build_observation(self) -> np.ndarray:
		"""构建 CAV 的观测数组。
		
		为每个 CAV 构建 5 维观测向量：
		[v, gap_lead, rel_v_lead, gap_foll, rel_v_foll]
		
		观测维度说明：
		- v: CAV的当前速度 (m/s)
		- gap_lead: 与前车的净间距 (m)
		- rel_v_lead: 与前车的相对速度 (m/s)
		- gap_foll: 与后车的净间距 (m)
		- rel_v_foll: 后车相对于 CAV 的速度 (m/s)
		
		Returns:
			形状为 (K, 5) 的 numpy 数组，K 为 CAV 数量
		"""
		assert self.sim is not None, "仿真器未初始化"
		assert len(self.cav_ids) >= 1, "CAV 数量不能为 0"
		
		# 对每个 CAV 构造 5 维观测并堆叠为 K×5 数组
		obs_list: List[np.ndarray] = []
		
		for cav_id in self.cav_ids:
			# 找到 CAV 在车辆列表中的索引
			idx = next(i for i, v in enumerate(self.sim.vehicles) if v.vehicle_id == cav_id)
			veh = self.sim.vehicles[idx]  # 当前 CAV
			
			# 获取前车和后车（如果存在）
			lead = self.sim.vehicles[idx - 1] if idx > 0 else None
			foll = self.sim.vehicles[idx + 1] if idx + 1 < len(self.sim.vehicles) else None

			# 计算与前车的间距和相对速度
			gap_lead = np.inf if lead is None else (lead.x_front - lead.length - veh.x_front)
			rel_v_lead = 0.0 if lead is None else (veh.speed - lead.speed)
			
			# 计算与后车的间距和相对速度
			gap_foll = np.inf if foll is None else (veh.x_front - veh.length - foll.x_front)
			rel_v_foll = 0.0 if foll is None else (foll.speed - veh.speed)

			# 构建 5 维观测向量
			obs = np.array([
				veh.speed,  # CAV 当前速度
				float(gap_lead if np.isfinite(gap_lead) else 1e6),   # 前车间距
				rel_v_lead,  # 前车相对速度
				float(gap_foll if np.isfinite(gap_foll) else 1e6),   # 后车间距
				rel_v_foll,  # 后车相对速度
			], dtype=np.float32)
			
			obs_list.append(obs)
			
		# 将所有 CAV 的观测堆叠成 (K, 5) 形状的数组
		return np.stack(obs_list, axis=0)

	def _reward_multi(self) -> float:
		"""计算多 CAV 的聚合奖励。
		
		奖励组成：
		1. 速度跟踪奖励：-|v - v_target|，鼓励 CAV 保持目标速度
		2. 加速度惩罚：-jerk_coeff * |a|，惩罚频繁加速/减速
		3. 碰撞惩罚：-collision_penalty，碰撞时的大额惩罚
		
		多 CAV 聚合方式：
		- "mean": 取平均值
		- "sum": 取总和
		
		Returns:
			聚合后的奖励值
		"""
		assert self.sim is not None, "仿真器未初始化"
		
		# 对每辆 CAV 计算单体奖励
		rewards = []
		for cav_id in self.cav_ids:
			# 找到 CAV 在车辆列表中的索引
			idx = next(i for i, v in enumerate(self.sim.vehicles) if v.vehicle_id == cav_id)
			veh = self.sim.vehicles[idx]
			
			# 计算奖励组成部分
			penalty_collision = -self.collision_penalty if self.sim.terminated else 0.0  # 碰撞惩罚
			penalty_jerk = -self.jerk_coeff * abs(veh.acceleration)                     # 加速度惩罚
			track = -abs(veh.speed - self.v_target)                                   # 速度跟踪奖励
			
			# 总奖励 = 跟踪奖励 + 加速度惩罚 + 碰撞惩罚
			rewards.append(track + penalty_jerk + penalty_collision)
			
		# 按指定方式聚合奖励
		if self.reward_aggregate == "sum":
			return float(np.sum(rewards))    # 总和
		return float(np.mean(rewards))       # 平均值
