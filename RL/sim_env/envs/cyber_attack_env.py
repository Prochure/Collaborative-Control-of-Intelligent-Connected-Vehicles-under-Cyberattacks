#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带网络攻击功能的单车道仿真环境

扩展原始 SingleLaneFollowingEnv，添加网络攻击能力。
支持对CAV的速度、位置、加速度数据进行攻击，可配置攻击频率、均值、方差等参数。
"""

import os
import sys
import numpy as np
from typing import Dict, Tuple, Any, List, Optional

# 添加项目路径
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sim_env.envs.single_lane_env import SingleLaneFollowingEnv
from sim_env.models.idm import compute_idm_acceleration, IDMParameters


class CyberAttackEnv(SingleLaneFollowingEnv):
    """
    带网络攻击功能的仿真环境
    
    继承自 SingleLaneFollowingEnv，添加了对CAV数据的网络攻击能力。
    支持多种攻击类型：数据篡改、丢包攻击、延迟攻击等。
    """
    
    def __init__(
        self,
        # 网络攻击相关参数
        enable_cyber_attack: bool = True,            # 是否启用网络攻击
        attack_frequency: float = 0.1,                # 攻击发生概率（每步）
        attack_mean: float = 0.0,                     # 攻击偏差均值（所有目标通用）
        attack_means: Optional[Dict[str, float]] = None, # 各目标独立均值
        attack_variance: Optional[float] = None,      # 单一方差（向后兼容）
        attack_variances: Optional[Dict[str, float]] = None,  # 各目标独立方差
        attack_targets: Optional[List[str]] = None,   # 攻击目标（"speed", "position", "acceleration"）
        attack_start_time: float = 5.0,               # 攻击开始时间（秒），前N秒免疫攻击
        attack_type: str = "data_tampering",          # 攻击类型："data_tampering", "packet_drop", "delay"
        attack_distribution: str = "normal",          # 攻击分布："normal", "uniform"
        delay_steps: int = 3,                         # 延迟步数（仅在attack_type="delay"时有效）
        # 三阶动力学模型参数
        epsilon: float = 0.2,                         # 时间常数参数
        kappa: float = 0.7,                           # 增益参数
        # 低通滤波器参数
        use_action_filter: bool = True,               # 是否使用动作低通滤波
        filter_alpha: float = 1,                    # 低通滤波器系数（0-1，越小越平滑）
        # CBF参数
        use_cbf: bool = False,                        # 是否使用CBF控制
        cbf_params: Optional[Dict[str, float]] = None, # CBF参数配置
        force_lead_cav_p_one: bool = False,           # 是否强制将lead_cav_p设为1
        **kwargs
    ):
        """
        初始化带网络攻击功能的环境
        
        Args:
            enable_cyber_attack: 是否启用网络攻击功能
            attack_frequency: 每个仿真步骤中攻击发生的概率（0.0-1.0）
            attack_mean: 攻击偏差的均值（通常为0.0，所有目标通用）
            attack_means: 各目标独立均值字典，例如：{"speed": 1.0, "position": 0.0}
            attack_variance: 单一方差值（向后兼容，如果设置则所有目标使用相同方差）
            attack_variances: 各目标独立方差字典，例如：
                             {"speed": 1.0, "position": 0.5, "acceleration": 2.0}
            attack_targets: 攻击目标列表，可选值: ["speed", "position", "acceleration"]
            attack_start_time: 攻击开始时间（秒），仿真开始后前N秒内不会发生攻击
            attack_type: 攻击类型，可选值: "data_tampering"（数据篡改）, "packet_drop"（丢包）, "delay"（延迟）
            delay_steps: 延迟步数（仅在attack_type="delay"时有效）
            epsilon: 三阶动力学模型的时间常数参数
            kappa: 三阶动力学模型的增益参数
            use_action_filter: 是否使用动作低通滤波器
            filter_alpha: 低通滤波器系数（0-1，越小越平滑）
            use_cbf: 是否使用CBF控制
            cbf_params: CBF参数配置字典，包含:
                       - tau_i: 时间常数参数 (默认: 1.2)
                       - tau_star: 时间常数参数 (默认: 2.5)
                       - rho1: CBF参数 (默认: 0.8)
                       - rho2: CBF参数 (默认: 0.8)
                       - u_min: 最小加速度限制 (默认: -3.0)
                       - u_max: 最大加速度限制 (默认: 2.0)
                       - v_max: 最大速度限制 (默认: 33.33)
            **kwargs: 传递给父类的其他参数
        """
        # 传递三阶动力学模型参数给父类
        kwargs['epsilon'] = epsilon
        kwargs['kappa'] = kappa
        
        # 调用父类初始化
        super().__init__(**kwargs)
        
        # 保存三阶动力学模型参数
        self.epsilon = epsilon
        self.kappa = kappa
        
        # 低通滤波器配置
        self.use_action_filter = use_action_filter
        self.filter_alpha = filter_alpha
        self.step_count=0
        
        # CBF配置
        self.use_cbf = bool(use_cbf)
        self.cbf_params = self._setup_cbf_params(cbf_params)
        
        # 实验配置
        self.force_lead_cav_p_one = force_lead_cav_p_one
        
        # 网络攻击配置
        self.enable_cyber_attack = bool(enable_cyber_attack)
        self.attack_frequency = float(np.clip(attack_frequency, 0.0, 1.0))
        self.attack_mean = float(attack_mean)
        self.attack_start_time = float(max(0.0, attack_start_time))  # 攻击开始时间
        self.attack_type = attack_type  # 攻击类型
        self.attack_distribution = attack_distribution  # 攻击分布
        self.delay_steps = max(1, int(delay_steps))  # 延迟步数
        
        # 攻击目标：默认对速度和加速度发起攻击
        self.attack_targets = attack_targets or ["speed", "acceleration"]
        
        # 验证攻击目标的有效性
        valid_targets = {"speed", "position", "acceleration"}
        for target in self.attack_targets:
            assert target in valid_targets, f"无效的攻击目标: {target}"
        
        # 验证攻击类型的有效性
        valid_attack_types = {"data_tampering", "packet_drop", "delay"}
        assert self.attack_type in valid_attack_types, f"无效的攻击类型: {self.attack_type}"
        
        # 验证攻击分布的有效性
        valid_distributions = {"normal", "uniform"}
        assert self.attack_distribution in valid_distributions, f"无效的攻击分布: {self.attack_distribution}"
        
        # 配置各目标的方差
        self.attack_variances = self._setup_attack_variances(
            attack_variance, attack_variances, self.attack_targets
        )
        
        # 配置各目标的均值
        self.attack_means = self._setup_attack_means(
            attack_mean, attack_means, self.attack_targets
        )
        
        # 网络攻击状态跟踪
        self._attack_history: List[Dict[str, Any]] = []
        
        # CAV状态管理：真实状态vs网络传输状态
        self._cav_real_states: Dict[int, Dict[str, float]] = {}      # CAV真实物理状态
        self._cav_network_states: Dict[int, Dict[str, float]] = {}  # CAV网络传输状态（可能被攻击）
        self._cav_real_states_history: Dict[int, List[Dict[str, float]]] = {}  # CAV真实状态历史记录
        self._cav_network_states_history: Dict[int, List[Dict[str, float]]] = {}  # CAV网络传输状态历史
        
        # CAV动作历史记录（用于卡尔曼滤波器的控制输入）
        self._cav_actions_history: Dict[int, List[float]] = {}  # CAV动作历史记录
        self._cav_filtered_actions: Dict[int, float] = {}  # CAV滤波后的动作
        
        # HV状态管理：真实状态历史记录
        self._hv_real_states_history: Dict[int, List[Dict[str, float]]] = {}  # HV真实状态历史记录
        
        # 延迟攻击相关状态
        self._delay_buffers: Dict[int, Dict[str, List[float]]] = {}  # 延迟缓冲区
        self._initialize_delay_buffers()
        
        self._current_attack_offsets: Dict[int, Dict[str, float]] = {}  # 当前攻击偏差
        
        # CBF u_cbf값记录
        self._cbf_u_values: Dict[int, List[Dict]] = {}  # CBF u_cbf값记录
        
        # 回合jerk均值记录
        self._episode_jerk_means: List[float] = []  # 存储每个回合的jerk均值
        self._current_episode_jerk_values: List[float] = []  # 存储当前回合的jerk값
        
        # CAV安全指标记录
        self.cav_safety_metrics: Dict[int, Dict[str, Any]] = {}

    def _setup_attack_variances(self, single_variance: Optional[float], 
                               variance_dict: Optional[Dict[str, float]], 
                               targets: List[str]) -> Dict[str, float]:
        """
        设置各目标变量的攻击方差
        
        Args:
            single_variance: 统一方差值（向后兼容）
            variance_dict: 各目标独立方差字典
            targets: 攻击目标列表
            
        Returns:
            各目标的方差配置字典
        """
        # 默认方差配置（基于物理意义设置合理的默认值）
        default_variances = {
            "speed": 6,        # 速度方差 (m/s)²
            "position": 30,     # 位置方差 (m)²  
            "acceleration": 3  # 加速度方差 (m/s²)²
        }
        
        if variance_dict is not None:

            
            # 使用用户提供的独立方差配置
            result = {}
            for target in targets:
                if target in variance_dict:
                    result[target] = float(max(0.0, variance_dict[target]))
                else:
                    result[target] = default_variances.get(target, 1.0)
                    print(f"警告: 未为目标 '{target}' 指定方差，使用默认值 {result[target]}")
            return result
            
        elif single_variance is not None:
            # 使用统一方差（向后兼容）
            var_value = float(max(0.0, single_variance))
            return {target: var_value for target in targets}
            
        else:
            # 使用默认方差配置
            return {target: default_variances.get(target, 1.0) for target in targets}
            
    def _setup_attack_means(self, single_mean: float, 
                           mean_dict: Optional[Dict[str, float]], 
                           targets: List[str]) -> Dict[str, float]:
        """
        设置各目标变量的攻击均值
        
        Args:
            single_mean: 统一均值
            mean_dict: 各目标独立均值字典
            targets: 攻击目标列表
            
        Returns:
            各目标的均值配置字典
        """
        if mean_dict is not None:
            result = {}
            for target in targets:
                if target in mean_dict:
                    result[target] = float(mean_dict[target])
                else:
                    result[target] = float(single_mean)
            return result
        else:
            return {target: float(single_mean) for target in targets}
        
    def _setup_cbf_params(self, cbf_params: Optional[Dict[str, float]]) -> Dict[str, float]:
        """
        设置CBF参数
        
        Args:
            cbf_params: 用户提供的CBF参数字典
            
        Returns:
            配置后的CBF参数字典
        """
        # 默认CBF参数配置
        default_cbf_params = {
            "tau_i": 2.5,        # 时间常数参数
            "tau_star": 2.5,     # 时间常数参数
            "rho1": 1.5,         # CBF参数
            "rho2": 1.5,         # CBF参数
            "u_min": -3.0,       # 最小加速度限制
            "u_max": 3.0,        # 最大加速度限制
            "v_max": 33.33,      # 最大速度限制
        }
        
        if cbf_params is not None:
            # 使用用户提供的参数配置
            result = default_cbf_params.copy()
            result.update(cbf_params)
            return result
        else:
            # 使用默认参数配置
            return default_cbf_params
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境，清空攻击历史"""
        # 调用父类reset
        result = super().reset(seed=seed, options=options)
        self.step_count = 0
        
        # 清空攻击历史记录和状态数据
        self._attack_history = []
        self._cav_real_states = {}
        self._cav_network_states = {}
        self._cav_real_states_history = {}  # 重置真实状态历史记录
        self._cav_network_states_history = {}  # 重置网络传输状态历史记录
        self._cav_actions_history = {}  # 重置动作历史记录
        self._hv_real_states_history = {}  # 重置HV真实状态历史记录
        self._current_attack_offsets = {}
        
        # 重置CBF u_cbf값记录
        self._cbf_u_values = {}
        
        # 重置回合jerk均值记录
        if self._current_episode_jerk_values:
            # 计算当前回合的jerk均值并添加到历史记录中
            avg_jerk = np.mean(self._current_episode_jerk_values) if self._current_episode_jerk_values else 0.0
            self._episode_jerk_means.append(avg_jerk)
        self._current_episode_jerk_values = []  # 清空当前回合的jerk값
        
        # 重置CAV安全指标记录
        self.cav_safety_metrics = {}
        
        # 重置延迟缓冲区
        self._initialize_delay_buffers()
        
        # 初始化CAV状态
        self._update_cav_states()
        
        return result
    
    def _update_cav_states(self):
        """
        更新CAV的真实状态和网络传输状态
        
        真实状态：用于CAV自身决策和HV读取
        网络传输状态：用于CAV之间的信息传输（可能被攻击）
        """
        if not self.sim or not self.sim.vehicles:
            return
        
        # 更新CAV状态
        for cav_id in self.cav_ids:
            # 找到CAV车辆
            idx = next((i for i, v in enumerate(self.sim.vehicles) if v.vehicle_id == cav_id), None)
            if idx is None:
                continue
                
            veh = self.sim.vehicles[idx]
            
            # 更新真实状态（直接从车辆获取）
            self._cav_real_states[cav_id] = {
                'speed': veh.speed,
                'position': veh.x_front,
                'acceleration': veh.acceleration
            }
            
            # 初始化真实状态历史记录列表
            if cav_id not in self._cav_real_states_history:
                self._cav_real_states_history[cav_id] = []
            
            # 初始化网络传输状态（默认与真实状态相同）
            if cav_id not in self._cav_network_states:
                self._cav_network_states[cav_id] = self._cav_real_states[cav_id].copy()
            
            # 初始化网络传输状态历史记录列表
            if cav_id not in self._cav_network_states_history:
                self._cav_network_states_history[cav_id] = []
                
        # 更新HV历史状态
        for i, veh in enumerate(self.sim.vehicles):
            if not veh.is_cav:
                hv_id = veh.vehicle_id
                # 更新HV真实状态历史记录
                hv_state = {
                    'speed': veh.speed,
                    'position': veh.x_front,
                    'acceleration': veh.acceleration
                }
                
                # 初始化HV真实状态历史记录列表
                if hv_id not in self._hv_real_states_history:
                    self._hv_real_states_history[hv_id] = []
                
                # 添加当前状态到历史记录
                self._hv_real_states_history[hv_id].append(hv_state.copy())
    
    def step(self, action):
        """执行一步仿真，包含网络攻击逻辑"""
        assert self.sim is not None, "环境未重置，请先调用reset()"
        # 步骤1：处理CAV动作输入，统一为包含所有CAV的字典格式
        actions = self._process_cav_actions(action)
        
        # 记录CAV动作历史（用于卡尔曼滤波器的控制输入）
        for cav_id, action_value in actions.items():
            if cav_id not in self._cav_actions_history:
                self._cav_actions_history[cav_id] = []
            self._cav_actions_history[cav_id].append(action_value)
            
            # 限制历史记录长度，避免内存占用过大
            if len(self._cav_actions_history[cav_id]) > 1000:
                self._cav_actions_history[cav_id].pop(0)

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

        # 执行一步仿真
        self.sim.step(actions)

        self.step_count += 1
        
        
        # 步骤4：更新CAV真实状态
        self._update_cav_states()
        
        # 步骤5：生成网络攻击和更新网络传输状态（如果启用）
        if self.enable_cyber_attack:
            self._generate_network_attack()
        else:
            # 没有攻击时，网络传输状态等于真实状态
            for cav_id in self.cav_ids:
                if cav_id in self._cav_real_states:
                    self._cav_network_states[cav_id] = self._cav_real_states[cav_id].copy()
                    # 保存真实状态历史
                    self._cav_real_states_history[cav_id].append(self._cav_real_states[cav_id].copy())
                    # 更新延迟缓冲区（即使没有攻击也需要维护缓冲区）
                    for target in self.attack_targets:
                        if cav_id in self._delay_buffers and target in self._delay_buffers[cav_id]:
                            # 将当前真实值添加到缓冲区末尾
                            self._delay_buffers[cav_id][target].append(self._cav_real_states[cav_id][target])
                            # 移除最旧的值
                            self._delay_buffers[cav_id][target].pop(0)
                    # 保存网络传输状态历史
                    self._cav_network_states_history[cav_id].append(self._cav_network_states[cav_id].copy())
        
        # 步骤5.5：更新CAV安全指标
        self._update_cav_safety_metrics()
        
        # 步骤6：根据cavnet提供的位置速度、加速度确定网络攻击情况，通过realtimkf计算每个CAV的p값
        self._compute_cav_p_values()

        # 步骤7：利用模型推测中间hvs的平均速度和加速度
        self._predict_intermediate_hv_states()

        # 步骤8：计算观测、奖励和终止条件
        obs = self._build_observation()      # 构建观测
        reward = self._reward_multi(obs,actions)        # 计算奖励
        terminated = bool(self.sim.terminated) # 检查是否终止（如碰撞）
        truncated = False                    # Gymnasium的truncated标志
        
        # 如果回合结束，计算并记录当前回合的jerk均值
        if terminated or truncated:
            if self._current_episode_jerk_values:
                avg_jerk = float(np.mean(self._current_episode_jerk_values))
                self._episode_jerk_means.append(avg_jerk)
                # 清空当前回合的jerk값记录
                self._current_episode_jerk_values = []
        
        # 步骤9：构建信息字典
        info = {
            "state": self.sim.get_state(),      # 仿真器状态
            "collision": self.sim.collision_info # 碰撞信息
        }
        
        # 添加攻击信息到info中
        if self.enable_cyber_attack:
            info["cyber_attack"] = self._get_attack_info()

        # 步骤10：根据Gymnasium/Gym兼容性返回结果
        # 导入supports_gymnasium变量
        try:
            from sim_env.envs.single_lane_env import supports_gymnasium
        except ImportError:
            supports_gymnasium = False
            
        if supports_gymnasium:
            return obs, reward, terminated, truncated, info
        else:
            return obs, reward, terminated, info
    
    def _process_cav_actions(self, action):
        """
        统一处理CAV动作输入，始终返回包含所有CAV的字典格式
        
        Args:
            action: 输入的动作，可以是字典、列表、元组或numpy数组
            
        Returns:
            Dict[int, float]: 包含所有CAV ID和对应加速度的字典
        """
        actions = {}
        
        # 如果是空字典，表示所有CAV都使用IDM模型
        if isinstance(action, dict) and len(action) == 0:
            # 所有CAV都使用IDM模型，但仍然在字典中为每个CAV计算IDM加速度
            for cav_id in self.cav_ids:
                actions[cav_id] = self._compute_idm_acceleration_for_cav(cav_id)
                
        # 如果是列表、元组或numpy数组
        elif isinstance(action, (list, tuple, np.ndarray)):
            # 处理混合动作输入：[None, 1.0] 表示第一辆CAV使用IDM，第二辆CAV使用加速度1.0
            assert len(action) == len(self.cav_ids), "action长度与CAV数量不一致"
            for i, a in enumerate(action):
                cav_id = self.cav_ids[i]
                if a is not None:
                    # 裁剪加速度到安全范围
                    clipped_a = float(np.clip(a, -3.0, 3.0))
                    # 应用低通滤波器
                    filtered_a = self._apply_action_filter(cav_id, clipped_a)
                    actions[cav_id] = filtered_a
                else:
                    # 使用IDM模型计算加速度
                    actions[cav_id] = self._compute_idm_acceleration_for_cav(cav_id)
                    
        # 其他情况（单个值或数组）
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
            for i, cav_id in enumerate(self.cav_ids):
                # 应用低通滤波器
                filtered_a = self._apply_action_filter(cav_id, float(a_arr[i]))
                actions[cav_id] = filtered_a
                
        return actions

    def _apply_action_filter(self, cav_id: int, action: float) -> float:
        """
        应用低通滤波器到CAV动作，使动作输出更加平滑
        
        使用一阶低通滤波器：y[n] = α * x[n] + (1-α) * y[n-1]
        其中 α 是滤波系数，x[n] 是当前输入，y[n] 是当前输出
        
        Args:
            cav_id: CAV的ID
            action: 原始动作值
            
        Returns:
            float: 滤波后的动作值
        """
        # 如果启用CBF，先应用CBF
        if self.use_cbf:
            action = self._apply_cbf_control(cav_id, action)
            
        # 如果不启用滤波器，直接返回原始动作
        if not self.use_action_filter:
            return action
            
        # 初始化滤波后的动作值
        if cav_id not in self._cav_filtered_actions:
            self._cav_filtered_actions[cav_id] = action
            return action
            
        # 应用低通滤波器
        # y[n] = α * x[n] + (1-α) * y[n-1]
        filtered_action = (self.filter_alpha * action + 
                          (1 - self.filter_alpha) * self._cav_filtered_actions[cav_id])
                          
        # 更新滤波后的动作值
        self._cav_filtered_actions[cav_id] = filtered_action
        
        return filtered_action
    
    def _apply_cbf_control(self, cav_id: int, action: float) -> float:
        """
        应用CBF控制到CAV动作
        
        Args:
            cav_id: CAV的ID
            action: 原始动作值（RL输出的期望加速度）
            
        Returns:
            float: CBF处理后的动作值
        """
        # 确保仿真器已初始化
        if not self.sim or not self.sim.vehicles:
            return action
            
        try:
            # 找到CAV在车辆列表中的索引
            cav_idx = next((i for i, v in enumerate(self.sim.vehicles) if v.vehicle_id == cav_id), None)
            if cav_idx is None:
                return action
                
            # 获取当前车辆
            cav_vehicle = self.sim.vehicles[cav_idx]
            
            # 构建CBF状态，添加更多数值稳定性检查
            state = {
                'dx_1': 50.0,  # 与前车的相对距离（默认值）
                'dx_n': 50.0,  # 与后车的相对距离（默认值）
                'v': max(0.0, cav_vehicle.speed),  # 当前车速，确保非负
                'a': cav_vehicle.acceleration,  # 当前加速度
                'v_prev': max(0.0, cav_vehicle.speed),  # 前车速度（默认值）
                'v_n': max(0.0, cav_vehicle.speed),  # 后车速度（默认值）
            }
            
            # 获取前车信息
            if cav_idx > 0:
                lead_vehicle = self.sim.vehicles[cav_idx - 1]
                dx_1 = lead_vehicle.x_front - lead_vehicle.length - cav_vehicle.x_front
                state['dx_1'] = max(0.1, dx_1)  # 确保距离为正
                state['v_prev'] = max(0.0, lead_vehicle.speed)
                
            # 获取后车信息
            if cav_idx < len(self.sim.vehicles) - 1:
                follow_vehicle = self.sim.vehicles[cav_idx + 1]
                dx_n = cav_vehicle.x_front - cav_vehicle.length - follow_vehicle.x_front
                state['dx_n'] = max(0.1, dx_n)  # 确保距离为正
                state['v_n'] = max(0.0, follow_vehicle.speed)
            
            # 添加三阶动力学模型参数到CBF参数中
            cbf_params = self.cbf_params.copy()
            cbf_params['eps'] = max(1e-6, self.epsilon)
            cbf_params['kappa'] = max(1e-6, self.kappa)
            cbf_params['T'] = max(1e-6, self.dt)
            
            # 确保参数在合理范围内，增加更多数值稳定性检查
            cbf_params['tau_i'] = max(0.1, min(5.0, cbf_params.get('tau_i', 1.2)))
            cbf_params['tau_star'] = max(0.1, min(5.0, cbf_params.get('tau_star', 2.5)))
            cbf_params['rho1'] = max(0.01, min(2.0, cbf_params.get('rho1', 0.8)))
            cbf_params['rho2'] = max(0.01, min(2.0, cbf_params.get('rho2', 0.8)))
            cbf_params['u_min'] = min(-1.0, cbf_params.get('u_min', -3.0))  # 确保合理的下限
            cbf_params['u_max'] = max(1.0, cbf_params.get('u_max', 2.0))   # 确保合理的上限
            cbf_params['v_max'] = max(5.0, cbf_params.get('v_max', 33.33))  # 确保合理的上限
            
            # 确保状态变量在合理范围内
            state['dx_1'] = max(0.1, min(1000.0, state['dx_1']))  # 距离限制在合理范围内
            state['dx_n'] = max(0.1, min(1000.0, state['dx_n']))  # 距离限制在合理范围内
            state['v'] = max(0.0, min(50.0, state['v']))          # 速度限制在合理范围内
            state['v_prev'] = max(0.0, min(50.0, state['v_prev']))  # 速度限制在合理范围内
            state['v_n'] = max(0.0, min(50.0, state['v_n']))      # 速度限制在合理范围内
            state['a'] = max(-10.0, min(10.0, state['a']))        # 加速度限制在合理范围内
            
            # 导入CBF求解器
            from sim_env.utils.cbf import solve_cbf_qp
            
            # 求解CBF QP问题
            u_applied, feasible, info = solve_cbf_qp(state, action, cbf_params)
            
            # 记录u_cbf值（如果存在）
            if hasattr(self, '_cbf_u_values'):
                if cav_id not in self._cbf_u_values:
                    self._cbf_u_values[cav_id] = []
                # 从info中获取u_cbf值，如果不存在则设为0.0
                u_cbf_value = info.get('u_cbf', 0.0)
                self._cbf_u_values[cav_id].append({
                    'time': self.sim.t,
                    'u_cbf': u_cbf_value,
                    'feasible': feasible,
                    'raw_action': action,
                    'applied_action': u_applied
                })
            
            # 如果求解失败，返回原始动作
            if not feasible:
                # 可以选择打印调试信息
                # print(f"CBF求解失败: {info.get('error', 'Unknown error')}")
                return action
            
            return u_applied
            
        except Exception as e:
            # 如果CBF求解失败，返回原始动作
            # print(f"CBF异常: {str(e)}")
            return action

    def _compute_idm_acceleration_for_cav(self, cav_id):
        """
        为指定的CAV计算IDM加速度
        
        Args:
            cav_id: CAV的ID
            
        Returns:
            float: 计算得到的加速度值
        """
        # 找到CAV在车辆列表中的索引
        cav_idx = next((i for i, v in enumerate(self.sim.vehicles) if v.vehicle_id == cav_id), None)
        if cav_idx is None:
            return 0.0
            
        # 获取CAV车辆对象
        cav_vehicle = self.sim.vehicles[cav_idx]
        
        # 获取前导车辆
        leader = self.sim.get_leader(cav_idx)
        
        # 计算与前导车辆的车距
        gap = self.sim.gap_to_leader(cav_vehicle, leader)
        
        # 获取前导车辆的速度，如果没有前导车辆则为0
        v_lead = 0.0 if leader is None else leader.speed
        
        # 计算相对速度
        rel_v = cav_vehicle.speed - v_lead
        
        # 使用车辆自己的IDM参数，如果没有则使用默认参数
        idm_params = cav_vehicle.idm_params if cav_vehicle.idm_params is not None else self.sim.config.idm_params
        
        # 计算IDM加速度
        idm_accel = compute_idm_acceleration(cav_vehicle.speed, gap, rel_v, idm_params)
        
        # 对于IDM，使用三阶动力学模型
        # ai(tk+1) = (1 - T/εi)ai(tk) + (Tκi/εi)ui(tk)

        # 裁剪到合理范围
        return float(np.clip(idm_accel, -3.0, 3.0))
    
    def _predict_intermediate_hv_states(self):
        """
        利用模型推测CAV和参考车辆之间的HV车辆的平均速度和加速度
        输入前车CAV的数据和后车CAV前一辆HV的数据
        """
        # 检查是否已加载模型
        if not hasattr(self, '_prediction_model') or not hasattr(self, '_feature_scaler') or not hasattr(self, '_target_scaler'):
            self._load_prediction_model()
        
        # 如果模型加载失败，直接返回
        if not hasattr(self, '_prediction_model'):
            return
            
        # 获取车辆列表
        if not self.sim or not self.sim.vehicles:
            return
            
        vehicle_names = [v.vehicle_id for v in self.sim.vehicles]
        
        # 查找CAV车辆
        cav_vehicles = []
        for i, vehicle in enumerate(self.sim.vehicles):
            if vehicle.is_cav:
                cav_vehicles.append((i, vehicle))
        
        # 如果少于2个CAV，无法进行预测
        if len(cav_vehicles) < 2:
            return
            
        # 对每一对相邻的CAV进行预测
        # 注意：索引更小的车辆是前车（位置更靠前），索引更大的车辆是后车
        for i in range(len(cav_vehicles) - 1):
            front_cav_idx, front_cav = cav_vehicles[i]      # 索引更小的是前车
            rear_cav_idx, rear_cav = cav_vehicles[i + 1]    # 索引更大的是后车
            
            # 确保两个CAV之间有HV车辆
            if rear_cav_idx - front_cav_idx <= 1:
                continue
                
            # 获取中间HV车辆
            intermediate_hv_indices = list(range(front_cav_idx + 1, rear_cav_idx))
            if not intermediate_hv_indices:
                continue
                
            # 获取后车CAV前一辆HV车辆的索引（即最靠近后车CAV的HV车辆）
            closest_hv_to_rear_cav_idx = rear_cav_idx - 1
            if closest_hv_to_rear_cav_idx <= front_cav_idx or closest_hv_to_rear_cav_idx >= rear_cav_idx:
                continue
                
            # 准备输入特征（使用前方CAV和后方CAV前一辆HV的历史数据）
            # 获取CAV的网络传输状态历史数据
            front_cav_id = front_cav.vehicle_id
            closest_hv_id = self.sim.vehicles[closest_hv_to_rear_cav_idx].vehicle_id
            rear_cav_id = rear_cav.vehicle_id
            
            # 检查是否有足够的历史数据
            if (front_cav_id not in self._cav_network_states_history or 
                closest_hv_id not in self._hv_real_states_history):  # HV车辆使用HV历史数据
                continue
                
            front_cav_history = self._cav_network_states_history[front_cav_id]
            closest_hv_history = self._hv_real_states_history[closest_hv_id]  # 使用HV的真实历史数据
            
            # 确保有足够的历史数据（至少50个时间步）
            if len(front_cav_history) < 50 or len(closest_hv_history) < 50:
                continue
            
            # 准备250维输入特征，使用滑动窗口的历史数据
            input_features = self._prepare_input_features_for_prediction_from_history(
                front_cav_history, closest_hv_history)
            
            if input_features is not None:
                # 使用模型进行预测
                try:
                    predicted_states = self._predict_with_model(input_features, len(intermediate_hv_indices))
                    
                    # 存储预测结果
                    if not hasattr(self, '_predicted_hv_states'):
                        self._predicted_hv_states = {}
                        
                    # 计算中间HV车辆的平均速度和加速度
                    avg_speed = predicted_states[0]
                    avg_accel = predicted_states[1]
                    
                    # 使用前方CAV的名字作为键存储预测结果（因为是前方CAV对后方HV的预测）
                    self._predicted_hv_states[rear_cav_id] = {
                        'speed': avg_speed,
                        'acceleration': avg_accel,
                        'timestamp': self.sim.t,
                        'target_hv_indices': intermediate_hv_indices  # 记录目标HV车辆索引
                    }
                    
                    # 记录真实HV的平均速度和加速度
                    if not hasattr(self, '_real_hv_states'):
                        self._real_hv_states = {}
                    
                    # 计算真实HV的平均速度和加速度
                    real_speeds = []
                    real_accels = []
                    for hv_idx in intermediate_hv_indices:
                        hv_vehicle = self.sim.vehicles[hv_idx]
                        real_speeds.append(hv_vehicle.speed)
                        real_accels.append(hv_vehicle.acceleration)
                    
                    real_avg_speed = np.mean(real_speeds) if real_speeds else 0.0
                    real_avg_accel = np.mean(real_accels) if real_accels else 0.0
                    
                    # 存储真实HV的平均状态
                    self._real_hv_states[rear_cav_id] = {
                        'speed': real_avg_speed,
                        'acceleration': real_avg_accel,
                        'timestamp': self.sim.t,
                        'target_hv_indices': intermediate_hv_indices
                    }
                except Exception as e:
                    # 预测失败，跳过
                    pass
    
    def _prepare_input_features_for_prediction_from_history(self, front_cav_history, closest_hv_history):
        """
        从历史数据准备250维输入特征用于预测，使用滑动窗口技术
        
        Args:
            front_cav_history: 前方CAV的历史状态列表
            closest_hv_history: 后方CAV前一辆HV的历史状态列表
            
        Returns:
            np.array: 250维输入特征向量，如果数据不足则返回None
        """
        try:
            # 使用最后50个时间步的数据（滑动窗口大小）
            window_size = 50
            
            # 确保有足够的历史数据
            if len(front_cav_history) < window_size or len(closest_hv_history) < window_size:
                return None
            
            # 提取最后50个时间步的数据
            front_recent = front_cav_history[-window_size:]
            closest_hv_recent = closest_hv_history[-window_size:]
            
            # 提取各个维度的数据
            front_speeds = [state['speed'] for state in front_recent]
            front_accels = [state['acceleration'] for state in front_recent]
            front_positions = [state['position'] for state in front_recent]
            
            closest_hv_speeds = [state['speed'] for state in closest_hv_recent]
            closest_hv_accels = [state['acceleration'] for state in closest_hv_recent]
            closest_hv_positions = [state['position'] for state in closest_hv_recent]
            
            # 计算相对于参考车辆的差值（与测试脚本保持一致）
            speed_diff = np.array(front_speeds) - np.array(closest_hv_speeds)
            accel_diff = np.array(front_accels) - np.array(closest_hv_accels)
            position_diff = np.array(front_positions) - np.array(closest_hv_positions)
            
            # 构建250维输入特征（与测试脚本保持一致）
            input_features = []
            input_features.extend(speed_diff.tolist())      # 50维：速度差
            input_features.extend(accel_diff.tolist())     # 50维：加速度差
            input_features.extend(position_diff.tolist())  # 50维：位置差
            input_features.extend(front_speeds)             # 50维：前方CAV速度
            input_features.extend(front_accels)             # 50维：前方CAV加速度
            
            return np.array(input_features)
        except Exception as e:
            return None

    def _load_prediction_model(self):
        """加载用于预测的模型"""
        try:
            # 导入必要的模块
            import os
            import sys
            import torch
            import pickle
            
            # 添加项目路径
            _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
            _PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
            if _PROJECT_ROOT not in sys.path:
                sys.path.insert(0, _PROJECT_ROOT)
            
            # 动态导入模型类
            from examples.test_enhanced_model_under_cyber_attack import TopologyAwareLSTM
            
            # 查找模型文件
            model_dir = os.path.join(_PROJECT_ROOT, "models")
            if not os.path.exists(model_dir):
                return
                
            model_files = [f for f in os.listdir(model_dir) if f.startswith("enhanced_traffic_lstm_model_cyber_attack_") and f.endswith(".pth")]
            if not model_files:
                return
                
            # 使用最新的模型文件
            latest_model = sorted(model_files)[-1]
            timestamp = latest_model.replace("enhanced_traffic_lstm_model_cyber_attack_", "").replace(".pth", "")
            
            model_path = os.path.join(model_dir, f"enhanced_traffic_lstm_model_cyber_attack_{timestamp}.pth")
            scaler_path = os.path.join(model_dir, f"enhanced_scalers_cyber_attack_{timestamp}.pkl")
            
            # 检查文件是否存在
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                return
            
            # 加载模型
            model = TopologyAwareLSTM(
                input_size=250,
                topology_size=2,
                hidden_size=96,
                num_layers=3,
                output_size=2,
                dropout=0.3,
                use_topology=True
            )
            
            # 加载模型权重
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            
            # 加载标准化器
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
                feature_scaler = scalers['feature_scaler']
                target_scaler = scalers['target_scaler']
                topology_scaler = scalers.get('topology_scaler', None)
            
            # 保存模型和标准化器
            self._prediction_model = model
            self._feature_scaler = feature_scaler
            self._target_scaler = target_scaler
            self._topology_scaler = topology_scaler
            #打印路径
            print(f"Model path: {model_path}")
            
        except Exception as e:
            print(f"Failed to load prediction model: {e}")
            pass
    
    def _predict_with_model(self, input_features, num_hv_vehicles):
        """
        使用模型进行预测
        
        Args:
            input_features: 250维输入特征
            num_hv_vehicles: 中间HV车辆数量
            
        Returns:
            list: [rear_hv_speed, front_hv_speed, rear_hv_accel, front_hv_accel]
        """
        try:
            import torch
            import numpy as np
            
            # 标准化输入特征
            input_scaled = self._feature_scaler.transform(input_features.reshape(1, -1))
            
            # 准备拓扑信息（车辆数量和CAV位置）
            # 简化处理：假设总共有2个CAV和num_hv_vehicles个HV
            total_vehicles = 2 + num_hv_vehicles
            cav_position = 0  # 后方CAV位置
            topology_info = np.array([[total_vehicles, cav_position]])
            
            if self._topology_scaler:
                topology_scaled = self._topology_scaler.transform(topology_info)
            else:
                topology_scaled = topology_info
            
            # 模型预测
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input_scaled)
                topology_tensor = torch.FloatTensor(topology_scaled)
                
                prediction = self._prediction_model(input_tensor, topology_tensor)
                prediction_np = self._target_scaler.inverse_transform(prediction.numpy())
                
                # 返回预测结果 [后方HV速度, 前方HV速度, 后方HV加速度, 前方HV加速度]
                # 由于模型输出是2维，我们假设第一个是速度，第二个是加速度
                result = [float(prediction_np[0, 0]), float(prediction_np[0, 1])]
                return result
        except Exception as e:
            # 预测失败，返回默认值
            return [0.0, 0.0]
    
    def get_predicted_hv_states(self):
        """
        获取预测的HV车辆状态
        
        Returns:
            Dict: 预测的HV车辆状态字典
        """
        if hasattr(self, '_predicted_hv_states'):
            return self._predicted_hv_states.copy()
        return {}
    
    def get_real_hv_states(self):
        """
        获取真实HV车辆的平均状态
        
        Returns:
            Dict: 真实HV车辆的平均状态字典
        """
        if hasattr(self, '_real_hv_states'):
            return self._real_hv_states.copy()
        return {}
    
    def _compute_cav_p_values(self):
        """
        使用RealTimeKFDetector计算每个CAV的p值，不修改观测值
        将结果存储在info中供外部使用
        """
        # 只有在启用网络攻击时才进行检测
        if not self.enable_cyber_attack:
            return
        
        # 确保在正确的时机进行检测（仿真时间足够长）
        if self.sim.t < self.attack_start_time:
            return
            
        # 为每个CAV创建或更新RealTimeKFDetector实例
        if not hasattr(self, '_cav_detectors'):
            self._cav_detectors = {}
            
        # 存储CAV的p值
        if not hasattr(self, '_cav_p_values'):
            self._cav_p_values = {}
            
        # 获取CAV对比数据
        cav_comparison_data = self.get_cav_comparison_data()
        
        # 对每个CAV进行攻击检测
        for cav_id in self.cav_ids:
            if cav_id not in cav_comparison_data:
                self._cav_p_values[cav_id] = 1.0  # 默认可信度为1.0
                continue
                
            cav_data = cav_comparison_data[cav_id]
            real_data = cav_data['real']
            network_data = cav_data['network']
            
            # 初始化该CAV的检测器（如果尚未创建）
            if cav_id not in self._cav_detectors:
                # 初始化三阶卡尔曼滤波器参数
                dt = self.dt
                epsilon = self.epsilon  # 从环境配置中获取三阶动力学参数
                kappa = self.kappa      # 从环境配置中获取三阶动力学参数
                
                # 噪声协方差矩阵
                Q = np.array([
                    [1e-4, 0, 0],
                    [0, 1e-4, 0],
                    [0, 0, 1e-4]
                ])
                R = np.array([
                    [1e-2, 0, 0],
                    [0, 1e-2, 0],
                    [0, 0, 1e-2]
                ])
                
                # 创建检测器实例
                from sim_env.utils.RealTimeKFDetector import RealTimeKFDetector3Full
                self._cav_detectors[cav_id] = RealTimeKFDetector3Full(
                    T=dt, 
                    eps=epsilon, 
                    kappa=kappa, 
                    Q=Q, 
                    R=R, 
                    alpha=0.05,
                    reset_P_scale=10.0
                )
                
                # 初始化检测器状态
                initial_position = real_data['position']
                initial_velocity = real_data['speed']
                initial_acceleration = real_data['acceleration']
                self._cav_detectors[cav_id].x_est = np.array([[initial_position], [initial_velocity], [initial_acceleration]])
            
            # 获取检测器
            detector = self._cav_detectors[cav_id]
            
            # 构造观测向量（使用网络传输数据，可能被攻击）
            # 处理可能为None的值
            position = network_data['position'] if network_data['position'] is not None else 0
            speed = network_data['speed'] if network_data['speed'] is not None else 0
            acceleration = network_data['acceleration'] if network_data['acceleration'] is not None else 0
            z = np.array([[position], [speed], [acceleration]])
            
            # 获取控制输入u（action），而不是实际加速度
            # 我们需要从CAV的动作历史中获取控制输入u
            control_input_u = 0.0  # 默认值
            
            # 如果有CAV动作历史，使用最近的动作作为控制输入
            if hasattr(self, '_cav_actions_history') and cav_id in self._cav_actions_history:
                if self._cav_actions_history[cav_id]:
                    control_input_u = self._cav_actions_history[cav_id][-1]  # 使用最近的动作
            
            # 使用RealTimeKFDetector处理网络传输数据
            result = detector.update(z, control_input_u)
            p_value = result['p_value_kf']
            trust_score = result['trust_score']
            
            # 存储该CAV的p值
            self._cav_p_values[cav_id] = p_value
            
    
    def get_cav_p_values(self):
        """
        获取所有CAV的p值
        
        Returns:
            Dict[int, float]: 每个CAV的p值字典
        """
        if hasattr(self, '_cav_p_values'):
            return self._cav_p_values.copy()
        return {}
    
    def get_cbf_u_values(self):
        """
        获取所有CAV的CBF u_cbf值记录
        
        Returns:
            Dict[int, List[Dict]]: 每个CAV的u_cbf值记录列表
        """
        if hasattr(self, '_cbf_u_values'):
            return self._cbf_u_values.copy()
        return {}
    
    def _generate_network_attack(self) -> None:
        """
        生成网络攻击，更新CAV的网络传输状态
        
        攻击仅影响网络传输状态，不影响CAV的真实物理状态。
        - CAV自身决策：使用真实状态
        - CAV之间通信：使用网络传输状态（可能被攻击）
        - HV读取CAV：使用真实状态（无网络传输）
        
        注意：在攻击开始时间之前不会发生攻击。
        """
        assert self.sim is not None, "仿真器未初始化"
        
        # 清空当前攻击偏差
        self._current_attack_offsets = {}
        
        # 检查是否已到攻击开始时间
        if self.sim.t < self.attack_start_time:
            # 攻击免疫期，网络传输状态等于真实状态
            for cav_id in self.cav_ids:
                if cav_id in self._cav_real_states:
                    self._cav_network_states[cav_id] = self._cav_real_states[cav_id].copy()
                    # 保存真实状态历史
                    self._cav_real_states_history[cav_id].append(self._cav_real_states[cav_id].copy())
                    # 保存网络传输状态历史
                    self._cav_network_states_history[cav_id].append(self._cav_network_states[cav_id].copy())
            return
        
        # 逐个CAV检查是否发生攻击
        for cav_id in self.cav_ids:
            # 默认网络传输状态等于真实状态
            if cav_id in self._cav_real_states:
                self._cav_network_states[cav_id] = self._cav_real_states[cav_id].copy()
            
            # 根据攻击类型执行不同的攻击逻辑
            if self.attack_type == "packet_drop":
                self._apply_packet_drop_attack(cav_id)
            elif self.attack_type == "delay":
                self._apply_delay_attack(cav_id)
            else:  # 默认为数据篡改攻击
                self._apply_data_tampering_attack(cav_id)
            
            # 保存真实状态历史（无论是否发生攻击都需要保存）
            self._cav_real_states_history[cav_id].append(self._cav_real_states[cav_id].copy())
            # 保存网络传输状态历史（无论是否发生攻击都需要保存）
            self._cav_network_states_history[cav_id].append(self._cav_network_states[cav_id].copy())
    
    def _apply_data_tampering_attack(self, cav_id: int) -> None:
        """
        应用数据篡改攻击
        
        Args:
            cav_id: CAV的ID
        """
        # 根据攻击频率随机决定是否攻击
        if self.rng.random() < self.attack_frequency:
            # 初始化攻击信息记录
            attack_record = {
                "time": self.sim.t,
                "cav_id": cav_id,
                "attack_type": "data_tampering",
                "attacked_targets": [],
                "attack_values": {},
                "original_values": {},
                "attack_offsets": {}  # 记录各目标的攻击偏差
            }
            
            # 初始化攻击偏差
            self._current_attack_offsets[cav_id] = {}
            
            # 对每个攻击目标生成攻击数据
            for target in self.attack_targets:
                # 获取该目标的均值和方差
                target_mean = self.attack_means[target]
                target_variance = self.attack_variances[target]
                
                # 根据分布类型生成攻击偏差
                if self.attack_distribution == "uniform":
                    # 均匀分布：U(mean - sqrt(3*var), mean + sqrt(3*var))
                    # 保持与正态分布相同的方差
                    bound = np.sqrt(3 * target_variance)
                    attack_offset = self.rng.uniform(target_mean - bound, target_mean + bound) 
                else:
                    # 正态分布（默认）
                    attack_offset = self.rng.normal(target_mean, np.sqrt(target_variance)) 
                attack_record["attack_offsets"][target] = attack_offset
                self._current_attack_offsets[cav_id][target] = attack_offset
                
                # 获取真实状态值
                real_value = self._cav_real_states[cav_id][target]
                
                # 计算被攻击的网络传输值
                if target == "speed":
                    attacked_value = max(0.0, real_value + attack_offset)  # 保证速度非负
                else:
                    attacked_value = real_value + attack_offset
                
                # 更新网络传输状态
                self._cav_network_states[cav_id][target] = attacked_value
                
                # 记录攻击信息
                attack_record["original_values"][target] = real_value
                attack_record["attack_values"][target] = attacked_value
                attack_record["attacked_targets"].append(target)
            
            # 将攻击记录添加到历史中
            if attack_record["attacked_targets"]:  # 只记录有效攻击
                self._attack_history.append(attack_record)
    
    def _apply_packet_drop_attack(self, cav_id: int) -> None:
        """
        应用丢包攻击
        
        Args:
            cav_id: CAV的ID
        """
        # 根据攻击频率随机决定是否丢包
        if self.rng.random() < self.attack_frequency:
            # 丢包攻击：使用上一时刻的网络传输状态替代当前状态，而不是设置为None
            # 如果没有历史状态，则使用真实状态
            if (cav_id in self._cav_network_states_history and 
                len(self._cav_network_states_history[cav_id]) > 0):
                # 使用上一时刻的网络传输状态
                last_network_state = self._cav_network_states_history[cav_id][-1]
                for target in self.attack_targets:
                    # 只有当上一时刻的状态不为None时才使用
                    if last_network_state.get(target) is not None:
                        self._cav_network_states[cav_id][target] = last_network_state[target]
                    else:
                        # 如果上一时刻也为None，则回退到真实状态
                        self._cav_network_states[cav_id][target] = self._cav_real_states[cav_id][target]
            else:
                # 没有历史状态时，使用真实状态
                for target in self.attack_targets:
                    self._cav_network_states[cav_id][target] = self._cav_real_states[cav_id][target]
            
            # 记录丢包攻击
            attack_record = {
                "time": self.sim.t,
                "cav_id": cav_id,
                "attack_type": "packet_drop",
                "attacked_targets": self.attack_targets.copy(),  # 所有目标都受影响
                "attack_values": self._cav_network_states[cav_id].copy(),
                "original_values": self._cav_real_states[cav_id].copy(),
                "attack_offsets": {target: 0.0 for target in self.attack_targets}  # 丢包攻击没有偏移量
            }
            
            self._attack_history.append(attack_record)
    
    def _apply_delay_attack(self, cav_id: int) -> None:
        """
        应用延迟攻击
        
        Args:
            cav_id: CAV的ID
        """
        # 根据攻击频率随机决定是否攻击
        if self.rng.random() < self.attack_frequency:
            # 延迟攻击：使用延迟缓冲区中的旧数据
            for target in self.attack_targets:
                # 将当前真实值添加到缓冲区末尾
                self._delay_buffers[cav_id][target].append(self._cav_real_states[cav_id][target])
                # 从缓冲区开头取出延迟的数据
                delayed_value = self._delay_buffers[cav_id][target].pop(0)
                # 更新网络传输状态
                self._cav_network_states[cav_id][target] = delayed_value
            
            # 记录延迟攻击
            attack_record = {
                "time": self.sim.t,
                "cav_id": cav_id,
                "attack_type": "delay",
                "attacked_targets": self.attack_targets.copy(),  # 所有目标都受影响
                "attack_values": self._cav_network_states[cav_id].copy(),
                "original_values": self._cav_real_states[cav_id].copy(),
                "attack_offsets": {target: 0.0 for target in self.attack_targets}  # 延迟攻击没有偏移量
            }
            
            self._attack_history.append(attack_record)
        else:
            # 没有攻击时，更新延迟缓冲区但使用当前真实值
            for target in self.attack_targets:
                # 将当前真实值添加到缓冲区末尾
                self._delay_buffers[cav_id][target].append(self._cav_real_states[cav_id][target])
                # 移除最旧的值
                self._delay_buffers[cav_id][target].pop(0)
    
    def get_cav_state(self, cav_id: int, requester_type: str = "cav", requester_id: Optional[int] = None) -> Dict[str, float]:
        """
        获取CAV的状态数据，根据请求者类型返回不同的状态
        
        Args:
            cav_id: 目标CAV的ID
            requester_type: 请求者类型，"cav"或"hv"
            requester_id: 请求者的ID（可选）
            
        Returns:
            CAV的状态数据字典
            
        逻辑：
        - CAV自身决策（requester_id == cav_id）：返回真实状态
        - CAV之间通信（requester_type == "cav" and requester_id != cav_id）：返回网络传输状态
        - HV读取CAV（requester_type == "hv"）：返回真实状态（无网络传输）
        """
        if cav_id not in self._cav_real_states:
            return {'speed': 0.0, 'position': 0.0, 'acceleration': 0.0}
        
        # CAV自身决策或HV读取：使用真实状态
        if (requester_type == "hv" or 
            (requester_type == "cav" and requester_id == cav_id)):
            return self._cav_real_states[cav_id].copy()
        
        # CAV之间通信：使用网络传输状态（可能被攻击）
        elif requester_type == "cav" and requester_id != cav_id:
            return self._cav_network_states.get(cav_id, self._cav_real_states[cav_id]).copy()
        
        # 默认返回真实状态
        return self._cav_real_states[cav_id].copy()
    
    def get_cav_network_states_history(self, cav_id: Optional[int] = None) -> Dict[int, List[Dict[str, float]]]:
        """
        获取CAV的网络传输状态历史记录
        
        Args:
            cav_id: 特定CAV的ID，如果为None则返回所有CAV的历史记录
            
        Returns:
            CAV网络传输状态历史记录字典
        """
        if cav_id is not None:
            return {cav_id: self._cav_network_states_history.get(cav_id, [])}
        return self._cav_network_states_history.copy()
    
    def get_cav_real_states_history(self, cav_id: Optional[int] = None) -> Dict[int, List[Dict[str, float]]]:
        """
        获取CAV的真实状态历史记录
        
        Args:
            cav_id: 特定CAV的ID，如果为None则返回所有CAV的历史记录
            
        Returns:
            CAV真实状态历史记录字典
        """
        if cav_id is not None:
            return {cav_id: self._cav_real_states_history.get(cav_id, [])}
        return self._cav_real_states_history.copy()
    
    def get_hv_real_states_history(self, hv_id: Optional[int] = None) -> Dict[int, List[Dict[str, float]]]:
        """
        获取HV的真实状态历史记录
        
        Args:
            hv_id: 特定HV的ID，如果为None则返回所有HV的历史记录
            
        Returns:
            HV真实状态历史记录字典
        """
        if hv_id is not None:
            return {hv_id: self._hv_real_states_history.get(hv_id, [])}
        return self._hv_real_states_history.copy()
    
    def get_cav_comparison_data(self):
        """
        获取CAV真实状态和网络传输状态的对比数据（用于可视化和分析）
        
        Returns:
            Dict: {
                cav_id: {
                    'real': {'speed': float, 'position': float, 'acceleration': float},
                    'network': {'speed': float, 'position': float, 'acceleration': float},
                    'offsets': {'speed': float, 'position': float, 'acceleration': float}
                }
            }
        """
        result = {}
        
        for cav_id in self.cav_ids:
            if cav_id in self._cav_real_states:
                # 获取攻击偏差
                offsets = self._current_attack_offsets.get(cav_id, {})
                
                result[cav_id] = {
                    'real': self._cav_real_states[cav_id].copy(),
                    'network': self._cav_network_states.get(cav_id, self._cav_real_states[cav_id]).copy(),
                    'offsets': {
                        'speed': offsets.get('speed', 0.0),
                        'position': offsets.get('position', 0.0),
                        'acceleration': offsets.get('acceleration', 0.0)
                    }
                }
        
        return result
    
    def _get_attack_info(self) -> Dict[str, Any]:
        """
        获取当前步骤的攻击信息
        
        Returns:
            包含攻击统计和最近攻击记录的字典
        """
        # 统计总攻击次数
        total_attacks = len(self._attack_history)
        
        # 统计各类型攻击次数
        target_counts = {}
        attack_type_counts = {}
        for record in self._attack_history:
            # 统计目标攻击次数
            for target in record["attacked_targets"]:
                target_counts[target] = target_counts.get(target, 0) + 1
            # 统计攻击类型次数
            attack_type = record.get("attack_type", "data_tampering")
            attack_type_counts[attack_type] = attack_type_counts.get(attack_type, 0) + 1
        
        # 获取最近的攻击记录（当前步骤）
        current_attacks = [
            record for record in self._attack_history 
            if abs(record["time"] - self.sim.t) < 1e-6
        ]
        
        return {
            "enabled": self.enable_cyber_attack,
            "attack_frequency": self.attack_frequency,
            "attack_mean": self.attack_mean,
            "attack_means": dict(self.attack_means),  # 新增：各目标独立均值
            "attack_start_time": self.attack_start_time,  # 新增：攻击开始时间
            "attack_type": self.attack_type,  # 新增：攻击类型
            "attack_distribution": self.attack_distribution,  # 新增：攻击分布
            "delay_steps": self.delay_steps if self.attack_type == "delay" else None,  # 延迟步数
            "current_simulation_time": self.sim.t,        # 新增：当前仿真时间
            "attack_active": self.sim.t >= self.attack_start_time,  # 新增：攻击是否激活
            "attack_variances": dict(self.attack_variances),  # 各目标独立方差
            "attack_targets": self.attack_targets,
            "total_attacks": total_attacks,
            "target_counts": target_counts,
            "attack_type_counts": attack_type_counts,  # 新增：攻击类型统计
            "current_step_attacks": current_attacks,
            "attack_history": self._attack_history[-10:]  # 只返回最近10次攻击记录
        }
    
    def get_cyber_attack_stats(self) -> Dict[str, Any]:
        """
        获取网络攻击的详细统计信息
        
        返回包括攻击次数、攻击率、各类型攻击分布等统计数据。
        
        Returns:
            攻击统计信息字典
        """
        if not self.enable_cyber_attack:
            return {"enabled": False, "message": "网络攻击功能未启用"}
        
        # 计算基本统计数据
        total_attacks = len(self._attack_history)
        total_steps = max(1, int(self.sim.t / self.dt))  # 总仿真步数
        actual_attack_rate = total_attacks / total_steps if total_steps > 0 else 0.0
        
        # 统计各目标攻击次数
        target_stats = {}
        for target in ["speed", "position", "acceleration"]:
            count = sum(1 for record in self._attack_history if target in record["attacked_targets"])
            target_stats[target] = {
                "count": count,
                "percentage": (count / total_attacks * 100) if total_attacks > 0 else 0.0
            }
        
        # 统计攻击类型次数
        attack_type_stats = {}
        for record in self._attack_history:
            attack_type = record.get("attack_type", "data_tampering")
            if attack_type not in attack_type_stats:
                attack_type_stats[attack_type] = 0
            attack_type_stats[attack_type] += 1
        
        # 计算各攻击类型百分比
        for attack_type, count in attack_type_stats.items():
            attack_type_stats[attack_type] = {
                "count": count,
                "percentage": (count / total_attacks * 100) if total_attacks > 0 else 0.0
            }
        
        # 统计每个CAV的攻击次数
        cav_stats = {}
        for cav_id in self.cav_ids:
            count = sum(1 for record in self._attack_history if record["cav_id"] == cav_id)
            cav_stats[cav_id] = {
                "count": count,
                "percentage": (count / total_attacks * 100) if total_attacks > 0 else 0.0
            }
        
        return {
            "enabled": True,
            "config": {
                "attack_frequency": self.attack_frequency,
                "attack_mean": self.attack_mean,
                "attack_means": dict(self.attack_means),  # 新增：各目标独立均值
                "attack_start_time": self.attack_start_time,  # 新增：攻击开始时间配置
                "attack_type": self.attack_type,  # 新增：攻击类型
                "attack_distribution": self.attack_distribution,  # 新增：攻击分布
                "delay_steps": self.delay_steps if self.attack_type == "delay" else None,  # 延迟步数
                "attack_variances": dict(self.attack_variances),  # 各目标独立方差
                "attack_targets": self.attack_targets
            },
            "statistics": {
                "total_attacks": total_attacks,
                "total_steps": total_steps,
                "expected_attack_rate": self.attack_frequency,
                "actual_attack_rate": actual_attack_rate,
                "target_distribution": target_stats,
                "attack_type_distribution": attack_type_stats,  # 新增：攻击类型分布
                "cav_distribution": cav_stats
            },
            "recent_attacks": self._attack_history[-20:]  # 最近20次攻击记录
        }
    
    def set_cyber_attack_config(self, **kwargs) -> None:
        """
        动态更新网络攻击配置
        
        允许在仿真过程中动态调整攻击参数。
        
        Args:
            enable_cyber_attack: 是否启用攻击
            attack_frequency: 攻击频率
            attack_mean: 攻击均值（统一）
            attack_means: 各目标独立均值字典
            attack_start_time: 攻击开始时间（秒）
            attack_start_time: 攻击开始时间（秒）
            attack_type: 攻击类型
            attack_distribution: 攻击分布 ("normal" 或 "uniform")
            delay_steps: 延迟步数
            attack_variance: 统一方差（向后兼容，会应用到所有目标）
            attack_variances: 各目标独立方差字典
            attack_targets: 攻击目标列表
        """
        if "enable_cyber_attack" in kwargs:
            self.enable_cyber_attack = bool(kwargs["enable_cyber_attack"])
            
        if "attack_frequency" in kwargs:
            self.attack_frequency = float(np.clip(kwargs["attack_frequency"], 0.0, 1.0))
            
        if "attack_mean" in kwargs:
            self.attack_mean = float(kwargs["attack_mean"])
            # 更新所有目标的均值
            for target in self.attack_targets:
                self.attack_means[target] = self.attack_mean
                
        if "attack_means" in kwargs:
            new_means = kwargs["attack_means"]
            valid_targets = {"speed", "position", "acceleration"}
            for target, mean_val in new_means.items():
                if target in valid_targets and target in self.attack_means:
                    self.attack_means[target] = float(mean_val)
                else:
                    print(f"警告: 忽略无效的攻击目标 '{target}'")
            
        if "attack_start_time" in kwargs:
            self.attack_start_time = float(max(0.0, kwargs["attack_start_time"]))
            
        if "attack_type" in kwargs:
            valid_attack_types = {"data_tampering", "packet_drop", "delay"}
            if kwargs["attack_type"] in valid_attack_types:
                self.attack_type = kwargs["attack_type"]
            else:
                print(f"警告: 忽略无效的攻击类型 '{kwargs['attack_type']}'")
        
        if "attack_distribution" in kwargs:
            valid_distributions = {"normal", "uniform"}
            if kwargs["attack_distribution"] in valid_distributions:
                self.attack_distribution = kwargs["attack_distribution"]
            else:
                print(f"警告: 忽略无效的攻击分布 '{kwargs['attack_distribution']}'")
        
        if "delay_steps" in kwargs:
            self.delay_steps = max(1, int(kwargs["delay_steps"]))
            # 重新初始化延迟缓冲区
            self._initialize_delay_buffers()
            
        # 处理方差更新（支持统一方差和独立方差）
        if "attack_variance" in kwargs:
            # 统一方差（向后兼容）
            var_value = float(max(0.0, kwargs["attack_variance"]))
            # 将统一方差应用到所有当前攻击目标
            for target in self.attack_targets:
                self.attack_variances[target] = var_value
                
        if "attack_variances" in kwargs:
            # 各目标独立方差
            new_variances = kwargs["attack_variances"]
            valid_targets = {"speed", "position", "acceleration"}
            for target, variance in new_variances.items():
                if target in valid_targets:
                    self.attack_variances[target] = float(max(0.0, variance))
                else:
                    print(f"警告: 忽略无效的攻击目标 '{target}'")
            
        if "attack_targets" in kwargs:
            valid_targets = {"speed", "position", "acceleration"}
            new_targets = kwargs["attack_targets"]
            for target in new_targets:
                assert target in valid_targets, f"无效的攻击目标: {target}"
            self.attack_targets = list(new_targets)
            
            # 确保新的攻击目标有对应的方差配置
            for target in self.attack_targets:
                if target not in self.attack_variances:
                    # 使用默认方差值
                    default_variances = {
                        "speed": 1.0,
                        "position": 0.5,
                        "acceleration": 2.0
                    }
                    self.attack_variances[target] = default_variances.get(target, 1.0)
                    print(f"信息: 为新攻击目标 '{target}' 设置默认方差 {self.attack_variances[target]}")
            
            # 确保新的攻击目标有对应的均值配置
            for target in self.attack_targets:
                if target not in self.attack_means:
                    self.attack_means[target] = self.attack_mean
                    print(f"信息: 为新攻击目标 '{target}' 设置默认均值 {self.attack_means[target]}")
    
    def _initialize_delay_buffers(self):
        """初始化延迟缓冲区"""
        self._delay_buffers = {}
        for cav_id in self.cav_ids:
            self._delay_buffers[cav_id] = {}
            for target in self.attack_targets:
                # 初始化延迟缓冲区，填充值为0.0
                self._delay_buffers[cav_id][target] = [0.0] * self.delay_steps
    
    def _build_observation(self) -> np.ndarray:
        """
        构建简洁的 CAV 观测 (N_CAV, 4)：
        [综合速度差, 综合加速度差, 时间车头距, 当前加速度]
        """
        if not self.sim or not self.sim.vehicles:
            return np.zeros((len(self.cav_ids), 4), dtype=np.float32)

        # 1. 预取数据
        predicted_states = self.get_predicted_hv_states()
        cav_p_values = self.get_cav_p_values()
        
        obs_list = []
        
        for cav_id in self.cav_ids:
            # 获取当前车辆信息
            cav_idx = next(i for i, v in enumerate(self.sim.vehicles) if v.vehicle_id == cav_id)
            curr_veh = self.sim.vehicles[cav_idx]
            
            # --- 2. 获取三个关键交通参与者的信息 ---
            
            # (A) 前方最近的 CAV (Lead CAV)
            lead_cav, lead_cav_idx = None, -1
            for i in range(cav_idx - 1, -1, -1):
                if self.sim.vehicles[i].is_cav:
                    lead_cav = self.sim.vehicles[i]
                    lead_cav_idx = i
                    break
            
            # (B) 紧邻前车 (Immediate Leader)
            lead_veh = self.sim.vehicles[cav_idx - 1] if cav_idx > 0 else None
            
            # (C) 预测的中间 HV 状态
            pred_state = predicted_states.get(cav_id)

            # --- 3. 提取状态差异 ---
            
            # A. vs Lead CAV
            d_v_cav, d_a_cav = 0.0, 0.0
            p_val = 1.0
            hv_count = 0 
            
            if lead_cav:
                # 获取网络状态 (可能被攻击)
                s = self.get_cav_state(lead_cav.vehicle_id, "cav", cav_id)
                # 处理丢包 (None) -> 0.0
                l_speed = s.get('speed') or 0.0
                l_accel = s.get('acceleration') or 0.0
                
                d_v_cav = curr_veh.speed - l_speed
                d_a_cav = curr_veh.acceleration - l_accel
                
                # 获取可信度 p
                p_val = cav_p_values.get(lead_cav.vehicle_id, 1.0)
                if self.force_lead_cav_p_one:
                    p_val = 1.0
                    
                # 计算间隔车辆数 (索引差)
                hv_count = cav_idx - lead_cav_idx

            # B. vs Immediate Leader (如果是HV才算，如果是CAV则归0，避免重复/冲突)
            d_v_hv, d_a_hv = 0.0, 0.0
            rel_dist = 1e6 # 默认大距离
            
            if lead_veh:
                rel_dist = lead_veh.x_front - lead_veh.length - curr_veh.x_front
                if not lead_veh.is_cav:
                    d_v_hv = curr_veh.speed - lead_veh.speed
                    d_a_hv = curr_veh.acceleration - lead_veh.acceleration

            # C. vs Predicted HV
            d_v_pred, d_a_pred = 0.0, 0.0
            if pred_state:
                d_v_pred = curr_veh.speed - pred_state['speed']
                d_a_pred = curr_veh.acceleration - pred_state['acceleration']

            # --- 4. 融合计算 ---
            
            # 权重因子
            weight = (0.5 ** hv_count) * p_val
            
            # 综合速度/加速度差
            # Formula: 1*HV + 2*w*Pred + w*CAV
            cond_speed = 1.0 * d_v_hv + 2 * weight * d_v_pred + weight * d_v_cav
            cond_accel = 1.0 * d_a_hv + 2 * weight * d_a_pred + weight * d_a_cav
            
            # 时间车头距 (Time Headway)
            v_safe = max(0.1, curr_veh.speed)
            thw = (rel_dist - 5.0) / v_safe - 1.0
            
            obs_list.append([cond_speed, cond_accel, thw, curr_veh.acceleration])
            
        return np.array(obs_list, dtype=np.float32)

    def _reward_multi(self, obs, actions) -> float:
        """计算多 CAV 的聚合奖励，使用观测值的平方和来计算奖励，并加入加加速度惩罚项
        x = 观测值的平方和（不包括时间车头距误差） + 时间车头距惩罚项 + 加加速度惩罚项
        r = e^(-x)
        obs=[速度差、加速度差和时间车头距]
        
        注意：只考虑第二辆CAV（索引为1）的观测值，忽略第一辆CAV的观测值
        时间车头距惩罚：只有当时间车头距小于0.5时才给予惩罚
        """
        # 检查是否发生碰撞，如果发生碰撞则添加惩罚
        collision_penalty = 0.0
        if self.sim.terminated and self.sim.collision_info is not None:
            collision_penalty = -1.0  # 碰撞惩罚值
        
        # 只使用第二辆CAV（索引为1）的观测值，忽略第一辆CAV（索引为0）的观测值
        if len(obs) > 10:
            obs_for_reward = obs[1:]  # 从第二辆CAV开始的所有观测值
        else:
            obs_for_reward = obs
        
        # 计算所有CAV的奖励列表
        rewards = []
        
        # 用于统计jerk均值
        jerk_values = []
        
        # 获取动作列表
        actions_list = list(actions.values())
        
        # 为每个CAV计算奖励
        for i in range(len(obs_for_reward)):
            # 计算观测值
            cav_obs = obs_for_reward[i]
            speed_diff = cav_obs[0]  # 速度差
            accel_diff = cav_obs[1]  # 加速度差
            time_headway_e = cav_obs[2]  # 时间车头距差距
            current_accel = cav_obs[3]  # 当前加速度
            
            # 获取对应的动作
            if i < len(actions_list):
                action = actions_list[i]
            else:
                action = 0.0
            
            # 计算观测值的平方和（不包括时间车头距）
            x = abs(speed_diff) + abs(accel_diff)
            
            # 时间车头距惩罚项：只有当时间车头距小于0.5时才给予惩罚
            time_headway_penalty = 0.0
            if abs(time_headway_e) > 0.05:
                # 当时间车头距误差大于于0.2时，惩罚
                time_headway_penalty = abs(time_headway_e) 
            
            # 计算加加速度（jerk）惩罚项
            jerk_penalty = 0.0
            current_action = 0
            current_cav_id = self.cav_ids[i]  # 获取当前CAV的ID
            if current_cav_id in self._cav_actions_history and len(self._cav_actions_history[current_cav_id]) >= 2:
                # 计算当前动作与上一动作的差值（加加速度）
                current_action = action
                previous_action = self._cav_actions_history[current_cav_id][-2]  # 倒数第二个动作
                jerk = current_action - previous_action
                # 记录jerk值用于计算均值
                jerk_values.append(abs(jerk))
                # 对加加速度进行惩罚
                if abs(jerk)> 0.1:  # 0.1是阈值，可根据需要调整
                    jerk_penalty = 10 * (abs(jerk)-0.1)
                
            # 添加uCBF惩罚项，希望uCBF趋于0
            ucbf_penalty = 0.0
            if hasattr(self, '_cbf_u_values') and current_cav_id in self._cbf_u_values:
                # 获取最新的u_cbf值
                if self._cbf_u_values[current_cav_id]:
                    latest_u_cbf = self._cbf_u_values[current_cav_id][-1]['u_cbf']
                    # uCBF惩罚项，希望uCBF趋于0
                    ucbf_penalty = abs(latest_u_cbf)
                    # print('ucbf_penalty:', ucbf_penalty)
            
            # 总的奖励计算项
            x = x + 1*time_headway_penalty + ucbf_penalty*5+0*jerk_penalty+abs(current_accel)
            
            # 计算奖励 r = e^(-x)
            reward = np.exp(-0.1 * x)

            
            # 添加碰撞惩罚
            reward += collision_penalty
            
            rewards.append(float(reward))
        
        # 计算并存储jerk均值到当前回合记录中
        if jerk_values:
            avg_jerk = np.mean(jerk_values)
            # 将当前时间步的jerk均值添加到当前回合的记录中
            self._current_episode_jerk_values.append(avg_jerk)
            
        # 返回所有奖励的平均值（保持与原来相同的返回类型）
        return rewards
    
    def get_episode_jerk_means(self):
        """
        获取所有回合的jerk均值
        
        Returns:
            List[float]: 每个回合的jerk均值列表
        """
        return self._episode_jerk_means.copy()
    
    def get_current_episode_jerk_mean(self):
        """
        获取当前回合的jerk均值
        
        Returns:
            float: 当前回合的jerk均值
        """
        if self._current_episode_jerk_values:
            return float(np.mean(self._current_episode_jerk_values))
        return 0.0

    def _update_cav_safety_metrics(self):
        """
        更新每个CAV的安全指标
        指标计算公式：(前车HV和前一辆CAV之间的距离) / (4.4 + 前车HV速度)
        注意：gap是CAV前车(HV)和CAV前一辆CAV之间的距离
        """
        if not self.sim or not self.sim.vehicles:
            return

        for cav_id in self.cav_ids:
            # 找到CAV索引
            cav_idx = next((idx for idx, v in enumerate(self.sim.vehicles) if v.vehicle_id == cav_id), None)
            
            # 如果没有找到或者没有前车，跳过
            if cav_idx is None or cav_idx == 0:
                continue

            # 获取当前CAV的前车 (Immediate Leader)
            leader = self.sim.vehicles[cav_idx - 1]
            
            # 寻找前一辆CAV (Preceding CAV)
            lead_cav = None
            for i in range(cav_idx - 1, -1, -1):
                if self.sim.vehicles[i].is_cav:
                    lead_cav = self.sim.vehicles[i]
                    break
            
            # 如果没有找到前一辆CAV，则不记录
            if lead_cav is None:
                continue

            # 计算Gap
            # 如果前一辆CAV就是前车本身（即CAV跟CAV），则gap为0
            if lead_cav.vehicle_id == leader.vehicle_id:
                gap = 0.0
            else:
                # 计算前一辆CAV和前车HV之间的Gap
                # Gap = LeadCAV尾部 - Leader头部
                gap = lead_cav.x_front - lead_cav.length - leader.x_front
            
            # 获取前车速度
            leader_speed = leader.speed
            
            # 计算指标
            # 避免除以零或极小值
            denominator = 4.4 + leader_speed
            if abs(denominator) < 1e-6:
                metric_value = 0.0 
            else:
                metric_value = gap / denominator

            # 初始化该CAV的指标记录
            if cav_id not in self.cav_safety_metrics:
                self.cav_safety_metrics[cav_id] = {
                    'values': [],
                    'max_value': -float('inf'),
                    'min_value': float('inf')
                }
            
            # 更新记录
            metrics = self.cav_safety_metrics[cav_id]
            metrics['values'].append(metric_value)
            
            if metric_value > metrics['max_value']:
                metrics['max_value'] = metric_value
            if metric_value < metrics['min_value']:
                metrics['min_value'] = metric_value

    def get_cav_safety_metrics(self) -> Dict[int, Dict[str, Any]]:
        """
        获取CAV的安全指标记录
        
        Returns:
            Dict: {cav_id: {'values': [], 'max_value': float, 'min_value': float}}
        """
        return self.cav_safety_metrics
