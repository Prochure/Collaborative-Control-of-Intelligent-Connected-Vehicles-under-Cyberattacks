import numpy as np
import cvxpy as cp
from typing import Dict, Any, Tuple, Optional
import os

# 禁用OSQP的详细输出
os.environ['OSQP_VERBOSITY'] = '0'
import os
os.environ["CVXPY_VERBOSE"] = "0"
os.environ["OSQP_ENABLE_PRINTING"] = "false"


def solve_cbf_qp(state: Dict[str, float], u_rl: float, params: Dict[str, float]) -> Tuple[float, bool, Dict[str, Any]]:
    """
    使用CBF（Control Barrier Function）求解二次规划问题，确保车辆安全运行
    
    Args:
        state: 状态字典，包含以下键值:
            - dx_1: 与前车的相对距离
            - dx_n: 与后车的相对距离
            - v: 当前车速
            - a: 当前加速度
            - v_prev: 前车速度
            - v_n: 后车速度
        u_rl: RL控制器输出的期望加速度
        params: CBF参数字典，包含:
            - tau_i: 时间常数参数
            - tau_star: 时间常数参数
            - eps: 三阶动力学模型的时间常数参数
            - kappa: 三阶动力学模型的增益参数
            - rho1: CBF参数
            - rho2: CBF参数
            - u_min: 最小加速度限制
            - u_max: 最大加速度限制
            - v_max: 最大速度限制
            - T: 仿真时间步长
    
    Returns:
        Tuple[float, bool, Dict[str, Any]]: (应用的加速度, 是否可行, 附加信息)
    """
    try:
        # 解包状态变量并进行数值稳定性检查
        dx_1 = max(0.1, float(state['dx_1']))        # 与前车的相对距离 Delta x_{i,i-1}
        dx_n = max(0.1, float(state['dx_n']))        # 与后车的相对距离 Delta x_{i,i-n}
        v_i = max(0.0, float(state['v']))            # 当前车速
        a_i = float(state['a'])                      # 当前加速度
        v_i_1 = float(state.get('v_prev', v_i))      # 前车速度 v_{i-1}
        v_i_n = float(state.get('v_n', v_i))         # 后车速度 v_{i-n}

        # 解包参数并进行数值稳定性检查
        tau = max(0.1, float(params['tau_i']))
        tau_star = max(0.1, float(params['tau_star']))
        eps = max(1e-6, float(params['eps']))
        kappa = max(1e-6, float(params['kappa']))
        rho1 = max(0.01, min(2.0, float(params['rho1'])))  # 限制在合理范围内
        rho2 = max(0.01, min(2.0, float(params['rho2'])))  # 限制在合理范围内
        u_min = float(params['u_min'])
        u_max = float(params['u_max'])
        v_max = max(1.0, float(params['v_max']))
        T = max(1e-6, float(params['T']))
        u_rl = float(u_rl)

        # CBF量计算 (基于论文中的解析表达式)
        # 对于 h1:
        h1 = dx_1 - tau * v_i
        Lf_h1 = v_i_1 - v_i - tau * a_i                 # 论文中的 L_f h1
        Lf2_h1 = (tau * eps - 1.0) * a_i               # 论文中的 L_f^2 h1
        Lg_Lf_h1 = - (kappa * tau) / eps               # 论文中的 L_g L_f h1

        # 对于 h2:
        h2 = dx_n - tau_star * v_i
        Lf_h2 = v_i_n - v_i - tau_star * a_i
        Lf2_h2 = (tau_star * eps - 1.0) * a_i
        Lg_Lf_h2 = - (kappa * tau_star) / eps

        # 构建变量 u_cbf (标量)
        u_cbf = cp.Variable(1)

        # 构建约束条件
        constraints = []

        # CBF约束: Lf2_h + LgLf_h * (u_rl + u_cbf) + 2*rho*Lf_h + rho^2 * h >= 0
        # 重新排列为 a* u_cbf + b >= 0 的形式
        a1 = Lg_Lf_h1
        b1 = Lf2_h1 + Lg_Lf_h1 * u_rl + 2.0 * rho1 * Lf_h1 + (rho1**2) * h1
        constraints.append(a1 * u_cbf + b1 >= 0)

        # a2 = Lg_Lf_h2
        # b2 = Lf2_h2 + Lg_Lf_h2 * u_rl + 2.0 * rho2 * Lf_h2 + (rho2**2) * h2
        # constraints.append(a2 * u_cbf + b2 >= 0)

        # 物理边界约束: u = u_rl + u_cbf
        u_eff_max = max(u_max, (v_max - v_i)/T)
        # 确保边界合理
        u_eff_max = max(u_min + 1e-6, u_eff_max)
        constraints.append(u_rl + u_cbf >= u_min)
        constraints.append(u_rl + u_cbf <= u_eff_max)

        # 可选: 可行性约束 (引理1)。 
        # 我们可以将其作为软约束添加或事先检查。为简单起见，这里跳过显式的引理。

        # QP目标: 最小化 norm(u_cbf) 即最小化平方
        objective = cp.Minimize(cp.sum_squares(u_cbf))

        # 创建并求解优化问题
        prob = cp.Problem(objective, constraints)

        # 尝试多种求解器以提高鲁棒性
        solvers_to_try = [cp.ECOS, cp.SCS]
        solution_found = False
        last_error = None
        
        for solver in solvers_to_try:
            try:
                if solver == cp.OSQP:
                    prob.solve(solver=solver, warm_start=True, max_iter=10000, verbose=False)
                elif solver == cp.ECOS:
                    prob.solve(solver=solver, max_iters=10000, verbose=False)
                else:
                    prob.solve(solver=solver, max_iters=10000, verbose=False)
                solution_found = u_cbf.value is not None and not np.isnan(u_cbf.value[0])
                if solution_found:
                    break
            except Exception as e:
                last_error = str(e)
                continue

        # 检查解决方案
        if not solution_found or u_cbf.value is None or np.isnan(u_cbf.value[0]):
            # 不可行 -> 备用方案: 将 u_rl 投影到边界内(夹紧)，或使用松弛
            u_applied = float(np.clip(u_rl, u_min, u_eff_max))
            return u_applied, False, {'u_cbf': 0.0, 'error': last_error or 'QP solution not found'}

        u_applied = float(u_rl + u_cbf.value[0])
        # 确保结果在物理边界内
        u_applied = float(np.clip(u_applied, u_min, u_eff_max))
        feasible = True
        # print('u_cbf:', u_cbf.value[0],u_rl)
        return u_applied, feasible, {'u_cbf': float(u_cbf.value[0])}
        
    except Exception as e:
        # 如果出现任何异常，使用保守的备用方案
        u_min = float(params.get('u_min', -3.0))
        u_max = float(params.get('u_max', 3.0))
        v_max = float(params.get('v_max', 33.33))
        T = float(params.get('T', 0.1))
        v_i = max(0.0, float(state.get('v', 0.0)))
        u_eff_max = max(u_max, (v_max - v_i)/T)
        u_applied = float(np.clip(u_rl, u_min, u_eff_max))
        return u_applied, False, {'u_cbf': 0.0, 'error': str(e)}


def compute_cbf_state_from_observation(obs: np.ndarray, cav_idx: int, vehicles: list) -> Dict[str, float]:
    """
    从观测值计算CBF所需的状态信息
    
    Args:
        obs: 观测数组 (K, 9) 或 (K, 4)
        cav_idx: 当前CAV的索引
        vehicles: 车辆列表
        
    Returns:
        CBF状态字典
    """
    # 确保观测是9维的
    if obs.shape[1] == 4:
        # 如果是浓缩观测，需要重构9维观测
        raise NotImplementedError("需要从浓缩观测重构9维观测")
    
    # 获取当前CAV的观测
    cav_obs = obs[cav_idx]
    
    # 从9维观测中提取信息
    # [与前方CAV的速度差, 与前方CAV的加速度差, 与前方CAV的相对位置,
    #  与前方HV的速度差, 与前方HV的加速度差, 与前方HV的相对位置,
    #  与推测的HVs的速度差, 与推测的HVs的加速度差, 与推测的HVs的相对位置]
    
    v_diff_lead_cav = cav_obs[0]    # 与前方CAV的速度差
    a_diff_lead_cav = cav_obs[1]    # 与前方CAV的加速度差
    dx_lead_cav = cav_obs[2]        # 与前方CAV的相对位置
    
    v_diff_lead_hv = cav_obs[3]     # 与前方HV的速度差
    a_diff_lead_hv = cav_obs[4]     # 与前方HV的加速度差
    dx_lead_hv = cav_obs[5]         # 与前方HV的相对位置
    
    # 获取当前车辆信息
    current_vehicle = vehicles[cav_idx]
    v_i = current_vehicle.speed
    a_i = current_vehicle.acceleration
    
    # 计算前车和后车速度
    v_prev = v_i + v_diff_lead_cav  # 前车速度
    v_n = 0.0  # 后车速度，需要从其他信息获取
    
    # 构建状态字典，确保数值稳定性
    state = {
        'dx_1': max(0.1, float(dx_lead_hv)),      # 与前车(前方HV)的相对距离
        'dx_n': 50.0,            # 与后车的相对距离(假设值)
        'v': max(0.0, float(v_i)),                # 当前车速
        'a': float(a_i),                # 当前加速度
        'v_prev': float(v_prev),        # 前车速度
        'v_n': float(v_n)               # 后车速度
    }
    
    return state