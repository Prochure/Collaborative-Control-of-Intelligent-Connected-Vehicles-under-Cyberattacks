import numpy as np
from scipy.stats import chi2

class RealTimeKFDetector3Full:
    """
    三阶状态（位置、速度、加速度）的实时卡尔曼滤波 + 两步检验可信度检测
    状态与观测均为:
        x = [position; velocity; acceleration]
    系统方程:
        x_{k+1} = F x_k + B u_k + w_k
        z_k     = H x_k + v_k

    连续离散化模型:
        x(t+T) = x + T v + 0.5 T^2 a
        v(t+T) = v + T a
        a(t+T) = (1 - T/ε) a + (T κ / ε) u
    """

    def __init__(self, T, eps, kappa, Q, R, x0=None, P0=None,
                 alpha=0.05, reset_P_scale=10.0):
        self.T = T
        self.eps = eps
        self.kappa = kappa

        # 状态维度和观测维度均为3
        self.dim_x = 3
        self.dim_z = 3

        # 系统矩阵
        self.F = np.array([
            [1.0, T, 0.5 * T**2],
            [0.0, 1.0, T],
            [0.0, 0.0, 1.0 - T / eps]
        ])
        self.B = np.array([[0.0], [0.0], [T * kappa / eps]])
        self.H = np.eye(3)  # 观测为全状态

        self.Q = Q.copy()
        self.R = R.copy()
        self.alpha = alpha
        self.reset_P_scale = reset_P_scale

        # 初始状态与协方差
        self.x_est = np.zeros((3, 1)) if x0 is None else x0.copy()
        self.P = np.eye(3) if P0 is None else P0.copy()

        # 用于两步检验
        self.prev_x_est = None
        self.prev_P = None
        self.prev_u = None

        self.step = 0

    def update(self, z, u_reported):
        """
        单步更新 + 两步检验
        Args:
            z: np.array shape (3,1), 当前观测 [pos, vel, acc]
            u_reported: 控制输入 (float)
        Returns:
            info: dict 诊断信息
        """
        # ---------- 1. KF预测 ----------
        x_pred = self.F @ self.x_est + self.B * u_reported
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # ---------- 2. KF残差统计 ----------
        r = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        invS = np.linalg.inv(S)
        D2 = float((r.T @ invS @ r).item())
        p_value_kf = 1 - chi2.cdf(D2, df=3)
        trust_score = p_value_kf

        # ---------- 3. 两步检验 ----------
        two_step_pass = False
        p_two = None
        if self.prev_x_est is not None and self.prev_P is not None and self.prev_u is not None:
            x_pred_two = self.F @ self.prev_x_est + self.B * self.prev_u
            P_pred_two = self.F @ self.prev_P @ self.F.T + self.Q
            r_two = z - self.H @ x_pred_two
            S_two = self.H @ P_pred_two @ self.H.T + self.R
            invS_two = np.linalg.inv(S_two)
            D2_two = float((r_two.T @ invS_two @ r_two).item())
            p_two = 1 - chi2.cdf(D2_two, df=3)
            two_step_pass = p_two > self.alpha

        # ---------- 4. 异常处理逻辑 ----------
        reset = False
        used_measurement = True

        if two_step_pass and p_value_kf < self.alpha:
            # 两步正常但KF异常 → 复位
            self.x_est = z.copy()
            self.P = np.eye(3) * self.reset_P_scale
            reset = True
            used_measurement = False

        elif two_step_pass and p_value_kf >= self.alpha:
            # 正常更新
            K = P_pred @ self.H.T @ invS
            self.x_est = x_pred + K @ r
            self.P = (np.eye(3) - K @ self.H) @ P_pred

        elif (not two_step_pass) and p_value_kf < self.alpha:
            # 双重异常 → 拒绝观测
            self.x_est = x_pred
            self.P = P_pred
            used_measurement = False

        else:
            # 其他情况（例如两步不通过但KF正常）
            K = P_pred @ self.H.T @ invS
            self.x_est = x_pred + K @ r
            self.P = (np.eye(3) - K @ self.H) @ P_pred

        # ---------- 5. 保存用于下一次两步检验 ----------
        self.prev_x_est = self.x_est.copy()
        self.prev_P = self.P.copy()
        self.prev_u = u_reported
        self.step += 1

        # ---------- 6. 返回诊断信息 ----------
        return {
            "step": self.step,
            "trust_score": trust_score,
            "p_value_kf": p_value_kf,
            "p_two": p_two,
            "two_step_pass": two_step_pass,
            "reset": reset,
            "used_measurement": used_measurement,
            "estimate": self.x_est.copy(),
            "residual": r.flatten()
        }
