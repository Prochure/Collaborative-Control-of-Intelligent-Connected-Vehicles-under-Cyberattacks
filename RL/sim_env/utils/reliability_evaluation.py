# 实时滑动窗口篡改检测器脚本（中文注释）
# 功能：每一时刻接收车辆的三维信息（位置、速度、加速度），基于卡尔曼滤波和滑动窗口内历史残差分布（均值+协方差）
#       计算马氏距离并给出是否篡改判断及置信度（p-value）。
# 特点：
# - 使用前面实现的简单3D卡尔曼滤波器作为预测器/更新器
# - 使用可配置的窗口大小（window_size）来维护历史残差统计（在线更新）
# - 对于p-value使用 scipy（若可用），否则提供基于卡方近似的简易映射（保守）

import numpy as np
from collections import deque
from numpy.linalg import inv, pinv
import math

# 尝试导入 scipy 以便计算精确的 p-value（若不可用则使用后备近似）
try:
    from scipy.stats import chi2
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# 复用 KalmanFilter3D 定义（简化版）
class KalmanFilter3D:
    def __init__(self, dt=1.0, process_var=1e-4, meas_var_pos=1e-2, meas_var_vel=1e-3, meas_var_acc=1e-4):
        self.dt = dt
        self.x = np.zeros((3,))
        self.P = np.eye(3) * 1.0
        self.A = np.array([
            [1.0, dt, 0.5 * dt**2],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0]
        ])
        self.H = np.eye(3)
        self.Q = np.diag([process_var, process_var, process_var * 0.1])
        self.R = np.diag([meas_var_pos, meas_var_vel, meas_var_acc])
        self.I = np.eye(3)
    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x.copy(), self.P.copy()
    def update(self, z):
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        try:
            S_inv = inv(S)
        except Exception:
            S_inv = pinv(S)
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P @ (self.I - K @ self.H).T + K @ self.R @ K.T
        return y.copy(), S.copy()

# 计算马氏距离
def mahalanobis_distance_squared(residual, cov, regularize=1e-8):
    cov_reg = cov + np.eye(cov.shape[0]) * regularize
    try:
        cov_inv = inv(cov_reg)
    except Exception:
        cov_inv = pinv(cov_reg)
    md2 = float(residual.T @ cov_inv @ residual)
    md = math.sqrt(md2) if md2 >= 0 else float('nan')
    return md, md2

# 主检测器类：维护滑动窗口统计并逐点输出检测结果
class SlidingWindowTamperDetector:
    def __init__(self, window_size=50, dt=1.0, chi2_alpha=0.01):
        """
        window_size: 用于估计历史残差分布的滑动窗口大小（至少应 >= 5）
        dt: 时间步长（用于卡尔曼滤波器）
        chi2_alpha: 用于阈值的显著性水平（默认0.01）
        """
        self.window_size = max(5, int(window_size))
        self.kf = KalmanFilter3D(dt=dt)
        self.residual_window = deque(maxlen=self.window_size)  # 存储最近 window_size 个残差向量
        self.res_mean = np.zeros(3)
        self.res_cov = np.eye(3) * 1e-6  # 初始化为小协方差避免奇异
        self.df = 3  # 自由度（残差维度）
        self.alpha = chi2_alpha
        # 计算卡方临界值（若scipy可用则精确）
        if SCIPY_AVAILABLE:
            self.chi2_threshold = float(chi2.ppf(1 - self.alpha, self.df))
        else:
            # 后备值（df=3, alpha=0.01约等于11.3449）
            self.chi2_threshold = 11.3449

    def initialize_with_history(self, history_array):
        """
        使用历史观测数组（形状 (N,3)）进行初始化。
        会用前N点依次更新卡尔曼滤波器并收集残差，最终计算窗口统计。
        """
        history = np.asarray(history_array, dtype=float)
        assert history.ndim == 2 and history.shape[1] == 3, "history_array 必须是 Nx3 数组"
        # 用第一条观测初始化滤波器状态
        self.kf.x = history[0].copy()
        self.residual_window.clear()
        for i in range(1, history.shape[0]):
            z = history[i]
            self.kf.predict()
            y, S = self.kf.update(z)
            # 将残差加入窗口
            self.residual_window.append(y)
        self._recompute_stats()

    def _recompute_stats(self):
        """从当前窗口（self.residual_window）重新计算均值和协方差"""
        if len(self.residual_window) == 0:
            self.res_mean = np.zeros(3)
            self.res_cov = np.eye(3) * 1e-6
            return
        arr = np.vstack(self.residual_window)
        self.res_mean = arr.mean(axis=0)
        # ddof=1 使用无偏估计；如果样本量为1则退化到小协方差
        if arr.shape[0] > 1:
            cov = np.cov(arr, rowvar=False, ddof=1)
            # 为数值稳定性添加微小对角正则
            self.res_cov = cov + np.eye(3) * 1e-9
        else:
            # 样本太少时使用很小的协方差，避免矩阵奇异
            self.res_cov = np.eye(3) * 1e-6

    def process_point(self, new_point):
        """
        处理一条新的观测点，并返回检测结果（马氏距离只考虑位置和速度两个维度）
        """
        z = np.asarray(new_point, dtype=float).reshape(3,)
        # 1) 预测
        x_prior, P_prior = self.kf.predict()
        # 2) 残差（测量 - 预测）
        residual = z - x_prior

        # ⚠️ 只取位置和速度的残差 (2维)
        residual_2d = residual[:2]
        centered = residual_2d - self.res_mean[:2]

        # 3) 马氏距离 (基于 2维协方差)
        md, md2 = mahalanobis_distance_squared(centered, self.res_cov[:2, :2])

        # 4) 计算 p-value，自由度改为 2
        if SCIPY_AVAILABLE:
            p_value = 1.0 - chi2.cdf(md2, df=2)
        else:
            p_value = math.exp(-md2 / 2.0)

        # 5) 更新卡尔曼滤波器
        self.kf.update(z)

        # 6) 窗口更新（依旧保存完整3维残差，但统计时只用前2维）
        self.residual_window.append(residual.copy())
        arr = np.vstack(self.residual_window)
        self.res_mean = arr.mean(axis=0)
        cov = np.cov(arr[:, :2], rowvar=False, ddof=1)  # 只统计前两列
        self.res_cov = np.eye(3) * 1e-6
        self.res_cov[:2, :2] = cov + np.eye(2) * 1e-9

        # 7) 判定
        tampered = bool(md2 > self.chi2_threshold)  # 阈值需改为 df=2 的卡方阈值

        return {
            'is_tampered': tampered,
            'md': md,
            'md2': md2,
            'p_value': p_value,
            'residual': residual,
        }

# --------------------------- 示例（模拟在线流）---------------------------
def main():
    # 为演示，先生成一段正常轨迹（使用简单模拟）
    def simulate_motion_simple(n_steps=200, dt=1.0, seed=0):
        rng = np.random.default_rng(seed)
        pos = 0.0; vel = 5.0; acc = 0.1
        jerk_sigma = 0.02
        meas_noise_pos = 0.05; meas_noise_vel = 0.02; meas_noise_acc = 0.005
        data = []
        for t in range(n_steps):
            meas_pos = pos + rng.normal(0, meas_noise_pos)
            meas_vel = vel + rng.normal(0, meas_noise_vel)
            meas_acc = acc + rng.normal(0, meas_noise_acc)
            data.append([meas_pos, meas_vel, meas_acc])
            pos = pos + vel * dt + 0.5 * acc * dt**2
            vel = vel + acc * dt
            acc = acc + rng.normal(0, jerk_sigma)
        return np.array(data)

    # 创建模拟数据流
    stream = simulate_motion_simple(200, seed=1)
    # 在流中插入一次明显的篡改（比如在第100步，把位置改大）
    tamper_index = 100
    stream[tamper_index, 0] += 8.0  # 位置突跳 +8m（明显异常）

    # 初始化检测器并用前面的一部分历史数据初始化统计（warm-up）
    detector = SlidingWindowTamperDetector(window_size=10, dt=1.0, chi2_alpha=0.05)
    warmup_history = stream[0:30]  # 用前30步作为初始历史
    detector.initialize_with_history(warmup_history)

    # 模拟实时接收后续观测并检测
    print("time\tis_tampered\tmd2\tp_value\t\tresidual(position,vel,acc)")
    for t in range(30, 140):
        meas = stream[t]
        result = detector.process_point(meas)
        # 仅在可疑或每隔若干步打印
        if result['is_tampered'] or (t % 10 == 0):
            print(f"{t}\t{result['is_tampered']}\t{result['md2']:.2f}\t{result['p_value']:.4g}\t{np.round(result['residual'],4)}")

    # 展示检测器在篡改点附近的输出窗口（前后各几步）
    print("\n-- 篡改点周边详细输出 --")
    detector2 = SlidingWindowTamperDetector(window_size=40, dt=1.0, chi2_alpha=0.1)
    detector2.initialize_with_history(stream[0:30])
    for t in range(30, tamper_index+4):
        res = detector2.process_point(stream[t])
        if tamper_index-3 <= t <= tamper_index+3:
            print(f"t={t}  is_tampered={res['is_tampered']}  md2={res['md2']:.2f}  p={res['p_value']:.4g}  residual={np.round(res['residual'],4)}")

if __name__ == "__main__":
    main()
