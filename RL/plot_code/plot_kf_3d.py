import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt

# =========================
# 1. 绘图风格设置 
# =========================
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13
})

np.random.seed(42)

# =========================
# 2. 类定义 (动力学模型)
# =========================
class RealTimeKFDetector3Full:
    def __init__(self, T, eps, kappa, Q, R, x0=None, P0=None,
                 alpha=0.05, reset_P_scale=10.0, enable_dual_test=True):
        self.T = T
        self.eps = eps
        self.kappa = kappa

        self.dim_x = 3
        self.dim_z = 3

        # 系统矩阵 (保留用户 Step 34 中的定义)
        self.F = np.array([
            [1.0, T, 0.5*T**2],
            [0.0, 1.0, T],
            [0.0, 0.0, 1.0 - T/eps]
        ])
        self.B = np.array([[0.0], [0.0], [T*kappa/eps]])
        self.H = np.eye(3)

        self.Q = Q.copy()
        self.R = R.copy()
        self.alpha = alpha
        self.reset_P_scale = reset_P_scale
        self.enable_dual_test = enable_dual_test

        self.x_est = np.zeros((3,1)) if x0 is None else x0.copy()
        self.P = np.eye(3) if P0 is None else P0.copy()

        self.prev_z = None
        self.prev_u = None
        self.step = 0
        
        # 统计
        self.reset_count = 0
        self.anomaly_count = 0
        self.reset_steps = []

    def update(self, z, u_reported):
        # ---------- 1. KF 预测 ----------
        x_pred = self.F @ self.x_est + self.B * u_reported
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # ---------- 2. 残差 & p-value ----------
        r_kf = z - self.H @ x_pred
        S_kf = self.H @ P_pred @ self.H.T + self.R
        invS = np.linalg.inv(S_kf)
        D2_kf = float((r_kf.T @ invS @ r_kf).item())
        p_value_kf = 1 - chi2.cdf(D2_kf, df=3)

        # ---------- 3. 基于上一观测的两步检验 ----------
        two_step_pass = False
        if self.enable_dual_test:
            if self.prev_z is not None and self.prev_u is not None:
                # 预测观测: z_{k|k-1} = H * (F * z_{k-1} + B * u_{k-1})
                # 假设 z_{k-1} 近似 x_{k-1}
                x_pred_obs = self.F @ self.prev_z + self.B * self.prev_u
                r_obs = z - self.H @ x_pred_obs # 注意这里加上 H
                
                # 简化协方差为 R (参考 plot_verify_kf_detection.py)
                D2_obs = float((r_obs.T @ np.linalg.inv(self.R) @ r_obs).item())
                p_obs = 1 - chi2.cdf(D2_obs, df=3)
                two_step_pass = p_obs > self.alpha
            else:
                two_step_pass = True # 初始步默认通过

        reset = False
        used_measurement = True

        # ---------- 4. 判定逻辑 ----------
        if self.enable_dual_test and two_step_pass and p_value_kf < self.alpha:
            # 双重检验通过 但 KF 异常 -> 说明 KF 状态跑偏 -> 复位
            self.x_est = z.copy()
            self.P = np.eye(3) * self.reset_P_scale
            reset = True
            used_measurement = False
            self.reset_count += 1
            self.reset_steps.append(self.step)
            
            # 复位后重新计算 p_value_kf 用于记录
            r_kf = z - self.H @ self.x_est
            S_kf = self.H @ self.P @ self.H.T + self.R
            D2_kf = float((r_kf.T @ np.linalg.inv(S_kf) @ r_kf).item())
            p_value_kf = 1 - chi2.cdf(D2_kf, df=3)
            
        elif p_value_kf >= self.alpha:
            # 正常 KF 更新
            K = P_pred @ self.H.T @ invS
            self.x_est = x_pred + K @ r_kf
            self.P = (np.eye(3) - K @ self.H) @ P_pred
        else:
            # KF 异常 (且不满足复位条件) -> 拒绝观测
            self.x_est = x_pred
            self.P = P_pred
            used_measurement = False
            self.anomaly_count += 1

        # ---------- 5. 记录 ----------
        self.prev_z = z.copy()
        self.prev_u = u_reported
        self.step += 1

        return {
            "step": self.step,
            "p_value_kf": p_value_kf,
            "reset": reset,
            "used_measurement": used_measurement,
            "estimate": self.x_est.copy(),
            "residual": r_kf.flatten()
        }

# =========================
# 3. 实验设置 (参考 plot_verify_kf_detection.py)
# =========================
T = 0.1
steps = 100
eps = 0.5
kappa = 1.0

# 噪声参数
Q = np.diag([1e-4, 1e-3, 5e-3])
R = np.diag([1e-2, 1e-2, 5e-2])

# 攻击参数 random  continuous
attack_type = "continuous"
attack_probability = 0.1
continuous_attack_start = 40
continuous_attack_duration = 20

# 攻击幅度 (Pos, Vel, Acc)
# 攻击幅度 (Pos, Vel, Acc)
attack_magnitude = np.array([10.0, 3.0, 1.0])
attack_magnitude_u = 3.0 # 增大输入攻击幅度以模拟显著的网络攻击

# 生成平滑输入 u
def generate_smooth_input(num_steps):
    t = np.linspace(0, 4*np.pi, num_steps)
    u = 0.5 * np.sin(t) + 0.3 * np.sin(2*t)
    return u

u_seq = generate_smooth_input(steps)

# 确定攻击步骤
def determine_attack_steps(attack_type, steps, attack_probability, start, duration):
    if attack_type == "random":
        attack_steps = []
        for i in range(steps):
            if np.random.random() < attack_probability:
                attack_steps.append(i)
        return attack_steps
    elif attack_type == "continuous":
        return list(range(start, min(start + duration, steps)))
    return []

attack_steps = determine_attack_steps(attack_type, steps, attack_probability, 
                                    continuous_attack_start, continuous_attack_duration)
print(f"攻击类型: {attack_type}")
print(f"攻击步骤: {attack_steps}")

# =========================
# 4. 仿真运行
# =========================
# 初始化两个检测器
detector_with = RealTimeKFDetector3Full(T, eps, kappa, Q, R, enable_dual_test=True)
detector_without = RealTimeKFDetector3Full(T, eps, kappa, Q, R, enable_dual_test=False)

# 初始状态
x_true = np.array([[0.0], [10.0], [0.0]])
detector_with.x_est = x_true.copy()
detector_without.x_est = x_true.copy()

# 数据存储
data_with = {'pos': [], 'vel': [], 'acc': [], 'p_val': []}
data_without = {'pos': [], 'vel': [], 'acc': [], 'p_val': []}
true_data = {'pos': [], 'vel': [], 'acc': []}

for k in range(steps):
    u = u_seq[k]
    
    # 1. 真实系统演化 (使用相同的 F, B)
    # 注意：这里使用 detector_with 的矩阵作为 Ground Truth 生成器
    process_noise = np.random.multivariate_normal(np.zeros(3), Q).reshape(3,1)
    x_true = detector_with.F @ x_true + detector_with.B * u 
    
    # 2. 生成观测
    meas_noise = np.random.multivariate_normal(np.zeros(3), R).reshape(3,1)
    z = detector_with.H @ x_true + meas_noise/2.5
    
    # 3. 施加攻击
    u_reported = u
    if k in attack_steps:
        # 观测攻击
        attack_vec = np.random.normal(0, 1, (3,1)) * attack_magnitude.reshape(3,1)
        z += attack_vec
        
        # 输入攻击
        u_reported += np.random.normal(0, attack_magnitude_u)
        
    # 4. 更新检测器
    # 4. 更新检测器
    out_with = detector_with.update(z, u_reported)
    out_without = detector_without.update(z, u_reported)
    
    # 5. 存储数据
    true_data['pos'].append(x_true[0,0])
    true_data['vel'].append(x_true[1,0])
    true_data['acc'].append(x_true[2,0])
    
    data_with['pos'].append(out_with['estimate'][0,0])
    data_with['vel'].append(out_with['estimate'][1,0])
    data_with['acc'].append(out_with['estimate'][2,0])
    data_with['p_val'].append(out_with['p_value_kf'])
    
    data_without['pos'].append(out_without['estimate'][0,0])
    data_without['vel'].append(out_without['estimate'][1,0])
    data_without['acc'].append(out_without['estimate'][2,0])
    data_without['p_val'].append(out_without['p_value_kf'])

# =========================
# 5. 数据处理与绘图
# =========================
steps_arr = np.arange(steps)

# 转换为 numpy 数组
for key in true_data: true_data[key] = np.array(true_data[key])
for key in data_with: data_with[key] = np.array(data_with[key])
for key in data_without: data_without[key] = np.array(data_without[key])

# 计算误差
err_pos_with = data_with['pos'] - true_data['pos']
err_vel_with = data_with['vel'] - true_data['vel']
err_acc_with = data_with['acc'] - true_data['acc']

err_pos_without = data_without['pos'] - true_data['pos']
err_vel_without = data_without['vel'] - true_data['vel']
err_acc_without = data_without['acc'] - true_data['acc']

fig, axes = plt.subplots(2, 2, figsize=(9, 8))
# 调整间距
fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.12, hspace=0.35, wspace=0.25)

def plot_subplot(ax, data_w, data_wo, ylabel, title_label, is_p_val=False):
    ax.plot(steps_arr, data_w, 'b-', label='DT-KF', linewidth=1.5)
    ax.plot(steps_arr, data_wo, 'm--', label='KF', linewidth=1.5)
    
    if is_p_val:
        ax.axhline(detector_with.alpha, color='r', linestyle='--', label=f'sl={detector_with.alpha}', linewidth=1.5)
        ax.set_ylabel('p-value')
    else:
        ax.axhline(0, color='k', linestyle=':', linewidth=1)
        ax.set_ylabel(ylabel)
        
    if attack_steps:
        y_min, y_max = ax.get_ylim()
        y_pos = y_min + (y_max - y_min) * 0.05
        ax.scatter(attack_steps, [y_pos]*len(attack_steps), color='red', marker='^', s=15, zorder=5, label='Attack' if not is_p_val else "")

    # 画出复位点 (红色圆圈)
    if not is_p_val and detector_with.reset_steps:
        reset_indices = np.array(detector_with.reset_steps)
        # 对应的数据点值
        reset_data = np.array(data_w)[reset_indices]
        ax.scatter(reset_indices, reset_data, s=50, facecolors='none', edgecolors='r', linewidth=1.5, label='Reset' if title_label=='(a)' else "", zorder=10)

    ax.set_xlabel('Time Step')
    ax.grid(True, alpha=0.3)
    if title_label == '(a)' or title_label == '(d)': # 仅在部分图显示图例避免遮挡
        ax.legend(fontsize=9, loc='center right' if is_p_val else 'best')
    
    ax.text(0.5, -0.2, title_label, transform=ax.transAxes, fontsize=14, fontweight='bold', ha='center', va='top')

plot_subplot(axes[0, 0], err_pos_with, err_pos_without, 'Position Error [m]', '(a)')
plot_subplot(axes[0, 1], err_vel_with, err_vel_without, 'Speed Error [m/s]', '(b)')
plot_subplot(axes[1, 0], err_acc_with, err_acc_without, 'Acceleration Error [m/s²]', '(c)')
plot_subplot(axes[1, 1], data_with['p_val'], data_without['p_val'], '', '(d)', is_p_val=True)

plt.savefig('kf_continuous_detection_results_3d_state_comparison.png', dpi=300, bbox_inches='tight')
print("Saved kf_detection_results_3d_state_comparison.png")

# =========================
# 6. 输出统计结果 (参考 plot_verify_kf_detection.py)
# =========================
def calc_mae(err): return np.mean(np.abs(err))

print("\n整体MAE结果:")
print("=" * 50)
print("考虑Dual-Test:")
print(f"  位置MAE: {calc_mae(err_pos_with):.4f} m")
print(f"  速度MAE: {calc_mae(err_vel_with):.4f} m/s")
print(f"  加速度MAE: {calc_mae(err_acc_with):.4f} m/s^2")
print(f"  复位次数: {detector_with.reset_count}")
print("不考虑Dual-Test:")
print(f"  位置MAE: {calc_mae(err_pos_without):.4f} m")
print(f"  速度MAE: {calc_mae(err_vel_without):.4f} m/s")
print(f"  加速度MAE: {calc_mae(err_acc_without):.4f} m/s^2")
print(f"  异常检测次数(拒绝更新): {detector_without.anomaly_count}")

# 保存到文件
with open("kf_random_detection_comparison_results.txt", "w", encoding="utf-8") as f:
    f.write("卡尔曼滤波检测结果比较：考虑Dual-Test vs 不考虑Dual-Test\n")
    f.write("=" * 60 + "\n")
    f.write(f"攻击类型: {attack_type}\n")
    f.write(f"攻击步骤: {attack_steps}\n\n")
    
    f.write("考虑Dual-Test:\n")
    f.write(f"  位置MAE: {calc_mae(err_pos_with):.4f} m\n")
    f.write(f"  速度MAE: {calc_mae(err_vel_with):.4f} m/s\n")
    f.write(f"  加速度MAE: {calc_mae(err_acc_with):.4f} m/s²\n")
    f.write(f"  复位次数: {detector_with.reset_count}\n\n")
    
    f.write("不考虑Dual-Test:\n")
    f.write(f"  位置MAE: {calc_mae(err_pos_without):.4f} m\n")
    f.write(f"  速度MAE: {calc_mae(err_vel_without):.4f} m/s\n")
    f.write(f"  加速度MAE: {calc_mae(err_acc_without):.4f} m/s²\n")
    f.write(f"  异常检测次数: {detector_without.anomaly_count}\n")

print("结果已保存到 kf_detection_comparison_results.txt")
