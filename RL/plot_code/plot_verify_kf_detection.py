import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 设置统一出版级绘图风格
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
# 系统参数
# =========================
dt = 0.1
steps = 100
F = np.array([[1, dt],
              [0, 1]])        # 状态转移矩阵 (位置, 速度)
B = np.array([[0.5*dt**2],
              [dt]])          # 控制输入矩阵 (加速度控制)
H = np.array([[1, 0],
              [0, 1]])        # 观测矩阵 (位置, 速度)

Q = np.array([[1e-4, 0],
              [0, 1e-4]])     # 过程噪声
R = np.array([[1e-2, 0],
              [0, 1e-2]])     # 观测噪声 (位置, 速度)

# =========================
# 生成平滑但有振幅的加速度输入
# =========================
def generate_smooth_acceleration(num_steps, base_freq=0.5, amplitude=2.0):
    """
    生成平滑但有振幅的加速度输入
    
    Args:
        num_steps: 步数
        base_freq: 基础频率
        amplitude: 振幅
    
    Returns:
        加速度序列
    """
    t = np.linspace(0, 4*np.pi, num_steps)
    
    # 组合多个正弦波以创建更复杂的平滑信号
    acc = (amplitude * np.sin(base_freq * t) + 
           0.5 * amplitude * np.sin(2 * base_freq * t) + 
           0.3 * amplitude * np.sin(3 * base_freq * t) +
           0.2 * amplitude * np.sin(5 * base_freq * t))
    
    # 添加一些随机噪声使信号更真实
    acc += np.random.normal(0, 0.1 * amplitude, num_steps)
    
    return acc

# 生成平滑加速度序列
smooth_accelerations = generate_smooth_acceleration(steps, base_freq=0.5, amplitude=2.5)

# =========================
# 攻击参数
# =========================
# 攻击类型: "random" (基于概率) 或 "continuous" (持续攻击)
attack_type = "random"  # 修改为持续攻击
attack_probability = 0.1    # 随机攻击概率
continuous_attack_start =40  # 持续攻击开始时间步
continuous_attack_duration = 20 # 持续攻击持续时间步数（延长攻击时间）

attack_magnitude_pos =3
attack_magnitude_vel = 2.0
attack_magnitude_acc = 1

alpha = 0.05  # 显著性水平

# =========================
# 初始化
# =========================
x_true = np.array([[100.0],
                   [10.0]])   # 真实状态
x_est_with_two_step = np.array([[100.0],
                               [10.0]])    # 考虑两步检验的估计状态
x_est_without_two_step = np.array([[100.0],
                                  [10.0]])    # 不考虑两步检验的估计状态
P_with_two_step = np.eye(2)
P_without_two_step = np.eye(2)

# =========================
# 确定攻击步骤
# =========================
def determine_attack_steps(attack_type, steps, attack_probability, continuous_attack_start, continuous_attack_duration):
    """
    根据攻击类型确定攻击步骤
    
    Args:
        attack_type: 攻击类型 ("random" 或 "continuous")
        steps: 总步数
        attack_probability: 随机攻击概率
        continuous_attack_start: 持续攻击开始时间步
        continuous_attack_duration: 持续攻击持续时间步数
    
    Returns:
        攻击步骤列表
    """
    if attack_type == "random":
        # 基于概率的随机攻击
        attack_steps = []
        for i in range(steps):
            if np.random.random() < attack_probability:
                attack_steps.append(i)
        return attack_steps
    elif attack_type == "continuous":
        # 持续一段时间的攻击
        return list(range(continuous_attack_start, min(continuous_attack_start + continuous_attack_duration, steps)))
    else:
        return []

# 确定攻击步骤
attack_steps = determine_attack_steps(attack_type, steps, attack_probability, continuous_attack_start, continuous_attack_duration)
print(f"攻击类型: {attack_type}")
print(f"攻击步骤: {attack_steps}")

# =========================
# 保存数据
# =========================
# 考虑两步检验的数据
true_pos_with = []
true_velocities = []
true_accelerations = []
pred_pos_with = []
pred_vel_with = []
meas_pos_with = []
p_values_kf_with = []
residuals_with = []
acc_inputs_reported_with = []
used_measurement_with = []
reset_points_with = []

# 不考虑两步检验的数据
true_pos_without = []
pred_pos_without = []
pred_vel_without = []
meas_pos_without = []
p_values_kf_without = []
residuals_without = []
acc_inputs_reported_without = []
used_measurement_without = []
reset_points_without = []

# =========================
# 前一观测及前一加速度（用于两步检验）
# =========================
prev_obs_with = None
prev_acc_with = None
prev_obs_without = None
prev_acc_without = None

# =========================
# 仿真循环
# =========================
for k in range(steps):
    # ---- 真实加速度 ----
    a_true = smooth_accelerations[k]

    # ---- 发送到估计器/网络传输的加速度（可能被攻击） ----
    a_reported_with = a_true
    a_reported_without = a_true
    if k in attack_steps:
        a_reported_with += np.random.normal(0, attack_magnitude_acc)
        a_reported_without += np.random.normal(0, attack_magnitude_acc)
    acc_inputs_reported_with.append(a_reported_with)
    acc_inputs_reported_without.append(a_reported_without)

    # ---- 真实状态更新 ----
    process_noise = np.random.multivariate_normal([0,0], Q).reshape(-1,1)
    x_true = F @ x_true + B * a_true + process_noise

    # ---- 观测 ----
    measurement_noise = np.random.multivariate_normal([0,0], R).reshape(-1,1)
    z = H @ x_true+measurement_noise/5
    if k in attack_steps:
        z[0,0] += np.random.normal(0, attack_magnitude_pos)
        z[1,0] += np.random.normal(0    , attack_magnitude_vel)

    # ===================================================================
    # 考虑两步检验的卡尔曼滤波处理
    # ===================================================================
    # ---- 卡尔曼预测 ----
    x_pred_kf_with = F @ x_est_with_two_step + B * a_reported_with
    P_pred_kf_with = F @ P_with_two_step @ F.T + Q

    # ---- 卡尔曼残差 & p值 ----
    r_kf_with = z - H @ x_pred_kf_with
    S_kf_with = H @ P_pred_kf_with @ H.T + R
    D2_kf_with = float((r_kf_with.T @ np.linalg.inv(S_kf_with) @ r_kf_with).item())
    p_value_kf_with = 1 - chi2.cdf(D2_kf_with, df=2)

    # ---- 两步检验 ----
    two_step_pass_with = False
    if prev_obs_with is not None and prev_acc_with is not None:
        x_pred_two_step_with = F @ prev_obs_with + B * prev_acc_with
        r_two_with = z - x_pred_two_step_with
        S_two_with = R
        D2_two_with = float((r_two_with.T @ np.linalg.inv(S_two_with) @ r_two_with).item())
        p_two_with = 1 - chi2.cdf(D2_two_with, df=2)
        two_step_pass_with = p_two_with > alpha

    # ---- 判定是否复位 ----
    if two_step_pass_with and p_value_kf_with < alpha:
        # 两步检验通过 + 卡尔曼异常 → 执行复位
        x_est_with_two_step = z.copy()
        P_with_two_step = np.eye(2) * 10.0
        used_measurement_with.append(False)
        reset_points_with.append(k)
        # 复位后重新计算当前残差和p值
        r_kf_with = z - H @ x_est_with_two_step
        S_kf_with = H @ P_with_two_step @ H.T + R
        D2_kf_with = float((r_kf_with.T @ np.linalg.inv(S_kf_with) @ r_kf_with).item())
        p_value_kf_with = 1 - chi2.cdf(D2_kf_with, df=2)
    elif p_value_kf_with > alpha:
        # 正常卡尔曼更新
        K_with = P_pred_kf_with @ H.T @ np.linalg.inv(S_kf_with)
        x_est_with_two_step = x_pred_kf_with + K_with @ r_kf_with
        P_with_two_step = (np.eye(2) - K_with @ H) @ P_pred_kf_with
        used_measurement_with.append(True)
    else:
        # 卡尔曼异常但两步检验未通过 → 不复位，只用预测
        x_est_with_two_step = x_pred_kf_with
        P_with_two_step = P_pred_kf_with
        used_measurement_with.append(False)

    # ---- 保存数据 ----
    prev_obs_with = z.copy()
    prev_acc_with = a_reported_with
    true_pos_with.append(x_true[0,0])
    true_velocities.append(x_true[1,0])
    true_accelerations.append(a_true)
    pred_pos_with.append(x_est_with_two_step[0,0])
    pred_vel_with.append(x_est_with_two_step[1,0])
    meas_pos_with.append(z[0,0])
    residuals_with.append(r_kf_with.flatten())
    p_values_kf_with.append(p_value_kf_with)

    # ===================================================================
    # 不考虑两步检验的卡尔曼滤波处理
    # ===================================================================
    # ---- 卡尔曼预测 ----
    x_pred_kf_without = F @ x_est_without_two_step + B * a_reported_without
    P_pred_kf_without = F @ P_without_two_step @ F.T + Q

    # ---- 卡尔曼残差 & p值 ----
    r_kf_without = z - H @ x_pred_kf_without
    S_kf_without = H @ P_pred_kf_without @ H.T + R
    D2_kf_without = float((r_kf_without.T @ np.linalg.inv(S_kf_without) @ r_kf_without).item())
    p_value_kf_without = 1 - chi2.cdf(D2_kf_without, df=2)

    # ---- 不考虑两步检验的处理 ----
    if p_value_kf_without > alpha:
        # 正常卡尔曼更新
        K_without = P_pred_kf_without @ H.T @ np.linalg.inv(S_kf_without)
        x_est_without_two_step = x_pred_kf_without + K_without @ r_kf_without
        P_without_two_step = (np.eye(2) - K_without @ H) @ P_pred_kf_without
        used_measurement_without.append(True)
    else:
        # 卡尔曼异常 → 不复位，只用预测
        x_est_without_two_step = x_pred_kf_without
        P_without_two_step = P_pred_kf_without
        used_measurement_without.append(False)

    # ---- 保存数据 ----
    prev_obs_without = z.copy()
    prev_acc_without = a_reported_without
    true_pos_without.append(x_true[0,0])
    pred_pos_without.append(x_est_without_two_step[0,0])
    pred_vel_without.append(x_est_without_two_step[1,0])
    meas_pos_without.append(z[0,0])
    residuals_without.append(r_kf_without.flatten())
    p_values_kf_without.append(p_value_kf_without)

# =========================
# 绘图 (适合论文的尺寸，2x2格式)
# =========================
steps_arr = np.arange(steps)
residuals_with = np.array(residuals_with)
residuals_without = np.array(residuals_without)
used_measurement_with = np.array(used_measurement_with)
used_measurement_without = np.array(used_measurement_without)

true_pos_arr = np.array(true_pos_with)
true_vel_arr = np.array(true_velocities)
true_acc_arr = np.array(true_accelerations)

pred_pos_with_arr = np.array(pred_pos_with)
pred_pos_without_arr = np.array(pred_pos_without)
pred_vel_with_arr = np.array(pred_vel_with)
pred_vel_without_arr = np.array(pred_vel_without)

acc_with_arr = np.array(acc_inputs_reported_with)
acc_without_arr = np.array(acc_inputs_reported_without)

pos_error_with = pred_pos_with_arr - true_pos_arr
pos_error_without = pred_pos_without_arr - true_pos_arr
vel_error_with = pred_vel_with_arr - true_vel_arr
vel_error_without = pred_vel_without_arr - true_vel_arr
acc_error_with = acc_with_arr - true_acc_arr
acc_error_without = acc_without_arr - true_acc_arr

# 设置适合论文的图表尺寸，2x2格式
fig, axes = plt.subplots(2, 2, figsize=(9,8))
fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.12,
                    hspace=0.28, wspace=0.19)


# 位置误差（估计-真实）
axes[0, 0].plot(steps_arr, pos_error_with, 'b-', label='Position Error (With Dual-Test)', linewidth=1.5)
axes[0, 0].plot(steps_arr, pos_error_without, 'm--', label='Position Error (Without Dual-Test)', linewidth=1.5)
axes[0, 0].axhline(0, color='k', linestyle=':', linewidth=1)
if attack_steps:
    y_min, y_max = axes[0, 0].get_ylim()
    y_position = y_min - (y_max - y_min) * 0.05
    axes[0, 0].scatter(np.array(attack_steps), [y_position]*len(attack_steps),
                       color='red', marker='^', s=10, label='Attack Points', zorder=5)
axes[0, 0].set_ylabel('Position Error [m]')
axes[0, 0].set_xlabel('Time Step [0.1 s]')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# 速度误差（估计-真实）
axes[0, 1].plot(steps_arr, vel_error_with, 'b-', label='Velocity Error (With Dual-Test)', linewidth=1.5)
axes[0, 1].plot(steps_arr, vel_error_without, 'm--', label='Velocity Error (Without Dual-Test)', linewidth=1.5)
axes[0, 1].axhline(0, color='k', linestyle=':', linewidth=1)
if attack_steps:
    y_min, y_max = axes[0, 1].get_ylim()
    y_position = y_min - (y_max - y_min) * 0.05
    axes[0, 1].scatter(np.array(attack_steps), [y_position]*len(attack_steps),
                       color='red', marker='^', s=10, zorder=5)
# axes[0, 1].set_ylabel('Velocity Error [m/s]')
axes[0, 1].set_ylabel('Velocity Error [m/s]')
axes[0, 1].set_xlabel('Time Step [0.1 s]')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# 加速度误差（输入-真实）
axes[1, 0].plot(steps_arr, acc_error_with, 'b-', label='Acceleration Error (With Dual-Test)', linewidth=1.5)
axes[1, 0].plot(steps_arr, acc_error_without, 'm--', label='Acceleration Error (Without Dual-Test)', linewidth=1.5)
axes[1, 0].axhline(0, color='k', linestyle=':', linewidth=1)
if attack_steps:
    y_min, y_max = axes[1, 0].get_ylim()
    y_position = y_min - (y_max - y_min) * 0.05
    axes[1, 0].scatter(np.array(attack_steps), [y_position]*len(attack_steps),
                       color='red', marker='^', s=10, zorder=5)
axes[1, 0].set_xlabel('Time Step [0.1 s]')
axes[1, 0].set_ylabel('Acceleration Error [m/s²]')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

# p值图对比
axes[1, 1].plot(steps_arr, p_values_kf_with, 'b-', label='p-value KF (With Dual-Test)', linewidth=1.5)
axes[1, 1].plot(steps_arr, p_values_kf_without, 'm-', label='p-value KF (Without Dual-Test)', linewidth=1.5)
axes[1, 1].axhline(alpha, color='r', linestyle='--', label=f'Significance Level ({alpha})', linewidth=1.5)
if attack_steps:
    y_min, y_max = axes[1, 1].get_ylim()
    y_position = y_min - (y_max - y_min) * 0.05
    axes[1, 1].scatter(np.array(attack_steps), [y_position]*len(attack_steps),
                       color='red', marker='^', s=10, zorder=5)
axes[1, 1].set_ylabel('p-value')
axes[1, 1].set_xlabel('Time Step [0.1 s]')
axes[1, 1].legend(fontsize=9, loc='center right')
axes[1, 1].grid(True, alpha=0.3)

# 每个子图添加 (a)-(d) 标注，字体更大
subplot_labels = {
    (0, 0): '(a)',
    (0, 1): '(b)',
    (1, 0): '(c)',
    (1, 1): '(d)'
}
for (row, col), label in subplot_labels.items():
    ax = axes[row, col]
    ax.text(0.5, -0.195, label, transform=ax.transAxes,
            fontsize=17, fontweight='bold', ha='center', va='top')

plt.savefig('kf_detection_results_random_comparison.png', dpi=800, bbox_inches='tight')


# =========================
# 计算性能指标（整体MAE）
# =========================
pos_mae_with = np.mean(np.abs(pos_error_with))
pos_mae_without = np.mean(np.abs(pos_error_without))
vel_mae_with = np.mean(np.abs(vel_error_with))
vel_mae_without = np.mean(np.abs(vel_error_without))
acc_mae_with = np.mean(np.abs(acc_error_with))
acc_mae_without = np.mean(np.abs(acc_error_without))

print("\n整体MAE结果:")
print("=" * 50)
print("考虑Dual-Test:")
print(f"  位置MAE: {pos_mae_with:.4f} m")
print(f"  速度MAE: {vel_mae_with:.4f} m/s")
print(f"  加速度MAE: {acc_mae_with:.4f} m/s²")
print("不考虑Dual-Test:")
print(f"  位置MAE: {pos_mae_without:.4f} m")
print(f"  速度MAE: {vel_mae_without:.4f} m/s")
print(f"  加速度MAE: {acc_mae_without:.4f} m/s²")

# 计算复位次数
print(f"\n复位次数比较:")
print("=" * 50)
print(f"考虑Dual-Test的复位次数: {len(reset_points_with)}")
print(f"不考虑Dual-Test的复位次数: 0 (无复位机制)")

# 计算p值小于显著性水平的次数（异常检测）
anomalies_with = np.sum(np.array(p_values_kf_with) < alpha)
anomalies_without = np.sum(np.array(p_values_kf_without) < alpha)

print(f"\n异常检测次数:")
print("=" * 50)
print(f"考虑Dual-Test的异常检测次数: {anomalies_with}")
print(f"不考虑Dual-Test的异常检测次数: {anomalies_without}")

# 保存结果到文本文件
with open("kf_detection_comparison_results_random.txt", "w", encoding="utf-8") as f:
    f.write("卡尔曼滤波检测结果比较：考虑Dual-Test vs 不考虑Dual-Test\n")
    f.write("=" * 60 + "\n")
    f.write(f"攻击类型: {attack_type}\n")
    f.write(f"攻击开始时间步: {continuous_attack_start}\n")
    f.write(f"攻击持续时间步: {continuous_attack_duration}\n")
    f.write(f"攻击步骤: {attack_steps}\n\n")
    
    f.write("整体MAE比较:\n")
    f.write("=" * 50 + "\n")
    f.write("考虑Dual-Test:\n")
    f.write(f"  位置MAE: {pos_mae_with:.4f} m\n")
    f.write(f"  速度MAE: {vel_mae_with:.4f} m/s\n")
    f.write(f"  加速度MAE: {acc_mae_with:.4f} m/s²\n\n")
    
    f.write("不考虑Dual-Test:\n")
    f.write(f"  位置MAE: {pos_mae_without:.4f} m\n")
    f.write(f"  速度MAE: {vel_mae_without:.4f} m/s\n")
    f.write(f"  加速度MAE: {acc_mae_without:.4f} m/s²\n\n")
    
    f.write("复位次数比较:\n")
    f.write("=" * 50 + "\n")
    f.write(f"考虑Dual-Test的复位次数: {len(reset_points_with)}\n")
    f.write(f"不考虑Dual-Test的复位次数: 0 (无复位机制)\n\n")
    
    f.write("异常检测次数:\n")
    f.write("=" * 50 + "\n")
    f.write(f"考虑Dual-Test的异常检测次数: {anomalies_with}\n")
    f.write(f"不考虑Dual-Test的异常检测次数: {anomalies_without}\n")

print(f"\n结果已保存到 kf_detection_comparison_results_continuous.txt 文件中")
