import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 统一的学术绘图风格设置，参考`plot_verify_kf_detection.py`
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'axes.linewidth': 1.2
})

def calculate_moving_stats(data, window=10):
    """
    计算滑动平均和标准差
    
    Args:
        data: 包含训练数据的DataFrame
        window: 滑动窗口大小
    
    Returns:
        tuple: (移动平均, 标准差)
    """
    scores = data['score'].values
    # 计算移动平均
    moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
    
    # 计算移动标准差
    moving_std = np.zeros_like(moving_avg)
    for i in range(len(moving_avg)):
        start_idx = max(0, i)
        end_idx = min(len(scores), i + window)
        window_data = scores[start_idx:end_idx]
        if len(window_data) > 0:
            moving_std[i] = np.std(window_data)
    
    return moving_avg, moving_std

def plot_training_comparison(window_size=10):
    """
    绘制DDPG训练结果对比图
    
    Args:
        window_size: 滑动窗口大小，默认为10
    """
    # 读取数据
    cbf_data = pd.read_csv('results/ddpg_training_scores_cbf.csv')
    no_cbf_data = pd.read_csv('results/ddpg_training_scores_no_cbf.csv')

    # 设置学术图表样式
    fig, ax = plt.subplots(figsize=(6, 4))

    # 计算滑动平均和标准差
    cbf_avg, cbf_std = calculate_moving_stats(cbf_data, window=window_size)
    no_cbf_avg, no_cbf_std = calculate_moving_stats(no_cbf_data, window=window_size)

    # 生成x轴数据（对应episode）
    # 移动平均会减少数据点数量，需要相应调整x轴
    start_offset = (window_size - 1) // 2
    end_offset = window_size // 2
    
    cbf_episodes = cbf_data['episode'].values[start_offset:-end_offset] if end_offset > 0 else cbf_data['episode'].values[start_offset:]
    no_cbf_episodes = no_cbf_data['episode'].values[start_offset:-end_offset] if end_offset > 0 else no_cbf_data['episode'].values[start_offset:]
    
    # 确保x轴和y轴长度一致
    cbf_length = min(len(cbf_episodes), len(cbf_avg), len(cbf_std))
    no_cbf_length = min(len(no_cbf_episodes), len(no_cbf_avg), len(no_cbf_std))
    
    cbf_episodes = cbf_episodes[:cbf_length]
    cbf_avg = cbf_avg[:cbf_length]
    cbf_std = cbf_std[:cbf_length]
    
    no_cbf_episodes = no_cbf_episodes[:no_cbf_length]
    no_cbf_avg = no_cbf_avg[:no_cbf_length]
    no_cbf_std = no_cbf_std[:no_cbf_length]

    # 绘制CBF数据
    ax.plot(
        cbf_episodes,
        cbf_avg,
        label='DDPG without CBF',
        color='#1f77b4',
        linewidth=2.2
    )
    ax.fill_between(
        cbf_episodes,
        cbf_avg - cbf_std,
        cbf_avg + cbf_std,
        color='#1f77b4',
        alpha=0.18,
        edgecolor='none'
    )

    # 绘制无CBF数据
    ax.plot(
        no_cbf_episodes,
        no_cbf_avg,
        label='DDPG with CBF',
        color='#ff7f0e',
        linewidth=2.2
    )
    ax.fill_between(
        no_cbf_episodes,
        no_cbf_avg - no_cbf_std,
        no_cbf_avg + no_cbf_std,
        color='#ff7f0e',
        alpha=0.18,
        edgecolor='none'
    )

    # 设置图表属性
    ax.set_xlabel('Episode', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax.tick_params(direction='in', length=5, width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    # 设置坐标轴范围
    ax.set_xlim(0, max(len(cbf_data), len(no_cbf_data)))
    ax.set_ylim(-50, 500)

    # 添加标签和标题
    plt.tight_layout(pad=1.1)

    # 保存图像
    output_filename = f'results/ddpg_training_comparison_window{window_size}.png'
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    print(f"Training comparison plot saved as '{output_filename}'")

    # 显示图像
    plt.show()
    plt.close()  # 关闭图像以释放内存

# 简化版本的函数，更容易使用
def simple_plot(window_size=10):
    """
    简化版绘图函数
    
    Args:
        window_size: 滑动窗口大小
    """
    plot_training_comparison(window_size)

# 从命令行接受窗口大小参数
if __name__ == "__main__":
    import sys
    # 默认使用窗口大小为5
    window_size = 8
    if len(sys.argv) > 1:
        try:
            window_size = int(sys.argv[1])
        except ValueError:
            print("无效的窗口大小，使用默认值5")
    
    print(f"使用窗口大小: {window_size}")
    plot_training_comparison(window_size)