#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多样化拓扑神经网络模型训练脚本
使用支持不同车辆数量和IDM参数的训练数据

功能：
1. 加载多样化拓扑训练数据
2. 训练能够处理不同拓扑结构的LSTM模型
3. 增强模型的泛化能力和鲁棒性
4. 支持域适应训练策略
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class EnhancedTrafficDataset(Dataset):
    """增强的交通仿真数据集类，支持拓扑信息"""
    
    def __init__(self, features, targets, topology_info=None):
        """
        初始化数据集
        
        参数:
        - features: 输入特征 (n_samples, 250)
        - targets: 目标标签 (n_samples, 2)
        - topology_info: 拓扑信息 (n_samples, n_topology_features)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.topology_info = torch.FloatTensor(topology_info) if topology_info is not None else None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.topology_info is not None:
            return self.features[idx], self.targets[idx], self.topology_info[idx]
        else:
            return self.features[idx], self.targets[idx]


class TopologyAwareLSTM(nn.Module):
    """拓扑感知LSTM模型"""
    
    def __init__(self, input_size=250, topology_size=2, hidden_size=64, num_layers=2, 
                 output_size=2, dropout=0.2, use_topology=True):
        """
        初始化拓扑感知LSTM模型
        
        参数:
        - input_size: 输入特征维度 (250)
        - topology_size: 拓扑信息维度 (车辆数量 + CAV位置)
        - hidden_size: LSTM隐藏层大小
        - num_layers: LSTM层数
        - output_size: 输出维度 (2: 速度和加速度)
        - dropout: Dropout比率
        - use_topology: 是否使用拓扑信息
        """
        super(TopologyAwareLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_topology = use_topology
        
        # 将250维特征重塑为时间序列: (batch, 50时间步, 5特征)
        self.sequence_length = 50
        self.feature_dim = 5  # 速度差、加速度差、位置差、CAV速度、CAV加速度
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 拓扑信息嵌入层
        if self.use_topology:
            self.topology_embedding = nn.Sequential(
                nn.Linear(topology_size, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            final_hidden_size = hidden_size + hidden_size // 4
        else:
            final_hidden_size = hidden_size
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(final_hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, output_size)
        )
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化模型权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)
                
        if self.use_topology:
            for layer in self.topology_embedding:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    layer.bias.data.fill_(0)
    
    def forward(self, x, topology_info=None):
        """
        前向传播
        
        参数:
        - x: 输入张量 (batch_size, 250)
        - topology_info: 拓扑信息 (batch_size, topology_size)
        
        返回:
        - output: 预测结果 (batch_size, 2)
        """
        batch_size = x.size(0)
        
        # 重塑输入：250维 -> (50时间步, 5特征)
        speed_diff = x[:, 0:50]      # (batch_size, 50)
        accel_diff = x[:, 50:100]    # (batch_size, 50)
        position_diff = x[:, 100:150] # (batch_size, 50)
        cav_speed = x[:, 150:200]    # (batch_size, 50)
        cav_accel = x[:, 200:250]    # (batch_size, 50)
        
        # 合并特征: (batch_size, 50, 5)
        x_reshaped = torch.stack([speed_diff, accel_diff, position_diff, cav_speed, cav_accel], dim=2)
        
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x_reshaped)
        
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # 如果使用拓扑信息，则融合拓扑特征
        if self.use_topology and topology_info is not None:
            topology_embedded = self.topology_embedding(topology_info)
            combined_features = torch.cat([last_output, topology_embedded], dim=1)
        else:
            combined_features = last_output
        
        # 全连接层预测
        output = self.fc_layers(combined_features)
        
        return output


class EnhancedTrafficTimeSeriesTrainer:
    """增强的交通时间序列模型训练器"""
    
    def __init__(self, model, device='cpu', use_topology=True):
        """
        初始化训练器
        
        参数:
        - model: 神经网络模型
        - device: 计算设备 ('cpu' 或 'cuda')
        - use_topology: 是否使用拓扑信息
        """
        self.model = model.to(device)
        self.device = device
        self.use_topology = use_topology
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def train_epoch(self, train_loader, optimizer, criterion):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_data in train_loader:
            if self.use_topology:
                batch_features, batch_targets, batch_topology = batch_data
                batch_topology = batch_topology.to(self.device)
            else:
                batch_features, batch_targets = batch_data
                batch_topology = None
            
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # 前向传播
            if self.use_topology:
                outputs = self.model(batch_features, batch_topology)
            else:
                outputs = self.model(batch_features)
            
            loss = criterion(outputs, batch_targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader, criterion):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                if self.use_topology:
                    batch_features, batch_targets, batch_topology = batch_data
                    batch_topology = batch_topology.to(self.device)
                else:
                    batch_features, batch_targets = batch_data
                    batch_topology = None
                
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                if self.use_topology:
                    outputs = self.model(batch_features, batch_topology)
                else:
                    outputs = self.model(batch_features)
                
                loss = criterion(outputs, batch_targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader, num_epochs=100, learning_rate=0.001, patience=15):
        """训练模型"""
        print(f"🚀 开始训练拓扑感知模型...")
        print(f"   - 设备: {self.device}")
        print(f"   - 使用拓扑信息: {self.use_topology}")
        print(f"   - 训练轮数: {num_epochs}")
        print(f"   - 学习率: {learning_rate}")
        print(f"   - 早停耐心值: {patience}")
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.MSELoss()
        
        # 早停计数器
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # 验证
            val_loss = self.validate_epoch(val_loader, criterion)
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{num_epochs}] - "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"LR: {current_lr:.6f}")
            
            # 早停检查
            if patience_counter >= patience:
                print(f"早停触发！在第 {epoch+1} 轮停止训练")
                break
        
        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"✅ 训练完成！最佳验证损失: {self.best_val_loss:.6f}")
        
        return self.train_losses, self.val_losses


def load_and_preprocess_diverse_data(data_path):
    """
    加载和预处理多样化数据
    
    参数:
    - data_path: 数据文件路径
    
    返回:
    - features: 输入特征
    - targets: 目标标签
    - topology_features: 拓扑特征
    - feature_scaler: 特征标准化器
    - target_scaler: 标签标准化器
    """
    print("📊 加载和预处理多样化训练数据...")
    
    # 加载数据
    if data_path.endswith('.pkl'):
        with open(data_path, 'rb') as f:
            df = pickle.load(f)
    else:
        df = pd.read_csv(data_path)
    
    print(f"   - 数据样本数: {len(df)}")
    print(f"   - 拓扑结构类型: {df['topology_id'].nunique()}")
    print(f"   - 车辆数量范围: {df['num_vehicles'].min()}-{df['num_vehicles'].max()}")
    print(f"   - IDM策略类型: {df['idm_strategy'].nunique() if 'idm_strategy' in df.columns else 'N/A'}")
    
    # 提取输入特征 (250维)
    feature_columns = [f'input_feature_{i:03d}' for i in range(250)]
    features = df[feature_columns].values
    
    # 提取目标标签 (2维: 速度和加速度)
    targets = df[['output_target_avg_speed', 'output_target_avg_acceleration']].values
    
    # 提取拓扑特征 (车辆数量和CAV位置)
    # 对于网络攻击数据集，我们需要正确处理CAV位置特征
    if 'cav_position' in df.columns:
        # 单个CAV位置的情况
        topology_features = df[['num_vehicles', 'cav_position']].values
    elif 'cav_positions' in df.columns:
        # 多个CAV位置的情况，取第一个作为特征
        # 在我们的新数据中，只有一个CAV位置
        def parse_cav_position(cav_pos_str):
            if isinstance(cav_pos_str, str):
                # 如果是字符串，尝试解析为整数索引
                try:
                    # 如果是单个数字，直接返回
                    if cav_pos_str.isdigit():
                        return int(cav_pos_str)
                    # 如果是逗号分隔的数字，取第一个
                    elif ',' in cav_pos_str:
                        return int(cav_pos_str.split(',')[0])
                    # 否则尝试转换为浮点数再转为整数
                    else:
                        return int(float(cav_pos_str))
                except:
                    return 0
            else:
                return 0
        
        cav_positions = df['cav_positions'].apply(parse_cav_position).values
        topology_features = np.column_stack([df['num_vehicles'].values, cav_positions])
    else:
        # 默认情况
        topology_features = df[['num_vehicles', 'cav_position']].values
    
    print(f"   - 输入特征形状: {features.shape}")
    print(f"   - 目标标签形状: {targets.shape}")
    print(f"   - 拓扑特征形状: {topology_features.shape}")
    
    # 数据标准化
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    topology_scaler = StandardScaler()
    
    features_scaled = feature_scaler.fit_transform(features)
    targets_scaled = target_scaler.fit_transform(targets)
    topology_scaled = topology_scaler.fit_transform(topology_features)
    
    print("   - 数据标准化完成")
    
    return features_scaled, targets_scaled, topology_scaled, feature_scaler, target_scaler, topology_scaler


def evaluate_enhanced_model(model, test_loader, target_scaler, device, use_topology):
    """评估增强模型性能"""
    print("📈 评估增强模型性能...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_topology = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            if use_topology:
                batch_features, batch_targets, batch_topology = batch_data
                batch_topology = batch_topology.to(device)
                all_topology.append(batch_topology.cpu().numpy())
            else:
                batch_features, batch_targets = batch_data
                batch_topology = None
            
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            if use_topology:
                outputs = model(batch_features, batch_topology)
            else:
                outputs = model(batch_features)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_targets.cpu().numpy())
    
    # 合并所有预测结果
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    if use_topology:
        topology_info = np.vstack(all_topology)
    
    # 反标准化
    predictions_orig = target_scaler.inverse_transform(predictions)
    targets_orig = target_scaler.inverse_transform(targets)
    
    # 计算评估指标
    metrics = {}
    
    for i, label in enumerate(['速度', '加速度']):
        pred_i = predictions_orig[:, i]
        target_i = targets_orig[:, i]
        
        mse = mean_squared_error(target_i, pred_i)
        mae = mean_absolute_error(target_i, pred_i)
        rmse = np.sqrt(mse)
        r2 = r2_score(target_i, pred_i)
        
        # 计算MAPE（平均绝对百分比误差）
        mape = np.mean(np.abs((target_i - pred_i) / (target_i + 1e-8))) * 100
        
        # 计算相关系数
        correlation = np.corrcoef(target_i, pred_i)[0, 1]
        
        metrics[f'{label}_MSE'] = mse
        metrics[f'{label}_MAE'] = mae
        metrics[f'{label}_RMSE'] = rmse
        metrics[f'{label}_R2'] = r2
        metrics[f'{label}_MAPE'] = mape
        metrics[f'{label}_Correlation'] = correlation
        
        print(f"   {label}预测性能:")
        print(f"     - MSE: {mse:.6f}")
        print(f"     - MAE: {mae:.6f}")
        print(f"     - RMSE: {rmse:.6f}")
        print(f"     - R²: {r2:.6f}")
        print(f"     - MAPE: {mape:.2f}%")
        print(f"     - 相关系数: {correlation:.6f}")
    
    return metrics, predictions_orig, targets_orig


def save_training_data_to_csv(train_losses, val_losses, csv_path):
    """保存训练损失数据到CSV文件"""
    import csv
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'val_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for epoch in range(len(train_losses)):
            writer.writerow({
                'epoch': epoch + 1,
                'train_loss': train_losses[epoch],
                'val_loss': val_losses[epoch]
            })
    
    print(f"   训练损失数据已保存: {csv_path}")


def save_prediction_data_to_csv(predictions, targets, csv_path):
    """保存预测数据到CSV文件"""
    import csv
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['sample_index', 'true_speed', 'pred_speed', 'true_acceleration', 'pred_acceleration']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(len(predictions)):
            writer.writerow({
                'sample_index': i,
                'true_speed': targets[i, 0],
                'pred_speed': predictions[i, 0],
                'true_acceleration': targets[i, 1],
                'pred_acceleration': predictions[i, 1]
            })
    
    print(f"   预测数据已保存: {csv_path}")


def plot_training_loss_curve(train_losses, val_losses, output_path):
    """
    绘制训练损失曲线（单独的图）
    使用期刊论文风格（Times New Roman字体，英文标签）
    """
    print("📊 绘制训练损失曲线图...")
    
    # 设置期刊论文风格
    plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 增大字体大小
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20
    })
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2.5, alpha=0.8)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2.5, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('Loss', fontsize=16)
    ax.legend(fontsize=14, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   训练损失曲线图已保存: {output_path}")


def plot_prediction_validation(predictions, targets, output_path):
    """
    绘制预测验证图（速度和加速度两个子图）
    使用期刊论文风格（Times New Roman字体，英文标签）
    """
    print("📊 绘制预测验证图...")
    
    # 设置期刊论文风格
    plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 增大字体大小
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20
    })
    
    # 创建图表：1行2列
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 提取速度和加速度数据
    true_speed = targets[:, 0]
    pred_speed = predictions[:, 0]
    true_acc = targets[:, 1]
    pred_acc = predictions[:, 1]
    
    # 计算R²
    r2_speed = r2_score(true_speed, pred_speed)
    r2_acc = r2_score(true_acc, pred_acc)
    
    # 子图1: 速度预测对比
    ax1 = axes[0]
    ax1.scatter(true_speed, pred_speed, alpha=0.6, s=30, c='blue', 
                edgecolors='darkblue', linewidths=0.5, label='Speed', 
                marker='o', zorder=3)
    
    # 绘制理想预测线（对角线）
    speed_min = min(np.min(true_speed), np.min(pred_speed))
    speed_max = max(np.max(true_speed), np.max(pred_speed))
    ax1.plot([speed_min, speed_max], [speed_min, speed_max], 'k--', linewidth=2, 
             label='Ideal Prediction', alpha=0.7, zorder=2)
    
    # 在图上添加R²信息
    ax1.text(0.95, 0.05, f'R² = {r2_speed:.4f}', 
             transform=ax1.transAxes, fontsize=14, verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax1.set_xlabel('Ground Truth (m/s)', fontsize=16)
    ax1.set_ylabel('Prediction (m/s)', fontsize=16)
    ax1.legend(fontsize=14, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 在子图1正下方添加(a)标签
    ax1.text(0.5, -0.15, '(a)', transform=ax1.transAxes, fontsize=18, 
             fontweight='bold', ha='center', va='top')
    
    # 子图2: 加速度预测对比
    ax2 = axes[1]
    ax2.scatter(true_acc, pred_acc, alpha=0.6, s=30, c='red', 
                edgecolors='darkred', linewidths=0.5, label='Acceleration', 
                marker='^', zorder=3)
    
    # 绘制理想预测线（对角线）
    acc_min = min(np.min(true_acc), np.min(pred_acc))
    acc_max = max(np.max(true_acc), np.max(pred_acc))
    ax2.plot([acc_min, acc_max], [acc_min, acc_max], 'k--', linewidth=2, 
             label='Ideal Prediction', alpha=0.7, zorder=2)
    
    # 在图上添加R²信息
    ax2.text(0.95, 0.05, f'R² = {r2_acc:.4f}', 
             transform=ax2.transAxes, fontsize=14, verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax2.set_xlabel('Ground Truth (m/s²)', fontsize=16)
    ax2.set_ylabel('Prediction (m/s²)', fontsize=16)
    ax2.legend(fontsize=14, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 在子图2正下方添加(b)标签
    ax2.text(0.5, -0.15, '(b)', transform=ax2.transAxes, fontsize=18, 
             fontweight='bold', ha='center', va='top')
    
    # 调整布局，为底部标签留出空间
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   预测验证图已保存: {output_path}")


def plot_training_curves_from_csv(csv_path, output_path):
    """从CSV文件读取数据并绘制训练损失曲线"""
    import csv
    
    epochs = []
    train_losses = []
    val_losses = []
    
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_losses.append(float(row['train_loss']))
            val_losses.append(float(row['val_loss']))
    
    # 设置期刊论文风格
    plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2.5, alpha=0.8)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2.5, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('Loss', fontsize=16)
    ax.set_title('Training and Validation Loss', fontsize=18, fontweight='bold')
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   训练损失曲线已保存: {output_path}")


def plot_predictions_from_csv(csv_path, output_path):
    """从CSV文件读取数据并绘制预测对比图"""
    import csv
    
    true_speed = []
    pred_speed = []
    true_acc = []
    pred_acc = []
    
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            true_speed.append(float(row['true_speed']))
            pred_speed.append(float(row['pred_speed']))
            true_acc.append(float(row['true_acceleration']))
            pred_acc.append(float(row['pred_acceleration']))
    
    true_speed = np.array(true_speed)
    pred_speed = np.array(pred_speed)
    true_acc = np.array(true_acc)
    pred_acc = np.array(pred_acc)
    
    # 设置期刊论文风格
    plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 计算R²
    r2_speed = r2_score(true_speed, pred_speed)
    r2_acc = r2_score(true_acc, pred_acc)
    
    # 绘制散点图
    ax.scatter(true_speed, pred_speed, alpha=0.5, s=25, c='blue', 
               edgecolors='none', label=f'Speed (R² = {r2_speed:.4f})', marker='o')
    ax.scatter(true_acc, pred_acc, alpha=0.5, s=25, c='red', 
               edgecolors='none', label=f'Acceleration (R² = {r2_acc:.4f})', marker='^')
    
    # 理想预测线
    all_values = np.concatenate([true_speed, pred_speed, true_acc, pred_acc])
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, 
            label='Ideal Prediction', alpha=0.7)
    
    ax.set_xlabel('Ground Truth', fontsize=16)
    ax.set_ylabel('Prediction', fontsize=16)
    ax.set_title('Prediction vs Ground Truth', fontsize=18, fontweight='bold')
    ax.legend(fontsize=13, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   预测对比图已保存: {output_path}")


def plot_prediction_time_series(predictions, targets, output_path, sample_size=500):
    """绘制时间序列预测对比图（采样显示）"""
    print("📊 绘制时间序列预测对比图...")
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 如果样本太多，进行采样
    if len(predictions) > sample_size:
        indices = np.linspace(0, len(predictions) - 1, sample_size, dtype=int)
        predictions_plot = predictions[indices]
        targets_plot = targets[indices]
        time_steps = indices
    else:
        predictions_plot = predictions
        targets_plot = targets
        time_steps = np.arange(len(predictions))
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    labels = ['速度', '加速度']
    units = ['m/s', 'm/s²']
    
    for i, (label, unit) in enumerate(zip(labels, units)):
        ax = axes[i]
        
        pred_i = predictions_plot[:, i]
        target_i = targets_plot[:, i]
        
        ax.plot(time_steps, target_i, 'b-', label=f'真实{label}', linewidth=1.5, alpha=0.7)
        ax.plot(time_steps, pred_i, 'r--', label=f'预测{label}', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('样本索引', fontsize=11)
        ax.set_ylabel(f'{label} ({unit})', fontsize=11)
        ax.set_title(f'{label}预测时间序列对比', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   时间序列对比图已保存: {output_path}")


def save_test_results_txt(metrics, predictions, targets, output_path, dataset_info=None):
    """保存测试结果到txt文件"""
    print("📝 保存测试结果到txt文档...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("模型测试结果报告\n")
        f.write("=" * 70 + "\n\n")
        
        if dataset_info:
            f.write("数据集信息:\n")
            f.write(f"  - 数据集类型: {dataset_info.get('type', 'N/A')}\n")
            f.write(f"  - 数据文件: {dataset_info.get('file', 'N/A')}\n")
            f.write(f"  - 测试样本数: {len(predictions)}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("评估指标详细结果\n")
        f.write("=" * 70 + "\n\n")
        
        labels = ['速度', '加速度']
        for label in labels:
            f.write(f"{label}预测性能:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  有效数据点数量: {len(predictions)}\n")
            f.write(f"  均方误差 (MSE): {metrics.get(f'{label}_MSE', 0):.6f}\n")
            f.write(f"  平均绝对误差 (MAE): {metrics.get(f'{label}_MAE', 0):.6f}\n")
            f.write(f"  均方根误差 (RMSE): {metrics.get(f'{label}_RMSE', 0):.6f}\n")
            f.write(f"  决定系数 (R²): {metrics.get(f'{label}_R2', 0):.6f}\n")
            f.write(f"  平均绝对百分比误差 (MAPE): {metrics.get(f'{label}_MAPE', 0):.2f}%\n")
            f.write(f"  预测相关系数: {metrics.get(f'{label}_Correlation', 0):.6f}\n")
            
            # 计算预测精度
            mape = metrics.get(f'{label}_MAPE', 0)
            accuracy = max(0, 100 - mape)
            f.write(f"  预测精度: {accuracy:.2f}%\n")
            f.write("\n")
        
        f.write("=" * 70 + "\n")
        f.write("模型配置信息\n")
        f.write("=" * 70 + "\n")
        f.write(f"  模型类型: TopologyAwareLSTM\n")
        f.write(f"  输入维度: 250\n")
        f.write(f"  输出维度: 2 (速度, 加速度)\n")
        f.write(f"  使用拓扑信息: {dataset_info.get('use_topology', 'N/A') if dataset_info else 'N/A'}\n")
        
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n")
    
    print(f"   测试结果文档已保存: {output_path}")


def main():
    """主函数"""
    print("🧠 多样化拓扑神经网络训练")
    print("=" * 60)
    
    # 查找最新的多样化数据集（支持两种命名格式）
    training_files = []
    
    # 查找网络攻击数据集（优先使用最新生成的数据）
    cyber_attack_files = [f for f in os.listdir("training_data") 
                         if f.startswith("cyber_attack_neural_network_dataset_complete_") and f.endswith(".pkl")]
    
    # 查找多样化拓扑数据集
    diverse_topology_files = [f for f in os.listdir("training_data") 
                             if f.startswith("diverse_topology_neural_network_dataset_") and f.endswith(".pkl")]
    
    # 优先使用具有250维特征的最新网络攻击数据集
    if cyber_attack_files:
        # 过滤出具有250维特征的数据集
        filtered_cyber_attack_files = []
        for f in cyber_attack_files:
            try:
                # 检查数据集是否具有250维特征
                file_path = f"training_data/{f}"
                with open(file_path, 'rb') as file:
                    df = pickle.load(file)
                feature_cols = [c for c in df.columns if c.startswith('input_feature_') and c.endswith(('0','1','2','3','4','5','6','7','8','9'))]
                if len(feature_cols) == 250:
                    filtered_cyber_attack_files.append(f)
            except:
                continue
        
        if filtered_cyber_attack_files:
            training_files = filtered_cyber_attack_files
            dataset_type = "网络攻击"
        elif diverse_topology_files:
            training_files = diverse_topology_files
            dataset_type = "多样化拓扑"
        else:
            print("❌ 未找到具有250维特征的训练数据文件")
            return
    elif diverse_topology_files:
        training_files = diverse_topology_files
        dataset_type = "多样化拓扑"
    else:
        print("❌ 未找到训练数据文件")
        return
    
    latest_file = sorted(training_files)[-1]
    DATA_PATH = f"training_data/{latest_file}"
    
    print(f"📁 使用{dataset_type}数据集: {latest_file}")
    
    # 配置参数
    BATCH_SIZE = 32
    NUM_EPOCHS = 150
    LEARNING_RATE = 0.001
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    USE_TOPOLOGY = True  # 是否使用拓扑信息
    
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    try:
        # 1. 加载和预处理数据
        features, targets, topology_features, feature_scaler, target_scaler, topology_scaler = load_and_preprocess_diverse_data(DATA_PATH)
        
        # 2. 创建数据集
        dataset = EnhancedTrafficDataset(features, targets, topology_features if USE_TOPOLOGY else None)
        
        # 3. 数据集分割
        total_size = len(dataset)
        train_size = int(TRAIN_RATIO * total_size)
        val_size = int(VAL_RATIO * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"📊 数据集分割:")
        print(f"   - 训练集: {len(train_dataset)} 样本")
        print(f"   - 验证集: {len(val_dataset)} 样本")
        print(f"   - 测试集: {len(test_dataset)} 样本")
        
        # 4. 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # 5. 创建模型
        model = TopologyAwareLSTM(
            input_size=250,
            topology_size=2,  # 车辆数量 + CAV位置
            hidden_size=96,   # 增加隐藏层大小以处理更复杂的数据
            num_layers=3,     # 增加层数
            output_size=2,
            dropout=0.3,      # 增加dropout以防止过拟合
            use_topology=USE_TOPOLOGY
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"🏗️  模型结构:")
        print(f"   - 输入维度: 250")
        print(f"   - 拓扑特征维度: 2")
        print(f"   - LSTM隐藏层大小: 96")
        print(f"   - LSTM层数: 3")
        print(f"   - 输出维度: 2")
        print(f"   - 使用拓扑信息: {USE_TOPOLOGY}")
        print(f"   - 总参数数量: {total_params:,}")
        
        # 6. 创建训练器并训练
        trainer = EnhancedTrafficTimeSeriesTrainer(model, device, USE_TOPOLOGY)
        train_losses, val_losses = trainer.train(
            train_loader, val_loader, 
            num_epochs=NUM_EPOCHS, 
            learning_rate=LEARNING_RATE
        )
        
        # 7. 评估模型
        metrics, predictions, targets_eval = evaluate_enhanced_model(
            model, test_loader, target_scaler, device, USE_TOPOLOGY
        )
        
        # 8. 保存模型
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_suffix = "cyber_attack" if dataset_type == "网络攻击" else "diverse_topology"
        
        model_path = f"models/enhanced_traffic_lstm_model_{dataset_suffix}_{timestamp}.pth"
        torch.save(model.state_dict(), model_path)
        
        # 保存标准化器
        scaler_path = f"models/enhanced_scalers_{dataset_suffix}_{timestamp}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'feature_scaler': feature_scaler,
                'target_scaler': target_scaler,
                'topology_scaler': topology_scaler if USE_TOPOLOGY else None,
                'use_topology': USE_TOPOLOGY
            }, f)
        
        # 保存评估指标
        metrics_path = f"models/enhanced_metrics_{dataset_suffix}_{timestamp}.pkl"
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
        
        print(f"\n💾 增强模型已保存:")
        print(f"   - 模型权重: {model_path}")
        print(f"   - 标准化器: {scaler_path}")
        print(f"   - 评估指标: {metrics_path}")
        
        # 9. 保存数据并生成可视化图表和测试结果文档
        print(f"\n📊 保存数据并生成可视化图表和测试结果文档...")
        os.makedirs("outputs", exist_ok=True)
        
        # 保存训练损失数据到CSV
        training_loss_csv_path = f"outputs/training_losses_{dataset_suffix}_{timestamp}.csv"
        save_training_data_to_csv(train_losses, val_losses, training_loss_csv_path)
        
        # 保存预测数据到CSV
        prediction_data_csv_path = f"outputs/prediction_data_{dataset_suffix}_{timestamp}.csv"
        save_prediction_data_to_csv(predictions, targets_eval, prediction_data_csv_path)
        
        # 绘制训练损失曲线图（单独一个图）
        training_loss_plot_path = f"outputs/training_loss_curve_{dataset_suffix}_{timestamp}.png"
        plot_training_loss_curve(train_losses, val_losses, training_loss_plot_path)
        
        # 绘制预测验证图（速度和加速度两个子图）
        prediction_validation_plot_path = f"outputs/prediction_validation_{dataset_suffix}_{timestamp}.png"
        plot_prediction_validation(predictions, targets_eval, prediction_validation_plot_path)
        
        # 绘制时间序列预测对比图（可选）
        prediction_timeseries_path = f"outputs/prediction_timeseries_{dataset_suffix}_{timestamp}.png"
        plot_prediction_time_series(predictions, targets_eval, prediction_timeseries_path)
        
        # 保存测试结果txt文档
        test_results_txt_path = f"outputs/test_results_{dataset_suffix}_{timestamp}.txt"
        dataset_info = {
            'type': dataset_type,
            'file': latest_file,
            'use_topology': USE_TOPOLOGY
        }
        save_test_results_txt(metrics, predictions, targets_eval, test_results_txt_path, dataset_info)
        
        print(f"\n📈 数据和可视化图表已保存:")
        print(f"   - 训练损失数据CSV: {training_loss_csv_path}")
        print(f"   - 预测数据CSV: {prediction_data_csv_path}")
        print(f"   - 训练损失曲线图: {training_loss_plot_path}")
        print(f"   - 预测验证图: {prediction_validation_plot_path}")
        print(f"   - 时间序列对比图: {prediction_timeseries_path}")
        print(f"   - 测试结果文档: {test_results_txt_path}")
        
        print(f"\n🎉 {dataset_type}训练完成！")
        print(f"模型性能总结:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.6f}")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    main()