#!/usr/bin/env python3
"""
Improved Proposition 2 Validation
改进的命题2验证：物理同构增强鲁棒性
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import Dict, List, Tuple

# 设置matplotlib后端
plt.switch_backend('Agg')

class PhysicsInformedLayer(nn.Module):
    """物理信息层：实现真正的故障诊断物理原理"""

    def __init__(self, seq_len: int = 50):
        super().__init__()
        self.seq_len = seq_len

        # 频域分析（模拟FFT物理意义）
        self.freq_transform = nn.Sequential(
            nn.Linear(seq_len, seq_len * 2),
            nn.ReLU(),
            nn.Linear(seq_len * 2, seq_len),
            nn.LayerNorm(seq_len)
        )

        # 包络检测（模拟希尔伯特变换）
        self.envelope_net = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(4, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

        # 能量守恒约束
        self.energy_constraint = nn.LayerNorm(seq_len)

    def forward(self, x):
        """
        物理信息处理
        Args:
            x: (batch, seq_len) 或 (batch, 1, seq_len)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size = x.size(0)

        # 频域分析（保持能量分布）
        x_reshaped = x.squeeze(1)
        freq_features = self.freq_transform(x_reshaped)

        # 包络检测
        envelope = self.envelope_net(x)
        envelope = envelope.squeeze(1)

        # 能量守恒
        x_energy = self.energy_constraint(x)
        x_energy = x_energy.squeeze(1)

        # 物理特征组合
        physics_features = torch.stack([freq_features, envelope, x_energy], dim=1)

        return {
            'physics_features': physics_features,
            'freq_features': freq_features,
            'envelope': envelope,
            'energy': x_energy
        }

    def compute_energy_ratio(self, x, physics_dict):
        """计算能量守恒比率"""
        input_energy = torch.mean(x**2, dim=1)
        freq_energy = torch.mean(physics_dict['freq_features']**2, dim=1)

        # 能量比率应接近1（能量守恒）
        energy_ratio = torch.min(freq_energy / (input_energy + 1e-8),
                                input_energy / (freq_energy + 1e-8))
        return energy_ratio.mean()


class ImprovedPhysicsModel(nn.Module):
    """改进的物理信息模型"""

    def __init__(self, input_dim=50, hidden_dim=100, num_classes=5):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 物理层
        self.physics_layer = PhysicsInformedLayer(input_dim)

        # 特征融合（物理特征+基础特征）
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 150, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # 分类器
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

        # 物理约束权重
        self.physics_weight = 0.1

    def forward(self, x):
        """前向传播"""
        # 基础特征
        base_features = self.feature_extractor(x)

        # 物理特征
        physics_dict = self.physics_layer(x)
        physics_features = physics_dict['physics_features']

        # 确保physics_features是2D的
        if physics_features.dim() == 3:
            physics_features = physics_features.view(physics_features.size(0), -1)

        # 融合特征
        combined = torch.cat([base_features, physics_features], dim=1)
        fused_features = self.fusion(combined)

        # 分类
        logits = self.classifier(fused_features)

        # 计算物理损失
        physics_loss = self.physics_weight * (
            1.0 - self.physics_layer.compute_energy_ratio(x, physics_dict)
        )

        return {
            'logits': logits,
            'physics_loss': physics_loss,
            'energy_ratio': self.physics_layer.compute_energy_ratio(x, physics_dict)
        }


def test_proposition_2_improved():
    """改进的命题2验证"""
    print("\n=== 命题2验证：改进的物理同构模型 ===")

    # 配置
    batch_size = 100
    input_dim = 50
    num_classes = 5
    num_experiments = 5
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

    # 生成数据
    train_data = torch.randn(batch_size * 10, input_dim)
    train_labels = torch.randint(0, num_classes, (batch_size * 10,))

    # 创建模型
    standard_model = nn.Sequential(
        nn.Linear(input_dim, 100),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(100, num_classes)
    )

    physics_model = ImprovedPhysicsModel(
        input_dim=input_dim,
        hidden_dim=100,
        num_classes=num_classes
    )

    # 训练设置
    optimizer_std = torch.optim.Adam(standard_model.parameters(), lr=0.001)
    optimizer_physics = torch.optim.Adam(physics_model.parameters(), lr=0.001)

    # 存储结果
    std_results = []
    physics_results = []

    print("训练模型并测试噪声鲁棒性...")

    # 预训练（简化版）
    for epoch in range(10):
        # 训练标准模型
        optimizer_std.zero_grad()
        std_outputs = standard_model(train_data)
        std_loss = nn.CrossEntropyLoss()(std_outputs, train_labels)
        std_loss.backward()
        optimizer_std.step()

        # 训练物理模型
        optimizer_physics.zero_grad()
        physics_outputs = physics_model(train_data)
        physics_loss_total = nn.CrossEntropyLoss()(physics_outputs['logits'], train_labels) + physics_outputs['physics_loss']
        physics_loss_total.backward()
        optimizer_physics.step()

    # 测试不同噪声水平
    for noise in noise_levels:
        print(f"\n噪声水平: {noise:.2f}")

        # 创建测试数据
        test_data = torch.randn(batch_size, input_dim)
        test_labels = torch.randint(0, num_classes, (batch_size,))

        # 添加噪声
        test_data_noisy = test_data + torch.randn_like(test_data) * noise

        # 评估标准模型
        standard_model.eval()
        with torch.no_grad():
            std_pred = standard_model(test_data_noisy)
            std_acc = (torch.argmax(std_pred, dim=1) == test_labels).float().mean()

        # 评估物理模型
        physics_model.eval()
        with torch.no_grad():
            physics_pred = physics_model(test_data_noisy)
            physics_acc = (torch.argmax(physics_pred['logits'], dim=1) == test_labels).float().mean()

        std_results.append(std_acc.item())
        physics_results.append(physics_acc.item())

        print(f"  标准模型准确率: {std_acc.item():.4f}")
        print(f"  物理模型准确率: {physics_acc.item():.4f}")

    # 计算性能下降率
    std_drops = [std_results[0] - r for r in std_results]
    physics_drops = [physics_results[0] - r for r in physics_results]

    # 修正：计算每单位噪声的性能下降
    std_drop_rate = np.std(std_drops[1:]) / (noise_levels[-1] if noise_levels[-1] > 0 else 1)
    physics_drop_rate = np.std(physics_drops[1:]) / (noise_levels[-1] if noise_levels[-1] > 0 else 1)

    # 改进：使用平均性能下降率
    std_avg_drop = np.mean(std_drops[1:]) / noise_levels[-1] if noise_levels[-1] > 0 else 0
    physics_avg_drop = np.mean(physics_drops[1:]) / noise_levels[-1] if noise_levels[-1] > 0 else 0

    print(f"\n性能下降分析:")
    print(f"  标准模型平均下降率: {std_avg_drop:.4f}")
    print(f"  物理模型平均下降率: {physics_avg_drop:.4f}")

    # 物理同构验证
    print(f"\n物理同构验证:")
    with torch.no_grad():
        physics_dict = physics_model.physics_layer(test_data)
        energy_ratios = []
        for i in range(10):
            ratio = physics_model.physics_layer.compute_energy_ratio(
                test_data[i:i+1], physics_dict
            )
            energy_ratios.append(ratio.item())

    avg_energy_ratio = np.mean(energy_ratios)
    print(f"  平均能量守恒比率: {avg_energy_ratio:.4f} (越接近1越好)")

    # 判断是否支持命题
    proposition_2_supported = physics_avg_drop < std_avg_drop and avg_energy_ratio > 0.8

    print(f"\n命题2验证结果: {'✓ 支持' if proposition_2_supported else '✗ 不支持'}")

    # 生成改进的图表
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, std_results, 'o-', label='标准模型', color='red', linewidth=2, markersize=8)
    plt.plot(noise_levels, physics_results, 's-', label='物理同构模型', color='blue', linewidth=2, markersize=8)

    plt.title('命题2验证：改进的物理同构模型性能对比', fontsize=14)
    plt.xlabel('噪声水平 (σ)', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # 添加注释
    plt.annotate(f'能量守恒比率: {avg_energy_ratio:.3f}',
                xy=(noise_levels[3], physics_results[3]),
                xytext=(noise_levels[3] + 0.05, physics_results[3] - 0.02),
                arrowprops=dict(arrowstyle='->', color='green'),
                color='green')

    plt.tight_layout()
    os.makedirs('./results/theory_validation', exist_ok=True)
    plt.savefig('./results/theory_validation/proposition_2_improved.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return {
        'noise_levels': noise_levels,
        'standard_performance': std_results,
        'physics_performance': physics_results,
        'energy_ratio': avg_energy_ratio,
        'proposition_supported': proposition_2_supported
    }


if __name__ == "__main__":
    results = test_proposition_2_improved()
    print("\n验证完成！")