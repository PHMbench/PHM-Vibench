"""
命题2实验简化版：物理同构增强鲁棒性验证
Simplified Proposition 2 Validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os


def generate_synthetic_data(num_samples=1000, seq_len=50, noise_level=0.0):
    """生成合成故障数据"""
    signals = []
    labels = []

    for i in range(num_samples):
        # 随机选择故障类型
        fault_type = np.random.randint(0, 4)
        t = np.linspace(0, 1, seq_len)
        base_freq = np.random.uniform(10, 50)

        # 生成信号
        if fault_type == 0:  # 正常
            signal = np.sin(2 * np.pi * base_freq * t)
        elif fault_type == 1:  # 内圈故障
            signal = np.sin(2 * np.pi * base_freq * t) + 0.5 * np.sin(2 * np.pi * base_freq * 2 * t)
        elif fault_type == 2:  # 外圈故障
            signal = np.sin(2 * np.pi * base_freq * t) + 0.3 * np.sin(2 * np.pi * base_freq * 3 * t)
        else:  # 滚动体故障
            signal = np.sin(2 * np.pi * base_freq * t) + 0.4 * np.sin(2 * np.pi * base_freq * 1.5 * t)

        # 添加噪声
        if noise_level > 0:
            signal += noise_level * np.random.randn(seq_len)

        signals.append(signal)
        labels.append(fault_type)

    return torch.FloatTensor(np.array(signals)), torch.LongTensor(labels)


class SimplePhysicsConstraint(nn.Module):
    """简化的物理约束层"""

    def __init__(self, input_dim, use_physics=True):
        super().__init__()
        self.input_dim = input_dim
        self.use_physics = use_physics

        if use_physics:
            # 物理约束：能量守恒 + 频域特性
            self.energy_scale = nn.Parameter(torch.ones(1))
            self.freq_filter = nn.Parameter(torch.ones(input_dim // 2 + 1))

    def forward(self, x):
        if not self.use_physics:
            return x

        # 1. 能量守恒约束
        energy = torch.mean(x**2, dim=-1, keepdim=True)
        x_normalized = x * self.energy_scale / torch.sqrt(energy + 1e-8)

        # 2. 频域滤波（模拟物理特性）
        x_fft = torch.fft.rfft(x_normalized, dim=-1)
        x_fft_filtered = x_fft * self.freq_filter
        x_filtered = torch.fft.irfft(x_fft_filtered, n=self.input_dim, dim=-1)

        return x_filtered

    def compute_physics_loss(self, x, x_out):
        if not self.use_physics:
            return torch.tensor(0.0)

        # 能量守恒损失
        input_energy = torch.mean(x**2)
        output_energy = torch.mean(x_out**2)
        energy_loss = torch.abs(input_energy - output_energy)

        # 平滑性约束（物理信号通常平滑）
        smooth_loss = torch.mean(torch.diff(x_out, dim=-1)**2)

        return energy_loss + 0.01 * smooth_loss


class SimpleModel(nn.Module):
    """简化的分类模型"""

    def __init__(self, input_dim, num_classes, use_physics=False):
        super().__init__()
        self.physics_constraint = SimplePhysicsConstraint(input_dim, use_physics)

        # 根据是否使用物理约束调整网络结构
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.use_physics = use_physics

    def forward(self, x):
        # 应用物理约束
        if self.use_physics:
            x = self.physics_constraint(x)

        # 分类
        output = self.classifier(x)
        return output

    def compute_loss(self, x, outputs, targets):
        # 分类损失
        ce_loss = F.cross_entropy(outputs, targets)

        # 物理约束损失
        if self.use_physics:
            physics_loss = self.physics_constraint.compute_physics_loss(x, x)
            return ce_loss + 0.1 * physics_loss
        else:
            return ce_loss


def train_and_evaluate(model, train_data, train_labels, test_data, test_labels,
                      num_epochs=20, seed=42):
    """训练并评估模型"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 准备数据
    dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练
    model.train()
    for epoch in range(num_epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()

            outputs = model(batch_x)

            # 计算损失
            if isinstance(model, SimpleModel) and model.use_physics:
                loss = model.compute_loss(batch_x, outputs, batch_y)
            else:
                loss = F.cross_entropy(outputs, batch_y)

            loss.backward()
            optimizer.step()

    # 评估
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_data)
        _, predicted = torch.max(test_outputs.data, 1)
        accuracy = (predicted == test_labels).float().mean().item()

    return accuracy


def run_proposition2_experiment():
    """运行命题2实验"""
    print("\n=== 命题2验证：物理同构增强鲁棒性 ===\n")

    # 实验配置
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2]
    seeds = [20, 42, 100]
    num_epochs = 30

    # 存储结果
    results = {
        'standard': {'accuracies': [], 'stds': []},
        'physics_informed': {'accuracies': [], 'stds': []}
    }

    for noise_level in noise_levels:
        print(f"噪声水平: {noise_level}")

        standard_accs = []
        physics_accs = []

        for seed in seeds:
            # 生成数据
            train_data, train_labels = generate_synthetic_data(800, 50, 0.0)
            test_data, test_labels = generate_synthetic_data(200, 50, noise_level)

            # 训练标准模型
            model_standard = SimpleModel(50, 4, use_physics=False)
            acc_standard = train_and_evaluate(
                model_standard, train_data, train_labels,
                test_data, test_labels, num_epochs, seed
            )
            standard_accs.append(acc_standard)

            # 训练物理信息模型
            model_physics = SimpleModel(50, 4, use_physics=True)
            acc_physics = train_and_evaluate(
                model_physics, train_data, train_labels,
                test_data, test_labels, num_epochs, seed
            )
            physics_accs.append(acc_physics)

        # 计算平均和标准差
        avg_std = np.mean(standard_accs)
        std_std = np.std(standard_accs)
        avg_phy = np.mean(physics_accs)
        std_phy = np.std(physics_accs)

        results['standard']['accuracies'].append(avg_std)
        results['standard']['stds'].append(std_std)
        results['physics_informed']['accuracies'].append(avg_phy)
        results['physics_informed']['stds'].append(std_phy)

        print(f"  标准模型: {avg_std:.4f} ± {std_std:.4f}")
        print(f"  物理模型: {avg_phy:.4f} ± {std_phy:.4f}")
        print()

    # 计算噪声敏感性（准确率下降率）
    std_sensitivity = (results['standard']['accuracies'][0] - results['standard']['accuracies'][-1]) / noise_levels[-1]
    phy_sensitivity = (results['physics_informed']['accuracies'][0] - results['physics_informed']['accuracies'][-1]) / noise_levels[-1]

    print("=== 实验结果总结 ===")
    print(f"标准模型噪声敏感性: {std_sensitivity:.4f}")
    print(f"物理模型噪声敏感性: {phy_sensitivity:.4f}")
    print(f"改进程度: {((std_sensitivity - phy_sensitivity) / std_sensitivity * 100):.1f}%")

    # 可视化结果
    plt.figure(figsize=(10, 6))

    # 绘制准确率曲线
    x = noise_levels
    plt.errorbar(x, results['standard']['accuracies'], yerr=results['standard']['stds'],
                label='标准模型', marker='o', capsize=5)
    plt.errorbar(x, results['physics_informed']['accuracies'], yerr=results['physics_informed']['stds'],
                label='物理信息模型', marker='s', capsize=5)

    plt.xlabel('噪声水平')
    plt.ylabel('准确率')
    plt.title('物理同构对模型鲁棒性的影响')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存图像
    os.makedirs('experiments/results/proposition2_12_14', exist_ok=True)
    plt.savefig('experiments/results/proposition2_12_14/simple_validation.png', dpi=300)
    plt.show()

    # 保存结果
    output = {
        'timestamp': datetime.now().isoformat(),
        'noise_levels': noise_levels,
        'results': results,
        'sensitivity': {
            'standard': std_sensitivity,
            'physics_informed': phy_sensitivity,
            'improvement_percent': (std_sensitivity - phy_sensitivity) / std_sensitivity * 100
        }
    }

    with open('experiments/results/proposition2_12_14/simple_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n结果已保存到 experiments/results/proposition2_12_14/simple_results.json")

    return results


if __name__ == "__main__":
    results = run_proposition2_experiment()