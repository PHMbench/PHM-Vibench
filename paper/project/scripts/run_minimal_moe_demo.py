#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NNSPN-MoE 最小演示脚本
运行一个简单的MoE训练和验证，展示专家激活和路由决策
"""

import sys
import os
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# 添加路径以导入我们的MoE模块
current_dir = Path(__file__).parent
code_dir = current_dir.parent / 'code'
sys.path.insert(0, str(code_dir))

from moe_model import NNSPNMoE


class SyntheticFaultDataGenerator:
    """合成故障数据生成器

    生成具有不同物理特征的故障信号用于演示MoE系统
    """

    def __init__(self, signal_length=4096, sample_rate=12000):
        self.signal_length = signal_length
        self.sample_rate = sample_rate
        self.time = np.linspace(0, signal_length/sample_rate, signal_length)

    def generate_low_freq_fault(self, num_samples=100):
        """生成低频故障信号（如转子不平衡）"""
        signals = []
        labels = []

        for i in range(num_samples):
            # 基础低频正弦波 + 噪声
            f1 = np.random.uniform(20, 50)  # 低频
            f2 = f1 * np.random.uniform(2, 3)  # 二倍频
            f3 = f1 * np.random.uniform(3, 4)  # 三倍频

            signal = (1.0 * np.sin(2*np.pi*f1*self.time) +
                     0.5 * np.sin(2*np.pi*f2*self.time) +
                     0.3 * np.sin(2*np.pi*f3*self.time) +
                     0.1 * np.random.randn(len(self.time)))

            signals.append(signal)
            labels.append(0)  # 低频故障标签

        return np.array(signals), np.array(labels)

    def generate_harmonic_fault(self, num_samples=100):
        """生成谐波故障信号（如不对中）"""
        signals = []
        labels = []

        for i in range(num_samples):
            # 明显的谐波特征
            f_base = np.random.uniform(30, 60)
            harmonics = [1, 2, 3, 0.5]  # 基频、2倍频、3倍频、0.5倍频
            amplitudes = [1.0, 0.6, 0.3, 0.2]

            signal = np.zeros_like(self.time)
            for h, amp in zip(harmonics, amplitudes):
                signal += amp * np.sin(2*np.pi*f_base*h*self.time)

            signal += 0.05 * np.random.randn(len(self.time))
            signals.append(signal)
            labels.append(1)  # 谐波故障标签

        return np.array(signals), np.array(labels)

    def generate_impact_fault(self, num_samples=100):
        """生成冲击故障信号（如轴承故障）"""
        signals = []
        labels = []

        for i in range(num_samples):
            # 高频载波 + 低频调制（冲击）
            carrier_freq = np.random.uniform(2000, 4000)  # 高频载波
            impact_freq = np.random.uniform(50, 100)     # 冲击频率

            # 生成冲击序列
            impact_train = np.zeros_like(self.time)
            impact_times = np.arange(0, self.time[-1], 1.0/impact_freq)

            for t_impact in impact_times:
                if t_impact < self.time[-1]:
                    # 指数衰减的冲击
                    impact_idx = np.argmin(np.abs(self.time - t_impact))
                    decay_length = int(0.01 * self.sample_rate)  # 10ms衰减
                    decay_idx = min(impact_idx + decay_length, len(self.time))

                    decay_envelope = np.exp(-3 * np.linspace(0, 1, decay_idx - impact_idx))
                    impact_train[impact_idx:decay_idx] = decay_envelope

            # 高频调制
            signal = impact_train * np.sin(2*np.pi*carrier_freq*self.time)
            signal += 0.05 * np.random.randn(len(self.time))
            signals.append(signal)
            labels.append(2)  # 冲击故障标签

        return np.array(signals), np.array(labels)

    def generate_dataset(self, samples_per_class=100):
        """生成完整数据集"""
        low_signals, low_labels = self.generate_low_freq_fault(samples_per_class)
        harmonic_signals, harmonic_labels = self.generate_harmonic_fault(samples_per_class)
        impact_signals, impact_labels = self.generate_impact_fault(samples_per_class)

        signals = np.vstack([low_signals, harmonic_signals, impact_signals])
        labels = np.hstack([low_labels, harmonic_labels, impact_labels])

        # 打乱数据
        indices = np.random.permutation(len(signals))
        signals = signals[indices]
        labels = labels[indices]

        return signals, labels


class MoEDemo:
    """MoE演示主类"""

    def __init__(self, output_root=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        base_root = Path(output_root) if output_root else (Path(__file__).resolve().parent.parent / "results")
        self.output_root = base_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.results = {}

    def setup_data(self, batch_size=32, train_samples_per_class=80, test_samples_per_class=20):
        """设置数据"""
        print("🔄 生成合成故障数据...")

        # 数据生成器
        data_gen = SyntheticFaultDataGenerator()

        # 生成训练和测试数据
        train_signals, train_labels = data_gen.generate_dataset(train_samples_per_class)
        test_signals, test_labels = data_gen.generate_dataset(test_samples_per_class)

        # 转换为PyTorch张量
        train_signals = torch.FloatTensor(train_signals).to(self.device)
        train_labels = torch.LongTensor(train_labels).to(self.device)
        test_signals = torch.FloatTensor(test_signals).to(self.device)
        test_labels = torch.LongTensor(test_labels).to(self.device)

        # 创建数据加载器
        train_dataset = TensorDataset(train_signals, train_labels)
        test_dataset = TensorDataset(test_signals, test_labels)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"✅ 数据设置完成:")
        print(f"   训练集: {len(train_signals)} 样本")
        print(f"   测试集: {len(test_signals)} 样本")
        print(f"   信号长度: {train_signals.shape[1]}")
        print(f"   类别数: 3 (低频、谐波、冲击)")

    def setup_model(self, num_classes=3, feature_dim=64):
        """设置模型"""
        print("🤖 初始化NNSPN-MoE模型...")

        self.model = NNSPNMoE(
            num_classes=num_classes,
            feature_dim=feature_dim,
            use_load_balance=True,
            use_sparsity=True,
            routing_temperature=1.0
        ).to(self.device)

        print(f"✅ 模型设置完成:")
        model_desc = self.model.get_model_description()
        print(f"   专家数量: {model_desc['num_experts']}")
        print(f"   特征维度: {feature_dim}")
        print(f"   类别数: {num_classes}")

    def train(self, num_epochs=50, learning_rate=0.001):
        """训练模型"""
        print("🚀 开始训练...")

        # 优化器
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # 训练历史
        train_losses = []
        train_accuracies = []
        expert_activations = []

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            batch_activations = []

            for batch_idx, (signals, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()

                # 前向传播
                logits, metadata = self.model(signals, return_explanations=True)

                # 计算分类损失
                cls_loss = criterion(logits, labels)

                # 计算正则化损失
                reg_losses = metadata['regularization_losses']
                total_reg_loss = sum(reg_losses.values())

                # 总损失
                total_loss_batch = cls_loss + 0.1 * total_reg_loss

                # 反向传播
                total_loss_batch.backward()
                optimizer.step()

                # 统计
                total_loss += total_loss_batch.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 记录专家激活
                batch_activations.append(metadata['routing_weights'].mean(dim=0).detach().cpu().numpy())

            # 计算指标
            avg_loss = total_loss / len(self.train_loader)
            accuracy = 100 * correct / total
            epoch_activations = np.mean(batch_activations, axis=0)

            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)
            expert_activations.append(epoch_activations)

            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1:3d}/{num_epochs}: "
                      f"Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

        # 保存训练历史
        self.results['train_losses'] = train_losses
        self.results['train_accuracies'] = train_accuracies
        self.results['expert_activations'] = expert_activations

        print(f"✅ 训练完成! 最终准确率: {train_accuracies[-1]:.2f}%")

    def evaluate(self):
        """评估模型"""
        print("📊 评估模型性能...")

        self.model.eval()
        all_predictions = []
        all_labels = []
        all_routing_weights = []
        all_explanations = []

        with torch.no_grad():
            for signals, labels in self.test_loader:
                logits, metadata = self.model(signals, return_explanations=True)
                _, predicted = torch.max(logits, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_routing_weights.extend(metadata['routing_weights'].cpu().numpy())
                all_explanations.extend(metadata['explanations']['sample_explanations'])

        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        print(f"✅ 测试准确率: {accuracy:.2f}%")
        print("\n📋 分类报告:")
        print(classification_report(all_labels, all_predictions,
                                   target_names=['低频故障', '谐波故障', '冲击故障']))

        # 保存结果
        self.results['test_accuracy'] = accuracy
        self.results['test_predictions'] = all_predictions
        self.results['test_labels'] = all_labels
        self.results['test_routing_weights'] = np.array(all_routing_weights)
        self.results['test_explanations'] = all_explanations

    def visualize_results(self):
        """可视化结果"""
        print("📈 生成可视化结果...")

        # 创建输出目录
        output_dir = self.output_root / 'demo_visualizations'
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 训练曲线
        plt.figure(figsize=(15, 10))

        # 训练损失和准确率
        plt.subplot(2, 3, 1)
        plt.plot(self.results['train_losses'])
        plt.title('训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(2, 3, 2)
        plt.plot(self.results['train_accuracies'])
        plt.title('训练准确率')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')

        # 2. 专家激活热力图
        plt.subplot(2, 3, 3)
        expert_act_matrix = np.array(self.results['expert_activations'])
        sns.heatmap(expert_act_matrix.T, cmap='YlOrRd', cbar=True)
        plt.title('专家激活热力图 (训练过程)')
        plt.xlabel('Epoch')
        plt.ylabel('专家')
        plt.yticks([0.5, 1.5, 2.5], ['低通', '谐波', '包络'])

        # 3. 测试样本的专家激活分布
        plt.subplot(2, 3, 4)
        test_weights = self.results['test_routing_weights']
        expert_means = np.mean(test_weights, axis=0)
        expert_stds = np.std(test_weights, axis=0)

        expert_names = ['低通专家', '谐波专家', '包络专家']
        x_pos = np.arange(len(expert_names))

        plt.bar(x_pos, expert_means, yerr=expert_stds, capsize=5, alpha=0.7)
        plt.title('测试样本专家激活分布')
        plt.xlabel('专家类型')
        plt.ylabel('平均激活权重')
        plt.xticks(x_pos, expert_names)

        # 4. 路径签名矩阵 (前20个测试样本)
        plt.subplot(2, 3, 5)
        sample_subset = min(20, len(test_weights))
        path_signature = test_weights[:sample_subset]
        sns.heatmap(path_signature.T, cmap='Blues', cbar=True)
        plt.title('路径签名矩阵 (前20个测试样本)')
        plt.xlabel('样本ID')
        plt.ylabel('专家')

        # 5. 类别-专家激活关系
        plt.subplot(2, 3, 6)
        class_names = ['低频故障', '谐波故障', '冲击故障']
        expert_names_short = ['低通', '谐波', '包络']

        # 计算每个类别的平均专家激活
        class_expert_activations = []
        for class_id in range(3):
            class_mask = np.array(self.results['test_labels']) == class_id
            class_weights = test_weights[class_mask]
            class_avg = np.mean(class_weights, axis=0)
            class_expert_activations.append(class_avg)

        class_expert_matrix = np.array(class_expert_activations)
        sns.heatmap(class_expert_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
                   xticklabels=expert_names_short, yticklabels=class_names)
        plt.title('类别-专家激活关系')
        plt.xlabel('专家')
        plt.ylabel('真实类别')

        plt.tight_layout()
        plt.savefig(output_dir / 'moe_demo_results.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 6. 打印一些解释示例
        print("\n🔍 样本解释示例:")
        for i, explanation in enumerate(self.results['test_explanations'][:3]):
            print(f"\n样本 {i+1}:")
            print(f"  信号统计: RMS={explanation['signal_statistics']['rms']:.3f}, "
                  f"峭度={explanation['signal_statistics']['kurtosis']:.2f}")
            print(f"  路由决策: {explanation['routing_decision']['selected_expert']} "
                  f"(置信度: {explanation['routing_decision']['expert_confidence']:.3f})")
            print(f"  物理解释: {explanation['physical_explanation']}")

        print(f"\n✅ 可视化结果已保存到: {output_dir}")

        route_weights = self.results['test_routing_weights']
        top_weights = np.max(route_weights, axis=1)
        entropy = -np.sum(route_weights * np.log(np.clip(route_weights, 1e-12, 1.0)), axis=1)
        summary = {
            'test_accuracy': float(self.results['test_accuracy']),
            'route_entropy': float(np.mean(entropy)),
            'top_expert_weight': float(np.mean(top_weights)),
            'figure_path': str(output_dir / 'moe_demo_results.png')
        }
        summary_path = self.output_root / 'demo_summary.json'
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f"✅ Demo summary saved to: {summary_path}")

    def run_demo(self):
        """运行完整演示"""
        print("🎯 NNSPN-MoE 最小演示开始")
        print("=" * 50)

        # 设置数据、模型、训练、评估
        self.setup_data()
        self.setup_model()
        self.train()
        self.evaluate()
        self.visualize_results()

        print("\n🎉 演示完成!")
        print("=" * 50)

        # 输出训练总结
        training_summary = self.model.get_training_summary()
        print("\n📊 训练总结:")
        for key, value in training_summary.items():
            print(f"   {key}: {value}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Run the minimal MoE demo')
    parser.add_argument('--output_root', type=str, default=None, help='Directory for demo artifacts')
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 运行演示
    demo = MoEDemo(output_root=args.output_root)
    demo.run_demo()


if __name__ == "__main__":
    main()