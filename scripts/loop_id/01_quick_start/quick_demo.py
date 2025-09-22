#!/usr/bin/env python3
"""
ContrastiveIDTask 5分钟快速演示

快速验证ContrastiveIDTask的核心功能，包括：
- 模拟数据生成和预处理
- 对比学习任务初始化
- InfoNCE损失计算演示
- 基础训练循环展示
- 结果可视化

本演示使用模拟数据，无需真实数据集，适合：
- 新用户快速了解系统功能
- 开发环境验证
- 算法原理演示

Usage:
    # 标准5分钟演示
    python quick_demo.py

    # 简化版演示（1分钟）
    python quick_demo.py --fast

    # 详细演示包含可视化
    python quick_demo.py --verbose --plot

Author: PHM-Vibench Team
Version: 1.0 (Quick Validation Demo)
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import warnings

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# 抑制警告以保持演示清洁
warnings.filterwarnings('ignore')

class MockContrastiveIDTask(nn.Module):
    """模拟的ContrastiveIDTask用于演示

    实现核心的对比学习功能：
    1. 特征编码器
    2. InfoNCE损失计算
    3. 对比学习准确率计算
    """

    def __init__(self, input_dim: int = 1024, d_model: int = 128, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

        # 简化的编码器网络
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 投影头（可选）
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 2)
        )

        print(f"✅ 模型初始化完成: 输入维度={input_dim}, 特征维度={d_model}, 温度={temperature}")

    def forward(self, anchor, positive):
        """前向传播"""
        # 特征提取
        z_anchor = self.encoder(anchor)
        z_positive = self.encoder(positive)

        # 投影
        z_anchor = self.projection_head(z_anchor)
        z_positive = self.projection_head(z_positive)

        return z_anchor, z_positive

    def infonce_loss(self, z_anchor, z_positive):
        """计算InfoNCE对比损失"""
        batch_size = z_anchor.size(0)

        # L2归一化
        z_anchor = F.normalize(z_anchor, dim=1)
        z_positive = F.normalize(z_positive, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.mm(z_anchor, z_positive.t()) / self.temperature

        # 创建标签（对角线为正样本）
        labels = torch.arange(batch_size, device=z_anchor.device)

        # 计算交叉熵损失
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss

    def compute_accuracy(self, z_anchor, z_positive):
        """计算对比学习准确率"""
        with torch.no_grad():
            z_anchor = F.normalize(z_anchor, dim=1)
            z_positive = F.normalize(z_positive, dim=1)

            similarity_matrix = torch.mm(z_anchor, z_positive.t()) / self.temperature
            predictions = torch.argmax(similarity_matrix, dim=1)
            labels = torch.arange(len(z_anchor), device=z_anchor.device)

            accuracy = (predictions == labels).float().mean()

        return accuracy

class QuickDemo:
    """ContrastiveIDTask快速演示类"""

    def __init__(self, fast_mode: bool = False, verbose: bool = False, enable_plot: bool = False):
        self.fast_mode = fast_mode
        self.verbose = verbose
        self.enable_plot = enable_plot

        # 演示参数
        if fast_mode:
            self.demo_params = {
                'batch_size': 16,
                'window_size': 512,
                'num_epochs': 3,
                'num_batches_per_epoch': 5,
                'd_model': 64
            }
            print("🚀 快速演示模式（约1分钟）")
        else:
            self.demo_params = {
                'batch_size': 32,
                'window_size': 1024,
                'num_epochs': 10,
                'num_batches_per_epoch': 10,
                'd_model': 128
            }
            print("🚀 标准演示模式（约5分钟）")

        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 使用设备: {self.device}")

        # 结果记录
        self.training_history = {
            'epochs': [],
            'losses': [],
            'accuracies': [],
            'batch_times': []
        }

    def generate_mock_data(self):
        """生成模拟的振动信号数据"""
        print("\n📊 生成模拟振动信号数据...")

        batch_size = self.demo_params['batch_size']
        window_size = self.demo_params['window_size']
        num_batches = self.demo_params['num_batches_per_epoch']

        # 模拟不同类型的振动信号
        def create_vibration_signal(signal_type, length):
            """创建不同类型的模拟振动信号"""
            t = np.linspace(0, 1, length)

            if signal_type == 'normal':
                # 正常信号：低幅度噪声
                signal = 0.1 * np.random.randn(length) + 0.05 * np.sin(2 * np.pi * 10 * t)
            elif signal_type == 'bearing_fault':
                # 轴承故障：周期性冲击
                signal = 0.2 * np.random.randn(length)
                impact_freq = 50  # 50Hz冲击频率
                for i in range(0, length, length // impact_freq):
                    if i < length - 20:
                        signal[i:i+20] += 0.8 * np.exp(-np.arange(20) / 5)
            elif signal_type == 'gear_fault':
                # 齿轮故障：谐波成分
                signal = 0.15 * np.random.randn(length)
                signal += 0.3 * np.sin(2 * np.pi * 25 * t)  # 基频
                signal += 0.2 * np.sin(2 * np.pi * 50 * t)  # 二次谐波
                signal += 0.1 * np.sin(2 * np.pi * 75 * t)  # 三次谐波
            else:
                # 默认随机信号
                signal = 0.2 * np.random.randn(length)

            return signal

        # 生成训练数据
        all_anchors = []
        all_positives = []

        signal_types = ['normal', 'bearing_fault', 'gear_fault']

        for batch_idx in range(num_batches):
            batch_anchors = []
            batch_positives = []

            for i in range(batch_size):
                # 随机选择信号类型
                signal_type = np.random.choice(signal_types)

                # 生成同一ID的两个不同窗口（正样本对）
                base_signal = create_vibration_signal(signal_type, window_size * 2)

                # 随机选择两个不重叠的窗口
                start1 = np.random.randint(0, window_size // 2)
                start2 = np.random.randint(window_size, window_size + window_size // 2)

                anchor = base_signal[start1:start1 + window_size]
                positive = base_signal[start2:start2 + window_size]

                batch_anchors.append(anchor)
                batch_positives.append(positive)

            all_anchors.append(np.array(batch_anchors))
            all_positives.append(np.array(batch_positives))

        # 转换为PyTorch张量
        anchor_data = torch.FloatTensor(np.concatenate(all_anchors, axis=0))
        positive_data = torch.FloatTensor(np.concatenate(all_positives, axis=0))

        print(f"✅ 数据生成完成: 锚点数据 {anchor_data.shape}, 正样本数据 {positive_data.shape}")

        return anchor_data, positive_data

    def create_dataloader(self, anchor_data, positive_data):
        """创建数据加载器"""
        dataset = TensorDataset(anchor_data, positive_data)
        dataloader = DataLoader(
            dataset,
            batch_size=self.demo_params['batch_size'],
            shuffle=True,
            num_workers=0  # 演示中使用单线程避免复杂性
        )

        print(f"✅ 数据加载器创建完成: {len(dataset)}个样本, 批大小={self.demo_params['batch_size']}")
        return dataloader

    def run_training_demo(self, dataloader):
        """运行训练演示"""
        print(f"\n🚀 开始对比学习训练演示...")

        # 初始化模型
        model = MockContrastiveIDTask(
            input_dim=self.demo_params['window_size'],
            d_model=self.demo_params['d_model']
        ).to(self.device)

        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        start_time = time.time()

        for epoch in range(self.demo_params['num_epochs']):
            epoch_losses = []
            epoch_accuracies = []
            epoch_start = time.time()

            for batch_idx, (anchor, positive) in enumerate(dataloader):
                if batch_idx >= self.demo_params['num_batches_per_epoch']:
                    break

                batch_start = time.time()

                # 数据转移到设备
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)

                # 前向传播
                z_anchor, z_positive = model(anchor, positive)

                # 计算损失
                loss = model.infonce_loss(z_anchor, z_positive)

                # 计算准确率
                accuracy = model.compute_accuracy(z_anchor, z_positive)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 记录结果
                epoch_losses.append(loss.item())
                epoch_accuracies.append(accuracy.item())

                batch_time = time.time() - batch_start

                if self.verbose or batch_idx % 3 == 0:
                    print(f"  Epoch {epoch+1:2d}, Batch {batch_idx+1:2d}: "
                          f"Loss={loss.item():.4f}, Acc={accuracy.item():.3f}, "
                          f"Time={batch_time*1000:.0f}ms")

            # 记录每个epoch的统计
            epoch_time = time.time() - epoch_start
            avg_loss = np.mean(epoch_losses)
            avg_accuracy = np.mean(epoch_accuracies)

            self.training_history['epochs'].append(epoch + 1)
            self.training_history['losses'].append(avg_loss)
            self.training_history['accuracies'].append(avg_accuracy)
            self.training_history['batch_times'].append(epoch_time)

            print(f"📊 Epoch {epoch+1:2d}: Loss={avg_loss:.4f}, "
                  f"Acc={avg_accuracy:.3f}, Time={epoch_time:.1f}s")

        total_time = time.time() - start_time
        print(f"\n✅ 训练完成! 总耗时: {total_time:.1f}秒")

        return model

    def evaluate_model(self, model, dataloader):
        """评估模型性能"""
        print(f"\n📈 模型评估...")

        model.eval()
        test_losses = []
        test_accuracies = []
        feature_similarities = []

        with torch.no_grad():
            for batch_idx, (anchor, positive) in enumerate(dataloader):
                if batch_idx >= 3:  # 只评估前3个批次
                    break

                anchor = anchor.to(self.device)
                positive = positive.to(self.device)

                z_anchor, z_positive = model(anchor, positive)
                loss = model.infonce_loss(z_anchor, z_positive)
                accuracy = model.compute_accuracy(z_anchor, z_positive)

                test_losses.append(loss.item())
                test_accuracies.append(accuracy.item())

                # 计算正样本对的相似度
                z_anchor_norm = F.normalize(z_anchor, dim=1)
                z_positive_norm = F.normalize(z_positive, dim=1)
                similarities = torch.sum(z_anchor_norm * z_positive_norm, dim=1)
                feature_similarities.extend(similarities.cpu().numpy())

        avg_test_loss = np.mean(test_losses)
        avg_test_accuracy = np.mean(test_accuracies)
        avg_similarity = np.mean(feature_similarities)

        print(f"✅ 测试结果: Loss={avg_test_loss:.4f}, "
              f"Acc={avg_test_accuracy:.3f}, "
              f"平均正样本相似度={avg_similarity:.3f}")

        return {
            'test_loss': avg_test_loss,
            'test_accuracy': avg_test_accuracy,
            'average_similarity': avg_similarity,
            'similarity_distribution': feature_similarities
        }

    def visualize_results(self, evaluation_results):
        """可视化结果"""
        if not self.enable_plot:
            return

        print(f"\n📊 生成可视化图表...")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('ContrastiveIDTask Demo Results', fontsize=16)

        # 1. 训练损失曲线
        axes[0, 0].plot(self.training_history['epochs'], self.training_history['losses'], 'b-o')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)

        # 2. 训练准确率曲线
        axes[0, 1].plot(self.training_history['epochs'], self.training_history['accuracies'], 'g-o')
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True)

        # 3. 批处理时间
        axes[1, 0].bar(self.training_history['epochs'], self.training_history['batch_times'])
        axes[1, 0].set_title('Epoch Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')

        # 4. 特征相似度分布
        similarities = evaluation_results['similarity_distribution']
        axes[1, 1].hist(similarities, bins=20, alpha=0.7, color='orange')
        axes[1, 1].axvline(evaluation_results['average_similarity'], color='red', linestyle='--',
                          label=f'Mean: {evaluation_results["average_similarity"]:.3f}')
        axes[1, 1].set_title('Feature Similarity Distribution')
        axes[1, 1].set_xlabel('Cosine Similarity')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()

        plt.tight_layout()

        # 保存图像
        output_dir = Path(__file__).parent
        plot_file = output_dir / f"demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"📈 图表已保存: {plot_file}")

        plt.show()

    def run_complete_demo(self):
        """运行完整演示"""
        demo_start_time = time.time()

        print("🎯 ContrastiveIDTask 快速演示开始")
        print("=" * 60)

        try:
            # 1. 生成模拟数据
            anchor_data, positive_data = self.generate_mock_data()

            # 2. 创建数据加载器
            dataloader = self.create_dataloader(anchor_data, positive_data)

            # 3. 训练演示
            model = self.run_training_demo(dataloader)

            # 4. 模型评估
            evaluation_results = self.evaluate_model(model, dataloader)

            # 5. 结果可视化
            if self.enable_plot:
                self.visualize_results(evaluation_results)

            # 6. 演示总结
            demo_end_time = time.time()
            total_demo_time = demo_end_time - demo_start_time

            print(f"\n🎉 演示完成总结")
            print("=" * 60)
            print(f"⏱️  总演示时间: {total_demo_time:.1f}秒")
            print(f"🏃 训练轮数: {self.demo_params['num_epochs']}")
            print(f"📊 最终训练损失: {self.training_history['losses'][-1]:.4f}")
            print(f"🎯 最终训练准确率: {self.training_history['accuracies'][-1]:.3f}")
            print(f"✅ 测试损失: {evaluation_results['test_loss']:.4f}")
            print(f"✅ 测试准确率: {evaluation_results['test_accuracy']:.3f}")

            print(f"\n💡 关键学习点:")
            print(f"   • InfoNCE损失有效优化了特征表示")
            print(f"   • 对比学习成功学习到了信号间的相似性")
            print(f"   • 模型在{self.device}上运行稳定")

            if evaluation_results['test_accuracy'] > 0.5:
                print(f"   🎊 演示结果良好！对比学习效果显著")
            else:
                print(f"   ⚠️  准确率偏低，实际应用中需要更多数据和调优")

            print(f"\n🔗 后续步骤:")
            print(f"   1. 尝试真实数据集训练: python main.py --config configs/id_contrastive/debug.yaml")
            print(f"   2. 运行消融实验: python scripts/loop_id/03_experiments/ablation_study.py --quick")
            print(f"   3. 性能基准测试: python scripts/loop_id/04_analysis/performance_benchmark.py --quick")

            return True

        except Exception as e:
            print(f"\n❌ 演示过程中出现错误: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ContrastiveIDTask 5分钟快速演示")

    parser.add_argument('--fast', action='store_true',
                       help='快速模式（约1分钟）')
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出模式')
    parser.add_argument('--plot', action='store_true',
                       help='生成结果可视化图表')

    args = parser.parse_args()

    # 创建演示实例
    demo = QuickDemo(
        fast_mode=args.fast,
        verbose=args.verbose,
        enable_plot=args.plot
    )

    try:
        # 运行完整演示
        success = demo.run_complete_demo()

        if success:
            return 0
        else:
            return 1

    except KeyboardInterrupt:
        print(f"\n⚠️ 演示被用户中断")
        return 130
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        return 1

if __name__ == "__main__":
    exit(main())