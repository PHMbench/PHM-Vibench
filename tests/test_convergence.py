#!/usr/bin/env python3
"""
ContrastiveIDTask收敛性测试
验证损失收敛和准确率提升
"""
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import sys
import os
from argparse import Namespace

# 添加项目路径
sys.path.append('.')

from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask


def create_synthetic_dataset(num_samples=50, save_dir="tests/test_results/"):
    """创建合成数据集用于收敛测试"""
    print("创建合成数据集...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建metadata
    metadata = {}
    data = []
    
    for i in range(num_samples):
        sample_id = f'synth_{i:03d}'
        metadata[sample_id] = {
            'Id': sample_id,
            'Label': i % 3,  # 3个类别
            'Domain_id': i % 2 + 1,  # 2个域
            'Sample_rate': 1000,
            'Sample_lenth (L)': 2048,
            'Channel (C)': 1
        }
    
    # 创建H5数据文件
    h5_path = os.path.join(save_dir, "synthetic_data.h5")
    with h5py.File(h5_path, 'w') as f:
        for sample_id, meta in metadata.items():
            length = 2048
            channels = 1
            
            # 生成具有不同模式的信号
            label = meta['Label']
            if label == 0:  # 正常信号
                signal = 0.5 * np.sin(np.linspace(0, 4*np.pi, length))
                signal += 0.1 * np.random.randn(length)
            elif label == 1:  # 故障类型1
                signal = 0.8 * np.sin(np.linspace(0, 8*np.pi, length))
                signal += 0.3 * np.sin(np.linspace(0, 2*np.pi, length))
                signal += 0.15 * np.random.randn(length)
            else:  # 故障类型2
                signal = 0.6 * np.square(np.sin(np.linspace(0, 6*np.pi, length)))
                signal += 0.2 * np.random.randn(length)
            
            signal_data = signal.reshape(-1, 1).astype(np.float32)
            f.create_dataset(sample_id, data=signal_data)
    
    print(f"✅ 合成数据集创建完成: {num_samples}个样本")
    return metadata, h5_path


class ConvergenceTracker:
    """收敛性跟踪器"""
    
    def __init__(self):
        self.losses = []
        self.accuracies = []
        self.epochs = []
    
    def update(self, epoch, loss, accuracy):
        """更新指标"""
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.accuracies.append(accuracy)
    
    def plot_curves(self, save_path="tests/test_results/convergence_curves.png"):
        """绘制收敛曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 损失曲线
        ax1.plot(self.epochs, self.losses, 'b-', linewidth=2, label='Contrastive Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Convergence')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 准确率曲线
        ax2.plot(self.epochs, self.accuracies, 'r-', linewidth=2, label='Contrastive Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy Improvement')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 收敛曲线已保存: {save_path}")
    
    def analyze_convergence(self):
        """分析收敛性"""
        if len(self.losses) < 3:
            return "数据不足，无法分析收敛性"
        
        # 分析损失下降
        initial_loss = np.mean(self.losses[:3])
        final_loss = np.mean(self.losses[-3:])
        loss_reduction = (initial_loss - final_loss) / initial_loss
        
        # 分析准确率提升
        initial_acc = np.mean(self.accuracies[:3])
        final_acc = np.mean(self.accuracies[-3:])
        acc_improvement = final_acc - initial_acc
        
        # 分析趋势
        loss_trend = "下降" if self.losses[-1] < self.losses[0] else "上升"
        acc_trend = "上升" if self.accuracies[-1] > self.accuracies[0] else "下降"
        
        analysis = {
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'loss_reduction_pct': loss_reduction * 100,
            'initial_accuracy': initial_acc,
            'final_accuracy': final_acc,
            'accuracy_improvement': acc_improvement,
            'loss_trend': loss_trend,
            'accuracy_trend': acc_trend,
            'converged': loss_reduction > 0.1 and acc_improvement > 0.05
        }
        
        return analysis


def test_training_convergence(num_epochs=10, batch_size=8):
    """测试训练收敛性"""
    print(f"\n=== 测试训练收敛性 ({num_epochs} epochs) ===")
    
    try:
        # 创建合成数据集
        metadata, h5_path = create_synthetic_dataset(40)
        
        # 配置参数
        args_data = Namespace(
            window_size=512,
            stride=256,
            num_window=2,
            window_sampling_strategy='random',
            normalization=True,
            dtype='float32'
        )
        
        args_task = Namespace(
            lr=1e-3,
            temperature=0.07,
            weight_decay=1e-4,
            loss="CE",
            metrics=["acc"]
        )
        
        args_model = Namespace(
            d_model=64,
            name="M_01_ISFM",
            backbone="B_08_PatchTST"
        )
        
        args_trainer = Namespace(
            epochs=num_epochs,
            gpus=0,
            accelerator="cpu"
        )
        
        args_environment = Namespace(
            save_dir="tests/test_results/"
        )
        
        # 创建网络和任务
        network = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512 * 1, 64),  # window_size * channels
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64)
        )
        
        task = ContrastiveIDTask(
            network=network,
            args_data=args_data,
            args_model=args_model,
            args_task=args_task,
            args_trainer=args_trainer,
            args_environment=args_environment,
            metadata=metadata
        )
        
        # 创建优化器
        optimizer = optim.AdamW(task.parameters(), lr=args_task.lr, weight_decay=args_task.weight_decay)
        
        # 准备训练数据
        batch_data = []
        with h5py.File(h5_path, 'r') as f:
            sample_ids = list(f.keys())[:batch_size * 2]  # 取足够的样本
            for sample_id in sample_ids:
                data_array = f[sample_id][:]
                meta = metadata[sample_id]
                batch_data.append((sample_id, data_array, meta))
        
        # 收敛性跟踪
        tracker = ConvergenceTracker()
        
        print("开始训练...")
        task.train()
        
        for epoch in range(num_epochs):
            # 准备批次
            batch = task.prepare_batch(batch_data)
            
            if len(batch['ids']) == 0:
                print(f"Epoch {epoch+1}: 空批次，跳过")
                continue
            
            # 前向传播
            optimizer.zero_grad()
            z_anchor = task.network(batch['anchor'])
            z_positive = task.network(batch['positive'])
            
            # 计算损失
            loss = task.infonce_loss(z_anchor, z_positive)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            with torch.no_grad():
                accuracy = task.compute_accuracy(z_anchor, z_positive)
            
            # 记录指标
            tracker.update(epoch + 1, loss.item(), accuracy.item())
            
            print(f"Epoch {epoch+1:2d}: Loss={loss.item():.4f}, Acc={accuracy.item():.4f}")
        
        # 分析收敛性
        analysis = tracker.analyze_convergence()
        
        # 绘制曲线
        tracker.plot_curves()
        
        # 输出分析结果
        print("\n收敛性分析:")
        print(f"  初始损失: {analysis['initial_loss']:.4f}")
        print(f"  最终损失: {analysis['final_loss']:.4f}")
        print(f"  损失下降: {analysis['loss_reduction_pct']:.1f}%")
        print(f"  初始准确率: {analysis['initial_accuracy']:.4f}")
        print(f"  最终准确率: {analysis['final_accuracy']:.4f}")
        print(f"  准确率提升: {analysis['accuracy_improvement']:.4f}")
        print(f"  损失趋势: {analysis['loss_trend']}")
        print(f"  准确率趋势: {analysis['accuracy_trend']}")
        print(f"  收敛状态: {'✅ 收敛' if analysis['converged'] else '⚠️ 未明显收敛'}")
        
        return analysis
        
    except Exception as e:
        import traceback
        print(f"❌ 收敛性测试失败: {e}")
        traceback.print_exc()
        return None


def test_different_temperatures():
    """测试不同温度参数的收敛性"""
    print("\n=== 测试不同温度参数的影响 ===")
    
    temperatures = [0.01, 0.07, 0.2, 0.5]
    results = {}
    
    try:
        # 创建合成数据集（复用）
        metadata, h5_path = create_synthetic_dataset(30)
        
        for temp in temperatures:
            print(f"\n测试温度 T={temp}")
            
            # 配置参数（与上面类似，只修改温度）
            args_task = Namespace(
                lr=1e-3,
                temperature=temp,
                weight_decay=1e-4,
                loss="CE",
                metrics=["acc"]
            )
            
            # ... 其他配置参数相同 ...
            args_data = Namespace(
                window_size=512, stride=256, num_window=2,
                window_sampling_strategy='random', normalization=True, dtype='float32'
            )
            
            args_model = Namespace(d_model=64, name="M_01_ISFM", backbone="B_08_PatchTST")
            args_trainer = Namespace(epochs=5, gpus=0, accelerator="cpu")
            args_environment = Namespace(save_dir="tests/test_results/")
            
            # 创建网络和任务
            network = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(512 * 1, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64)
            )
            
            task = ContrastiveIDTask(
                network=network, args_data=args_data, args_model=args_model,
                args_task=args_task, args_trainer=args_trainer,
                args_environment=args_environment, metadata=metadata
            )
            
            # 快速训练几轮
            optimizer = optim.AdamW(task.parameters(), lr=1e-3)
            
            batch_data = []
            with h5py.File(h5_path, 'r') as f:
                for sample_id in list(f.keys())[:16]:
                    batch_data.append((sample_id, f[sample_id][:], metadata[sample_id]))
            
            losses = []
            for epoch in range(5):
                batch = task.prepare_batch(batch_data)
                if len(batch['ids']) > 0:
                    optimizer.zero_grad()
                    z_anchor = task.network(batch['anchor'])
                    z_positive = task.network(batch['positive'])
                    loss = task.infonce_loss(z_anchor, z_positive)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
            
            if losses:
                results[temp] = {
                    'initial_loss': losses[0],
                    'final_loss': losses[-1],
                    'avg_loss': np.mean(losses)
                }
                print(f"  T={temp}: 初始损失={losses[0]:.3f}, 最终损失={losses[-1]:.3f}")
        
        print("\n温度参数影响分析:")
        for temp, result in results.items():
            reduction = (result['initial_loss'] - result['final_loss']) / result['initial_loss']
            print(f"  T={temp}: 平均损失={result['avg_loss']:.3f}, 下降={reduction*100:.1f}%")
        
        return results
        
    except Exception as e:
        print(f"❌ 温度测试失败: {e}")
        return {}


def run_convergence_tests():
    """运行所有收敛性测试"""
    print("开始ContrastiveIDTask收敛性测试...")
    print("=" * 60)
    
    results = {}
    
    # 测试训练收敛性
    convergence_result = test_training_convergence(num_epochs=15)
    if convergence_result:
        results['convergence'] = convergence_result
    
    # 测试温度参数影响
    temperature_results = test_different_temperatures()
    if temperature_results:
        results['temperature_analysis'] = temperature_results
    
    print("\n" + "=" * 60)
    print("收敛性测试总结:")
    
    if 'convergence' in results:
        conv = results['convergence']
        if conv['converged']:
            print("✅ 模型收敛性良好")
            print(f"   损失下降: {conv['loss_reduction_pct']:.1f}%")
            print(f"   准确率提升: {conv['accuracy_improvement']:.3f}")
        else:
            print("⚠️ 模型收敛性需要优化")
    
    if 'temperature_analysis' in results:
        print("✅ 温度参数敏感性分析完成")
        print("   建议查看输出选择最优温度参数")
    
    return results


if __name__ == "__main__":
    results = run_convergence_tests()
    
    # 保存结果（简化版本避免循环引用）
    import json
    
    # 提取关键结果
    summary_results = {}
    if 'convergence' in results:
        conv = results['convergence']
        summary_results['convergence'] = {
            'initial_loss': float(conv['initial_loss']),
            'final_loss': float(conv['final_loss']),
            'loss_reduction_pct': float(conv['loss_reduction_pct']),
            'initial_accuracy': float(conv['initial_accuracy']),
            'final_accuracy': float(conv['final_accuracy']),
            'accuracy_improvement': float(conv['accuracy_improvement']),
            'converged': bool(conv['converged'])
        }
    
    if 'temperature_analysis' in results:
        summary_results['temperature_analysis'] = {}
        for temp, result in results['temperature_analysis'].items():
            summary_results['temperature_analysis'][str(temp)] = {
                'initial_loss': float(result['initial_loss']),
                'final_loss': float(result['final_loss']),
                'avg_loss': float(result['avg_loss'])
            }
    
    with open("tests/test_results/convergence_results.json", "w") as f:
        json.dump(summary_results, f, indent=2)
    
    print(f"\n✅ 收敛性测试结果已保存: tests/test_results/convergence_results.json")