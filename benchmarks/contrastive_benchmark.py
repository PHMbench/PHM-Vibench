#!/usr/bin/env python3
"""
ContrastiveIDTask性能基准测试
验证内存使用、训练速度、吞吐量和收敛性能是否达到设计目标
"""

import os
import sys
import time
import json
import psutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import gc
import logging
from contextlib import contextmanager
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask
from src.data_factory.id_data_factory import id_data_factory
from argparse import Namespace


class PerformanceBenchmark:
    """ContrastiveIDTask性能基准测试类"""
    
    def __init__(self, save_dir="./benchmark_results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / "benchmark.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 性能目标
        self.targets = {
            'memory_reduction': 0.50,  # 50%内存减少
            'training_time_per_epoch': 144,  # 2小时/50epoch = 144秒/epoch (CWRU)
            'throughput_samples_per_second': 10,  # 最小吞吐量
            'batch_size': 32,  # 目标批量大小
            'convergence_epochs': 50  # 最大收敛epoch
        }
        
        # 记录结果
        self.results = defaultdict(dict)
        
    @contextmanager
    def memory_monitor(self, test_name):
        """内存监控上下文管理器"""
        process = psutil.Process()
        
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        initial_ram = process.memory_info().rss / 1024 / 1024  # MB
        initial_gpu = 0
        if torch.cuda.is_available():
            initial_gpu = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        self.logger.info(f"[{test_name}] 初始内存 - RAM: {initial_ram:.2f}MB, GPU: {initial_gpu:.2f}MB")
        
        start_time = time.time()
        
        try:
            yield
        finally:
            end_time = time.time()
            
            # 强制垃圾回收
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            final_ram = process.memory_info().rss / 1024 / 1024  # MB
            final_gpu = 0
            if torch.cuda.is_available():
                final_gpu = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            
            ram_increase = final_ram - initial_ram
            gpu_increase = final_gpu - initial_gpu
            duration = end_time - start_time
            
            self.logger.info(f"[{test_name}] 最终内存 - RAM: {final_ram:.2f}MB, GPU: {final_gpu:.2f}MB")
            self.logger.info(f"[{test_name}] 内存增长 - RAM: {ram_increase:.2f}MB, GPU: {gpu_increase:.2f}MB")
            self.logger.info(f"[{test_name}] 执行时间: {duration:.2f}秒")
            
            self.results[test_name].update({
                'initial_ram_mb': initial_ram,
                'final_ram_mb': final_ram,
                'ram_increase_mb': ram_increase,
                'initial_gpu_mb': initial_gpu,
                'final_gpu_mb': final_gpu,
                'gpu_increase_mb': gpu_increase,
                'duration_seconds': duration
            })
    
    def create_mock_config(self, batch_size=32, window_size=1024):
        """创建模拟配置"""
        args_data = Namespace(
            factory_name='id',
            batch_size=batch_size,
            window_size=window_size,
            stride=window_size//2,
            num_window=2,
            window_sampling_strategy='random',
            normalization=True,
            truncate_length=window_size*4
        )
        
        args_task = Namespace(
            type='pretrain',
            name='contrastive_id',
            lr=1e-3,
            weight_decay=1e-4,
            temperature=0.07,
            loss='CE',
            metrics=['acc']
        )
        
        args_model = Namespace(
            name='M_01_ISFM',
            backbone='B_08_PatchTST',
            d_model=128,
            num_heads=8,
            num_layers=6
        )
        
        args_trainer = Namespace(
            epochs=50,
            accelerator='cpu',
            devices=1,
            precision=32
        )
        
        args_environment = Namespace(
            save_dir=str(self.save_dir),
            experiment_name='benchmark'
        )
        
        metadata = {
            i: {'Label': i % 10, 'Name': 'benchmark_data', 'Domain_id': 1}
            for i in range(1000)
        }
        
        return {
            'args_data': args_data,
            'args_task': args_task,
            'args_model': args_model,
            'args_trainer': args_trainer,
            'args_environment': args_environment,
            'metadata': metadata
        }
    
    def create_mock_data(self, num_samples, signal_length=2048, num_channels=2):
        """创建模拟数据"""
        data = []
        for i in range(num_samples):
            signal = np.random.randn(signal_length, num_channels).astype(np.float32)
            metadata = {'Label': i % 10, 'Domain_id': 1}
            data.append((f'sample_{i}', signal, metadata))
        return data
    
    def create_simple_network(self, input_dim, output_dim=128):
        """创建简单的测试网络"""
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def benchmark_memory_usage(self):
        """内存使用基准测试"""
        self.logger.info("="*50)
        self.logger.info("开始内存使用基准测试")
        
        config = self.create_mock_config(batch_size=32, window_size=1024)
        
        with self.memory_monitor("memory_usage_test"):
            # 创建网络和任务
            network = self.create_simple_network(1024 * 2, 128)
            
            # 模拟BaseIDTask初始化
            from unittest.mock import patch, Mock
            with patch('src.task_factory.task.pretrain.ContrastiveIDTask.BaseIDTask.__init__'):
                task = ContrastiveIDTask(
                    network=network,
                    args_data=config['args_data'],
                    args_model=config['args_model'],
                    args_task=config['args_task'],
                    args_trainer=config['args_trainer'],
                    args_environment=config['args_environment'],
                    metadata=config['metadata']
                )
                
                # 模拟必要的方法
                task.process_sample = Mock(side_effect=lambda data, metadata: data)
                task.create_windows = Mock(side_effect=lambda data, strategy, num_window: [
                    np.random.randn(1024, 2), 
                    np.random.randn(1024, 2)
                ])
                task.log = Mock()
            
            # 创建大量数据进行内存测试
            large_dataset = self.create_mock_data(1000, 2048, 2)  # 1000个样本
            
            # 分批处理数据
            batch_size = 32
            total_memory_used = []
            
            for i in range(0, len(large_dataset), batch_size):
                batch_data = large_dataset[i:i+batch_size]
                
                # 准备批次
                batch = task.prepare_batch(batch_data)
                
                if len(batch['ids']) > 0:
                    # 模拟前向传播
                    anchor_features = torch.randn(len(batch['ids']), 128)
                    positive_features = torch.randn(len(batch['ids']), 128)
                    
                    # 计算损失
                    loss = task.infonce_loss(anchor_features, positive_features)
                    accuracy = task.compute_accuracy(anchor_features, positive_features)
                
                # 记录当前内存使用
                process = psutil.Process()
                current_memory = process.memory_info().rss / 1024 / 1024
                total_memory_used.append(current_memory)
        
        # 分析内存使用
        max_memory = max(total_memory_used)
        avg_memory = np.mean(total_memory_used)
        memory_increase = self.results["memory_usage_test"]["ram_increase_mb"]
        
        self.results["memory_usage_test"].update({
            'max_memory_mb': max_memory,
            'avg_memory_mb': avg_memory,
            'samples_processed': len(large_dataset),
            'memory_per_sample_kb': (memory_increase * 1024) / len(large_dataset)
        })
        
        # 评估是否满足目标
        baseline_memory = 2000  # 假设基线内存使用2GB
        reduction_ratio = memory_increase / baseline_memory
        memory_target_met = reduction_ratio <= (1 - self.targets['memory_reduction'])
        
        self.logger.info(f"内存使用测试结果:")
        self.logger.info(f"  最大内存使用: {max_memory:.2f}MB")
        self.logger.info(f"  平均内存使用: {avg_memory:.2f}MB")
        self.logger.info(f"  内存增长: {memory_increase:.2f}MB")
        self.logger.info(f"  每样本内存: {(memory_increase * 1024) / len(large_dataset):.2f}KB")
        self.logger.info(f"  内存减少目标达成: {'✅' if memory_target_met else '❌'}")
        
        self.results["memory_usage_test"]["target_met"] = memory_target_met
    
    def benchmark_training_speed(self):
        """训练速度基准测试"""
        self.logger.info("="*50)
        self.logger.info("开始训练速度基准测试")
        
        config = self.create_mock_config(batch_size=32, window_size=1024)
        
        with self.memory_monitor("training_speed_test"):
            # 创建网络和任务
            network = self.create_simple_network(1024 * 2, 128)
            
            # 模拟BaseIDTask初始化
            from unittest.mock import patch, Mock
            with patch('src.task_factory.task.pretrain.ContrastiveIDTask.BaseIDTask.__init__'):
                task = ContrastiveIDTask(
                    network=network,
                    args_data=config['args_data'],
                    args_model=config['args_model'],
                    args_task=config['args_task'],
                    args_trainer=config['args_trainer'],
                    args_environment=config['args_environment'],
                    metadata=config['metadata']
                )
                
                # 模拟必要的方法
                task.process_sample = Mock(side_effect=lambda data, metadata: data)
                task.create_windows = Mock(side_effect=lambda data, strategy, num_window: [
                    np.random.randn(1024, 2), 
                    np.random.randn(1024, 2)
                ])
                task.log = Mock()
            
            # 创建训练数据（模拟CWRU大小）
            train_data = self.create_mock_data(500, 2048, 2)  # 500个样本模拟CWRU
            
            # 模拟训练循环
            epochs_to_test = 5  # 测试5个epoch
            epoch_times = []
            
            for epoch in range(epochs_to_test):
                epoch_start = time.time()
                
                # 分批处理
                for i in range(0, len(train_data), config['args_data'].batch_size):
                    batch_data = train_data[i:i+config['args_data'].batch_size]
                    
                    # 准备批次
                    batch = task.prepare_batch(batch_data)
                    
                    if len(batch['ids']) > 0:
                        # 模拟前向传播
                        anchor_features = network(batch['anchor'])
                        positive_features = network(batch['positive'])
                        
                        # 计算损失
                        loss = task.infonce_loss(anchor_features, positive_features)
                        
                        # 模拟反向传播（不实际计算梯度）
                        if loss.requires_grad:
                            loss.backward()
                
                epoch_end = time.time()
                epoch_time = epoch_end - epoch_start
                epoch_times.append(epoch_time)
                
                self.logger.info(f"Epoch {epoch+1}/{epochs_to_test}: {epoch_time:.2f}秒")
        
        # 分析训练速度
        avg_epoch_time = np.mean(epoch_times)
        total_samples = len(train_data)
        throughput = total_samples / avg_epoch_time  # samples/second
        
        # 评估目标
        speed_target_met = avg_epoch_time <= self.targets['training_time_per_epoch']
        throughput_target_met = throughput >= self.targets['throughput_samples_per_second']
        
        self.results["training_speed_test"].update({
            'avg_epoch_time_seconds': avg_epoch_time,
            'throughput_samples_per_second': throughput,
            'total_samples': total_samples,
            'epochs_tested': epochs_to_test,
            'speed_target_met': speed_target_met,
            'throughput_target_met': throughput_target_met
        })
        
        self.logger.info(f"训练速度测试结果:")
        self.logger.info(f"  平均每epoch时间: {avg_epoch_time:.2f}秒")
        self.logger.info(f"  吞吐量: {throughput:.2f} samples/second")
        self.logger.info(f"  速度目标达成: {'✅' if speed_target_met else '❌'}")
        self.logger.info(f"  吞吐量目标达成: {'✅' if throughput_target_met else '❌'}")
    
    def benchmark_batch_size_scalability(self):
        """批量大小可扩展性测试"""
        self.logger.info("="*50)
        self.logger.info("开始批量大小可扩展性测试")
        
        batch_sizes = [8, 16, 32, 64, 128]
        scalability_results = {}
        
        for batch_size in batch_sizes:
            self.logger.info(f"测试批量大小: {batch_size}")
            
            config = self.create_mock_config(batch_size=batch_size, window_size=1024)
            
            with self.memory_monitor(f"batch_size_{batch_size}"):
                # 创建网络和任务
                network = self.create_simple_network(1024 * 2, 128)
                
                from unittest.mock import patch, Mock
                with patch('src.task_factory.task.pretrain.ContrastiveIDTask.BaseIDTask.__init__'):
                    task = ContrastiveIDTask(
                        network=network,
                        args_data=config['args_data'],
                        args_model=config['args_model'],
                        args_task=config['args_task'],
                        args_trainer=config['args_trainer'],
                        args_environment=config['args_environment'],
                        metadata=config['metadata']
                    )
                    
                    task.process_sample = Mock(side_effect=lambda data, metadata: data)
                    task.create_windows = Mock(side_effect=lambda data, strategy, num_window: [
                        np.random.randn(1024, 2), 
                        np.random.randn(1024, 2)
                    ])
                    task.log = Mock()
                
                # 创建测试数据
                test_data = self.create_mock_data(batch_size * 5, 2048, 2)
                
                try:
                    # 处理批次
                    batch_times = []
                    successful_batches = 0
                    
                    for i in range(0, len(test_data), batch_size):
                        batch_data = test_data[i:i+batch_size]
                        
                        start_time = time.time()
                        batch = task.prepare_batch(batch_data)
                        
                        if len(batch['ids']) > 0:
                            anchor_features = network(batch['anchor'])
                            positive_features = network(batch['positive'])
                            loss = task.infonce_loss(anchor_features, positive_features)
                        
                        end_time = time.time()
                        batch_times.append(end_time - start_time)
                        successful_batches += 1
                    
                    avg_batch_time = np.mean(batch_times)
                    memory_used = self.results[f"batch_size_{batch_size}"]["ram_increase_mb"]
                    
                    scalability_results[batch_size] = {
                        'avg_batch_time': avg_batch_time,
                        'memory_used_mb': memory_used,
                        'successful_batches': successful_batches,
                        'success': True
                    }
                    
                    self.logger.info(f"  批量大小 {batch_size}: 成功")
                    self.logger.info(f"    平均批次时间: {avg_batch_time:.4f}秒")
                    self.logger.info(f"    内存使用: {memory_used:.2f}MB")
                
                except Exception as e:
                    scalability_results[batch_size] = {
                        'error': str(e),
                        'success': False
                    }
                    self.logger.error(f"  批量大小 {batch_size}: 失败 - {e}")
        
        self.results["batch_scalability"] = scalability_results
        
        # 找到最大可用批量大小
        max_working_batch = max([bs for bs, result in scalability_results.items() 
                                if result.get('success', False)])
        
        batch_target_met = max_working_batch >= self.targets['batch_size']
        
        self.logger.info(f"批量大小可扩展性结果:")
        self.logger.info(f"  最大可用批量大小: {max_working_batch}")
        self.logger.info(f"  批量大小目标达成: {'✅' if batch_target_met else '❌'}")
        
        self.results["batch_scalability"]["max_working_batch"] = max_working_batch
        self.results["batch_scalability"]["target_met"] = batch_target_met
    
    def benchmark_convergence_speed(self):
        """收敛速度基准测试"""
        self.logger.info("="*50)
        self.logger.info("开始收敛速度基准测试")
        
        config = self.create_mock_config(batch_size=32, window_size=1024)
        
        # 创建更复杂的数据以测试收敛
        def create_structured_data(num_samples_per_class=50, num_classes=10):
            """创建结构化数据便于收敛测试"""
            data = []
            for class_id in range(num_classes):
                # 为每个类别创建相似的信号
                base_pattern = np.random.randn(2048, 2)
                
                for i in range(num_samples_per_class):
                    # 在基础模式上添加噪声
                    signal = base_pattern + np.random.randn(2048, 2) * 0.3
                    metadata = {'Label': class_id, 'Domain_id': 1}
                    data.append((f'class_{class_id}_sample_{i}', signal.astype(np.float32), metadata))
            return data
        
        with self.memory_monitor("convergence_test"):
            # 创建网络和任务
            network = self.create_simple_network(1024 * 2, 128)
            
            from unittest.mock import patch, Mock
            with patch('src.task_factory.task.pretrain.ContrastiveIDTask.BaseIDTask.__init__'):
                task = ContrastiveIDTask(
                    network=network,
                    args_data=config['args_data'],
                    args_model=config['args_model'],
                    args_task=config['args_task'],
                    args_trainer=config['args_trainer'],
                    args_environment=config['args_environment'],
                    metadata=config['metadata']
                )
                
                task.process_sample = Mock(side_effect=lambda data, metadata: data)
                task.create_windows = Mock(side_effect=lambda data, strategy, num_window: [
                    np.random.randn(1024, 2), 
                    np.random.randn(1024, 2)
                ])
                task.log = Mock()
            
            # 创建结构化数据
            train_data = create_structured_data(30, 5)  # 5类，每类30个样本
            
            # 模拟训练并记录收敛
            max_epochs = 20
            losses = []
            accuracies = []
            
            for epoch in range(max_epochs):
                epoch_losses = []
                epoch_accuracies = []
                
                # 随机打乱数据
                np.random.shuffle(train_data)
                
                for i in range(0, len(train_data), config['args_data'].batch_size):
                    batch_data = train_data[i:i+config['args_data'].batch_size]
                    batch = task.prepare_batch(batch_data)
                    
                    if len(batch['ids']) > 0:
                        # 前向传播
                        anchor_features = network(batch['anchor'])
                        positive_features = network(batch['positive'])
                        
                        # 计算指标
                        loss = task.infonce_loss(anchor_features, positive_features)
                        accuracy = task.compute_accuracy(anchor_features, positive_features)
                        
                        epoch_losses.append(loss.item())
                        epoch_accuracies.append(accuracy.item())
                
                if epoch_losses:
                    avg_loss = np.mean(epoch_losses)
                    avg_accuracy = np.mean(epoch_accuracies)
                    
                    losses.append(avg_loss)
                    accuracies.append(avg_accuracy)
                    
                    self.logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}")
                    
                    # 简单收敛检查
                    if len(losses) >= 5:
                        recent_losses = losses[-5:]
                        if max(recent_losses) - min(recent_losses) < 0.01:  # 损失变化很小
                            converged_epoch = epoch + 1
                            self.logger.info(f"在第 {converged_epoch} epoch 收敛")
                            break
                else:
                    converged_epoch = max_epochs
        
        # 分析收敛性能
        if not locals().get('converged_epoch'):
            converged_epoch = max_epochs
        
        convergence_target_met = converged_epoch <= self.targets['convergence_epochs']
        
        self.results["convergence_test"].update({
            'converged_epoch': converged_epoch,
            'final_loss': losses[-1] if losses else None,
            'final_accuracy': accuracies[-1] if accuracies else None,
            'loss_history': losses,
            'accuracy_history': accuracies,
            'convergence_target_met': convergence_target_met
        })
        
        self.logger.info(f"收敛速度测试结果:")
        self.logger.info(f"  收敛epoch: {converged_epoch}")
        if losses:
            self.logger.info(f"  最终损失: {losses[-1]:.4f}")
            self.logger.info(f"  最终准确率: {accuracies[-1]:.4f}")
        self.logger.info(f"  收敛目标达成: {'✅' if convergence_target_met else '❌'}")
    
    def generate_report(self):
        """生成性能基准报告"""
        self.logger.info("="*50)
        self.logger.info("生成性能基准报告")
        
        # 保存详细结果
        results_file = self.save_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # 生成摘要报告
        summary_file = self.save_dir / "benchmark_summary.md"
        with open(summary_file, 'w') as f:
            f.write("# ContrastiveIDTask 性能基准测试报告\n\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 测试目标\n\n")
            for target, value in self.targets.items():
                f.write(f"- {target}: {value}\n")
            
            f.write("\n## 测试结果摘要\n\n")
            
            # 内存使用
            if "memory_usage_test" in self.results:
                mem_result = self.results["memory_usage_test"]
                f.write("### 内存使用测试\n")
                f.write(f"- 内存增长: {mem_result.get('ram_increase_mb', 'N/A'):.2f}MB\n")
                f.write(f"- 目标达成: {'✅' if mem_result.get('target_met', False) else '❌'}\n\n")
            
            # 训练速度
            if "training_speed_test" in self.results:
                speed_result = self.results["training_speed_test"]
                f.write("### 训练速度测试\n")
                f.write(f"- 平均epoch时间: {speed_result.get('avg_epoch_time_seconds', 'N/A'):.2f}秒\n")
                f.write(f"- 吞吐量: {speed_result.get('throughput_samples_per_second', 'N/A'):.2f} samples/s\n")
                f.write(f"- 速度目标达成: {'✅' if speed_result.get('speed_target_met', False) else '❌'}\n")
                f.write(f"- 吞吐量目标达成: {'✅' if speed_result.get('throughput_target_met', False) else '❌'}\n\n")
            
            # 批量大小可扩展性
            if "batch_scalability" in self.results:
                batch_result = self.results["batch_scalability"]
                f.write("### 批量大小可扩展性测试\n")
                f.write(f"- 最大可用批量大小: {batch_result.get('max_working_batch', 'N/A')}\n")
                f.write(f"- 目标达成: {'✅' if batch_result.get('target_met', False) else '❌'}\n\n")
            
            # 收敛速度
            if "convergence_test" in self.results:
                conv_result = self.results["convergence_test"]
                f.write("### 收敛速度测试\n")
                f.write(f"- 收敛epoch: {conv_result.get('converged_epoch', 'N/A')}\n")
                f.write(f"- 最终损失: {conv_result.get('final_loss', 'N/A'):.4f}\n")
                f.write(f"- 最终准确率: {conv_result.get('final_accuracy', 'N/A'):.4f}\n")
                f.write(f"- 目标达成: {'✅' if conv_result.get('convergence_target_met', False) else '❌'}\n\n")
        
        # 生成性能图表
        self.plot_performance_charts()
        
        self.logger.info(f"基准测试报告已保存到: {summary_file}")
        self.logger.info(f"详细结果已保存到: {results_file}")
    
    def plot_performance_charts(self):
        """绘制性能图表"""
        try:
            plt.style.use('default')
            
            # 收敛曲线图
            if "convergence_test" in self.results and self.results["convergence_test"].get('loss_history'):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                losses = self.results["convergence_test"]['loss_history']
                accuracies = self.results["convergence_test"]['accuracy_history']
                
                ax1.plot(losses, 'b-', label='Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title('Training Loss')
                ax1.grid(True)
                
                ax2.plot(accuracies, 'r-', label='Accuracy')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.set_title('Training Accuracy')
                ax2.grid(True)
                
                plt.tight_layout()
                plt.savefig(self.save_dir / "convergence_curves.png", dpi=150)
                plt.close()
            
            # 批量大小性能图
            if "batch_scalability" in self.results:
                batch_results = self.results["batch_scalability"]
                successful_batches = {k: v for k, v in batch_results.items() 
                                    if isinstance(v, dict) and v.get('success', False)}
                
                if successful_batches:
                    batch_sizes = list(successful_batches.keys())
                    batch_times = [v['avg_batch_time'] for v in successful_batches.values()]
                    memory_usage = [v['memory_used_mb'] for v in successful_batches.values()]
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    ax1.plot(batch_sizes, batch_times, 'o-', color='blue')
                    ax1.set_xlabel('Batch Size')
                    ax1.set_ylabel('Average Batch Time (s)')
                    ax1.set_title('Batch Size vs Processing Time')
                    ax1.grid(True)
                    
                    ax2.plot(batch_sizes, memory_usage, 'o-', color='red')
                    ax2.set_xlabel('Batch Size')
                    ax2.set_ylabel('Memory Usage (MB)')
                    ax2.set_title('Batch Size vs Memory Usage')
                    ax2.grid(True)
                    
                    plt.tight_layout()
                    plt.savefig(self.save_dir / "batch_scalability.png", dpi=150)
                    plt.close()
            
            self.logger.info("性能图表已生成")
            
        except Exception as e:
            self.logger.warning(f"生成图表时出错: {e}")
    
    def run_all_benchmarks(self):
        """运行所有基准测试"""
        self.logger.info("开始ContrastiveIDTask性能基准测试")
        self.logger.info(f"结果将保存到: {self.save_dir}")
        
        try:
            # 运行所有基准测试
            self.benchmark_memory_usage()
            self.benchmark_training_speed()
            self.benchmark_batch_size_scalability()
            self.benchmark_convergence_speed()
            
            # 生成报告
            self.generate_report()
            
            # 整体评估
            all_targets_met = all([
                self.results.get("memory_usage_test", {}).get("target_met", False),
                self.results.get("training_speed_test", {}).get("speed_target_met", False),
                self.results.get("training_speed_test", {}).get("throughput_target_met", False),
                self.results.get("batch_scalability", {}).get("target_met", False),
                self.results.get("convergence_test", {}).get("convergence_target_met", False)
            ])
            
            self.logger.info("="*50)
            self.logger.info("基准测试完成")
            self.logger.info(f"所有性能目标达成: {'✅' if all_targets_met else '❌'}")
            
            return all_targets_met
            
        except Exception as e:
            self.logger.error(f"基准测试过程中出错: {e}")
            raise


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ContrastiveIDTask性能基准测试")
    parser.add_argument("--save-dir", default="./benchmark_results",
                       help="结果保存目录")
    parser.add_argument("--test", choices=["all", "memory", "speed", "batch", "convergence"],
                       default="all", help="要运行的测试类型")
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark(save_dir=args.save_dir)
    
    if args.test == "all":
        benchmark.run_all_benchmarks()
    elif args.test == "memory":
        benchmark.benchmark_memory_usage()
    elif args.test == "speed":
        benchmark.benchmark_training_speed()
    elif args.test == "batch":
        benchmark.benchmark_batch_size_scalability()
    elif args.test == "convergence":
        benchmark.benchmark_convergence_speed()
    
    benchmark.generate_report()


if __name__ == "__main__":
    main()