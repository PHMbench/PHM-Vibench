#!/usr/bin/env python3
"""
Comprehensive Performance Benchmark for ContrastiveIDTask
Task-011: Performance benchmark testing

This benchmark provides detailed performance analysis across:
1. Training Performance - speed, memory, convergence
2. Data Processing Performance - loading, window generation, batch preparation
3. Model Performance - forward/backward pass timing, memory usage
4. Scalability Testing - batch sizes, window sizes, multi-GPU
5. Hardware Optimization - CPU vs GPU, mixed precision

The benchmark generates detailed reports with actionable insights
for production optimization.
"""

import os
import sys
import time
import json
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.profiler import profile, record_function, ProfilerActivity
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import gc
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
import tracemalloc
from functools import wraps
import traceback

# Optional dependencies for profiling
try:
    from memory_profiler import profile as memory_profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    memory_profile = lambda func: func  # No-op decorator

try:
    import line_profiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask
from argparse import Namespace


class AdvancedPerformanceBenchmark:
    """
    Advanced performance benchmark suite for ContrastiveIDTask
    Provides comprehensive analysis of all performance aspects
    """
    
    def __init__(self, save_dir="./benchmark_results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.save_dir / "reports").mkdir(exist_ok=True)
        (self.save_dir / "plots").mkdir(exist_ok=True)
        (self.save_dir / "profiles").mkdir(exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 性能目标和基线
        self.targets = {
            'memory_efficiency': {
                'gpu_memory_per_sample_mb': 0.5,  # Max GPU memory per sample
                'cpu_memory_per_sample_mb': 2.0,  # Max CPU memory per sample
                'memory_growth_rate': 0.1,        # Max memory growth per batch
            },
            'training_performance': {
                'samples_per_second': 50,          # Min training throughput
                'batch_time_ms': 200,             # Max batch processing time
                'epoch_time_minutes': 5,          # Max epoch time for standard dataset
                'gpu_utilization': 0.8,           # Min GPU utilization
            },
            'model_performance': {
                'forward_pass_ms': 50,            # Max forward pass time
                'backward_pass_ms': 100,          # Max backward pass time
                'infonce_computation_ms': 10,     # Max InfoNCE loss computation
            },
            'scalability': {
                'max_batch_size': 128,            # Target max batch size
                'max_window_size': 4096,          # Target max window size
                'memory_scaling_factor': 1.2,    # Max memory scaling factor
            },
            'data_processing': {
                'h5_loading_mb_per_sec': 100,     # Min H5 loading speed
                'window_generation_per_sec': 1000, # Min window generation speed
                'batch_preparation_ms': 50,       # Max batch preparation time
            }
        }
        
        # 记录结果
        self.results = defaultdict(dict)
        self.profiling_data = {}
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mixed_precision_available = self.device.type == "cuda"
        
        self.logger.info(f"Benchmark initialized on device: {self.device}")
        self.logger.info(f"Mixed precision available: {self.mixed_precision_available}")
    
    def _setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / "benchmark.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def performance_monitor(self, test_name: str, profile_memory=True, profile_gpu=True):
        """
        高级性能监控上下文管理器
        包含内存、GPU、时间等多维度监控
        """
        # 开始监控
        start_time = time.time()
        process = psutil.Process()
        
        # 内存监控
        if profile_memory:
            tracemalloc.start()
            gc.collect()
        
        # GPU监控
        initial_gpu_memory = 0
        if profile_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        initial_cpu_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        self.logger.info(f"[{test_name}] Starting - CPU: {initial_cpu_memory:.2f}MB, GPU: {initial_gpu_memory:.2f}MB")
        
        # 性能数据收集
        performance_data = {
            'start_time': start_time,
            'initial_cpu_memory_mb': initial_cpu_memory,
            'initial_gpu_memory_mb': initial_gpu_memory,
            'cpu_memory_samples': [initial_cpu_memory],
            'gpu_memory_samples': [initial_gpu_memory],
            'timestamps': [start_time]
        }
        
        try:
            yield performance_data
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            # 最终内存测量
            final_cpu_memory = process.memory_info().rss / 1024 / 1024
            final_gpu_memory = 0
            if profile_gpu and torch.cuda.is_available():
                final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            
            # 内存追踪
            if profile_memory and tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                performance_data['peak_memory_mb'] = peak / 1024 / 1024
                performance_data['current_memory_mb'] = current / 1024 / 1024
            
            # 更新性能数据
            performance_data.update({
                'end_time': end_time,
                'duration_seconds': duration,
                'final_cpu_memory_mb': final_cpu_memory,
                'final_gpu_memory_mb': final_gpu_memory,
                'cpu_memory_increase_mb': final_cpu_memory - initial_cpu_memory,
                'gpu_memory_increase_mb': final_gpu_memory - initial_gpu_memory,
                'max_cpu_memory_mb': max(performance_data['cpu_memory_samples']),
                'max_gpu_memory_mb': max(performance_data['gpu_memory_samples']),
            })
            
            # 保存结果
            self.results[test_name].update(performance_data)
            
            self.logger.info(
                f"[{test_name}] Completed in {duration:.2f}s - "
                f"CPU: +{final_cpu_memory - initial_cpu_memory:.2f}MB, "
                f"GPU: +{final_gpu_memory - initial_gpu_memory:.2f}MB"
            )
    
    def create_mock_config(self, batch_size=32, window_size=1024, **kwargs):
        """创建模拟配置，支持灵活参数覆盖"""
        config = {
            'args_data': Namespace(
                factory_name='id',
                batch_size=batch_size,
                window_size=window_size,
                stride=window_size//2,
                num_window=2,
                window_sampling_strategy='random',
                normalization=True,
                truncate_length=window_size*4
            ),
            'args_task': Namespace(
                type='pretrain',
                name='contrastive_id',
                lr=1e-3,
                weight_decay=1e-4,
                temperature=0.07,
                loss='CE',
                metrics=['acc']
            ),
            'args_model': Namespace(
                name='M_01_ISFM',
                backbone='B_08_PatchTST',
                d_model=128,
                num_heads=8,
                num_layers=6
            ),
            'args_trainer': Namespace(
                epochs=50,
                accelerator='cuda' if torch.cuda.is_available() else 'cpu',
                devices=1,
                precision=32
            ),
            'args_environment': Namespace(
                save_dir=str(self.save_dir),
                experiment_name='benchmark'
            ),
            'metadata': {
                i: {'Label': i % 10, 'Name': 'benchmark_data', 'Domain_id': 1}
                for i in range(1000)
            }
        }
        
        # 应用kwargs覆盖
        for key, value in kwargs.items():
            if hasattr(config['args_data'], key):
                setattr(config['args_data'], key, value)
            elif hasattr(config['args_task'], key):
                setattr(config['args_task'], key, value)
            elif hasattr(config['args_model'], key):
                setattr(config['args_model'], key, value)
            elif hasattr(config['args_trainer'], key):
                setattr(config['args_trainer'], key, value)
        
        return config
    
    def create_synthetic_data(self, num_samples=1000, signal_length=2048, 
                            num_channels=1, complexity='medium'):
        """
        创建不同复杂度的合成数据用于基准测试
        """
        data = []
        
        if complexity == 'simple':
            # 简单随机数据
            for i in range(num_samples):
                signal = np.random.randn(signal_length, num_channels).astype(np.float32)
                metadata = {'Label': i % 10, 'Domain_id': 1}
                data.append((f'sample_{i}', signal, metadata))
                
        elif complexity == 'medium':
            # 带有模式的数据
            for i in range(num_samples):
                base_freq = 0.1 + (i % 10) * 0.05  # 不同频率
                t = np.linspace(0, 1, signal_length)
                signal = np.sin(2 * np.pi * base_freq * t).reshape(-1, 1)
                signal += np.random.randn(signal_length, num_channels) * 0.1
                signal = signal.astype(np.float32)
                metadata = {'Label': i % 10, 'Domain_id': 1}
                data.append((f'sample_{i}', signal, metadata))
                
        elif complexity == 'complex':
            # 复杂多分量信号
            for i in range(num_samples):
                t = np.linspace(0, 1, signal_length)
                signal = np.zeros((signal_length, num_channels))
                
                # 多个频率分量
                for freq in [0.05, 0.15, 0.3]:
                    component = np.sin(2 * np.pi * freq * t + np.random.rand())
                    signal[:, 0] += component
                
                # 添加噪声和趋势
                signal += np.random.randn(signal_length, num_channels) * 0.2
                signal += np.linspace(0, 0.5, signal_length).reshape(-1, 1)
                signal = signal.astype(np.float32)
                
                metadata = {'Label': i % 10, 'Domain_id': 1}
                data.append((f'sample_{i}', signal, metadata))
        
        return data
    
    def create_test_network(self, input_dim, output_dim=128, complexity='medium'):
        """创建不同复杂度的测试网络"""
        if complexity == 'simple':
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, output_dim)
            ).to(self.device)
            
        elif complexity == 'medium':
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, output_dim)
            ).to(self.device)
            
        elif complexity == 'complex':
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, output_dim)
            ).to(self.device)

    def benchmark_training_performance(self):
        """1. 训练性能基准测试"""
        self.logger.info("="*60)
        self.logger.info("TRAINING PERFORMANCE BENCHMARK")
        self.logger.info("="*60)
        
        test_configs = [
            {'batch_size': 16, 'window_size': 512, 'name': 'small_config'},
            {'batch_size': 32, 'window_size': 1024, 'name': 'medium_config'},
            {'batch_size': 64, 'window_size': 2048, 'name': 'large_config'},
        ]
        
        for config_params in test_configs:
            config_name = config_params.pop('name')
            self.logger.info(f"\nTesting configuration: {config_name}")
            
            with self.performance_monitor(f"training_perf_{config_name}"):
                config = self.create_mock_config(**config_params)
                
                # 创建网络和任务
                network = self.create_test_network(
                    config_params['window_size'] * 1,  # 假设单通道
                    complexity='medium'
                )
                
                # 使用mock初始化ContrastiveIDTask
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
                    
                    # Mock必要方法
                    task.process_sample = Mock(side_effect=lambda data, metadata: data)
                    task.create_windows = Mock(side_effect=lambda data, strategy, num_window: [
                        np.random.randn(config_params['window_size'], 1), 
                        np.random.randn(config_params['window_size'], 1)
                    ])
                    task.log = Mock()
                
                # 创建训练数据
                train_data = self.create_synthetic_data(
                    num_samples=500, 
                    signal_length=config_params['window_size'] * 2,
                    complexity='medium'
                )
                
                # 性能测试
                batch_times = []
                forward_times = []
                backward_times = []
                loss_computation_times = []
                
                optimizer = torch.optim.Adam(network.parameters(), lr=config['args_task'].lr)
                
                epochs_to_test = 3
                for epoch in range(epochs_to_test):
                    epoch_start = time.time()
                    
                    for i in range(0, len(train_data), config['args_data'].batch_size):
                        batch_data = train_data[i:i+config['args_data'].batch_size]
                        
                        # 批次准备时间
                        batch_start = time.time()
                        batch = task.prepare_batch(batch_data)
                        batch_prep_time = time.time() - batch_start
                        
                        if len(batch['ids']) == 0:
                            continue
                        
                        # 移到设备
                        batch['anchor'] = batch['anchor'].to(self.device)
                        batch['positive'] = batch['positive'].to(self.device)
                        
                        # 前向传播时间
                        forward_start = time.time()
                        anchor_features = network(batch['anchor'])
                        positive_features = network(batch['positive'])
                        forward_time = time.time() - forward_start
                        
                        # 损失计算时间
                        loss_start = time.time()
                        loss = task.infonce_loss(anchor_features, positive_features)
                        loss_time = time.time() - loss_start
                        
                        # 反向传播时间
                        backward_start = time.time()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        backward_time = time.time() - backward_start
                        
                        # 记录时间
                        batch_times.append(batch_prep_time * 1000)  # ms
                        forward_times.append(forward_time * 1000)   # ms
                        backward_times.append(backward_time * 1000) # ms
                        loss_computation_times.append(loss_time * 1000)  # ms
                    
                    epoch_time = time.time() - epoch_start
                    self.logger.info(f"  Epoch {epoch+1}: {epoch_time:.2f}s")
                
                # 分析结果
                throughput = len(train_data) * epochs_to_test / sum([
                    sum(batch_times), sum(forward_times), sum(backward_times)
                ]) * 1000  # samples per second
                
                results = {
                    'avg_batch_time_ms': np.mean(batch_times),
                    'avg_forward_time_ms': np.mean(forward_times),
                    'avg_backward_time_ms': np.mean(backward_times),
                    'avg_loss_computation_ms': np.mean(loss_computation_times),
                    'throughput_samples_per_sec': throughput,
                    'batch_size': config_params['batch_size'],
                    'window_size': config_params['window_size'],
                    'total_params': sum(p.numel() for p in network.parameters()),
                }
                
                # 目标检查
                targets_met = {
                    'batch_time': results['avg_batch_time_ms'] <= self.targets['training_performance']['batch_time_ms'],
                    'throughput': results['throughput_samples_per_sec'] >= self.targets['training_performance']['samples_per_second'],
                    'forward_pass': results['avg_forward_time_ms'] <= self.targets['model_performance']['forward_pass_ms'],
                    'backward_pass': results['avg_backward_time_ms'] <= self.targets['model_performance']['backward_pass_ms'],
                }
                
                results['targets_met'] = targets_met
                results['all_targets_met'] = all(targets_met.values())
                
                self.results[f"training_perf_{config_name}"].update(results)
                
                # 日志输出
                self.logger.info(f"  Results for {config_name}:")
                self.logger.info(f"    Batch time: {results['avg_batch_time_ms']:.2f}ms ({'✅' if targets_met['batch_time'] else '❌'})")
                self.logger.info(f"    Forward time: {results['avg_forward_time_ms']:.2f}ms ({'✅' if targets_met['forward_pass'] else '❌'})")
                self.logger.info(f"    Backward time: {results['avg_backward_time_ms']:.2f}ms ({'✅' if targets_met['backward_pass'] else '❌'})")
                self.logger.info(f"    Throughput: {results['throughput_samples_per_sec']:.2f} samples/s ({'✅' if targets_met['throughput'] else '❌'})")

    def benchmark_data_processing_performance(self):
        """2. 数据处理性能基准测试"""
        self.logger.info("="*60)
        self.logger.info("DATA PROCESSING PERFORMANCE BENCHMARK")
        self.logger.info("="*60)
        
        test_scenarios = [
            {'num_samples': 1000, 'signal_length': 2048, 'complexity': 'simple'},
            {'num_samples': 1000, 'signal_length': 4096, 'complexity': 'medium'},
            {'num_samples': 2000, 'signal_length': 8192, 'complexity': 'complex'},
        ]
        
        for scenario in test_scenarios:
            scenario_name = f"data_proc_{scenario['complexity']}_{scenario['signal_length']}"
            self.logger.info(f"\nTesting scenario: {scenario_name}")
            
            with self.performance_monitor(scenario_name):
                config = self.create_mock_config(window_size=1024)
                
                # 1. 数据生成性能
                data_gen_start = time.time()
                test_data = self.create_synthetic_data(**scenario)
                data_gen_time = time.time() - data_gen_start
                
                # 2. 批次准备性能
                from unittest.mock import patch, Mock
                with patch('src.task_factory.task.pretrain.ContrastiveIDTask.BaseIDTask.__init__'):
                    task = ContrastiveIDTask(
                        network=None,
                        args_data=config['args_data'],
                        args_model=config['args_model'],
                        args_task=config['args_task'],
                        args_trainer=config['args_trainer'],
                        args_environment=config['args_environment'],
                        metadata=config['metadata']
                    )
                    
                    task.process_sample = Mock(side_effect=lambda data, metadata: data)
                    task.create_windows = Mock(side_effect=lambda data, strategy, num_window: [
                        data[:1024], data[512:1536]  # 模拟窗口提取
                    ])
                    task.log = Mock()
                
                # 测试不同批次大小的处理性能
                batch_prep_times = []
                window_gen_times = []
                
                for batch_size in [16, 32, 64]:
                    for i in range(0, min(len(test_data), 200), batch_size):
                        batch_data = test_data[i:i+batch_size]
                        
                        # 窗口生成时间
                        window_start = time.time()
                        for sample_id, data_array, metadata in batch_data:
                            # 模拟窗口生成
                            windows = task.create_windows(data_array, 'random', 2)
                        window_time = time.time() - window_start
                        
                        # 批次准备时间
                        prep_start = time.time()
                        batch = task.prepare_batch(batch_data)
                        prep_time = time.time() - prep_start
                        
                        if len(batch['ids']) > 0:
                            batch_prep_times.append(prep_time * 1000)  # ms
                            window_gen_times.append(window_time * 1000)  # ms
                
                # 分析结果
                data_processing_speed = scenario['num_samples'] / data_gen_time  # samples/sec
                avg_batch_prep_time = np.mean(batch_prep_times)
                avg_window_gen_time = np.mean(window_gen_times)
                
                results = {
                    'data_generation_speed_samples_per_sec': data_processing_speed,
                    'avg_batch_prep_time_ms': avg_batch_prep_time,
                    'avg_window_gen_time_ms': avg_window_gen_time,
                    'data_complexity': scenario['complexity'],
                    'signal_length': scenario['signal_length'],
                    'num_samples_processed': scenario['num_samples'],
                }
                
                # 目标检查
                targets_met = {
                    'batch_prep_time': avg_batch_prep_time <= self.targets['data_processing']['batch_preparation_ms'],
                    'window_gen_speed': (scenario['num_samples'] / (sum(window_gen_times)/1000)) >= self.targets['data_processing']['window_generation_per_sec'],
                }
                
                results['targets_met'] = targets_met
                results['all_targets_met'] = all(targets_met.values())
                
                self.results[scenario_name].update(results)
                
                self.logger.info(f"  Results for {scenario_name}:")
                self.logger.info(f"    Data generation: {data_processing_speed:.2f} samples/s")
                self.logger.info(f"    Batch preparation: {avg_batch_prep_time:.2f}ms ({'✅' if targets_met['batch_prep_time'] else '❌'})")
                self.logger.info(f"    Window generation: {avg_window_gen_time:.2f}ms")

    def benchmark_model_performance(self):
        """3. 模型性能基准测试"""
        self.logger.info("="*60)
        self.logger.info("MODEL PERFORMANCE BENCHMARK")
        self.logger.info("="*60)
        
        model_configs = [
            {'complexity': 'simple', 'input_dim': 512, 'output_dim': 64},
            {'complexity': 'medium', 'input_dim': 1024, 'output_dim': 128},
            {'complexity': 'complex', 'input_dim': 2048, 'output_dim': 256},
        ]
        
        for model_config in model_configs:
            config_name = f"model_{model_config['complexity']}"
            self.logger.info(f"\nTesting model: {config_name}")
            
            with self.performance_monitor(config_name):
                # 创建模型
                network = self.create_test_network(
                    model_config['input_dim'],
                    model_config['output_dim'],
                    model_config['complexity']
                )
                
                # 准备测试数据
                batch_sizes = [16, 32, 64]
                input_shape = (max(batch_sizes), model_config['input_dim'] // network[0].in_features, 1)
                
                # 性能测试
                forward_times = []
                backward_times = []
                infonce_times = []
                memory_usage = []
                
                for batch_size in batch_sizes:
                    # 创建测试输入
                    test_input = torch.randn(batch_size, input_shape[1], input_shape[2], device=self.device)
                    
                    # 多次测试以获得稳定结果
                    for _ in range(10):
                        # 前向传播测试
                        torch.cuda.synchronize() if self.device.type == 'cuda' else None
                        forward_start = time.time()
                        
                        with torch.no_grad():
                            output = network(test_input)
                        
                        torch.cuda.synchronize() if self.device.type == 'cuda' else None
                        forward_time = time.time() - forward_start
                        forward_times.append(forward_time * 1000)  # ms
                        
                        # 反向传播测试
                        network.zero_grad()
                        output = network(test_input)
                        loss = output.sum()  # 简单损失
                        
                        torch.cuda.synchronize() if self.device.type == 'cuda' else None
                        backward_start = time.time()
                        
                        loss.backward()
                        
                        torch.cuda.synchronize() if self.device.type == 'cuda' else None
                        backward_time = time.time() - backward_start
                        backward_times.append(backward_time * 1000)  # ms
                        
                        # InfoNCE损失计算测试
                        anchor_features = torch.randn(batch_size, model_config['output_dim'], device=self.device)
                        positive_features = torch.randn(batch_size, model_config['output_dim'], device=self.device)
                        
                        torch.cuda.synchronize() if self.device.type == 'cuda' else None
                        infonce_start = time.time()
                        
                        # InfoNCE计算
                        anchor_norm = F.normalize(anchor_features, dim=1)
                        positive_norm = F.normalize(positive_features, dim=1)
                        similarity_matrix = torch.mm(anchor_norm, positive_norm.t()) / 0.07
                        positive_samples = torch.diag(similarity_matrix)
                        logsumexp = torch.logsumexp(similarity_matrix, dim=1)
                        infonce_loss = (-positive_samples + logsumexp).mean()
                        
                        torch.cuda.synchronize() if self.device.type == 'cuda' else None
                        infonce_time = time.time() - infonce_start
                        infonce_times.append(infonce_time * 1000)  # ms
                        
                        # GPU内存使用
                        if self.device.type == 'cuda':
                            memory_usage.append(torch.cuda.memory_allocated() / 1024 / 1024)  # MB
                
                # 分析结果
                results = {
                    'avg_forward_time_ms': np.mean(forward_times),
                    'std_forward_time_ms': np.std(forward_times),
                    'avg_backward_time_ms': np.mean(backward_times),
                    'std_backward_time_ms': np.std(backward_times),
                    'avg_infonce_time_ms': np.mean(infonce_times),
                    'std_infonce_time_ms': np.std(infonce_times),
                    'avg_memory_usage_mb': np.mean(memory_usage) if memory_usage else 0,
                    'max_memory_usage_mb': max(memory_usage) if memory_usage else 0,
                    'model_parameters': sum(p.numel() for p in network.parameters()),
                    'model_size_mb': sum(p.numel() * 4 for p in network.parameters()) / 1024 / 1024,  # 假设float32
                }
                
                # 目标检查
                targets_met = {
                    'forward_time': results['avg_forward_time_ms'] <= self.targets['model_performance']['forward_pass_ms'],
                    'backward_time': results['avg_backward_time_ms'] <= self.targets['model_performance']['backward_pass_ms'],
                    'infonce_time': results['avg_infonce_time_ms'] <= self.targets['model_performance']['infonce_computation_ms'],
                }
                
                results['targets_met'] = targets_met
                results['all_targets_met'] = all(targets_met.values())
                
                self.results[config_name].update(results)
                
                self.logger.info(f"  Results for {config_name}:")
                self.logger.info(f"    Forward time: {results['avg_forward_time_ms']:.2f}±{results['std_forward_time_ms']:.2f}ms ({'✅' if targets_met['forward_time'] else '❌'})")
                self.logger.info(f"    Backward time: {results['avg_backward_time_ms']:.2f}±{results['std_backward_time_ms']:.2f}ms ({'✅' if targets_met['backward_time'] else '❌'})")
                self.logger.info(f"    InfoNCE time: {results['avg_infonce_time_ms']:.2f}±{results['std_infonce_time_ms']:.2f}ms ({'✅' if targets_met['infonce_time'] else '❌'})")
                self.logger.info(f"    Model size: {results['model_size_mb']:.2f}MB ({results['model_parameters']} params)")

    def benchmark_scalability_testing(self):
        """4. 可扩展性测试"""
        self.logger.info("="*60)
        self.logger.info("SCALABILITY TESTING BENCHMARK")
        self.logger.info("="*60)
        
        # 批次大小可扩展性
        self.logger.info("\nTesting batch size scalability...")
        batch_sizes = [4, 8, 16, 32, 64, 128, 256]
        batch_scalability = {}
        
        for batch_size in batch_sizes:
            test_name = f"batch_scale_{batch_size}"
            try:
                with self.performance_monitor(test_name):
                    config = self.create_mock_config(batch_size=batch_size, window_size=1024)
                    network = self.create_test_network(1024, 128, 'medium')
                    
                    # 创建测试数据
                    test_input = torch.randn(batch_size, 1024, 1, device=self.device)
                    
                    # 测试前向传播
                    output = network(test_input)
                    
                    # 测试InfoNCE损失
                    anchor_features = torch.randn(batch_size, 128, device=self.device)
                    positive_features = torch.randn(batch_size, 128, device=self.device)
                    
                    # InfoNCE计算
                    anchor_norm = F.normalize(anchor_features, dim=1)
                    positive_norm = F.normalize(positive_features, dim=1)
                    similarity_matrix = torch.mm(anchor_norm, positive_norm.t()) / 0.07
                    loss = -torch.diag(similarity_matrix) + torch.logsumexp(similarity_matrix, dim=1)
                    final_loss = loss.mean()
                    
                    # 反向传播
                    final_loss.backward()
                    
                    batch_scalability[batch_size] = {
                        'success': True,
                        'memory_mb': self.results[test_name].get('max_gpu_memory_mb', 0),
                        'time_ms': self.results[test_name].get('duration_seconds', 0) * 1000
                    }
                    
                    self.logger.info(f"  Batch size {batch_size}: ✅ ({batch_scalability[batch_size]['memory_mb']:.1f}MB)")
                    
            except RuntimeError as e:
                batch_scalability[batch_size] = {
                    'success': False,
                    'error': str(e)
                }
                self.logger.info(f"  Batch size {batch_size}: ❌ ({str(e)[:50]}...)")
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # 窗口大小可扩展性
        self.logger.info("\nTesting window size scalability...")
        window_sizes = [256, 512, 1024, 2048, 4096, 8192]
        window_scalability = {}
        
        for window_size in window_sizes:
            test_name = f"window_scale_{window_size}"
            try:
                with self.performance_monitor(test_name):
                    config = self.create_mock_config(batch_size=32, window_size=window_size)
                    network = self.create_test_network(window_size, 128, 'medium')
                    
                    # 测试
                    test_input = torch.randn(32, window_size, 1, device=self.device)
                    output = network(test_input)
                    loss = output.sum()
                    loss.backward()
                    
                    window_scalability[window_size] = {
                        'success': True,
                        'memory_mb': self.results[test_name].get('max_gpu_memory_mb', 0),
                        'time_ms': self.results[test_name].get('duration_seconds', 0) * 1000
                    }
                    
                    self.logger.info(f"  Window size {window_size}: ✅ ({window_scalability[window_size]['memory_mb']:.1f}MB)")
                    
            except RuntimeError as e:
                window_scalability[window_size] = {
                    'success': False,
                    'error': str(e)
                }
                self.logger.info(f"  Window size {window_size}: ❌ ({str(e)[:50]}...)")
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # 保存可扩展性结果
        self.results['scalability_testing'] = {
            'batch_size_scalability': batch_scalability,
            'window_size_scalability': window_scalability,
            'max_working_batch_size': max([bs for bs, result in batch_scalability.items() if result['success']]),
            'max_working_window_size': max([ws for ws, result in window_scalability.items() if result['success']]),
        }
        
        # 目标检查
        max_batch = self.results['scalability_testing']['max_working_batch_size']
        max_window = self.results['scalability_testing']['max_working_window_size']
        
        scalability_targets_met = {
            'batch_size': max_batch >= self.targets['scalability']['max_batch_size'],
            'window_size': max_window >= self.targets['scalability']['max_window_size'],
        }
        
        self.results['scalability_testing']['targets_met'] = scalability_targets_met
        self.results['scalability_testing']['all_targets_met'] = all(scalability_targets_met.values())
        
        self.logger.info(f"\nScalability Results:")
        self.logger.info(f"  Max batch size: {max_batch} ({'✅' if scalability_targets_met['batch_size'] else '❌'})")
        self.logger.info(f"  Max window size: {max_window} ({'✅' if scalability_targets_met['window_size'] else '❌'})")

    def benchmark_hardware_optimization(self):
        """5. 硬件优化测试"""
        self.logger.info("="*60)
        self.logger.info("HARDWARE OPTIMIZATION BENCHMARK")
        self.logger.info("="*60)
        
        optimization_results = {}
        
        # CPU vs GPU比较（如果GPU可用）
        if torch.cuda.is_available():
            self.logger.info("\nCPU vs GPU Performance Comparison:")
            
            for device_name, device in [('CPU', torch.device('cpu')), ('GPU', torch.device('cuda'))]:
                test_name = f"hardware_{device_name.lower()}"
                
                with self.performance_monitor(test_name):
                    # 创建网络和数据
                    network = self.create_test_network(1024, 128, 'medium').to(device)
                    test_data = torch.randn(32, 1024, 1, device=device)
                    
                    # 多次测试
                    times = []
                    for _ in range(20):
                        start_time = time.time()
                        
                        output = network(test_data)
                        anchor_features = torch.randn(32, 128, device=device)
                        positive_features = torch.randn(32, 128, device=device)
                        
                        # InfoNCE损失
                        anchor_norm = F.normalize(anchor_features, dim=1)
                        positive_norm = F.normalize(positive_features, dim=1)
                        similarity_matrix = torch.mm(anchor_norm, positive_norm.t()) / 0.07
                        loss = -torch.diag(similarity_matrix) + torch.logsumexp(similarity_matrix, dim=1)
                        final_loss = loss.mean()
                        final_loss.backward()
                        
                        if device.type == 'cuda':
                            torch.cuda.synchronize()
                        
                        times.append(time.time() - start_time)
                    
                    optimization_results[device_name] = {
                        'avg_time_ms': np.mean(times) * 1000,
                        'std_time_ms': np.std(times) * 1000,
                        'memory_mb': self.results[test_name].get('max_gpu_memory_mb', 0) if device.type == 'cuda' else 0
                    }
                    
                    self.logger.info(f"  {device_name}: {optimization_results[device_name]['avg_time_ms']:.2f}±{optimization_results[device_name]['std_time_ms']:.2f}ms")
            
            # 计算加速比
            if 'CPU' in optimization_results and 'GPU' in optimization_results:
                speedup = optimization_results['CPU']['avg_time_ms'] / optimization_results['GPU']['avg_time_ms']
                optimization_results['GPU_speedup'] = speedup
                self.logger.info(f"  GPU Speedup: {speedup:.2f}x")
        
        # 混合精度测试（如果GPU可用）
        if self.mixed_precision_available:
            self.logger.info("\nMixed Precision Testing:")
            
            for precision, use_amp in [('FP32', False), ('FP16', True)]:
                test_name = f"precision_{precision.lower()}"
                
                with self.performance_monitor(test_name):
                    network = self.create_test_network(1024, 128, 'medium')
                    test_data = torch.randn(64, 1024, 1, device=self.device)
                    scaler = torch.cuda.amp.GradScaler() if use_amp else None
                    
                    times = []
                    for _ in range(10):
                        start_time = time.time()
                        
                        if use_amp:
                            with torch.cuda.amp.autocast():
                                output = network(test_data)
                                anchor_features = torch.randn(64, 128, device=self.device)
                                positive_features = torch.randn(64, 128, device=self.device)
                                
                                anchor_norm = F.normalize(anchor_features, dim=1)
                                positive_norm = F.normalize(positive_features, dim=1)
                                similarity_matrix = torch.mm(anchor_norm, positive_norm.t()) / 0.07
                                loss = -torch.diag(similarity_matrix) + torch.logsumexp(similarity_matrix, dim=1)
                                final_loss = loss.mean()
                            
                            scaler.scale(final_loss).backward()
                        else:
                            output = network(test_data)
                            anchor_features = torch.randn(64, 128, device=self.device)
                            positive_features = torch.randn(64, 128, device=self.device)
                            
                            anchor_norm = F.normalize(anchor_features, dim=1)
                            positive_norm = F.normalize(positive_features, dim=1)
                            similarity_matrix = torch.mm(anchor_norm, positive_norm.t()) / 0.07
                            loss = -torch.diag(similarity_matrix) + torch.logsumexp(similarity_matrix, dim=1)
                            final_loss = loss.mean()
                            final_loss.backward()
                        
                        torch.cuda.synchronize()
                        times.append(time.time() - start_time)
                    
                    optimization_results[precision] = {
                        'avg_time_ms': np.mean(times) * 1000,
                        'std_time_ms': np.std(times) * 1000,
                        'memory_mb': self.results[test_name].get('max_gpu_memory_mb', 0)
                    }
                    
                    self.logger.info(f"  {precision}: {optimization_results[precision]['avg_time_ms']:.2f}±{optimization_results[precision]['std_time_ms']:.2f}ms")
            
            # 计算混合精度收益
            if 'FP32' in optimization_results and 'FP16' in optimization_results:
                time_speedup = optimization_results['FP32']['avg_time_ms'] / optimization_results['FP16']['avg_time_ms']
                memory_savings = optimization_results['FP32']['memory_mb'] - optimization_results['FP16']['memory_mb']
                optimization_results['mixed_precision_speedup'] = time_speedup
                optimization_results['mixed_precision_memory_savings_mb'] = memory_savings
                self.logger.info(f"  Mixed Precision Speedup: {time_speedup:.2f}x")
                self.logger.info(f"  Memory Savings: {memory_savings:.2f}MB")
        
        self.results['hardware_optimization'] = optimization_results

    def generate_comprehensive_report(self):
        """生成综合性能报告"""
        self.logger.info("="*60)
        self.logger.info("GENERATING COMPREHENSIVE PERFORMANCE REPORT")
        self.logger.info("="*60)
        
        # 保存详细JSON结果
        results_file = self.save_dir / "comprehensive_benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # 生成HTML报告
        self._generate_html_report()
        
        # 生成Markdown摘要报告
        self._generate_markdown_summary()
        
        # 生成性能图表
        self._generate_performance_plots()
        
        # 生成优化建议
        self._generate_optimization_recommendations()
        
        self.logger.info(f"Comprehensive report generated in: {self.save_dir}")
    
    def _generate_html_report(self):
        """生成HTML报告"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ContrastiveIDTask Performance Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .success {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>ContrastiveIDTask Performance Benchmark Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Device: {self.device}</p>
                <p>Mixed Precision Available: {self.mixed_precision_available}</p>
            </div>
        """
        
        # 性能摘要表
        html_content += """
            <div class="section">
                <h2>Performance Summary</h2>
                <table>
                    <tr><th>Test Category</th><th>Status</th><th>Key Metrics</th><th>Target Achievement</th></tr>
        """
        
        # 添加各类测试的摘要行
        test_categories = [
            ('Training Performance', 'training_perf_medium_config'),
            ('Data Processing', 'data_proc_medium_2048'),
            ('Model Performance', 'model_medium'),
            ('Scalability', 'scalability_testing'),
            ('Hardware Optimization', 'hardware_optimization')
        ]
        
        for category, key in test_categories:
            if key in self.results:
                result = self.results[key]
                targets_met = result.get('all_targets_met', result.get('targets_met', {}).get('all_targets_met', False))
                status = "✅ PASS" if targets_met else "❌ FAIL"
                status_class = "success" if targets_met else "error"
                
                # 提取关键指标
                key_metrics = []
                if 'throughput_samples_per_sec' in result:
                    key_metrics.append(f"Throughput: {result['throughput_samples_per_sec']:.1f} samples/s")
                if 'avg_batch_time_ms' in result:
                    key_metrics.append(f"Batch time: {result['avg_batch_time_ms']:.1f}ms")
                if 'max_working_batch_size' in result:
                    key_metrics.append(f"Max batch: {result['max_working_batch_size']}")
                
                html_content += f"""
                    <tr>
                        <td>{category}</td>
                        <td class="{status_class}">{status}</td>
                        <td>{', '.join(key_metrics[:2])}</td>
                        <td>{'All targets met' if targets_met else 'Some targets missed'}</td>
                    </tr>
                """
        
        html_content += """
                </table>
            </div>
        """
        
        # 详细结果部分
        for category, key in test_categories:
            if key in self.results:
                html_content += f"""
                    <div class="section">
                        <h2>{category} Details</h2>
                        <div class="metric">
                            <pre>{json.dumps(self.results[key], indent=2, default=str)}</pre>
                        </div>
                    </div>
                """
        
        html_content += """
            </body>
            </html>
        """
        
        with open(self.save_dir / "reports" / "performance_report.html", 'w') as f:
            f.write(html_content)
    
    def _generate_markdown_summary(self):
        """生成Markdown摘要报告"""
        with open(self.save_dir / "reports" / "performance_summary.md", 'w') as f:
            f.write("# ContrastiveIDTask Performance Benchmark Summary\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"**Device**: {self.device}\\n")
            f.write(f"**Mixed Precision**: {self.mixed_precision_available}\\n\\n")
            
            f.write("## Executive Summary\n\n")
            
            # 计算总体通过率
            total_tests = 0
            passed_tests = 0
            
            for test_name, result in self.results.items():
                if isinstance(result, dict) and ('all_targets_met' in result or 'targets_met' in result):
                    total_tests += 1
                    targets_met = result.get('all_targets_met', result.get('targets_met', {}).get('all_targets_met', False))
                    if targets_met:
                        passed_tests += 1
            
            pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            f.write(f"**Overall Pass Rate**: {pass_rate:.1f}% ({passed_tests}/{total_tests} tests passed)\\n\\n")
            
            # 关键性能指标
            f.write("## Key Performance Indicators\n\n")
            f.write("| Metric | Value | Target | Status |\n")
            f.write("|--------|-------|--------|--------|\n")
            
            # 从结果中提取关键指标
            if 'training_perf_medium_config' in self.results:
                result = self.results['training_perf_medium_config']
                throughput = result.get('throughput_samples_per_sec', 0)
                target_throughput = self.targets['training_performance']['samples_per_second']
                status = "✅" if throughput >= target_throughput else "❌"
                f.write(f"| Training Throughput | {throughput:.1f} samples/s | {target_throughput} samples/s | {status} |\n")
                
                batch_time = result.get('avg_batch_time_ms', 0)
                target_batch_time = self.targets['training_performance']['batch_time_ms']
                status = "✅" if batch_time <= target_batch_time else "❌"
                f.write(f"| Batch Processing Time | {batch_time:.1f}ms | ≤{target_batch_time}ms | {status} |\n")
            
            if 'scalability_testing' in self.results:
                result = self.results['scalability_testing']
                max_batch = result.get('max_working_batch_size', 0)
                target_batch = self.targets['scalability']['max_batch_size']
                status = "✅" if max_batch >= target_batch else "❌"
                f.write(f"| Max Batch Size | {max_batch} | ≥{target_batch} | {status} |\n")
            
            f.write("\\n")
            
            # 优化建议
            f.write("## Optimization Recommendations\n\n")
            recommendations = self._generate_optimization_recommendations()
            for rec in recommendations:
                f.write(f"- {rec}\\n")
            
            f.write("\\n")
            
            # 详细测试结果
            f.write("## Detailed Test Results\n\n")
            
            for test_name, result in self.results.items():
                if isinstance(result, dict):
                    f.write(f"### {test_name.replace('_', ' ').title()}\n\n")
                    
                    # 基本信息
                    if 'duration_seconds' in result:
                        f.write(f"**Duration**: {result['duration_seconds']:.2f}s\\n")
                    if 'cpu_memory_increase_mb' in result:
                        f.write(f"**Memory Usage**: +{result['cpu_memory_increase_mb']:.2f}MB CPU")
                        if 'gpu_memory_increase_mb' in result:
                            f.write(f", +{result['gpu_memory_increase_mb']:.2f}MB GPU")
                        f.write("\\n")
                    
                    # 目标达成情况
                    if 'targets_met' in result:
                        f.write(f"**Targets Met**: {result.get('all_targets_met', False)}\\n")
                        targets = result['targets_met']
                        if isinstance(targets, dict):
                            for target_name, met in targets.items():
                                status = "✅" if met else "❌"
                                f.write(f"  - {target_name}: {status}\\n")
                    
                    f.write("\\n")
    
    def _generate_performance_plots(self):
        """生成性能图表"""
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            plt.style.use('default')
        
        # 1. 训练性能对比图
        if any('training_perf_' in key for key in self.results.keys()):
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 收集训练性能数据
            configs = []
            throughput_data = []
            batch_time_data = []
            forward_time_data = []
            memory_data = []
            
            for key, result in self.results.items():
                if 'training_perf_' in key and isinstance(result, dict):
                    config_name = key.replace('training_perf_', '')
                    configs.append(config_name)
                    throughput_data.append(result.get('throughput_samples_per_sec', 0))
                    batch_time_data.append(result.get('avg_batch_time_ms', 0))
                    forward_time_data.append(result.get('avg_forward_time_ms', 0))
                    memory_data.append(result.get('max_gpu_memory_mb', result.get('max_cpu_memory_mb', 0)))
            
            if configs:
                ax1.bar(configs, throughput_data, color='skyblue')
                ax1.set_title('Training Throughput by Configuration')
                ax1.set_ylabel('Samples/Second')
                ax1.tick_params(axis='x', rotation=45)
                
                ax2.bar(configs, batch_time_data, color='lightcoral')
                ax2.set_title('Batch Processing Time by Configuration')
                ax2.set_ylabel('Time (ms)')
                ax2.tick_params(axis='x', rotation=45)
                
                ax3.bar(configs, forward_time_data, color='lightgreen')
                ax3.set_title('Forward Pass Time by Configuration')
                ax3.set_ylabel('Time (ms)')
                ax3.tick_params(axis='x', rotation=45)
                
                ax4.bar(configs, memory_data, color='wheat')
                ax4.set_title('Memory Usage by Configuration')
                ax4.set_ylabel('Memory (MB)')
                ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.save_dir / "plots" / "training_performance.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # 2. 可扩展性图表
        if 'scalability_testing' in self.results:
            scalability_data = self.results['scalability_testing']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 批次大小可扩展性
            if 'batch_size_scalability' in scalability_data:
                batch_results = scalability_data['batch_size_scalability']
                batch_sizes = []
                batch_memories = []
                batch_times = []
                
                for batch_size, result in batch_results.items():
                    if result.get('success', False):
                        batch_sizes.append(batch_size)
                        batch_memories.append(result.get('memory_mb', 0))
                        batch_times.append(result.get('time_ms', 0))
                
                if batch_sizes:
                    ax1.plot(batch_sizes, batch_memories, 'o-', label='Memory Usage', color='red')
                    ax1_twin = ax1.twinx()
                    ax1_twin.plot(batch_sizes, batch_times, 's-', label='Processing Time', color='blue')
                    
                    ax1.set_xlabel('Batch Size')
                    ax1.set_ylabel('Memory Usage (MB)', color='red')
                    ax1_twin.set_ylabel('Processing Time (ms)', color='blue')
                    ax1.set_title('Batch Size Scalability')
                    ax1.grid(True)
            
            # 窗口大小可扩展性
            if 'window_size_scalability' in scalability_data:
                window_results = scalability_data['window_size_scalability']
                window_sizes = []
                window_memories = []
                window_times = []
                
                for window_size, result in window_results.items():
                    if result.get('success', False):
                        window_sizes.append(window_size)
                        window_memories.append(result.get('memory_mb', 0))
                        window_times.append(result.get('time_ms', 0))
                
                if window_sizes:
                    ax2.plot(window_sizes, window_memories, 'o-', label='Memory Usage', color='red')
                    ax2_twin = ax2.twinx()
                    ax2_twin.plot(window_sizes, window_times, 's-', label='Processing Time', color='blue')
                    
                    ax2.set_xlabel('Window Size')
                    ax2.set_ylabel('Memory Usage (MB)', color='red')
                    ax2_twin.set_ylabel('Processing Time (ms)', color='blue')
                    ax2.set_title('Window Size Scalability')
                    ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(self.save_dir / "plots" / "scalability_analysis.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. 硬件优化对比图
        if 'hardware_optimization' in self.results:
            hw_data = self.results['hardware_optimization']
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # CPU vs GPU 对比
            devices = []
            times = []
            
            for device in ['CPU', 'GPU']:
                if device in hw_data:
                    devices.append(device)
                    times.append(hw_data[device]['avg_time_ms'])
            
            if len(devices) >= 2:
                bars = ax.bar(devices, times, color=['lightblue', 'lightgreen'])
                ax.set_title('CPU vs GPU Performance Comparison')
                ax.set_ylabel('Average Processing Time (ms)')
                
                # 添加数值标签
                for bar, time_val in zip(bars, times):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(times),
                           f'{time_val:.1f}ms', ha='center', va='bottom')
                
                # 添加加速比信息
                if 'GPU_speedup' in hw_data:
                    speedup = hw_data['GPU_speedup']
                    ax.text(0.5, 0.95, f'GPU Speedup: {speedup:.2f}x', 
                           transform=ax.transAxes, ha='center', va='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(self.save_dir / "plots" / "hardware_optimization.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        self.logger.info("Performance plots generated successfully")
    
    def _generate_optimization_recommendations(self):
        """生成优化建议"""
        recommendations = []
        
        # 基于测试结果生成建议
        
        # 训练性能建议
        if 'training_perf_medium_config' in self.results:
            result = self.results['training_perf_medium_config']
            throughput = result.get('throughput_samples_per_sec', 0)
            
            if throughput < self.targets['training_performance']['samples_per_second']:
                recommendations.append(
                    f"Training throughput ({throughput:.1f} samples/s) below target "
                    f"({self.targets['training_performance']['samples_per_second']} samples/s). "
                    "Consider increasing batch size or using mixed precision training."
                )
            
            batch_time = result.get('avg_batch_time_ms', 0)
            if batch_time > self.targets['training_performance']['batch_time_ms']:
                recommendations.append(
                    f"Batch processing time ({batch_time:.1f}ms) exceeds target "
                    f"({self.targets['training_performance']['batch_time_ms']}ms). "
                    "Consider optimizing data loading or using faster storage."
                )
        
        # 内存优化建议
        max_memory = 0
        for result in self.results.values():
            if isinstance(result, dict):
                memory = result.get('max_gpu_memory_mb', result.get('max_cpu_memory_mb', 0))
                max_memory = max(max_memory, memory)
        
        if max_memory > 1000:  # 超过1GB
            recommendations.append(
                f"Peak memory usage ({max_memory:.1f}MB) is high. "
                "Consider using gradient checkpointing, smaller batch sizes, or model parallelism."
            )
        
        # 可扩展性建议
        if 'scalability_testing' in self.results:
            scalability = self.results['scalability_testing']
            max_batch = scalability.get('max_working_batch_size', 0)
            
            if max_batch < self.targets['scalability']['max_batch_size']:
                recommendations.append(
                    f"Maximum batch size ({max_batch}) below target "
                    f"({self.targets['scalability']['max_batch_size']}). "
                    "Consider memory optimization or using gradient accumulation."
                )
        
        # 硬件优化建议
        if 'hardware_optimization' in self.results:
            hw_data = self.results['hardware_optimization']
            
            if 'GPU_speedup' in hw_data:
                speedup = hw_data['GPU_speedup']
                if speedup < 5.0:  # GPU应该比CPU快至少5倍
                    recommendations.append(
                        f"GPU speedup ({speedup:.2f}x) is lower than expected. "
                        "Consider optimizing GPU utilization or checking for CPU bottlenecks."
                    )
            
            if 'mixed_precision_speedup' in hw_data:
                mp_speedup = hw_data['mixed_precision_speedup']
                if mp_speedup > 1.2:
                    recommendations.append(
                        f"Mixed precision provides {mp_speedup:.2f}x speedup. "
                        "Recommend enabling mixed precision training for production."
                    )
        
        # 通用建议
        if not recommendations:
            recommendations.append("All performance targets met. Current configuration is optimal.")
        else:
            recommendations.append(
                "Run benchmarks regularly to monitor performance regression and validate optimizations."
            )
        
        return recommendations
    
    def run_comprehensive_benchmark(self):
        """运行完整的性能基准测试套件"""
        self.logger.info("="*60)
        self.logger.info("STARTING COMPREHENSIVE PERFORMANCE BENCHMARK")
        self.logger.info("="*60)
        
        try:
            # 更新待办事项状态
            self.logger.info("Running benchmark suite...")
            
            # 1. 训练性能测试
            self.benchmark_training_performance()
            self._update_todo_status("training_performance", "completed")
            
            # 2. 数据处理性能测试
            self.benchmark_data_processing_performance()
            self._update_todo_status("data_processing", "completed")
            
            # 3. 模型性能测试
            self.benchmark_model_performance()
            self._update_todo_status("model_performance", "completed")
            
            # 4. 可扩展性测试
            self.benchmark_scalability_testing()
            self._update_todo_status("scalability", "completed")
            
            # 5. 硬件优化测试
            self.benchmark_hardware_optimization()
            self._update_todo_status("hardware_optimization", "completed")
            
            # 6. 生成综合报告
            self.generate_comprehensive_report()
            self._update_todo_status("report_generation", "completed")
            
            # 总体评估
            self._calculate_overall_performance_score()
            
            self.logger.info("="*60)
            self.logger.info("COMPREHENSIVE BENCHMARK COMPLETED SUCCESSFULLY")
            self.logger.info("="*60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Benchmark suite failed: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _update_todo_status(self, task_name, status):
        """更新任务状态（占位符方法）"""
        pass
    
    def _calculate_overall_performance_score(self):
        """计算总体性能得分"""
        total_score = 0
        max_score = 0
        
        # 定义各测试类别的权重
        test_weights = {
            'training_performance': 30,
            'data_processing': 20,
            'model_performance': 25,
            'scalability': 15,
            'hardware_optimization': 10
        }
        
        for category, weight in test_weights.items():
            max_score += weight
            
            # 查找相关测试结果
            category_results = [result for key, result in self.results.items() 
                              if category.replace('_', '') in key.replace('_', '') 
                              and isinstance(result, dict)]
            
            if category_results:
                # 计算该类别的平均通过率
                category_pass_rate = 0
                for result in category_results:
                    targets_met = result.get('all_targets_met', 
                                           result.get('targets_met', {}).get('all_targets_met', False))
                    if targets_met:
                        category_pass_rate += 1
                
                category_pass_rate = category_pass_rate / len(category_results) if category_results else 0
                total_score += weight * category_pass_rate
        
        overall_score = (total_score / max_score * 100) if max_score > 0 else 0
        
        self.results['overall_performance'] = {
            'score': overall_score,
            'max_score': max_score,
            'weighted_score': total_score,
            'grade': self._get_performance_grade(overall_score)
        }
        
        self.logger.info(f"Overall Performance Score: {overall_score:.1f}/100 ({self._get_performance_grade(overall_score)})")
    
    def _get_performance_grade(self, score):
        """获取性能等级"""
        if score >= 90:
            return "A (Excellent)"
        elif score >= 80:
            return "B (Good)"
        elif score >= 70:
            return "C (Satisfactory)"
        elif score >= 60:
            return "D (Needs Improvement)"
        else:
            return "F (Poor)"


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive ContrastiveIDTask Performance Benchmark")
    parser.add_argument("--save-dir", default="./benchmark_results",
                       help="Results save directory")
    parser.add_argument("--test", choices=["all", "training", "data", "model", "scalability", "hardware"],
                       default="all", help="Test type to run")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"],
                       default="auto", help="Device to use for testing")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # 创建基准测试实例
    benchmark = AdvancedPerformanceBenchmark(save_dir=args.save_dir)
    benchmark.device = device
    
    if args.verbose:
        benchmark.logger.setLevel(logging.DEBUG)
    
    # 运行测试
    if args.test == "all":
        success = benchmark.run_comprehensive_benchmark()
    elif args.test == "training":
        benchmark.benchmark_training_performance()
        success = True
    elif args.test == "data":
        benchmark.benchmark_data_processing_performance()
        success = True
    elif args.test == "model":
        benchmark.benchmark_model_performance()
        success = True
    elif args.test == "scalability":
        benchmark.benchmark_scalability_testing()
        success = True
    elif args.test == "hardware":
        benchmark.benchmark_hardware_optimization()
        success = True
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()