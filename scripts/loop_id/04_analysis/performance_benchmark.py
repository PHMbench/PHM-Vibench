#!/usr/bin/env python3
"""
ContrastiveIDTask性能基准测试

专注于实验过程中的核心性能指标监控和分析，为科研人员提供：
- 训练性能分析（速度、内存、收敛）
- 模型性能评估（前向/后向传播、InfoNCE计算）
- 可扩展性测试（批量大小、窗口大小影响）
- 硬件效率分析（CPU vs GPU、内存使用）

Usage:
    # 快速基准测试
    python performance_benchmark.py --quick --config debug

    # 完整性能分析
    python performance_benchmark.py --full --config production

    # 特定组件测试
    python performance_benchmark.py --test memory training --output_dir bench_results

Author: PHM-Vibench Team
Version: 2.0 (Research-focused)
"""

import os
import sys
import time
import json
import psutil
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import gc
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
import argparse
from contextlib import contextmanager

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.configs import load_config

class PerformanceBenchmark:
    """ContrastiveIDTask性能基准测试器

    核心测试维度：
    1. 内存效率 - GPU/CPU内存使用、内存泄露检测
    2. 训练性能 - 批处理速度、吞吐量、收敛速度
    3. 模型性能 - 前向/后向传播耗时、InfoNCE计算
    4. 可扩展性 - 不同配置下的性能表现
    """

    def __init__(self,
                 config_path: str = "configs/id_contrastive/debug.yaml",
                 output_dir: str = "save/performance_benchmark",
                 quick_mode: bool = False):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.quick_mode = quick_mode

        # 创建目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)

        # 设置日志
        self._setup_logging()

        # 性能目标 - 基于实际硬件合理设置
        self.performance_targets = {
            'memory': {
                'gpu_memory_per_batch_mb': 1000,   # 单批次GPU内存使用上限
                'cpu_memory_per_batch_mb': 500,    # 单批次CPU内存使用上限
                'memory_growth_threshold': 0.1,   # 内存增长率阈值
            },
            'training': {
                'min_samples_per_second': 20,      # 最小训练吞吐量
                'max_batch_time_ms': 500,         # 最大批处理时间
                'target_gpu_utilization': 0.7,    # 目标GPU利用率
            },
            'model': {
                'max_forward_time_ms': 100,        # 最大前向传播时间
                'max_infonce_time_ms': 20,        # 最大InfoNCE计算时间
            }
        }

        # 测试配置
        if quick_mode:
            self.test_configs = {
                'batch_sizes': [8, 16],
                'window_sizes': [512, 1024],
                'num_batches': 5,
                'warmup_batches': 2
            }
        else:
            self.test_configs = {
                'batch_sizes': [8, 16, 32, 64],
                'window_sizes': [256, 512, 1024, 2048],
                'num_batches': 20,
                'warmup_batches': 5
            }

        # 结果存储
        self.results = defaultdict(list)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info(f"性能基准测试初始化完成")
        self.logger.info(f"配置文件: {config_path}")
        self.logger.info(f"输出目录: {output_dir}")
        self.logger.info(f"测试设备: {self.device}")
        self.logger.info(f"快速模式: {quick_mode}")

    def _setup_logging(self):
        """设置日志系统"""
        log_file = self.output_dir / "benchmark.log"

        self.logger = logging.getLogger('PerformanceBenchmark')
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            # 文件处理器
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)

            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    @contextmanager
    def memory_monitor(self, test_name: str):
        """内存监控上下文管理器"""
        # 清理内存
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # 记录初始状态
        process = psutil.Process()
        initial_cpu_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 if self.device.type == "cuda" else 0

        start_time = time.time()

        try:
            yield
        finally:
            end_time = time.time()

            # 记录最终状态
            final_cpu_memory = process.memory_info().rss / 1024 / 1024
            final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 if self.device.type == "cuda" else 0
            peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 if self.device.type == "cuda" else 0

            memory_result = {
                'test_name': test_name,
                'duration_sec': end_time - start_time,
                'cpu_memory_initial_mb': initial_cpu_memory,
                'cpu_memory_final_mb': final_cpu_memory,
                'cpu_memory_delta_mb': final_cpu_memory - initial_cpu_memory,
                'gpu_memory_initial_mb': initial_gpu_memory,
                'gpu_memory_final_mb': final_gpu_memory,
                'gpu_memory_peak_mb': peak_gpu_memory,
                'gpu_memory_delta_mb': final_gpu_memory - initial_gpu_memory
            }

            self.results['memory'].append(memory_result)

    def create_mock_task(self, batch_size: int = 32, window_size: int = 1024) -> torch.nn.Module:
        """创建模拟的ContrastiveIDTask用于测试"""
        class MockContrastiveTask(torch.nn.Module):
            def __init__(self, d_model=128, temperature=0.07):
                super().__init__()
                self.temperature = temperature
                self.encoder = torch.nn.Sequential(
                    torch.nn.Linear(window_size, d_model * 2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(d_model * 2, d_model),
                    torch.nn.ReLU(),
                    torch.nn.Linear(d_model, d_model)
                )

            def forward(self, anchor, positive):
                z_anchor = self.encoder(anchor)
                z_positive = self.encoder(positive)
                return self.infonce_loss(z_anchor, z_positive)

            def infonce_loss(self, z_anchor, z_positive):
                # L2归一化
                z_anchor = F.normalize(z_anchor, dim=1)
                z_positive = F.normalize(z_positive, dim=1)

                # 相似度矩阵
                sim_matrix = torch.mm(z_anchor, z_positive.t()) / self.temperature

                # InfoNCE损失
                labels = torch.arange(z_anchor.size(0), device=z_anchor.device)
                loss = F.cross_entropy(sim_matrix, labels)

                return loss

        return MockContrastiveTask().to(self.device)

    def test_memory_usage(self) -> Dict[str, Any]:
        """测试内存使用情况"""
        self.logger.info("开始内存使用测试...")

        memory_results = []

        for batch_size in self.test_configs['batch_sizes']:
            for window_size in self.test_configs['window_sizes']:
                test_name = f"memory_b{batch_size}_w{window_size}"

                with self.memory_monitor(test_name):
                    # 创建模型和数据
                    model = self.create_mock_task(batch_size, window_size)

                    # 生成测试数据
                    anchor = torch.randn(batch_size, window_size, device=self.device)
                    positive = torch.randn(batch_size, window_size, device=self.device)

                    # 多次前向传播测试内存稳定性
                    for i in range(self.test_configs['num_batches']):
                        loss = model(anchor, positive)
                        if i > self.test_configs['warmup_batches']:
                            loss.backward()
                            model.zero_grad()

        # 分析内存使用结果
        memory_summary = self._analyze_memory_results()
        return memory_summary

    def test_training_performance(self) -> Dict[str, Any]:
        """测试训练性能"""
        self.logger.info("开始训练性能测试...")

        training_results = []

        for batch_size in self.test_configs['batch_sizes']:
            for window_size in self.test_configs['window_sizes']:
                # 创建模型
                model = self.create_mock_task(batch_size, window_size)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                # 生成测试数据
                anchor = torch.randn(batch_size, window_size, device=self.device)
                positive = torch.randn(batch_size, window_size, device=self.device)

                # 预热
                for _ in range(self.test_configs['warmup_batches']):
                    loss = model(anchor, positive)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # 性能测试
                batch_times = []
                throughputs = []

                for i in range(self.test_configs['num_batches']):
                    start_time = time.time()

                    loss = model(anchor, positive)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

                    end_time = time.time()
                    batch_time = (end_time - start_time) * 1000  # 转换为毫秒
                    throughput = batch_size / (end_time - start_time)  # 样本/秒

                    batch_times.append(batch_time)
                    throughputs.append(throughput)

                # 记录结果
                result = {
                    'batch_size': batch_size,
                    'window_size': window_size,
                    'avg_batch_time_ms': np.mean(batch_times),
                    'std_batch_time_ms': np.std(batch_times),
                    'min_batch_time_ms': np.min(batch_times),
                    'max_batch_time_ms': np.max(batch_times),
                    'avg_throughput_samples_per_sec': np.mean(throughputs),
                    'std_throughput_samples_per_sec': np.std(throughputs)
                }

                training_results.append(result)
                self.logger.info(f"批量{batch_size}×窗口{window_size}: {result['avg_batch_time_ms']:.1f}ms, {result['avg_throughput_samples_per_sec']:.1f}样本/秒")

        self.results['training'] = training_results
        return self._analyze_training_results()

    def test_model_components(self) -> Dict[str, Any]:
        """测试模型组件性能"""
        self.logger.info("开始模型组件性能测试...")

        component_results = []

        # 使用中等配置进行测试
        batch_size = 32
        window_size = 1024
        d_model = 128

        model = self.create_mock_task(batch_size, window_size)

        # 生成测试数据
        anchor = torch.randn(batch_size, window_size, device=self.device)
        positive = torch.randn(batch_size, window_size, device=self.device)

        # 测试前向传播时间
        forward_times = []
        for _ in range(20):
            start_time = time.time()

            with torch.no_grad():
                z_anchor = model.encoder(anchor)
                z_positive = model.encoder(positive)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            forward_time = (time.time() - start_time) * 1000
            forward_times.append(forward_time)

        # 测试InfoNCE计算时间
        infonce_times = []
        for _ in range(20):
            with torch.no_grad():
                z_anchor = model.encoder(anchor)
                z_positive = model.encoder(positive)

            start_time = time.time()

            loss = model.infonce_loss(z_anchor, z_positive)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            infonce_time = (time.time() - start_time) * 1000
            infonce_times.append(infonce_time)

        # 测试反向传播时间
        backward_times = []
        for _ in range(20):
            loss = model(anchor, positive)

            start_time = time.time()
            loss.backward()
            if self.device.type == "cuda":
                torch.cuda.synchronize()

            backward_time = (time.time() - start_time) * 1000
            backward_times.append(backward_time)

            model.zero_grad()

        component_result = {
            'batch_size': batch_size,
            'window_size': window_size,
            'd_model': d_model,
            'avg_forward_time_ms': np.mean(forward_times),
            'std_forward_time_ms': np.std(forward_times),
            'avg_infonce_time_ms': np.mean(infonce_times),
            'std_infonce_time_ms': np.std(infonce_times),
            'avg_backward_time_ms': np.mean(backward_times),
            'std_backward_time_ms': np.std(backward_times)
        }

        component_results.append(component_result)
        self.results['components'] = component_results

        return component_result

    def test_scalability(self) -> Dict[str, Any]:
        """测试可扩展性"""
        self.logger.info("开始可扩展性测试...")

        scalability_results = []

        # 测试批量大小扩展性
        window_size = 1024
        for batch_size in [8, 16, 32, 64, 128]:
            try:
                with self.memory_monitor(f"scalability_batch_{batch_size}"):
                    model = self.create_mock_task(batch_size, window_size)
                    anchor = torch.randn(batch_size, window_size, device=self.device)
                    positive = torch.randn(batch_size, window_size, device=self.device)

                    # 性能测试
                    times = []
                    for _ in range(5):
                        start_time = time.time()
                        loss = model(anchor, positive)
                        if self.device.type == "cuda":
                            torch.cuda.synchronize()
                        times.append((time.time() - start_time) * 1000)

                    scalability_results.append({
                        'test_type': 'batch_size',
                        'value': batch_size,
                        'avg_time_ms': np.mean(times),
                        'throughput_samples_per_sec': batch_size / (np.mean(times) / 1000),
                        'success': True
                    })

                    self.logger.info(f"批量大小 {batch_size}: {np.mean(times):.1f}ms")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    scalability_results.append({
                        'test_type': 'batch_size',
                        'value': batch_size,
                        'success': False,
                        'error': 'OOM'
                    })
                    self.logger.warning(f"批量大小 {batch_size}: 内存不足")
                    break
                else:
                    raise

        # 测试窗口大小扩展性
        batch_size = 32
        for window_size in [256, 512, 1024, 2048, 4096]:
            try:
                with self.memory_monitor(f"scalability_window_{window_size}"):
                    model = self.create_mock_task(batch_size, window_size)
                    anchor = torch.randn(batch_size, window_size, device=self.device)
                    positive = torch.randn(batch_size, window_size, device=self.device)

                    times = []
                    for _ in range(5):
                        start_time = time.time()
                        loss = model(anchor, positive)
                        if self.device.type == "cuda":
                            torch.cuda.synchronize()
                        times.append((time.time() - start_time) * 1000)

                    scalability_results.append({
                        'test_type': 'window_size',
                        'value': window_size,
                        'avg_time_ms': np.mean(times),
                        'throughput_samples_per_sec': batch_size / (np.mean(times) / 1000),
                        'success': True
                    })

                    self.logger.info(f"窗口大小 {window_size}: {np.mean(times):.1f}ms")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    scalability_results.append({
                        'test_type': 'window_size',
                        'value': window_size,
                        'success': False,
                        'error': 'OOM'
                    })
                    self.logger.warning(f"窗口大小 {window_size}: 内存不足")
                    break
                else:
                    raise

        self.results['scalability'] = scalability_results
        return scalability_results

    def _analyze_memory_results(self) -> Dict[str, Any]:
        """分析内存使用结果"""
        memory_data = self.results['memory']

        if not memory_data:
            return {'error': 'No memory data available'}

        df = pd.DataFrame(memory_data)

        analysis = {
            'avg_cpu_memory_usage_mb': df['cpu_memory_delta_mb'].mean(),
            'max_cpu_memory_usage_mb': df['cpu_memory_delta_mb'].max(),
            'avg_gpu_memory_usage_mb': df['gpu_memory_delta_mb'].mean(),
            'max_gpu_memory_usage_mb': df['gpu_memory_delta_mb'].max(),
            'avg_peak_gpu_memory_mb': df['gpu_memory_peak_mb'].mean(),
            'memory_efficient': df['gpu_memory_peak_mb'].max() < self.performance_targets['memory']['gpu_memory_per_batch_mb']
        }

        return analysis

    def _analyze_training_results(self) -> Dict[str, Any]:
        """分析训练性能结果"""
        training_data = self.results['training']

        if not training_data:
            return {'error': 'No training data available'}

        df = pd.DataFrame(training_data)

        analysis = {
            'avg_batch_time_ms': df['avg_batch_time_ms'].mean(),
            'min_batch_time_ms': df['avg_batch_time_ms'].min(),
            'max_batch_time_ms': df['avg_batch_time_ms'].max(),
            'avg_throughput_samples_per_sec': df['avg_throughput_samples_per_sec'].mean(),
            'max_throughput_samples_per_sec': df['avg_throughput_samples_per_sec'].max(),
            'performance_target_met': df['avg_throughput_samples_per_sec'].max() >= self.performance_targets['training']['min_samples_per_second']
        }

        return analysis

    def generate_visualizations(self):
        """生成性能可视化图表"""
        self.logger.info("生成可视化图表...")

        # 设置绘图风格
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)

        # 1. 训练性能图表
        if 'training' in self.results and self.results['training']:
            self._plot_training_performance()

        # 2. 内存使用图表
        if 'memory' in self.results and self.results['memory']:
            self._plot_memory_usage()

        # 3. 可扩展性图表
        if 'scalability' in self.results and self.results['scalability']:
            self._plot_scalability()

        # 4. 组件性能图表
        if 'components' in self.results and self.results['components']:
            self._plot_component_performance()

        self.logger.info("可视化图表生成完成")

    def _plot_training_performance(self):
        """绘制训练性能图表"""
        df = pd.DataFrame(self.results['training'])

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Performance Analysis', fontsize=16)

        # 批处理时间热力图
        pivot_time = df.pivot_table(values='avg_batch_time_ms', index='window_size', columns='batch_size')
        im1 = axes[0, 0].imshow(pivot_time.values, cmap='YlOrRd', aspect='auto')
        axes[0, 0].set_title('Batch Processing Time (ms)')
        axes[0, 0].set_xlabel('Batch Size')
        axes[0, 0].set_ylabel('Window Size')
        axes[0, 0].set_xticks(range(len(pivot_time.columns)))
        axes[0, 0].set_xticklabels(pivot_time.columns)
        axes[0, 0].set_yticks(range(len(pivot_time.index)))
        axes[0, 0].set_yticklabels(pivot_time.index)
        plt.colorbar(im1, ax=axes[0, 0])

        # 吞吐量热力图
        pivot_throughput = df.pivot_table(values='avg_throughput_samples_per_sec', index='window_size', columns='batch_size')
        im2 = axes[0, 1].imshow(pivot_throughput.values, cmap='YlGnBu', aspect='auto')
        axes[0, 1].set_title('Throughput (samples/sec)')
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Window Size')
        axes[0, 1].set_xticks(range(len(pivot_throughput.columns)))
        axes[0, 1].set_xticklabels(pivot_throughput.columns)
        axes[0, 1].set_yticks(range(len(pivot_throughput.index)))
        axes[0, 1].set_yticklabels(pivot_throughput.index)
        plt.colorbar(im2, ax=axes[0, 1])

        # 批量大小vs性能
        batch_perf = df.groupby('batch_size').agg({
            'avg_batch_time_ms': 'mean',
            'avg_throughput_samples_per_sec': 'mean'
        }).reset_index()

        ax1 = axes[1, 0]
        ax2 = ax1.twinx()
        line1 = ax1.plot(batch_perf['batch_size'], batch_perf['avg_batch_time_ms'], 'b-o', label='Batch Time (ms)')
        line2 = ax2.plot(batch_perf['batch_size'], batch_perf['avg_throughput_samples_per_sec'], 'r-s', label='Throughput (samples/sec)')
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Batch Time (ms)', color='b')
        ax2.set_ylabel('Throughput (samples/sec)', color='r')
        ax1.set_title('Performance vs Batch Size')

        # 窗口大小vs性能
        window_perf = df.groupby('window_size').agg({
            'avg_batch_time_ms': 'mean',
            'avg_throughput_samples_per_sec': 'mean'
        }).reset_index()

        ax3 = axes[1, 1]
        ax4 = ax3.twinx()
        line3 = ax3.plot(window_perf['window_size'], window_perf['avg_batch_time_ms'], 'b-o', label='Batch Time (ms)')
        line4 = ax4.plot(window_perf['window_size'], window_perf['avg_throughput_samples_per_sec'], 'r-s', label='Throughput (samples/sec)')
        ax3.set_xlabel('Window Size')
        ax3.set_ylabel('Batch Time (ms)', color='b')
        ax4.set_ylabel('Throughput (samples/sec)', color='r')
        ax3.set_title('Performance vs Window Size')

        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "training_performance.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_memory_usage(self):
        """绘制内存使用图表"""
        df = pd.DataFrame(self.results['memory'])

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Memory Usage Analysis', fontsize=16)

        # CPU内存使用
        axes[0, 0].bar(range(len(df)), df['cpu_memory_delta_mb'])
        axes[0, 0].set_title('CPU Memory Usage (MB)')
        axes[0, 0].set_xlabel('Test Index')
        axes[0, 0].set_ylabel('Memory Delta (MB)')

        # GPU内存使用
        if df['gpu_memory_delta_mb'].sum() > 0:
            axes[0, 1].bar(range(len(df)), df['gpu_memory_peak_mb'])
            axes[0, 1].set_title('Peak GPU Memory Usage (MB)')
            axes[0, 1].set_xlabel('Test Index')
            axes[0, 1].set_ylabel('Peak Memory (MB)')

        # 内存使用随时间变化
        axes[1, 0].plot(df['duration_sec'], df['cpu_memory_delta_mb'], 'o-', label='CPU Memory')
        if df['gpu_memory_delta_mb'].sum() > 0:
            axes[1, 0].plot(df['duration_sec'], df['gpu_memory_delta_mb'], 's-', label='GPU Memory')
        axes[1, 0].set_xlabel('Test Duration (sec)')
        axes[1, 0].set_ylabel('Memory Delta (MB)')
        axes[1, 0].set_title('Memory Usage vs Test Duration')
        axes[1, 0].legend()

        # 内存效率分析
        if df['gpu_memory_peak_mb'].sum() > 0:
            axes[1, 1].hist(df['gpu_memory_peak_mb'], bins=20, alpha=0.7, label='Peak GPU Memory')
            axes[1, 1].axvline(self.performance_targets['memory']['gpu_memory_per_batch_mb'],
                              color='r', linestyle='--', label='Target Threshold')
            axes[1, 1].set_xlabel('Peak GPU Memory (MB)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Memory Usage Distribution')
            axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "memory_usage.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_scalability(self):
        """绘制可扩展性图表"""
        scalability_data = self.results['scalability']
        successful_data = [item for item in scalability_data if item['success']]

        if not successful_data:
            return

        df = pd.DataFrame(successful_data)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Scalability Analysis', fontsize=16)

        # 批量大小扩展性
        batch_data = df[df['test_type'] == 'batch_size']
        if not batch_data.empty:
            axes[0].plot(batch_data['value'], batch_data['avg_time_ms'], 'o-', label='Processing Time')
            axes[0].set_xlabel('Batch Size')
            axes[0].set_ylabel('Processing Time (ms)')
            axes[0].set_title('Batch Size Scalability')
            axes[0].grid(True)

            # 添加吞吐量的第二个y轴
            ax2 = axes[0].twinx()
            ax2.plot(batch_data['value'], batch_data['throughput_samples_per_sec'], 's-', color='red', label='Throughput')
            ax2.set_ylabel('Throughput (samples/sec)', color='red')

        # 窗口大小扩展性
        window_data = df[df['test_type'] == 'window_size']
        if not window_data.empty:
            axes[1].plot(window_data['value'], window_data['avg_time_ms'], 'o-', label='Processing Time')
            axes[1].set_xlabel('Window Size')
            axes[1].set_ylabel('Processing Time (ms)')
            axes[1].set_title('Window Size Scalability')
            axes[1].grid(True)

            # 添加吞吐量的第二个y轴
            ax3 = axes[1].twinx()
            ax3.plot(window_data['value'], window_data['throughput_samples_per_sec'], 's-', color='red', label='Throughput')
            ax3.set_ylabel('Throughput (samples/sec)', color='red')

        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "scalability.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_component_performance(self):
        """绘制组件性能图表"""
        component_data = self.results['components'][0]  # 假设只有一个配置

        components = ['Forward Pass', 'InfoNCE Loss', 'Backward Pass']
        times = [
            component_data['avg_forward_time_ms'],
            component_data['avg_infonce_time_ms'],
            component_data['avg_backward_time_ms']
        ]
        errors = [
            component_data['std_forward_time_ms'],
            component_data['std_infonce_time_ms'],
            component_data['std_backward_time_ms']
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(components, times, yerr=errors, capsize=5, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax.set_ylabel('Time (ms)')
        ax.set_title('Model Component Performance')

        # 添加数值标签
        for bar, time_val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{time_val:.1f}ms', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "component_performance.png", dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self) -> str:
        """生成性能基准测试报告"""
        self.logger.info("生成基准测试报告...")

        # 分析结果
        memory_analysis = self._analyze_memory_results()
        training_analysis = self._analyze_training_results()

        report_file = self.output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        with open(report_file, 'w') as f:
            f.write("# ContrastiveIDTask 性能基准测试报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**测试设备**: {self.device}\n")
            f.write(f"**配置文件**: {self.config_path}\n\n")

            # 执行摘要
            f.write("## 执行摘要\n\n")

            # 内存使用分析
            if memory_analysis and 'error' not in memory_analysis:
                f.write("### 内存使用\n")
                f.write(f"- **平均GPU内存使用**: {memory_analysis['avg_gpu_memory_usage_mb']:.1f} MB\n")
                f.write(f"- **峰值GPU内存使用**: {memory_analysis['max_gpu_memory_usage_mb']:.1f} MB\n")
                f.write(f"- **内存效率**: {'✅ 达标' if memory_analysis['memory_efficient'] else '❌ 需优化'}\n\n")

            # 训练性能分析
            if training_analysis and 'error' not in training_analysis:
                f.write("### 训练性能\n")
                f.write(f"- **平均批处理时间**: {training_analysis['avg_batch_time_ms']:.1f} ms\n")
                f.write(f"- **最大吞吐量**: {training_analysis['max_throughput_samples_per_sec']:.1f} 样本/秒\n")
                f.write(f"- **性能目标**: {'✅ 达标' if training_analysis['performance_target_met'] else '❌ 需优化'}\n\n")

            # 组件性能
            if 'components' in self.results and self.results['components']:
                comp = self.results['components'][0]
                f.write("### 模型组件性能\n")
                f.write(f"- **前向传播**: {comp['avg_forward_time_ms']:.1f} ± {comp['std_forward_time_ms']:.1f} ms\n")
                f.write(f"- **InfoNCE损失**: {comp['avg_infonce_time_ms']:.1f} ± {comp['std_infonce_time_ms']:.1f} ms\n")
                f.write(f"- **反向传播**: {comp['avg_backward_time_ms']:.1f} ± {comp['std_backward_time_ms']:.1f} ms\n\n")

            # 可扩展性分析
            if 'scalability' in self.results:
                f.write("### 可扩展性\n")
                successful_tests = [t for t in self.results['scalability'] if t['success']]
                failed_tests = [t for t in self.results['scalability'] if not t['success']]

                if successful_tests:
                    batch_sizes = [t['value'] for t in successful_tests if t['test_type'] == 'batch_size']
                    window_sizes = [t['value'] for t in successful_tests if t['test_type'] == 'window_size']

                    if batch_sizes:
                        f.write(f"- **最大支持批量大小**: {max(batch_sizes)}\n")
                    if window_sizes:
                        f.write(f"- **最大支持窗口大小**: {max(window_sizes)}\n")

                if failed_tests:
                    f.write("- **内存限制**:\n")
                    for test in failed_tests:
                        f.write(f"  - {test['test_type']} {test['value']}: {test.get('error', 'Failed')}\n")
                f.write("\n")

            # 性能建议
            f.write("## 性能优化建议\n\n")

            if training_analysis and 'error' not in training_analysis:
                if training_analysis['avg_batch_time_ms'] > self.performance_targets['training']['max_batch_time_ms']:
                    f.write("1. **批处理时间优化**:\n")
                    f.write("   - 考虑减少模型复杂度\n")
                    f.write("   - 优化数据加载流程\n")
                    f.write("   - 启用混合精度训练\n\n")

                if training_analysis['max_throughput_samples_per_sec'] < self.performance_targets['training']['min_samples_per_second']:
                    f.write("2. **吞吐量优化**:\n")
                    f.write("   - 增加批量大小（如果内存允许）\n")
                    f.write("   - 使用数据并行训练\n")
                    f.write("   - 优化InfoNCE损失计算\n\n")

            if memory_analysis and 'error' not in memory_analysis:
                if not memory_analysis['memory_efficient']:
                    f.write("3. **内存优化**:\n")
                    f.write("   - 使用梯度累积减少批量大小\n")
                    f.write("   - 启用混合精度训练\n")
                    f.write("   - 优化模型架构\n\n")

            # 详细测试结果
            f.write("## 详细测试结果\n\n")

            # 训练性能表格
            if 'training' in self.results:
                f.write("### 训练性能详情\n\n")
                df = pd.DataFrame(self.results['training'])
                f.write(df.to_markdown(index=False))
                f.write("\n\n")

            # 可视化图表
            f.write("## 可视化分析\n\n")
            plot_files = [
                "training_performance.png",
                "memory_usage.png",
                "scalability.png",
                "component_performance.png"
            ]

            for plot_file in plot_files:
                plot_path = self.output_dir / "plots" / plot_file
                if plot_path.exists():
                    f.write(f"![{plot_file.replace('_', ' ').title()}](plots/{plot_file})\n\n")

            f.write("---\n")
            f.write("*报告由 ContrastiveIDTask 性能基准测试系统自动生成*\n")

        self.logger.info(f"基准测试报告已保存: {report_file}")
        return str(report_file)

    def run_full_benchmark(self) -> str:
        """运行完整的性能基准测试"""
        self.logger.info("开始完整性能基准测试...")

        start_time = time.time()

        try:
            # 1. 内存使用测试
            self.test_memory_usage()

            # 2. 训练性能测试
            self.test_training_performance()

            # 3. 模型组件测试
            self.test_model_components()

            # 4. 可扩展性测试
            self.test_scalability()

            # 5. 生成可视化
            self.generate_visualizations()

            # 6. 生成报告
            report_path = self.generate_report()

            end_time = time.time()
            duration = (end_time - start_time) / 60

            self.logger.info(f"基准测试完成，耗时 {duration:.1f} 分钟")
            self.logger.info(f"报告保存在: {report_path}")

            return report_path

        except Exception as e:
            self.logger.error(f"基准测试失败: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ContrastiveIDTask性能基准测试")

    parser.add_argument('--config', default='configs/id_contrastive/debug.yaml',
                       help='配置文件路径')
    parser.add_argument('--output_dir', default='save/performance_benchmark',
                       help='输出目录')
    parser.add_argument('--quick', action='store_true',
                       help='快速测试模式')
    parser.add_argument('--test', nargs='*',
                       choices=['memory', 'training', 'components', 'scalability'],
                       help='指定测试类型')

    args = parser.parse_args()

    # 创建基准测试实例
    benchmark = PerformanceBenchmark(
        config_path=args.config,
        output_dir=args.output_dir,
        quick_mode=args.quick
    )

    try:
        if args.test:
            # 运行指定的测试
            if 'memory' in args.test:
                benchmark.test_memory_usage()
            if 'training' in args.test:
                benchmark.test_training_performance()
            if 'components' in args.test:
                benchmark.test_model_components()
            if 'scalability' in args.test:
                benchmark.test_scalability()

            benchmark.generate_visualizations()
            report_path = benchmark.generate_report()
        else:
            # 运行完整基准测试
            report_path = benchmark.run_full_benchmark()

        print(f"\n{'='*60}")
        print("性能基准测试完成!")
        print(f"报告: {report_path}")
        print(f"结果目录: {args.output_dir}")
        print(f"{'='*60}")

        return 0

    except KeyboardInterrupt:
        benchmark.logger.warning("测试被中断")
        return 130
    except Exception as e:
        benchmark.logger.error(f"测试失败: {e}")
        return 1

if __name__ == "__main__":
    exit(main())