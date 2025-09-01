#!/usr/bin/env python3
"""
ContrastiveIDTask增强测试套件
包含单元测试、梯度测试、内存测试等
"""
import torch
import numpy as np
import unittest
import time
import psutil
import os
from argparse import Namespace

# 添加项目路径
import sys
sys.path.append('.')

from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask


class TestContrastiveIDTaskEnhanced(unittest.TestCase):
    
    def setUp(self):
        """测试初始化"""
        self.args_data = Namespace(
            window_size=128,
            stride=64,
            num_window=2,
            window_sampling_strategy='random',
            normalization=True,
            dtype='float32'
        )
        
        self.args_task = Namespace(
            lr=1e-3,
            temperature=0.07,
            weight_decay=1e-4,
            loss="CE",
            metrics=["acc"]
        )
        
        self.args_model = Namespace(
            d_model=64,
            name="M_01_ISFM",
            backbone="B_08_PatchTST"
        )
        
        self.args_trainer = Namespace(
            epochs=50,
            gpus=0,
            accelerator="cpu"
        )
        
        self.args_environment = Namespace(
            save_dir="tests/test_results/"
        )
    
    def create_task(self):
        """创建测试任务实例"""
        # 创建适配窗口大小的网络 (window_size * channels -> d_model)
        network = torch.nn.Sequential(
            torch.nn.Flatten(),  # 展平输入
            torch.nn.Linear(128 * 2, 64)  # 128窗口大小 * 2通道
        )
        return ContrastiveIDTask(
            network=network,
            args_data=self.args_data,
            args_model=self.args_model,
            args_task=self.args_task,
            args_trainer=self.args_trainer,
            args_environment=self.args_environment,
            metadata={}
        )
    
    def test_basic_functionality(self):
        """测试基本功能"""
        print("\n=== 测试基本功能 ===")
        task = self.create_task()
        
        # 测试任务创建
        self.assertIsInstance(task, ContrastiveIDTask)
        self.assertEqual(task.temperature, 0.07)
        
        # 测试窗口生成
        data = np.random.randn(1000, 2)
        windows = task.create_windows(data, num_window=2, strategy='random')
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].shape, (128, 2))
        
        print("✅ 基本功能测试通过")
    
    def test_batch_processing_variations(self):
        """测试不同批处理情况"""
        print("\n=== 测试批处理变体 ===")
        task = self.create_task()
        
        # 测试不同批大小
        for batch_size in [1, 2, 4, 8]:
            batch_data = [
                (f'id{i}', np.random.randn(500 + i * 100, 2), {'Label': i % 3})
                for i in range(batch_size)
            ]
            
            batch = task.prepare_batch(batch_data)
            
            self.assertEqual(len(batch['ids']), batch_size)
            self.assertEqual(batch['anchor'].shape[0], batch_size)
            self.assertEqual(batch['positive'].shape[0], batch_size)
        
        print("✅ 批处理变体测试通过")
    
    def test_gradient_flow(self):
        """测试梯度流"""
        print("\n=== 测试梯度流 ===")
        task = self.create_task()
        
        # 创建需要梯度的输入
        batch_size = 4
        feature_dim = 64
        z_anchor = torch.randn(batch_size, feature_dim, requires_grad=True)
        z_positive = torch.randn(batch_size, feature_dim, requires_grad=True)
        
        # 计算损失
        loss = task.infonce_loss(z_anchor, z_positive)
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        self.assertIsNotNone(z_anchor.grad)
        self.assertIsNotNone(z_positive.grad)
        self.assertFalse(torch.isnan(z_anchor.grad).any())
        self.assertFalse(torch.isnan(z_positive.grad).any())
        
        print(f"✅ 梯度流测试通过，损失值: {loss.item():.4f}")
    
    def test_memory_usage(self):
        """测试内存使用"""
        print("\n=== 测试内存使用 ===")
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        task = self.create_task()
        
        # 处理大批量数据（使用2通道匹配网络）
        large_batch_data = [
            (f'id{i}', np.random.randn(2000, 2), {'Label': i % 5})
            for i in range(20)
        ]
        
        batch = task.prepare_batch(large_batch_data)
        
        # 前向传播
        if len(batch['ids']) > 0:
            z_anchor = task.network(batch['anchor'])
            z_positive = task.network(batch['positive'])
            loss = task.infonce_loss(z_anchor, z_positive)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        self.assertLess(memory_increase, 500, f"内存增长过大: {memory_increase:.2f}MB")
        
        print(f"✅ 内存使用测试通过，内存增长: {memory_increase:.2f}MB")
    
    def test_performance_benchmarking(self):
        """测试性能基准"""
        print("\n=== 测试性能基准 ===")
        task = self.create_task()
        
        # 准备测试数据
        batch_data = [
            (f'id{i}', np.random.randn(1000, 2), {'Label': i % 3})
            for i in range(16)
        ]
        
        # 测量批处理时间
        start_time = time.time()
        for _ in range(10):
            batch = task.prepare_batch(batch_data)
            if len(batch['ids']) > 0:
                z_anchor = task.network(batch['anchor'])
                z_positive = task.network(batch['positive'])
                loss = task.infonce_loss(z_anchor, z_positive)
        
        total_time = time.time() - start_time
        avg_time = total_time / 10
        batches_per_second = 1.0 / avg_time
        
        self.assertGreater(batches_per_second, 1.0, "批处理速度过慢")
        
        print(f"✅ 性能测试通过，速度: {batches_per_second:.2f} batches/sec")
    
    def test_different_window_strategies(self):
        """测试不同窗口采样策略"""
        print("\n=== 测试窗口采样策略 ===")
        task = self.create_task()
        
        data = np.random.randn(2000, 2)
        strategies = ['random', 'sequential', 'evenly_spaced']
        
        for strategy in strategies:
            windows = task.create_windows(
                data, 
                num_window=3, 
                strategy=strategy
            )
            
            self.assertEqual(len(windows), 3)
            self.assertEqual(windows[0].shape, (128, 2))
            
            # 检查窗口不完全相同（除了evenly_spaced可能重复）
            if strategy != 'evenly_spaced':
                self.assertFalse(np.array_equal(windows[0], windows[1]))
        
        print("✅ 窗口策略测试通过")
    
    def test_temperature_sensitivity(self):
        """测试温度参数敏感性"""
        print("\n=== 测试温度参数敏感性 ===")
        
        # 测试不同温度值
        temperatures = [0.01, 0.07, 0.5, 1.0]
        losses = []
        
        for temp in temperatures:
            self.args_task.temperature = temp
            task = self.create_task()
            
            # 相同的输入特征
            z_anchor = torch.randn(4, 64)
            z_positive = torch.randn(4, 64)
            
            loss = task.infonce_loss(z_anchor, z_positive)
            losses.append(loss.item())
        
        # 温度越低，损失通常越大（更严格的对比）
        self.assertGreater(losses[0], losses[-1], "温度参数影响不符合预期")
        
        print("✅ 温度敏感性测试通过")
    
    def test_edge_cases_comprehensive(self):
        """全面测试边界情况"""
        print("\n=== 测试边界情况 ===")
        task = self.create_task()
        
        # 测试1: 空批次
        empty_batch = task.prepare_batch([])
        self.assertEqual(len(empty_batch['ids']), 0)
        
        # 测试2: 单样本
        single_sample = [('single', np.random.randn(300, 1), {'Label': 0})]
        batch = task.prepare_batch(single_sample)
        self.assertEqual(len(batch['ids']), 1)
        
        # 测试3: 极短序列
        short_sample = [('short', np.random.randn(50, 1), {'Label': 0})]
        batch = task.prepare_batch(short_sample)
        self.assertEqual(len(batch['ids']), 0)  # 应该被过滤
        
        # 测试4: 极长序列
        long_sample = [('long', np.random.randn(10000, 1), {'Label': 0})]
        batch = task.prepare_batch(long_sample)
        self.assertEqual(len(batch['ids']), 1)  # 应该成功处理
        
        # 测试5: NaN数据
        nan_data = np.random.randn(1000, 1)
        nan_data[100:200] = np.nan
        nan_sample = [('nan', nan_data, {'Label': 0})]
        batch = task.prepare_batch(nan_sample)
        # 根据实现，可能被过滤或处理
        
        print("✅ 边界情况测试通过")
    
    def test_device_compatibility(self):
        """测试设备兼容性"""
        print("\n=== 测试设备兼容性 ===")
        
        # CPU测试
        task_cpu = self.create_task()
        z_anchor = torch.randn(2, 64)
        z_positive = torch.randn(2, 64)
        loss_cpu = task_cpu.infonce_loss(z_anchor, z_positive)
        self.assertFalse(loss_cpu.is_cuda)
        
        # GPU测试（如果可用）
        if torch.cuda.is_available():
            self.args_trainer.gpus = 1
            task_gpu = self.create_task()
            z_anchor_gpu = z_anchor.cuda()
            z_positive_gpu = z_positive.cuda()
            loss_gpu = task_gpu.infonce_loss(z_anchor_gpu, z_positive_gpu)
            self.assertTrue(loss_gpu.is_cuda)
        
        print("✅ 设备兼容性测试通过")


def run_enhanced_tests():
    """运行增强测试套件"""
    print("开始ContrastiveIDTask增强测试...")
    print("=" * 60)
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestContrastiveIDTaskEnhanced)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 所有增强测试通过！")
        print(f"运行测试: {result.testsRun}")
        return True
    else:
        print("❌ 部分测试失败")
        print(f"失败: {len(result.failures)}, 错误: {len(result.errors)}")
        return False


if __name__ == "__main__":
    success = run_enhanced_tests()
    exit(0 if success else 1)