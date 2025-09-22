#!/usr/bin/env python3
"""
ContrastiveIDTask性能测试套件
专为研究流程设计的性能基准测试
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[3]))

import torch
import numpy as np
import time
import gc
from argparse import Namespace
import warnings
warnings.filterwarnings("ignore")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil未安装，内存监控功能受限")

from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask


class PerformanceBenchmark:
    """性能基准测试器"""

    def __init__(self):
        self.results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 测试设备: {self.device}")

    def create_test_task(self, window_size=256, d_model=64):
        """创建测试任务"""
        args_data = Namespace(
            window_size=window_size,
            stride=window_size // 2,
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
            d_model=d_model,
            name="M_01_ISFM",
            backbone="B_08_PatchTST"
        )

        args_trainer = Namespace(
            epochs=1,
            accelerator=str(self.device),
            gpus=1 if torch.cuda.is_available() else 0
        )

        args_environment = Namespace(
            save_dir="save/"
        )

        # 创建网络
        network = torch.nn.Sequential(
            torch.nn.Linear(window_size * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, d_model)
        ).to(self.device)

        task = ContrastiveIDTask(
            network=network,
            args_data=args_data,
            args_model=args_model,
            args_task=args_task,
            args_trainer=args_trainer,
            args_environment=args_environment,
            metadata={}
        )

        return task, network

    def create_performance_dataset(self, num_samples, signal_length, num_channels=2):
        """创建性能测试数据集"""
        dataset = []

        for i in range(num_samples):
            # 生成高质量测试信号
            t = np.linspace(0, 1, signal_length)
            signal = np.zeros((signal_length, num_channels))

            # 多频率成分
            frequencies = [50, 120, 200, 350]
            amplitudes = [0.8, 0.4, 0.2, 0.1]

            for ch in range(num_channels):
                combined_signal = np.zeros(signal_length)
                for freq, amp in zip(frequencies, amplitudes):
                    phase = np.random.uniform(0, 2*np.pi)
                    combined_signal += amp * np.sin(2 * np.pi * freq * t + phase)

                # 添加调制和噪声
                modulation = 1 + 0.1 * np.sin(2 * np.pi * 10 * t)  # 10Hz调制
                noise = 0.05 * np.random.randn(signal_length)
                signal[:, ch] = combined_signal * modulation + noise

            metadata = {'Label': i % 4, 'ID': f'perf_{i:04d}'}
            dataset.append((f'perf_sample_{i:04d}', signal.astype(np.float32), metadata))

        return dataset

    def test_window_creation_performance(self):
        """测试窗口创建性能"""
        print("\n=== 测试窗口创建性能 ===")

        task, _ = self.create_test_task()

        # 不同信号长度测试
        signal_lengths = [1000, 5000, 10000, 20000]
        window_creation_results = {}

        for length in signal_lengths:
            print(f"📏 测试信号长度: {length}")

            # 创建测试信号
            signal = np.random.randn(length, 2).astype(np.float32)

            # 测试不同窗口数量
            num_windows = [2, 4, 8, 16]
            length_results = {}

            for num_win in num_windows:
                times = []

                # 多次测试取平均
                for _ in range(10):
                    start_time = time.time()
                    windows = task.create_windows(signal, num_window=num_win, strategy='random')
                    end_time = time.time()

                    times.append(end_time - start_time)

                avg_time = np.mean(times)
                length_results[num_win] = {
                    'avg_time': avg_time,
                    'windows_per_sec': num_win / avg_time if avg_time > 0 else 0
                }

                print(f"  {num_win}窗口: {avg_time*1000:.2f}ms ({num_win/avg_time:.1f} 窗口/秒)")

            window_creation_results[length] = length_results

        self.results['window_creation'] = window_creation_results
        print("✅ 窗口创建性能测试完成")

    def test_batch_processing_performance(self):
        """测试批处理性能"""
        print("\n=== 测试批处理性能 ===")

        batch_sizes = [4, 8, 16, 32] if not torch.cuda.is_available() else [4, 8, 16, 32, 64]
        batch_results = {}

        for batch_size in batch_sizes:
            print(f"📦 测试批大小: {batch_size}")

            task, _ = self.create_test_task()

            # 创建测试数据集
            dataset = self.create_performance_dataset(
                num_samples=batch_size * 4,
                signal_length=1024
            )

            # 测试批处理时间
            batch_times = []

            for i in range(0, len(dataset), batch_size):
                batch_data = dataset[i:i+batch_size]

                start_time = time.time()
                batch = task.prepare_batch(batch_data)
                end_time = time.time()

                if len(batch['ids']) > 0:
                    batch_times.append(end_time - start_time)

            if batch_times:
                avg_batch_time = np.mean(batch_times)
                samples_per_sec = batch_size / avg_batch_time

                batch_results[batch_size] = {
                    'avg_time': avg_batch_time,
                    'samples_per_sec': samples_per_sec
                }

                print(f"  平均时间: {avg_batch_time*1000:.2f}ms")
                print(f"  吞吐量: {samples_per_sec:.1f} 样本/秒")

        self.results['batch_processing'] = batch_results
        print("✅ 批处理性能测试完成")

    def test_infonce_computation_performance(self):
        """测试InfoNCE计算性能"""
        print("\n=== 测试InfoNCE计算性能 ===")

        task, _ = self.create_test_task()

        # 不同特征维度和批大小
        test_configs = [
            (4, 64), (8, 64), (16, 64), (32, 64),
            (4, 128), (8, 128), (16, 128),
            (4, 256), (8, 256)
        ]

        infonce_results = {}

        for batch_size, feature_dim in test_configs:
            config_key = f"batch_{batch_size}_dim_{feature_dim}"
            print(f"🧮 测试配置: Batch={batch_size}, Dim={feature_dim}")

            # 创建特征张量
            z_anchor = torch.randn(batch_size, feature_dim).to(self.device)
            z_positive = torch.randn(batch_size, feature_dim).to(self.device)

            # 预热（GPU情况下）
            if self.device.type == 'cuda':
                for _ in range(5):
                    _ = task.infonce_loss(z_anchor, z_positive)
                torch.cuda.synchronize()

            # 性能测试
            times = []
            num_iterations = 100

            for _ in range(num_iterations):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                start_time = time.time()
                loss = task.infonce_loss(z_anchor, z_positive)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                end_time = time.time()
                times.append(end_time - start_time)

            avg_time = np.mean(times)
            ops_per_sec = 1 / avg_time

            infonce_results[config_key] = {
                'avg_time_ms': avg_time * 1000,
                'ops_per_sec': ops_per_sec,
                'batch_size': batch_size,
                'feature_dim': feature_dim
            }

            print(f"  平均时间: {avg_time*1000:.3f}ms")
            print(f"  计算速度: {ops_per_sec:.1f} 次/秒")

        self.results['infonce_computation'] = infonce_results
        print("✅ InfoNCE计算性能测试完成")

    def test_memory_usage(self):
        """测试内存使用"""
        print("\n=== 测试内存使用 ===")

        if not PSUTIL_AVAILABLE and not torch.cuda.is_available():
            print("⚠️ 无法监控内存使用，跳过测试")
            return

        memory_results = {}

        # 不同数据规模测试
        test_scales = [
            (16, 1024, "小规模"),
            (64, 2048, "中规模"),
            (128, 4096, "大规模")
        ]

        for num_samples, signal_length, scale_name in test_scales:
            print(f"🧠 测试{scale_name}: {num_samples}样本 × {signal_length}长度")

            # 记录初始内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            else:
                initial_gpu_mem = 0

            if PSUTIL_AVAILABLE:
                initial_cpu_mem = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            else:
                initial_cpu_mem = 0

            # 创建任务和数据
            task, network = self.create_test_task(window_size=256, d_model=128)
            dataset = self.create_performance_dataset(num_samples, signal_length)

            # 处理数据
            all_batches = []
            for i in range(0, len(dataset), 8):
                batch_data = dataset[i:i+8]
                batch = task.prepare_batch(batch_data)
                if len(batch['ids']) > 0:
                    all_batches.append(batch)

            # 记录峰值内存
            if torch.cuda.is_available():
                peak_gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                gpu_usage = peak_gpu_mem - initial_gpu_mem
            else:
                gpu_usage = 0

            if PSUTIL_AVAILABLE:
                peak_cpu_mem = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                cpu_usage = peak_cpu_mem - initial_cpu_mem
            else:
                cpu_usage = 0

            memory_results[scale_name] = {
                'num_samples': num_samples,
                'signal_length': signal_length,
                'cpu_memory_mb': cpu_usage,
                'gpu_memory_mb': gpu_usage
            }

            print(f"  CPU内存使用: {cpu_usage:.1f} MB")
            print(f"  GPU内存使用: {gpu_usage:.1f} MB")

            # 清理内存
            del task, network, dataset, all_batches
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.results['memory_usage'] = memory_results
        print("✅ 内存使用测试完成")

    def test_scalability(self):
        """测试可扩展性"""
        print("\n=== 测试可扩展性 ===")

        # 测试不同数据集大小的处理时间
        dataset_sizes = [50, 100, 200, 500] if not torch.cuda.is_available() else [50, 100, 200, 500, 1000]
        scalability_results = {}

        task, network = self.create_test_task()

        for size in dataset_sizes:
            print(f"📊 测试数据集大小: {size}")

            # 创建数据集
            dataset = self.create_performance_dataset(size, 1024)

            # 测试完整处理时间
            start_time = time.time()

            processed_samples = 0
            for i in range(0, len(dataset), 16):  # 固定批大小
                batch_data = dataset[i:i+16]
                batch = task.prepare_batch(batch_data)

                if len(batch['ids']) > 0:
                    # 模拟前向传播
                    batch_size, seq_len, channels = batch['anchor'].shape
                    anchor_flat = batch['anchor'].reshape(batch_size, -1).to(self.device)
                    positive_flat = batch['positive'].reshape(batch_size, -1).to(self.device)

                    z_anchor = network(anchor_flat)
                    z_positive = network(positive_flat)

                    loss = task.infonce_loss(z_anchor, z_positive)
                    accuracy = task.compute_accuracy(z_anchor, z_positive)

                    processed_samples += len(batch['ids'])

            end_time = time.time()
            total_time = end_time - start_time

            scalability_results[size] = {
                'total_time': total_time,
                'processed_samples': processed_samples,
                'samples_per_sec': processed_samples / total_time,
                'time_per_sample': total_time / processed_samples if processed_samples > 0 else 0
            }

            print(f"  总时间: {total_time:.2f}s")
            print(f"  处理样本: {processed_samples}")
            print(f"  吞吐量: {processed_samples/total_time:.1f} 样本/秒")

        self.results['scalability'] = scalability_results
        print("✅ 可扩展性测试完成")

    def test_temperature_sensitivity(self):
        """测试温度参数敏感性"""
        print("\n=== 测试温度参数性能影响 ===")

        temperatures = [0.01, 0.05, 0.07, 0.1, 0.2, 0.5]
        temp_results = {}

        for temp in temperatures:
            print(f"🌡️ 测试温度: {temp}")

            task, _ = self.create_test_task()
            task.args_task.temperature = temp

            # 创建测试数据
            batch_size = 32
            feature_dim = 64
            z_anchor = torch.randn(batch_size, feature_dim).to(self.device)
            z_positive = torch.randn(batch_size, feature_dim).to(self.device)

            # 性能测试
            times = []
            for _ in range(50):
                start_time = time.time()
                loss = task.infonce_loss(z_anchor, z_positive)
                end_time = time.time()
                times.append(end_time - start_time)

            avg_time = np.mean(times)

            temp_results[temp] = {
                'avg_time_ms': avg_time * 1000,
                'loss_value': loss.item()
            }

            print(f"  平均时间: {avg_time*1000:.3f}ms")
            print(f"  损失值: {loss.item():.4f}")

        self.results['temperature_sensitivity'] = temp_results
        print("✅ 温度参数性能测试完成")

    def run_all_performance_tests(self):
        """运行所有性能测试"""
        print("🚀 开始ContrastiveIDTask性能基准测试")
        print("=" * 60)

        test_methods = [
            ("窗口创建性能", self.test_window_creation_performance),
            ("批处理性能", self.test_batch_processing_performance),
            ("InfoNCE计算性能", self.test_infonce_computation_performance),
            ("内存使用", self.test_memory_usage),
            ("可扩展性", self.test_scalability),
            ("温度参数敏感性", self.test_temperature_sensitivity)
        ]

        for test_name, test_method in test_methods:
            try:
                print(f"\n🔍 开始 {test_name} 测试")
                test_method()
            except Exception as e:
                print(f"❌ {test_name} 测试失败: {e}")
                continue

        # 生成性能报告
        self.generate_performance_report()

    def generate_performance_report(self):
        """生成性能报告"""
        print("\n📊 性能测试总结报告")
        print("=" * 60)

        # 批处理性能总结
        if 'batch_processing' in self.results:
            print("\n🔸 批处理性能:")
            for batch_size, metrics in self.results['batch_processing'].items():
                print(f"  批大小 {batch_size}: {metrics['samples_per_sec']:.1f} 样本/秒")

        # InfoNCE计算性能总结
        if 'infonce_computation' in self.results:
            print("\n🔸 InfoNCE计算性能:")
            for config, metrics in self.results['infonce_computation'].items():
                print(f"  {config}: {metrics['avg_time_ms']:.3f}ms")

        # 内存使用总结
        if 'memory_usage' in self.results:
            print("\n🔸 内存使用:")
            for scale, metrics in self.results['memory_usage'].items():
                print(f"  {scale}: CPU {metrics['cpu_memory_mb']:.1f}MB, GPU {metrics['gpu_memory_mb']:.1f}MB")

        # 可扩展性总结
        if 'scalability' in self.results:
            print("\n🔸 可扩展性:")
            for size, metrics in self.results['scalability'].items():
                print(f"  {size}样本: {metrics['samples_per_sec']:.1f} 样本/秒")

        print("\n✅ 性能基准测试完成")
        print("📈 详细结果已保存到self.results")


def run_performance_tests():
    """运行性能测试套件"""
    benchmark = PerformanceBenchmark()

    try:
        benchmark.run_all_performance_tests()
        return True
    except Exception as e:
        print(f"💥 性能测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_performance_tests()
    exit(0 if success else 1)