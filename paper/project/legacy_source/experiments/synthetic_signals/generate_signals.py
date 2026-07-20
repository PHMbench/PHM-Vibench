#!/usr/bin/env python3
"""
合成信号生成器 - 用于验证算子注意力机制

设计目的：
1. 验证算子注意力机制在可控信号上的有效性
2. 展示不同算子在不同类型信号上的权重分布
3. 为理论分析提供可控实验环境
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os


class SyntheticSignalGenerator:
    """合成信号生成器"""

    def __init__(self, sample_rate: int = 1024, duration: float = 1.0):
        """
        初始化信号生成器

        Args:
            sample_rate: 采样率 (Hz)
            duration: 信号时长 (秒)
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        self.time = np.linspace(0, duration, self.num_samples)

    def generate_single_frequency(self, freq: float, amplitude: float = 1.0,
                                noise_level: float = 0.0) -> np.ndarray:
        """
        生成单频正弦信号

        Args:
            freq: 频率 (Hz)
            amplitude: 幅度
            noise_level: 噪声水平 (0-1)

        Returns:
            信号数组
        """
        signal = amplitude * np.sin(2 * np.pi * freq * self.time)

        if noise_level > 0:
            noise = noise_level * np.random.randn(self.num_samples)
            signal += noise

        return signal

    def generate_dual_frequency(self, freq1: float, freq2: float,
                               amp1: float = 1.0, amp2: float = 0.5,
                               noise_level: float = 0.0) -> np.ndarray:
        """
        生成双频信号

        Args:
            freq1, freq2: 两个频率分量
            amp1, amp2: 对应的幅度
            noise_level: 噪声水平

        Returns:
            合成信号
        """
        signal1 = amp1 * np.sin(2 * np.pi * freq1 * self.time)
        signal2 = amp2 * np.sin(2 * np.pi * freq2 * self.time)
        signal = signal1 + signal2

        if noise_level > 0:
            noise = noise_level * np.random.randn(self.num_samples)
            signal += noise

        return signal

    def generate_am_modulated(self, carrier_freq: float, mod_freq: float,
                             mod_index: float = 0.5, noise_level: float = 0.0) -> np.ndarray:
        """
        生成幅度调制信号 (AM)

        Args:
            carrier_freq: 载波频率
            mod_freq: 调制频率
            mod_index: 调制指数 (0-1)
            noise_level: 噪声水平

        Returns:
            AM调制信号
        """
        # 载波信号
        carrier = np.sin(2 * np.pi * carrier_freq * self.time)
        # 调制信号
        modulator = 1 + mod_index * np.sin(2 * np.pi * mod_freq * self.time)

        signal = carrier * modulator

        if noise_level > 0:
            noise = noise_level * np.random.randn(self.num_samples)
            signal += noise

        return signal

    def generate_impulse(self, num_pulses: int = 3, pulse_width: float = 0.01,
                        amplitude: float = 1.0, noise_level: float = 0.0) -> np.ndarray:
        """
        生成脉冲信号

        Args:
            num_pulses: 脉冲数量
            pulse_width: 脉冲宽度 (秒)
            amplitude: 脉冲幅度
            noise_level: 噪声水平

        Returns:
            脉冲信号
        """
        signal = np.zeros(self.num_samples)
        pulse_samples = int(pulse_width * self.sample_rate)

        # 生成等间距脉冲
        pulse_interval = self.num_samples // (num_pulses + 1)

        for i in range(num_pulses):
            start_idx = (i + 1) * pulse_interval
            end_idx = start_idx + pulse_samples
            if end_idx < self.num_samples:
                signal[start_idx:end_idx] = amplitude

        if noise_level > 0:
            noise = noise_level * np.random.randn(self.num_samples)
            signal += noise

        return signal

    def generate_chirp(self, start_freq: float, end_freq: float,
                      method: str = 'linear', noise_level: float = 0.0) -> np.ndarray:
        """
        生成扫频信号

        Args:
            start_freq: 起始频率
            end_freq: 结束频率
            method: 'linear' 或 'logarithmic'
            noise_level: 噪声水平

        Returns:
            扫频信号
        """
        if method == 'linear':
            phase = 2 * np.pi * ((start_freq + (end_freq - start_freq) *
                                 self.time / self.duration) * self.time)
        else:  # logarithmic
            phase = 2 * np.pi * start_freq * self.duration * (
                np.exp(self.time * np.log(end_freq / start_freq)) / np.log(end_freq / start_freq) - 1)

        signal = np.sin(phase)

        if noise_level > 0:
            noise = noise_level * np.random.randn(self.num_samples)
            signal += noise

        return signal

    def generate_bearing_signal(self, fault_type: str = 'inner_race',
                               shaft_freq: float = 30.0,
                               ball_pass_freq: float = 3.6,
                               noise_level: float = 0.0) -> np.ndarray:
        """
        生成轴承故障信号

        Args:
            fault_type: 故障类型 ('inner_race', 'outer_race', 'ball', 'cage')
            shaft_freq: 轴频
            ball_pass_freq: 滚珠通过频率
            noise_level: 噪声水平

        Returns:
            轴承故障信号
        """
        # 基础轴频分量
        signal = self.generate_single_frequency(shaft_freq, 1.0, 0.0)

        # 故障特征频率
        fault_freqs = {
            'inner_race': ball_pass_freq * (1 + ball_count/2),
            'outer_race': ball_pass_freq * (1 - ball_count/2),
            'ball': 2 * ball_pass_freq,
            'cage': ball_pass_freq / 2
        }

        # 模拟8个滚珠
        ball_count = 8

        if fault_type in fault_freqs:
            fault_freq = fault_freqs[fault_type]
            signal += 0.3 * self.generate_single_frequency(fault_freq, 1.0, 0.0)

        # 添加调制效应
        signal += 0.2 * self.generate_am_modulated(shaft_freq, 1.0, 0.1, 0.0)

        if noise_level > 0:
            noise = noise_level * np.random.randn(self.num_samples)
            signal += noise

        return signal

    def visualize_signal(self, signal: np.ndarray, title: str = "Signal",
                         save_path: str = None):
        """
        可视化信号

        Args:
            signal: 信号数组
            title: 图标题
            save_path: 保存路径（可选）
        """
        plt.figure(figsize=(12, 6))

        # 时域图
        plt.subplot(2, 1, 1)
        plt.plot(self.time[:1000], signal[:1000])  # 只显示前1000个点
        plt.title(f'{title} - Time Domain')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        # 频域图
        plt.subplot(2, 1, 2)
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/self.sample_rate)
        magnitude = np.abs(fft[:len(fft)//2])

        plt.plot(freqs[:len(freqs)//2], magnitude[:len(magnitude)])
        plt.title(f'{title} - Frequency Domain')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.xlim(0, 200)  # 显示0-200Hz范围

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


class SignalDataset:
    """信号数据集管理"""

    def __init__(self):
        self.generator = SyntheticSignalGenerator()
        self.signals = {}
        self.metadata = {}

    def create_dataset(self, save_dir: str = "synthetic_data"):
        """
        创建完整的信号数据集

        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)

        # 1. 单频信号 - 验证FFT算子
        single_freq_50 = self.generator.generate_single_frequency(50)
        single_freq_100 = self.generator.generate_single_frequency(100)

        self.signals['single_freq_50'] = single_freq_50
        self.signals['single_freq_100'] = single_freq_100
        self.metadata['single_freq_50'] = {'type': 'single', 'freq': 50, 'operator_expectation': 'FFT'}
        self.metadata['single_freq_100'] = {'type': 'single', 'freq': 100, 'operator_expectation': 'FFT'}

        # 2. 双频信号 - 验证多算子协同
        dual_freq = self.generator.generate_dual_frequency(50, 150, amp1=1.0, amp2=0.5)
        self.signals['dual_freq'] = dual_freq
        self.metadata['dual_freq'] = {'type': 'dual', 'freqs': [50, 150], 'operator_expectation': 'FFT,WF'}

        # 3. AM调制信号 - 验证包络检测算子
        am_signal = self.generator.generate_am_modulated(100, 10, mod_index=0.8)
        self.signals['am_modulated'] = am_signal
        self.metadata['am_modulated'] = {'type': 'modulated', 'carrier': 100, 'mod': 10, 'operator_expectation': 'HT,Envelope'}

        # 4. 脉冲信号 - 验证时域算子
        impulse_signal = self.generator.generate_impulse(num_pulses=5, pulse_width=0.01)
        self.signals['impulse'] = impulse_signal
        self.metadata['impulse'] = {'type': 'impulse', 'pulses': 5, 'operator_expectation': 'I,Kurtosis'}

        # 5. 扫频信号 - 验证自适应算子选择
        chirp_signal = self.generator.generate_chirp(20, 200, method='linear')
        self.signals['chirp'] = chirp_signal
        self.metadata['chirp'] = {'type': 'chirp', 'freq_range': [20, 200], 'operator_expectation': 'Wavelet,FFT'}

        # 6. 轴承故障信号 - 验证故障检测算子
        inner_race = self.generator.generate_bearing_signal('inner_race', shaft_freq=30)
        outer_race = self.generator.generate_bearing_signal('outer_race', shaft_freq=30)
        ball_fault = self.generator.generate_bearing_signal('ball', shaft_freq=30)

        self.signals['bearing_inner'] = inner_race
        self.signals['bearing_outer'] = outer_race
        self.signals['bearing_ball'] = ball_fault
        self.metadata['bearing_inner'] = {'type': 'bearing_fault', 'fault': 'inner_race', 'operator_expectation': 'Envelope,HT'}
        self.metadata['bearing_outer'] = {'type': 'bearing_fault', 'fault': 'outer_race', 'operator_expectation': 'Envelope,HT'}
        self.metadata['bearing_ball'] = {'type': 'bearing_fault', 'fault': 'ball', 'operator_expectation': 'Kurtosis,Envelope'}

        # 7. 噪声信号 - 验证鲁棒性
        noisy_signal = self.generator.generate_single_frequency(100, noise_level=0.2)
        self.signals['noisy'] = noisy_signal
        self.metadata['noisy'] = {'type': 'noisy', 'base_freq': 100, 'operator_expectation': 'WF,I'}

        # 保存数据
        self.save_dataset(save_dir)

        # 生成可视化
        self.visualize_all(save_dir)

    def save_dataset(self, save_dir: str):
        """保存信号数据集"""
        os.makedirs(f"{save_dir}/signals", exist_ok=True)
        os.makedirs(f"{save_dir}/visualizations", exist_ok=True)

        for name, signal in self.signals.items():
            # 保存原始信号
            np.save(f"{save_dir}/signals/{name}.npy", signal)

            # 保存元数据
            import json
            with open(f"{save_dir}/signals/{name}_metadata.json", 'w') as f:
                json.dump(self.metadata[name], f, indent=2)

    def visualize_all(self, save_dir: str):
        """可视化所有信号"""
        for name, signal in self.signals.items():
            title = f"{name.replace('_', ' ').title()} - {self.metadata[name].get('type', '')}"
            save_path = f"{save_dir}/visualizations/{name}.png"
            self.generator.visualize_signal(signal, title, save_path)


def main():
    """主函数：生成合成信号数据集"""
    print("="*60)
    print("合成信号数据集生成器 - Operator Attention 机制验证")
    print("="*60)

    # 创建数据集
    dataset = SignalDataset()
    dataset.create_dataset("synthetic_data")

    print("\n✅ 合成信号数据集生成完成！")
    print("📁 保存位置: synthetic_data/")
    print("📊 信号类型:")

    for name, metadata in dataset.metadata.items():
        print(f"   • {name}: {metadata['type']}")
        print(f"     预期激活算子: {metadata.get('operator_expectation', 'N/A')}")

    print("\n🔥 下一步: 运行 test_operator_attention.py 验证算子注意力机制")


if __name__ == "__main__":
    main()