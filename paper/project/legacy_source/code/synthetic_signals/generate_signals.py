"""
合成信号生成器 - Operator Attention理论验证
用于生成验证算子选择机制的各类测试信号
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SignalConfig:
    """信号配置参数"""
    sample_rate: float = 4096.0  # 采样率 (Hz)
    duration: float = 1.0        # 信号时长 (s)
    num_samples: int = 4096      # 采样点数

    def __post_init__(self):
        self.num_samples = int(self.sample_rate * self.duration)


class SyntheticSignalGenerator:
    """合成信号生成器基类"""

    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        self.time = np.linspace(0, self.config.duration, self.config.num_samples)

    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        """信号标准化到[-1, 1]范围"""
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            return signal / max_val
        return signal

    def add_noise(self, signal: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        """添加高斯白噪声"""
        signal_power = np.mean(signal ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power) * np.random.randn(len(signal))
        return signal + noise


class SingleFrequencyGenerator(SyntheticSignalGenerator):
    """单频信号生成器

    用于验证FFT算子在特定频段的激活强度
    """

    def generate(self, frequency: float, amplitude: float = 1.0,
                 phase: float = 0.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        生成单频正弦信号

        Args:
            frequency: 信号频率 (Hz)
            amplitude: 信号幅度
            phase: 初始相位 (rad)

        Returns:
            signal: 生成的信号
            metadata: 信号元数据
        """
        signal = amplitude * np.sin(2 * np.pi * frequency * self.time + phase)
        signal = self._normalize(signal)

        metadata = {
            'type': 'single_frequency',
            'frequency': frequency,
            'amplitude': amplitude,
            'phase': phase,
            'expected_dominant_operator': 'FFT',
            'expected_fft_weight': 0.8,
            'expected_other_weight': 0.05
        }

        return signal, metadata


class DualFrequencyGenerator(SyntheticSignalGenerator):
    """双频信号生成器

    用于验证多算子的协同激活，特别是FFT与WF的组合
    """

    def generate(self, f1: float, f2: float, ratio: float = 0.5,
                 phase1: float = 0.0, phase2: float = 0.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        生成双频调制信号

        Args:
            f1, f2: 两个频率分量 (Hz)
            ratio: 第二个频率的幅度比例
            phase1, phase2: 初始相位 (rad)

        Returns:
            signal: 生成的信号
            metadata: 信号元数据
        """
        signal1 = np.sin(2 * np.pi * f1 * self.time + phase1)
        signal2 = ratio * np.sin(2 * np.pi * f2 * self.time + phase2)
        signal = self._normalize(signal1 + signal2)

        metadata = {
            'type': 'dual_frequency',
            'frequencies': [f1, f2],
            'ratio': ratio,
            'expected_dominant_operators': ['FFT', 'WF'],
            'expected_combined_weight': 0.7,
            'frequency_ratio': f2/f1 if f1 > 0 else 0
        }

        return signal, metadata


class TransientSignalGenerator(SyntheticSignalGenerator):
    """瞬态信号生成器

    用于验证HT（希尔伯特变换）算子对瞬态特征的敏感性
    """

    def generate(self, carrier_freq: float, center_time: float,
                 bandwidth: float = 100.0, amplitude: float = 1.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        生成高斯包络调制的瞬态信号

        Args:
            carrier_freq: 载波频率 (Hz)
            center_time: 瞬态中心时间 (s)
            bandwidth: 时域带宽 (控制瞬态持续时间)
            amplitude: 信号幅度

        Returns:
            signal: 生成的信号
            metadata: 信号元数据
        """
        # 高斯包络
        envelope = np.exp(-((self.time - center_time) ** 2) / (2 * (1/bandwidth) ** 2))

        # 调制信号
        carrier = np.sin(2 * np.pi * carrier_freq * self.time)
        signal = amplitude * envelope * carrier
        signal = self._normalize(signal)

        metadata = {
            'type': 'transient',
            'carrier_frequency': carrier_freq,
            'center_time': center_time,
            'bandwidth': bandwidth,
            'expected_dominant_operator': 'HT',
            'expected_ht_weight': 0.6,
            'transient_duration': 4.0 / bandwidth
        }

        return signal, metadata


class NoisySignalGenerator(SyntheticSignalGenerator):
    """噪声信号生成器

    用于验证算子注意力在噪声环境下的鲁棒性
    """

    def generate(self, base_signal: np.ndarray, noise_types: list = None,
                 snr_db: float = 20.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        生成带噪声的信号

        Args:
            base_signal: 基础干净信号
            noise_types: 噪声类型列表 ['gaussian', 'impulse', 'colored']
            snr_db: 信噪比 (dB)

        Returns:
            signal: 带噪声的信号
            metadata: 信号元数据
        """
        if noise_types is None:
            noise_types = ['gaussian']

        signal = base_signal.copy()

        for noise_type in noise_types:
            if noise_type == 'gaussian':
                # 高斯白噪声
                signal = self.add_noise(signal, snr_db)
            elif noise_type == 'impulse':
                # 脉冲噪声
                num_impulses = int(0.01 * len(signal))
                impulse_indices = np.random.choice(len(signal), num_impulses, replace=False)
                impulse_strength = 0.5 * np.max(np.abs(signal))
                signal[impulse_indices] += impulse_strength * np.random.randn(num_impulses)
            elif noise_type == 'colored':
                # 有色噪声（通过滤波产生）
                white_noise = np.random.randn(len(signal))
                # 简单的低通滤波产生有色噪声
                alpha = 0.9
                colored_noise = np.zeros_like(white_noise)
                colored_noise[0] = white_noise[0]
                for i in range(1, len(white_noise)):
                    colored_noise[i] = alpha * colored_noise[i-1] + (1-alpha) * white_noise[i]
                # 调整有色噪声功率
                signal += self._normalize(colored_noise) * 0.1 * np.max(np.abs(signal))

        metadata = {
            'type': 'noisy',
            'noise_types': noise_types,
            'snr_db': snr_db,
            'noise_power_ratio': 10 ** (-snr_db / 10.0),
            'expected_weight_stability': 0.9  # 期望权重保持90%的稳定性
        }

        return signal, metadata


class FaultSimulatorGenerator(SyntheticSignalGenerator):
    """故障模拟信号生成器

    模拟典型的机械故障特征信号
    """

    def generate_bearing_fault(self, shaft_freq: float, fault_freq: float,
                              fault_type: str = 'inner') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        生成轴承故障信号

        Args:
            shaft_freq: 转轴频率 (Hz)
            fault_freq: 故障特征频率 (Hz)
            fault_type: 故障类型 ('inner', 'outer', 'ball', 'cage')

        Returns:
            signal: 故障信号
            metadata: 故障元数据
        """
        # 基础转频信号
        shaft_signal = 0.3 * np.sin(2 * np.pi * shaft_freq * self.time)

        # 故障特征（冲击序列）
        impact_interval = int(self.config.sample_rate / fault_freq)
        impact_positions = np.arange(0, len(self.time), impact_interval)

        # 冲击响应（指数衰减正弦）
        decay_rate = 100.0  # 衰减率
        carrier_freq_fault = fault_freq * 3  # 故障共振频率

        fault_signal = np.zeros_like(self.time)
        for pos in impact_positions:
            if pos < len(self.time):
                # 生成单个冲击
                t_impact = self.time[pos:] - self.time[pos]
                decay_envelope = np.exp(-decay_rate * t_impact)
                impact = np.sin(2 * np.pi * carrier_freq_fault * t_impact[:len(decay_envelope)]) * decay_envelope
                fault_signal[pos:pos+len(impact)] += impact

        # 合成信号
        signal = self._normalize(shaft_signal + 0.5 * fault_signal)

        metadata = {
            'type': 'bearing_fault',
            'fault_type': fault_type,
            'shaft_frequency': shaft_freq,
            'fault_frequency': fault_freq,
            'impact_count': len(impact_positions),
            'expected_dominant_operators': ['WF', 'HT'],
            'expected_combined_weight': 0.6
        }

        return signal, metadata

    def generate_gear_fault(self, gear_mesh_freq: float, fault_freq: float,
                           amplitude_modulation: float = 0.3) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        生成齿轮故障信号（幅值调制）

        Args:
            gear_mesh_freq: 齿轮啮合频率 (Hz)
            fault_freq: 故障调制频率 (Hz)
            amplitude_modulation: 调制深度

        Returns:
            signal: 齿轮故障信号
            metadata: 故障元数据
        """
        # 载波（啮合频率）
        carrier = np.sin(2 * np.pi * gear_mesh_freq * self.time)

        # 调制信号（故障频率）
        modulator = 1 + amplitude_modulation * np.sin(2 * np.pi * fault_freq * self.time)

        # 调制信号
        signal = carrier * modulator
        signal = self._normalize(signal)

        metadata = {
            'type': 'gear_fault',
            'gear_mesh_frequency': gear_mesh_freq,
            'fault_frequency': fault_freq,
            'modulation_depth': amplitude_modulation,
            'expected_dominant_operators': ['HT', 'FFT'],
            'expected_combined_weight': 0.7
        }

        return signal, metadata


def create_test_signals() -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
    """
    创建所有测试信号的集合

    Returns:
        signals: 字典，包含所有测试信号及其元数据
    """
    config = SignalConfig(sample_rate=4096, duration=1.0)
    signals = {}

    # 1. 单频信号测试集（覆盖不同频段）
    single_gen = SingleFrequencyGenerator(config)
    for freq in [10, 100, 500, 1000, 2000]:
        signal, metadata = single_gen.generate(frequency=freq)
        signals[f'single_freq_{freq}Hz'] = (signal, metadata)

    # 2. 双频信号测试集
    dual_gen = DualFrequencyGenerator(config)
    test_cases = [
        (100, 300, 0.5),   # 低频组合
        (500, 1500, 0.3),  # 中频组合
        (1000, 3000, 0.2), # 高频组合
    ]
    for f1, f2, ratio in test_cases:
        signal, metadata = dual_gen.generate(f1=f1, f2=f2, ratio=ratio)
        signals[f'dual_freq_{f1}_{f2}Hz'] = (signal, metadata)

    # 3. 瞬态信号测试集
    transient_gen = TransientSignalGenerator(config)
    test_cases = [
        (100, 0.2, 200),    # 低频载波早期瞬态
        (1000, 0.5, 500),   # 中频载波中期瞬态
        (2000, 0.8, 1000),  # 高频载波晚期瞬态
    ]
    for carrier, center, bw in test_cases:
        signal, metadata = transient_gen.generate(
            carrier_freq=carrier,
            center_time=center,
            bandwidth=bw
        )
        signals[f'transient_{carrier}Hz_t{center}s'] = (signal, metadata)

    # 4. 噪声鲁棒性测试
    noisy_gen = NoisySignalGenerator(config)
    # 使用单频信号作为基础
    base_signal, _ = single_gen.generate(frequency=100)
    for snr in [30, 20, 10, 0]:
        signal, metadata = noisy_gen.generate(
            base_signal=base_signal,
            snr_db=snr
        )
        signals[f'noisy_snr{snr}dB'] = (signal, metadata)

    # 5. 故障模拟信号
    fault_gen = FaultSimulatorGenerator(config)

    # 轴承故障
    signal, metadata = fault_gen.generate_bearing_fault(
        shaft_freq=30,
        fault_freq=150,
        fault_type='inner'
    )
    signals['bearing_fault'] = (signal, metadata)

    # 齿轮故障
    signal, metadata = fault_gen.generate_gear_fault(
        gear_mesh_freq=500,
        fault_freq=50,
        amplitude_modulation=0.4
    )
    signals['gear_fault'] = (signal, metadata)

    return signals


def convert_to_tensor(signal: np.ndarray) -> torch.Tensor:
    """将numpy信号转换为PyTorch张量"""
    return torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0)  # [1, 1, N]


if __name__ == "__main__":
    # 测试信号生成器
    print("Generating test signals for Operator Attention validation...")
    signals = create_test_signals()

    print(f"\nGenerated {len(signals)} test signals:")
    for name, (signal, metadata) in signals.items():
        print(f"  - {name}: {metadata['type']}")

    # 可视化示例信号
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()

    test_names = list(signals.keys())[:9]
    for idx, name in enumerate(test_names):
        signal, metadata = signals[name]
        time_axis = np.linspace(0, 1, len(signal))

        axes[idx].plot(time_axis, signal, linewidth=0.8)
        axes[idx].set_title(f"{name}\n{metadata['type']}")
        axes[idx].set_xlabel("Time (s)")
        axes[idx].set_ylabel("Amplitude")
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/Paper/TII_operator_attention/results/synthetic_signal_examples.png",
                dpi=300, bbox_inches='tight')
    print("\nSaved example signals visualization to: results/synthetic_signal_examples.png")