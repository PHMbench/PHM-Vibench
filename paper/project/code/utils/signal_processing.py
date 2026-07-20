import torch
import torch.nn as nn
import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple, Optional


class SignalProcessingUtils:
    """信号处理工具函数集合"""

    @staticmethod
    def low_pass_filter(x: torch.Tensor, cutoff_freq: float, sample_rate: float = 12000, order: int = 4) -> torch.Tensor:
        """低通滤波器
        Args:
            x: 输入信号 [batch_size, signal_length]
            cutoff_freq: 截止频率 (Hz)
            sample_rate: 采样率 (Hz)
            order: 滤波器阶数
        Returns:
            filtered_x: 滤波后信号 [batch_size, signal_length]
        """
        batch_size, signal_len = x.shape
        x_np = x.detach().cpu().numpy()

        # 设计巴特沃斯低通滤波器
        nyquist_freq = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist_freq
        b, a = scipy_signal.butter(order, normalized_cutoff, btype='low')

        # 对每个样本进行滤波
        filtered_signals = []
        for i in range(batch_size):
            filtered_signal = scipy_signal.filtfilt(b, a, x_np[i])
            filtered_signals.append(filtered_signal)

        filtered_x = torch.tensor(np.array(filtered_signals), dtype=x.dtype, device=x.device)
        return filtered_x

    @staticmethod
    def band_pass_filter(x: torch.Tensor, low_freq: float, high_freq: float,
                       sample_rate: float = 12000, order: int = 4) -> torch.Tensor:
        """带通滤波器
        Args:
            x: 输入信号 [batch_size, signal_length]
            low_freq: 低频截止频率 (Hz)
            high_freq: 高频截止频率 (Hz)
            sample_rate: 采样率 (Hz)
            order: 滤波器阶数
        Returns:
            filtered_x: 滤波后信号 [batch_size, signal_length]
        """
        batch_size, signal_len = x.shape
        x_np = x.detach().cpu().numpy()

        # 设计巴特沃斯带通滤波器
        nyquist_freq = sample_rate / 2
        low_normalized = low_freq / nyquist_freq
        high_normalized = high_freq / nyquist_freq
        b, a = scipy_signal.butter(order, [low_normalized, high_normalized], btype='band')

        # 对每个样本进行滤波
        filtered_signals = []
        for i in range(batch_size):
            filtered_signal = scipy_signal.filtfilt(b, a, x_np[i])
            filtered_signals.append(filtered_signal)

        filtered_x = torch.tensor(np.array(filtered_signals), dtype=x.dtype, device=x.device)
        return filtered_x

    @staticmethod
    def compute_envelope(x: torch.Tensor) -> torch.Tensor:
        """计算信号包络
        Args:
            x: 输入信号 [batch_size, signal_length]
        Returns:
            envelope: 包络信号 [batch_size, signal_length]
        """
        x_np = x.detach().cpu().numpy()
        batch_size = x_np.shape[0]

        envelopes = []
        for i in range(batch_size):
            # 使用希尔伯特变换计算包络
            analytic_signal = scipy_signal.hilbert(x_np[i])
            envelope = np.abs(analytic_signal)
            envelopes.append(envelope)

        envelope_tensor = torch.tensor(np.array(envelopes), dtype=x.dtype, device=x.device)
        return envelope_tensor

    @staticmethod
    def extract_harmonics(x: torch.Tensor, fundamental_freq: float,
                         sample_rate: float = 12000, num_harmonics: int = 5) -> torch.Tensor:
        """提取谐波分量
        Args:
            x: 输入信号 [batch_size, signal_length]
            fundamental_freq: 基频 (Hz)
            sample_rate: 采样率 (Hz)
            num_harmonics: 谐波数量
        Returns:
            harmonics: 谐波能量 [batch_size, num_harmonics]
        """
        batch_size, signal_len = x.shape
        x_np = x.detach().cpu().numpy()

        # 计算频谱
        fft_x = np.fft.fft(x_np, axis=-1)
        freqs = np.fft.fftfreq(signal_len, d=1.0/sample_rate)

        harmonic_energies = []
        for i in range(batch_size):
            spectrum_mag = np.abs(fft_x[i])
            energies = []

            for h in range(1, num_harmonics + 1):
                harmonic_freq = fundamental_freq * h
                # 在谐波频率附近找到峰值
                freq_idx = np.argmin(np.abs(freqs - harmonic_freq))
                # 取谐波频率附近的平均能量
                window_size = 3
                start_idx = max(0, freq_idx - window_size)
                end_idx = min(len(freqs), freq_idx + window_size + 1)
                harmonic_energy = np.mean(spectrum_mag[start_idx:end_idx])
                energies.append(harmonic_energy)

            harmonic_energies.append(energies)

        harmonics = torch.tensor(np.array(harmonic_energies), dtype=x.dtype, device=x.device)
        return harmonics

    @staticmethod
    def compute_rms_energy(x: torch.Tensor, window_size: int = 256) -> torch.Tensor:
        """计算RMS能量
        Args:
            x: 输入信号 [batch_size, signal_length]
            window_size: 窗口大小
        Returns:
            rms_energy: RMS能量 [batch_size, signal_length // window_size]
        """
        batch_size, signal_len = x.shape

        # 重塑为窗口
        num_windows = signal_len // window_size
        x_windowed = x[:, :num_windows * window_size].reshape(batch_size, num_windows, window_size)

        # 计算每个窗口的RMS
        rms_energy = torch.sqrt(torch.mean(x_windowed ** 2, dim=-1))
        return rms_energy

    @staticmethod
    def apply_window(x: torch.Tensor, window_type: str = 'hann') -> torch.Tensor:
        """应用窗函数
        Args:
            x: 输入信号 [batch_size, signal_length]
            window_type: 窗函数类型 ('hann', 'hamming', 'blackman')
        Returns:
            windowed_x: 加窗后信号 [batch_size, signal_length]
        """
        signal_len = x.shape[-1]

        if window_type == 'hann':
            window = torch.hann_window(signal_len, device=x.device)
        elif window_type == 'hamming':
            window = torch.hamming_window(signal_len, device=x.device)
        elif window_type == 'blackman':
            window = torch.blackman_window(signal_len, device=x.device)
        else:
            raise ValueError(f"Unsupported window type: {window_type}")

        windowed_x = x * window.unsqueeze(0)
        return windowed_x