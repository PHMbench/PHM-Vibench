import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional

try:
    from ..utils.signal_processing import SignalProcessingUtils
    from ..utils.statistical_features import StatisticalFeatureExtractor
except ImportError:
    from utils.signal_processing import SignalProcessingUtils
    from utils.statistical_features import StatisticalFeatureExtractor


class HarmonicExpert(nn.Module):
    """谐波分析专家

    专门针对谐波故障（如转子不对中、齿轮故障）设计的专家。
    物理机理：转频谐波振动，基频与倍频并存，通过梳状滤波器提取谐波分量。

    目标故障类型：
    - 转子不对中
    - 齿轮故障
    - 偏心故障

    核心算子：
    - 梳状滤波器 (提取基频及其倍频)
    - FFT谐波分析
    - 谐波能量统计
    """

    def __init__(self,
                 fundamental_freq: float = 50.0,  # 基频(Hz)，可根据实际情况调整
                 num_harmonics: int = 5,
                 sample_rate: float = 12000.0,
                 feature_dim: int = 64):
        super().__init__()

        self.fundamental_freq = fundamental_freq
        self.num_harmonics = num_harmonics
        self.sample_rate = sample_rate
        self.feature_dim = feature_dim

        # 信号处理器
        self.signal_utils = SignalProcessingUtils()

        # 特征提取器
        self.feature_extractor = StatisticalFeatureExtractor()

        # 谐波特征提取网络
        self.harmonic_net = nn.Sequential(
            nn.Linear(num_harmonics * 2, 32),  # 谐波幅值 + 相位
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, feature_dim // 2),
            nn.ReLU()
        )

        # 统计特征网络
        self.stats_net = nn.Sequential(
            nn.Linear(15, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, feature_dim // 2),
            nn.ReLU()
        )

        # 特征融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 可学习的基频偏移 (用于适应实际工况)
        self.freq_offset = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: 输入信号 [batch_size, signal_length]
        Returns:
            features: 专家输出特征 [batch_size, feature_dim]
            metadata: 专家处理过程的元数据
        """
        batch_size, signal_len = x.shape

        # 1. 提取谐波分量
        harmonic_features = self.signal_utils.extract_harmonics(
            x, self.fundamental_freq + self.freq_offset.item(),
            self.sample_rate, self.num_harmonics
        )

        # 2. 计算谐波相位信息
        harmonic_phases = self._extract_harmonic_phases(x)

        # 3. 组合谐波特征
        harmonic_combined = torch.cat([harmonic_features, harmonic_phases], dim=-1)
        harmonic_features = self.harmonic_net(harmonic_combined)

        # 4. 提取原始信号的统计特征
        stats_features = self.feature_extractor(x)
        stats_features = self.stats_net(stats_features)

        # 5. 特征融合
        combined_features = torch.cat([harmonic_features, stats_features], dim=-1)
        final_features = self.fusion_net(combined_features)

        # 6. 计算专家置信度
        confidence = self._compute_expert_confidence(harmonic_features, harmonic_combined)

        # 7. 收集元数据
        metadata = {
            'expert_type': 'harmonic',
            'fundamental_freq': self.fundamental_freq + self.freq_offset.item(),
            'num_harmonics': self.num_harmonics,
            'harmonic_energies': harmonic_features,
            'harmonic_phases': harmonic_phases,
            'confidence': confidence,
            'harmonic_spectrum': self._compute_harmonic_spectrum(x),
            'feature_stats': {
                'harmonic_mean': torch.mean(harmonic_features),
                'harmonic_std': torch.std(harmonic_features),
                'stats_mean': torch.mean(stats_features),
                'stats_std': torch.std(stats_features)
            }
        }

        return final_features, metadata

    def _extract_harmonic_phases(self, x: torch.Tensor) -> torch.Tensor:
        """提取谐波相位信息"""
        batch_size = x.shape[0]
        x_np = x.detach().cpu().numpy()

        harmonic_phases = []
        for i in range(batch_size):
            signal_i = x_np[i]
            fft_signal = np.fft.fft(signal_i)
            freqs = np.fft.fftfreq(len(signal_i), d=1.0/self.sample_rate)

            phases = []
            for h in range(1, self.num_harmonics + 1):
                harmonic_freq = self.fundamental_freq * h + self.freq_offset.item()
                freq_idx = np.argmin(np.abs(freqs - harmonic_freq))

                # 提取相位
                phase = np.angle(fft_signal[freq_idx])
                phases.append(phase)

            harmonic_phases.append(phases)

        return torch.tensor(np.array(harmonic_phases), dtype=x.dtype, device=x.device)

    def _compute_harmonic_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """计算谐波频谱"""
        fft_x = torch.fft.fft(x, dim=-1)
        spectrum_mag = torch.abs(fft_x)

        # 提取谐波频率点
        signal_len = x.shape[-1]
        freqs = torch.fft.fftfreq(signal_len, d=1.0/self.sample_rate).to(x.device)

        harmonic_spectrum = []
        for h in range(1, self.num_harmonics + 1):
            harmonic_freq = self.fundamental_freq * h + self.freq_offset.item()
            freq_idx = torch.argmin(torch.abs(freqs - harmonic_freq))
            harmonic_spectrum.append(spectrum_mag[:, freq_idx])

        return torch.stack(harmonic_spectrum, dim=-1)  # [batch_size, num_harmonics]

    def _compute_expert_confidence(self, harmonic_features: torch.Tensor,
                                  harmonic_combined: torch.Tensor) -> torch.Tensor:
        """计算专家置信度

        基于谐波能量的规律性和幅值衰减模式
        """
        # 谐波幅值特征 (前num_harmonics个)
        harmonic_magnitudes = harmonic_combined[:, :self.num_harmonics]

        # 理想谐波模式：高次谐波幅值递减
        magnitude_ratios = []
        for h in range(1, self.num_harmonics):
            ratio = harmonic_magnitudes[:, h] / (harmonic_magnitudes[:, 0] + 1e-8)
            magnitude_ratios.append(ratio)

        # 计算谐波规律性得分
        harmonic_regularity = 1.0 - torch.std(torch.stack(magnitude_ratios, dim=-1), dim=-1)
        harmonic_regularity = torch.sigmoid(harmonic_regularity)

        # 基于特征激活强度的置信度
        feature_activation = torch.mean(torch.abs(harmonic_features), dim=-1)
        feature_confidence = torch.sigmoid(feature_activation - 1.0)

        # 综合置信度
        confidence = 0.6 * harmonic_regularity + 0.4 * feature_confidence

        return confidence

    def get_harmonic_analysis(self, x: torch.Tensor) -> Dict:
        """详细的谐波分析结果"""
        with torch.no_grad():
            harmonic_spectrum = self._compute_harmonic_spectrum(x)
            harmonic_phases = self._extract_harmonic_phases(x)

            # 计算谐波失真度
            fundamental_amp = harmonic_spectrum[:, 0]
            total_harmonics = torch.sum(harmonic_spectrum[:, 1:], dim=-1)
            thd = total_harmonics / (fundamental_amp + 1e-8)

            return {
                'harmonic_amplitudes': harmonic_spectrum,
                'harmonic_phases': harmonic_phases,
                'total_harmonic_distortion': thd,
                'dominant_harmonic': torch.argmax(harmonic_spectrum, dim=-1),
                'fundamental_frequency': self.fundamental_freq + self.freq_offset.item()
            }

    def get_expert_description(self) -> Dict:
        """获取专家描述信息"""
        return {
            'expert_name': 'HarmonicExpert',
            'expert_id': 'E2',
            'target_faults': ['转子不对中', '齿轮故障', '偏心故障'],
            'physical_mechanism': '转频谐波振动，基频与倍频并存，通过梳状滤波器提取谐波分量',
            'key_parameters': {
                'fundamental_freq': self.fundamental_freq,
                'num_harmonics': self.num_harmonics,
                'sample_rate': self.sample_rate,
                'feature_dim': self.feature_dim
            },
            'frequency_range': f'{self.fundamental_freq}-{self.fundamental_freq * self.num_harmonics} Hz',
            'strengths': ['对谐波故障敏感', '能识别不对中特征', '相位信息丰富'],
            'limitations': ['依赖准确的基频', '对非谐波故障不敏感', '计算复杂度较高']
        }