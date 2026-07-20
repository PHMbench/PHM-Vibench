import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

try:
    from ..utils.signal_processing import SignalProcessingUtils
    from ..utils.statistical_features import StatisticalFeatureExtractor
except ImportError:
    from utils.signal_processing import SignalProcessingUtils
    from utils.statistical_features import StatisticalFeatureExtractor


class LowPassExpert(nn.Module):
    """低频通带专家

    专门针对低频故障（如转子不平衡、基础振动）设计的专家。
    物理机理：低频段能量集中，通过低通滤波器突出低频特征。

    目标故障类型：
    - 转子不平衡
    - 基础振动
    - 低频机械松动

    核心算子：
    - 低通滤波器 (500Hz截止)
    - RMS能量统计
    - 低频频谱特征提取
    """

    def __init__(self,
                 cutoff_freq: float = 500.0,
                 sample_rate: float = 12000.0,
                 feature_dim: int = 64):
        super().__init__()

        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.feature_dim = feature_dim

        # 信号处理器
        self.signal_utils = SignalProcessingUtils()

        # 特征提取器
        self.feature_extractor = StatisticalFeatureExtractor()

        # 特征变换网络
        self.feature_net = nn.Sequential(
            nn.Linear(15, 32),  # 15个统计特征
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, feature_dim),
            nn.ReLU()
        )

        # 频域特征提取
        self.freq_net = nn.Sequential(
            nn.Linear(128, 64),  # 低频段频谱特征
            nn.ReLU(),
            nn.Linear(64, feature_dim // 2)
        )

        # 最终特征融合
        self.fusion_net = nn.Sequential(
            nn.Linear(feature_dim + feature_dim // 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: 输入信号 [batch_size, signal_length]
        Returns:
            features: 专家输出特征 [batch_size, feature_dim]
            metadata: 专家处理过程的元数据
        """
        batch_size, signal_len = x.shape

        # 1. 低通滤波处理
        with torch.no_grad():  # 滤波操作不需要梯度
            filtered_signal = self.signal_utils.low_pass_filter(
                x.detach(), self.cutoff_freq, self.sample_rate
            )
        filtered_signal = filtered_signal.clone().detach().requires_grad_(True)  # 重新添加梯度

        # 2. 提取统计特征
        stats_features = self.feature_extractor(filtered_signal)  # [batch_size, 15]
        stats_features = self.feature_net(stats_features)  # [batch_size, feature_dim]

        # 3. 提取低频频谱特征
        fft_filtered = torch.fft.fft(filtered_signal, dim=-1)
        spectrum_mag = torch.abs(fft_filtered[:, :signal_len//2])  # 取正频率部分

        # 只考虑低频部分（0-1000Hz）
        freq_bins = signal_len // 2
        low_freq_bins = min(freq_bins, int(1000 * freq_bins / (self.sample_rate / 2)))
        low_freq_spectrum = spectrum_mag[:, :low_freq_bins]

        # 统一到固定长度
        if low_freq_spectrum.shape[-1] < 128:
            # 填充到128
            padding_size = 128 - low_freq_spectrum.shape[-1]
            low_freq_spectrum = torch.cat([
                low_freq_spectrum,
                torch.zeros(batch_size, padding_size, device=x.device)
            ], dim=-1)
        else:
            # 截取前128个频率点
            low_freq_spectrum = low_freq_spectrum[:, :128]

        freq_features = self.freq_net(low_freq_spectrum)  # [batch_size, feature_dim//2]

        # 4. 特征融合
        combined_features = torch.cat([stats_features, freq_features], dim=-1)
        final_features = self.fusion_net(combined_features)  # [batch_size, feature_dim]

        # 5. 计算专家置信度
        confidence = self._compute_expert_confidence(filtered_signal, final_features)

        # 6. 收集元数据
        metadata = {
            'expert_type': 'low_pass',
            'cutoff_freq': self.cutoff_freq,
            'filtered_signal': filtered_signal,
            'spectrum_magnitude': spectrum_mag,
            'low_freq_energy': torch.mean(low_freq_spectrum ** 2, dim=-1),
            'confidence': confidence,
            'feature_stats': {
                'mean': torch.mean(stats_features),
                'std': torch.std(stats_features),
                'max': torch.max(stats_features),
                'min': torch.min(stats_features)
            }
        }

        return final_features, metadata

    def _compute_expert_confidence(self, filtered_signal: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """计算专家对当前信号的置信度

        基于低频能量占比和特征激活强度计算置信度
        """
        # 计算低频能量占比
        total_energy = torch.mean(filtered_signal ** 2, dim=-1)
        fft_signal = torch.fft.fft(filtered_signal, dim=-1)
        spectrum = torch.abs(fft_signal)

        # 低频段能量 (0-500Hz)
        signal_len = filtered_signal.shape[-1]
        freq_bins = signal_len // 2
        low_freq_bins = min(freq_bins, int(500 * freq_bins / (self.sample_rate / 2)))

        low_freq_energy = torch.sum(spectrum[:, :low_freq_bins] ** 2, dim=-1) / freq_bins
        total_spectrum_energy = torch.sum(spectrum[:, :freq_bins] ** 2, dim=-1) / freq_bins

        energy_ratio = low_freq_energy / (total_spectrum_energy + 1e-8)

        # 基于特征激活强度的置信度
        feature_activation = torch.mean(torch.abs(features), dim=-1)
        feature_confidence = torch.sigmoid(feature_activation - 1.0)  # 归一化到[0,1]

        # 综合置信度
        confidence = torch.sigmoid(2.0 * energy_ratio + feature_confidence - 1.0)

        return confidence

    def get_expert_description(self) -> Dict:
        """获取专家描述信息"""
        return {
            'expert_name': 'LowPassExpert',
            'expert_id': 'E1',
            'target_faults': ['转子不平衡', '基础振动', '低频机械松动'],
            'physical_mechanism': '低频段能量集中，通过低通滤波器突出低频特征',
            'key_parameters': {
                'cutoff_freq': self.cutoff_freq,
                'sample_rate': self.sample_rate,
                'feature_dim': self.feature_dim
            },
            'frequency_range': f'0-{self.cutoff_freq} Hz',
            'strengths': ['对低频故障敏感', '抗高频噪声能力强', '物理意义明确'],
            'limitations': ['对高频故障不敏感', '可能丢失瞬态冲击信息']
        }