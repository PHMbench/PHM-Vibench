import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple


class StatisticalFeatureExtractor(nn.Module):
    """统计特征提取器

    提取13种可解释统计特征，用于路由决策：
    - 时域特征：均值、标准差、方差、熵、最大值、最小值、绝对均值、峰度、均方根、峰值因子、偏度、间隙因子、形状因子
    - 频域特征：频谱均值、频谱重心
    """

    def __init__(self, sample_rate: float = 12000.0):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入信号 [batch_size, signal_length]
        Returns:
            stats: 统计特征 [batch_size, 15]
        """
        batch_size, signal_len = x.shape
        stats = []

        # 时域特征
        # 1. 均值
        mean_val = torch.mean(x, dim=-1)
        stats.append(mean_val)

        # 2. 标准差
        std_val = torch.std(x, dim=-1)
        stats.append(std_val)

        # 3. 方差
        var_val = torch.var(x, dim=-1)
        stats.append(var_val)

        # 4. 最大值
        max_val = torch.max(x, dim=-1)[0]
        stats.append(max_val)

        # 5. 最小值
        min_val = torch.min(x, dim=-1)[0]
        stats.append(min_val)

        # 6. 绝对均值
        abs_mean = torch.mean(torch.abs(x), dim=-1)
        stats.append(abs_mean)

        # 7. 均方根 (RMS)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1))
        stats.append(rms)

        # 8. 峰值因子
        peak_factor = max_val / (rms + 1e-8)
        stats.append(peak_factor)

        # 9. 偏度
        x_centered = x - mean_val.unsqueeze(-1)
        skewness = torch.mean(x_centered ** 3, dim=-1) / (torch.std(x, dim=-1) ** 3 + 1e-8)
        stats.append(skewness)

        # 10. 间隙因子
        clearance_factor = torch.max(x ** 2, dim=-1)[0] / (torch.mean(torch.sqrt(torch.abs(x)), dim=-1) ** 2 + 1e-8)
        stats.append(clearance_factor)

        # 11. 形状因子
        shape_factor = rms / (abs_mean + 1e-8)
        stats.append(shape_factor)

        # 12. 峭度
        kurtosis = torch.mean(x_centered ** 4, dim=-1) / (torch.var(x, dim=-1) ** 2 + 1e-8) - 3
        stats.append(kurtosis)

        # 13. 熵 (简化版)
        hist = torch.histc(x, bins=50, min=min_val.min().item(), max=max_val.max().item())
        hist_norm = hist / (torch.sum(hist) + 1e-8)
        entropy = -torch.sum(hist_norm * torch.log(hist_norm + 1e-8))
        entropy = entropy.expand(batch_size)  # 扩展到batch维度
        stats.append(entropy)

        # 频域特征
        # 14. 频谱均值
        fft_x = torch.fft.rfft(x, dim=-1)
        spectrum_mag = torch.abs(fft_x)
        spectrum_mean = torch.mean(spectrum_mag, dim=-1)
        stats.append(spectrum_mean)

        # 15. 频谱重心
        freqs = torch.fft.rfftfreq(signal_len, d=1.0 / self.sample_rate).to(x.device)
        spectral_centroid = torch.sum(freqs.unsqueeze(0) * spectrum_mag, dim=-1) / (torch.sum(spectrum_mag, dim=-1) + 1e-8)
        stats.append(spectral_centroid)

        return torch.stack(stats, dim=-1)

    def get_feature_names(self) -> List[str]:
        """返回特征名称列表"""
        return [
            'mean', 'std', 'var', 'max', 'min', 'abs_mean', 'rms',
            'peak_factor', 'skewness', 'clearance_factor', 'shape_factor',
            'kurtosis', 'entropy', 'spectrum_mean', 'spectral_centroid'
        ]

    def interpret_features(self, features: torch.Tensor) -> List[Dict]:
        """解释统计特征的物理意义"""
        feature_names = self.get_feature_names()
        features_np = features.detach().cpu().numpy()
        batch_size = features_np.shape[0]

        interpretations = []
        for i in range(batch_size):
            interpretation = {}

            # 能量指标
            interpretation['energy_level'] = 'high' if features_np[i, 6] > 0.5 else 'normal'  # RMS
            interpretation['impact_intensity'] = 'strong' if features_np[i, 11] > 3.0 else 'weak'  # Kurtosis

            # 频率特征
            interpretation['frequency_characteristic'] = (
                'low_freq' if features_np[i, 14] < 500 else
                'mid_freq' if features_np[i, 14] < 2000 else
                'high_freq'
            )  # Spectral centroid

            # 波形特征
            interpretation['waveform_pattern'] = (
                'impulsive' if features_np[i, 7] > 4.0 else
                'periodic' if features_np[i, 10] > 1.2 else
                'random'
            )  # Peak factor & Shape factor

            # 偏态特征
            interpretation['asymmetry'] = (
                'positive_skew' if features_np[i, 8] > 0.5 else
                'negative_skew' if features_np[i, 8] < -0.5 else
                'symmetric'
            )  # Skewness

            interpretations.append(interpretation)

        return interpretations
