import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

try:
    from ..utils.signal_processing import SignalProcessingUtils
    from ..utils.statistical_features import StatisticalFeatureExtractor
except ImportError:
    from utils.signal_processing import SignalProcessingUtils
    from utils.statistical_features import StatisticalFeatureExtractor


class EnvelopeExpert(nn.Module):
    """包络分析专家

    专门针对高频冲击故障（如轴承故障）设计的专家。
    物理机理：高频冲击共振，通过带通滤波+包络解调提取故障特征。

    目标故障类型：
    - 轴承外圈故障
    - 轴承内圈故障
    - 滚动体故障

    核心算子：
    - 带通滤波器 (2000-5000Hz)
    - 希尔伯特变换包络提取
    - 包络谱分析
    """

    def __init__(self,
                 band_freq: Tuple[float, float] = (2000.0, 5000.0),
                 sample_rate: float = 12000.0,
                 feature_dim: int = 64):
        super().__init__()

        self.low_freq, self.high_freq = band_freq
        self.sample_rate = sample_rate
        self.feature_dim = feature_dim

        # 信号处理器
        self.signal_utils = SignalProcessingUtils()

        # 特征提取器
        self.feature_extractor = StatisticalFeatureExtractor()

        # 包络特征提取网络
        self.envelope_net = nn.Sequential(
            nn.Linear(128, 64),  # 包络谱特征
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, feature_dim // 2),
            nn.ReLU()
        )

        # 时域包络特征网络
        self.time_envelope_net = nn.Sequential(
            nn.Linear(15, 32),  # 包络信号的统计特征
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

        # 可学习的频率偏移
        self.freq_offset_low = nn.Parameter(torch.zeros(1))
        self.freq_offset_high = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: 输入信号 [batch_size, signal_length]
        Returns:
            features: 专家输出特征 [batch_size, feature_dim]
            metadata: 专家处理过程的元数据
        """
        batch_size, signal_len = x.shape

        # 1. 带通滤波提取高频段
        with torch.no_grad():  # 滤波操作不需要梯度
            band_filtered = self.signal_utils.band_pass_filter(
                x.detach(),
                self.low_freq + self.freq_offset_low.item(),
                self.high_freq + self.freq_offset_high.item(),
                self.sample_rate
            )
        band_filtered = band_filtered.clone().detach().requires_grad_(True)  # 重新添加梯度

        # 2. 计算包络信号
        envelope_signal = self.signal_utils.compute_envelope(band_filtered)

        # 3. 提取包络的统计特征
        envelope_stats = self.feature_extractor(envelope_signal)
        envelope_time_features = self.time_envelope_net(envelope_stats)

        # 4. 包络谱分析
        envelope_spectrum = self._compute_envelope_spectrum(envelope_signal)
        envelope_freq_features = self.envelope_net(envelope_spectrum)

        # 5. 特征融合
        combined_features = torch.cat([envelope_time_features, envelope_freq_features], dim=-1)
        final_features = self.fusion_net(combined_features)

        # 6. 计算专家置信度
        confidence = self._compute_expert_confidence(envelope_signal, envelope_spectrum)

        # 7. 收集元数据
        metadata = {
            'expert_type': 'envelope',
            'band_freq': (self.low_freq + self.freq_offset_low,
                         self.high_freq + self.freq_offset_high),
            'filtered_signal': band_filtered,
            'envelope_signal': envelope_signal,
            'envelope_spectrum': envelope_spectrum,
            'confidence': confidence,
            'impact_intensity': self._compute_impact_intensity(envelope_signal),
            'feature_stats': {
                'envelope_mean': torch.mean(envelope_signal),
                'envelope_std': torch.std(envelope_signal),
                'envelope_rms': torch.sqrt(torch.mean(envelope_signal ** 2))
            }
        }

        return final_features, metadata

    def _compute_envelope_spectrum(self, envelope_signal: torch.Tensor) -> torch.Tensor:
        """计算包络谱"""
        # 对包络信号进行FFT
        envelope_fft = torch.fft.fft(envelope_signal, dim=-1)
        envelope_spectrum = torch.abs(envelope_fft[:, :envelope_signal.shape[-1]//2])

        # 统一到固定长度
        target_length = 128
        if envelope_spectrum.shape[-1] < target_length:
            # 填充
            padding_size = target_length - envelope_spectrum.shape[-1]
            envelope_spectrum = torch.cat([
                envelope_spectrum,
                torch.zeros(envelope_spectrum.shape[0], padding_size, device=envelope_spectrum.device)
            ], dim=-1)
        else:
            # 截取
            envelope_spectrum = envelope_spectrum[:, :target_length]

        return envelope_spectrum  # [batch_size, 128]

    def _compute_expert_confidence(self, envelope_signal: torch.Tensor,
                                  envelope_spectrum: torch.Tensor) -> torch.Tensor:
        """计算专家置信度

        基于包络信号的冲击特征和谱峰值
        """
        # 1. 冲击特征：峭度
        envelope_centered = envelope_signal - torch.mean(envelope_signal, dim=-1, keepdim=True)
        kurtosis = torch.mean(envelope_centered ** 4, dim=-1) / (torch.var(envelope_signal, dim=-1) ** 2 + 1e-8)

        # 峭度得分 (冲击性强峭度大)
        kurtosis_score = torch.sigmoid((kurtosis - 3.0) / 2.0)  # 归一化

        # 2. 谱峰值特征
        spectrum_mean = torch.mean(envelope_spectrum, dim=-1)
        spectrum_max = torch.max(envelope_spectrum, dim=-1)[0]
        peak_ratio = spectrum_max / (spectrum_mean + 1e-8)

        # 峰值得分
        peak_score = torch.sigmoid((peak_ratio - 3.0) / 2.0)

        # 3. 包络能量稳定性
        envelope_rms = torch.sqrt(torch.mean(envelope_signal ** 2, dim=-1))
        energy_score = torch.sigmoid(envelope_rms / torch.max(envelope_rms))

        # 综合置信度
        confidence = 0.4 * kurtosis_score + 0.4 * peak_score + 0.2 * energy_score

        return confidence

    def _compute_impact_intensity(self, envelope_signal: torch.Tensor) -> torch.Tensor:
        """计算冲击强度指标"""
        # RMS
        rms = torch.sqrt(torch.mean(envelope_signal ** 2, dim=-1))

        # 峰值因子
        peak = torch.max(torch.abs(envelope_signal), dim=-1)[0]
        crest_factor = peak / (rms + 1e-8)

        # 峭度
        envelope_centered = envelope_signal - torch.mean(envelope_signal, dim=-1, keepdim=True)
        kurtosis = torch.mean(envelope_centered ** 4, dim=-1) / (torch.var(envelope_signal, dim=-1) ** 2 + 1e-8)

        # 冲击强度综合指标
        impact_intensity = rms * crest_factor * torch.relu(kurtosis - 3.0)

        return impact_intensity

    def detect_bearing_fault_freq(self, envelope_signal: torch.Tensor,
                                 bpfi: float = None, bpfo: float = None,
                                 bsf: float = None) -> Dict:
        """检测轴承故障特征频率

        Args:
            envelope_signal: 包络信号
            bpfi: 内圈故障频率 (Hz)
            bpfo: 外圈故障频率 (Hz)
            bsf: 滚动体故障频率 (Hz)
        """
        if all(f is None for f in [bpfi, bpfo, bsf]):
            return {'warning': 'No bearing fault frequencies provided'}

        # 计算包络谱
        envelope_fft = torch.fft.fft(envelope_signal, dim=-1)
        envelope_spectrum = torch.abs(envelope_fft)
        freqs = torch.fft.fftfreq(envelope_signal.shape[-1], d=1.0/self.sample_rate).to(envelope_spectrum.device)

        fault_detection = {}
        for fault_name, fault_freq in [('BPFI', bpfi), ('BPFO', bpfo), ('BSF', bsf)]:
            if fault_freq is not None:
                # 在故障频率附近找峰值
                freq_idx = torch.argmin(torch.abs(freqs - fault_freq))
                window_size = 5
                start_idx = max(0, freq_idx - window_size)
                end_idx = min(len(freqs), freq_idx + window_size + 1)

                peak_amplitude = torch.max(envelope_spectrum[:, start_idx:end_idx], dim=-1)[0]
                background_amplitude = torch.mean(envelope_spectrum[:, :start_idx//2], dim=-1)

                fault_ratio = peak_amplitude / (background_amplitude + 1e-8)

                fault_detection[fault_name] = {
                    'frequency': fault_freq,
                    'detected_amplitude': peak_amplitude,
                    'fault_ratio': fault_ratio,
                    'is_detected': torch.any(fault_ratio > 3.0)  # 简单阈值判断
                }

        return fault_detection

    def get_expert_description(self) -> Dict:
        """获取专家描述信息"""
        return {
            'expert_name': 'EnvelopeExpert',
            'expert_id': 'E3',
            'target_faults': ['轴承外圈故障', '轴承内圈故障', '滚动体故障'],
            'physical_mechanism': '高频冲击共振，通过带通滤波+包络解调提取故障特征',
            'key_parameters': {
                'band_freq': (self.low_freq, self.high_freq),
                'sample_rate': self.sample_rate,
                'feature_dim': self.feature_dim
            },
            'frequency_range': f'{self.low_freq}-{self.high_freq} Hz',
            'strengths': ['对冲击故障敏感', '能提取调制特征', '轴承故障诊断专用'],
            'limitations': ['依赖合适的滤波频带', '对低频故障不敏感', '需要较长的信号长度']
        }