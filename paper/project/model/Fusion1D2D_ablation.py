"""
Fusion1D2D with Ablation Study Support
支持消融实验的1D-2D融合模型版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import sys
import os

# 添加主项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

class Fusion1D2D_Ablation(nn.Module):
    """
    支持消融实验的1D-2D融合模型
    """

    def __init__(self,
                 signal_processing_modules: Dict,
                 feature_extractor_modules: Dict,
                 args: Any):
        """
        初始化模型
        """
        super(Fusion1D2D_Ablation, self).__init__()

        # 提取参数
        self.input_dim = getattr(args, 'in_dim', 4096)
        self.in_channels = getattr(args, 'in_channels', 2)
        self.out_channels = getattr(args, 'out_channels', 3)
        self.num_classes = getattr(args, 'num_classes', 5)
        self.skip_connection = getattr(args, 'skip_connection', True)

        # 消融实验开关
        self.ablation_1d_only = getattr(args, 'ablation_1d_only', False)
        self.ablation_2d_only = getattr(args, 'ablation_2d_only', False)
        self.ablation_no_statistical = getattr(args, 'ablation_no_statistical', False)

        # 信号处理层（简化版）
        self.signal_processing_layers = nn.ModuleList()
        for i in range(4):
            config_key = f'layer{i+1}'
            layer_config = getattr(args, config_key, ['I', 'WF', 'I'])
            actual_input_dim = self.input_dim * self.in_channels
            self.signal_processing_layers.append(
                nn.Sequential(
                    nn.Linear(actual_input_dim, actual_input_dim),
                    nn.ReLU(inplace=True)
                )
            )

        # 特征提取器
        def simple_feature_extractor(x):
            """提取统计特征"""
            mean = torch.mean(x, dim=-1)
            std = torch.std(x, dim=-1)
            max_val = torch.max(x, dim=-1)[0]
            min_val = torch.min(x, dim=-1)[0]
            rms = torch.sqrt(torch.mean(x**2, dim=-1))

            features = torch.cat([mean, std, max_val, min_val, rms], dim=-1)
            return features

        self.feature_extractor = simple_feature_extractor

        # 1D分支
        self.one_d_branch = nn.Sequential(
            nn.Conv1d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # 2D分支
        def create_spectrogram(signal):
            """将1D信号转换为2D频谱图"""
            batch_size, channels, seq_len = signal.shape

            spectrograms = []
            for i in range(batch_size):
                x = signal[i, 0, :]
                stft = torch.stft(x, n_fft=256, hop_length=64, return_complex=True)
                magnitude = torch.abs(stft)
                log_mag = torch.log1p(magnitude)
                log_mag = F.interpolate(log_mag.unsqueeze(0).unsqueeze(0), size=(64, 64), mode='bilinear').squeeze()
                spectrograms.append(log_mag)

            return torch.stack(spectrograms).unsqueeze(1)

        self.spectrogram_converter = create_spectrogram

        self.two_d_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # 根据消融配置计算融合维度
        if self.ablation_1d_only:
            fusion_dim = 64  # 仅1D分支
        elif self.ablation_2d_only:
            fusion_dim = 64  # 仅2D分支
        elif self.ablation_no_statistical:
            fusion_dim = 64 + 64  # 1D + 2D，无统计特征
        else:
            fusion_dim = 64 + 64 + self.in_channels * 5  # 完整融合

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        # 输入格式处理
        if x.dim() == 3 and x.size(-1) in [2, 3]:
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
        elif x.dim() == 3:
            batch_size = x.size(0)
            x = x.transpose(1, 2).contiguous().view(batch_size, -1)

        # 信号处理
        for layer in self.signal_processing_layers:
            x = layer(x)

        # 重塑为CNN格式
        batch_size = x.size(0)
        total_features = x.size(1)
        target_channels = self.in_channels
        target_seq_len = total_features // target_channels

        if target_seq_len * target_channels != total_features:
            usable_features = target_seq_len * target_channels
            x = x[:, :usable_features]

        x = x.view(batch_size, target_channels, target_seq_len)

        max_seq_len = 1024
        if target_seq_len > max_seq_len:
            x = x[:, :, :max_seq_len]
            target_seq_len = max_seq_len

        # 收集各分支特征
        features_list = []

        # 1D分支
        if not self.ablation_2d_only:
            one_d_features = self.one_d_branch(x)
            features_list.append(one_d_features)

        # 2D分支
        if not self.ablation_1d_only:
            spectrogram = self.spectrogram_converter(x)
            two_d_features = self.two_d_branch(spectrogram)
            features_list.append(two_d_features)

        # 统计特征
        if not self.ablation_no_statistical:
            stat_features = self.feature_extractor(x)
            features_list.append(stat_features)

        # 融合特征
        if len(features_list) > 0:
            fused = torch.cat(features_list, dim=1)
            logits = self.classifier(fused)
        else:
            # 边缘情况：如果所有分支都被禁用
            batch_size = x.size(0)
            logits = torch.zeros(batch_size, self.num_classes, device=x.device)

        return logits