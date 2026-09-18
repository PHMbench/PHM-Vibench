"""Native TimesNet classification path (THUML, MIT).

The architecture is frequency-selected periodic 2D convolution, not Transformer
attention. Padding inside a periodic grid is part of TimesNet, not input repair.
See ../NATIVE_MODELS.md for source and supported scope.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .._native_classification import check_signal, class_count, positive_int, probability


def fft_periods(x, top_k):
    """Select positive-frequency bins; a constant signal cannot select DC."""
    amplitude = torch.fft.rfft(x, dim=1).abs()
    frequencies = amplitude.mean(0).mean(-1)[1:].topk(top_k).indices + 1
    periods = torch.div(x.shape[1], frequencies, rounding_mode="floor")
    return periods.tolist(), amplitude.mean(-1).index_select(1, frequencies)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels):
        super().__init__()
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
            for i in range(num_kernels)
        ])
        for kernel in self.kernels:
            nn.init.kaiming_normal_(kernel.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(kernel.bias)

    def forward(self, x):
        return torch.stack([kernel(x) for kernel in self.kernels], dim=-1).mean(-1)


class TimesBlock(nn.Module):
    def __init__(self, dim, ff_dim, top_k, num_kernels):
        super().__init__()
        self.top_k = top_k
        self.conv = nn.Sequential(
            InceptionBlock(dim, ff_dim, num_kernels), nn.GELU(),
            InceptionBlock(ff_dim, dim, num_kernels),
        )

    def forward(self, x):
        batch, length, dim = x.shape
        periods, weights = fft_periods(x, self.top_k)
        outputs = []
        for period in periods:
            padded_length = ((length + period - 1) // period) * period
            grid = F.pad(x, (0, 0, 0, padded_length - length))
            grid = grid.reshape(batch, padded_length // period, period, dim).permute(0, 3, 1, 2)
            result = self.conv(grid).permute(0, 2, 3, 1).reshape(batch, padded_length, dim)
            outputs.append(result[:, :length])
        return x + (torch.stack(outputs, dim=-1) * weights.softmax(1)[:, None, None, :]).sum(-1)


class Model(nn.Module):
    """Classify fixed-length, fully observed float32 signals without resampling."""

    def __init__(self, args, metadata=None):
        super().__init__()
        self.seq_len = positive_int(args, "seq_len")
        self.input_dim = positive_int(args, "input_dim")
        dim = positive_int(args, "d_model")
        ff_dim = positive_int(args, "d_ff")
        depth = positive_int(args, "e_layers")
        top_k = positive_int(args, "top_k")
        num_kernels = positive_int(args, "num_kernels")
        dropout = probability(args, "dropout")
        if dim % 2:
            raise ValueError("TimesNet sinusoidal embedding requires an even model.d_model")
        if not 1 <= top_k <= self.seq_len // 2:
            raise ValueError("model.top_k must not exceed the number of positive FFT bins")
        self.value_embedding = nn.Conv1d(self.input_dim, dim, 3, padding=1, padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.value_embedding.weight, mode="fan_in", nonlinearity="leaky_relu")
        position = torch.arange(self.seq_len, dtype=torch.float32)[:, None]
        frequency = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, self.seq_len, dim)
        pe[0, :, 0::2] = torch.sin(position * frequency)
        pe[0, :, 1::2] = torch.cos(position * frequency)
        self.register_buffer("position_embedding", pe)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TimesBlock(dim, ff_dim, top_k, num_kernels) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(dim * self.seq_len, class_count(args))

    def forward(self, x, file_id=None, task_id=None, return_feature=False):
        check_signal(x, self.seq_len, self.input_dim, task_id)
        x = self.embedding_dropout(self.value_embedding(x.transpose(1, 2)).transpose(1, 2) + self.position_embedding)
        for block in self.blocks:
            x = self.norm(block(x))
        features = self.dropout(F.gelu(x)).flatten(1)
        logits = self.projection(features)
        return (logits, features) if return_feature else logits
