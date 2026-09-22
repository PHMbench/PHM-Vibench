"""TimesNet classification, adapted from THUML Time-Series-Library (MIT).

Source: models/TimesNet.py, layers/Embed.py and layers/Conv_Blocks.py at
4e938a1767106324dd753b2a44832bf870a0252e. Scope: fixed-length, fully observed
float32 BLC signals. The forecast and missing-data paths are not implemented.

Copyright (c) 2021 THUML @ Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
from collections.abc import Mapping
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F


def _positive_int(args, name):
    value = getattr(args, name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"model.{name} must be a positive integer, got {value!r}")
    return value


def fft_periods(x, top_k):
    """Return period lengths and per-sample weights, preserving batch selection.

    Upstream sets DC strength to zero, which still lets DC win a zero-spectrum
    tie. Mask only that bin to -inf; never change the observed signal itself.
    """
    amplitude = torch.fft.rfft(x, dim=1).abs()
    strength = amplitude.mean(0).mean(-1)
    strength[0] = -torch.inf
    bins = strength.topk(top_k).indices
    periods = torch.div(x.shape[1], bins, rounding_mode="floor")
    return periods.tolist(), amplitude.mean(-1).index_select(1, bins)


class InceptionBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_kernels):
        super().__init__()
        self.kernels = nn.ModuleList([
            nn.Conv2d(input_dim, output_dim, 2 * i + 1, padding=i)
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
            # Period-grid padding is part of TimesNet, not user-input repair.
            grid = F.pad(x, (0, 0, 0, padded_length - length))
            grid = grid.reshape(batch, padded_length // period, period, dim).permute(0, 3, 1, 2)
            result = self.conv(grid).permute(0, 2, 3, 1).reshape(batch, padded_length, dim)
            outputs.append(result[:, :length])
        return x + (torch.stack(outputs, dim=-1) * weights.softmax(1)[:, None, None, :]).sum(-1)


class _Embedding(nn.Module):
    def __init__(self, channels, dim, length, dropout):
        super().__init__()
        self.token = nn.Conv1d(channels, dim, 3, padding=1, padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.token.weight, mode="fan_in", nonlinearity="leaky_relu")
        position = torch.arange(length, dtype=torch.float32)[:, None]
        frequency = (torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim)).exp()
        pe = torch.zeros(1, length, dim)
        pe[0, :, 0::2] = torch.sin(position * frequency)
        pe[0, :, 1::2] = torch.cos(position * frequency)
        self.register_buffer("position", pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.token(x.transpose(1, 2)).transpose(1, 2) + self.position)


class Model(nn.Module):
    """Model(args_model, metadata); logits [B,K], or (logits, features).

    Requires seq_len, input_dim, num_classes, d_model, d_ff, e_layers, top_k,
    num_kernels and dropout. Metadata/file_id do not change the network path.
    Only the all-valid classification mask is supported; no implicit scaling.
    """
    def __init__(self, args_model, metadata=None):
        super().__init__()
        self.seq_len = _positive_int(args_model, "seq_len")
        self.input_dim = _positive_int(args_model, "input_dim")
        dim = _positive_int(args_model, "d_model")
        ff_dim = _positive_int(args_model, "d_ff")
        depth = _positive_int(args_model, "e_layers")
        top_k = _positive_int(args_model, "top_k")
        num_kernels = _positive_int(args_model, "num_kernels")
        if dim % 2:
            raise ValueError("TimesNet model.d_model must be even for sinusoidal positions")
        if not 2 <= self.seq_len <= 5000:
            raise ValueError("TimesNet classification supports model.seq_len in [2, 5000]")
        if top_k > self.seq_len // 2:
            raise ValueError("model.top_k exceeds the positive FFT bins")
        dropout = args_model.dropout
        if isinstance(dropout, bool) or not isinstance(dropout, Real) or not 0 <= dropout < 1:
            raise ValueError("model.dropout must be a number in [0, 1)")
        classes = args_model.num_classes
        if isinstance(classes, Mapping):
            if len(classes) != 1:
                raise ValueError("TimesNet has one class head; num_classes must define one ontology")
            classes = next(iter(classes.values()))
        if isinstance(classes, bool) or not isinstance(classes, int) or classes < 2:
            raise ValueError("model.num_classes must be an integer >= 2")
        self.model = nn.ModuleList([TimesBlock(dim, ff_dim, top_k, num_kernels) for _ in range(depth)])
        self.enc_embedding = _Embedding(self.input_dim, dim, self.seq_len, float(dropout))
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(float(dropout))
        self.projection = nn.Linear(dim * self.seq_len, classes)

    def forward(self, x, file_id=None, task_id=None, return_feature=False):
        if task_id not in (None, "classification"):
            raise ValueError(f"TimesNet implements classification only, got task_id={task_id!r}")
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("TimesNet input must be a float32 tensor [B,L,C]")
        if x.shape[0] < 1 or tuple(x.shape[1:]) != (self.seq_len, self.input_dim):
            raise ValueError(f"Expected non-empty [B,{self.seq_len},{self.input_dim}], got {tuple(x.shape)}")
        if x.dtype != torch.float32:
            raise TypeError(f"TimesNet input must be float32, got {x.dtype}")
        if not torch.isfinite(x).all():
            raise FloatingPointError("TimesNet input contains NaN or Inf")
        encoded = self.enc_embedding(x)
        for block in self.model:
            encoded = self.layer_norm(block(encoded))
        features = self.dropout(F.gelu(encoded)).flatten(1)
        logits = self.projection(features)
        return (logits, features) if return_feature else logits
