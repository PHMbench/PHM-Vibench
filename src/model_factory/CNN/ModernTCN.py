"""ModernTCN classification, adapted from luodhhh/ModernTCN (MIT).

Copyright (c) 2024 luodhhh. Full MIT terms accompany the port below.
Preserves unmerged large/small depthwise kernels, variable-wise/feature-wise
FFNs, per-variable stem and the classification head. No unused forecasting
head, RevIN, structural reparameterization or training runner is imported.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .._native_classification import check_signal, class_count, positive_int, probability


def _positive_sequence(value, field):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"model.{field} must be a nonempty sequence")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in value):
        raise ValueError(f"model.{field} must contain positive integers")
    return tuple(value)


class ModernBlock(nn.Module):
    def __init__(self, variables, width, ffn_ratio, large_kernel, small_kernel, dropout):
        super().__init__()
        channels = variables * width
        hidden = width * ffn_ratio
        self.large = nn.Sequential(nn.Conv1d(channels, channels, large_kernel,
                                  padding=large_kernel // 2, groups=channels, bias=False), nn.BatchNorm1d(channels))
        self.small = nn.Sequential(nn.Conv1d(channels, channels, small_kernel,
                                  padding=small_kernel // 2, groups=channels, bias=False), nn.BatchNorm1d(channels))
        self.norm = nn.BatchNorm1d(width)
        self.ffn1pw1 = nn.Conv1d(channels, variables * hidden, 1, groups=variables)
        self.ffn1pw2 = nn.Conv1d(variables * hidden, channels, 1, groups=variables)
        self.ffn2pw1 = nn.Conv1d(channels, variables * hidden, 1, groups=width)
        self.ffn2pw2 = nn.Conv1d(variables * hidden, channels, 1, groups=width)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.drop3, self.drop4 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x):
        batch, variables, width, tokens = x.shape
        packed = x.reshape(batch, variables * width, tokens)
        mixed = self.large(packed) + self.small(packed)
        mixed = self.norm(mixed.reshape(batch * variables, width, tokens)).reshape(batch, variables * width, tokens)
        mixed = self.drop2(self.ffn1pw2(F.gelu(self.drop1(self.ffn1pw1(mixed)))))
        mixed = mixed.reshape(batch, variables, width, tokens).permute(0, 2, 1, 3).reshape(batch, width * variables, tokens)
        mixed = self.drop4(self.ffn2pw2(F.gelu(self.drop3(self.ffn2pw1(mixed)))))
        return x + mixed.reshape(batch, width, variables, tokens).permute(0, 2, 1, 3)


class Model(nn.Module):
    def __init__(self, args, metadata=None):
        super().__init__()
        self.seq_len, self.input_dim = positive_int(args, "seq_len"), positive_int(args, "input_dim")
        self.patch_size, self.patch_stride = positive_int(args, "patch_size"), positive_int(args, "patch_stride")
        downsample = positive_int(args, "downsample_ratio")
        ffn_ratio = positive_int(args, "ffn_ratio")
        dims = _positive_sequence(args.dims, "dims")
        blocks = _positive_sequence(args.num_blocks, "num_blocks")
        large = _positive_sequence(args.large_size, "large_size")
        small = _positive_sequence(args.small_size, "small_size")
        if not len(dims) == len(blocks) == len(large) == len(small):
            raise ValueError("ModernTCN dims, num_blocks and kernel lists must have identical stage counts")
        if not self.patch_stride <= self.patch_size <= self.seq_len or self.seq_len % self.patch_stride:
            raise ValueError("ModernTCN requires stride <= patch_size <= seq_len and seq_len divisible by stride")
        if any(lg % 2 == 0 or sm % 2 == 0 or sm > lg for lg, sm in zip(large, small)):
            raise ValueError("ModernTCN kernels must be odd and small_size <= large_size")
        dropout, class_dropout = probability(args, "dropout"), probability(args, "class_dropout")
        tokens = self.seq_len // self.patch_stride
        if tokens % (downsample ** (len(dims) - 1)):
            raise ValueError("ModernTCN supported stages require divisible downsampling; no tail repair")
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(nn.Conv1d(1, dims[0], self.patch_size, stride=self.patch_stride), nn.BatchNorm1d(dims[0]))
        ])
        for before, after in zip(dims, dims[1:]):
            self.downsample_layers.append(nn.Sequential(nn.BatchNorm1d(before), nn.Conv1d(before, after, downsample, stride=downsample)))
        self.stages = nn.ModuleList([
            nn.Sequential(*[ModernBlock(self.input_dim, dim, ffn_ratio, lg, sm, dropout) for _ in range(count)])
            for dim, count, lg, sm in zip(dims, blocks, large, small)
        ])
        final_tokens = tokens // (downsample ** (len(dims) - 1))
        self.class_dropout = nn.Dropout(class_dropout)
        self.classifier = nn.Linear(self.input_dim * dims[-1] * final_tokens, class_count(args))

    def forward(self, x, file_id=None, task_id=None, return_feature=False):
        check_signal(x, self.seq_len, self.input_dim, task_id)
        batch = x.shape[0]
        x = x.transpose(1, 2).unsqueeze(2)
        for index, (downsample, stage) in enumerate(zip(self.downsample_layers, self.stages)):
            packed = x.reshape(batch * self.input_dim, x.shape[2], x.shape[3])
            if index == 0 and self.patch_size != self.patch_stride:
                packed = F.pad(packed, (0, self.patch_size - self.patch_stride), mode="replicate")
            packed = downsample(packed)
            x = stage(packed.reshape(batch, self.input_dim, packed.shape[1], packed.shape[2]))
        features = self.class_dropout(F.gelu(x)).flatten(1)
        logits = self.classifier(features)
        return (logits, features) if return_feature else logits


# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
