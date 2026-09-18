"""SOFTS point forecasting, adapted from Secilia-Cxy/SOFTS (MIT).

Copyright (c) 2024 Xu-Yang Chen. The full MIT notice is included below.
No calendar covariates, external runner, classification head or weight downloads.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .._native_forecasting import boolean, check_history, positive_int, probability


class STAR(nn.Module):
    """Series-core fusion with upstream stochastic training pooling."""
    def __init__(self, d_series, d_core):
        super().__init__()
        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, x):
        batch, channels, _ = x.shape
        core = self.gen2(F.gelu(self.gen1(x)))
        weights = core.softmax(dim=1)
        if self.training:
            indices = torch.multinomial(weights.permute(0, 2, 1).reshape(-1, channels), 1)
            indices = indices.view(batch, -1, 1).permute(0, 2, 1)
            pooled = core.gather(1, indices)
        else:
            pooled = (core * weights).sum(dim=1, keepdim=True)
        fused = torch.cat((x, pooled.expand(-1, channels, -1)), dim=-1)
        return self.gen4(F.gelu(self.gen3(fused)))


class SOFTSLayer(nn.Module):
    def __init__(self, dim, core_dim, ff_dim, dropout, activation):
        super().__init__()
        self.star = STAR(dim, core_dim)
        self.conv1, self.conv2 = nn.Conv1d(dim, ff_dim, 1), nn.Conv1d(ff_dim, dim, 1)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.star(x)))
        y = self.dropout(self.activation(self.conv1(x.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        return self.norm2(x + y)


class Model(nn.Module):
    def __init__(self, args, metadata=None):
        super().__init__()
        self.seq_len, self.pred_len = positive_int(args, "seq_len"), positive_int(args, "pred_len")
        self.input_dim = positive_int(args, "input_dim")
        dim, core_dim = positive_int(args, "d_model"), positive_int(args, "d_core")
        ff_dim, depth = positive_int(args, "d_ff"), positive_int(args, "e_layers")
        dropout = probability(args, "dropout")
        self.use_norm = boolean(args, "use_norm")
        if args.activation not in ("relu", "gelu"):
            raise ValueError("model.activation must be relu or gelu")
        self.embedding = nn.Linear(self.seq_len, dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([SOFTSLayer(dim, core_dim, ff_dim, dropout, args.activation) for _ in range(depth)])
        self.projection = nn.Linear(dim, self.pred_len)

    def forward(self, x, file_id=None, task_id=None, return_feature=False):
        check_history(x, self.seq_len, self.input_dim, task_id, return_feature)
        if self.use_norm:
            mean = x.mean(1, keepdim=True).detach()
            centered = x - mean
            std = (centered.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
            x = centered / std
        encoded = self.dropout(self.embedding(x.transpose(1, 2)))
        for layer in self.layers:
            encoded = layer(encoded)
        result = self.projection(encoded).transpose(1, 2)
        return result * std + mean if self.use_norm else result


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
