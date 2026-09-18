"""LightTS point-forecasting path from THUML Time-Series-Library (MIT).

Continuous/interval sampling and the autoregressive highway are retained.
The configured chunk size must divide history; this scope avoids silently
increasing context length or reducing the user's requested chunk size.
"""
import torch
from torch import nn
from .._native_forecasting import check_history, positive_int


class IEBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nodes):
        super().__init__()
        self.spatial_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, hidden_dim // 4))
        self.channel_proj = nn.Linear(nodes, nodes)
        nn.init.eye_(self.channel_proj.weight)
        self.output_proj = nn.Linear(hidden_dim // 4, output_dim)

    def forward(self, x):
        x = self.spatial_proj(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel_proj(x)
        return self.output_proj(x.transpose(1, 2)).transpose(1, 2)


class Model(nn.Module):
    def __init__(self, args, metadata=None):
        super().__init__()
        self.seq_len, self.pred_len = positive_int(args, "seq_len"), positive_int(args, "pred_len")
        self.input_dim = positive_int(args, "input_dim")
        self.chunk_size, width = positive_int(args, "chunk_size"), positive_int(args, "d_model")
        if self.chunk_size > min(self.seq_len, self.pred_len) or self.seq_len % self.chunk_size:
            raise ValueError("LightTS chunk_size must divide seq_len and not exceed pred_len; no implicit padding/clamping")
        if width < 16 or width % 4:
            raise ValueError("LightTS model.d_model must be >= 16 and divisible by four")
        self.num_chunks = self.seq_len // self.chunk_size
        self.layer1 = IEBlock(self.chunk_size, width // 4, width // 4, self.num_chunks)
        self.layer2 = IEBlock(self.chunk_size, width // 4, width // 4, self.num_chunks)
        self.chunk_proj1, self.chunk_proj2 = nn.Linear(self.num_chunks, 1), nn.Linear(self.num_chunks, 1)
        self.layer3 = IEBlock(width // 2, width // 2, self.pred_len, self.input_dim)
        self.ar = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, file_id=None, task_id=None, return_feature=False):
        check_history(x, self.seq_len, self.input_dim, task_id, return_feature)
        batch = x.shape[0]
        continuous = x.reshape(batch, self.num_chunks, self.chunk_size, self.input_dim).permute(0, 3, 2, 1)
        interval = x.reshape(batch, self.chunk_size, self.num_chunks, self.input_dim).permute(0, 3, 1, 2)
        continuous = self.chunk_proj1(self.layer1(continuous.reshape(-1, self.chunk_size, self.num_chunks))).squeeze(-1)
        interval = self.chunk_proj2(self.layer2(interval.reshape(-1, self.chunk_size, self.num_chunks))).squeeze(-1)
        fused = torch.cat((continuous, interval), dim=-1).reshape(batch, self.input_dim, -1).transpose(1, 2)
        return self.layer3(fused) + self.ar(x.transpose(1, 2)).transpose(1, 2)
