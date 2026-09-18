"""PAttn forecasting from THUML Time-Series-Library (MIT).

A single patch-attention encoder per series, without positional/calendar tokens.
Reuses the exact THUML-style encoder already ported for iTransformer. Internal
replication padding is the upstream patch algorithm, not missing-data repair.
"""
import torch
from torch import nn
from .._native_forecasting import check_history, positive_int, probability
from .iTransformer import InvertedEncoderLayer


class Model(nn.Module):
    def __init__(self, args, metadata=None):
        super().__init__()
        self.seq_len, self.pred_len = positive_int(args, "seq_len"), positive_int(args, "pred_len")
        self.input_dim = positive_int(args, "input_dim")
        self.patch_size, self.stride = positive_int(args, "patch_size"), positive_int(args, "stride")
        width, heads, ff = positive_int(args, "d_model"), positive_int(args, "n_heads"), positive_int(args, "d_ff")
        dropout = probability(args, "dropout")
        if not 1 <= self.stride <= self.patch_size <= self.seq_len:
            raise ValueError("PAttn requires stride <= patch_size <= seq_len")
        if width % heads:
            raise ValueError("model.d_model must be divisible by model.n_heads")
        if args.activation not in ("relu", "gelu"):
            raise ValueError("model.activation must be relu or gelu")
        patches = (self.seq_len - self.patch_size) // self.stride + 2
        self.padding = nn.ReplicationPad1d((0, self.stride))
        self.in_layer = nn.Linear(self.patch_size, width)
        self.encoder = InvertedEncoderLayer(width, heads, ff, dropout, args.activation)
        self.norm = nn.LayerNorm(width)
        self.out_layer = nn.Linear(width * patches, self.pred_len)

    def forward(self, x, file_id=None, task_id=None, return_feature=False):
        check_history(x, self.seq_len, self.input_dim, task_id, return_feature)
        batch = x.shape[0]
        mean = x.mean(1, keepdim=True).detach()
        centered = x - mean
        std = (centered.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
        patches = self.padding((centered / std).transpose(1, 2)).unfold(-1, self.patch_size, self.stride)
        encoded = self.in_layer(patches).reshape(batch * self.input_dim, patches.shape[2], -1)
        encoded = self.norm(self.encoder(encoded))
        forecast = self.out_layer(encoded.reshape(batch, self.input_dim, -1)).transpose(1, 2)
        return forecast * std + mean
