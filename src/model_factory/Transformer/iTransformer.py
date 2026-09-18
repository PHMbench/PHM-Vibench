"""Native classification path from THUML's iTransformer (MIT).

Adapted from Time-Series-Library/models/iTransformer.py and its attention/encoder
layers. See ../NATIVE_MODELS.md for source, license and supported scope.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .._native_classification import check_signal, class_count, positive_int, probability


class InvertedEncoderLayer(nn.Module):
    """Attention between variate tokens with the upstream post-norm FFN order."""

    def __init__(self, d_model, n_heads, d_ff, dropout, activation):
        super().__init__()
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x):
        batch, tokens, dim = x.shape
        head_dim = dim // self.n_heads
        shape = (batch, tokens, self.n_heads, head_dim)
        q = self.query_projection(x).reshape(shape)
        k = self.key_projection(x).reshape(shape)
        v = self.value_projection(x).reshape(shape)
        scores = torch.einsum("blhe,bshe->bhls", q, k) / math.sqrt(head_dim)
        attention = self.attention_dropout(scores.softmax(dim=-1))
        values = torch.einsum("bhls,bshd->blhd", attention, v).reshape(batch, tokens, dim)
        x = self.norm1(x + self.dropout(self.out_projection(values)))
        y = self.dropout(self.activation(self.conv1(x.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        return self.norm2(x + y)


class Model(nn.Module):
    """Fixed-length BLC signals -> logits, optionally (logits, features).

    Uses the original classification branch: variate tokens, no forecasting
    normalization, GELU/dropout/flatten classifier. No calendar covariates.
    """

    def __init__(self, args, metadata=None):
        super().__init__()
        self.seq_len = positive_int(args, "seq_len")
        self.input_dim = positive_int(args, "input_dim")
        dim = positive_int(args, "d_model")
        heads = positive_int(args, "n_heads")
        depth = positive_int(args, "e_layers")
        ff_dim = positive_int(args, "d_ff")
        dropout = probability(args, "dropout")
        if dim % heads:
            raise ValueError("model.d_model must be divisible by model.n_heads")
        if args.activation not in ("gelu", "relu"):
            raise ValueError("model.activation must be 'gelu' or 'relu'")
        self.embedding = nn.Linear(self.seq_len, dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            InvertedEncoderLayer(dim, heads, ff_dim, dropout, args.activation)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(dim * self.input_dim, class_count(args))

    def forward(self, x, file_id=None, task_id=None, return_feature=False):
        check_signal(x, self.seq_len, self.input_dim, task_id)
        x = self.embedding_dropout(self.embedding(x.transpose(1, 2)))
        for layer in self.layers:
            x = layer(x)
        features = self.dropout(F.gelu(self.norm(x))).flatten(1)
        logits = self.projection(features)
        return (logits, features) if return_feature else logits
