"""TSMixer forecast path from THUML Time-Series-Library (MIT).

This identifies the TSL implementation: residual temporal/channel MLPs without
extra normalization. It does not claim equivalence to Google's full TSMixer
training pipeline. The installed MIT notice is in _native_classification.py.
"""
from torch import nn
from .._native_forecasting import check_history, positive_int, probability


class ResBlock(nn.Module):
    def __init__(self, length, channels, hidden, dropout):
        super().__init__()
        self.temporal = nn.Sequential(nn.Linear(length, hidden), nn.ReLU(), nn.Linear(hidden, length), nn.Dropout(dropout))
        self.channel = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        return x + self.channel(x)


class Model(nn.Module):
    def __init__(self, args, metadata=None):
        super().__init__()
        self.seq_len, self.pred_len = positive_int(args, "seq_len"), positive_int(args, "pred_len")
        self.input_dim = positive_int(args, "input_dim")
        hidden, depth = positive_int(args, "d_model"), positive_int(args, "e_layers")
        dropout = probability(args, "dropout")
        self.blocks = nn.ModuleList([ResBlock(self.seq_len, self.input_dim, hidden, dropout) for _ in range(depth)])
        self.projection = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, file_id=None, task_id=None, return_feature=False):
        check_history(x, self.seq_len, self.input_dim, task_id, return_feature)
        for block in self.blocks:
            x = block(x)
        return self.projection(x.transpose(1, 2)).transpose(1, 2)
