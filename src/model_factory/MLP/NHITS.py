"""NHITS forecast network from Nixtla/neuralforecast (Apache-2.0).

Native no-exogenous, point-output path. Each channel is an independent series
with shared network parameters. Task owns the MSE objective; no Nixtla runner,
scaler, random-seed mutation, horizon targets or masks enter model construction.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .._native_forecasting import check_history, positive_int, probability


def _positive_list(value, field):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"model.{field} must be a nonempty list of positive integers")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in value):
        raise ValueError(f"model.{field} must contain only positive integers")
    return tuple(value)


class NHITSBlock(nn.Module):
    def __init__(self, length, horizon, units, pool_size, downsample, pooling, interpolation, dropout, activation):
        super().__init__()
        self.length, self.horizon = length, horizon
        self.interpolation = interpolation
        self.knots = max(horizon // downsample, 1)
        self.pool = getattr(nn, pooling)(pool_size, stride=pool_size, ceil_mode=True)
        layers = [nn.Linear(math.ceil(length / pool_size), units[0][0])]
        act = getattr(nn, activation)()
        for before, after in units:
            layers.extend((nn.Linear(before, after), act))
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(units[-1][1], length + self.knots))
        self.layers = nn.Sequential(*layers)

    def forward(self, residual):
        theta = self.layers(self.pool(residual.unsqueeze(1)).squeeze(1))
        backcast = theta[:, :self.length]
        knots = theta[:, self.length:].unsqueeze(1)
        forecast = F.interpolate(knots, size=self.horizon, mode=self.interpolation)
        return backcast, forecast.squeeze(1)


class Model(nn.Module):
    def __init__(self, args, metadata=None):
        super().__init__()
        self.seq_len, self.pred_len = positive_int(args, "seq_len"), positive_int(args, "pred_len")
        self.input_dim = positive_int(args, "input_dim")
        n_blocks = _positive_list(args.n_blocks, "n_blocks")
        pools = _positive_list(args.n_pool_kernel_size, "n_pool_kernel_size")
        frequencies = _positive_list(args.n_freq_downsample, "n_freq_downsample")
        if not len(n_blocks) == len(pools) == len(frequencies):
            raise ValueError("NHITS n_blocks, pooling and downsample lists must have identical stack counts")
        if not isinstance(args.mlp_units, (list, tuple)) or not args.mlp_units:
            raise ValueError("model.mlp_units requires nonempty [input, output] pairs")
        units = tuple(_positive_list(pair, "mlp_units") for pair in args.mlp_units)
        if any(len(pair) != 2 for pair in units) or any(a[1] != b[0] for a, b in zip(units, units[1:])):
            raise ValueError("model.mlp_units pairs must be width-compatible")
        if args.pooling_mode not in ("MaxPool1d", "AvgPool1d"):
            raise ValueError("model.pooling_mode must be MaxPool1d or AvgPool1d")
        if args.interpolation_mode not in ("linear", "nearest"):
            raise ValueError("This NHITS port supports explicit linear or nearest interpolation")
        if args.activation not in ("ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"):
            raise ValueError("Unsupported NHITS activation")
        dropout = probability(args, "dropout_prob_theta")
        self.blocks = nn.ModuleList([
            NHITSBlock(self.seq_len, self.pred_len, units, pool, down, args.pooling_mode,
                       args.interpolation_mode, dropout, args.activation)
            for count, pool, down in zip(n_blocks, pools, frequencies) for _ in range(count)
        ])

    def forward(self, x, file_id=None, task_id=None, return_feature=False):
        check_history(x, self.seq_len, self.input_dim, task_id, return_feature)
        batch = x.shape[0]
        series = x.transpose(1, 2).reshape(batch * self.input_dim, self.seq_len)
        residual = series.flip(-1)
        forecast = series[:, -1:].expand(-1, self.pred_len)
        for block in self.blocks:
            backcast, increment = block(residual)
            residual = residual - backcast
            forecast = forecast + increment
        return forecast.reshape(batch, self.input_dim, self.pred_len).transpose(1, 2)
