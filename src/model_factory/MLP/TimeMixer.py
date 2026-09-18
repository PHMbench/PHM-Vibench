"""TimeMixer forecasting from THUML Time-Series-Library (MIT).

Moving-average decomposition and average/max scale pooling are explicit. The
port has local history normalization statistics, no calendar covariates and no
forward-created convolution. MIT notice: ../_native_classification.py.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .._native_forecasting import boolean, check_history, positive_int, probability


def decompose(x, kernel):
    padded = F.pad(x.transpose(1, 2), ((kernel - 1) // 2, (kernel - 1) // 2), mode="replicate")
    trend = F.avg_pool1d(padded, kernel, stride=1).transpose(1, 2)
    return x - trend, trend


class HistoryNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight, self.bias = nn.Parameter(torch.ones(channels)), nn.Parameter(torch.zeros(channels))

    def normalize(self, x):
        mean = x.mean(1, keepdim=True).detach()
        std = (x.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
        return (x - mean) / std * self.weight + self.bias, mean, std

    def denormalize(self, y, mean, std):
        return (y - self.bias) / (self.weight + 1e-10) * std + mean


class PastMixing(nn.Module):
    def __init__(self, lengths, width, ff_width, kernel, independent):
        super().__init__()
        self.kernel, self.independent = kernel, independent
        if independent:
            self.out_cross = nn.Sequential(nn.Linear(width, ff_width), nn.GELU(), nn.Linear(ff_width, width))
        else:
            self.cross = nn.Sequential(nn.Linear(width, ff_width), nn.GELU(), nn.Linear(ff_width, width))
        self.season_down = nn.ModuleList([
            nn.Sequential(nn.Linear(high, low), nn.GELU(), nn.Linear(low, low))
            for high, low in zip(lengths, lengths[1:])
        ])
        self.trend_up = nn.ModuleList([
            nn.Sequential(nn.Linear(low, high), nn.GELU(), nn.Linear(high, high))
            for high, low in reversed(list(zip(lengths, lengths[1:])))
        ])

    def forward(self, scales):
        seasons, trends = [], []
        for x in scales:
            season, trend = decompose(x, self.kernel)
            if not self.independent:
                season, trend = self.cross(season), self.cross(trend)
            seasons.append(season.transpose(1, 2))
            trends.append(trend.transpose(1, 2))
        season_mixed = [seasons[0]]
        for index, down in enumerate(self.season_down):
            season_mixed.append(seasons[index + 1] + down(season_mixed[-1]))
        trend_mixed = [trends[-1]]
        for index, up in enumerate(self.trend_up):
            trend_mixed.append(trends[-index - 2] + up(trend_mixed[-1]))
        trend_mixed.reverse()
        outputs = []
        for original, season, trend in zip(scales, season_mixed, trend_mixed):
            mixed = (season + trend).transpose(1, 2)
            outputs.append(original + self.out_cross(mixed) if self.independent else mixed)
        return outputs


class Model(nn.Module):
    def __init__(self, args, metadata=None):
        super().__init__()
        self.seq_len, self.pred_len = positive_int(args, "seq_len"), positive_int(args, "pred_len")
        self.input_dim = positive_int(args, "input_dim")
        self.window = positive_int(args, "down_sampling_window")
        scales = positive_int(args, "down_sampling_layers")
        width, ff_width, depth = positive_int(args, "d_model"), positive_int(args, "d_ff"), positive_int(args, "e_layers")
        self.kernel = positive_int(args, "moving_avg")
        self.independent = boolean(args, "channel_independence")
        self.use_norm = boolean(args, "use_norm")
        dropout = probability(args, "dropout")
        if args.decomp_method != "moving_avg":
            raise ValueError("This TimeMixer port supports explicit moving_avg decomposition only")
        if args.down_sampling_method not in ("avg", "max"):
            raise ValueError("TimeMixer supports explicit avg or max pooling, not a forward-created convolution")
        if self.window < 2 or self.seq_len % self.window ** scales:
            raise ValueError("TimeMixer scale pooling must divide the configured history length")
        if self.kernel % 2 == 0:
            raise ValueError("TimeMixer moving_avg must be odd")
        self.pool = nn.AvgPool1d(self.window) if args.down_sampling_method == "avg" else nn.MaxPool1d(self.window)
        self.lengths = [self.seq_len // self.window ** i for i in range(scales + 1)]
        self.norms = nn.ModuleList([HistoryNorm(self.input_dim) for _ in self.lengths]) if self.use_norm else nn.ModuleList()
        self.embedding = nn.Conv1d(1 if self.independent else self.input_dim, width, 3, padding=1, padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.embedding.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.dropout = nn.Dropout(dropout)
        self.pdm = nn.ModuleList([PastMixing(self.lengths, width, ff_width, self.kernel, self.independent) for _ in range(depth)])
        self.predict = nn.ModuleList([nn.Linear(length, self.pred_len) for length in self.lengths])
        self.projection = nn.Linear(width, 1 if self.independent else self.input_dim)
        if not self.independent:
            self.out_res = nn.ModuleList([nn.Linear(length, length) for length in self.lengths])
            self.regression = nn.ModuleList([nn.Linear(length, self.pred_len) for length in self.lengths])

    def forward(self, x, file_id=None, task_id=None, return_feature=False):
        check_history(x, self.seq_len, self.input_dim, task_id, return_feature)
        batch = x.shape[0]
        raw_scales = [x]
        for _ in self.lengths[1:]:
            raw_scales.append(self.pool(raw_scales[-1].transpose(1, 2)).transpose(1, 2))
        values, trends, statistics = [], [], []
        for index, raw in enumerate(raw_scales):
            if self.use_norm:
                raw, mean, std = self.norms[index].normalize(raw)
                statistics.append((mean, std))
            if self.independent:
                raw = raw.transpose(1, 2).reshape(batch * self.input_dim, raw.shape[1], 1)
            else:
                raw, trend = decompose(raw, self.kernel)
                trends.append(trend)
            values.append(self.dropout(self.embedding(raw.transpose(1, 2)).transpose(1, 2)))
        for pdm in self.pdm:
            values = pdm(values)
        forecasts = []
        for index, value in enumerate(values):
            predicted = self.projection(self.predict[index](value.transpose(1, 2)).transpose(1, 2))
            if self.independent:
                predicted = predicted.reshape(batch, self.input_dim, self.pred_len).transpose(1, 2)
            else:
                residual = self.regression[index](self.out_res[index](trends[index].transpose(1, 2))).transpose(1, 2)
                predicted = predicted + residual
            forecasts.append(predicted)
        result = torch.stack(forecasts, dim=-1).sum(-1)
        return self.norms[0].denormalize(result, *statistics[0]) if self.use_norm else result
