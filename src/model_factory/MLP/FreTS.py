"""FreTS forecast port, from aikunyi/FreTS/models/FreTS.py (Apache-2.0).

The upstream einsum('bijd,dd->bijd') uses only diagonal matrix entries; this
port retains that operation rather than silently replacing it with dense mixing.
`channel_mixing=True` matches the upstream `channel_independence == '1'` branch.
"""
from numbers import Real
import math
import torch
from torch import nn
from torch.nn import functional as F
from .._native_forecasting import boolean, check_history, positive_int


class Model(nn.Module):
    def __init__(self, args, metadata=None):
        super().__init__()
        self.seq_len, self.pred_len = positive_int(args, "seq_len"), positive_int(args, "pred_len")
        self.input_dim = positive_int(args, "input_dim")
        self.embed_size, hidden = positive_int(args, "embed_size"), positive_int(args, "hidden_size")
        self.channel_mixing = boolean(args, "channel_mixing")
        threshold = args.sparsity_threshold
        if isinstance(threshold, bool) or not isinstance(threshold, Real) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("model.sparsity_threshold must be a finite nonnegative number")
        self.sparsity_threshold = float(threshold)
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        for suffix in ("1", "2"):
            for part in ("r", "i"):
                setattr(self, part + suffix, nn.Parameter(0.02 * torch.randn(self.embed_size, self.embed_size)))
                setattr(self, part + "b" + suffix, nn.Parameter(0.02 * torch.randn(self.embed_size)))
        self.fc = nn.Sequential(nn.Linear(self.seq_len * self.embed_size, hidden), nn.LeakyReLU(), nn.Linear(hidden, self.pred_len))

    def frequency_mlp(self, x, suffix):
        r, i = getattr(self, "r" + suffix), getattr(self, "i" + suffix)
        rb, ib = getattr(self, "rb" + suffix), getattr(self, "ib" + suffix)
        real = F.relu(x.real * r.diagonal() - x.imag * i.diagonal() + rb)
        imaginary = F.relu(x.imag * r.diagonal() + x.real * i.diagonal() + ib)
        parts = F.softshrink(torch.stack((real, imaginary), dim=-1), lambd=self.sparsity_threshold)
        return torch.view_as_complex(parts)

    def forward(self, x, file_id=None, task_id=None, return_feature=False):
        check_history(x, self.seq_len, self.input_dim, task_id, return_feature)
        embedded = x.transpose(1, 2).unsqueeze(-1) * self.embeddings
        encoded = embedded
        if self.channel_mixing:
            spectrum = torch.fft.rfft(encoded.transpose(1, 2), dim=2, norm="ortho")
            encoded = torch.fft.irfft(self.frequency_mlp(spectrum, "1"), n=self.input_dim, dim=2, norm="ortho").transpose(1, 2)
        spectrum = torch.fft.rfft(encoded, dim=2, norm="ortho")
        encoded = torch.fft.irfft(self.frequency_mlp(spectrum, "2"), n=self.seq_len, dim=2, norm="ortho")
        return self.fc((encoded + embedded).flatten(2)).transpose(1, 2)
