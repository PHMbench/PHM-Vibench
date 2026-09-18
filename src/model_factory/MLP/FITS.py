"""FITS frequency interpolation (VEWOXIC/FITS, Apache-2.0).

Adaptation exposes the forecast tail only. The Task optimizes horizon MSE, not
FITS's optional full context-plus-horizon objective. Sample variance (correction=1)
and length-ratio compensation follow the original. Explicit irfft length also
preserves odd requested output lengths instead of silently returning one fewer.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .._native_forecasting import boolean, check_history, positive_int


class Model(nn.Module):
    def __init__(self, args, metadata=None):
        super().__init__()
        self.seq_len = positive_int(args, "seq_len")
        self.pred_len = positive_int(args, "pred_len")
        self.input_dim = positive_int(args, "input_dim")
        self.cut_freq = positive_int(args, "cut_freq")
        self.individual = boolean(args, "individual")
        if self.seq_len < 2 or self.cut_freq > self.seq_len // 2 + 1:
            raise ValueError("FITS requires seq_len >= 2 and cut_freq within history FFT bins")
        self.total_len = self.seq_len + self.pred_len
        self.length_ratio = self.total_len / self.seq_len
        self.output_freq = int(self.cut_freq * self.length_ratio)
        if self.output_freq > self.total_len // 2 + 1:
            raise ValueError("FITS interpolated cutoff exceeds output FFT bins")
        def projection():
            # Retain the upstream real initialization with zero imaginary part.
            return nn.Linear(self.cut_freq, self.output_freq).to(torch.complex64)
        self.upsampler = (nn.ModuleList([projection() for _ in range(self.input_dim)])
                          if self.individual else projection())

    def forward(self, x, file_id=None, task_id=None, return_feature=False):
        check_history(x, self.seq_len, self.input_dim, task_id, return_feature)
        mean = x.mean(1, keepdim=True)
        centered = x - mean
        std = (centered.var(1, keepdim=True, unbiased=True) + 1e-5).sqrt()
        low = torch.fft.rfft(centered / std, dim=1)[:, :self.cut_freq]
        if self.individual:
            spectrum = torch.stack([layer(low[:, :, i])
                                    for i, layer in enumerate(self.upsampler)], dim=-1)
        else:
            spectrum = self.upsampler(low.transpose(1, 2)).transpose(1, 2)
        spectrum = F.pad(spectrum, (0, 0, 0, self.total_len // 2 + 1 - self.output_freq))
        restored = torch.fft.irfft(spectrum, n=self.total_len, dim=1) * self.length_ratio
        return (restored * std + mean)[:, -self.pred_len:]
