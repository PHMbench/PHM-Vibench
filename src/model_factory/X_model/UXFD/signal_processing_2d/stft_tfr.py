from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class STFTConfig:
    n_fft: int = 256
    hop_length: int = 128
    win_length: Optional[int] = None
    center: bool = True
    normalized: bool = False
    onesided: bool = True
    magnitude: bool = True


class STFTTimeFrequency(nn.Module):
    """Compute a differentiable STFT time-frequency representation.

    Input
    - `x`: (B, L, C)

    Output
    - `(B, T, F, C)` magnitude (default) or complex STFT converted to real/imag stacking.
    """

    def __init__(self, cfg: Optional[STFTConfig] = None):
        super().__init__()
        self.cfg = cfg or STFTConfig()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B,L,C), got {tuple(x.shape)}")
        b, l, c = x.shape
        x_bcL = x.permute(0, 2, 1).contiguous().view(b * c, l)

        win_length = self.cfg.win_length or self.cfg.n_fft
        window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)

        stft = torch.stft(
            x_bcL,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=win_length,
            window=window,
            center=self.cfg.center,
            normalized=self.cfg.normalized,
            onesided=self.cfg.onesided,
            return_complex=True,
        )  # (B*C, F, T)

        stft = stft.permute(0, 2, 1)  # (B*C, T, F)
        if self.cfg.magnitude:
            stft_out = stft.abs()
        else:
            stft_out = torch.stack([stft.real, stft.imag], dim=-1)  # (B*C, T, F, 2)

        if stft_out.ndim == 3:
            stft_out = stft_out.view(b, c, stft_out.shape[1], stft_out.shape[2]).permute(0, 2, 3, 1)
        else:
            # (B, C, T, F, 2) -> (B, T, F, C, 2)
            stft_out = stft_out.view(b, c, stft_out.shape[1], stft_out.shape[2], stft_out.shape[3]).permute(
                0, 2, 3, 1, 4
            )
        return stft_out

