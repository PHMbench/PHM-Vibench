from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class OperatorAttentionConfig:
    operators: Sequence[str] = ("I", "HT", "FFT")
    hidden_dim: int = 128
    temperature: float = 1.0


class OperatorAttention1D(nn.Module):
    """Operator-attention over a small set of 1D signal operators.

    Input:  x (B, L, C)
    Output: y (B, L, C), attention_weights (B, K)
    """

    def __init__(self, in_channels: int, cfg: Optional[OperatorAttentionConfig] = None):
        super().__init__()
        self.cfg = cfg or OperatorAttentionConfig()
        self.operators = [str(op) for op in self.cfg.operators]
        if not self.operators:
            raise ValueError("OperatorAttention1D requires a non-empty operators list.")

        self.temperature = float(self.cfg.temperature)
        self.gate = nn.Sequential(
            nn.Linear(int(in_channels), int(self.cfg.hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(self.cfg.hidden_dim), len(self.operators)),
        )
        self.last_attention_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape (B,L,C), got {tuple(x.shape)}")

        b, l, c = x.shape
        global_features = x.mean(dim=1)  # (B, C)
        logits = self.gate(global_features) / max(self.temperature, 1e-6)
        weights = F.softmax(logits, dim=1)  # (B, K)

        outputs: List[torch.Tensor] = []
        for op in self.operators:
            outputs.append(_apply_operator(op, x))

        y = torch.zeros_like(outputs[0])
        for i, out in enumerate(outputs):
            y = y + weights[:, i].view(b, 1, 1) * out

        self.last_attention_weights = weights.detach()
        return y, weights


def _apply_operator(op: str, x: torch.Tensor) -> torch.Tensor:
    op_norm = op.strip().upper()
    if op_norm in {"I", "IDENTITY"}:
        return x
    if op_norm in {"HT", "HILBERT"}:
        return _hilbert_envelope(x)
    if op_norm in {"FFT"}:
        return _fft_magnitude_resample(x)
    raise ValueError(f"Unsupported operator for OperatorAttention1D: {op}")


def _hilbert_envelope(x: torch.Tensor) -> torch.Tensor:
    # Based on Signal_processing.HilbertTransform but device-agnostic.
    x_bcl = x.permute(0, 2, 1)  # (B, C, L)
    n = x_bcl.shape[-1]
    xf = torch.fft.fft(x_bcl, dim=2)
    if n % 2 == 0:
        xf[..., 1 : n // 2] *= 2
        xf[..., n // 2 + 1 :] = 0
    else:
        xf[..., 1 : (n + 1) // 2] *= 2
        xf[..., (n + 1) // 2 :] = 0
    env = torch.fft.ifft(xf, dim=2).abs()
    return env.permute(0, 2, 1)  # (B, L, C)


def _fft_magnitude_resample(x: torch.Tensor) -> torch.Tensor:
    b, l, c = x.shape
    fft = torch.fft.rfft(x, dim=1, norm="ortho")  # (B, F, C) complex
    mag = fft.abs()  # (B, F, C)
    mag_bcf = mag.permute(0, 2, 1)  # (B, C, F)
    mag_up = F.interpolate(mag_bcf, size=l, mode="linear", align_corners=False)  # (B, C, L)
    return mag_up.permute(0, 2, 1).contiguous()  # (B, L, C)
