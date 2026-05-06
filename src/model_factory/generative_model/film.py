from __future__ import annotations

import torch
import torch.nn as nn


class FiLM1D(nn.Module):
    """Feature-wise modulation for `[N, C, L]` tensors."""

    def __init__(self, condition_dim: int, channels: int) -> None:
        super().__init__()
        self.to_scale_shift = nn.Linear(condition_dim, channels * 2)

    def forward(self, x: torch.Tensor, condition_embedding: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(condition_embedding).chunk(2, dim=1)
        scale = torch.tanh(scale).unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        return x * (1.0 + scale) + shift
