from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .condition_encoder import ConditionEncoder
from .film import FiLM1D


def _match_length(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[-1] == length:
        return x
    if x.shape[-1] > length:
        return x[..., :length]
    return F.pad(x, (0, length - x.shape[-1]))


class Model(nn.Module):
    """Small conditional UNet1D backbone for PHM generative smoke runs."""

    stateless = True

    def __init__(self, args_model: Any, metadata: Any = None) -> None:
        super().__init__()
        channels = int(getattr(args_model, "in_channels", getattr(args_model, "channels", 2)))
        hidden_dim = int(getattr(args_model, "hidden_dim", 64))
        condition_dim = int(getattr(args_model, "condition_dim", 32))
        self.channels = channels
        self.condition_encoder = ConditionEncoder(
            metadata=metadata,
            embedding_dim=condition_dim,
            num_fault_classes=getattr(args_model, "num_fault_classes", None),
            num_domains=getattr(args_model, "num_domains", None),
        )
        self.enc1 = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.down = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1),
        )
        self.film = FiLM1D(condition_dim=condition_dim, channels=hidden_dim * 2)
        self.up = nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.dec = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: dict[str, torch.Tensor]) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be [N, C, L], got shape={tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"x channel mismatch: expected {self.channels}, got {x.shape[1]}")
        cond = self.condition_encoder(condition, t)
        skip = self.enc1(x.float())
        h = self.down(skip)
        h = self.film(h, cond)
        h = _match_length(self.up(h), skip.shape[-1])
        return self.dec(torch.cat([h, skip], dim=1))
