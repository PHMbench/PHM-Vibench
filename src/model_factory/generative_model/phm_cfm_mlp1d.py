from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .condition_encoder import ConditionEncoder
from .film import FiLM1D


class Model(nn.Module):
    """Minimal 1D velocity model for PHM Conditional Flow Matching."""

    def __init__(self, args_model: Any, metadata: Any = None) -> None:
        super().__init__()
        channels = int(getattr(args_model, "in_channels", getattr(args_model, "channels", 2)))
        hidden_dim = int(getattr(args_model, "hidden_dim", 64))
        condition_dim = int(getattr(args_model, "condition_dim", 32))
        num_fault_classes = getattr(args_model, "num_fault_classes", None)
        num_domains = getattr(args_model, "num_domains", None)

        self.channels = channels
        self.condition_encoder = ConditionEncoder(
            metadata=metadata,
            embedding_dim=condition_dim,
            num_fault_classes=num_fault_classes,
            num_domains=num_domains,
        )
        self.in_proj = nn.Conv1d(channels, hidden_dim, kernel_size=3, padding=1)
        self.film = FiLM1D(condition_dim=condition_dim, channels=hidden_dim)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if x_t.ndim != 3:
            raise ValueError(f"x_t must be [N, C, L], got shape={tuple(x_t.shape)}")
        if x_t.shape[1] != self.channels:
            raise ValueError(f"x_t channel mismatch: expected {self.channels}, got {x_t.shape[1]}")
        cond_emb = self.condition_encoder(condition, t)
        h = self.in_proj(x_t.float())
        h = self.film(h, cond_emb)
        return self.net(h)
