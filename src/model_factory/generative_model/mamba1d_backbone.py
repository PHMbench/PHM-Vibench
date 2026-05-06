from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .condition_encoder import ConditionEncoder
from .film import FiLM1D


class Model(nn.Module):
    """Stateless Mamba/SSM-style 1D adapter placeholder for future generators.

    This implementation avoids mandatory Mamba CUDA dependencies. It preserves
    the future backbone contract for velocity/epsilon/score heads: input
    `[N, C, L]`, time `t`, conditions `fault_label/domain_id`, output
    `[N, C, L]`, and no sampler-managed hidden cache.
    """

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
        self.in_proj = nn.Conv1d(channels, hidden_dim, kernel_size=1)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=7,
            padding=3,
            groups=hidden_dim,
        )
        self.film = FiLM1D(condition_dim=condition_dim, channels=hidden_dim)
        self.out_proj = nn.Sequential(nn.SiLU(), nn.Conv1d(hidden_dim, channels, kernel_size=1))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be [N, C, L], got shape={tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"x channel mismatch: expected {self.channels}, got {x.shape[1]}")
        cond = self.condition_encoder(condition, t)
        h = self.in_proj(x.float())
        h = h + self.depthwise(h)
        h = self.film(h, cond)
        return self.out_proj(h)

