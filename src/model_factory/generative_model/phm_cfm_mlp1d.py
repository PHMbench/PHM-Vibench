from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .condition_encoder import ConditionEncoder
from .film import FiLM1D


class Model(nn.Module):
    """Minimal stateless 1D velocity model for Conditional Flow Matching."""

    def __init__(self, args_model: Any, metadata: Any = None) -> None:
        super().__init__()
        self.channels = int(getattr(args_model, "in_channels", 2))
        hidden_dim = int(getattr(args_model, "hidden_dim", 64))
        condition_dim = int(getattr(args_model, "condition_dim", 32))

        if self.channels <= 0 or hidden_dim <= 0 or condition_dim <= 0:
            raise ValueError(
                "in_channels, hidden_dim, and condition_dim must be positive"
            )

        self.condition_encoder = ConditionEncoder(
            metadata=metadata,
            embedding_dim=condition_dim,
            num_fault_classes=getattr(args_model, "num_fault_classes", None),
            num_domains=getattr(args_model, "num_domains", None),
        )
        self.input_projection = nn.Conv1d(
            self.channels,
            hidden_dim,
            kernel_size=3,
            padding=1,
        )
        self.conditioning = FiLM1D(
            condition_dim=condition_dim,
            channels=hidden_dim,
        )
        self.velocity_head = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, self.channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if x_t.ndim != 3:
            raise ValueError(f"x_t must be [N,C,L], got {tuple(x_t.shape)}")
        if x_t.shape[1] != self.channels:
            raise ValueError(
                f"x_t channel mismatch: expected {self.channels}, got {x_t.shape[1]}"
            )
        if not torch.is_floating_point(x_t):
            raise ValueError(f"x_t must be floating point, got {x_t.dtype}")
        if not torch.isfinite(x_t).all():
            raise ValueError("x_t contains NaN/Inf")

        parameter = self.input_projection.weight
        if x_t.device != parameter.device:
            raise ValueError(
                f"x_t device mismatch: input={x_t.device}, model={parameter.device}"
            )
        if x_t.dtype != parameter.dtype:
            raise ValueError(
                f"x_t dtype mismatch: input={x_t.dtype}, model={parameter.dtype}"
            )

        t = torch.as_tensor(t, device=x_t.device, dtype=x_t.dtype).reshape(-1)
        if t.numel() != x_t.shape[0]:
            raise ValueError(
                f"t batch mismatch: expected {x_t.shape[0]}, got {t.numel()}"
            )
        if not torch.isfinite(t).all():
            raise ValueError("t contains NaN/Inf")

        condition_embedding = self.condition_encoder(condition, t)
        hidden = self.input_projection(x_t)
        hidden = self.conditioning(hidden, condition_embedding)
        velocity = self.velocity_head(hidden)

        if velocity.shape != x_t.shape:
            raise ValueError(
                f"velocity shape mismatch: {tuple(velocity.shape)} vs {tuple(x_t.shape)}"
            )
        if velocity.dtype != x_t.dtype:
            raise ValueError(
                f"velocity dtype mismatch: {velocity.dtype} vs {x_t.dtype}"
            )
        if velocity.device != x_t.device:
            raise ValueError(
                f"velocity device mismatch: {velocity.device} vs {x_t.device}"
            )
        if not torch.isfinite(velocity).all():
            raise ValueError("predicted velocity contains NaN/Inf")
        return velocity
