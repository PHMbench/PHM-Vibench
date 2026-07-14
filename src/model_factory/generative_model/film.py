from __future__ import annotations

import torch
import torch.nn as nn


class FiLM1D(nn.Module):
    """Apply stateless feature-wise affine conditioning to ``[N,C,L]`` data."""

    def __init__(self, condition_dim: int, channels: int) -> None:
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.channels = int(channels)
        if self.condition_dim <= 0 or self.channels <= 0:
            raise ValueError(
                "condition_dim and channels must both be positive, got "
                f"{self.condition_dim} and {self.channels}"
            )
        self.affine = nn.Linear(self.condition_dim, self.channels * 2)

    def forward(
        self,
        features: torch.Tensor,
        condition_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                f"features must be [N,C,L], got {tuple(features.shape)}"
            )
        if condition_embedding.ndim != 2:
            raise ValueError(
                "condition_embedding must be [N,D], got "
                f"{tuple(condition_embedding.shape)}"
            )
        if features.shape[0] != condition_embedding.shape[0]:
            raise ValueError(
                "FiLM batch mismatch: "
                f"features={features.shape[0]}, condition={condition_embedding.shape[0]}"
            )
        if features.shape[1] != self.channels:
            raise ValueError(
                f"FiLM channel mismatch: expected {self.channels}, "
                f"got {features.shape[1]}"
            )
        if condition_embedding.shape[1] != self.condition_dim:
            raise ValueError(
                f"FiLM condition width mismatch: expected {self.condition_dim}, "
                f"got {condition_embedding.shape[1]}"
            )

        scale, shift = self.affine(condition_embedding).chunk(2, dim=1)
        return features * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
