from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn


FusionType = Literal["concat", "sum", "gated"]


@dataclass(frozen=True)
class FusionConfig:
    fusion_type: FusionType = "concat"


class ConcatProjectFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([a, b], dim=1))


class SumFusion(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


class GatedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([a, b], dim=1))
        return gate * a + (1.0 - gate) * b


def build_fusion(dim: int, cfg: Optional[FusionConfig] = None) -> nn.Module:
    cfg = cfg or FusionConfig()
    if cfg.fusion_type == "concat":
        return ConcatProjectFusion(dim)
    if cfg.fusion_type == "sum":
        return SumFusion()
    if cfg.fusion_type == "gated":
        return GatedFusion(dim)
    raise ValueError(f"Unknown fusion_type: {cfg.fusion_type}")
