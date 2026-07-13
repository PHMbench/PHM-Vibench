from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LogicConfig:
    """Minimal neuro-symbolic reasoning config.

    This is a best-effort placeholder slot for the UXFD merge. It is designed to:
    - be device-agnostic (no import-time .cuda())
    - never hard-crash when enabled
    - provide a stable knob surface under `model.uxfd.logic.*`
    """

    hidden_dim: int = 128
    logit_scale: float = 1.0


class LogicReasoner(nn.Module):
    """A minimal "logic residual" head over learned features.

    Input:  features (B, D)
    Output: logits_residual (B, num_classes)
    """

    def __init__(self, dim_in: int, num_classes: int, cfg: Optional[LogicConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or LogicConfig()
        self.net = nn.Sequential(
            nn.Linear(int(dim_in), int(self.cfg.hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(self.cfg.hidden_dim), int(num_classes)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            features = features.view(features.size(0), -1)
        return self.net(features)
