from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FuzzyConfig:
    num_fuzzy_features: int = 32
    num_membership_functions: int = 3  # low/medium/high (Gaussian)
    num_rules: int = 10
    logit_scale: float = 1.0


class FuzzyReasoner(nn.Module):
    """A small fuzzy-rule head over a feature vector.

    Input:  (B, D)
    Output: (B, num_classes) logits (to be added to a base classifier).
    """

    def __init__(self, dim_in: int, num_classes: int, cfg: Optional[FuzzyConfig] = None):
        super().__init__()
        self.cfg = cfg or FuzzyConfig()

        self.feature_reducer = nn.Sequential(
            nn.Linear(dim_in, int(self.cfg.num_fuzzy_features)),
            nn.ReLU(inplace=True),
        )

        f = int(self.cfg.num_fuzzy_features)
        m = int(self.cfg.num_membership_functions)
        r = int(self.cfg.num_rules)
        k = int(num_classes)

        self.centers = nn.Parameter(torch.randn(f, m) * 0.5)
        self.widths = nn.Parameter(torch.ones(f, m) * 0.3)
        self.rule_weights = nn.Parameter(torch.ones(r, f) / max(1, f))
        self.rule_outputs = nn.Parameter(torch.randn(r, k) * 0.1)

        self.classifier = nn.Sequential(
            nn.Linear(k, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, k),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        reduced = self.feature_reducer(features)  # (B, F)
        membership = self._compute_membership(reduced)  # (B, F, M)
        fuzzy_logits = self._apply_rules(membership)  # (B, K)
        return self.classifier(fuzzy_logits)

    def _compute_membership(self, x: torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(-1)  # (B, F, 1)
        centers = self.centers.unsqueeze(0)  # (1, F, M)
        widths = torch.abs(self.widths).unsqueeze(0).clamp_min(1e-6)  # (1, F, M)
        return torch.exp(-((x_expanded - centers) ** 2) / (2.0 * widths**2))

    def _apply_rules(self, membership: torch.Tensor) -> torch.Tensor:
        aggregated = membership.mean(dim=2)  # (B, F)
        activation = torch.sum(
            aggregated.unsqueeze(1) * self.rule_weights.unsqueeze(0),
            dim=2,
        )  # (B, R)
        activation = F.softmax(activation, dim=-1)
        outputs = activation.unsqueeze(-1) * self.rule_outputs.unsqueeze(0)  # (B, R, K)
        return outputs.sum(dim=1)

