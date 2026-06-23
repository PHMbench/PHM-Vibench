from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .legacy_collection.Physics_informed_PDN import PhysicsInformedPDN as _PhysicsInformedPDN


class _CompatibilityPDN(nn.Module):
    """Fallback when legacy PDN initialization fails."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(hidden_dim // 2, 8), num_classes),
        )

    def forward(self, x: torch.Tensor, return_uncertainty: bool = False):
        logits = self.net(x)
        uncertainty = torch.zeros(x.shape[0], device=x.device)
        explanations: Dict[str, Any] = {}
        return logits, uncertainty, explanations


class Model(nn.Module):
    """Factory entry for legacy Physics-informed PDN."""

    def __init__(self, args: Any, metadata: Any = None):
        super().__init__()
        self.args = args if args is not None else SimpleNamespace()
        self.input_dim = int(getattr(self.args, "input_dim", getattr(self.args, "in_dim", 4096)))
        self.num_classes = int(getattr(self.args, "num_classes", getattr(self.args, "output_dim", 2)))
        hidden_dim = int(getattr(self.args, "hidden_dim", 128))
        num_samples = int(getattr(self.args, "num_samples", 10))
        physics_params: Dict[str, Any] = getattr(self.args, "physics_params", None) or {
            "resonance_freq": 100.0,
            "damping_ratio": 0.1,
            "freq_range": [0, 1000],
        }
        try:
            self.network = _PhysicsInformedPDN(
                input_dim=self.input_dim,
                num_classes=self.num_classes,
                hidden_dim=hidden_dim,
                num_samples=num_samples,
                physics_params=physics_params,
            )
        except TypeError:
            self.network = _CompatibilityPDN(
                input_dim=self.input_dim,
                hidden_dim=hidden_dim,
                num_classes=self.num_classes,
            )

    def _to_flat(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        elif x.ndim != 2:
            raise ValueError(f"Physics_informed_PDN expects 1D/2D/3D input, got shape={tuple(x.shape)}")

        if x.shape[1] > self.input_dim:
            x = x[:, : self.input_dim]
        elif x.shape[1] < self.input_dim:
            x = F.pad(x, (0, self.input_dim - x.shape[1]))
        return x.float()

    def forward(self, x: torch.Tensor, data_id=None, task_id=None) -> torch.Tensor:
        logits, _, _ = self.network(self._to_flat(x), return_uncertainty=False)
        return logits
