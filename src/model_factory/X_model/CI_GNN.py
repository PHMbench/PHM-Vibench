from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

try:
    from .legacy_collection.CI_GNN import ExplainableGNN as _ExplainableGNN
except ModuleNotFoundError as exc:  # pragma: no cover - dependency gate
    if exc.name and exc.name.startswith("torch_geometric"):
        raise ModuleNotFoundError(
            "CI_GNN requires `torch_geometric`. Install it in your environment "
            "(e.g. `pip install torch_geometric`) before using model.name='CI_GNN'."
        ) from exc
    raise


class Model(nn.Module):
    """Factory entry for legacy CI-GNN with `(B, L, C)` input support."""

    def __init__(self, args: Any, metadata: Any = None):
        super().__init__()
        self.args = args if args is not None else SimpleNamespace()
        self.num_sensors = int(getattr(self.args, "num_sensors", getattr(self.args, "in_channels", 8)))
        self.num_classes = int(getattr(self.args, "num_classes", getattr(self.args, "output_dim", 2)))
        hidden_dim = int(getattr(self.args, "hidden_dim", 128))
        num_layers = int(getattr(self.args, "num_layers", 3))
        dropout = float(getattr(self.args, "dropout", 0.2))
        self.network = _ExplainableGNN(
            num_sensors=self.num_sensors,
            num_classes=self.num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        # Legacy implementation hardcodes sensor_embedding input dim=1.
        # Align it with the configured sensor count for stable forward passes.
        self.network.causality_layer.sensor_embedding = nn.Linear(self.num_sensors, hidden_dim)
        # Legacy implementation hardcodes temporal_embedding input dim=10.
        self.network.causality_layer.temporal_embedding = nn.Linear(self.num_sensors, hidden_dim)

    def _to_sensor_first(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B, 1, L)
        elif x.ndim == 3:
            if x.shape[1] == self.num_sensors:
                pass
            elif x.shape[2] == self.num_sensors:
                x = x.permute(0, 2, 1).contiguous()
            else:
                x = x.permute(0, 2, 1).contiguous()  # assume (B, L, C)
        else:
            raise ValueError(f"CI_GNN expects 2D/3D input, got shape={tuple(x.shape)}")

        channels = x.shape[1]
        if channels < self.num_sensors:
            repeats = (self.num_sensors + channels - 1) // channels
            x = x.repeat(1, repeats, 1)[:, : self.num_sensors, :]
        elif channels > self.num_sensors:
            x = x[:, : self.num_sensors, :]
        return x.float()

    def forward(self, x: torch.Tensor, data_id=None, task_id=None) -> torch.Tensor:
        sensor_first = self._to_sensor_first(x)
        logits, _ = self.network(sensor_first)
        return logits
