from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Compatibility implementation for legacy F_EQL entry.

    The imported legacy `F_EQL.py` is mostly commented research draft code.
    This module provides a minimal runnable classifier with the same factory
    entry semantics so `model.name='F_EQL'` remains usable.
    """

    def __init__(self, args: Any, metadata: Any = None):
        super().__init__()
        self.args = args if args is not None else SimpleNamespace()
        self.in_channels = int(getattr(self.args, "in_channels", 1))
        self.num_classes = int(getattr(self.args, "num_classes", getattr(self.args, "output_dim", 2)))
        hidden = int(getattr(self.args, "hidden_dim", 64))
        self.features = nn.Sequential(
            nn.Conv1d(self.in_channels, hidden, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(hidden, self.num_classes)

    def _to_bcl(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 3:
            if x.shape[1] == self.in_channels:
                pass
            elif x.shape[2] == self.in_channels:
                x = x.permute(0, 2, 1).contiguous()
            else:
                x = x.permute(0, 2, 1).contiguous()
        else:
            raise ValueError(f"F_EQL expects 2D/3D input, got shape={tuple(x.shape)}")

        channels = x.shape[1]
        if channels < self.in_channels:
            repeats = (self.in_channels + channels - 1) // channels
            x = x.repeat(1, repeats, 1)[:, : self.in_channels, :]
        elif channels > self.in_channels:
            x = x[:, : self.in_channels, :]
        return x.float()

    def forward(self, x: torch.Tensor, data_id=None, task_id=None) -> torch.Tensor:
        x = self._to_bcl(x)
        x = self.features(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.head(x)

