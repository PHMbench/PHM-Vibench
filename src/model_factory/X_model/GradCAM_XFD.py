from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from .legacy_collection.GradCAM_XFD import ExplainableCNN as _ExplainableCNN


class Model(nn.Module):
    """Factory entry for legacy 1D Grad-CAM CNN."""

    def __init__(self, args: Any, metadata: Any = None):
        super().__init__()
        self.args = args if args is not None else SimpleNamespace()
        self.input_channels = int(getattr(self.args, "input_channels", getattr(self.args, "in_channels", 1)))
        self.num_classes = int(getattr(self.args, "num_classes", getattr(self.args, "output_dim", 2)))
        seq_length = int(getattr(self.args, "seq_length", getattr(self.args, "in_dim", 4096)))
        dropout = float(getattr(self.args, "dropout", 0.2))
        self.network = _ExplainableCNN(
            input_channels=self.input_channels,
            num_classes=self.num_classes,
            seq_length=seq_length,
            dropout=dropout,
        )

    def _to_bcl(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B, 1, L)
        elif x.ndim == 3:
            if x.shape[1] == self.input_channels:
                pass
            elif x.shape[2] == self.input_channels:
                x = x.permute(0, 2, 1).contiguous()
            else:
                x = x.permute(0, 2, 1).contiguous()  # assume (B, L, C)
        else:
            raise ValueError(f"GradCAM_XFD expects 2D/3D input, got shape={tuple(x.shape)}")

        channels = x.shape[1]
        if channels < self.input_channels:
            repeats = (self.input_channels + channels - 1) // channels
            x = x.repeat(1, repeats, 1)[:, : self.input_channels, :]
        elif channels > self.input_channels:
            x = x[:, : self.input_channels, :]
        return x.float()

    def forward(self, x: torch.Tensor, data_id=None, task_id=None) -> torch.Tensor:
        return self.network(self._to_bcl(x))

