from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from .legacy_collection.EELM import Dong_ELM


class Model(nn.Module):
    """Factory entry for legacy Dong_ELM."""

    def __init__(self, args: Any, metadata: Any = None):
        super().__init__()
        self.args = args if args is not None else SimpleNamespace()
        self.num_classes = int(getattr(self.args, "num_class", getattr(self.args, "num_classes", 2)))
        self.network = Dong_ELM(num_class=self.num_classes)

    def _to_bcl(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B, 1, L)
        elif x.ndim == 3:
            if x.shape[1] == 1:
                pass
            elif x.shape[2] == 1:
                x = x.permute(0, 2, 1).contiguous()
            else:
                x = x.permute(0, 2, 1).contiguous()  # (B, L, C) -> (B, C, L)
                x = x[:, :1, :]
        else:
            raise ValueError(f"EELM expects 2D/3D input, got shape={tuple(x.shape)}")
        return x.float()

    def forward(self, x: torch.Tensor, data_id=None, task_id=None) -> torch.Tensor:
        return self.network(self._to_bcl(x))

