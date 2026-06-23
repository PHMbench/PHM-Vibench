from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

import torch
import torch.nn as nn

from .legacy_collection.Resnet import BasicBlock, Bottleneck, ResNet as _LegacyResNet


def _parse_layers(value: Any) -> List[int]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return [int(x) for x in value]
    return [2, 2, 2, 2]


class Model(nn.Module):
    """Factory entry for legacy ResNet-family model."""

    def __init__(self, args: Any, metadata: Any = None):
        super().__init__()
        self.args = args if args is not None else SimpleNamespace()
        self.in_channels = int(getattr(self.args, "in_channel", getattr(self.args, "in_channels", 1)))
        self.num_classes = int(getattr(self.args, "num_class", getattr(self.args, "num_classes", 2)))
        layers = _parse_layers(getattr(self.args, "layers", [2, 2, 2, 2]))
        block_type = str(getattr(self.args, "block_type", "basic")).strip().lower()
        block = Bottleneck if block_type in {"bottleneck", "bot"} else BasicBlock
        first_kernel = str(getattr(self.args, "first_kernel", "conv"))
        zero_init = bool(getattr(self.args, "zero_init_residual", False))
        self.network = _LegacyResNet(
            block=block,
            layers=layers,
            in_channel=self.in_channels,
            num_class=self.num_classes,
            zero_init_residual=zero_init,
            first_kernel=first_kernel,
        )

    def _to_blc(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # (B, L, 1)
        elif x.ndim == 3:
            if x.shape[2] == self.in_channels:
                pass
            elif x.shape[1] == self.in_channels:
                x = x.permute(0, 2, 1).contiguous()
            else:
                x = x if x.shape[1] >= x.shape[2] else x.permute(0, 2, 1).contiguous()
        else:
            raise ValueError(f"Resnet expects 2D/3D input, got shape={tuple(x.shape)}")

        channels = x.shape[2]
        if channels < self.in_channels:
            repeats = (self.in_channels + channels - 1) // channels
            x = x.repeat(1, 1, repeats)[:, :, : self.in_channels]
        elif channels > self.in_channels:
            x = x[:, :, : self.in_channels]
        return x.float()

    def forward(self, x: torch.Tensor, data_id=None, task_id=None) -> torch.Tensor:
        return self.network(self._to_blc(x))

