from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from .legacy_collection.TFN.Models.TFN import TFN_Chirplet, TFN_Morlet, TFN_STTF


class Model(nn.Module):
    """Factory entry for legacy TFN models (default: TFN_Morlet)."""

    def __init__(self, args: Any, metadata: Any = None):
        super().__init__()
        self.args = args if args is not None else SimpleNamespace()
        self.in_channels = int(getattr(self.args, "in_channels", 1))
        self.num_classes = int(getattr(self.args, "num_classes", getattr(self.args, "output_dim", 2)))
        self.mid_channel = int(getattr(self.args, "mid_channel", 16))
        variant = str(getattr(self.args, "variant", "morlet")).strip().lower()
        cls_map = {
            "morlet": TFN_Morlet,
            "sttf": TFN_STTF,
            "chirplet": TFN_Chirplet,
        }
        model_cls = cls_map.get(variant, TFN_Morlet)
        self.network = model_cls(
            in_channels=self.in_channels,
            out_channels=self.num_classes,
            mid_channel=self.mid_channel,
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
            raise ValueError(f"TFN expects 2D/3D input, got shape={tuple(x.shape)}")

        channels = x.shape[2]
        if channels < self.in_channels:
            repeats = (self.in_channels + channels - 1) // channels
            x = x.repeat(1, 1, repeats)[:, :, : self.in_channels]
        elif channels > self.in_channels:
            x = x[:, :, : self.in_channels]
        return x.float()

    def forward(self, x: torch.Tensor, data_id=None, task_id=None) -> torch.Tensor:
        return self.network(self._to_blc(x))
