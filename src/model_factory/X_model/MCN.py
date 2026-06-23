from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .legacy_collection.MCN.models import MultiChannel_MCN_GFK, MultiChannel_MCN_WFK


class Model(nn.Module):
    """Factory entry for legacy MCN models (default: GFK)."""

    def __init__(self, args: Any, metadata: Any = None):
        super().__init__()
        self.args = args if args is not None else SimpleNamespace()
        self.in_channels = int(getattr(self.args, "in_channels", 1))
        self.num_classes = int(getattr(self.args, "num_classes", getattr(self.args, "output_dim", 2)))
        self.seq_len = int(getattr(self.args, "seq_len", getattr(self.args, "in_dim", 128)))
        self.num_mfks = int(getattr(self.args, "num_mfks", 8))
        mode = str(getattr(self.args, "mode", "gfk")).strip().lower()
        self.use_wfk = mode == "wfk"

        fft_bins = self.seq_len // 2 + 1
        ff = np.arange(0, fft_bins, dtype=np.float32) / float(fft_bins)
        cls = MultiChannel_MCN_WFK if self.use_wfk else MultiChannel_MCN_GFK
        self.network = cls(
            ff=ff,
            in_channels=self.in_channels,
            num_MFKs=self.num_mfks,
            num_classes=self.num_classes,
        )

    def _to_blc(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        elif x.ndim == 3:
            if x.shape[2] == self.in_channels:
                pass
            elif x.shape[1] == self.in_channels:
                x = x.permute(0, 2, 1).contiguous()
            else:
                x = x if x.shape[1] >= x.shape[2] else x.permute(0, 2, 1).contiguous()
        else:
            raise ValueError(f"MCN expects 2D/3D input, got shape={tuple(x.shape)}")

        channels = x.shape[2]
        if channels < self.in_channels:
            repeats = (self.in_channels + channels - 1) // channels
            x = x.repeat(1, 1, repeats)[:, :, : self.in_channels]
        elif channels > self.in_channels:
            x = x[:, :, : self.in_channels]

        length = x.shape[1]
        if length < self.seq_len:
            x = torch.nn.functional.pad(x, (0, 0, 0, self.seq_len - length))
        elif length > self.seq_len:
            x = x[:, : self.seq_len, :]
        return x.float()

    def forward(self, x: torch.Tensor, data_id=None, task_id=None) -> torch.Tensor:
        return self.network(self._to_blc(x))

