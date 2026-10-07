"""Multilayer wavelet attention CNN with explicit four/six-stage identity.

Architecture reference: PHM-Code/MWA-CNN, revision
692020e1dc9d41131844b77935fa0318af709b49, Network_Code/MWA-CNN-6.ipynb.
The six-stage path extends this repository's existing four-stage implementation;
it does not import the reference notebook's data or test-selection procedure.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import torch
from torch import nn
from pytorch_wavelets import DWT1DForward


class A_cSE(nn.Module):
    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, in_ch // 2, kernel_size=1),
            nn.BatchNorm1d(in_ch // 2),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_ch // 2, in_ch, kernel_size=1),
            nn.BatchNorm1d(in_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.conv0(x).mean(dim=-1, keepdim=True)
        weights = torch.sigmoid(self.conv2(self.conv1(weights)))
        return x * weights + x


class SConv_1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, pad: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
            nn.GroupNorm(6, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Model(nn.Module):
    """Factory classifier taking (batch, samples, channels) waveforms.

    Set depth=6 for MWA-CNN-6. Omitted depth preserves the existing four-stage
    model and its checkpoint names. in_channels (or input_dim) and num_classes
    are required positive integers. Wavelet db16, zero boundary extension,
    width 12, GroupNorm(6), and dropout 0.1 are fixed architecture choices.
    The Trainer owns device placement, preprocessing and supervision.

    Windows must contain at least one db16 filter support (32 samples); this
    input contract does not claim unpadded support for every deeper coefficient.
    """

    minimum_window_size = 32

    def __init__(self, args: Any, metadata: Any = None) -> None:
        super().__init__()
        self.depth = _positive_integer(getattr(args, "depth", 4), "depth")
        if self.depth not in (4, 6):
            raise ValueError("MWA_CNN depth must be 4 (legacy) or 6 (MWA-CNN-6)")
        channels = getattr(args, "in_channels", None)
        input_dim = getattr(args, "input_dim", None)
        if channels is not None and input_dim is not None and channels != input_dim:
            raise ValueError("MWA_CNN in_channels and input_dim must agree")
        self.in_channels = _positive_integer(
            channels if channels is not None else input_dim, "in_channels/input_dim"
        )
        self.num_classes = _positive_integer(
            getattr(args, "num_classes", None), "num_classes"
        )

        self.DWT0 = DWT1DForward(J=1, wave="db16", mode="zero")
        input_channels = 2 * self.in_channels
        for stage in range(1, self.depth):
            output_channels = 12 * 2 ** (stage - 1)
            # Keep existing module names so four-stage checkpoints load strictly.
            setattr(self, f"SConv{stage}", SConv_1D(input_channels, output_channels, 3, 0))
            setattr(self, f"DWT{stage}", DWT1DForward(J=1, wave="db16", mode="zero"))
            setattr(self, f"dropout{stage}", nn.Dropout(p=0.1))
            setattr(self, f"cSE{stage}", A_cSE(2 * output_channels))
            input_channels = 2 * output_channels
        self.SConv6 = SConv_1D(input_channels, input_channels, 3, 0)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(input_channels, self.num_classes)

    def forward(
        self, input: torch.Tensor, data_id: Any = None, task_id: Any = None
    ) -> torch.Tensor:
        if input.ndim != 3:
            raise ValueError("MWA_CNN expects (batch, samples, channels)")
        if input.shape[2] != self.in_channels:
            raise ValueError(
                f"MWA_CNN expects {self.in_channels} channels, got {input.shape[2]}"
            )
        if input.shape[1] < self.minimum_window_size:
            raise ValueError(
                f"MWA_CNN needs at least {self.minimum_window_size} measured samples, "
                f"got {input.shape[1]}"
            )
        if input.shape[0] == 0 or (self.training and input.shape[0] < 2):
            raise ValueError("MWA_CNN needs a nonempty batch and at least 2 samples in training")
        if not input.is_floating_point():
            raise TypeError("MWA_CNN expects a floating-point waveform")

        low, high = self.DWT0(input.transpose(1, 2))
        output = torch.cat((low, high[0]), dim=1)
        for stage in range(1, self.depth):
            output = getattr(self, f"SConv{stage}")(output)
            low, high = getattr(self, f"DWT{stage}")(output)
            output = torch.cat((low, high[0]), dim=1)
            output = getattr(self, f"dropout{stage}")(output)
            output = getattr(self, f"cSE{stage}")(output)
        output = self.SConv6(output)
        return self.fc(self.avg_pool(output).flatten(1))


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"MWA_CNN {name} must be a positive integer, got {value!r}")
    return int(value)
