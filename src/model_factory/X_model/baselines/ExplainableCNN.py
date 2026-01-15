from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Simple 1D CNN baseline (torch-only).

    Input: `(B, L, C)`
    Output: `(B, num_classes)`
    """

    def __init__(self, args: Any, metadata: Optional[Any] = None):
        super().__init__()
        in_channels = int(getattr(args, "in_channels", 1))
        num_classes = int(getattr(args, "num_classes", 2))
        width = int(getattr(args, "width", 64))
        dropout = float(getattr(args, "dropout", 0.1))

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=9, padding=4),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(width, width * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(width * 2, width * 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(width * 4),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(width * 4, num_classes)

        device = getattr(args, "device", None)
        if device:
            self.to(device)

    def forward(self, x: torch.Tensor, data_id=None, task_id=None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B,L,C), got {tuple(x.shape)}")
        x = x.permute(0, 2, 1).contiguous()  # (B,C,L)
        x = self.features(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = self.dropout(x)
        return self.head(x)

