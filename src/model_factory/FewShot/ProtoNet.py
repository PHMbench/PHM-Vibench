"""Minimal implementation of a Prototypical Network encoder."""

import torch
from torch import nn


class Model(nn.Module):
    """A lightweight CNN used in few-shot baselines."""

    def __init__(self, args_model, metadata=None):
        super().__init__()
        in_channels = getattr(args_model, "in_channels", 1)
        embedding_dim = getattr(args_model, "embedding_dim", 64)

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:  # (B, L) -> (B, 1, L)
            x = x.unsqueeze(1)
        return self.encoder(x)
