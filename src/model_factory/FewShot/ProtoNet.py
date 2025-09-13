"""Minimal implementation of a Prototypical Network encoder."""

import torch
from torch import nn


class Model(nn.Module):
    """Lightweight CNN encoder for prototypical networks.

    Parameters
    ----------
    args_model : Namespace
        Hyper parameters such as ``in_channels`` and ``embedding_dim``.
    metadata : Any, optional
        Unused but kept for compatibility.

    Notes
    -----
    Expects input tensors shaped ``(B, L, C)`` or ``(B, L)`` and returns
    embeddings of shape ``(B, embedding_dim)``.
    """

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
        """Encode a batch of sequences.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, L, C)`` or ``(B, L)``.

        Returns
        -------
        torch.Tensor
            Embedding tensor of shape ``(B, embedding_dim)``.
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.encoder(x)

