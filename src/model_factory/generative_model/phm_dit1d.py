from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .condition_encoder import ConditionEncoder


def _match_length(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[-1] == length:
        return x
    if x.shape[-1] > length:
        return x[..., :length]
    return F.pad(x, (0, length - x.shape[-1]))


class Model(nn.Module):
    """Tiny DiT-style 1D transformer backbone for PHM generative signals."""

    stateless = True

    def __init__(self, args_model: Any, metadata: Any = None) -> None:
        super().__init__()
        channels = int(getattr(args_model, "in_channels", getattr(args_model, "channels", 2)))
        d_model = int(getattr(args_model, "hidden_dim", getattr(args_model, "d_model", 64)))
        condition_dim = int(getattr(args_model, "condition_dim", d_model))
        patch_size = int(getattr(args_model, "patch_size", 8))
        num_layers = int(getattr(args_model, "num_layers", 2))
        num_heads = int(getattr(args_model, "num_heads", 4))
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if d_model % num_heads != 0:
            raise ValueError("hidden_dim/d_model must be divisible by num_heads")
        self.channels = channels
        self.patch_size = patch_size
        self.condition_encoder = ConditionEncoder(
            metadata=metadata,
            embedding_dim=condition_dim,
            num_fault_classes=getattr(args_model, "num_fault_classes", None),
            num_domains=getattr(args_model, "num_domains", None),
        )
        self.cond_proj = nn.Linear(condition_dim, d_model)
        self.patch = nn.Conv1d(channels, d_model, kernel_size=patch_size, stride=patch_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.unpatch = nn.ConvTranspose1d(d_model, channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: dict[str, torch.Tensor]) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be [N, C, L], got shape={tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"x channel mismatch: expected {self.channels}, got {x.shape[1]}")
        length = x.shape[-1]
        pad = (-length) % self.patch_size
        x_in = F.pad(x.float(), (0, pad)) if pad else x.float()
        tokens = self.patch(x_in).transpose(1, 2)
        cond = self.cond_proj(self.condition_encoder(condition, t)).unsqueeze(1)
        tokens = self.transformer(tokens + cond)
        out = self.unpatch(tokens.transpose(1, 2))
        return _match_length(out, length)
