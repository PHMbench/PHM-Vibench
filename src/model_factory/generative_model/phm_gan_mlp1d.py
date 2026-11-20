"""Fixed-window conditional GAN with separate generator/discriminator parameters.

The objective follows Goodfellow et al. (https://arxiv.org/abs/1406.2661) and
conditioning follows Mirza and Osindero (https://arxiv.org/abs/1411.1784).
This is a direct implementation of those equations, not copied vendor code.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .condition_encoder import ConditionEncoder


class _Generator(nn.Module):
    def __init__(self, encoder, latent_dim, channels, length, hidden_dim) -> None:
        super().__init__()
        self.condition_encoder = encoder
        self.latent_dim = latent_dim
        self.channels = channels
        self.length = length
        self.layers = nn.Sequential(
            nn.Linear(latent_dim + encoder.embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, channels * length),
        )

    def forward(self, noise, condition):
        if noise.ndim != 2 or noise.shape[1] != self.latent_dim:
            raise ValueError(f"GAN noise must have shape [N,{self.latent_dim}]")
        time = noise.new_zeros(noise.shape[0])
        embedding = self.condition_encoder(condition, time)
        return self.layers(torch.cat((noise, embedding), dim=1)).reshape(
            noise.shape[0], self.channels, self.length
        )


class _Discriminator(nn.Module):
    def __init__(self, encoder, channels, length, hidden_dim) -> None:
        super().__init__()
        self.condition_encoder = encoder
        self.channels = channels
        self.length = length
        self.layers = nn.Sequential(
            nn.Linear(channels * length + encoder.embedding_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, windows, condition):
        if windows.ndim != 3 or windows.shape[1:] != (self.channels, self.length):
            raise ValueError(
                f"GAN windows must have shape [N,{self.channels},{self.length}]"
            )
        time = windows.new_zeros(windows.shape[0])
        embedding = self.condition_encoder(condition, time)
        return self.layers(
            torch.cat((windows.flatten(1), embedding), dim=1)
        ).squeeze(1)


class Model(nn.Module):
    """Map Gaussian latent vectors ``[N,latent_dim]`` to windows ``[N,C,L]``.

    ``discriminator(windows, condition)`` returns real/fake logits ``[N]``.
    Separate condition encoders prevent sharing parameters across optimizers.
    Both networks use the parameter dtype/device; float32 CPU is the smoke path.
    """

    def __init__(self, args_model: Any, metadata: Any = None) -> None:
        super().__init__()
        self.channels = int(getattr(args_model, "in_channels", 2))
        self.length = int(getattr(args_model, "window_size", 128))
        self.latent_dim = int(getattr(args_model, "latent_dim", 16))
        hidden_dim = int(getattr(args_model, "hidden_dim", 64))
        condition_dim = int(getattr(args_model, "condition_dim", 32))
        if min(self.channels, self.length, self.latent_dim, hidden_dim, condition_dim) <= 0:
            raise ValueError("GAN dimensions must all be positive")
        encoder_args = dict(
            metadata=metadata,
            embedding_dim=condition_dim,
            num_fault_classes=getattr(args_model, "num_fault_classes", None),
            num_domains=getattr(args_model, "num_domains", None),
        )
        self.generator = _Generator(
            ConditionEncoder(**encoder_args),
            self.latent_dim,
            self.channels,
            self.length,
            hidden_dim,
        )
        self.discriminator = _Discriminator(
            ConditionEncoder(**encoder_args), self.channels, self.length, hidden_dim
        )

    def forward(self, noise, condition):
        return self.generator(noise, condition)
