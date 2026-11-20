"""Conditional Gaussian VAE for fixed-length, real-valued signal windows.

Direct implementation of the reparameterized Gaussian VAE equations in
Kingma and Welling, https://arxiv.org/abs/1312.6114, with label/domain inputs.
The Task specifies the fixed observation variance and samples that likelihood.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .condition_encoder import ConditionEncoder


class Model(nn.Module):
    """Encode ``[N,C,L]`` into diagonal-Gaussian latent parameters.

    ``forward(x, condition)`` returns decoder mean, posterior mean and log-variance.
    ``decode(z, condition)`` returns a Gaussian observation mean ``[N,C,L]``.
    Inputs must use the model parameter dtype/device; the runtime uses float32.
    """

    def __init__(self, args_model: Any, metadata: Any = None) -> None:
        super().__init__()
        self.channels = int(getattr(args_model, "in_channels", 2))
        self.length = int(getattr(args_model, "window_size", 128))
        self.latent_dim = int(getattr(args_model, "latent_dim", 16))
        hidden_dim = int(getattr(args_model, "hidden_dim", 64))
        condition_dim = int(getattr(args_model, "condition_dim", 32))
        if min(self.channels, self.length, self.latent_dim, hidden_dim, condition_dim) <= 0:
            raise ValueError("VAE dimensions must all be positive")
        self.condition_encoder = ConditionEncoder(
            metadata=metadata,
            embedding_dim=condition_dim,
            num_fault_classes=getattr(args_model, "num_fault_classes", None),
            num_domains=getattr(args_model, "num_domains", None),
        )
        self.encoder = nn.Sequential(
            nn.Linear(self.channels * self.length + condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim + condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.channels * self.length),
        )

    def encode(self, windows, condition):
        if windows.ndim != 3 or windows.shape[1:] != (self.channels, self.length):
            raise ValueError(
                f"VAE windows must have shape [N,{self.channels},{self.length}]"
            )
        embedding = self.condition_encoder(condition, windows.new_zeros(windows.shape[0]))
        return self.encoder(
            torch.cat((windows.flatten(1), embedding), dim=1)
        ).chunk(2, dim=1)

    def decode(self, latent, condition):
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"VAE latent must have shape [N,{self.latent_dim}]")
        embedding = self.condition_encoder(condition, latent.new_zeros(latent.shape[0]))
        return self.decoder(torch.cat((latent, embedding), dim=1)).reshape(
            latent.shape[0], self.channels, self.length
        )

    @staticmethod
    def reparameterize(mean, log_variance):
        return mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)

    def forward(self, windows, condition):
        mean, log_variance = self.encode(windows, condition)
        latent = self.reparameterize(mean, log_variance)
        return self.decode(latent, condition), mean, log_variance
