"""Conditional VAE with a declared Gaussian observation likelihood."""

from __future__ import annotations

import math
from numbers import Integral

import torch

from .base import GenerativeTask


def gaussian_vae_loss(windows, reconstruction, mean, log_variance, *, observation_std, kl_weight):
    """Return per-window summed NLL + weighted KL, averaged over the batch."""
    nll = (
        0.5 * ((windows - reconstruction) / observation_std).square()
        + math.log(observation_std) + 0.5 * math.log(2.0 * math.pi)
    ).flatten(1).sum(1).mean()
    kl = 0.5 * (mean.square() + log_variance.exp() - log_variance - 1.0).sum(1).mean()
    return {"loss": nll + kl_weight * kl, "reconstruction_nll": nll, "kl": kl}


class ConditionalVAETask(GenerativeTask):
    """Use the same observation variance during loss evaluation and generation.

    At ``kl_weight=1`` the objective is a one-sample negative Gaussian ELBO.
    Changing that weight selects a beta-weighted objective.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.observation_std = float(getattr(self.args_task, "observation_std", 1.0))
        self.kl_weight = float(getattr(self.args_task, "kl_weight", 1.0))
        if not math.isfinite(self.observation_std) or self.observation_std <= 0:
            raise ValueError("task.observation_std must be finite and positive")
        if not math.isfinite(self.kl_weight) or self.kl_weight < 0:
            raise ValueError("task.kl_weight must be finite and non-negative")
        self.method_id = "conditional_vae"
        self.loss_id = "gaussian_nll_plus_weighted_kl"
        self.sampler_id = "gaussian_latent_and_observation"
        self.save_hyperparameters(
            {
                "method_id": self.method_id,
                "observation_std": self.observation_std,
                "kl_weight": self.kl_weight,
                "loss_reduction": "sum_per_window_mean_over_batch",
            }
        )

    def forward(self, windows, condition):
        return self.network(windows, condition)

    def _shared_step(self, batch, stage):
        windows = self._to_ncl(batch["x"])
        condition = self._extract_condition(batch, windows.shape[0])
        reconstruction, mean, log_variance = self.network(windows, condition)
        values = gaussian_vae_loss(
            windows, reconstruction, mean, log_variance,
            observation_std=self.observation_std, kl_weight=self.kl_weight,
        )
        for name, value in values.items():
            if not torch.isfinite(value):
                raise ValueError(f"VAE {name} is non-finite")
            self.log(
                f"{stage}_{name}", value,
                on_step=stage == "train", on_epoch=True,
                batch_size=windows.shape[0], logger=True,
            )
        return values["loss"]

    def sampler_metadata(self):
        return {
            "sampler_id": self.sampler_id,
            "nfe": 1,
            "latent_prior": "standard_normal",
            "observation_likelihood": "isotropic_gaussian",
            "observation_std": self.observation_std,
            "sample_decoder_mean_only": False,
        }

    @torch.no_grad()
    def sample(self, condition, *, num_samples, length, channels, num_steps, device=None):
        if isinstance(num_steps, bool) or not isinstance(num_steps, Integral):
            raise TypeError(f"num_steps must be an integer, got {num_steps!r}")
        if num_steps != 1:
            raise ValueError("conditional VAE sampling requires num_steps=1")
        if num_samples <= 0 or (channels, length) != (self.network.channels, self.network.length):
            raise ValueError("VAE sampling requires a positive count and the configured [C,L]")
        sample_device = torch.device(device or self.device)
        self.network.to(sample_device)
        latent = torch.randn(
            num_samples, self.network.latent_dim, device=sample_device,
            dtype=next(self.network.parameters()).dtype,
        )
        mean = self.network.decode(latent, condition)
        samples = mean + self.observation_std * torch.randn_like(mean)
        if not torch.isfinite(samples).all():
            raise ValueError("VAE generated non-finite samples")
        return samples


task = ConditionalVAETask
