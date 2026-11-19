"""Conditional non-saturating GAN with one D and one G update per batch."""

from __future__ import annotations

from numbers import Integral

import torch
import torch.nn.functional as F

from .base import GenerativeTask


class ConditionalGANTask(GenerativeTask):
    """Own adversarial losses and separate manual optimizer steps.

    ``val_loss`` is the non-saturating generator BCE against the current
    discriminator on validation conditions, not a distribution-quality score.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False
        self.method_id = "conditional_gan"
        self.loss_id = "non_saturating_adversarial_bce"
        self.sampler_id = "gaussian_latent_generator"
        self.save_hyperparameters(
            {
                "method_id": self.method_id,
                "loss_id": self.loss_id,
                "generator_updates_per_batch": 1,
                "discriminator_updates_per_batch": 1,
                "validation_criterion": "generator_non_saturating_bce",
            }
        )

    def forward(self, noise, condition):
        return self.network(noise, condition)

    def _noise(self, windows):
        return torch.randn(
            windows.shape[0], self.network.latent_dim,
            dtype=windows.dtype, device=windows.device,
        )

    def _log_losses(self, stage, generator_loss, discriminator_loss, batch_size):
        for name, value in (("loss", generator_loss), ("discriminator_loss", discriminator_loss)):
            if not torch.isfinite(value):
                raise ValueError(f"GAN {stage}_{name} is non-finite")
            self.log(
                f"{stage}_{name}", value,
                on_step=stage == "train", on_epoch=True,
                batch_size=batch_size, logger=True,
            )

    def training_step(self, batch, *args, **kwargs):
        windows = self._to_ncl(batch["x"])
        condition = self._extract_condition(batch, windows.shape[0])
        generator_optimizer, discriminator_optimizer = self.optimizers()

        self.toggle_optimizer(discriminator_optimizer)
        discriminator_optimizer.zero_grad()
        with torch.no_grad():
            fake = self.network.generator(self._noise(windows), condition)
        real_logits = self.network.discriminator(windows, condition)
        fake_logits = self.network.discriminator(fake, condition)
        discriminator_loss = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
        if not torch.isfinite(discriminator_loss):
            raise ValueError("GAN discriminator loss is non-finite")
        self.manual_backward(discriminator_loss)
        discriminator_optimizer.step()
        self.untoggle_optimizer(discriminator_optimizer)

        self.toggle_optimizer(generator_optimizer)
        generator_optimizer.zero_grad()
        fake = self.network.generator(self._noise(windows), condition)
        fake_logits = self.network.discriminator(fake, condition)
        generator_loss = F.softplus(-fake_logits).mean()
        if not torch.isfinite(generator_loss):
            raise ValueError("GAN generator loss is non-finite")
        self.manual_backward(generator_loss)
        generator_optimizer.step()
        self.untoggle_optimizer(generator_optimizer)
        self._log_losses("train", generator_loss, discriminator_loss, windows.shape[0])
        return generator_loss.detach()

    def _shared_step(self, batch, stage):
        windows = self._to_ncl(batch["x"])
        condition = self._extract_condition(batch, windows.shape[0])
        fake = self.network.generator(self._noise(windows), condition)
        real_logits = self.network.discriminator(windows, condition)
        fake_logits = self.network.discriminator(fake, condition)
        generator_loss = F.softplus(-fake_logits).mean()
        discriminator_loss = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
        self._log_losses(stage, generator_loss, discriminator_loss, windows.shape[0])
        return generator_loss

    def configure_optimizers(self):
        name = str(getattr(self.args_task, "optimizer", "adamw")).lower()
        if name not in {"adam", "adamw"}:
            raise ValueError(f"unsupported optimizer for generative task: {name}")
        optimizer = torch.optim.Adam if name == "adam" else torch.optim.AdamW
        options = dict(
            lr=float(getattr(self.args_task, "lr", 1e-4)),
            weight_decay=float(getattr(self.args_task, "weight_decay", 1e-4)),
        )
        return [
            optimizer(self.network.generator.parameters(), **options),
            optimizer(self.network.discriminator.parameters(), **options),
        ]

    def sampler_metadata(self):
        return {
            "sampler_id": self.sampler_id,
            "nfe": 1,
            "latent_prior": "standard_normal",
            "generator_updates_per_batch": 1,
            "discriminator_updates_per_batch": 1,
        }

    @torch.no_grad()
    def sample(self, condition, *, num_samples, length, channels, num_steps, device=None):
        if isinstance(num_steps, bool) or not isinstance(num_steps, Integral):
            raise TypeError(f"num_steps must be an integer, got {num_steps!r}")
        if num_steps != 1:
            raise ValueError("conditional GAN sampling requires num_steps=1")
        if num_samples <= 0 or (channels, length) != (self.network.channels, self.network.length):
            raise ValueError("GAN sampling requires a positive count and the configured [C,L]")
        sample_device = torch.device(device or self.device)
        self.network.to(sample_device)
        noise = torch.randn(
            num_samples, self.network.latent_dim, device=sample_device,
            dtype=next(self.network.parameters()).dtype,
        )
        samples = self.network.generator(noise, condition)
        if not torch.isfinite(samples).all():
            raise ValueError("GAN generated non-finite samples")
        return samples


task = ConditionalGANTask
