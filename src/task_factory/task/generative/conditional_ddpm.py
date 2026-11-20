"""Conditional DDPM for continuous waveforms (Ho et al., arXiv:2006.11239).

Uses Eq. (14)'s epsilon MSE and Algorithm 2 with Eq. (7)'s fixed posterior
variance. Waveform amplitudes are not clipped to the image interval [-1,1].
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.task_factory.Components.generative.euler_ode import _normalized_condition

from .base import GenerativeTask


class ConditionalDDPMTask(GenerativeTask):
    """Discrete noise-prediction objective with full ancestral sampling."""

    def __init__(
        self,
        network: nn.Module,
        args_data: Any,
        args_model: Any,
        args_task: Any,
        args_trainer: Any,
        args_environment: Any,
        metadata: Any,
    ) -> None:
        super().__init__(
            network, args_data, args_model, args_task, args_trainer,
            args_environment, metadata,
        )
        steps = getattr(args_task, "diffusion_steps", 1000)
        if isinstance(steps, bool) or not isinstance(steps, Integral):
            raise TypeError("diffusion_steps must be an integer")
        if steps <= 0:
            raise ValueError("diffusion_steps must be positive")
        self.diffusion_steps = int(steps)
        self.beta_start = float(getattr(args_task, "beta_start", 1e-4))
        self.beta_end = float(getattr(args_task, "beta_end", 0.02))
        if not (
            math.isfinite(self.beta_start) and math.isfinite(self.beta_end)
            and 0 < self.beta_start <= self.beta_end < 1
        ):
            raise ValueError("DDPM requires 0 < beta_start <= beta_end < 1")

        # Form cumulative products in float64 before casting to waveform dtype.
        betas = torch.linspace(
            self.beta_start, self.beta_end, self.diffusion_steps,
            dtype=torch.float64,
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        previous_alpha_bars = torch.cat((torch.ones_like(alpha_bars[:1]), alpha_bars[:-1]))
        posterior_variance = betas * (1.0 - previous_alpha_bars) / (1.0 - alpha_bars)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("posterior_variance", posterior_variance)
        self.method_id = "conditional_ddpm"
        self.loss_id = "ddpm_epsilon_mse"
        self.sampler_id = "ddpm_ancestral"
        self._last_sample_nfe: int | None = None
        self.save_hyperparameters({
            "diffusion_steps": self.diffusion_steps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
        })

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # Fixed diffusion buffers are protocol settings, not learned parameters.
        # Reject a different schedule before loading any network weights.
        for name in ("betas", "alphas", "alpha_bars", "posterior_variance"):
            if name not in state_dict:
                continue  # Preserve PyTorch's normal missing-key handling.
            saved = state_dict[name]
            expected = getattr(self, name)
            if (
                not isinstance(saved, torch.Tensor)
                or not torch.is_floating_point(saved)
                or not torch.equal(
                    saved.detach().cpu(),
                    expected.detach().to(device="cpu", dtype=saved.dtype),
                )
            ):
                raise ValueError(
                    f"DDPM checkpoint schedule {name!r} does not match configured "
                    f"diffusion_steps={self.diffusion_steps}, beta_start={self.beta_start}, "
                    f"beta_end={self.beta_end}; restore with the training schedule"
                )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _forward_diffusion(
        self, clean: torch.Tensor, noise: torch.Tensor, steps: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.alpha_bars.to(clean)[steps].view(-1, 1, 1)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    def _reverse_parameters(
        self, x_t: torch.Tensor, predicted_noise: torch.Tensor, steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self.alphas.to(x_t)[steps].view(-1, 1, 1)
        alpha_bar = self.alpha_bars.to(x_t)[steps].view(-1, 1, 1)
        beta = self.betas.to(x_t)[steps].view(-1, 1, 1)
        mean = (x_t - beta / (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha.sqrt()
        variance = self.posterior_variance.to(x_t)[steps].view(-1, 1, 1)
        return mean, variance

    def _predict_noise(
        self, x_t: torch.Tensor, steps: torch.Tensor,
        condition: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # Internal index 0 represents paper timestep 1, not clean data.
        time = (steps.to(dtype=x_t.dtype) + 1.0) / self.diffusion_steps
        predicted = self.network(x_t, time, condition)
        if predicted.shape != x_t.shape:
            raise ValueError("predicted noise must preserve the [N,C,L] waveform shape")
        if predicted.device != x_t.device or predicted.dtype != x_t.dtype:
            raise ValueError("predicted noise must preserve waveform device and dtype")
        if not torch.isfinite(predicted).all():
            raise ValueError("predicted noise contains NaN/Inf")
        return predicted

    def _shared_step(self, batch: Mapping[str, Any], stage: str) -> torch.Tensor:
        if "x" not in batch:
            raise ValueError("generative batch is missing x")
        clean = self._to_ncl(batch["x"])
        condition = self._extract_condition(batch, clean.shape[0])
        noise = torch.randn_like(clean)
        steps = torch.randint(self.diffusion_steps, (clean.shape[0],), device=clean.device)
        x_t = self._forward_diffusion(clean, noise, steps)
        predicted_noise = self._predict_noise(x_t, steps, condition)
        loss = F.mse_loss(predicted_noise, noise)
        self.log(
            f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True,
            prog_bar=False, logger=True, batch_size=clean.shape[0],
        )
        return loss

    def sampler_metadata(self) -> dict[str, Any]:
        return {
            "sampler_id": self.sampler_id,
            "prediction_type": "epsilon",
            "beta_schedule": "linear",
            "beta_start": float(self.betas[0]),
            "beta_end": float(self.betas[-1]),
            "diffusion_steps": self.diffusion_steps,
            "nfe": self._last_sample_nfe,
            "variance": "fixed_posterior",
            "terminal_alpha_bar": float(self.alpha_bars[-1]),
            "clip_denoised": False,
        }

    @torch.no_grad()
    def sample(
        self,
        condition: dict[str, torch.Tensor],
        *,
        num_samples: int,
        length: int,
        channels: int,
        num_steps: int,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        self._last_sample_nfe = None
        if isinstance(num_steps, bool) or not isinstance(num_steps, Integral):
            raise TypeError("num_steps must be an integer")
        if num_steps != self.diffusion_steps:
            raise ValueError(
                "DDPM ancestral sampling requires num_steps == diffusion_steps "
                f"({self.diffusion_steps}); got {num_steps}"
            )
        for name, value in (("num_samples", num_samples), ("length", length), ("channels", channels)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if channels != int(getattr(self.args_model, "in_channels", 2)):
            raise ValueError("sample channels must match model.in_channels")
        sample_device = torch.device(device or self.device)
        network = self.network.to(sample_device)
        dtype = next(network.parameters()).dtype
        condition = _normalized_condition(
            condition, batch_size=num_samples, device=sample_device,
        )
        training_state = network.training
        network.eval()
        try:
            state = torch.randn(num_samples, channels, length, device=sample_device, dtype=dtype)
            nfe = 0
            for step in reversed(range(self.diffusion_steps)):
                steps = torch.full((num_samples,), step, device=sample_device, dtype=torch.long)
                predicted_noise = self._predict_noise(state, steps, condition)
                nfe += 1
                mean, variance = self._reverse_parameters(state, predicted_noise, steps)
                state = mean if step == 0 else mean + variance.sqrt() * torch.randn_like(state)
            if not torch.isfinite(state).all():
                raise ValueError("DDPM sampling produced NaN/Inf")
            self._last_sample_nfe = nfe
        finally:
            network.train(training_state)
        return state


task = ConditionalDDPMTask
