from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.task_factory.Components.generative.schedulers.ddpm import DDPMScheduler


def _check_finite(name: str, tensor: torch.Tensor) -> None:
    if torch.isfinite(tensor).all():
        return
    raise ValueError(
        f"{name} contains NaN/Inf; shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}, device={tensor.device}"
    )


class DDPMEpsilonPredictionLoss(nn.Module):
    """DDPM epsilon-prediction loss for `[N, C, L]` PHM windows."""

    def __init__(self, scheduler: DDPMScheduler | None = None, num_train_timesteps: int = 1000) -> None:
        super().__init__()
        self.scheduler = scheduler or DDPMScheduler(num_train_timesteps=num_train_timesteps)

    def sample_timesteps(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.randint(0, self.scheduler.num_train_timesteps, (batch_size,), device=device)

    def q_sample(self, x0: torch.Tensor, epsilon: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x0.shape != epsilon.shape:
            raise ValueError(f"x0 and epsilon shape mismatch: {tuple(x0.shape)} vs {tuple(epsilon.shape)}")
        _check_finite("x0", x0)
        _check_finite("epsilon", epsilon)
        alpha_bar = self.scheduler.alpha_bar_at(t, device=x0.device, dtype=x0.dtype).view(-1, 1, 1)
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * epsilon

    def forward(
        self,
        pred_epsilon: torch.Tensor,
        x0: torch.Tensor,
        epsilon: torch.Tensor,
        t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if pred_epsilon.shape != x0.shape:
            raise ValueError(
                f"pred_epsilon and x0 shape mismatch: {tuple(pred_epsilon.shape)} vs {tuple(x0.shape)}"
            )
        _check_finite("pred_epsilon", pred_epsilon)
        _check_finite("x0", x0)
        _check_finite("epsilon", epsilon)
        loss = F.mse_loss(pred_epsilon, epsilon)
        _check_finite("loss", loss)
        return {"loss": loss, "mse_epsilon": loss.detach()}
