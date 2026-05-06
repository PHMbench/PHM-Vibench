from __future__ import annotations

import torch


class DDPMScheduler:
    """Minimal finite DDPM beta/alpha/alpha_bar scheduler."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        if num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("Require 0 < beta_start < beta_end < 1")
        self.num_train_timesteps = int(num_train_timesteps)
        self.betas = torch.linspace(beta_start, beta_end, self.num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        for name, value in {
            "betas": self.betas,
            "alphas": self.alphas,
            "alpha_bars": self.alpha_bars,
        }.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN/Inf")

    def _index(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.view(1)
        idx = t.long().view(-1)
        if (idx < 0).any() or (idx >= self.num_train_timesteps).any():
            raise ValueError(
                f"timestep out of range [0,{self.num_train_timesteps - 1}]: {idx.tolist()}"
            )
        return idx

    def alpha_bar_at(
        self,
        t: torch.Tensor,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        idx = self._index(t).cpu()
        return self.alpha_bars[idx].to(device=device, dtype=dtype)

    def alpha_at(
        self,
        t: torch.Tensor,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        idx = self._index(t).cpu()
        return self.alphas[idx].to(device=device, dtype=dtype)

    def beta_at(
        self,
        t: torch.Tensor,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        idx = self._index(t).cpu()
        return self.betas[idx].to(device=device, dtype=dtype)

