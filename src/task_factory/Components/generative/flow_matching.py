from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_finite(
    name: str,
    tensor: torch.Tensor,
    t: torch.Tensor | None = None,
) -> None:
    if torch.isfinite(tensor).all():
        return
    details = [
        f"{name} contains NaN/Inf",
        f"shape={tuple(tensor.shape)}",
        f"dtype={tensor.dtype}",
        f"device={tensor.device}",
    ]
    if t is not None and t.numel():
        t_float = t.detach().float()
        details.append(
            f"t_range=({float(t_float.min()):.6g}, {float(t_float.max()):.6g})"
        )
    raise ValueError("; ".join(details))


class ConditionalFlowMatchingLoss(nn.Module):
    """Velocity matching for ``x_t=(1-t)z+t*x_1``."""

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = float(eps)
        if not 0.0 <= self.eps < 0.5:
            raise ValueError(f"eps must be in [0, 0.5), got {self.eps}")

    def sample_t(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        return (
            torch.rand(int(batch_size), device=device, dtype=dtype)
            * (1.0 - 2.0 * self.eps)
            + self.eps
        )

    @staticmethod
    def _view_t(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype)
        if t.ndim == 1:
            t = t.view(-1, 1, 1)
        elif t.ndim == 2 and t.shape[1] == 1:
            t = t.view(-1, 1, 1)
        elif t.ndim == 3 and tuple(t.shape[1:]) == (1, 1):
            pass
        else:
            raise ValueError(
                "t must have shape [N], [N,1], or [N,1,1], got "
                f"{tuple(t.shape)}"
            )
        if t.shape[0] != x.shape[0]:
            raise ValueError(
                f"t batch mismatch: expected {x.shape[0]}, got {t.shape[0]}"
            )
        _check_finite("t", t, t)
        if torch.any(t < 0.0) or torch.any(t > 1.0):
            raise ValueError("t values must be in [0, 1]")
        return t

    def sample_xt(
        self,
        x1: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if x1.shape != noise.shape:
            raise ValueError(
                f"x1/noise shape mismatch: {tuple(x1.shape)} vs {tuple(noise.shape)}"
            )
        _check_finite("x1", x1, t)
        _check_finite("noise", noise, t)
        t_view = self._view_t(t, x1)
        x_t = (1.0 - t_view) * noise + t_view * x1
        _check_finite("x_t", x_t, t)
        return x_t

    @staticmethod
    def target_velocity(
        x1: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        if x1.shape != noise.shape:
            raise ValueError(
                f"x1/noise shape mismatch: {tuple(x1.shape)} vs {tuple(noise.shape)}"
            )
        return x1 - noise

    def forward(
        self,
        predicted_velocity: torch.Tensor,
        x1: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if predicted_velocity.shape != x1.shape:
            raise ValueError(
                "predicted_velocity/x1 shape mismatch: "
                f"{tuple(predicted_velocity.shape)} vs {tuple(x1.shape)}"
            )
        self._view_t(t, x1)
        _check_finite("predicted_velocity", predicted_velocity, t)
        target = self.target_velocity(x1, noise)
        _check_finite("target_velocity", target, t)
        loss = F.mse_loss(predicted_velocity, target)
        _check_finite("loss", loss, t)
        return {"loss": loss, "mse_v": loss.detach()}
