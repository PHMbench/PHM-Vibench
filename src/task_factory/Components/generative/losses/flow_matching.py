from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_finite(name: str, tensor: torch.Tensor, t: torch.Tensor | None = None) -> None:
    if torch.isfinite(tensor).all():
        return
    parts = [
        f"{name} contains NaN/Inf",
        f"shape={tuple(tensor.shape)}",
        f"dtype={tensor.dtype}",
        f"device={tensor.device}",
    ]
    if t is not None:
        t_float = t.float()
        parts.append(f"t_range=({float(t_float.min()):.6g}, {float(t_float.max()):.6g})")
    raise ValueError("; ".join(parts))


class ConditionalFlowMatchingLoss(nn.Module):
    """Conditional Flow Matching loss for PHM signals."""

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        if not 0.0 <= eps < 0.5:
            raise ValueError(f"eps must be in [0, 0.5), got {eps}")
        self.eps = float(eps)

    def sample_t(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.rand(batch_size, device=device) * (1.0 - 2.0 * self.eps) + self.eps

    def _view_t(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            return t.view(-1, 1, 1)
        if t.ndim == 2 and t.shape[1] == 1:
            return t.view(-1, 1, 1)
        if t.ndim == 3 and t.shape[1:] == (1, 1):
            return t
        raise ValueError(f"Expected t shape [N], [N,1], or [N,1,1], got {tuple(t.shape)}")

    def sample_xt(self, x1: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x1.shape != z.shape:
            raise ValueError(f"x1 and z shape mismatch: {tuple(x1.shape)} vs {tuple(z.shape)}")
        _check_finite("x1", x1, t)
        _check_finite("z", z, t)
        _check_finite("t", t, t)
        t_view = self._view_t(t, x1)
        return (1.0 - t_view) * z + t_view * x1

    def target_velocity(self, x1: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if x1.shape != z.shape:
            raise ValueError(f"x1 and z shape mismatch: {tuple(x1.shape)} vs {tuple(z.shape)}")
        return x1 - z

    def forward(
        self,
        pred_velocity: torch.Tensor,
        x1: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if pred_velocity.shape != x1.shape:
            raise ValueError(
                f"pred_velocity and x1 shape mismatch: {tuple(pred_velocity.shape)} vs {tuple(x1.shape)}"
            )
        _check_finite("pred_velocity", pred_velocity, t)
        target = self.target_velocity(x1, z)
        _check_finite("target_velocity", target, t)
        loss = F.mse_loss(pred_velocity, target)
        _check_finite("loss", loss, t)
        return {"loss": loss, "mse_v": loss.detach()}

