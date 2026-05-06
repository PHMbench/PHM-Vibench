from __future__ import annotations

import torch


def sample_euler_ode(
    model,
    noise: torch.Tensor,
    condition: dict[str, torch.Tensor],
    num_steps: int,
    t0: float = 0.0,
    t1: float = 1.0,
) -> torch.Tensor:
    """Sample CFM trajectories with explicit Euler integration."""
    if noise.ndim != 3:
        raise ValueError(f"noise must be [N, C, L], got shape={tuple(noise.shape)}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    x = noise
    dt = float(t1 - t0) / float(num_steps)
    for step in range(num_steps):
        t_value = t0 + step * dt
        t = torch.full((x.shape[0],), t_value, dtype=x.dtype, device=x.device)
        velocity = model(x, t, condition)
        if velocity.shape != x.shape:
            raise ValueError(f"velocity shape mismatch at step={step}: {tuple(velocity.shape)} vs {tuple(x.shape)}")
        if not torch.isfinite(velocity).all():
            raise ValueError(
                f"velocity contains NaN/Inf at step={step}, t={t_value}, "
                f"shape={tuple(velocity.shape)}, dtype={velocity.dtype}, device={velocity.device}"
            )
        x = x + dt * velocity
    return x

