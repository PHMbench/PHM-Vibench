from __future__ import annotations

import torch


def sample_score_sde_annealed_langevin(
    model,
    noise: torch.Tensor,
    condition: dict[str, torch.Tensor],
    *,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    step_size: float,
    seed: int | None = None,
) -> torch.Tensor:
    """Research-grade annealed Langevin sampler for score-SDE smoke tests."""
    if noise.ndim != 3:
        raise ValueError(f"noise must be [N, C, L], got shape={tuple(noise.shape)}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if not 0.0 < sigma_min < sigma_max:
        raise ValueError("Require 0 < sigma_min < sigma_max")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=noise.device)
        generator.manual_seed(int(seed))

    x = noise * float(sigma_max)
    sigmas = torch.linspace(float(sigma_max), float(sigma_min), steps=num_steps, device=noise.device)
    for step, sigma in enumerate(sigmas):
        t = torch.full((x.shape[0],), float(sigma.item()), dtype=x.dtype, device=x.device)
        score = model(x, t, condition)
        if score.shape != x.shape:
            raise ValueError(
                f"score shape mismatch at step={step}, sigma={float(sigma):.6g}: "
                f"{tuple(score.shape)} vs {tuple(x.shape)}"
            )
        if not torch.isfinite(score).all():
            raise ValueError(f"score contains NaN/Inf at step={step}, sigma={float(sigma):.6g}")
        randn_kwargs = {"device": x.device, "dtype": x.dtype}
        if generator is not None:
            randn_kwargs["generator"] = generator
        z = torch.randn(x.shape, **randn_kwargs)
        x_next = x + float(step_size) * score + (2.0 * float(step_size)) ** 0.5 * z
        if x_next.shape != x.shape or x_next.dtype != x.dtype or x_next.device != x.device:
            raise ValueError(f"state contract changed at step={step}, sigma={float(sigma):.6g}")
        if not torch.isfinite(x_next).all():
            raise ValueError(f"state contains NaN/Inf at step={step}, sigma={float(sigma):.6g}")
        x = x_next
    return x
