from __future__ import annotations

import torch


def sample(
    model,
    noise: torch.Tensor,
    condition: dict[str, torch.Tensor],
    scheduler,
    num_steps: int,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample DDPM reverse process with a stateless epsilon model."""
    if noise.ndim != 3:
        raise ValueError(f"noise must be [N, C, L], got shape={tuple(noise.shape)}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=noise.device)
        generator.manual_seed(int(seed))

    x = noise
    max_step = scheduler.num_train_timesteps - 1
    timesteps = torch.linspace(max_step, 0, steps=num_steps, device=noise.device).long()
    for step, t_value in enumerate(timesteps):
        t = torch.full((x.shape[0],), int(t_value.item()), device=x.device, dtype=torch.long)
        model_t = t.float() / float(max(max_step, 1))
        pred_epsilon = model(x, model_t, condition)
        if pred_epsilon.shape != x.shape:
            raise ValueError(
                f"epsilon shape mismatch at step={step}, timestep={int(t_value)}: "
                f"{tuple(pred_epsilon.shape)} vs {tuple(x.shape)}"
            )
        if not torch.isfinite(pred_epsilon).all():
            raise ValueError(f"epsilon contains NaN/Inf at step={step}, timestep={int(t_value)}")

        beta = scheduler.beta_at(t, device=x.device, dtype=x.dtype).view(-1, 1, 1)
        alpha = scheduler.alpha_at(t, device=x.device, dtype=x.dtype).view(-1, 1, 1)
        alpha_bar = scheduler.alpha_bar_at(t, device=x.device, dtype=x.dtype).view(-1, 1, 1)
        mean = (x - beta / (1.0 - alpha_bar).sqrt() * pred_epsilon) / alpha.sqrt()
        if int(t_value.item()) > 0:
            randn_kwargs = {"device": x.device, "dtype": x.dtype}
            if generator is not None:
                randn_kwargs["generator"] = generator
            z = torch.randn(x.shape, **randn_kwargs)
            x = mean + beta.sqrt() * z
        else:
            x = mean
        if not torch.isfinite(x).all():
            raise ValueError(f"sample contains NaN/Inf at step={step}, timestep={int(t_value)}")
    return x
