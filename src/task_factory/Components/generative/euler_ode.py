from __future__ import annotations

import torch


def _normalized_condition(
    condition: dict[str, torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not isinstance(condition, dict):
        raise ValueError("condition must be a dict")
    normalized: dict[str, torch.Tensor] = {}
    for key in ("fault_label", "domain_id"):
        if key not in condition:
            raise ValueError(f"condition missing required key: {key}")
        value = torch.as_tensor(
            condition[key],
            device=device,
            dtype=torch.long,
        ).reshape(-1)
        if value.numel() == 1 and batch_size > 1:
            value = value.repeat(batch_size)
        if value.numel() != batch_size:
            raise ValueError(
                f"condition {key} must contain 1 or {batch_size} values, "
                f"got {value.numel()}"
            )
        normalized[key] = value
    return normalized


@torch.no_grad()
def sample_euler_ode(
    model,
    noise: torch.Tensor,
    condition: dict[str, torch.Tensor],
    num_steps: int,
    t0: float = 0.0,
    t1: float = 1.0,
) -> torch.Tensor:
    """Integrate a stateless velocity field using explicit Euler steps."""

    if noise.ndim != 3:
        raise ValueError(f"noise must be [N,C,L], got {tuple(noise.shape)}")
    if not torch.is_floating_point(noise):
        raise ValueError(f"noise must be floating point, got {noise.dtype}")
    if not torch.isfinite(noise).all():
        raise ValueError("noise contains NaN/Inf")
    if int(num_steps) <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if not float(t1) > float(t0):
        raise ValueError(f"t1 must be greater than t0, got {t0} and {t1}")

    condition = _normalized_condition(
        condition,
        batch_size=noise.shape[0],
        device=noise.device,
    )
    state = noise
    expected_shape = state.shape
    expected_dtype = state.dtype
    expected_device = state.device
    dt = (float(t1) - float(t0)) / float(num_steps)

    training_state = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()
    try:
        for step in range(int(num_steps)):
            t_value = float(t0) + step * dt
            t = torch.full(
                (state.shape[0],),
                t_value,
                dtype=state.dtype,
                device=state.device,
            )
            velocity = model(state, t, condition)
            if velocity.shape != expected_shape:
                raise ValueError(
                    f"velocity shape mismatch at step={step}: "
                    f"{tuple(velocity.shape)} vs {tuple(expected_shape)}"
                )
            if velocity.dtype != expected_dtype:
                raise ValueError(
                    f"velocity dtype changed at step={step}: "
                    f"{velocity.dtype} vs {expected_dtype}"
                )
            if velocity.device != expected_device:
                raise ValueError(
                    f"velocity device changed at step={step}: "
                    f"{velocity.device} vs {expected_device}"
                )
            if not torch.isfinite(velocity).all():
                raise ValueError(
                    f"velocity contains NaN/Inf at step={step}, t={t_value}"
                )

            next_state = state + dt * velocity
            if next_state.shape != expected_shape:
                raise ValueError(
                    f"state shape changed at step={step}: "
                    f"{tuple(next_state.shape)} vs {tuple(expected_shape)}"
                )
            if next_state.dtype != expected_dtype:
                raise ValueError(
                    f"state dtype changed at step={step}: "
                    f"{next_state.dtype} vs {expected_dtype}"
                )
            if next_state.device != expected_device:
                raise ValueError(
                    f"state device changed at step={step}: "
                    f"{next_state.device} vs {expected_device}"
                )
            if not torch.isfinite(next_state).all():
                raise ValueError(
                    f"state contains NaN/Inf at step={step}, t={t_value}"
                )
            state = next_state
    finally:
        if training_state is not None and hasattr(model, "train"):
            model.train(bool(training_state))

    return state
