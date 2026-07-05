from __future__ import annotations

from typing import Any

import torch


def _check_condition_batch(condition: dict[str, torch.Tensor], batch_size: int) -> None:
    for key, value in condition.items():
        if not torch.is_tensor(value):
            raise ValueError(f"condition {key} must be a tensor")
        if value.view(-1).numel() != batch_size:
            raise ValueError(
                f"condition {key} batch size mismatch: expected {batch_size}, "
                f"got {value.view(-1).numel()}"
            )


def sample_one_step_map(
    model: Any,
    noise: torch.Tensor,
    condition: dict[str, torch.Tensor],
) -> torch.Tensor:
    if noise.ndim != 3:
        raise ValueError(f"noise must be [N, C, L], got shape={tuple(noise.shape)}")
    _check_condition_batch(condition, noise.shape[0])
    t = torch.zeros(noise.shape[0], dtype=noise.dtype, device=noise.device)
    sample = model(noise, t, condition)
    if sample.shape != noise.shape:
        raise ValueError(f"sample shape mismatch: {tuple(sample.shape)} vs {tuple(noise.shape)}")
    if not torch.isfinite(sample).all():
        raise ValueError(
            "sample contains NaN/Inf; "
            f"shape={tuple(sample.shape)}, dtype={sample.dtype}, device={sample.device}"
        )
    return sample
