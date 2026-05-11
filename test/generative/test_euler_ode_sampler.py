from __future__ import annotations

import pytest
import torch

from src.task_factory.Components.generative.samplers.euler_ode import sample_euler_ode


class _ConstantVelocity(torch.nn.Module):
    def __init__(self, value: float = 1.0, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.value = value
        self.dtype = dtype

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: dict[str, torch.Tensor]) -> torch.Tensor:
        out = torch.full_like(x, self.value)
        if self.dtype is not None:
            out = out.to(self.dtype)
        return out


def _condition(n: int = 2) -> dict[str, torch.Tensor]:
    return {
        "fault_label": torch.zeros(n, dtype=torch.long),
        "domain_id": torch.zeros(n, dtype=torch.long),
    }


def test_euler_sampler_returns_finite_state() -> None:
    noise = torch.zeros(2, 1, 4)

    samples = sample_euler_ode(_ConstantVelocity(1.0), noise, _condition(), num_steps=4)

    assert samples.shape == noise.shape
    assert samples.dtype == noise.dtype
    assert samples.device == noise.device
    assert torch.isfinite(samples).all()


def test_euler_sampler_rejects_state_overflow_after_update() -> None:
    noise = torch.zeros(2, 1, 4, dtype=torch.float32)

    with pytest.raises(ValueError, match="state contains NaN/Inf after update"):
        sample_euler_ode(
            _ConstantVelocity(3.0e38),
            noise,
            _condition(),
            num_steps=1,
            t0=0.0,
            t1=2.0,
        )


def test_euler_sampler_rejects_state_dtype_change_after_update() -> None:
    noise = torch.zeros(2, 1, 4, dtype=torch.float32)

    with pytest.raises(ValueError, match="state dtype changed"):
        sample_euler_ode(
            _ConstantVelocity(1.0, dtype=torch.float64),
            noise,
            _condition(),
            num_steps=1,
        )


def test_euler_sampler_rejects_velocity_shape_mismatch() -> None:
    class BadShape(torch.nn.Module):
        def forward(self, x: torch.Tensor, t: torch.Tensor, condition: dict[str, torch.Tensor]) -> torch.Tensor:
            return x[:, :, :-1]

    with pytest.raises(ValueError, match="velocity shape mismatch"):
        sample_euler_ode(BadShape(), torch.zeros(2, 1, 4), _condition(), num_steps=1)
