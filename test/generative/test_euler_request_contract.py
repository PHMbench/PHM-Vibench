"""Analytic tests for the sampler's requested step count and integration interval."""
from __future__ import annotations

import pytest
import torch

from src.task_factory.Components.generative.euler_ode import sample_euler_ode


class ConstantVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.times = []

    def forward(self, state, time, condition):
        self.times.append(time.detach().clone())
        return torch.ones_like(state)


def _condition():
    return {"fault_label": torch.tensor([0]), "domain_id": torch.tensor([0])}


@pytest.mark.parametrize("steps", [1, 2, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_step_count_and_constant_velocity_endpoint(steps, dtype):
    model = ConstantVelocity()
    noise = torch.zeros(2, 2, 16, dtype=dtype)
    result = sample_euler_ode(model, noise, _condition(), steps, t0=0.25, t1=0.75)
    assert len(model.times) == steps
    torch.testing.assert_close(result, torch.full_like(noise, 0.5))
    torch.testing.assert_close(noise, torch.zeros_like(noise))
    expected = 0.25 + torch.arange(steps, dtype=dtype) * (0.5 / steps)
    torch.testing.assert_close(torch.stack(model.times)[:, 0], expected)
    assert model.training is True


@pytest.mark.parametrize("steps", [True, False, 2.5, 2.0, "2"])
def test_noninteger_step_requests_fail_before_model_call(steps):
    model = ConstantVelocity()
    with pytest.raises(TypeError, match="num_steps.*integer"):
        sample_euler_ode(model, torch.zeros(1, 2, 8), _condition(), steps)
    assert model.times == []
    assert model.training is True


@pytest.mark.parametrize("steps", [0, -1])
def test_nonpositive_step_requests_fail_before_model_call(steps):
    model = ConstantVelocity()
    with pytest.raises(ValueError, match="num_steps.*positive"):
        sample_euler_ode(model, torch.zeros(1, 2, 8), _condition(), steps)
    assert model.times == []


@pytest.mark.parametrize("t0,t1", [(0.0, float("inf")), (float("-inf"), 1.0),
                                  (float("nan"), 1.0), (0.0, float("nan"))])
def test_nonfinite_intervals_fail_before_model_call(t0, t1):
    model = ConstantVelocity()
    with pytest.raises(ValueError, match="finite"):
        sample_euler_ode(model, torch.zeros(1, 2, 8), _condition(), 2, t0, t1)
    assert model.times == []
