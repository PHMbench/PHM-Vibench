import pytest
import torch

from src.task_factory.Components.generative.losses.flow_matching import (
    ConditionalFlowMatchingLoss,
)


def test_flow_matching_loss_returns_scalar_velocity_mse():
    loss_fn = ConditionalFlowMatchingLoss(eps=0.01)
    x1 = torch.randn(4, 2, 16)
    z = torch.randn_like(x1)
    t = loss_fn.sample_t(4, x1.device)
    x_t = loss_fn.sample_xt(x1, z, t)
    pred_velocity = x1 - z

    out = loss_fn(pred_velocity, x1, z, t)

    assert x_t.shape == x1.shape
    assert out["loss"].ndim == 0
    assert torch.isclose(out["loss"], torch.tensor(0.0))


def test_flow_matching_loss_rejects_shape_mismatch():
    loss_fn = ConditionalFlowMatchingLoss()
    x1 = torch.randn(2, 2, 8)
    z = torch.randn(2, 2, 7)

    with pytest.raises(ValueError, match="shape mismatch"):
        loss_fn.sample_xt(x1, z, torch.rand(2))

