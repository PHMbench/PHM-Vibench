import pytest
import torch

from src.task_factory.Components.generative.losses.rectified_flow import RectifiedFlowLoss


def test_rectified_flow_zero_loss_for_target_velocity():
    loss_fn = RectifiedFlowLoss()
    x1 = torch.randn(3, 2, 16)
    z = torch.randn_like(x1)
    t = torch.rand(3)
    pred = x1 - z

    out = loss_fn(pred, x1, z, t)

    assert torch.isclose(out["loss"], torch.tensor(0.0))
    assert loss_fn.sample_xt(x1, z, t).shape == x1.shape


def test_rectified_flow_rejects_shape_mismatch():
    loss_fn = RectifiedFlowLoss()
    with pytest.raises(ValueError, match="shape mismatch"):
        loss_fn(torch.randn(2, 2, 8), torch.randn(2, 2, 8), torch.randn(2, 2, 7), torch.rand(2))

