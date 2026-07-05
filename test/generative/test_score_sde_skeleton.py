import pytest
import torch

from src.task_factory.Components.generative.losses.score_sde import ScoreSDEResearchLoss


def test_score_sde_skeleton_shape_and_status():
    loss_fn = ScoreSDEResearchLoss()
    target = torch.randn(2, 2, 16)
    out = loss_fn(target, target)

    assert torch.isclose(out["loss"], torch.tensor(0.0))
    assert out["status"] == "research-only"
    assert loss_fn.condition_keys == ("fault_label", "domain_id")


def test_score_sde_skeleton_rejects_shape_mismatch():
    loss_fn = ScoreSDEResearchLoss()
    with pytest.raises(ValueError, match="shape mismatch"):
        loss_fn(torch.randn(2, 2, 16), torch.randn(2, 2, 15))

