from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.task_factory.Components.gradient_constraints import FisherGradientConstraint
from src.task_factory.Default_task import Default_task


def test_fic_scales_squared_gradient_norm_to_epsilon():
    parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([3.0, 4.0])
    constraint = FisherGradientConstraint(epsilon=4.0)

    result = constraint.apply([parameter])

    assert result.norm.item() == pytest.approx(25.0)
    assert result.scale.item() == pytest.approx(0.4)
    torch.testing.assert_close(parameter.grad, torch.tensor([1.2, 1.6]))


def test_fic_leaves_small_gradients_unchanged_and_rejects_nonfinite():
    parameter = nn.Parameter(torch.ones(2))
    parameter.grad = torch.tensor([0.5, 0.5])
    result = FisherGradientConstraint(epsilon=2.0).apply([parameter])
    assert result.scale.item() == 1.0
    torch.testing.assert_close(parameter.grad, torch.tensor([0.5, 0.5]))

    parameter.grad = torch.tensor([float("nan"), 0.0])
    with pytest.raises(FloatingPointError, match="non-finite"):
        FisherGradientConstraint().apply([parameter])


def test_default_task_applies_fic_in_optimizer_hook(monkeypatch):
    task = Default_task(
        network=nn.Linear(2, 2, bias=False),
        args_data=SimpleNamespace(),
        args_model=SimpleNamespace(),
        args_task=SimpleNamespace(
            loss="CE",
            metrics=[],
            gradient_constraint={"name": "fic", "epsilon": 1.0},
        ),
        args_trainer=SimpleNamespace(gpus=0),
        args_environment=SimpleNamespace(),
        metadata={1: {"Name": "Dummy_Data", "Label": 1}},
    )
    for parameter in task.network.parameters():
        parameter.grad = torch.full_like(parameter, 2.0)
    logged = {}
    monkeypatch.setattr(task, "log", lambda name, value, **kwargs: logged.setdefault(name, value))

    task.on_before_optimizer_step(None)

    assert logged["train_fic_norm"].item() == pytest.approx(16.0)
    assert logged["train_fic_scale"].item() == pytest.approx(0.25)
    for parameter in task.network.parameters():
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 0.5))


def test_default_task_rejects_fic_with_non_ce_loss():
    with pytest.raises(ValueError, match="requires task.loss=CE"):
        Default_task(
            network=nn.Linear(2, 2),
            args_data=SimpleNamespace(),
            args_model=SimpleNamespace(),
            args_task=SimpleNamespace(
                loss="MSE",
                metrics=[],
                gradient_constraint={"name": "fic", "epsilon": 1.0},
            ),
            args_trainer=SimpleNamespace(gpus=0),
            args_environment=SimpleNamespace(),
            metadata={1: {"Name": "Dummy_Data", "Label": 1}},
        )
