from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.model_factory.generative_model.condition_encoder import ConditionEncoder
from src.model_factory.generative_model.phm_cfm_mlp1d import Model
from src.task_factory.Components.generative import (
    ConditionalFlowMatchingLoss,
    sample_euler_ode,
)
from src.task_factory.task.generative.conditional_flow_matching import (
    ConditionalFlowMatchingTask,
)


METADATA = {
    1: {"Label": 0, "Domain_id": 0},
    2: {"Label": 1, "Domain_id": 1},
}


def _model_args() -> SimpleNamespace:
    return SimpleNamespace(
        type="generative_model",
        name="phm_cfm_mlp1d",
        in_channels=2,
        hidden_dim=16,
        condition_dim=8,
        num_fault_classes=2,
        num_domains=2,
    )


def _task_args() -> SimpleNamespace:
    return SimpleNamespace(
        type="generative",
        name="conditional_flow_matching",
        lr=1e-4,
        weight_decay=1e-4,
        optimizer="adamw",
        t_eps=1e-3,
    )


def _task() -> ConditionalFlowMatchingTask:
    model_args = _model_args()
    task = ConditionalFlowMatchingTask(
        network=Model(model_args, METADATA),
        args_data=SimpleNamespace(normalization="standardization"),
        args_model=model_args,
        args_task=_task_args(),
        args_trainer=SimpleNamespace(),
        args_environment=SimpleNamespace(),
        metadata=METADATA,
    )
    task.log = lambda *args, **kwargs: None
    return task


def test_condition_encoder_infers_metadata_cardinality() -> None:
    encoder = ConditionEncoder(metadata=METADATA, embedding_dim=4)
    encoded = encoder(
        {
            "fault_label": torch.tensor([0, 1]),
            "domain_id": torch.tensor([1, 0]),
        },
        torch.tensor([0.25, 0.75]),
    )

    assert encoder.num_fault_classes == 2
    assert encoder.num_domains == 2
    assert encoded.shape == (2, 4)
    assert torch.isfinite(encoded).all()


def test_condition_encoder_rejects_invalid_contracts() -> None:
    encoder = ConditionEncoder(
        metadata=METADATA,
        embedding_dim=4,
        num_fault_classes=2,
        num_domains=2,
    )

    with pytest.raises(ValueError, match="missing required key"):
        encoder(
            {"fault_label": torch.tensor([0])},
            torch.tensor([0.5]),
        )

    with pytest.raises(ValueError, match="exceeds configured embedding"):
        encoder(
            {
                "fault_label": torch.tensor([2]),
                "domain_id": torch.tensor([0]),
            },
            torch.tensor([0.5]),
        )


def test_cfm_model_preserves_shape_and_finite_values() -> None:
    model = Model(_model_args(), METADATA)
    x_t = torch.randn(3, 2, 32)
    velocity = model(
        x_t,
        torch.tensor([0.1, 0.5, 0.9]),
        {
            "fault_label": torch.tensor([0, 1, 0]),
            "domain_id": torch.tensor([0, 1, 1]),
        },
    )

    assert velocity.shape == x_t.shape
    assert velocity.dtype == x_t.dtype
    assert velocity.device == x_t.device
    assert torch.isfinite(velocity).all()


def test_cfm_interpolation_and_zero_loss_for_exact_velocity() -> None:
    loss_fn = ConditionalFlowMatchingLoss(eps=0.0)
    x1 = torch.tensor([[[2.0, 4.0]]])
    noise = torch.tensor([[[0.0, 2.0]]])
    t = torch.tensor([0.25])

    x_t = loss_fn.sample_xt(x1, noise, t)
    target = loss_fn.target_velocity(x1, noise)
    result = loss_fn(target, x1, noise, t)

    assert torch.allclose(x_t, torch.tensor([[[0.5, 2.5]]]))
    assert torch.allclose(target, torch.tensor([[[2.0, 2.0]]]))
    assert torch.allclose(result["loss"], torch.zeros(()))


def test_cfm_loss_rejects_nonfinite_prediction() -> None:
    loss_fn = ConditionalFlowMatchingLoss()
    x1 = torch.zeros(1, 2, 4)
    noise = torch.zeros_like(x1)
    prediction = torch.full_like(x1, float("nan"))

    with pytest.raises(ValueError, match="NaN/Inf"):
        loss_fn(prediction, x1, noise, torch.tensor([0.5]))


class _ConstantVelocity(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    def forward(self, x, t, condition):
        return torch.full_like(x, self.value)


def test_euler_sampler_integrates_velocity_and_restores_mode() -> None:
    model = _ConstantVelocity(0.5)
    model.train()
    noise = torch.zeros(2, 1, 4)

    samples = sample_euler_ode(
        model,
        noise,
        {
            "fault_label": torch.tensor([0, 1]),
            "domain_id": torch.tensor([0, 1]),
        },
        num_steps=4,
    )

    assert model.training is True
    assert samples.shape == noise.shape
    assert torch.allclose(samples, torch.full_like(samples, 0.5))


def test_euler_sampler_rejects_invalid_steps() -> None:
    with pytest.raises(ValueError, match="num_steps must be positive"):
        sample_euler_ode(
            _ConstantVelocity(0.0),
            torch.zeros(1, 1, 4),
            {
                "fault_label": torch.tensor([0]),
                "domain_id": torch.tensor([0]),
            },
            num_steps=0,
        )


def test_cfm_task_training_step_uses_file_condition_provenance() -> None:
    task = _task()
    batch = {
        "x": torch.randn(2, 32, 2),
        "y": torch.tensor([0, 1]),
        "file_id": torch.tensor([1, 2]),
    }

    loss = task.training_step(batch)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_cfm_task_rejects_missing_file_id() -> None:
    task = _task()
    with pytest.raises(ValueError, match="file_id"):
        task.training_step(
            {
                "x": torch.randn(2, 32, 2),
                "y": torch.tensor([0, 1]),
            }
        )


def test_cfm_task_sample_contract() -> None:
    task = _task()
    samples = task.sample(
        {
            "fault_label": torch.tensor([0]),
            "domain_id": torch.tensor([1]),
        },
        num_samples=2,
        length=16,
        channels=2,
        num_steps=2,
        device="cpu",
    )

    assert samples.shape == (2, 2, 16)
    assert torch.isfinite(samples).all()
