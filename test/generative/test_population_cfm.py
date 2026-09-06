from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from src.config_schema import ExperimentConfig
from src.model_factory.generative_model.phm_cfm_mlp1d import Model
from src.task_factory.Components.generative import (
    PopulationCorrelationMMD,
    evaluate_smoke_metrics,
    multi_rbf_mmd,
    pearson_correlation_vectors,
)
from src.task_factory.task.generative.conditional_flow_matching import (
    ConditionalFlowMatchingTask,
)


METADATA = {
    1: {"Label": 0, "Domain_id": 0},
    2: {"Label": 1, "Domain_id": 1},
}


def test_population_features_and_mmd_contract() -> None:
    real = torch.randn(4, 3, 32)
    fake = torch.randn(4, 3, 32, requires_grad=True)
    real_features = pearson_correlation_vectors(real)
    fake_features = pearson_correlation_vectors(fake)

    assert real_features.shape == (4, 3)
    forward = multi_rbf_mmd(real_features, fake_features, [0.1, 0.5, 1.0])
    reverse = multi_rbf_mmd(fake_features, real_features, [0.1, 0.5, 1.0])
    assert forward.item() >= 0.0
    torch.testing.assert_close(forward, reverse)
    forward.backward()
    assert fake.grad is not None
    assert torch.isfinite(fake.grad).all()


def test_population_mmd_is_zero_for_identical_windows() -> None:
    windows = torch.randn(4, 2, 16)
    value = PopulationCorrelationMMD([0.1, 0.5, 1.0, 2.0])(windows, windows)
    assert value.item() == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("shape", [(1, 2, 16), (2, 1, 16), (2, 2, 1)])
def test_population_features_reject_insufficient_population(shape) -> None:
    with pytest.raises(ValueError, match="requires"):
        pearson_correlation_vectors(torch.randn(*shape))


class _RecordingModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.last_t = None

    def forward(self, x, t, condition):
        self.last_t = t.detach().clone()
        return self.model(x, t, condition)


def _task(*, enabled: bool) -> tuple[ConditionalFlowMatchingTask, _RecordingModel]:
    model_args = SimpleNamespace(
        type="generative_model",
        name="phm_cfm_mlp1d",
        in_channels=2,
        hidden_dim=16,
        condition_dim=8,
        num_fault_classes=2,
        num_domains=2,
    )
    recording = _RecordingModel(Model(model_args, METADATA))
    task_args = SimpleNamespace(
        type="generative",
        name="conditional_flow_matching",
        lr=1e-4,
        weight_decay=1e-4,
        optimizer="adamw",
        t_eps=1e-3,
        population_regularization=SimpleNamespace(
            enabled=enabled,
            weight=0.1,
            dependency="pearson_correlation",
            estimator="biased",
            rbf_bandwidths=[0.1, 0.5, 1.0, 2.0],
            same_time_per_batch=True,
        ),
    )
    task = ConditionalFlowMatchingTask(
        network=recording,
        args_data=SimpleNamespace(normalization="standardization"),
        args_model=model_args,
        args_task=task_args,
        args_trainer=SimpleNamespace(),
        args_environment=SimpleNamespace(),
        metadata=METADATA,
    )
    return task, recording


def _batch() -> dict:
    return {
        "x": torch.randn(3, 32, 2),
        "y": torch.tensor([0, 1, 0]),
        "file_id": torch.tensor([1, 2, 1]),
    }


def test_population_cfm_uses_shared_time_and_backpropagates() -> None:
    task, recording = _task(enabled=True)
    logged = {}
    task.log = lambda name, value, **kwargs: logged.setdefault(name, value)
    loss = task.training_step(_batch())

    assert task.method_id == "population_aware_cfm"
    assert torch.unique(recording.last_t).numel() == 1
    assert "train_population_correlation_mmd" in logged
    assert torch.isfinite(loss)
    loss.backward()
    assert next(recording.model.parameters()).grad is not None


def test_disabled_population_path_preserves_baseline_contract() -> None:
    task, recording = _task(enabled=False)
    task.log = lambda *args, **kwargs: None
    loss = task.training_step(_batch())

    assert task.method_id == "conditional_flow_matching"
    assert recording.last_t.numel() == 3
    assert torch.isfinite(loss)


def test_population_metric_is_optional_but_structured() -> None:
    real = torch.randn(3, 2, 16)
    fake = torch.randn(3, 2, 16)
    metrics = evaluate_smoke_metrics(real, fake)

    assert metrics["population_dependency_mmd"]["status"] == "ok"
    assert "population_dependency_mmd" in metrics["summary"]["optional"]


def test_population_metric_uses_requested_bandwidths() -> None:
    torch.manual_seed(7)
    real = torch.randn(3, 2, 16)
    fake = torch.randn(3, 2, 16)
    expected = PopulationCorrelationMMD([7.0])(real, fake).item()

    metrics = evaluate_smoke_metrics(
        real,
        fake,
        population_rbf_bandwidths=[7.0],
    )

    assert metrics["population_dependency_mmd"]["value"] == pytest.approx(expected)


def test_population_config_rejects_single_sample_batches() -> None:
    config = {
        "pipeline": "Pipeline_06_Generative_Modeling",
        "environment": {
            "project": "population-contract",
            "output_dir": "results/test-population-contract",
        },
        "data": {
            "data_dir": "data",
            "metadata_file": "metadata_dummy.csv",
            "batch_size": 1,
        },
        "model": {
            "type": "generative_model",
            "name": "phm_cfm_mlp1d",
            "in_channels": 2,
        },
        "task": {
            "type": "generative",
            "name": "conditional_flow_matching",
            "population_regularization": {"enabled": True},
        },
        "trainer": {"name": "Default_trainer"},
    }

    with pytest.raises(ValidationError, match="data.batch_size >= 2"):
        ExperimentConfig.model_validate(config)
