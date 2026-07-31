from __future__ import annotations

import copy
import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

import src.Pipeline_06_Generative_Modeling as pipeline06
from src.config_schema import ExperimentConfig, PopulationRegularizationConfig
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
    torch.manual_seed(0)
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


def test_population_features_reject_nonfinite_values() -> None:
    windows = torch.randn(2, 2, 16)
    windows[0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="NaN/Inf"):
        pearson_correlation_vectors(windows)


@pytest.mark.parametrize("bandwidths", [[], [0.0], [float("inf")]])
def test_population_mmd_rejects_invalid_bandwidths(bandwidths) -> None:
    with pytest.raises(ValueError, match="bandwidths"):
        PopulationCorrelationMMD(bandwidths)


class _RecordingModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.last_t = None

    def forward(self, x, t, condition):
        self.last_t = t.detach().clone()
        return self.model(x, t, condition)


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


def _task(
    population_enabled: bool,
    *,
    population_weight: float = 0.1,
) -> tuple[ConditionalFlowMatchingTask, _RecordingModel]:
    model_args = _model_args()
    recording = _RecordingModel(Model(model_args, METADATA))
    task_args = SimpleNamespace(
        type="generative",
        name="conditional_flow_matching",
        lr=1e-4,
        weight_decay=1e-4,
        optimizer="adamw",
        t_eps=1e-3,
        population_regularization=SimpleNamespace(
            enabled=population_enabled,
            weight=population_weight,
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


def test_population_cfm_uses_shared_time_and_combined_loss() -> None:
    task, recording = _task(True)
    logged = {}
    task.log = lambda name, value, **kwargs: logged.setdefault(name, value)
    loss = task.training_step(
        {
            "x": torch.randn(3, 32, 2),
            "y": torch.tensor([0, 1, 0]),
            "file_id": torch.tensor([1, 2, 1]),
        }
    )

    assert task.method_id == "population_aware_cfm"
    assert task.loss_id == "conditional_flow_matching+population_correlation_mmd"
    assert torch.unique(recording.last_t).numel() == 1
    torch.testing.assert_close(
        loss,
        logged["train_mse_v"]
        + task.population_weight * logged["train_population_correlation_mmd"],
    )
    loss.backward()
    assert next(recording.model.parameters()).grad is not None


def test_disabled_population_keeps_baseline_task_contract() -> None:
    task, _ = _task(False)

    assert task.method_id == "conditional_flow_matching"
    assert task.loss_id == "conditional_flow_matching"
    assert task.population_loss is None


@pytest.mark.parametrize("weight", [0.0, float("nan"), float("inf")])
def test_population_task_rejects_invalid_runtime_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        _task(True, population_weight=weight)


def test_population_metric_is_absent_for_baseline() -> None:
    metrics = evaluate_smoke_metrics(torch.randn(3, 2, 16), torch.randn(3, 2, 16))

    assert "population_dependency_mmd" not in metrics
    assert "required_for_method" not in metrics["summary"]


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
    assert metrics["summary"]["required_for_method"] == ["population_dependency_mmd"]
    assert metrics["summary"]["method_required_ok"] is True


def test_pipeline_population_bandwidths_are_enabled_only() -> None:
    disabled = SimpleNamespace(population_regularization=SimpleNamespace(enabled=False))
    enabled = {
        "population_regularization": {
            "enabled": True,
            "rbf_bandwidths": [0.25, 0.75],
        }
    }

    assert pipeline06._population_rbf_bandwidths(disabled) is None
    assert pipeline06._population_rbf_bandwidths(enabled) == (0.25, 0.75)


def test_population_metric_is_written_to_csv(tmp_path: Path) -> None:
    metrics = evaluate_smoke_metrics(
        torch.randn(3, 2, 16),
        torch.randn(3, 2, 16),
        population_rbf_bandwidths=[0.5],
    )
    path = tmp_path / "metrics.csv"

    pipeline06._write_metrics_csv(path, metrics)

    with path.open(encoding="utf-8", newline="") as handle:
        names = [row["metric"] for row in csv.DictReader(handle)]
    assert names[-1] == "population_dependency_mmd"


def _experiment_config() -> dict:
    return {
        "pipeline": "Pipeline_06_Generative_Modeling",
        "environment": {
            "project": "population-contract",
            "output_dir": "results/test-population-contract",
        },
        "data": {
            "data_dir": "data",
            "metadata_file": "metadata_dummy.csv",
            "batch_size": 2,
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


def test_population_config_accepts_valid_contract() -> None:
    validated = ExperimentConfig.model_validate(_experiment_config())

    assert validated.task.population_regularization.enabled is True


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("data", "batch_size", 1, "data.batch_size >= 2"),
        ("model", "in_channels", 1, "model.in_channels >= 2"),
        ("task", "name", "other", "only by conditional_flow_matching"),
    ],
)
def test_population_config_rejects_invalid_coupling(
    section: str,
    field: str,
    value,
    message: str,
) -> None:
    config = copy.deepcopy(_experiment_config())
    config[section][field] = value

    with pytest.raises(ValidationError, match=message):
        ExperimentConfig.model_validate(config)


def test_population_config_rejects_invalid_numeric_contract() -> None:
    with pytest.raises(ValidationError, match="finite and positive"):
        PopulationRegularizationConfig(rbf_bandwidths=[float("inf")])
