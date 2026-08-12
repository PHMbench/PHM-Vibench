from __future__ import annotations

import csv
import copy
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch
import yaml

from src.model_factory import build_model
from src.model_factory.X_model.P01Alignment import Model
from src.task_factory.Default_task import Default_task


CONFIG_PATH = Path("configs/base/model/p01_alignment.yaml")
SMOKE_CONFIG_PATH = Path("configs/experiments/p01/p01_e0_smoke.yaml")
DUMMY_METADATA_PATH = Path("data/metadata_dummy.csv")


def _namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(
            **{key: _namespace(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _args(condition: str = "M5") -> SimpleNamespace:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["model"]
    payload = copy.deepcopy(payload)
    payload["condition"] = condition
    return _namespace(payload)


def _batch(seed: int = 23) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "x": torch.randn(4, 256, 2, generator=generator),
        "y": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        "file_id": torch.tensor([1, 2, 1, 2], dtype=torch.long),
    }


def _task(model: Model, *, regularization: dict[str, Any] | None = None) -> Default_task:
    args_task = SimpleNamespace(
        loss="CE",
        metrics=[],
        optimizer="adam",
        lr=1.0e-3,
        weight_decay=0.0,
    )
    if regularization is not None:
        args_task.regularization = regularization
    task = Default_task(
        network=model,
        args_data=SimpleNamespace(batch_size=4),
        args_model=_args("M5"),
        args_task=args_task,
        args_trainer=SimpleNamespace(gpus=0, devices=0),
        args_environment=SimpleNamespace(seed=23),
        metadata={
            1: {"Name": "Dummy_Data", "Label": 0},
            2: {"Name": "Dummy_Data", "Label": 1},
        },
    )
    task.eval()
    return task


def _shared_parameters(model: Model) -> list[torch.nn.Parameter]:
    assert model.project_1d is not None
    assert model.project_2d is not None
    return [
        *model.project_1d.parameters(),
        *model.project_2d.parameters(),
    ]


def _gradient_norm(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squared_norm = torch.zeros((), dtype=torch.float64)
    for gradient in gradients:
        if gradient is not None:
            squared_norm = squared_norm + gradient.detach().double().square().sum()
    return float(squared_norm.sqrt().item())


def _controlled_audit(seed: int) -> tuple[dict[str, float], dict[str, float]]:
    torch.manual_seed(seed)
    model = build_model(_args("M5"), metadata=None)
    model.eval()
    batch = _batch(seed)
    _, losses = model.forward_with_alignment(batch["x"], batch["y"])
    config = model.alignment_config
    assert config is not None
    weighted = {
        "physical": config.lambda_p * losses["physical"],
        "semantic": config.lambda_s * losses["semantic"],
        "geometric": config.lambda_g * losses["geometric"],
    }
    parameters = _shared_parameters(model)
    norms = {
        name: _gradient_norm(
            value,
            parameters,
            retain_graph=index < len(weighted) - 1,
        )
        for index, (name, value) in enumerate(weighted.items())
    }
    values = {
        name: float(value.detach().item())
        for name, value in losses.items()
    }
    return values, norms


def test_task_returns_exact_reconstructable_backward_scalar() -> None:
    torch.manual_seed(29)
    task = _task(build_model(_args("M5"), metadata=None))
    batch = _batch(29)
    metrics = task._shared_step(batch, "train")

    expected = (
        metrics["train_classification_loss"]
        + metrics["train_weighted_physical_loss"]
        + metrics["train_weighted_semantic_loss"]
        + metrics["train_weighted_geometric_loss"]
    )
    torch.testing.assert_close(
        metrics["train_total_loss"], expected, rtol=0.0, atol=0.0
    )
    assert metrics["train_loss"] is metrics["train_classification_loss"]

    task._shared_step = lambda *_args, **_kwargs: metrics  # type: ignore[method-assign]
    task._log_metrics = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    assert task.training_step(batch) is metrics["train_total_loss"]


def test_each_enabled_term_has_reproducible_shared_gradient_above_threshold() -> None:
    first_values, first_norms = _controlled_audit(seed=31)
    second_values, second_norms = _controlled_audit(seed=31)
    threshold = float(_args("M5").alignment.gradient_min_norm)

    assert first_values.keys() == second_values.keys()
    for name in first_values:
        assert first_values[name] == pytest.approx(
            second_values[name], rel=0.0, abs=0.0
        )
    for name, norm in first_norms.items():
        assert math.isfinite(norm)
        assert norm > threshold, f"{name} gradient norm {norm} <= {threshold}"
        assert norm == pytest.approx(second_norms[name], rel=0.0, abs=0.0)


@pytest.mark.parametrize("switch", ("a_p", "a_s", "a_g"))
def test_switch_isolation_changes_only_objective_consumption(switch: str) -> None:
    torch.manual_seed(37)
    baseline = build_model(_args("M5"), metadata=None)
    baseline.eval()
    state_dict = copy.deepcopy(baseline.state_dict())
    batch = _batch(37)
    baseline_logits, baseline_losses = baseline.forward_with_alignment(
        batch["x"], batch["y"]
    )
    classification = torch.tensor(2.0)
    baseline_objective = baseline.compose_training_objective(
        classification, baseline_losses
    )

    switched_args = _args("M5")
    setattr(switched_args.alignment, switch, 0)
    torch.manual_seed(37)
    switched = build_model(switched_args, metadata=None)
    for name, value in switched.state_dict().items():
        torch.testing.assert_close(value, state_dict[name], rtol=0.0, atol=0.0)
    switched.eval()
    switched_logits, switched_losses = switched.forward_with_alignment(
        batch["x"], batch["y"]
    )
    switched_objective = switched.compose_training_objective(
        classification, switched_losses
    )

    assert switched.trainable_parameter_count == baseline.trainable_parameter_count
    assert set(switched.state_dict()) == set(baseline.state_dict())
    torch.testing.assert_close(switched_logits, baseline_logits, rtol=0.0, atol=0.0)
    for name in baseline_losses:
        torch.testing.assert_close(
            switched_losses[name], baseline_losses[name], rtol=0.0, atol=0.0
        )

    selected = {
        "a_p": "weighted_physical",
        "a_s": "weighted_semantic",
        "a_g": "weighted_geometric",
    }[switch]
    assert switched_objective[selected].item() == 0.0
    torch.testing.assert_close(
        baseline_objective["total"] - switched_objective["total"],
        baseline_objective[selected],
        rtol=1.0e-5,
        atol=1.0e-7,
    )
    for name in (
        "weighted_physical",
        "weighted_semantic",
        "weighted_geometric",
    ):
        if name != selected:
            torch.testing.assert_close(
                switched_objective[name],
                baseline_objective[name],
                rtol=0.0,
                atol=0.0,
            )


def test_semantic_targets_exclude_same_class_nonpaired_samples() -> None:
    model = build_model(_args("M5"), metadata=None)
    labels = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    masks = model.semantic_pair_masks(labels)

    assert masks["positive"].diagonal().all()
    assert not masks["negative"].diagonal().any()
    assert not masks["admissible"][0, 1]
    assert not masks["admissible"][1, 0]
    assert masks["negative"][0, 2]
    assert masks["negative"][0, 3]


@pytest.mark.parametrize(
    "labels, message",
    (
        (torch.tensor([[0, 1]]), "shape"),
        (torch.tensor([0.0, 1.0]), "integer dtype"),
        (torch.tensor([0, 4]), "within"),
        (torch.tensor([1, 1, 1]), "different-class negative"),
    ),
)
def test_invalid_semantic_labels_fail_closed(
    labels: torch.Tensor, message: str
) -> None:
    model = build_model(_args("M5"), metadata=None)
    with pytest.raises(ValueError, match=message):
        model.semantic_pair_masks(labels)


def test_physical_components_and_pre_log_parseval_audit_are_explicit() -> None:
    torch.manual_seed(41)
    model = build_model(_args("M5"), metadata=None)
    model.eval()
    batch = _batch(41)
    _, losses = model.forward_with_alignment(batch["x"], batch["y"])
    config = model.alignment_config
    assert config is not None

    assert model.renderer_identity() == {
        "n_fft": 128,
        "hop_length": 32,
        "win_length": 128,
        "window": "hann",
        "window_periodic": True,
        "center": True,
        "pad_mode": "reflect",
        "normalized": False,
        "onesided": True,
        "representation": "magnitude",
        "scaling": "log1p",
        "resize": "none",
        "normalization": "none",
    }
    for name in (
        "physical",
        "physical_energy",
        "physical_spectral",
        "physical_parseval",
    ):
        assert losses[name].ndim == 0
        assert torch.isfinite(losses[name])
    expected = (
        config.physical_energy_weight * losses["physical_energy"]
        + config.physical_spectral_weight * losses["physical_spectral"]
        + config.physical_parseval_weight * losses["physical_parseval"]
    )
    torch.testing.assert_close(losses["physical"], expected, rtol=0.0, atol=0.0)
    assert losses["physical_parseval"].item() < 1.0e-10


def test_geometric_term_is_differentiable_and_rejects_degenerate_batches() -> None:
    torch.manual_seed(43)
    model = build_model(_args("M5"), metadata=None)
    first = torch.randn(4, 32, requires_grad=True)
    second = torch.randn(4, 32, requires_grad=True)
    loss = model._geometric_alignment(first, second)
    gradients = torch.autograd.grad(loss, (first, second))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(torch.linalg.vector_norm(gradient).item() > 0.0 for gradient in gradients)

    with pytest.raises(ValueError, match="B >= 3"):
        model._geometric_alignment(first[:2], second[:2])


def test_invalid_transform_and_objective_semantics_fail_closed() -> None:
    invalid_transform = _args("M5")
    invalid_transform.renderer.win_length = 64
    model = build_model(invalid_transform, metadata=None)
    batch = _batch(47)
    with pytest.raises(ValueError, match="win_length == renderer.n_fft"):
        model.forward_with_alignment(batch["x"], batch["y"])

    invalid_switch = _args("M5")
    invalid_switch.alignment.a_p = True
    with pytest.raises(RuntimeError, match="integer 0 or 1"):
        build_model(invalid_switch, metadata=None)

    missing_weight = _args("M5")
    del missing_weight.alignment.lambda_s
    with pytest.raises(RuntimeError, match="alignment.lambda_s is required"):
        build_model(missing_weight, metadata=None)


def test_validation_does_not_consume_labels_in_alignment_objective() -> None:
    model = build_model(_args("M5"), metadata=None)
    task = _task(model)
    model.forward_with_alignment = lambda *_args, **_kwargs: pytest.fail(  # type: ignore[method-assign]
        "validation must not invoke the training-only alignment objective"
    )
    metrics = task._shared_step(_batch(53), "val")
    assert "val_total_loss" in metrics
    assert "val_weighted_semantic_loss" not in metrics


def test_generic_regularization_cannot_change_the_m5_objective() -> None:
    model = build_model(_args("M5"), metadata=None)
    task = _task(
        model,
        regularization={"regularization": {"l2": 1.0e-4}},
    )
    with pytest.raises(ValueError, match="forbids generic regularization"):
        task._shared_step(_batch(59), "train")


def test_smoke_model_class_count_matches_dummy_metadata() -> None:
    smoke = yaml.safe_load(SMOKE_CONFIG_PATH.read_text(encoding="utf-8"))
    with DUMMY_METADATA_PATH.open(encoding="utf-8", newline="") as handle:
        labels = {int(row["Label"]) for row in csv.DictReader(handle)}

    expected_labels = set(range(int(smoke["model"]["num_classes"])))
    assert labels == expected_labels
