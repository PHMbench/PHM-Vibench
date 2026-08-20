"""Focused counterexamples for Task Factory objective and estimator truth."""

from __future__ import annotations

import pytorch_lightning as pl
import pytest
import torch
import torch.nn as nn
import torchmetrics

from phmfactory.task_semantics import validate_loss_metric_contract
from src.task_factory.Components.loss import (
    compute_task_loss,
    prepare_loss_inputs,
)
from src.task_factory.Components.metrics import (
    get_metrics,
    prepare_metric_inputs,
)
from src.task_factory.Default_task import Default_task


_BINARY_METADATA = {
    1: {"Name": "demo", "Dataset_id": 0, "Label": 0},
    2: {"Name": "demo", "Dataset_id": 0, "Label": 1},
}


def test_regression_loss_preserves_fractional_targets() -> None:
    predictions = torch.tensor([1.75], requires_grad=True)
    targets = torch.tensor([1.75])

    loss = compute_task_loss(nn.MSELoss(), "MSE", predictions, targets)

    assert torch.equal(loss, torch.tensor(0.0))
    assert targets.dtype == torch.float32
    loss.backward()
    assert predictions.grad is not None


def test_class_index_loss_converts_only_integer_valued_targets() -> None:
    logits = torch.tensor(
        [[3.0, -1.0], [-1.0, 3.0]],
        requires_grad=True,
    )
    integer_valued_float = torch.tensor([0.0, 1.0])

    loss = compute_task_loss(
        nn.CrossEntropyLoss(),
        "CE",
        logits,
        integer_valued_float,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    with pytest.raises(ValueError, match="integer class indices"):
        prepare_loss_inputs("CE", logits, torch.tensor([0.0, 0.5]))


def test_auroc_receives_logits_instead_of_argmax_indices() -> None:
    logits = torch.tensor(
        [[2.0, -1.0], [-0.5, 1.5], [0.25, 0.75]],
    )
    targets = torch.tensor([0, 1, 1])

    observed_predictions, observed_targets = prepare_metric_inputs(
        "auroc",
        logits,
        targets,
        loss_name="CE",
    )

    assert torch.equal(observed_predictions, logits)
    assert observed_predictions.shape == (3, 2)
    assert torch.equal(observed_targets, targets)


def _macro_f1(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    metric = torchmetrics.F1Score(
        task="multiclass",
        num_classes=2,
        average="macro",
    )
    return metric(logits, targets)


def test_f1_is_computed_over_the_complete_epoch_population() -> None:
    task = Default_task.__new__(Default_task)
    pl.LightningModule.__init__(task)
    task.loss_name = "CE"
    task.metrics = get_metrics(
        ["f1"],
        _BINARY_METADATA,
        loss_name="CE",
    )

    first_logits = torch.tensor([[4.0, -1.0], [4.0, -1.0]])
    first_targets = torch.tensor([0, 1])
    second_logits = torch.tensor([[-1.0, 4.0], [-1.0, 4.0]])
    second_targets = torch.tensor([1, 1])

    first = task._compute_metrics(
        first_logits,
        first_targets,
        "demo",
        "test",
    )
    second = task._compute_metrics(
        second_logits,
        second_targets,
        "demo",
        "test",
    )

    assert first["test_f1_demo"] is second["test_f1_demo"]
    observed = task.metrics["demo"]["test_f1"].compute()
    expected = _macro_f1(
        torch.cat([first_logits, second_logits], dim=0),
        torch.cat([first_targets, second_targets], dim=0),
    )
    batch_mean = torch.stack(
        [
            _macro_f1(first_logits, first_targets),
            _macro_f1(second_logits, second_targets),
        ]
    ).mean()

    assert torch.allclose(observed, expected)
    assert not torch.allclose(observed, batch_mean)


def test_loss_and_metric_families_cannot_be_mixed() -> None:
    with pytest.raises(ValueError, match="requires regression metrics"):
        validate_loss_metric_contract("MSE", ["acc"])
    with pytest.raises(ValueError, match="requires classification metrics"):
        validate_loss_metric_contract("CE", ["mse"])
    with pytest.raises(ValueError, match="cannot mix classification and regression"):
        validate_loss_metric_contract("CUSTOM", ["f1", "mae"])


def test_two_logit_ce_uses_multiclass_metrics_even_for_two_classes() -> None:
    metrics = get_metrics(
        ["acc", "auroc"],
        _BINARY_METADATA,
        loss_name="CE",
    )
    logits = torch.tensor(
        [[3.0, -1.0], [-1.0, 3.0], [2.0, 0.0], [0.0, 2.0]],
    )
    targets = torch.tensor([0, 1, 0, 1])

    metrics["demo"]["test_acc"].update(logits, targets)
    metrics["demo"]["test_auroc"].update(logits, targets)

    assert torch.isfinite(metrics["demo"]["test_acc"].compute())
    assert torch.isfinite(metrics["demo"]["test_auroc"].compute())


def test_bce_rejects_ambiguous_two_logit_outputs() -> None:
    with pytest.raises(ValueError, match="one logit and one target per sample"):
        prepare_loss_inputs(
            "BCE",
            torch.tensor([[2.0, -2.0], [-1.0, 1.0]]),
            torch.tensor([0.0, 1.0]),
        )
