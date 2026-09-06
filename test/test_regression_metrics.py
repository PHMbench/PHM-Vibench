"""Focused tests for classification and regression metric construction."""

import pytest
import torch
import torchmetrics

from src.task_factory.Components.metrics import get_metrics


_CLASSIFICATION_METADATA = {
    1: {"Name": "dataset1", "Dataset_id": 1, "Label": 0},
    2: {"Name": "dataset1", "Dataset_id": 1, "Label": 1},
    3: {"Name": "dataset1", "Dataset_id": 1, "Label": 2},
}

_MIXED_METADATA = {
    **_CLASSIFICATION_METADATA,
    4: {"Name": "dataset2", "Dataset_id": 2, "Label": 0},
    5: {"Name": "dataset2", "Dataset_id": 2, "Label": 1},
}


def test_classification_metrics_are_built_for_every_stage():
    names = ["acc", "f1", "precision", "recall", "auroc"]
    metrics = get_metrics(names, _CLASSIFICATION_METADATA)

    for stage in ("train", "val", "test"):
        for name in names:
            assert f"{stage}_{name}" in metrics["dataset1"]

    predictions = torch.tensor([0, 1, 2, 1])
    targets = torch.tensor([0, 1, 2, 2])
    value = metrics["dataset1"]["train_acc"](predictions, targets)
    assert 0 <= value <= 1


def test_regression_metrics_do_not_require_classification_parameters():
    names = ["mse", "mae", "r2", "mape"]
    metrics = get_metrics(names, _CLASSIFICATION_METADATA)

    assert isinstance(
        metrics["dataset1"]["train_mse"],
        torchmetrics.MeanSquaredError,
    )
    assert isinstance(
        metrics["dataset1"]["train_mae"],
        torchmetrics.MeanAbsoluteError,
    )
    assert isinstance(metrics["dataset1"]["train_r2"], torchmetrics.R2Score)
    assert isinstance(
        metrics["dataset1"]["train_mape"],
        torchmetrics.MeanAbsolutePercentageError,
    )

    predictions = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.1, 2.2, 2.9])
    assert metrics["dataset1"]["train_mse"](predictions, targets) >= 0
    assert metrics["dataset1"]["train_mae"](predictions, targets) >= 0


def test_mixed_metric_families_and_multiple_datasets_are_explicit():
    names = ["acc", "mse", "mae"]
    metrics = get_metrics(names, _MIXED_METADATA)

    assert set(metrics.keys()) == {"dataset1", "dataset2"}
    for dataset_name in metrics:
        for stage in ("train", "val", "test"):
            for name in names:
                assert f"{stage}_{name}" in metrics[dataset_name]


def test_binary_and_multiclass_ontologies_select_matching_metric_modes():
    binary_metadata = {
        1: {"Name": "binary", "Dataset_id": 1, "Label": 0},
        2: {"Name": "binary", "Dataset_id": 1, "Label": 1},
    }
    multiclass_metrics = get_metrics(["acc", "f1"], _CLASSIFICATION_METADATA)
    binary_metrics = get_metrics(["acc", "f1"], binary_metadata)

    assert "train_acc" in binary_metrics["binary"]
    assert "train_acc" in multiclass_metrics["dataset1"]


def test_unknown_metric_fails_instead_of_being_skipped():
    with pytest.raises(ValueError, match="Unknown task metric"):
        get_metrics(["acc", "unsupported_metric", "mse"], _CLASSIFICATION_METADATA)


def test_empty_metadata_and_invalid_ontologies_fail_closed():
    with pytest.raises(ValueError, match="at least one row"):
        get_metrics(["acc"], {})

    nonzero = {
        1: {"Name": "bad", "Dataset_id": 1, "Label": 1},
        2: {"Name": "bad", "Dataset_id": 1, "Label": 2},
    }
    with pytest.raises(ValueError, match="zero-based and contiguous"):
        get_metrics(["acc"], nonzero)

    gapped = {
        1: {"Name": "bad", "Dataset_id": 1, "Label": 0},
        2: {"Name": "bad", "Dataset_id": 1, "Label": 2},
    }
    with pytest.raises(ValueError, match="zero-based and contiguous"):
        get_metrics(["acc"], gapped)
