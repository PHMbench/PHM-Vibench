"""Utilities for building and updating maintained evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torchmetrics

from phmfactory.task_semantics import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    loss_family,
    normalize_loss_name,
    normalize_metric_names,
    validate_loss_metric_contract,
)
from src.utils.label_ontology import (
    metadata_rows,
    validate_metadata_label_ontology,
)

from .loss import prepare_loss_inputs


_CLASSIFICATION_METRIC_CLASSES = {
    "acc": torchmetrics.Accuracy,
    "f1": torchmetrics.F1Score,
    "precision": torchmetrics.Precision,
    "recall": torchmetrics.Recall,
    "auroc": torchmetrics.AUROC,
}

_REGRESSION_METRIC_CLASSES = {
    "mse": torchmetrics.MeanSquaredError,
    "mae": torchmetrics.MeanAbsoluteError,
    "r2": torchmetrics.R2Score,
    "mape": torchmetrics.MeanAbsolutePercentageError,
}


def _classification_metric(
    metric_name: str,
    num_classes: int,
    *,
    loss_name: str | None,
):
    """Build a binary or multiclass metric that matches the declared output."""

    if num_classes < 2:
        raise ValueError(
            "classification metrics require at least two classes, "
            f"but the validated ontology has K={num_classes}"
        )

    family = loss_family(loss_name) if loss_name is not None else "legacy"
    if family == "binary":
        if num_classes != 2:
            raise ValueError(
                "task.loss=BCE requires a two-class metadata ontology for the "
                f"maintained binary metric contract, got K={num_classes}"
            )
        return _CLASSIFICATION_METRIC_CLASSES[metric_name](task="binary")

    # CE/NLL models expose one logit per class, including the K=2 case.  Those
    # outputs therefore use multiclass metrics rather than silently discarding one
    # of the two logits through a binary threshold interface.
    use_multiclass = family == "multiclass" or num_classes > 2
    if not use_multiclass:
        return _CLASSIFICATION_METRIC_CLASSES[metric_name](task="binary")

    kwargs: dict[str, Any] = {
        "task": "multiclass",
        "num_classes": num_classes,
    }
    if metric_name in {"f1", "precision", "recall", "auroc"}:
        kwargs["average"] = "macro"
    return _CLASSIFICATION_METRIC_CLASSES[metric_name](**kwargs)


def prepare_metric_inputs(
    metric_name: str,
    predictions: Any,
    target: Any,
    *,
    loss_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the prediction representation required by one estimator.

    Classification metrics receive logits or binary scores directly.  AUROC therefore
    never receives class indices produced by ``argmax``.  Regression metrics receive the
    continuous prediction and target tensors used by regression losses.
    """

    normalized_metric = normalize_metric_names([metric_name])[0]
    normalized_loss = normalize_loss_name(loss_name)
    family = loss_family(normalized_loss)

    if normalized_metric in REGRESSION_METRICS:
        if family in {"multiclass", "binary"}:
            raise ValueError(
                f"metric {normalized_metric!r} is incompatible with "
                f"task.loss={normalized_loss}"
            )
        return prepare_loss_inputs("MSE", predictions, target)

    if normalized_metric not in CLASSIFICATION_METRICS:
        raise ValueError(f"Unknown task metric {metric_name!r}")
    if family == "regression":
        raise ValueError(
            f"metric {normalized_metric!r} is incompatible with "
            f"task.loss={normalized_loss}"
        )
    if family in {"multiclass", "binary"}:
        return prepare_loss_inputs(normalized_loss, predictions, target)

    # A custom loss may still expose one explicit classification representation.
    # Infer only from an unambiguous output shape; do not squeeze or threshold a
    # general tensor to force compatibility.
    if torch.is_tensor(predictions) and predictions.ndim == 2 and predictions.shape[1] >= 2:
        return prepare_loss_inputs("CE", predictions, target)
    return prepare_loss_inputs("BCE", predictions, target)


def get_metrics(
    metric_names: Sequence[str],
    metadata: Any,
    *,
    loss_name: str | None = None,
) -> nn.ModuleDict:
    """Build every requested metric or fail at the Task Factory boundary.

    ``loss_name`` is optional for low-level construction compatibility.  Maintained Task
    construction always supplies it, which rejects classification/regression mixtures
    before training begins and selects the metric interface matching the model output.
    """

    normalized_names = (
        validate_loss_metric_contract(loss_name, metric_names)
        if loss_name is not None
        else normalize_metric_names(metric_names)
    )

    rows = metadata_rows(metadata)
    dataset_names: list[Any] = []
    for index, row in enumerate(rows):
        if "Name" not in row:
            raise KeyError(f"metadata row {index} is missing required field 'Name'")
        if row["Name"] not in dataset_names:
            dataset_names.append(row["Name"])

    classification_requested = any(
        name in CLASSIFICATION_METRICS for name in normalized_names
    )
    class_counts = (
        validate_metadata_label_ontology(
            metadata,
            group_field="Name",
            require_labels=True,
        )
        if classification_requested
        else {}
    )

    metrics = nn.ModuleDict()
    for raw_data_name in dataset_names:
        data_name = str(raw_data_name)
        data_metrics = nn.ModuleDict()
        for stage in ("train", "val", "test"):
            for metric_name in normalized_names:
                key = f"{stage}_{metric_name}"
                if metric_name in CLASSIFICATION_METRICS:
                    data_metrics[key] = _classification_metric(
                        metric_name,
                        class_counts[raw_data_name],
                        loss_name=loss_name,
                    )
                else:
                    data_metrics[key] = _REGRESSION_METRIC_CLASSES[metric_name]()
        metrics[data_name] = data_metrics

    return metrics


__all__ = ["get_metrics", "prepare_metric_inputs"]
