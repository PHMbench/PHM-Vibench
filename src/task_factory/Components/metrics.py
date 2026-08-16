"""Utilities for building common evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch.nn as nn
import torchmetrics

from ...utils.label_ontology import (
    metadata_rows,
    validate_metadata_label_ontology,
)


_CLASSIFICATION_METRICS = {
    "acc": torchmetrics.Accuracy,
    "f1": torchmetrics.F1Score,
    "precision": torchmetrics.Precision,
    "recall": torchmetrics.Recall,
    "auroc": torchmetrics.AUROC,
}

_REGRESSION_METRICS = {
    "mse": torchmetrics.MeanSquaredError,
    "mae": torchmetrics.MeanAbsoluteError,
    "r2": torchmetrics.R2Score,
    "mape": torchmetrics.MeanAbsolutePercentageError,
}

_METRIC_CLASSES = {**_CLASSIFICATION_METRICS, **_REGRESSION_METRICS}


def _classification_metric(metric_name: str, num_classes: int):
    if num_classes < 2:
        raise ValueError(
            "classification metrics require at least two classes, "
            f"but the validated ontology has K={num_classes}"
        )
    metric_class = _CLASSIFICATION_METRICS[metric_name]
    if num_classes == 2:
        return metric_class(task="binary")
    return metric_class(task="multiclass", num_classes=num_classes)


def get_metrics(metric_names: Sequence[str], metadata: Any) -> nn.ModuleDict:
    """Build every requested metric or fail at the Task Factory boundary."""

    if isinstance(metric_names, (str, bytes)) or not isinstance(
        metric_names, Sequence
    ):
        raise TypeError("task.metrics must be a non-empty sequence of metric names")
    normalized_names = [str(name).strip().lower() for name in metric_names]
    if not normalized_names or any(not name for name in normalized_names):
        raise ValueError("task.metrics must contain at least one non-empty metric name")

    unknown = sorted(set(normalized_names) - set(_METRIC_CLASSES))
    if unknown:
        available = ", ".join(sorted(_METRIC_CLASSES))
        raise ValueError(
            f"Unknown task metric(s): {unknown}. Available metrics: {available}. "
            "PHMFactory does not silently skip requested metrics."
        )

    rows = metadata_rows(metadata)
    dataset_names: list[Any] = []
    for index, row in enumerate(rows):
        if "Name" not in row:
            raise KeyError(f"metadata row {index} is missing required field 'Name'")
        if row["Name"] not in dataset_names:
            dataset_names.append(row["Name"])

    classification_requested = any(
        name in _CLASSIFICATION_METRICS for name in normalized_names
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
                if metric_name in _CLASSIFICATION_METRICS:
                    data_metrics[key] = _classification_metric(
                        metric_name,
                        class_counts[raw_data_name],
                    )
                else:
                    data_metrics[key] = _REGRESSION_METRICS[metric_name]()
        metrics[data_name] = data_metrics

    return metrics


if __name__ == "__main__":
    dummy_meta = {
        1: {"Name": "ds", "Dataset_id": 1, "Label": 0},
        2: {"Name": "ds", "Dataset_id": 1, "Label": 1},
    }
    built = get_metrics(["acc", "f1"], dummy_meta)
    print("Built metrics:", list(built["ds"].keys()))
