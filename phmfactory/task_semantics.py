"""Pure task-semantic validation shared by config and runtime boundaries.

This module deliberately has no Torch or Lightning import.  It only decides whether a
configured loss and metric set describe one coherent mathematical problem.  Tensor
shape, dtype, and value checks remain next to the loss and metric implementations that
consume them.
"""

from __future__ import annotations

from collections.abc import Sequence


CLASSIFICATION_INDEX_LOSSES = frozenset({"CE", "NLL"})
BINARY_CLASSIFICATION_LOSSES = frozenset({"BCE"})
REGRESSION_LOSSES = frozenset({"MSE", "MAE"})

CLASSIFICATION_METRICS = frozenset(
    {"acc", "f1", "precision", "recall", "auroc"}
)
REGRESSION_METRICS = frozenset({"mse", "mae", "r2", "mape"})
KNOWN_METRICS = CLASSIFICATION_METRICS | REGRESSION_METRICS


def normalize_loss_name(loss_name: object) -> str:
    """Return one non-empty uppercase loss identifier."""

    if not isinstance(loss_name, str) or not loss_name.strip():
        raise TypeError("task.loss must be a non-empty string")
    return loss_name.strip().upper()


def normalize_metric_names(metric_names: object) -> tuple[str, ...]:
    """Return explicit, unique, known metric identifiers in configured order."""

    if isinstance(metric_names, (str, bytes)) or not isinstance(
        metric_names, Sequence
    ):
        raise TypeError("task.metrics must be a non-empty sequence of metric names")

    normalized: list[str] = []
    for index, raw_name in enumerate(metric_names):
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise TypeError(
                f"task.metrics[{index}] must be a non-empty string, got {raw_name!r}"
            )
        normalized.append(raw_name.strip().lower())

    if not normalized:
        raise ValueError("task.metrics must contain at least one metric")
    if len(set(normalized)) != len(normalized):
        duplicates = sorted(
            name for name in set(normalized) if normalized.count(name) > 1
        )
        raise ValueError(f"task.metrics contains duplicate metric(s): {duplicates}")

    unknown = sorted(set(normalized) - KNOWN_METRICS)
    if unknown:
        available = ", ".join(sorted(KNOWN_METRICS))
        raise ValueError(
            f"Unknown task metric(s): {unknown}. Available metrics: {available}. "
            "PHMFactory does not silently skip requested metrics."
        )
    return tuple(normalized)


def loss_family(loss_name: object) -> str:
    """Return the declared prediction/target family for one loss."""

    normalized = normalize_loss_name(loss_name)
    if normalized in CLASSIFICATION_INDEX_LOSSES:
        return "multiclass"
    if normalized in BINARY_CLASSIFICATION_LOSSES:
        return "binary"
    if normalized in REGRESSION_LOSSES:
        return "regression"
    return "custom"


def validate_loss_metric_contract(
    loss_name: object,
    metric_names: object,
) -> tuple[str, ...]:
    """Reject configurations that mix incompatible estimands.

    Classification and regression metrics cannot share one generic output tensor because
    they require different target dtypes and prediction representations.  Known losses
    additionally constrain the metric family.  Custom losses may use either one family,
    but never both in the same task.
    """

    family = loss_family(loss_name)
    normalized = normalize_metric_names(metric_names)
    requested = set(normalized)
    has_classification = bool(requested & CLASSIFICATION_METRICS)
    has_regression = bool(requested & REGRESSION_METRICS)

    if has_classification and has_regression:
        raise ValueError(
            "task.metrics cannot mix classification and regression estimators in one "
            f"generic task: metrics={list(normalized)}"
        )
    if family in {"multiclass", "binary"} and has_regression:
        raise ValueError(
            f"task.loss={normalize_loss_name(loss_name)} requires classification "
            f"metrics, got {list(normalized)}"
        )
    if family == "regression" and has_classification:
        raise ValueError(
            f"task.loss={normalize_loss_name(loss_name)} requires regression metrics, "
            f"got {list(normalized)}"
        )
    return normalized


__all__ = [
    "BINARY_CLASSIFICATION_LOSSES",
    "CLASSIFICATION_INDEX_LOSSES",
    "CLASSIFICATION_METRICS",
    "KNOWN_METRICS",
    "REGRESSION_LOSSES",
    "REGRESSION_METRICS",
    "loss_family",
    "normalize_loss_name",
    "normalize_metric_names",
    "validate_loss_metric_contract",
]
