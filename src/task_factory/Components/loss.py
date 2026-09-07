"""Common loss utilities used across PHM-Vibench tasks."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from phmfactory.task_semantics import (
    BINARY_CLASSIFICATION_LOSSES,
    CLASSIFICATION_INDEX_LOSSES,
    REGRESSION_LOSSES,
    normalize_loss_name,
)

from .contrastive_losses import (
    BarlowTwinsLoss,
    InfoNCELoss,
    PrototypicalLoss,
    SupConLoss,
    TripletLoss,
    VICRegLoss,
)
from .metric_loss import MatchingLoss
from .prediction_loss import *  # noqa: F403 - historical prediction losses


def get_loss_fn(loss_name: str) -> nn.Module:
    """Return the explicitly configured loss implementation."""

    key = normalize_loss_name(loss_name)
    loss_mapping = {
        "CE": nn.CrossEntropyLoss(),
        "MSE": nn.MSELoss(),
        "MAE": nn.L1Loss(),
        "BCE": nn.BCEWithLogitsLoss(),
        "NLL": nn.NLLLoss(),
        "MATCHING": MatchingLoss,
        "SIGNAL_MASK_LOSS": Signal_mask_Loss,  # noqa: F405
        "INFONCE": InfoNCELoss(),
        "TRIPLET": TripletLoss(),
        "SUPCON": SupConLoss(),
        "PROTOTYPICAL": PrototypicalLoss(),
        "BARLOWTWINS": BarlowTwinsLoss(),
        "VICREG": VICRegLoss(),
    }
    if key not in loss_mapping:
        available = ", ".join(sorted(loss_mapping))
        raise ValueError(
            f"Unsupported task loss {loss_name!r}. Available losses: {available}."
        )
    return loss_mapping[key]


def _require_tensor(value: Any, *, context: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{context} must be a torch.Tensor, got {type(value).__name__}")
    return value


def _require_real_finite_predictions(
    predictions: Any,
    *,
    context: str,
) -> torch.Tensor:
    tensor = _require_tensor(predictions, context=context)
    if not torch.is_floating_point(tensor) or torch.is_complex(tensor):
        raise TypeError(
            f"{context} must contain real floating-point values, got {tensor.dtype}"
        )
    if tensor.numel() == 0:
        raise ValueError(f"{context} must be non-empty")
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{context} contains NaN or Inf")
    return tensor


def _require_real_finite_target(target: Any, *, context: str) -> torch.Tensor:
    tensor = _require_tensor(target, context=context)
    if tensor.dtype == torch.bool or torch.is_complex(tensor):
        raise TypeError(f"{context} must contain real numeric values, got {tensor.dtype}")
    if tensor.numel() == 0:
        raise ValueError(f"{context} must be non-empty")
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{context} contains NaN or Inf")
    return tensor


def _scalar_target_vector(target: torch.Tensor, *, batch_size: int, context: str) -> torch.Tensor:
    if target.ndim == 0 and batch_size == 1:
        target = target.reshape(1)
    elif target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 1:
        raise ValueError(
            f"{context} must have shape [B] or [B,1], got {tuple(target.shape)}"
        )
    if target.shape[0] != batch_size:
        raise ValueError(
            f"{context} batch size mismatch: predictions={batch_size}, "
            f"target={target.shape[0]}"
        )
    return target


def _prepare_multiclass_inputs(
    predictions: Any,
    target: Any,
    *,
    loss_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = _require_real_finite_predictions(
        predictions,
        context=f"task.loss={loss_name} predictions",
    )
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError(
            f"task.loss={loss_name} predictions must have shape [B,C] with C>=2, "
            f"got {tuple(logits.shape)}"
        )

    labels = _require_real_finite_target(
        target,
        context=f"task.loss={loss_name} target",
    )
    labels = _scalar_target_vector(
        labels,
        batch_size=logits.shape[0],
        context=f"task.loss={loss_name} target",
    )
    if torch.is_floating_point(labels) and not torch.equal(labels, labels.round()):
        raise ValueError(
            f"task.loss={loss_name} target must contain integer class indices"
        )
    labels = labels.to(device=logits.device, dtype=torch.long)
    minimum = int(labels.min().item())
    maximum = int(labels.max().item())
    if minimum < 0 or maximum >= logits.shape[1]:
        raise ValueError(
            f"task.loss={loss_name} target is outside the logits class range: "
            f"observed=[{minimum}, {maximum}], expected=[0, {logits.shape[1] - 1}]"
        )
    return logits, labels


def _prepare_binary_inputs(
    predictions: Any,
    target: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = _require_real_finite_predictions(
        predictions,
        context="task.loss=BCE predictions",
    )
    labels = _require_real_finite_target(
        target,
        context="task.loss=BCE target",
    )

    if logits.ndim == 0:
        logits = logits.reshape(1)
    elif logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if labels.ndim == 0:
        labels = labels.reshape(1)
    elif labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]

    if logits.ndim != 1 or labels.ndim != 1 or logits.shape != labels.shape:
        raise ValueError(
            "task.loss=BCE requires one logit and one target per sample with matching "
            f"shape [B] or [B,1], got predictions={tuple(logits.shape)}, "
            f"target={tuple(labels.shape)}"
        )
    if (labels < 0).any() or (labels > 1).any():
        raise ValueError("task.loss=BCE target values must lie in [0, 1]")
    return logits, labels.to(device=logits.device, dtype=logits.dtype)


def _prepare_regression_inputs(
    predictions: Any,
    target: Any,
    *,
    loss_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    estimates = _require_real_finite_predictions(
        predictions,
        context=f"task.loss={loss_name} predictions",
    )
    values = _require_real_finite_target(
        target,
        context=f"task.loss={loss_name} target",
    )

    if estimates.ndim == 0:
        estimates = estimates.reshape(1)
    if values.ndim == 0:
        values = values.reshape(1)
    if estimates.ndim == 2 and estimates.shape[1] == 1 and values.ndim == 1:
        estimates = estimates[:, 0]
    elif values.ndim == 2 and values.shape[1] == 1 and estimates.ndim == 1:
        values = values[:, 0]

    if estimates.shape != values.shape:
        raise ValueError(
            f"task.loss={loss_name} requires matching prediction and target shapes, "
            f"got predictions={tuple(estimates.shape)}, target={tuple(values.shape)}"
        )
    return estimates, values.to(device=estimates.device, dtype=estimates.dtype)


def prepare_loss_inputs(
    loss_name: str,
    predictions: Any,
    target: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and prepare the exact tensors consumed by one declared loss.

    Integer conversion occurs only for class-index losses.  Regression and binary
    targets retain their numeric values, and PHMFactory never reshapes a general tensor
    to force compatibility.
    """

    normalized = normalize_loss_name(loss_name)
    if normalized in CLASSIFICATION_INDEX_LOSSES:
        return _prepare_multiclass_inputs(
            predictions,
            target,
            loss_name=normalized,
        )
    if normalized in BINARY_CLASSIFICATION_LOSSES:
        return _prepare_binary_inputs(predictions, target)
    if normalized in REGRESSION_LOSSES:
        return _prepare_regression_inputs(
            predictions,
            target,
            loss_name=normalized,
        )

    checked_predictions = _require_real_finite_predictions(
        predictions,
        context=f"task.loss={normalized} predictions",
    )
    checked_target = _require_real_finite_target(
        target,
        context=f"task.loss={normalized} target",
    )
    return checked_predictions, checked_target


def compute_task_loss(
    loss_fn: Any,
    loss_name: str,
    predictions: Any,
    target: Any,
) -> torch.Tensor:
    """Compute one finite scalar objective from its declared tensor contract."""

    prepared_predictions, prepared_target = prepare_loss_inputs(
        loss_name,
        predictions,
        target,
    )
    result = loss_fn(prepared_predictions, prepared_target)
    if not torch.is_tensor(result):
        raise TypeError(
            f"task.loss={normalize_loss_name(loss_name)} must return a torch.Tensor, "
            f"got {type(result).__name__}"
        )
    if result.ndim != 0:
        raise ValueError(
            f"task.loss={normalize_loss_name(loss_name)} must return one scalar, "
            f"got shape {tuple(result.shape)}"
        )
    if not torch.isfinite(result):
        raise FloatingPointError(
            f"task.loss={normalize_loss_name(loss_name)} returned NaN or Inf"
        )
    return result


__all__ = [
    "compute_task_loss",
    "get_loss_fn",
    "prepare_loss_inputs",
]
