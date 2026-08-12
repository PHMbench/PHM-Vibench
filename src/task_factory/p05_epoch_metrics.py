"""Strict weighted training semantics for the P05 evidence path."""

from __future__ import annotations

import torch
import torch.nn as nn


def _validated_weights(
    sample_weight: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a finite non-negative float64 weight vector with positive mass."""

    if sample_weight is None:
        raise KeyError("P05 evidence mode requires sample_weight")
    weights = torch.as_tensor(sample_weight, device=device)
    if weights.ndim != 1 or int(weights.shape[0]) != int(batch_size):
        raise ValueError(
            "sample_weight must have shape "
            f"({batch_size},), got {tuple(weights.shape)}"
        )
    weights = weights.detach().to(dtype=torch.float64)
    if not torch.isfinite(weights).all():
        raise FloatingPointError("sample_weight contains non-finite values")
    if (weights < 0).any():
        raise ValueError("sample_weight must be non-negative")
    if not bool(weights.sum() > 0):
        raise ValueError("sample_weight must have a positive total")
    return weights


def weighted_mean_loss(
    per_sample_loss: torch.Tensor,
    sample_weight: torch.Tensor | None,
) -> torch.Tensor:
    """Compute exactly ``sum(weight * loss) / sum(weight)`` without broadcasting."""

    if per_sample_loss.ndim != 1:
        raise ValueError(
            "P05 evidence loss must be unreduced with shape (batch,), "
            f"got {tuple(per_sample_loss.shape)}"
        )
    if not torch.isfinite(per_sample_loss).all():
        raise FloatingPointError("per-sample loss contains non-finite values")
    weights64 = _validated_weights(
        sample_weight,
        batch_size=int(per_sample_loss.shape[0]),
        device=per_sample_loss.device,
    )
    weights = weights64.to(dtype=per_sample_loss.dtype)
    return (weights * per_sample_loss).sum() / weights.sum()


def _class_indices(values: torch.Tensor, *, name: str) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError(f"{name} must have shape (batch,), got {tuple(values.shape)}")
    if values.dtype.is_floating_point:
        if not torch.isfinite(values).all():
            raise FloatingPointError(f"{name} contains non-finite values")
        if not torch.equal(values, values.round()):
            raise ValueError(f"{name} must contain integer class indices")
    return values.detach().to(dtype=torch.long)


class WeightedEpochConfusionMatrix(nn.Module):
    """Accumulate one float64 weighted confusion matrix for an entire epoch."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if isinstance(num_classes, bool) or int(num_classes) != num_classes or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2")
        self.num_classes = int(num_classes)
        self.register_buffer(
            "matrix",
            torch.zeros((self.num_classes, self.num_classes), dtype=torch.float64),
            persistent=False,
        )

    @torch.no_grad()
    def reset(self) -> None:
        self.matrix.zero_()

    @torch.no_grad()
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: torch.Tensor | None,
    ) -> None:
        predictions = _class_indices(predictions, name="predictions").to(self.matrix.device)
        targets = _class_indices(targets, name="targets").to(self.matrix.device)
        if predictions.shape != targets.shape:
            raise ValueError("predictions and targets must have identical shapes")
        if predictions.numel() == 0:
            raise ValueError("cannot update confusion matrix with an empty batch")
        if (predictions < 0).any() or (predictions >= self.num_classes).any():
            raise ValueError("predictions contain an out-of-range class index")
        if (targets < 0).any() or (targets >= self.num_classes).any():
            raise ValueError("targets contain an out-of-range class index")

        weights = _validated_weights(
            sample_weight,
            batch_size=int(targets.shape[0]),
            device=self.matrix.device,
        )
        flat_indices = targets * self.num_classes + predictions
        batch_matrix = torch.bincount(
            flat_indices,
            weights=weights,
            minlength=self.num_classes * self.num_classes,
        ).reshape(self.num_classes, self.num_classes)
        self.matrix.add_(batch_matrix)

    @torch.no_grad()
    def compute_macro_f1(self) -> torch.Tensor:
        """Return macro-F1 with absent-class divisions defined as zero."""

        if not bool(self.matrix.sum() > 0):
            raise RuntimeError("cannot compute macro-F1 from an empty confusion matrix")
        true_positive = self.matrix.diagonal()
        denominator = self.matrix.sum(dim=1) + self.matrix.sum(dim=0)
        per_class_f1 = torch.zeros_like(true_positive)
        valid = denominator > 0
        per_class_f1[valid] = 2.0 * true_positive[valid] / denominator[valid]
        return per_class_f1.mean()

    @torch.no_grad()
    def compute_accuracy(self) -> torch.Tensor:
        """Return exact weighted accuracy from the complete epoch matrix."""

        total = self.matrix.sum()
        if not bool(total > 0):
            raise RuntimeError("cannot compute accuracy from an empty confusion matrix")
        return self.matrix.diagonal().sum() / total


class WeightedEpochLoss(nn.Module):
    """Accumulate ``sum(w*loss) / sum(w)`` across all epoch batches."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "numerator",
            torch.zeros((), dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "denominator",
            torch.zeros((), dtype=torch.float64),
            persistent=False,
        )

    @torch.no_grad()
    def reset(self) -> None:
        self.numerator.zero_()
        self.denominator.zero_()

    @torch.no_grad()
    def update(
        self,
        per_sample_loss: torch.Tensor,
        sample_weight: torch.Tensor | None,
    ) -> None:
        if per_sample_loss.ndim != 1 or per_sample_loss.numel() == 0:
            raise ValueError("epoch loss requires a non-empty per-sample vector")
        losses = per_sample_loss.detach().to(
            device=self.numerator.device,
            dtype=torch.float64,
        )
        if not torch.isfinite(losses).all():
            raise FloatingPointError("per-sample loss contains non-finite values")
        weights = _validated_weights(
            sample_weight,
            batch_size=int(losses.shape[0]),
            device=self.numerator.device,
        )
        self.numerator.add_((weights * losses).sum(dtype=torch.float64))
        self.denominator.add_(weights.sum(dtype=torch.float64))

    @torch.no_grad()
    def compute(self) -> torch.Tensor:
        if not bool(self.denominator > 0):
            raise RuntimeError("cannot compute loss from an empty epoch accumulator")
        return self.numerator / self.denominator
