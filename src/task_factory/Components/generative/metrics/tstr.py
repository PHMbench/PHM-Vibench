from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def tstr_placeholder() -> dict[str, float]:
    """Numeric placeholder used when labels are unavailable."""
    return {
        "tstr_nearest_centroid_accuracy": float("nan"),
        "trts_nearest_centroid_accuracy": float("nan"),
        "utility_classifier_tstr_accuracy": float("nan"),
        "utility_classifier_trts_accuracy": float("nan"),
        "utility_classifier_real_only_accuracy": float("nan"),
        "utility_classifier_real_plus_synth_accuracy": float("nan"),
        "utility_classifier_real_plus_synth_gain": float("nan"),
        "tstr_accuracy": float("nan"),
        "trts_accuracy": float("nan"),
        "tstr_status_code": 0.0,
    }


def _nearest_centroid_accuracy(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
) -> float:
    labels = torch.unique(train_y)
    if labels.numel() < 1 or test_x.numel() == 0:
        return float("nan")
    centroids = []
    valid_labels = []
    for label in labels:
        mask = train_y == label
        if bool(mask.any()):
            centroids.append(train_x[mask].mean(dim=0))
            valid_labels.append(label)
    if not centroids:
        return float("nan")
    centroid_tensor = torch.stack(centroids, dim=0)
    label_tensor = torch.stack(valid_labels).to(test_y.device)
    pred = label_tensor[torch.cdist(test_x, centroid_tensor).argmin(dim=1)]
    return float((pred == test_y).float().mean().cpu())


def _standardize(train_x: torch.Tensor, test_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (train_x - mean) / std, (test_x - mean) / std


def _linear_probe_accuracy(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    epochs: int = 80,
    lr: float = 0.1,
) -> float:
    train_labels = torch.unique(train_y)
    if train_labels.numel() < 2 or test_x.numel() == 0:
        return float("nan")
    if train_x.shape[0] != train_y.numel() or test_x.shape[0] != test_y.numel():
        return float("nan")
    train_x, test_x = _standardize(train_x.float(), test_x.float())
    classes = train_labels.sort().values
    class_to_index = {int(label.item()): index for index, label in enumerate(classes)}
    mapped_train = torch.tensor(
        [class_to_index[int(label)] for label in train_y.tolist()],
        dtype=torch.long,
        device=train_x.device,
    )
    weights = torch.zeros(
        train_x.shape[1],
        classes.numel(),
        dtype=train_x.dtype,
        device=train_x.device,
        requires_grad=True,
    )
    bias = torch.zeros(
        classes.numel(), dtype=train_x.dtype, device=train_x.device, requires_grad=True
    )
    for _ in range(int(epochs)):
        logits = train_x @ weights + bias
        loss = F.cross_entropy(logits, mapped_train)
        grad_w, grad_b = torch.autograd.grad(loss, (weights, bias))
        with torch.no_grad():
            weights -= float(lr) * grad_w
            bias -= float(lr) * grad_b
    with torch.no_grad():
        pred_index = (test_x @ weights + bias).argmax(dim=1)
        pred = classes[pred_index].to(test_y.device)
        return float((pred == test_y).float().mean().cpu())


def _real_holdout_indices(num_items: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.arange(num_items, device=device)
    train_mask = indices % 2 == 0
    test_mask = ~train_mask
    return indices[train_mask], indices[test_mask]


def _classifier_utility_metrics(
    real_x: torch.Tensor,
    fake_x: torch.Tensor,
    real_y: torch.Tensor,
    fake_y: torch.Tensor,
) -> dict[str, float]:
    tstr = _linear_probe_accuracy(fake_x, fake_y, real_x, real_y)
    trts = _linear_probe_accuracy(real_x, real_y, fake_x, fake_y)
    train_idx, test_idx = _real_holdout_indices(real_x.shape[0], real_x.device)
    if train_idx.numel() < 2 or test_idx.numel() < 1:
        real_only = float("nan")
        real_plus_synth = float("nan")
    else:
        real_only = _linear_probe_accuracy(
            real_x[train_idx],
            real_y[train_idx],
            real_x[test_idx],
            real_y[test_idx],
        )
        real_plus_synth = _linear_probe_accuracy(
            torch.cat([real_x[train_idx], fake_x], dim=0),
            torch.cat([real_y[train_idx], fake_y], dim=0),
            real_x[test_idx],
            real_y[test_idx],
        )
    gain = (
        real_plus_synth - real_only
        if math.isfinite(real_only) and math.isfinite(real_plus_synth)
        else float("nan")
    )
    return {
        "utility_classifier_tstr_accuracy": tstr,
        "utility_classifier_trts_accuracy": trts,
        "utility_classifier_real_only_accuracy": real_only,
        "utility_classifier_real_plus_synth_accuracy": real_plus_synth,
        "utility_classifier_real_plus_synth_gain": gain,
    }


def tstr_metrics(
    real: torch.Tensor,
    fake: torch.Tensor,
    *,
    real_labels: torch.Tensor | None = None,
    fake_labels: torch.Tensor | None = None,
) -> dict[str, float]:
    """Lightweight downstream utility via nearest-centroid probes.

    TSTR trains class centroids on synthetic samples and tests on real samples.
    TRTS trains class centroids on real samples and tests on synthetic samples.
    """
    if real_labels is None or fake_labels is None:
        return tstr_placeholder()
    with torch.no_grad():
        real_x = real.float().reshape(real.shape[0], -1)
        fake_x = fake.float().reshape(fake.shape[0], -1)
        real_y = torch.as_tensor(real_labels, device=real_x.device).long().view(-1)
        fake_y = torch.as_tensor(fake_labels, device=fake_x.device).long().view(-1)
        n_real = min(real_x.shape[0], real_y.numel())
        n_fake = min(fake_x.shape[0], fake_y.numel())
        real_x, real_y = real_x[:n_real], real_y[:n_real]
        fake_x, fake_y = fake_x[:n_fake], fake_y[:n_fake]
        tstr_accuracy = _nearest_centroid_accuracy(fake_x, fake_y, real_x, real_y)
        trts_accuracy = _nearest_centroid_accuracy(real_x, real_y, fake_x, fake_y)
        metrics = {
            "tstr_nearest_centroid_accuracy": tstr_accuracy,
            "trts_nearest_centroid_accuracy": trts_accuracy,
            "tstr_accuracy": tstr_accuracy,
            "trts_accuracy": trts_accuracy,
            "tstr_accuracy_deprecated_reason": (
                "deprecated alias; use tstr_nearest_centroid_accuracy"
            ),
            "trts_accuracy_deprecated_reason": (
                "deprecated alias; use trts_nearest_centroid_accuracy"
            ),
            "tstr_status_code": 1.0,
        }
        with torch.enable_grad():
            metrics.update(_classifier_utility_metrics(real_x, fake_x, real_y, fake_y))
        return metrics
