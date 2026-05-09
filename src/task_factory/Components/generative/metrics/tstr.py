from __future__ import annotations

import torch


def tstr_placeholder() -> dict[str, float]:
    """Numeric placeholder used when labels are unavailable."""
    return {"tstr_accuracy": float("nan"), "tstr_status_code": 0.0}


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
        return {
            "tstr_accuracy": _nearest_centroid_accuracy(fake_x, fake_y, real_x, real_y),
            "trts_accuracy": _nearest_centroid_accuracy(real_x, real_y, fake_x, fake_y),
            "tstr_status_code": 1.0,
        }
