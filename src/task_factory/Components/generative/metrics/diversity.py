from __future__ import annotations

import torch


def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.float().reshape(x.shape[0], -1)


def _knn_radius(distances: torch.Tensor, k: int) -> torch.Tensor:
    k = min(max(int(k), 1), max(distances.shape[1] - 1, 1))
    return torch.topk(distances, k=k + 1, dim=1, largest=False).values[:, -1]


def _prdc(real: torch.Tensor, fake: torch.Tensor, k: int = 3) -> dict[str, float]:
    if real.shape[0] < 2 or fake.shape[0] < 2:
        return {
            "diversity_prdc_precision": float("nan"),
            "diversity_prdc_recall": float("nan"),
            "diversity_prdc_density": float("nan"),
            "diversity_prdc_coverage": float("nan"),
        }
    real_dist = torch.cdist(real, real)
    fake_dist = torch.cdist(fake, fake)
    cross = torch.cdist(fake, real)
    real_radius = _knn_radius(real_dist, k)
    fake_radius = _knn_radius(fake_dist, k)
    precision_mask = cross <= real_radius.view(1, -1)
    recall_mask = cross <= fake_radius.view(-1, 1)
    precision = precision_mask.any(dim=1).float().mean()
    recall = recall_mask.any(dim=0).float().mean()
    density = precision_mask.float().sum(dim=1).mean() / float(k)
    coverage = precision_mask.any(dim=0).float().mean()
    return {
        "diversity_prdc_precision": float(precision.cpu()),
        "diversity_prdc_recall": float(recall.cpu()),
        "diversity_prdc_density": float(density.cpu()),
        "diversity_prdc_coverage": float(coverage.cpu()),
    }


def _intra_class_variance_ratio(
    real: torch.Tensor,
    fake: torch.Tensor,
    real_labels: torch.Tensor | None,
    fake_labels: torch.Tensor | None,
) -> float:
    if real_labels is None or fake_labels is None:
        return float("nan")
    ratios = []
    for label in torch.unique(torch.cat([real_labels.view(-1), fake_labels.view(-1)])):
        real_mask = real_labels.view(-1) == label
        fake_mask = fake_labels.view(-1) == label
        if int(real_mask.sum()) < 2 or int(fake_mask.sum()) < 2:
            continue
        real_var = real[real_mask].var(dim=0, unbiased=False).mean()
        fake_var = fake[fake_mask].var(dim=0, unbiased=False).mean()
        ratios.append(fake_var / real_var.clamp_min(1e-8))
    if not ratios:
        return float("nan")
    return float(torch.stack(ratios).mean().cpu())


def diversity_metrics(
    real: torch.Tensor,
    fake: torch.Tensor,
    *,
    real_labels: torch.Tensor | None = None,
    fake_labels: torch.Tensor | None = None,
) -> dict[str, float]:
    """Diversity and coverage metrics for `[N, C, L]` tensors."""
    with torch.no_grad():
        if real.ndim != 3 or fake.ndim != 3:
            return {
                "diversity_prdc_precision": float("nan"),
                "diversity_prdc_recall": float("nan"),
                "diversity_prdc_density": float("nan"),
                "diversity_prdc_coverage": float("nan"),
                "diversity_intra_class_variance_ratio": float("nan"),
                "diversity_status_code": 0.0,
            }
        real_flat = _flatten(real)
        fake_flat = _flatten(fake)
        metrics = _prdc(real_flat, fake_flat)
        metrics["diversity_intra_class_variance_ratio"] = _intra_class_variance_ratio(
            real_flat, fake_flat, real_labels, fake_labels
        )
        metrics["diversity_status_code"] = 1.0
        return metrics
