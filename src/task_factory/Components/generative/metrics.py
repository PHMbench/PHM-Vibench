from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import torch

from .population import PopulationCorrelationMMD


REQUIRED_METRICS = (
    "time_domain_statistics_distance",
    "spectral_distance",
    "condition_consistency_distance",
    "nearest_neighbor_leakage_l2",
    "duplicate_rate",
    "downstream_classifier_utility",
    "fid_like_embedding_distance",
    "training_wall_clock_seconds",
)


def _metric_result(
    value: float | None,
    *,
    status: str,
    reason: str = "",
) -> dict[str, Any]:
    if status not in {"ok", "not_computable", "failed"}:
        raise ValueError(f"invalid metric status: {status}")
    if status == "ok":
        if value is None or not math.isfinite(float(value)):
            raise ValueError("metric status=ok requires a finite numeric value")
        return {"value": float(value), "status": status, "reason": ""}
    if not reason:
        raise ValueError(f"metric status={status} requires a reason")
    return {"value": None, "status": status, "reason": reason}


def _validate_windows(
    real: torch.Tensor,
    fake: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    real_tensor = torch.as_tensor(real).detach().cpu().float()
    fake_tensor = torch.as_tensor(fake).detach().cpu().float()
    if real_tensor.ndim != 3 or fake_tensor.ndim != 3:
        raise ValueError(
            "real and fake windows must both be [N,C,L], got "
            f"{tuple(real_tensor.shape)} and {tuple(fake_tensor.shape)}"
        )
    if real_tensor.shape[1:] != fake_tensor.shape[1:]:
        raise ValueError(
            "real/fake channel and length mismatch: "
            f"{tuple(real_tensor.shape)} vs {tuple(fake_tensor.shape)}"
        )
    if real_tensor.shape[0] <= 0 or fake_tensor.shape[0] <= 0:
        raise ValueError("real and fake windows must not be empty")
    if not torch.isfinite(real_tensor).all() or not torch.isfinite(fake_tensor).all():
        raise ValueError("real or fake windows contain NaN/Inf")
    return real_tensor, fake_tensor


def _feature_embedding(windows: torch.Tensor) -> torch.Tensor:
    mean = windows.mean(dim=2)
    standard_deviation = windows.std(dim=2, unbiased=False)
    rms = windows.square().mean(dim=2).sqrt()
    absolute_mean = windows.abs().mean(dim=2)
    spectrum = torch.fft.rfft(windows, dim=2, norm="ortho").abs()
    frequency_bins = spectrum.shape[2]
    edges = [
        0,
        frequency_bins // 4,
        frequency_bins // 2,
        3 * frequency_bins // 4,
        frequency_bins,
    ]
    bands = []
    for start, end in zip(edges[:-1], edges[1:]):
        bands.append(
            spectrum[:, :, start:end].mean(dim=2)
            if end > start
            else torch.zeros_like(mean)
        )
    return torch.cat(
        [mean, standard_deviation, rms, absolute_mean, *bands],
        dim=1,
    )


def _time_domain_distance(real: torch.Tensor, fake: torch.Tensor) -> float:
    def statistics(windows: torch.Tensor) -> torch.Tensor:
        mean = windows.mean(dim=(0, 2))
        standard_deviation = windows.std(dim=(0, 2), unbiased=False)
        rms = windows.square().mean(dim=(0, 2)).sqrt()
        absolute_mean = windows.abs().mean(dim=(0, 2))
        centered = windows - windows.mean(dim=(0, 2), keepdim=True)
        denominator = centered.square().mean(dim=(0, 2)).pow(1.5).clamp_min(1e-8)
        skewness = centered.pow(3).mean(dim=(0, 2)) / denominator
        return torch.cat(
            [mean, standard_deviation, rms, absolute_mean, skewness],
            dim=0,
        )

    return float(torch.mean(torch.abs(statistics(real) - statistics(fake))).item())


def _spectral_distance(real: torch.Tensor, fake: torch.Tensor) -> float:
    real_spectrum = torch.fft.rfft(real, dim=2, norm="ortho").abs().mean(dim=0)
    fake_spectrum = torch.fft.rfft(fake, dim=2, norm="ortho").abs().mean(dim=0)
    real_spectrum = real_spectrum / real_spectrum.sum(dim=1, keepdim=True).clamp_min(1e-8)
    fake_spectrum = fake_spectrum / fake_spectrum.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return float(torch.mean(torch.abs(real_spectrum - fake_spectrum)).item())


def _label_vector(value: torch.Tensor | None, expected: int) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value).detach().cpu().long().reshape(-1)
    if tensor.numel() != expected:
        raise ValueError(
            f"label length mismatch: expected {expected}, got {tensor.numel()}"
        )
    return tensor


def _condition_distance(
    real_labels: torch.Tensor,
    fake_labels: torch.Tensor,
    real_domains: torch.Tensor,
    fake_domains: torch.Tensor,
) -> float:
    real_pairs = list(zip(real_labels.tolist(), real_domains.tolist()))
    fake_pairs = list(zip(fake_labels.tolist(), fake_domains.tolist()))
    keys = sorted(set(real_pairs) | set(fake_pairs))
    real_total = max(1, len(real_pairs))
    fake_total = max(1, len(fake_pairs))
    return float(
        0.5
        * sum(
            abs(
                real_pairs.count(key) / real_total
                - fake_pairs.count(key) / fake_total
            )
            for key in keys
        )
    )


def _leakage_values(
    real: torch.Tensor,
    fake: torch.Tensor,
    duplicate_threshold: float,
) -> tuple[float, float]:
    if duplicate_threshold < 0:
        raise ValueError("duplicate_threshold must be non-negative")
    distances = torch.cdist(
        fake.reshape(fake.shape[0], -1),
        real.reshape(real.shape[0], -1),
    )
    nearest = distances.min(dim=1).values
    return (
        float(nearest.mean().item()),
        float((nearest <= float(duplicate_threshold)).float().mean().item()),
    )


def _nearest_centroid_accuracy(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
) -> float:
    classes = torch.unique(train_labels)
    if classes.numel() < 2:
        raise ValueError("at least two generated label classes are required")
    centroids = torch.stack(
        [train_features[train_labels == label].mean(dim=0) for label in classes]
    )
    predictions = classes[torch.cdist(test_features, centroids).argmin(dim=1)]
    return float((predictions == test_labels).float().mean().item())


def _fid_like_distance(real: torch.Tensor, fake: torch.Tensor) -> float:
    if real.shape[0] < 2 or fake.shape[0] < 2:
        raise ValueError("at least two real and fake samples are required")
    real_features = _feature_embedding(real)
    fake_features = _feature_embedding(fake)
    real_mean = real_features.mean(dim=0)
    fake_mean = fake_features.mean(dim=0)
    real_variance = real_features.var(dim=0, unbiased=False).clamp_min(0.0)
    fake_variance = fake_features.var(dim=0, unbiased=False).clamp_min(0.0)
    mean_term = (real_mean - fake_mean).square().sum()
    covariance_term = (
        real_variance
        + fake_variance
        - 2.0 * torch.sqrt(real_variance * fake_variance)
    ).sum()
    return float((mean_term + covariance_term).clamp_min(0.0).item())


def _safe_metric(callback: Callable[[], float]) -> dict[str, Any]:
    try:
        value = callback()
    except ValueError as exc:
        return _metric_result(None, status="not_computable", reason=str(exc))
    except Exception as exc:  # pragma: no cover - defensive evidence boundary.
        return _metric_result(None, status="failed", reason=repr(exc))
    if not math.isfinite(float(value)):
        return _metric_result(
            None,
            status="failed",
            reason="metric returned NaN/Inf",
        )
    return _metric_result(float(value), status="ok")


def evaluate_smoke_metrics(
    real: torch.Tensor,
    fake: torch.Tensor,
    *,
    real_labels: torch.Tensor | None = None,
    fake_labels: torch.Tensor | None = None,
    real_domains: torch.Tensor | None = None,
    fake_domains: torch.Tensor | None = None,
    duplicate_threshold: float = 1e-6,
    training_wall_clock_seconds: float | None = None,
    population_rbf_bandwidths: Sequence[float] = (0.1, 0.5, 1.0, 2.0),
) -> dict[str, Any]:
    """Compute the eight required structured Pipeline 06 smoke metrics."""

    real_tensor, fake_tensor = _validate_windows(real, fake)
    real_labels = _label_vector(real_labels, real_tensor.shape[0])
    fake_labels = _label_vector(fake_labels, fake_tensor.shape[0])
    real_domains = _label_vector(real_domains, real_tensor.shape[0])
    fake_domains = _label_vector(fake_domains, fake_tensor.shape[0])

    metrics: dict[str, Any] = {
        "time_domain_statistics_distance": _safe_metric(
            lambda: _time_domain_distance(real_tensor, fake_tensor)
        ),
        "spectral_distance": _safe_metric(
            lambda: _spectral_distance(real_tensor, fake_tensor)
        ),
    }
    metrics["population_dependency_mmd"] = _safe_metric(
        lambda: float(
            PopulationCorrelationMMD(population_rbf_bandwidths)(
                real_tensor,
                fake_tensor,
            ).item()
        )
    )

    if any(
        value is None
        for value in (real_labels, fake_labels, real_domains, fake_domains)
    ):
        metrics["condition_consistency_distance"] = _metric_result(
            None,
            status="not_computable",
            reason="fault labels and domain ids are required for condition consistency",
        )
    else:
        metrics["condition_consistency_distance"] = _safe_metric(
            lambda: _condition_distance(
                real_labels,
                fake_labels,
                real_domains,
                fake_domains,
            )
        )

    leakage = _safe_metric(
        lambda: _leakage_values(
            real_tensor,
            fake_tensor,
            duplicate_threshold,
        )[0]
    )
    duplicates = _safe_metric(
        lambda: _leakage_values(
            real_tensor,
            fake_tensor,
            duplicate_threshold,
        )[1]
    )
    metrics["nearest_neighbor_leakage_l2"] = leakage
    metrics["duplicate_rate"] = duplicates

    if real_labels is None or fake_labels is None:
        metrics["downstream_classifier_utility"] = _metric_result(
            None,
            status="not_computable",
            reason="real and generated fault labels are required",
        )
    else:
        metrics["downstream_classifier_utility"] = _safe_metric(
            lambda: _nearest_centroid_accuracy(
                _feature_embedding(fake_tensor),
                fake_labels,
                _feature_embedding(real_tensor),
                real_labels,
            )
        )

    metrics["fid_like_embedding_distance"] = _safe_metric(
        lambda: _fid_like_distance(real_tensor, fake_tensor)
    )
    metrics["training_wall_clock_seconds"] = (
        _safe_metric(lambda: float(training_wall_clock_seconds))
        if training_wall_clock_seconds is not None
        and float(training_wall_clock_seconds) >= 0.0
        else _metric_result(
            None,
            status="not_computable",
            reason="training stage ledger does not contain a finite wall-clock value",
        )
    )
    metrics["summary"] = {
        "required": list(REQUIRED_METRICS),
        "optional": ["population_dependency_mmd"],
        "ok": sum(
            metrics[name]["status"] == "ok" for name in REQUIRED_METRICS
        ),
        "not_computable": sum(
            metrics[name]["status"] == "not_computable"
            for name in REQUIRED_METRICS
        ),
        "failed": sum(
            metrics[name]["status"] == "failed" for name in REQUIRED_METRICS
        ),
    }
    return metrics
