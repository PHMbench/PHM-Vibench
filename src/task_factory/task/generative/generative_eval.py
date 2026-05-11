from __future__ import annotations

import math

import torch

from src.task_factory.Components.generative.metrics.distribution import distribution_metrics
from src.task_factory.Components.generative.metrics.diversity import diversity_metrics
from src.task_factory.Components.generative.metrics.leakage import leakage_metrics
from src.task_factory.Components.generative.metrics.spectral import spectral_metrics
from src.task_factory.Components.generative.metrics.temporal import temporal_metrics
from src.task_factory.Components.generative.metrics.tstr import tstr_metrics


def _label_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return torch.as_tensor(value).long().view(-1)


def _numeric_metric_value(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_metric_value_key(key: str) -> bool:
    return not (
        key.endswith("_status")
        or key.endswith("_reason")
        or key.endswith("_status_code")
    )


def _invalid_window_reason(real: torch.Tensor, fake: torch.Tensor) -> str | None:
    if real.ndim != 3 or fake.ndim != 3:
        return "real and fake windows must be rank-3 [N,C,L] tensors"
    if real.shape != fake.shape:
        return f"real and fake windows must have matching shape, got {tuple(real.shape)} and {tuple(fake.shape)}"
    return None


def _reason_for_metric(
    key: str,
    *,
    real: torch.Tensor,
    fake: torch.Tensor,
    real_labels: torch.Tensor | None,
    fake_labels: torch.Tensor | None,
    real_domains: torch.Tensor | None,
    fake_domains: torch.Tensor | None,
) -> str:
    shape_reason = _invalid_window_reason(real, fake)
    if shape_reason:
        return shape_reason
    if not torch.isfinite(real).all() or not torch.isfinite(fake).all():
        return "real or fake windows contain NaN/Inf"
    if key.startswith("diversity_prdc_") and (real.shape[0] < 2 or fake.shape[0] < 2):
        return "at least two real and fake samples are required for PRDC diversity metrics"
    if key.startswith(("tstr_", "trts_")) and (real_labels is None or fake_labels is None):
        return "real_labels and fake_labels are required for downstream utility metrics"
    if key.startswith("diversity_intra_class") and (real_labels is None or fake_labels is None):
        return "real_labels and fake_labels are required for intra-class diversity"
    if "_fault_" in key and (real_labels is None or fake_labels is None):
        return "fault group labels are required for fault-conditioned metrics"
    if "_domain_" in key and (real_domains is None or fake_domains is None):
        return "domain ids are required for domain-conditioned metrics"
    return "metric returned NaN or Inf for the available samples"


def _annotate_metric_statuses(
    metrics: dict[str, object],
    *,
    real: torch.Tensor,
    fake: torch.Tensor,
    real_labels: torch.Tensor | None,
    fake_labels: torch.Tensor | None,
    real_domains: torch.Tensor | None,
    fake_domains: torch.Tensor | None,
) -> None:
    for key, value in list(metrics.items()):
        if not _is_metric_value_key(key) or not _numeric_metric_value(value):
            continue
        if math.isfinite(float(value)):
            metrics.setdefault(f"{key}_status", "ok")
            metrics.setdefault(f"{key}_reason", "")
        else:
            metrics.setdefault(f"{key}_status", "not_computable")
            metrics.setdefault(
                f"{key}_reason",
                _reason_for_metric(
                    key,
                    real=real,
                    fake=fake,
                    real_labels=real_labels,
                    fake_labels=fake_labels,
                    real_domains=real_domains,
                    fake_domains=fake_domains,
                ),
            )


def _add_group_metrics(
    metrics: dict[str, object],
    *,
    group_name: str,
    real: torch.Tensor,
    fake: torch.Tensor,
    real_group: torch.Tensor | None,
    fake_group: torch.Tensor | None,
) -> None:
    if real_group is None or fake_group is None:
        return
    for group_value in torch.unique(torch.cat([real_group, fake_group])).tolist():
        real_mask = real_group == int(group_value)
        fake_mask = fake_group == int(group_value)
        if not bool(real_mask.any()) or not bool(fake_mask.any()):
            continue
        n = min(int(real_mask.sum()), int(fake_mask.sum()))
        if n <= 0:
            continue
        real_sub = real[real_mask][:n]
        fake_sub = fake[fake_mask][:n]
        group_metrics: dict[str, float] = {}
        group_metrics.update(temporal_metrics(real_sub, fake_sub))
        group_metrics.update(spectral_metrics(real_sub, fake_sub))
        group_metrics.update(distribution_metrics(real_sub, fake_sub))
        prefix = f"{group_name}_{int(group_value)}"
        for key, value in group_metrics.items():
            metrics[f"{key}_{prefix}"] = value


def evaluate_generated_windows(
    real: torch.Tensor,
    fake: torch.Tensor,
    *,
    real_labels: torch.Tensor | None = None,
    fake_labels: torch.Tensor | None = None,
    real_domains: torch.Tensor | None = None,
    fake_domains: torch.Tensor | None = None,
) -> dict[str, object]:
    """Compute the lightweight generative metric bundle for `[N, C, L]` tensors."""
    real_labels = _label_tensor(real_labels)
    fake_labels = _label_tensor(fake_labels)
    real_domains = _label_tensor(real_domains)
    fake_domains = _label_tensor(fake_domains)
    metrics: dict[str, object] = {}
    metrics.update(temporal_metrics(real, fake))
    metrics.update(spectral_metrics(real, fake))
    metrics.update(distribution_metrics(real, fake))
    metrics.update(leakage_metrics(real, fake))
    metrics.update(diversity_metrics(real, fake, real_labels=real_labels, fake_labels=fake_labels))
    metrics.update(tstr_metrics(real, fake, real_labels=real_labels, fake_labels=fake_labels))
    metrics["eval_has_fault_labels"] = float(real_labels is not None and fake_labels is not None)
    metrics["eval_has_domain_ids"] = float(real_domains is not None and fake_domains is not None)
    _add_group_metrics(
        metrics,
        group_name="fault",
        real=real,
        fake=fake,
        real_group=real_labels,
        fake_group=fake_labels,
    )
    _add_group_metrics(
        metrics,
        group_name="domain",
        real=real,
        fake=fake,
        real_group=real_domains,
        fake_group=fake_domains,
    )
    _annotate_metric_statuses(
        metrics,
        real=real,
        fake=fake,
        real_labels=real_labels,
        fake_labels=fake_labels,
        real_domains=real_domains,
        fake_domains=fake_domains,
    )
    return metrics
