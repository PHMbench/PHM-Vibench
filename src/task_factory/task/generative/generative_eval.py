from __future__ import annotations

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


def _add_group_metrics(
    metrics: dict[str, float],
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
) -> dict[str, float]:
    """Compute the lightweight generative metric bundle for `[N, C, L]` tensors."""
    real_labels = _label_tensor(real_labels)
    fake_labels = _label_tensor(fake_labels)
    real_domains = _label_tensor(real_domains)
    fake_domains = _label_tensor(fake_domains)
    metrics: dict[str, float] = {}
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
    return metrics
