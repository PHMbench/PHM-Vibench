from __future__ import annotations

import torch

from src.task_factory.Components.generative.metrics.distribution import distribution_metrics
from src.task_factory.Components.generative.metrics.leakage import leakage_metrics
from src.task_factory.Components.generative.metrics.spectral import spectral_metrics
from src.task_factory.Components.generative.metrics.temporal import temporal_metrics
from src.task_factory.Components.generative.metrics.tstr import tstr_placeholder


def evaluate_generated_windows(real: torch.Tensor, fake: torch.Tensor) -> dict[str, float]:
    """Compute the V0 lightweight generative metric bundle."""
    metrics: dict[str, float] = {}
    metrics.update(temporal_metrics(real, fake))
    metrics.update(spectral_metrics(real, fake))
    metrics.update(distribution_metrics(real, fake))
    metrics.update(leakage_metrics(real, fake))
    metrics.update(tstr_placeholder())
    return metrics
