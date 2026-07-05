import math

import torch

from src.task_factory.Components.generative.metrics.distribution import distribution_metrics
from src.task_factory.Components.generative.metrics.diversity import diversity_metrics
from src.task_factory.Components.generative.metrics.leakage import leakage_metrics
from src.task_factory.Components.generative.metrics.spectral import spectral_metrics
from src.task_factory.Components.generative.metrics.temporal import temporal_metrics
from src.task_factory.Components.generative.metrics.tstr import tstr_metrics


def test_extended_metrics_are_zero_or_finite_for_identical_inputs():
    real = torch.randn(4, 2, 32)
    labels = torch.tensor([0, 0, 1, 1])

    metrics = {}
    metrics.update(temporal_metrics(real, real.clone()))
    metrics.update(spectral_metrics(real, real.clone()))
    metrics.update(distribution_metrics(real, real.clone()))
    metrics.update(leakage_metrics(real, real.clone()))
    metrics.update(diversity_metrics(real, real.clone(), real_labels=labels, fake_labels=labels))
    metrics.update(tstr_metrics(real, real.clone(), real_labels=labels, fake_labels=labels))

    assert metrics["temporal_l1"] == 0.0
    assert metrics["spectral_fft_l1"] == 0.0
    assert abs(metrics["distribution_mmd_rbf"]) < 1e-6
    assert metrics["leakage_duplicate_rate"] == 1.0
    assert math.isfinite(metrics["diversity_prdc_precision"])
    assert math.isfinite(metrics["tstr_accuracy"])


def test_shape_mismatch_returns_status_code_zero():
    real = torch.randn(2, 2, 16)
    fake = torch.randn(2, 2, 15)

    temporal = temporal_metrics(real, fake)
    spectral = spectral_metrics(real, fake)

    assert math.isnan(temporal["temporal_l1"])
    assert temporal["temporal_l1_status_code"] == 0.0
    assert math.isnan(spectral["spectral_fft_l1"])
    assert spectral["spectral_fft_l1_status_code"] == 0.0
