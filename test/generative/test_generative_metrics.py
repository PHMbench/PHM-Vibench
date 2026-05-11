from __future__ import annotations

import math

import torch

from src.task_factory.task.generative.generative_eval import evaluate_generated_windows


def test_noncomputable_metrics_include_status_and_reason_for_invalid_shape() -> None:
    metrics = evaluate_generated_windows(torch.randn(2, 8), torch.randn(2, 8))

    for key in [
        "temporal_l1",
        "spectral_fft_l1",
        "distribution_mean_distance",
        "diversity_prdc_precision",
        "leakage_nearest_neighbor_l2",
    ]:
        assert math.isnan(float(metrics[key]))
        assert metrics[f"{key}_status"] == "not_computable"
        assert "[N,C,L]" in str(metrics[f"{key}_reason"])


def test_label_dependent_metrics_record_missing_label_reasons() -> None:
    real = torch.randn(4, 2, 16)
    fake = torch.randn(4, 2, 16)

    metrics = evaluate_generated_windows(real, fake)

    assert math.isnan(float(metrics["tstr_accuracy"]))
    assert metrics["tstr_accuracy_status"] == "not_computable"
    assert "real_labels and fake_labels" in str(metrics["tstr_accuracy_reason"])
    assert math.isnan(float(metrics["diversity_intra_class_variance_ratio"]))
    assert metrics["diversity_intra_class_variance_ratio_status"] == "not_computable"
    assert "intra-class diversity" in str(metrics["diversity_intra_class_variance_ratio_reason"])


def test_computable_metrics_include_computed_status() -> None:
    real = torch.randn(4, 2, 16)
    fake = real + 0.1 * torch.randn(4, 2, 16)
    labels = torch.tensor([0, 0, 1, 1])

    metrics = evaluate_generated_windows(
        real,
        fake,
        real_labels=labels,
        fake_labels=labels,
        real_domains=labels,
        fake_domains=labels,
    )

    assert metrics["temporal_l1_status"] == "ok"
    assert metrics["temporal_l1_reason"] == ""
    assert metrics["tstr_accuracy_status"] == "ok"
    assert metrics["tstr_accuracy_reason"] == ""


def test_nonfinite_inputs_record_explicit_reason() -> None:
    real = torch.randn(2, 2, 16)
    fake = torch.randn(2, 2, 16)
    fake[0, 0, 0] = float("nan")

    metrics = evaluate_generated_windows(real, fake)

    assert metrics["temporal_l1_status"] == "not_computable"
    assert "NaN/Inf" in str(metrics["temporal_l1_reason"])


def test_sample_count_reason_is_recorded_for_prdc_metrics() -> None:
    real = torch.randn(1, 2, 16)
    fake = torch.randn(1, 2, 16)

    metrics = evaluate_generated_windows(real, fake)

    assert metrics["diversity_prdc_precision_status"] == "not_computable"
    assert "at least two real and fake samples" in str(metrics["diversity_prdc_precision_reason"])
