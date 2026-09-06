from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.p09.g060_metrics import (
    compute_episode_metrics,
    equal_mass_ece,
    normalized_aurc,
    recompute_episode_row,
    risk_at_coverage,
)


def synthetic_row() -> dict[str, object]:
    return {
        "labels": [0, 1, 2, 3],
        "probabilities": [
            [0.90, 0.05, 0.03, 0.02],
            [0.10, 0.70, 0.10, 0.10],
            [0.10, 0.10, 0.60, 0.20],
            [0.40, 0.10, 0.20, 0.30],
        ],
        "class_ids": [0, 1, 2, 3],
        "accepted": [True, True, True, False],
        "base_class_ids": [0, 1],
        "novel_class_ids": [2, 3],
        "adaptation_wall_time_seconds": 0.2,
        "inference_latency_seconds": 0.03,
        "adapted_parameters": 4096,
        "peak_accelerator_memory_bytes": 1234,
    }


def test_episode_metrics_are_prediction_recomputable() -> None:
    row = synthetic_row()
    observed = recompute_episode_row(row)
    direct = compute_episode_metrics(**row)  # type: ignore[arg-type]
    assert observed == direct
    assert observed["base_accuracy"] == 1.0
    assert observed["novel_accuracy"] == 0.5
    assert observed["harmonic_mean"] == pytest.approx(2.0 / 3.0)
    assert observed["joint_accuracy"] == 0.75
    assert observed["coverage"] == 0.75
    assert observed["error_risk"] == 0.0
    assert observed["matched_coverage_realized_0_80"] == 1.0
    assert observed["matched_coverage_risk_0_80"] == 0.25
    assert math.isfinite(float(observed["negative_log_likelihood"]))


def test_zero_acceptance_is_retained_conservatively() -> None:
    row = synthetic_row()
    row["accepted"] = [False] * 4
    observed = recompute_episode_row(row)
    assert observed["coverage"] == 0.0
    assert observed["error_risk"] == 1.0
    assert observed["zero_acceptance"] == 1


def test_equal_mass_ece_uses_weighted_bins() -> None:
    correctness = np.asarray([True, False, True, False])
    confidence = np.asarray([0.9, 0.8, 0.7, 0.6])
    assert equal_mass_ece(correctness, confidence, bins=2) == pytest.approx(0.25)


def test_aurc_uses_boundary_interpolation_and_trapezoids() -> None:
    errors = np.asarray([False, False, False, True])
    confidence = np.asarray([0.9, 0.8, 0.7, 0.6])
    assert normalized_aurc(errors, confidence) == pytest.approx(0.0625)
    risk, coverage, count = risk_at_coverage(errors, confidence, coverage=0.8)
    assert (risk, coverage, count) == (0.25, 1.0, 4)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("probabilities", [[0.2, 0.2, 0.2, 0.2]] * 4, "sum to one"),
        ("labels", [0, 1, 2, 9], "unregistered"),
        ("accepted", [True], "match labels"),
    ],
)
def test_invalid_prediction_artifacts_fail(field: str, value: object, match: str) -> None:
    row = synthetic_row()
    row[field] = value
    with pytest.raises(ValueError, match=match):
        recompute_episode_row(row)
