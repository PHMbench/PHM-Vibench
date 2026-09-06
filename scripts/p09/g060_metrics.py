"""Strict, prediction-recomputable metrics for P09-G060.

All selective metrics use the joint four-class confidence.  AURC follows the
frozen statistical protocol: linear interpolation at the coverage bounds and
trapezoidal integration of the prefix-risk curve over coverage [0.5, 1.0]. If
an arm accepts no observations, its observed selective risk is conservatively
recorded as one rather than omitted.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


PROBABILITY_TOLERANCE = 1.0e-6


def _validated_predictions(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    class_ids: Sequence[int] | np.ndarray,
    accepted: Sequence[bool] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels_array = np.asarray(labels, dtype=np.int64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    class_ids_array = np.asarray(class_ids, dtype=np.int64)
    accepted_array = np.asarray(accepted, dtype=np.bool_)
    if labels_array.ndim != 1 or labels_array.size == 0:
        raise ValueError("labels must be a non-empty rank-1 array")
    if probabilities_array.shape != (labels_array.size, class_ids_array.size):
        raise ValueError("probabilities must have shape [observations, classes]")
    if class_ids_array.ndim != 1 or class_ids_array.size != 4:
        raise ValueError("P09-G060 requires exactly four joint class ids")
    if np.unique(class_ids_array).size != class_ids_array.size:
        raise ValueError("class_ids must be unique")
    if accepted_array.shape != labels_array.shape:
        raise ValueError("accepted must match labels")
    if not np.isfinite(probabilities_array).all():
        raise ValueError("probabilities contain non-finite values")
    if np.any(probabilities_array < 0.0) or np.any(probabilities_array > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.allclose(
        probabilities_array.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=PROBABILITY_TOLERANCE,
    ):
        raise ValueError("probability rows must sum to one")
    unknown = np.setdiff1d(np.unique(labels_array), class_ids_array)
    if unknown.size:
        raise ValueError(f"labels contain unregistered class ids: {unknown.tolist()}")
    return labels_array, probabilities_array, class_ids_array, accepted_array


def equal_mass_ece(
    correctness: Sequence[bool] | np.ndarray,
    confidence: Sequence[float] | np.ndarray,
    *,
    bins: int = 15,
) -> float:
    correctness_array = np.asarray(correctness, dtype=np.float64)
    confidence_array = np.asarray(confidence, dtype=np.float64)
    if correctness_array.ndim != 1 or correctness_array.size == 0:
        raise ValueError("correctness must be a non-empty rank-1 array")
    if confidence_array.shape != correctness_array.shape:
        raise ValueError("confidence must match correctness")
    if bins <= 0:
        raise ValueError("bins must be positive")
    if not np.isfinite(confidence_array).all() or np.any(
        (confidence_array < 0.0) | (confidence_array > 1.0)
    ):
        raise ValueError("confidence must be finite and lie in [0, 1]")
    order = np.argsort(confidence_array, kind="stable")
    groups = np.array_split(order, min(int(bins), order.size))
    return float(
        sum(
            group.size
            / order.size
            * abs(
                float(correctness_array[group].mean())
                - float(confidence_array[group].mean())
            )
            for group in groups
            if group.size
        )
    )


def normalized_aurc(
    errors: Sequence[bool] | np.ndarray,
    confidence: Sequence[float] | np.ndarray,
    *,
    coverage_min: float = 0.5,
    coverage_max: float = 1.0,
) -> float:
    errors_array = np.asarray(errors, dtype=np.float64)
    confidence_array = np.asarray(confidence, dtype=np.float64)
    if errors_array.ndim != 1 or errors_array.size == 0:
        raise ValueError("errors must be a non-empty rank-1 array")
    if confidence_array.shape != errors_array.shape:
        raise ValueError("confidence must match errors")
    if not 0.0 <= coverage_min < coverage_max <= 1.0:
        raise ValueError("coverage interval must satisfy 0 <= min < max <= 1")
    if not np.isfinite(confidence_array).all():
        raise ValueError("confidence contains non-finite values")
    order = np.argsort(-confidence_array, kind="stable")
    prefix_risk = np.cumsum(errors_array[order]) / np.arange(
        1, errors_array.size + 1, dtype=np.float64
    )
    coverage = np.arange(1, errors_array.size + 1, dtype=np.float64) / errors_array.size
    interior = (coverage > coverage_min) & (coverage < coverage_max)
    selected_coverage = np.concatenate(
        ([coverage_min], coverage[interior], [coverage_max])
    )
    selected_risk = np.interp(selected_coverage, coverage, prefix_risk)
    return float(
        np.trapz(selected_risk, selected_coverage)
        / (coverage_max - coverage_min)
    )


def risk_at_coverage(
    errors: Sequence[bool] | np.ndarray,
    confidence: Sequence[float] | np.ndarray,
    *,
    coverage: float,
) -> tuple[float, float, int]:
    errors_array = np.asarray(errors, dtype=np.float64)
    confidence_array = np.asarray(confidence, dtype=np.float64)
    if errors_array.ndim != 1 or errors_array.size == 0:
        raise ValueError("errors must be a non-empty rank-1 array")
    if confidence_array.shape != errors_array.shape:
        raise ValueError("confidence must match errors")
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must lie in (0, 1]")
    count = max(1, int(np.ceil(coverage * errors_array.size)))
    order = np.argsort(-confidence_array, kind="stable")[:count]
    return float(errors_array[order].mean()), count / errors_array.size, count


def compute_episode_metrics(
    *,
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    class_ids: Sequence[int] | np.ndarray,
    accepted: Sequence[bool] | np.ndarray,
    base_class_ids: Sequence[int] = (0, 1),
    novel_class_ids: Sequence[int] = (2, 3),
    adaptation_wall_time_seconds: float,
    inference_latency_seconds: float,
    adapted_parameters: int,
    peak_accelerator_memory_bytes: int,
) -> dict[str, float | int]:
    labels_array, probabilities_array, class_ids_array, accepted_array = (
        _validated_predictions(labels, probabilities, class_ids, accepted)
    )
    base_ids = np.asarray(base_class_ids, dtype=np.int64)
    novel_ids = np.asarray(novel_class_ids, dtype=np.int64)
    if set(base_ids.tolist()) & set(novel_ids.tolist()):
        raise ValueError("base and novel class ids must be disjoint")
    if set(base_ids.tolist()) | set(novel_ids.tolist()) != set(class_ids_array.tolist()):
        raise ValueError("base and novel class ids must partition joint class ids")
    for name, value in (
        ("adaptation_wall_time_seconds", adaptation_wall_time_seconds),
        ("inference_latency_seconds", inference_latency_seconds),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if adapted_parameters < 0 or peak_accelerator_memory_bytes < 0:
        raise ValueError("resource counts must be non-negative")

    prediction_indices = probabilities_array.argmax(axis=1)
    predictions = class_ids_array[prediction_indices]
    confidence = probabilities_array[np.arange(labels_array.size), prediction_indices]
    correctness = predictions == labels_array
    errors = ~correctness
    base_mask = np.isin(labels_array, base_ids)
    novel_mask = np.isin(labels_array, novel_ids)
    if not base_mask.any() or not novel_mask.any():
        raise ValueError("one episode must contain base and novel query observations")
    base_accuracy = float(correctness[base_mask].mean())
    novel_accuracy = float(correctness[novel_mask].mean())
    denominator = base_accuracy + novel_accuracy
    harmonic_mean = (
        0.0 if denominator == 0.0 else 2.0 * base_accuracy * novel_accuracy / denominator
    )
    true_indices = np.asarray(
        [{int(class_id): index for index, class_id in enumerate(class_ids_array)}[int(label)] for label in labels_array],
        dtype=np.int64,
    )
    negative_log_likelihood = float(
        -np.log(
            np.clip(
                probabilities_array[np.arange(labels_array.size), true_indices],
                np.finfo(np.float64).tiny,
                1.0,
            )
        ).mean()
    )
    accepted_count = int(accepted_array.sum())
    observed_risk = (
        float(errors[accepted_array].mean()) if accepted_count else 1.0
    )
    matched_risk, matched_coverage, matched_count = risk_at_coverage(
        errors, confidence, coverage=0.80
    )
    return {
        "observations": int(labels_array.size),
        "base_observations": int(base_mask.sum()),
        "novel_observations": int(novel_mask.sum()),
        "base_accuracy": base_accuracy,
        "novel_accuracy": novel_accuracy,
        "harmonic_mean": harmonic_mean,
        "joint_accuracy": float(correctness.mean()),
        "negative_log_likelihood": negative_log_likelihood,
        "equal_mass_ece_15_bins": equal_mass_ece(correctness, confidence, bins=15),
        "normalized_aurc_coverage_0_5_to_1_0": normalized_aurc(
            errors, confidence, coverage_min=0.5, coverage_max=1.0
        ),
        "coverage": accepted_count / labels_array.size,
        "error_risk": observed_risk,
        "accepted_observations": accepted_count,
        "zero_acceptance": int(accepted_count == 0),
        "matched_coverage_risk_0_80": matched_risk,
        "matched_coverage_realized_0_80": matched_coverage,
        "matched_coverage_observations_0_80": matched_count,
        "adaptation_wall_time_seconds": float(adaptation_wall_time_seconds),
        "inference_latency_seconds": float(inference_latency_seconds),
        "adapted_parameters": int(adapted_parameters),
        "peak_accelerator_memory_bytes": int(peak_accelerator_memory_bytes),
    }


def recompute_episode_row(row: Mapping[str, Any]) -> dict[str, float | int]:
    return compute_episode_metrics(
        labels=row["labels"],
        probabilities=row["probabilities"],
        class_ids=row["class_ids"],
        accepted=row["accepted"],
        base_class_ids=row.get("base_class_ids", (0, 1)),
        novel_class_ids=row.get("novel_class_ids", (2, 3)),
        adaptation_wall_time_seconds=float(row["adaptation_wall_time_seconds"]),
        inference_latency_seconds=float(row["inference_latency_seconds"]),
        adapted_parameters=int(row["adapted_parameters"]),
        peak_accelerator_memory_bytes=int(row["peak_accelerator_memory_bytes"]),
    )


__all__ = [
    "compute_episode_metrics",
    "equal_mass_ece",
    "normalized_aurc",
    "recompute_episode_row",
    "risk_at_coverage",
]
