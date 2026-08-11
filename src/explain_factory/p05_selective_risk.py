"""Frozen selective-risk calculations for the P05 C3 protocol."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class LogisticRiskRanker:
    name: str
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    coefficient: tuple[float, ...]
    intercept: float

    def score(self, features: np.ndarray) -> np.ndarray:
        values = _finite_matrix(features, name=f"{self.name} features")
        if values.shape[1] != len(self.feature_mean):
            raise ValueError(
                f"{self.name} feature count mismatch: expected {len(self.feature_mean)}, "
                f"got {values.shape[1]}"
            )
        standardized = (
            values - np.asarray(self.feature_mean, dtype=np.float64)
        ) / np.asarray(self.feature_std, dtype=np.float64)
        scores = expit(
            standardized @ np.asarray(self.coefficient, dtype=np.float64)
            + self.intercept
        )
        if not np.isfinite(scores).all():
            raise FloatingPointError(f"{self.name} produced non-finite risk scores")
        return scores.astype(np.float64, copy=False)


@dataclass(frozen=True)
class ValidationRiskBundle:
    trace_ranker: LogisticRiskRanker
    trace_free_ranker: LogisticRiskRanker
    temperature: float
    thresholds: Mapping[str, float]


def _finite_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return array


def _labels(values: np.ndarray, *, sample_count: int) -> np.ndarray:
    labels = np.asarray(values)
    if labels.shape != (sample_count,):
        raise ValueError(f"labels must have shape ({sample_count},), got {labels.shape}")
    if labels.dtype.kind == "f":
        if not np.isfinite(labels).all() or not np.equal(labels, np.round(labels)).all():
            raise ValueError("labels must contain finite integer class indices")
    labels = labels.astype(np.int64, copy=False)
    return labels


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = _finite_matrix(logits, name="logits")
    probabilities = np.exp(values - logsumexp(values, axis=1, keepdims=True))
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("softmax probabilities are non-finite")
    return probabilities


def predictive_risk_features(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R1 features and the original softmax probabilities."""

    probabilities = _softmax(logits)
    classes = probabilities.shape[1]
    if classes < 2:
        raise ValueError("selective risk requires at least two classes")
    ordered = np.sort(probabilities, axis=1)
    msp = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, np.finfo(np.float64).tiny, 1.0)),
        axis=1,
    ) / math.log(classes)
    features = np.column_stack((1.0 - msp, entropy, 1.0 - margin))
    return features.astype(np.float64, copy=False), probabilities


def trace_risk_features(logits: np.ndarray, firing: np.ndarray) -> np.ndarray:
    prediction_features, _ = predictive_risk_features(logits)
    firing_values = _finite_matrix(firing, name="normalized rule firing")
    if firing_values.shape[0] != prediction_features.shape[0]:
        raise ValueError("firing and logits must contain the same number of samples")
    if firing_values.shape[1] < 2:
        raise ValueError("trace risk requires at least two rules")
    if np.any(firing_values < 0.0):
        raise ValueError("normalized rule firing cannot be negative")
    row_sums = firing_values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("rule firing rows must sum to one")
    entropy = -np.sum(
        firing_values
        * np.log(np.clip(firing_values, np.finfo(np.float64).tiny, 1.0)),
        axis=1,
    ) / math.log(firing_values.shape[1])
    top_share = firing_values.max(axis=1)
    return np.column_stack((prediction_features[:, 0], entropy, 1.0 - top_share))


def equal_group_window_weights(groups: Sequence[str]) -> np.ndarray:
    group_values = np.asarray([str(group) for group in groups], dtype=object)
    if group_values.ndim != 1 or len(group_values) == 0 or any(not value for value in group_values):
        raise ValueError("groups must be a non-empty sequence of non-empty identifiers")
    unique, counts = np.unique(group_values, return_counts=True)
    count_by_group = dict(zip(unique.tolist(), counts.tolist()))
    raw = np.asarray(
        [1.0 / (len(unique) * count_by_group[group]) for group in group_values],
        dtype=np.float64,
    )
    weights = raw / raw.mean(dtype=np.float64)
    if not np.isclose(weights.mean(), 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("risk sample weights do not have mean one")
    return weights


def fit_logistic_risk_ranker(
    features: np.ndarray,
    error_target: np.ndarray,
    sample_weight: np.ndarray,
    *,
    name: str,
) -> LogisticRiskRanker:
    values = _finite_matrix(features, name=f"{name} features")
    targets = _labels(error_target, sample_count=values.shape[0])
    if set(targets.tolist()) != {0, 1}:
        raise ValueError(f"{name} validation error target must contain both outcomes")
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != (values.shape[0],):
        raise ValueError(f"{name} sample weights have the wrong shape")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError(f"{name} sample weights must be finite and positive")
    weights = weights / weights.mean(dtype=np.float64)
    weight_sum = weights.sum(dtype=np.float64)
    mean = np.sum(values * weights[:, None], axis=0, dtype=np.float64) / weight_sum
    variance = np.sum(
        np.square(values - mean) * weights[:, None], axis=0, dtype=np.float64
    ) / weight_sum
    std = np.sqrt(variance)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 1e-8):
        raise ValueError(f"{name} validation feature standard deviation is below 1e-8")
    standardized = (values - mean) / std
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        fit_intercept=True,
        tol=1e-4,
        class_weight=None,
        max_iter=1000,
        random_state=None,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(standardized, targets, sample_weight=weights)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError(f"{name} logistic ranker did not converge")
    if int(model.n_iter_[0]) >= 1000:
        raise RuntimeError(f"{name} logistic ranker reached its iteration limit")
    coefficient = model.coef_.reshape(-1).astype(np.float64)
    intercept = float(model.intercept_[0])
    if not np.isfinite(coefficient).all() or not math.isfinite(intercept):
        raise FloatingPointError(f"{name} fitted non-finite coefficients")
    return LogisticRiskRanker(
        name=name,
        feature_mean=tuple(float(value) for value in mean),
        feature_std=tuple(float(value) for value in std),
        coefficient=tuple(float(value) for value in coefficient),
        intercept=intercept,
    )


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
) -> float:
    values = _finite_matrix(logits, name="validation logits")
    targets = _labels(labels, sample_count=values.shape[0])
    if np.any(targets < 0) or np.any(targets >= values.shape[1]):
        raise ValueError("validation labels contain an out-of-range class")
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != (len(targets),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("temperature-fit weights must be finite, positive, and one-dimensional")
    weights = weights / weights.mean(dtype=np.float64)

    def objective(temperature: float) -> float:
        scaled = values / float(temperature)
        nll = logsumexp(scaled, axis=1) - scaled[np.arange(len(targets)), targets]
        return float(np.sum(weights * nll, dtype=np.float64) / weights.sum(dtype=np.float64))

    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=(0.05, 20.0),
        options={"xatol": 1e-8, "maxiter": 500},
    )
    temperature = float(result.x)
    if not result.success or not math.isfinite(temperature) or not (0.05 <= temperature <= 20.0):
        raise RuntimeError("validation temperature optimization failed")
    return temperature


def select_validation_threshold(
    scores: np.ndarray,
    groups: Sequence[str],
    *,
    target_coverage: float = 0.90,
) -> float:
    values = np.asarray(scores, dtype=np.float64)
    group_values = np.asarray([str(group) for group in groups], dtype=object)
    if values.shape != group_values.shape or values.ndim != 1 or len(values) == 0:
        raise ValueError("validation scores and groups must be equal non-empty vectors")
    if not np.isfinite(values).all():
        raise FloatingPointError("validation scores contain non-finite values")
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must be in (0, 1]")
    unique_groups = sorted(set(group_values.tolist()))
    for threshold in np.unique(values):
        coverage = np.mean(
            [np.mean(values[group_values == group] <= threshold) for group in unique_groups]
        )
        if coverage >= target_coverage:
            return float(threshold)
    raise AssertionError("the maximum validation score did not reach full coverage")


def score_risk_methods(
    bundle: ValidationRiskBundle,
    logits: np.ndarray,
    firing: np.ndarray,
) -> dict[str, np.ndarray]:
    trace_features = trace_risk_features(logits, firing)
    prediction_features, probabilities = predictive_risk_features(logits)
    temperature_probabilities = _softmax(np.asarray(logits, dtype=np.float64) / bundle.temperature)
    return {
        "trace": bundle.trace_ranker.score(trace_features),
        "R0": prediction_features[:, 0],
        "R1": bundle.trace_free_ranker.score(prediction_features),
        "R2": prediction_features[:, 1],
        "R3": 1.0 - temperature_probabilities.max(axis=1),
        "confidence": probabilities.max(axis=1),
    }


def fit_validation_risk_bundle(
    *,
    sample_ids: Sequence[str],
    groups: Sequence[str],
    logits: np.ndarray,
    firing: np.ndarray,
    labels: np.ndarray,
) -> ValidationRiskBundle:
    values = _finite_matrix(logits, name="validation logits")
    if len(sample_ids) != values.shape[0] or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("validation sample_ids must be unique and match logits")
    if len(groups) != values.shape[0]:
        raise ValueError("validation groups must match logits")
    targets = _labels(labels, sample_count=values.shape[0])
    prediction_features, probabilities = predictive_risk_features(values)
    predictions = probabilities.argmax(axis=1)
    error = (predictions != targets).astype(np.int64)
    weights = equal_group_window_weights(groups)
    trace_features = trace_risk_features(values, firing)
    trace_ranker = fit_logistic_risk_ranker(
        trace_features, error, weights, name="trace"
    )
    trace_free_ranker = fit_logistic_risk_ranker(
        prediction_features, error, weights, name="R1"
    )
    temperature = fit_temperature(values, targets, weights)
    provisional = ValidationRiskBundle(
        trace_ranker=trace_ranker,
        trace_free_ranker=trace_free_ranker,
        temperature=temperature,
        thresholds={},
    )
    scores = score_risk_methods(provisional, values, firing)
    thresholds = {
        method: select_validation_threshold(score, groups, target_coverage=0.90)
        for method, score in scores.items()
        if method != "confidence"
    }
    return ValidationRiskBundle(
        trace_ranker=trace_ranker,
        trace_free_ranker=trace_free_ranker,
        temperature=temperature,
        thresholds=thresholds,
    )


def _macro_f1(
    predictions: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[int] = (0, 1),
) -> float:
    classes = tuple(int(value) for value in classes)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("macro-F1 classes must contain at least two unique IDs")
    values = []
    for class_id in classes:
        true_positive = np.sum((predictions == class_id) & (labels == class_id))
        false_positive = np.sum((predictions == class_id) & (labels != class_id))
        false_negative = np.sum((predictions != class_id) & (labels == class_id))
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return float(np.mean(values))


def retrospective_selective_metrics(
    *,
    sample_ids: Sequence[str],
    groups: Sequence[str],
    scores: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    coverages: Sequence[float] = (0.70, 0.80, 0.90, 0.95),
) -> dict[str, object]:
    values = np.asarray(scores, dtype=np.float64)
    ids = np.asarray([str(value) for value in sample_ids], dtype=object)
    group_values = np.asarray([str(value) for value in groups], dtype=object)
    predictions = _labels(predictions, sample_count=len(values))
    targets = _labels(labels, sample_count=len(values))
    if values.ndim != 1 or ids.shape != values.shape or group_values.shape != values.shape:
        raise ValueError("selective metric inputs must be equal one-dimensional vectors")
    if len(set(ids.tolist())) != len(ids) or any(not value for value in ids):
        raise ValueError("sample_ids must be non-empty and unique")
    if not np.isfinite(values).all():
        raise FloatingPointError("risk scores contain non-finite values")
    requested = tuple(float(value) for value in coverages)
    if len(set(requested)) != len(requested) or any(not 0 < value <= 1 for value in requested):
        raise ValueError("coverages must be unique values in (0, 1]")

    group_results: dict[str, object] = {}
    for group in sorted(set(group_values.tolist())):
        indices = np.flatnonzero(group_values == group)
        if len(set(targets[indices].tolist())) < 2:
            raise ValueError(f"group {group!r} is missing a protocol class")
        order = np.lexsort((ids[indices], values[indices]))
        ordered_indices = indices[order]
        ordered_errors = (predictions[ordered_indices] != targets[ordered_indices]).astype(np.float64)
        cumulative_risk = np.cumsum(ordered_errors) / np.arange(1, len(indices) + 1)
        coverage_results = {}
        for coverage in requested:
            accepted_count = int(math.ceil(coverage * len(indices)))
            accepted_indices = ordered_indices[:accepted_count]
            accepted_labels = targets[accepted_indices]
            accepted_predictions = predictions[accepted_indices]
            classwise = {
                str(class_id): float(
                    np.sum(accepted_labels == class_id) / np.sum(targets[indices] == class_id)
                )
                for class_id in sorted(set(targets[indices].tolist()))
            }
            negatives = targets[accepted_indices] == 0
            positives = targets[accepted_indices] == 1
            fnr = float(np.mean(accepted_predictions[positives] == 0)) if positives.any() else 0.0
            fpr = float(np.mean(accepted_predictions[negatives] == 1)) if negatives.any() else 0.0
            coverage_results[str(coverage)] = {
                "accepted": accepted_count,
                "total": int(len(indices)),
                "coverage": float(accepted_count / len(indices)),
                "risk": float(np.mean(accepted_predictions != accepted_labels)),
                "classwise_coverage": classwise,
                "fnr": fnr,
                "fpr": fpr,
                "macro_f1": _macro_f1(accepted_predictions, accepted_labels),
                "accepted_sample_ids": ids[accepted_indices].tolist(),
            }
        group_results[group] = {
            "aurc": float(np.mean(cumulative_risk)),
            "coverages": coverage_results,
        }
    return {"groups": group_results}


def equal_mass_ece(
    *,
    sample_ids: Sequence[str],
    groups: Sequence[str],
    confidence: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    bins: int = 15,
) -> dict[str, object]:
    """Compute the registered 15-bin within-bearing equal-mass ECE."""

    confidence_values = np.asarray(confidence, dtype=np.float64)
    ids = np.asarray([str(value) for value in sample_ids], dtype=object)
    group_values = np.asarray([str(value) for value in groups], dtype=object)
    predictions = _labels(predictions, sample_count=len(confidence_values))
    targets = _labels(labels, sample_count=len(confidence_values))
    if (
        confidence_values.ndim != 1
        or ids.shape != confidence_values.shape
        or group_values.shape != confidence_values.shape
    ):
        raise ValueError("ECE inputs must be equal one-dimensional vectors")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("ECE sample_ids must be unique")
    if (
        not np.isfinite(confidence_values).all()
        or np.any(confidence_values < 0.0)
        or np.any(confidence_values > 1.0)
    ):
        raise ValueError("ECE confidence values must be finite and within [0, 1]")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("ECE bins must be a positive integer")

    per_group = {}
    for group in sorted(set(group_values.tolist())):
        indices = np.flatnonzero(group_values == group)
        if len(indices) < bins:
            raise ValueError(
                f"group {group!r} has {len(indices)} samples, fewer than {bins} ECE bins"
            )
        order = np.lexsort((ids[indices], confidence_values[indices]))
        ordered = indices[order]
        quotient, remainder = divmod(len(ordered), bins)
        sizes = [quotient + (1 if index < remainder else 0) for index in range(bins)]
        if min(sizes) <= 0 or max(sizes) - min(sizes) > 1:
            raise AssertionError("equal-mass ECE bin sizes violate the registered contract")
        start = 0
        ece = np.float64(0.0)
        bin_results = []
        for size in sizes:
            selected = ordered[start:start + size]
            start += size
            accuracy = float(np.mean(predictions[selected] == targets[selected]))
            mean_confidence = float(np.mean(confidence_values[selected]))
            ece += np.float64(size / len(ordered) * abs(accuracy - mean_confidence))
            bin_results.append(
                {
                    "count": int(size),
                    "accuracy": accuracy,
                    "mean_confidence": mean_confidence,
                }
            )
        per_group[group] = {"ece": float(ece), "bins": bin_results}
    return {
        "groups": per_group,
        "equal_group_mean_ece": float(
            np.mean([value["ece"] for value in per_group.values()])
        ),
    }


def frozen_threshold_metrics(
    *,
    groups: Sequence[str],
    scores: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, dict[str, float | int]]:
    values = np.asarray(scores, dtype=np.float64)
    group_values = np.asarray([str(value) for value in groups], dtype=object)
    predictions = _labels(predictions, sample_count=len(values))
    targets = _labels(labels, sample_count=len(values))
    if group_values.shape != values.shape or not np.isfinite(values).all() or not math.isfinite(threshold):
        raise ValueError("invalid frozen-threshold inputs")
    results = {}
    for group in sorted(set(group_values.tolist())):
        group_mask = group_values == group
        accepted = group_mask & (values <= threshold)
        accepted_count = int(accepted.sum())
        total = int(group_mask.sum())
        results[group] = {
            "accepted": accepted_count,
            "total": total,
            "coverage": float(accepted_count / total),
            "risk": (
                float(np.mean(predictions[accepted] != targets[accepted]))
                if accepted_count
                else float("nan")
            ),
        }
    return results


def operational_wording_gate(
    trace: Mapping[str, Mapping[str, float]],
    r0: Mapping[str, Mapping[str, float]],
    r1: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    groups = set(trace)
    failures = []
    if groups != set(r0) or groups != set(r1) or len(groups) != 5:
        failures.append("methods must contain the same five bearings")
    for group in sorted(groups & set(r0) & set(r1)):
        values = [
            trace[group].get("coverage"),
            trace[group].get("risk"),
            r0[group].get("coverage"),
            r0[group].get("risk"),
            r1[group].get("coverage"),
            r1[group].get("risk"),
        ]
        if any(value is None or not math.isfinite(float(value)) for value in values):
            failures.append(f"{group}: non-finite or missing result")
            continue
        trace_coverage, trace_risk, r0_coverage, r0_risk, r1_coverage, r1_risk = map(float, values)
        if not 0.85 <= trace_coverage <= 0.95:
            failures.append(f"{group}: trace coverage outside [0.85, 0.95]")
        if abs(trace_coverage - r0_coverage) > 0.05 or abs(trace_coverage - r1_coverage) > 0.05:
            failures.append(f"{group}: comparator coverage difference exceeds 0.05")
        if not trace_risk < r0_risk or not trace_risk < r1_risk:
            failures.append(f"{group}: trace risk is not strictly lower than R0 and R1")
    return {"passed": not failures, "failures": failures, "confirmatory_p_value": None}
