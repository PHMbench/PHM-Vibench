"""Deterministic float64 CPU implementation of the frozen P05-B4 arm."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

from src.data_factory.p05_weighting import WeightPlan


SCHEMA_NAME = "p05.b4_classical_fuzzy"
SCHEMA_VERSION = 1
MODEL_NAME = "model.npz"
PREDICTIONS_NAME = "predictions.npz"
MANIFEST_NAME = "manifest.json"

CHANNELS = 2
FEATURES_PER_CHANNEL = ("mean", "std_population", "rms", "kurtosis_pearson")
FEATURE_COUNT = CHANNELS * len(FEATURES_PER_CHANNEL)
CLUSTERS = 10
FUZZIFIER = 2.0
FCM_INITIALIZATION_SEED = 20260801
FCM_MAX_ITERATIONS = 300
FCM_CENTER_SHIFT_TOLERANCE = 1.0e-5
WINDOW_SECOND_MOMENT_FLOOR = 1.0e-12
FEATURE_STD_FLOOR = 1.0e-8
GAUSSIAN_WIDTH_FLOOR = 1.0e-4

_EXPECTED_CLASSES = {1: 4, 2: 2}
_EXPECTED_WINDOWS_PER_RECORD = {1: 16, 2: 4}
_EXPECTED_TRAIN_WEIGHT_FORMULA = {
    1: "1/(4*n_recordings_in_class*16)",
    2: "1/(10*n_records_in_bearing_class_cell*4)",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class P05B4TrainingSplit:
    """Externally standardized training windows and registered train weights."""

    sample_ids: Sequence[str]
    record_ids: Sequence[int]
    windows: np.ndarray
    labels: Sequence[int]
    weight_plan: WeightPlan


@dataclass(frozen=True)
class P05B4PredictionSplit:
    """One label-free split for ordinary per-window B4 prediction."""

    sample_ids: Sequence[str]
    windows: np.ndarray


@dataclass(frozen=True)
class P05B4Model:
    """All deterministic state required for B4 inference."""

    dataset_id: int
    num_classes: int
    feature_mean: np.ndarray
    feature_std: np.ndarray
    centers: np.ndarray
    widths: np.ndarray
    consequents: np.ndarray


@dataclass(frozen=True)
class P05B4Prediction:
    """Per-window model outputs consumed by the common evaluator."""

    features: np.ndarray
    standardized_features: np.ndarray
    normalized_rule_firing: np.ndarray
    class_scores: np.ndarray
    predicted_labels: np.ndarray


@dataclass(frozen=True)
class P05B4RunResult:
    """Paths and hashes for one create-only dataset fit and prediction package."""

    package_dir: Path
    model_path: Path
    predictions_path: Path
    manifest_path: Path
    dataset_id: int
    iterations: int
    final_max_center_shift: float
    semantic_sha256: str
    model_sha256: str
    predictions_sha256: str
    manifest_sha256: str
    status: str


@dataclass(frozen=True)
class _PreparedTraining:
    dataset_id: int
    num_classes: int
    sample_ids: tuple[str, ...]
    record_ids: tuple[int, ...]
    windows: np.ndarray
    labels: np.ndarray
    sample_weights: np.ndarray
    weight_plan_contract: Mapping[str, Any]


@dataclass(frozen=True)
class _FCMResult:
    initial_memberships: np.ndarray
    memberships: np.ndarray
    centers: np.ndarray
    iterations: int
    final_max_center_shift: float


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    metadata = _canonical_json_bytes(
        {
            "dtype": contiguous.dtype.str,
            "shape": [int(size) for size in contiguous.shape],
        }
    )
    return _sha256_bytes(metadata + b"\0" + contiguous.tobytes(order="C"))


def _array_descriptor(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": contiguous.dtype.str,
        "shape": [int(size) for size in contiguous.shape],
        "sha256": _array_sha256(contiguous),
    }


def _string_sequence(
    values: Sequence[str],
    *,
    name: str,
    count: int,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings")
    normalized = tuple(values)
    if len(normalized) != count:
        raise ValueError(f"{name} length must equal the window count")
    if any(not isinstance(value, str) or not value or "\x00" in value for value in normalized):
        raise ValueError(f"{name} must contain non-empty strings without NUL bytes")
    if len(set(normalized)) != count:
        raise ValueError(f"{name} must be unique")
    return normalized


def _string_array(values: Sequence[str]) -> np.ndarray:
    width = max(len(value) for value in values)
    return np.asarray(tuple(values), dtype=f"<U{width}")


def _record_id(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must contain integer record IDs")
    return int(value)


def _float64_windows(value: Any, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a CPU NumPy array")
    if value.dtype not in {np.dtype("float32"), np.dtype("float64")}:
        raise ValueError(f"{name} must use float32 or float64 storage")
    if value.ndim != 3 or value.shape[0] <= 0 or value.shape[1] < 2:
        raise ValueError(f"{name} must have shape (windows, points, {CHANNELS})")
    if int(value.shape[2]) != CHANNELS:
        raise ValueError(f"{name} must contain exactly {CHANNELS} channels")
    converted = np.ascontiguousarray(value, dtype="<f8")
    if not np.isfinite(converted).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return converted


def _weight_plan_contract(
    plan: WeightPlan,
    *,
    record_ids: tuple[int, ...],
) -> tuple[dict[str, Any], np.ndarray]:
    if not isinstance(plan, WeightPlan):
        raise TypeError("train.weight_plan must be a WeightPlan")
    if type(plan.dataset_id) is not int or plan.dataset_id not in _EXPECTED_CLASSES:
        raise ValueError("P05-B4 weight plan must use registered dataset 1 or 2")
    if plan.role != "train":
        raise ValueError("P05-B4 requires the registered train-only weight plan")
    expected_windows = _EXPECTED_WINDOWS_PER_RECORD[plan.dataset_id]
    if type(plan.windows_per_record) is not int or plan.windows_per_record != expected_windows:
        raise ValueError(
            f"P05 dataset {plan.dataset_id} requires {expected_windows} windows per record"
        )
    expected_formula = _EXPECTED_TRAIN_WEIGHT_FORMULA[plan.dataset_id]
    if plan.formula != expected_formula:
        raise ValueError(
            f"P05-B4 train weight formula must be {expected_formula!r}"
        )
    if not isinstance(plan.record_weights, Mapping) or not plan.record_weights:
        raise TypeError("train weight plan must contain record weights")

    weights_by_record: dict[int, float] = {}
    for raw_record_id, raw_weight in plan.record_weights.items():
        record_id = _record_id(raw_record_id, name="weight_plan.record_weights")
        if record_id in weights_by_record:
            raise ValueError("weight plan contains duplicate canonical record IDs")
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, Real):
            raise TypeError("weight plan values must be real numbers")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("weight plan values must be finite and positive")
        weights_by_record[record_id] = weight

    counts = Counter(record_ids)
    if set(counts) != set(weights_by_record):
        raise ValueError("training records must exactly cover the registered weight plan")
    if any(count != expected_windows for count in counts.values()):
        raise ValueError(
            f"every P05 dataset {plan.dataset_id} training record must have "
            f"{expected_windows} windows"
        )

    rows = [
        {"Id": record_id, "window_weight": weights_by_record[record_id]}
        for record_id in sorted(weights_by_record)
    ]
    contract = {
        "schema_version": 1,
        "paper_id": "P05",
        "dataset_id": plan.dataset_id,
        "role": "train",
        "windows_per_record": expected_windows,
        "formula": expected_formula,
        "normalization": "mean_train_or_evaluation_window_weight_equals_one",
        "record_weights": rows,
    }
    expected_hash = _sha256_bytes(_canonical_json_bytes(contract))
    if _required_sha256(plan.sha256, name="weight_plan.sha256") != expected_hash:
        raise ValueError("weight_plan source SHA-256 does not match its contract")

    sample_weights = np.asarray(
        [weights_by_record[record_id] for record_id in record_ids],
        dtype="<f8",
    )
    if not np.isclose(
        sample_weights.mean(dtype=np.float64),
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("registered train-window weights must have mean one")
    return {**contract, "sha256": expected_hash}, sample_weights


def _prepare_training(train: P05B4TrainingSplit) -> _PreparedTraining:
    if not isinstance(train, P05B4TrainingSplit):
        raise TypeError("train must be a P05B4TrainingSplit")
    windows = _float64_windows(train.windows, name="train.windows")
    count = int(windows.shape[0])
    if count < CLUSTERS:
        raise ValueError(f"P05-B4 needs at least {CLUSTERS} training windows")
    sample_ids = _string_sequence(train.sample_ids, name="train.sample_ids", count=count)
    if isinstance(train.record_ids, (str, bytes)):
        raise TypeError("train.record_ids must be an integer sequence")
    record_ids = tuple(
        _record_id(value, name="train.record_ids") for value in train.record_ids
    )
    if len(record_ids) != count:
        raise ValueError("train.record_ids length must equal the window count")
    plan_contract, sample_weights = _weight_plan_contract(
        train.weight_plan,
        record_ids=record_ids,
    )
    dataset_id = int(train.weight_plan.dataset_id)
    num_classes = _EXPECTED_CLASSES[dataset_id]

    if isinstance(train.labels, (str, bytes)):
        raise TypeError("train.labels must be an integer sequence")
    raw_labels = tuple(train.labels)
    if len(raw_labels) != count:
        raise ValueError("train.labels length must equal the window count")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw_labels):
        raise TypeError("train.labels must contain integer class IDs")
    labels = np.asarray(raw_labels, dtype="<i8")
    expected_labels = set(range(num_classes))
    observed_labels = set(int(value) for value in labels)
    if observed_labels != expected_labels:
        raise ValueError(
            f"P05 dataset {dataset_id} training labels must be {sorted(expected_labels)}"
        )
    label_by_record: dict[int, int] = {}
    for record_id, label in zip(record_ids, labels):
        previous = label_by_record.setdefault(record_id, int(label))
        if previous != int(label):
            raise ValueError(f"training record {record_id} maps to multiple labels")

    return _PreparedTraining(
        dataset_id=dataset_id,
        num_classes=num_classes,
        sample_ids=sample_ids,
        record_ids=record_ids,
        windows=windows,
        labels=labels,
        sample_weights=sample_weights,
        weight_plan_contract=plan_contract,
    )


def p05_b4_extract_features(windows: np.ndarray) -> np.ndarray:
    """Compute per-channel Mean/Std(pop)/RMS/Pearson kurtosis in float64."""

    values = _float64_windows(windows, name="windows")
    means = values.mean(axis=1, dtype=np.float64)
    centered = values - means[:, None, :]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        second_moments = np.square(centered).mean(axis=1, dtype=np.float64)
        root_mean_squares = np.sqrt(
            np.square(values).mean(axis=1, dtype=np.float64)
        )
        fourth_moments = np.square(np.square(centered)).mean(
            axis=1,
            dtype=np.float64,
        )
        kurtosis = fourth_moments / np.square(second_moments)
    if not np.isfinite(second_moments).all():
        raise FloatingPointError("P05-B4 window second moments are non-finite")
    below = np.argwhere(second_moments < WINDOW_SECOND_MOMENT_FLOOR)
    if below.size:
        window_index, channel_index = (int(value) for value in below[0])
        raise ValueError(
            "P05-B4 window second central moment is below 1e-12 at "
            f"window={window_index}, channel={channel_index}"
        )
    features = np.stack(
        (means, np.sqrt(second_moments), root_mean_squares, kurtosis),
        axis=2,
    ).reshape(values.shape[0], FEATURE_COUNT)
    features = np.ascontiguousarray(features, dtype="<f8")
    if not np.isfinite(features).all():
        raise FloatingPointError("P05-B4 extracted features are non-finite")
    return features


def _weighted_feature_standardization(
    features: np.ndarray,
    sample_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_weight = sample_weights.sum(dtype=np.float64)
    if not math.isfinite(float(total_weight)) or total_weight <= 0.0:
        raise ValueError("P05-B4 training weight sum must be finite and positive")
    feature_mean = (
        (sample_weights[:, None] * features).sum(axis=0, dtype=np.float64)
        / total_weight
    )
    centered = features - feature_mean
    feature_variance = (
        (sample_weights[:, None] * np.square(centered)).sum(
            axis=0,
            dtype=np.float64,
        )
        / total_weight
    )
    if not np.isfinite(feature_mean).all() or not np.isfinite(feature_variance).all():
        raise FloatingPointError("P05-B4 weighted feature moments are non-finite")
    if np.any(feature_variance < 0.0):
        raise ValueError("P05-B4 weighted feature variance must be non-negative")
    feature_std = np.sqrt(feature_variance)
    below = np.flatnonzero(feature_std < FEATURE_STD_FLOOR)
    if below.size:
        raise ValueError(
            "P05-B4 weighted feature standard deviation is below 1e-8 at "
            f"feature={int(below[0])}"
        )
    standardized = np.ascontiguousarray(centered / feature_std, dtype="<f8")
    return (
        standardized,
        np.ascontiguousarray(feature_mean, dtype="<f8"),
        np.ascontiguousarray(feature_std, dtype="<f8"),
    )


def _initialize_memberships(sample_count: int) -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64(FCM_INITIALIZATION_SEED))
    memberships = generator.random((sample_count, CLUSTERS), dtype=np.float64)
    memberships /= memberships.sum(axis=1, keepdims=True, dtype=np.float64)
    return np.ascontiguousarray(memberships, dtype="<f8")


def _weighted_centers(
    features: np.ndarray,
    sample_weights: np.ndarray,
    memberships: np.ndarray,
) -> np.ndarray:
    effective_weights = sample_weights[:, None] * np.square(memberships)
    denominators = effective_weights.sum(axis=0, dtype=np.float64)
    if not np.isfinite(denominators).all() or np.any(denominators <= 0.0):
        raise RuntimeError("P05-B4 fuzzy c-means produced an empty cluster")
    centers = effective_weights.T @ features
    centers /= denominators[:, None]
    if not np.isfinite(centers).all():
        raise FloatingPointError("P05-B4 fuzzy c-means centers are non-finite")
    return np.ascontiguousarray(centers, dtype="<f8")


def _memberships_for_centers(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
    if not np.isfinite(distances).all():
        raise FloatingPointError("P05-B4 fuzzy c-means distances are non-finite")
    memberships = np.zeros_like(distances, dtype="<f8")
    zero_mask = distances == 0.0
    rows_with_zero = zero_mask.any(axis=1)
    if np.any(rows_with_zero):
        zero_counts = zero_mask[rows_with_zero].sum(axis=1, keepdims=True)
        memberships[rows_with_zero] = zero_mask[rows_with_zero] / zero_counts

    nonzero_rows = ~rows_with_zero
    if np.any(nonzero_rows):
        exponent = 2.0 / (FUZZIFIER - 1.0)
        log_inverse = -exponent * np.log(distances[nonzero_rows])
        log_inverse -= log_inverse.max(axis=1, keepdims=True)
        inverse = np.exp(log_inverse)
        memberships[nonzero_rows] = inverse / inverse.sum(
            axis=1,
            keepdims=True,
            dtype=np.float64,
        )
    if not np.isfinite(memberships).all() or not np.allclose(
        memberships.sum(axis=1, dtype=np.float64),
        1.0,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise RuntimeError("P05-B4 fuzzy c-means memberships failed normalization")
    return np.ascontiguousarray(memberships, dtype="<f8")


def _fit_weighted_fuzzy_c_means(
    features: np.ndarray,
    sample_weights: np.ndarray,
    *,
    max_iterations: int = FCM_MAX_ITERATIONS,
    tolerance: float = FCM_CENTER_SHIFT_TOLERANCE,
) -> _FCMResult:
    initial_memberships = _initialize_memberships(int(features.shape[0]))
    memberships = initial_memberships.copy()
    centers = _weighted_centers(features, sample_weights, memberships)
    for iteration in range(1, max_iterations + 1):
        updated_memberships = _memberships_for_centers(features, centers)
        updated_centers = _weighted_centers(
            features,
            sample_weights,
            updated_memberships,
        )
        center_shifts = np.linalg.norm(updated_centers - centers, axis=1)
        maximum_shift = float(center_shifts.max())
        if not math.isfinite(maximum_shift):
            raise FloatingPointError("P05-B4 fuzzy c-means center shift is non-finite")
        memberships = updated_memberships
        centers = updated_centers
        if maximum_shift <= tolerance:
            return _FCMResult(
                initial_memberships=initial_memberships,
                memberships=memberships,
                centers=centers,
                iterations=iteration,
                final_max_center_shift=maximum_shift,
            )
    raise RuntimeError(
        "P05-B4 fuzzy c-means failed to converge within 300 updates"
    )


def _derive_widths_and_consequents(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    memberships: np.ndarray,
    centers: np.ndarray,
    *,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    effective_weights = sample_weights[:, None] * np.square(memberships)
    total_mass = effective_weights.sum(axis=0, dtype=np.float64)
    squared_deviation = np.square(features[:, None, :] - centers[None, :, :])
    width_variance = (
        (effective_weights[:, :, None] * squared_deviation).sum(
            axis=0,
            dtype=np.float64,
        )
        / total_mass[:, None]
    )
    widths = np.maximum(np.sqrt(width_variance), GAUSSIAN_WIDTH_FLOOR)

    class_mass = np.empty((CLUSTERS, num_classes), dtype="<f8")
    for class_id in range(num_classes):
        class_mass[:, class_id] = effective_weights[labels == class_id].sum(
            axis=0,
            dtype=np.float64,
        )
    consequents = (1.0 + class_mass) / (num_classes + total_mass[:, None])
    if not np.isfinite(widths).all() or not np.isfinite(consequents).all():
        raise FloatingPointError("P05-B4 widths or consequents are non-finite")
    if np.any(widths < GAUSSIAN_WIDTH_FLOOR):
        raise AssertionError("P05-B4 Gaussian width floor was not applied")
    if not np.allclose(
        consequents.sum(axis=1, dtype=np.float64),
        1.0,
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise AssertionError("P05-B4 Laplace consequents are not normalized")
    return (
        np.ascontiguousarray(widths, dtype="<f8"),
        np.ascontiguousarray(consequents, dtype="<f8"),
    )


def _predict_from_features(features: np.ndarray, model: P05B4Model) -> P05B4Prediction:
    standardized = (features - model.feature_mean) / model.feature_std
    standardized = np.ascontiguousarray(standardized, dtype="<f8")
    scaled = (
        standardized[:, None, :] - model.centers[None, :, :]
    ) / model.widths[None, :, :]
    log_geometric_firing = (-0.5 * np.square(scaled)).mean(
        axis=2,
        dtype=np.float64,
    )
    shifted = log_geometric_firing - log_geometric_firing.max(axis=1, keepdims=True)
    firing = np.exp(shifted)
    firing /= firing.sum(axis=1, keepdims=True, dtype=np.float64)
    scores = firing @ model.consequents
    predicted_labels = np.argmax(scores, axis=1).astype("<i8", copy=False)
    if not np.isfinite(standardized).all() or not np.isfinite(firing).all():
        raise FloatingPointError("P05-B4 prediction produced non-finite values")
    if not np.isfinite(scores).all():
        raise FloatingPointError("P05-B4 class scores are non-finite")
    return P05B4Prediction(
        features=np.ascontiguousarray(features, dtype="<f8"),
        standardized_features=standardized,
        normalized_rule_firing=np.ascontiguousarray(firing, dtype="<f8"),
        class_scores=np.ascontiguousarray(scores, dtype="<f8"),
        predicted_labels=np.ascontiguousarray(predicted_labels, dtype="<i8"),
    )


def _prepare_predictions(
    prediction_splits: Mapping[str, P05B4PredictionSplit],
    *,
    model: P05B4Model,
) -> tuple[
    dict[str, tuple[tuple[str, ...], np.ndarray]],
    dict[str, P05B4Prediction],
]:
    if not isinstance(prediction_splits, Mapping) or not prediction_splits:
        raise TypeError("prediction_splits must be a non-empty mapping")
    allowed_roles = {"train", "validation", "test"}
    prepared: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    outputs: dict[str, P05B4Prediction] = {}
    all_sample_ids: set[str] = set()
    for role in sorted(prediction_splits):
        if role not in allowed_roles:
            raise ValueError(f"unknown P05-B4 prediction role {role!r}")
        split = prediction_splits[role]
        if not isinstance(split, P05B4PredictionSplit):
            raise TypeError(f"prediction_splits[{role!r}] must be P05B4PredictionSplit")
        windows = _float64_windows(split.windows, name=f"{role}.windows")
        sample_ids = _string_sequence(
            split.sample_ids,
            name=f"{role}.sample_ids",
            count=int(windows.shape[0]),
        )
        overlap = all_sample_ids.intersection(sample_ids)
        if overlap:
            raise ValueError("P05-B4 prediction sample IDs must be disjoint across roles")
        all_sample_ids.update(sample_ids)
        features = p05_b4_extract_features(windows)
        prepared[role] = (sample_ids, windows)
        outputs[role] = _predict_from_features(features, model)
    return prepared, outputs


def _model_arrays(
    model: P05B4Model,
    fcm: _FCMResult,
) -> dict[str, np.ndarray]:
    return {
        "centers": model.centers,
        "consequents": model.consequents,
        "fcm_final_memberships": fcm.memberships,
        "fcm_initial_memberships": fcm.initial_memberships,
        "feature_mean": model.feature_mean,
        "feature_std": model.feature_std,
        "widths": model.widths,
    }


def _prediction_arrays(
    prepared: Mapping[str, tuple[tuple[str, ...], np.ndarray]],
    outputs: Mapping[str, P05B4Prediction],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for role in sorted(outputs):
        sample_ids, _windows = prepared[role]
        prediction = outputs[role]
        arrays[f"{role}__sample_ids"] = _string_array(sample_ids)
        arrays[f"{role}__features"] = prediction.features
        arrays[f"{role}__standardized_features"] = prediction.standardized_features
        arrays[f"{role}__normalized_rule_firing"] = prediction.normalized_rule_firing
        arrays[f"{role}__class_scores"] = prediction.class_scores
        arrays[f"{role}__predicted_labels"] = prediction.predicted_labels
    return arrays


def _semantic_manifest(
    *,
    training: _PreparedTraining,
    train_features: np.ndarray,
    standardized_train_features: np.ndarray,
    model: P05B4Model,
    fcm: _FCMResult,
    prepared_predictions: Mapping[str, tuple[tuple[str, ...], np.ndarray]],
    predictions: Mapping[str, P05B4Prediction],
    model_arrays: Mapping[str, np.ndarray],
    prediction_arrays: Mapping[str, np.ndarray],
    channel_standardization_sha256: str,
    split_manifest_sha256: str,
    signal_cache_manifest_sha256: str,
    expected_window_size: int,
) -> dict[str, Any]:
    feature_names = [
        f"channel_{channel}_{feature}"
        for channel in range(CHANNELS)
        for feature in FEATURES_PER_CHANNEL
    ]
    prediction_provenance = {}
    for role in sorted(predictions):
        sample_ids, windows = prepared_predictions[role]
        prediction = predictions[role]
        prediction_provenance[role] = {
            "sample_count": len(sample_ids),
            "sample_ids_sha256": _array_sha256(_string_array(sample_ids)),
            "externally_standardized_windows": _array_descriptor(windows),
            "features": _array_descriptor(prediction.features),
            "standardized_features": _array_descriptor(
                prediction.standardized_features
            ),
            "normalized_rule_firing": _array_descriptor(
                prediction.normalized_rule_firing
            ),
            "class_scores": _array_descriptor(prediction.class_scores),
            "predicted_labels": _array_descriptor(prediction.predicted_labels),
        }
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "paper_id": "P05",
        "baseline_id": "P05-B4",
        "role": "classical_fuzzy",
        "evidence_status": "unadjudicated",
        "dataset_id": training.dataset_id,
        "fit_id": f"P05-B4-dataset-{training.dataset_id}",
        "device": "cpu",
        "precision": "float64",
        "artifact_policy": "atomic_create_only_model_and_predictions",
        "fit_contract": {
            "fits_per_dataset": 1,
            "validation_grid": "none",
            "model_seed_repetition": "forbidden_as_redundant_deterministic_fit",
            "feature_names": feature_names,
            "feature_source": "externally_channel_standardized_windows",
            "window_size": expected_window_size,
            "window_second_moment_below_1e-12": "hard_error",
            "feature_standardization": (
                "train_only_registered_train_window_weighted_population_moments"
            ),
            "feature_standard_deviation_below_1e-8": "hard_error",
            "clustering": {
                "method": "fuzzy_c_means",
                "clusters": CLUSTERS,
                "fuzzifier": FUZZIFIER,
                "initialization_seed": FCM_INITIALIZATION_SEED,
                "initialization": (
                    "n_by_10_PCG64_independent_uniform_then_row_normalize"
                ),
                "update_weight": "train_window_weight_times_membership_squared",
                "distance": "Euclidean",
                "zero_distance": (
                    "split_mass_equally_among_exact_zero_distance_centers"
                ),
                "max_iterations": FCM_MAX_ITERATIONS,
                "stop": "maximum_center_L2_shift_at_most_1e-5",
                "iterations": fcm.iterations,
                "final_max_center_shift": fcm.final_max_center_shift,
                "converged": True,
            },
            "gaussian_width": (
                "weighted_RMS_feature_deviation_using_train_weight_times_membership_squared"
            ),
            "gaussian_width_floor": GAUSSIAN_WIDTH_FLOOR,
            "consequents": (
                "one_plus_weighted_class_mass_divided_by_K_plus_total_weighted_mass"
            ),
            "prediction_firing": (
                "normalized_geometric_mean_Gaussian_across_features"
            ),
            "prediction_score": "firing_weighted_sum_of_rule_class_consequents",
            "argmax_tie": "lower_class_id",
        },
        "model": {
            "num_features": FEATURE_COUNT,
            "num_rules": CLUSTERS,
            "num_classes": model.num_classes,
            "arrays": {
                name: _array_descriptor(array)
                for name, array in sorted(model_arrays.items())
            },
        },
        "provenance": {
            "channel_standardization_sha256": channel_standardization_sha256,
            "split_manifest_sha256": split_manifest_sha256,
            "signal_cache_manifest_sha256": signal_cache_manifest_sha256,
            "software": {"numpy": np.__version__},
            "train": {
                "sample_count": len(training.sample_ids),
                "sample_ids_sha256": _array_sha256(
                    _string_array(training.sample_ids)
                ),
                "record_ids": _array_descriptor(
                    np.asarray(training.record_ids, dtype="<i8")
                ),
                "externally_standardized_windows": _array_descriptor(training.windows),
                "labels": _array_descriptor(training.labels),
                "sample_weights": _array_descriptor(training.sample_weights),
                "weight_plan": training.weight_plan_contract,
                "features": _array_descriptor(train_features),
                "standardized_features": _array_descriptor(
                    standardized_train_features
                ),
            },
            "predictions": prediction_provenance,
        },
        "prediction_artifact_arrays": {
            name: _array_descriptor(array)
            for name, array in sorted(prediction_arrays.items())
        },
    }


def _assert_create_only_target(target: Path) -> None:
    if target.is_symlink():
        raise FileExistsError(f"refusing create-only P05-B4 export through symlink: {target}")
    if target.exists():
        raise FileExistsError(f"P05-B4 artifact conflicts with existing target: {target}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic create-only P05-B4 export requires Linux renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            "P05-B4 artifact conflicts with existing target",
            str(target),
        )
    raise OSError(error_number, os.strerror(error_number), str(target))


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as handle:
        np.savez(handle, **{name: arrays[name] for name in sorted(arrays)})
        handle.flush()
        os.fsync(handle.fileno())


def _write_bytes(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _write_package(
    target: Path,
    *,
    model_arrays: Mapping[str, np.ndarray],
    prediction_arrays: Mapping[str, np.ndarray],
    semantic_manifest: Mapping[str, Any],
    iterations: int,
    final_max_center_shift: float,
    dataset_id: int,
) -> P05B4RunResult:
    _assert_create_only_target(target)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("P05-B4 artifact parent must be a real directory")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=parent)
    )
    try:
        model_path = temporary / MODEL_NAME
        predictions_path = temporary / PREDICTIONS_NAME
        manifest_path = temporary / MANIFEST_NAME
        _write_npz(model_path, model_arrays)
        _write_npz(predictions_path, prediction_arrays)
        semantic_sha256 = _sha256_bytes(_canonical_json_bytes(semantic_manifest))
        manifest = {
            **semantic_manifest,
            "semantic_sha256": semantic_sha256,
            "files": {
                MODEL_NAME: _sha256_file(model_path),
                PREDICTIONS_NAME: _sha256_file(predictions_path),
            },
        }
        _write_bytes(manifest_path, _pretty_json_bytes(manifest))
        _fsync_directory(temporary)
        _rename_directory_noreplace(temporary, target)
        _fsync_directory(parent)
        return P05B4RunResult(
            package_dir=target,
            model_path=target / MODEL_NAME,
            predictions_path=target / PREDICTIONS_NAME,
            manifest_path=target / MANIFEST_NAME,
            dataset_id=dataset_id,
            iterations=iterations,
            final_max_center_shift=final_max_center_shift,
            semantic_sha256=semantic_sha256,
            model_sha256=_sha256_file(target / MODEL_NAME),
            predictions_sha256=_sha256_file(target / PREDICTIONS_NAME),
            manifest_sha256=_sha256_file(target / MANIFEST_NAME),
            status="created",
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def run_p05_b4_classical_fuzzy(
    package_dir: str | Path,
    *,
    train: P05B4TrainingSplit,
    prediction_splits: Mapping[str, P05B4PredictionSplit],
    channel_standardization_sha256: str,
    split_manifest_sha256: str,
    signal_cache_manifest_sha256: str,
    expected_window_size: int = 4096,
) -> P05B4RunResult:
    """Fit B4 exactly once and atomically create model/prediction provenance.

    There is intentionally no ``model_seed`` argument. The only randomness is
    the frozen PCG64 fuzzy-c-means initialization seed 20260801, so repeating
    B4 under neural model-seed labels would be a protocol violation.
    """

    target = Path(os.path.abspath(os.fspath(package_dir)))
    _assert_create_only_target(target)
    if type(expected_window_size) is not int or expected_window_size <= 0:
        raise ValueError("expected_window_size must be a positive integer")
    normalization_hash = _required_sha256(
        channel_standardization_sha256,
        name="channel_standardization_sha256",
    )
    split_hash = _required_sha256(
        split_manifest_sha256,
        name="split_manifest_sha256",
    )
    cache_hash = _required_sha256(
        signal_cache_manifest_sha256,
        name="signal_cache_manifest_sha256",
    )
    training = _prepare_training(train)
    if int(training.windows.shape[1]) != expected_window_size:
        raise ValueError(
            "P05-B4 training windows must use the registered window size "
            f"{expected_window_size}, got {training.windows.shape[1]}"
        )
    train_features = p05_b4_extract_features(training.windows)
    standardized_features, feature_mean, feature_std = (
        _weighted_feature_standardization(
            train_features,
            training.sample_weights,
        )
    )
    fcm = _fit_weighted_fuzzy_c_means(
        standardized_features,
        training.sample_weights,
    )
    widths, consequents = _derive_widths_and_consequents(
        standardized_features,
        training.labels,
        training.sample_weights,
        fcm.memberships,
        fcm.centers,
        num_classes=training.num_classes,
    )
    model = P05B4Model(
        dataset_id=training.dataset_id,
        num_classes=training.num_classes,
        feature_mean=feature_mean,
        feature_std=feature_std,
        centers=fcm.centers,
        widths=widths,
        consequents=consequents,
    )
    prepared_predictions, predictions = _prepare_predictions(
        prediction_splits,
        model=model,
    )
    for role, (_sample_ids, windows) in prepared_predictions.items():
        if int(windows.shape[1]) != expected_window_size:
            raise ValueError(
                f"P05-B4 {role} windows must use the registered window size "
                f"{expected_window_size}, got {windows.shape[1]}"
            )
    model_arrays = _model_arrays(model, fcm)
    prediction_arrays = _prediction_arrays(prepared_predictions, predictions)
    semantic_manifest = _semantic_manifest(
        training=training,
        train_features=train_features,
        standardized_train_features=standardized_features,
        model=model,
        fcm=fcm,
        prepared_predictions=prepared_predictions,
        predictions=predictions,
        model_arrays=model_arrays,
        prediction_arrays=prediction_arrays,
        channel_standardization_sha256=normalization_hash,
        split_manifest_sha256=split_hash,
        signal_cache_manifest_sha256=cache_hash,
        expected_window_size=expected_window_size,
    )
    return _write_package(
        target,
        model_arrays=model_arrays,
        prediction_arrays=prediction_arrays,
        semantic_manifest=semantic_manifest,
        iterations=fcm.iterations,
        final_max_center_shift=fcm.final_max_center_shift,
        dataset_id=training.dataset_id,
    )
