"""Frozen record-level and E1 metrics for the P08-LOSO-v1.1 protocol."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Hashable, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


NUM_CLASSES = 4
PROBABILITY_SUM_TOLERANCE = 1.0e-6
NLL_CLIP_MINIMUM = 1.0e-12
ECE_BIN_COUNT = 15
DEFAULT_E1_CLASSES = (0, 1, 2, 3)
DEFAULT_E1_SEEDS = (42, 123, 456, 789, 999)
DEFAULT_E1_RATES_HZ = (12000, 20480, 25600, 48000, 50000, 200000)


def _probabilities(value: ArrayLike) -> NDArray[np.float64]:
    probabilities = np.asarray(value, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"probabilities must have shape (N, {NUM_CLASSES}), got "
            f"{probabilities.shape}"
        )
    if probabilities.shape[0] == 0:
        raise ValueError("probabilities cannot be empty")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities contain non-finite values")
    if np.any(probabilities < -PROBABILITY_SUM_TOLERANCE) or np.any(
        probabilities > 1.0 + PROBABILITY_SUM_TOLERANCE
    ):
        raise ValueError("inputs are not four-class probabilities in [0, 1]")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(
        row_sums, 1.0, rtol=0.0, atol=PROBABILITY_SUM_TOLERANCE
    ):
        raise ValueError("four-class probabilities must sum to one per row")

    # Correct only floating-point boundary noise across the full four-class
    # vector.  Target-supported subsets are never renormalized.
    clipped = np.clip(probabilities, 0.0, 1.0)
    return clipped / clipped.sum(axis=1, keepdims=True)


def _integer_vector(value: ArrayLike, name: str) -> NDArray[np.int64]:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if raw.size == 0:
        raise ValueError(f"{name} cannot be empty")
    try:
        converted = raw.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integers") from exc
    try:
        numerically_equal = np.equal(raw, converted)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integers") from exc
    if not np.all(numerically_equal):
        raise ValueError(f"{name} must contain exact integers")
    return converted


def _canonical_identifier(value: Any) -> tuple[str, Hashable]:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("signal/group identifiers must be strings or non-boolean integers")
    return type(value).__name__, value


def _supported_classes(classes: Iterable[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in classes)
    if not result or len(set(result)) != len(result):
        raise ValueError("supported_classes must be a non-empty unique sequence")
    if any(value < 0 or value >= NUM_CLASSES for value in result):
        raise ValueError(f"supported classes must be within [0, {NUM_CLASSES - 1}]")
    return result


@dataclass(frozen=True)
class E1Predictions:
    """Columnar uncalibrated predictions for one E1 arm.

    There must be one row per ``(class, underlying signal, model seed, rate)``.
    Underlying signal identifiers may repeat between classes because class is
    part of the frozen analytic-bank identity.
    """

    probabilities: NDArray[np.float64]
    labels: NDArray[np.int64]
    signal_ids: NDArray[np.object_]
    model_seeds: NDArray[np.int64]
    rates_hz: NDArray[np.int64]

    @classmethod
    def from_columns(
        cls,
        *,
        probabilities: ArrayLike,
        labels: ArrayLike,
        signal_ids: Sequence[str | int],
        model_seeds: ArrayLike,
        rates_hz: ArrayLike,
    ) -> "E1Predictions":
        probability_array = _probabilities(probabilities)
        label_array = _integer_vector(labels, "labels")
        seed_array = _integer_vector(model_seeds, "model_seeds")
        rate_array = _integer_vector(rates_hz, "rates_hz")
        id_array = np.asarray(signal_ids, dtype=object)
        if id_array.ndim != 1:
            raise ValueError("signal_ids must be one-dimensional")
        row_count = probability_array.shape[0]
        if any(
            len(column) != row_count
            for column in (label_array, id_array, seed_array, rate_array)
        ):
            raise ValueError("all E1 columns must have the same row count")
        for identifier in id_array:
            _canonical_identifier(identifier)
        return cls(
            probabilities=probability_array,
            labels=label_array,
            signal_ids=id_array,
            model_seeds=seed_array,
            rates_hz=rate_array,
        )


def aggregate_probabilities_by_group(
    *,
    probabilities: ArrayLike,
    group_ids: Sequence[str | int],
    labels: ArrayLike | None = None,
) -> dict[str, Any]:
    """Arithmetic-mean window probabilities into record/signal probabilities."""

    probability_array = _probabilities(probabilities)
    id_array = np.asarray(group_ids, dtype=object)
    if id_array.ndim != 1 or len(id_array) != len(probability_array):
        raise ValueError("group_ids must provide exactly one identifier per row")
    label_array = None if labels is None else _integer_vector(labels, "labels")
    if label_array is not None and len(label_array) != len(probability_array):
        raise ValueError("labels must provide exactly one value per row")

    indices_by_id: dict[tuple[str, Hashable], list[int]] = {}
    original_ids: dict[tuple[str, Hashable], str | int] = {}
    for index, identifier in enumerate(id_array):
        key = _canonical_identifier(identifier)
        indices_by_id.setdefault(key, []).append(index)
        original_ids[key] = identifier.item() if isinstance(identifier, np.generic) else identifier

    ordered_keys = sorted(indices_by_id, key=lambda item: (item[0], repr(item[1])))
    grouped_probabilities: list[NDArray[np.float64]] = []
    grouped_labels: list[int] = []
    counts: list[int] = []
    for key in ordered_keys:
        indices = np.asarray(indices_by_id[key], dtype=np.int64)
        grouped_probabilities.append(probability_array[indices].mean(axis=0))
        counts.append(int(len(indices)))
        if label_array is not None:
            unique_labels = np.unique(label_array[indices])
            if len(unique_labels) != 1:
                raise ValueError(
                    f"group {original_ids[key]!r} has inconsistent true labels"
                )
            grouped_labels.append(int(unique_labels[0]))

    result: dict[str, Any] = {
        "group_ids": np.asarray([original_ids[key] for key in ordered_keys], dtype=object),
        "probabilities": np.stack(grouped_probabilities),
        "counts": np.asarray(counts, dtype=np.int64),
        "aggregation": "arithmetic_mean_probabilities",
    }
    if label_array is not None:
        result["labels"] = np.asarray(grouped_labels, dtype=np.int64)
    return result


def balanced_accuracy(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    supported_classes: Iterable[int],
) -> float:
    truth = _integer_vector(y_true, "y_true")
    prediction = _integer_vector(y_pred, "y_pred")
    if len(truth) != len(prediction):
        raise ValueError("y_true and y_pred lengths differ")
    if np.any(prediction < 0) or np.any(prediction >= NUM_CLASSES):
        raise ValueError(f"y_pred values must be within [0, {NUM_CLASSES - 1}]")
    classes = _supported_classes(supported_classes)
    if not set(np.unique(truth)).issubset(classes):
        raise ValueError("y_true contains a class outside the prespecified support")
    recalls = []
    for class_id in classes:
        mask = truth == class_id
        denominator = int(mask.sum())
        if denominator == 0:
            raise ValueError(f"supported class {class_id} has no true observations")
        recalls.append(float(np.mean(prediction[mask] == class_id)))
    return float(np.mean(recalls))


def record_classification_metrics(
    *,
    probabilities: ArrayLike,
    labels: ArrayLike,
    supported_classes: Iterable[int],
) -> dict[str, Any]:
    """Compute frozen uncalibrated record/signal classification metrics."""

    probability_array = _probabilities(probabilities)
    truth = _integer_vector(labels, "labels")
    if len(truth) != len(probability_array):
        raise ValueError("labels and probabilities lengths differ")
    classes = _supported_classes(supported_classes)
    if not set(np.unique(truth)).issubset(classes):
        raise ValueError("labels contain a class outside the prespecified support")

    prediction = np.argmax(probability_array, axis=1).astype(np.int64)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for true_value, predicted_value in zip(truth, prediction, strict=True):
        confusion[true_value, predicted_value] += 1

    recalls: dict[str, dict[str, float | int]] = {}
    f1_terms: dict[str, dict[str, float | int]] = {}
    class_f1_values: list[float] = []
    for class_id in classes:
        true_positive = int(confusion[class_id, class_id])
        false_negative = int(confusion[class_id, :].sum() - true_positive)
        false_positive = int(confusion[:, class_id].sum() - true_positive)
        support = true_positive + false_negative
        if support == 0:
            raise ValueError(f"supported class {class_id} has no true observations")
        recall = true_positive / support
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 0.0 if denominator == 0 else (2 * true_positive) / denominator
        recalls[str(class_id)] = {
            "numerator": true_positive,
            "denominator": support,
            "value": float(recall),
        }
        f1_terms[str(class_id)] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "value": float(f1),
        }
        class_f1_values.append(float(f1))

    confidence = np.max(probability_array, axis=1)
    correct = prediction == truth
    bin_indices = np.minimum((confidence * ECE_BIN_COUNT).astype(np.int64), 14)
    ece = 0.0
    ece_bins: list[dict[str, float | int]] = []
    for bin_id in range(ECE_BIN_COUNT):
        mask = bin_indices == bin_id
        count = int(mask.sum())
        bin_accuracy = float(np.mean(correct[mask])) if count else 0.0
        mean_confidence = float(np.mean(confidence[mask])) if count else 0.0
        contribution = (
            (count / len(truth)) * abs(bin_accuracy - mean_confidence)
            if count
            else 0.0
        )
        ece += contribution
        ece_bins.append(
            {
                "bin": bin_id,
                "count": count,
                "accuracy": bin_accuracy,
                "mean_confidence": mean_confidence,
                "contribution": float(contribution),
            }
        )

    true_probabilities = probability_array[np.arange(len(truth)), truth]
    nll = -float(np.mean(np.log(np.clip(true_probabilities, NLL_CLIP_MINIMUM, 1.0))))
    macro_f1 = float(np.mean(class_f1_values))
    four_class_macro_f1 = macro_f1 if set(classes) == set(range(NUM_CLASSES)) else None
    return {
        "scoring_unit": "raw_recording_or_underlying_signal",
        "probability_source": "uncalibrated_softmax",
        "probability_dimension": NUM_CLASSES,
        "renormalized_over_supported_classes": False,
        "num_scored_units": int(len(truth)),
        "supported_classes": list(classes),
        "accuracy": float(np.mean(correct)),
        "balanced_accuracy": float(np.mean([item["value"] for item in recalls.values()])),
        "macro_f1": macro_f1,
        "four_class_macro_f1": four_class_macro_f1,
        "ece_15_bin": float(ece),
        "negative_log_likelihood": nll,
        "per_class_recall": recalls,
        "per_class_f1": f1_terms,
        "ece_bins": ece_bins,
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
    }


@dataclass(frozen=True)
class _E1Layout:
    classes: tuple[int, ...]
    seeds: tuple[int, ...]
    rates_hz: tuple[int, ...]
    signal_keys_by_class: tuple[tuple[tuple[str, Hashable], ...], ...]
    index_cubes: tuple[NDArray[np.int64], ...]


def _e1_layout(
    table: E1Predictions,
    *,
    classes: Iterable[int],
    seeds: Iterable[int],
    rates_hz: Iterable[int],
) -> _E1Layout:
    expected_classes = _supported_classes(classes)
    expected_seeds = tuple(int(value) for value in seeds)
    expected_rates = tuple(int(value) for value in rates_hz)
    if len(set(expected_seeds)) != len(expected_seeds) or not expected_seeds:
        raise ValueError("expected E1 seeds must be non-empty and unique")
    if len(set(expected_rates)) != len(expected_rates) or len(expected_rates) < 2:
        raise ValueError("expected E1 rates must contain at least two unique values")
    if set(np.unique(table.labels)) != set(expected_classes):
        raise ValueError("E1 table class set disagrees with the frozen class set")
    if set(np.unique(table.model_seeds)) != set(expected_seeds):
        raise ValueError("E1 table seed set disagrees with the frozen seed set")
    if set(np.unique(table.rates_hz)) != set(expected_rates):
        raise ValueError("E1 table rate set disagrees with the frozen rate set")

    row_by_key: dict[tuple[int, tuple[str, Hashable], int, int], int] = {}
    signal_sets: dict[int, set[tuple[str, Hashable]]] = {
        class_id: set() for class_id in expected_classes
    }
    for row, (label, identifier, seed, rate) in enumerate(
        zip(
            table.labels,
            table.signal_ids,
            table.model_seeds,
            table.rates_hz,
            strict=True,
        )
    ):
        class_id = int(label)
        signal_key = _canonical_identifier(identifier)
        key = (class_id, signal_key, int(seed), int(rate))
        if key in row_by_key:
            raise ValueError(f"duplicate E1 prediction key: {key}")
        row_by_key[key] = row
        signal_sets[class_id].add(signal_key)

    key_groups: list[tuple[tuple[str, Hashable], ...]] = []
    cubes: list[NDArray[np.int64]] = []
    for class_id in expected_classes:
        signal_keys = tuple(
            sorted(signal_sets[class_id], key=lambda item: (item[0], repr(item[1])))
        )
        if not signal_keys:
            raise ValueError(f"E1 class {class_id} has no underlying signals")
        cube = np.empty(
            (len(expected_seeds), len(signal_keys), len(expected_rates)),
            dtype=np.int64,
        )
        for seed_index, seed in enumerate(expected_seeds):
            for signal_index, signal_key in enumerate(signal_keys):
                for rate_index, rate in enumerate(expected_rates):
                    key = (class_id, signal_key, seed, rate)
                    if key not in row_by_key:
                        raise ValueError(f"incomplete E1 Cartesian grid; missing {key}")
                    cube[seed_index, signal_index, rate_index] = row_by_key[key]
        key_groups.append(signal_keys)
        cubes.append(cube)

    expected_rows = sum(
        len(keys) * len(expected_seeds) * len(expected_rates) for keys in key_groups
    )
    if expected_rows != len(table.labels):
        raise ValueError("E1 table contains rows outside the complete Cartesian grid")
    return _E1Layout(
        classes=expected_classes,
        seeds=expected_seeds,
        rates_hz=expected_rates,
        signal_keys_by_class=tuple(key_groups),
        index_cubes=tuple(cubes),
    )


def jensen_shannon_divergence(p: ArrayLike, q: ArrayLike) -> float:
    """Natural-log binary JSD, not SciPy's square-root distance."""

    p_array = _probabilities(np.asarray(p, dtype=np.float64).reshape(1, -1))[0]
    q_array = _probabilities(np.asarray(q, dtype=np.float64).reshape(1, -1))[0]
    midpoint = 0.5 * (p_array + q_array)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_term = np.where(p_array > 0, p_array * np.log(p_array / midpoint), 0.0)
        q_term = np.where(q_array > 0, q_array * np.log(q_array / midpoint), 0.0)
    return float(0.5 * p_term.sum() + 0.5 * q_term.sum())


def _jsd_to_rate_centroid(
    probabilities: NDArray[np.float64],
) -> NDArray[np.float64]:
    centroid = probabilities.mean(axis=-2, keepdims=True)
    midpoint = 0.5 * (probabilities + centroid)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_term = np.where(
            probabilities > 0,
            probabilities * np.log(probabilities / midpoint),
            0.0,
        )
        q_term = np.where(
            centroid > 0,
            centroid * np.log(centroid / midpoint),
            0.0,
        )
    return 0.5 * p_term.sum(axis=-1) + 0.5 * q_term.sum(axis=-1)


def _probability_cubes(
    table: E1Predictions, layout: _E1Layout
) -> tuple[NDArray[np.float64], ...]:
    return tuple(table.probabilities[index_cube] for index_cube in layout.index_cubes)


def _e1_summaries_from_cubes(
    probability_cubes: Sequence[NDArray[np.float64]],
    class_ids: Sequence[int],
    *,
    seed_indices: NDArray[np.int64] | None = None,
    signal_indices_by_class: Sequence[NDArray[np.int64]] | None = None,
) -> tuple[NDArray[np.float64], float]:
    seed_count = probability_cubes[0].shape[0]
    selected_seeds = (
        np.arange(seed_count, dtype=np.int64)
        if seed_indices is None
        else np.asarray(seed_indices, dtype=np.int64)
    )
    rate_ba_by_class: list[NDArray[np.float64]] = []
    jsd_by_class: list[NDArray[np.float64]] = []
    for class_position, (class_id, cube) in enumerate(
        zip(class_ids, probability_cubes, strict=True)
    ):
        selected_signals = (
            np.arange(cube.shape[1], dtype=np.int64)
            if signal_indices_by_class is None
            else np.asarray(signal_indices_by_class[class_position], dtype=np.int64)
        )
        selected = cube[selected_seeds][:, selected_signals, :, :]
        correct = np.argmax(selected, axis=-1) == class_id
        rate_ba_by_class.append(correct.mean(axis=1))  # sampled seed x rate
        per_signal_seed_jsd = _jsd_to_rate_centroid(selected).mean(axis=-1)
        jsd_by_class.append(per_signal_seed_jsd.mean(axis=1))  # sampled seed

    rate_ba = np.stack(rate_ba_by_class, axis=0).mean(axis=0).mean(axis=0)
    mean_jsd = float(np.stack(jsd_by_class, axis=0).mean(axis=0).mean())
    return rate_ba, mean_jsd


def _e1_components(
    probability_cubes: Sequence[NDArray[np.float64]],
    class_ids: Sequence[int],
) -> tuple[
    tuple[NDArray[np.float64], ...], tuple[NDArray[np.float64], ...]
]:
    correctness = tuple(
        (np.argmax(cube, axis=-1) == class_id).astype(np.float64)
        for class_id, cube in zip(class_ids, probability_cubes, strict=True)
    )
    mean_rate_jsd = tuple(
        _jsd_to_rate_centroid(cube).mean(axis=-1) for cube in probability_cubes
    )
    return correctness, mean_rate_jsd


def _bootstrap_count_weights(
    rng: np.random.Generator,
    *,
    replicates: int,
    population_size: int,
) -> NDArray[np.float64]:
    draws = rng.integers(
        0,
        population_size,
        size=(replicates, population_size),
        dtype=np.int64,
    )
    weights = np.zeros((replicates, population_size), dtype=np.float64)
    replicate_rows = np.repeat(np.arange(replicates), population_size)
    np.add.at(weights, (replicate_rows, draws.ravel()), 1.0)
    return weights


def _vectorized_bootstrap_summaries(
    correctness_by_class: Sequence[NDArray[np.float64]],
    jsd_by_class: Sequence[NDArray[np.float64]],
    *,
    seed_weights: NDArray[np.float64],
    signal_weights_by_class: Sequence[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    replicate_count, seed_count = seed_weights.shape
    rate_count = correctness_by_class[0].shape[-1]
    rate_ba = np.zeros((replicate_count, rate_count), dtype=np.float64)
    mean_jsd = np.zeros(replicate_count, dtype=np.float64)
    for correct, divergence, signal_weights in zip(
        correctness_by_class,
        jsd_by_class,
        signal_weights_by_class,
        strict=True,
    ):
        signal_count = correct.shape[1]
        rate_ba += np.einsum(
            "bs,snr,bn->br",
            seed_weights,
            correct,
            signal_weights,
            optimize=True,
        ) / (seed_count * signal_count)
        mean_jsd += np.einsum(
            "bs,sn,bn->b",
            seed_weights,
            divergence,
            signal_weights,
            optimize=True,
        ) / (seed_count * signal_count)
    class_count = len(correctness_by_class)
    return rate_ba / class_count, mean_jsd / class_count


def e1_prediction_consistency(
    table: E1Predictions,
    *,
    classes: Iterable[int] = DEFAULT_E1_CLASSES,
    seeds: Iterable[int] = DEFAULT_E1_SEEDS,
    rates_hz: Iterable[int] = DEFAULT_E1_RATES_HZ,
) -> dict[str, Any]:
    layout = _e1_layout(table, classes=classes, seeds=seeds, rates_hz=rates_hz)
    cubes = _probability_cubes(table, layout)
    _, mean_jsd = _e1_summaries_from_cubes(cubes, layout.classes)
    per_seed: list[float] = []
    for seed_index in range(len(layout.seeds)):
        _, value = _e1_summaries_from_cubes(
            cubes,
            layout.classes,
            seed_indices=np.asarray([seed_index], dtype=np.int64),
        )
        per_seed.append(value)
    return {
        "mean_jsd_to_within_signal_centroid": mean_jsd,
        "per_seed": dict(zip((str(seed) for seed in layout.seeds), per_seed, strict=True)),
        "logarithm": "natural",
        "scipy_sqrt_distance": False,
        "aggregation_order": [
            "rates_within_signal",
            "signals_equal_within_class",
            "classes_equal",
            "seeds_equal",
        ],
    }


def e1_worst_rate_balanced_accuracy(
    table: E1Predictions,
    *,
    classes: Iterable[int] = DEFAULT_E1_CLASSES,
    seeds: Iterable[int] = DEFAULT_E1_SEEDS,
    rates_hz: Iterable[int] = DEFAULT_E1_RATES_HZ,
) -> dict[str, Any]:
    layout = _e1_layout(table, classes=classes, seeds=seeds, rates_hz=rates_hz)
    rate_values, _ = _e1_summaries_from_cubes(
        _probability_cubes(table, layout), layout.classes
    )
    by_rate = {
        str(rate): float(value)
        for rate, value in zip(layout.rates_hz, rate_values, strict=True)
    }
    worst_index = int(np.argmin(rate_values))
    return {
        "worst_rate_balanced_accuracy": float(rate_values[worst_index]),
        "worst_rate_hz": int(layout.rates_hz[worst_index]),
        "balanced_accuracy_by_rate_hz": by_rate,
        "aggregation_order": [
            "signals_equal_within_class",
            "classes_equal",
            "seeds_equal",
            "minimum_over_rates",
        ],
    }


def e1_representation_distance(
    table: E1Predictions,
    embeddings: ArrayLike,
    *,
    classes: Iterable[int] = DEFAULT_E1_CLASSES,
    seeds: Iterable[int] = DEFAULT_E1_SEEDS,
    rates_hz: Iterable[int] = DEFAULT_E1_RATES_HZ,
) -> dict[str, Any]:
    """Mean rate-pair cosine distance after token mean-pooling and L2 norm."""

    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim == 3:
        values = values.mean(axis=1)
    if values.ndim != 2 or values.shape[0] != len(table.labels):
        raise ValueError("embeddings must have shape (N,D) or (N,T,D)")
    if not np.all(np.isfinite(values)):
        raise ValueError("embeddings contain non-finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("zero-norm representation cannot be L2-normalized")
    normalized = values / norms

    layout = _e1_layout(table, classes=classes, seeds=seeds, rates_hz=rates_hz)
    pair_indices = tuple(combinations(range(len(layout.rates_hz)), 2))
    class_seed_values: list[NDArray[np.float64]] = []
    for index_cube in layout.index_cubes:
        cube = normalized[index_cube]  # seed x signal x rate x feature
        distances = []
        for left, right in pair_indices:
            cosine = np.sum(cube[:, :, left, :] * cube[:, :, right, :], axis=-1)
            distances.append(1.0 - np.clip(cosine, -1.0, 1.0))
        per_signal_seed = np.stack(distances, axis=-1).mean(axis=-1)
        class_seed_values.append(per_signal_seed.mean(axis=1))
    per_seed_values = np.stack(class_seed_values, axis=0).mean(axis=0)
    return {
        "mean_rate_pair_cosine_distance": float(per_seed_values.mean()),
        "per_seed": {
            str(seed): float(value)
            for seed, value in zip(layout.seeds, per_seed_values, strict=True)
        },
        "rate_pair_count": len(pair_indices),
        "pooling": "mean_over_tokens_then_l2_normalize",
        "aggregation_order": [
            "unordered_rate_pairs_within_signal",
            "signals_equal_within_class",
            "classes_equal",
            "seeds_equal",
        ],
    }


def bootstrap_e1_paired_contrast(
    *,
    mechanism: E1Predictions,
    baseline: E1Predictions,
    classes: Iterable[int] = DEFAULT_E1_CLASSES,
    seeds: Iterable[int] = DEFAULT_E1_SEEDS,
    rates_hz: Iterable[int] = DEFAULT_E1_RATES_HZ,
    replicates: int = 10_000,
    bootstrap_seed: int = 20_260_801,
    confidence_level: float = 0.95,
    include_samples: bool = False,
) -> dict[str, Any]:
    """Frozen paired E1 bootstrap with crossed signal and seed weights."""

    if isinstance(replicates, bool) or int(replicates) != replicates or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")

    mechanism_layout = _e1_layout(
        mechanism, classes=classes, seeds=seeds, rates_hz=rates_hz
    )
    baseline_layout = _e1_layout(
        baseline, classes=classes, seeds=seeds, rates_hz=rates_hz
    )
    if (
        mechanism_layout.classes != baseline_layout.classes
        or mechanism_layout.seeds != baseline_layout.seeds
        or mechanism_layout.rates_hz != baseline_layout.rates_hz
        or mechanism_layout.signal_keys_by_class
        != baseline_layout.signal_keys_by_class
    ):
        raise ValueError("mechanism and baseline E1 grids are not exactly paired")

    mechanism_cubes = _probability_cubes(mechanism, mechanism_layout)
    baseline_cubes = _probability_cubes(baseline, baseline_layout)
    mechanism_rate_ba, mechanism_jsd = _e1_summaries_from_cubes(
        mechanism_cubes, mechanism_layout.classes
    )
    baseline_rate_ba, baseline_jsd = _e1_summaries_from_cubes(
        baseline_cubes, baseline_layout.classes
    )
    point_worst_ba = float(np.min(mechanism_rate_ba) - np.min(baseline_rate_ba))
    point_jsd_reduction = float(baseline_jsd - mechanism_jsd)

    mechanism_correct, mechanism_signal_jsd = _e1_components(
        mechanism_cubes, mechanism_layout.classes
    )
    baseline_correct, baseline_signal_jsd = _e1_components(
        baseline_cubes, baseline_layout.classes
    )
    rng = np.random.Generator(np.random.PCG64(int(bootstrap_seed)))
    seed_count = len(mechanism_layout.seeds)
    seed_weights = _bootstrap_count_weights(
        rng,
        replicates=int(replicates),
        population_size=seed_count,
    )
    signal_weights = tuple(
        _bootstrap_count_weights(
            rng,
            replicates=int(replicates),
            population_size=len(keys),
        )
        for keys in mechanism_layout.signal_keys_by_class
    )
    mechanism_rate_samples, mechanism_jsd_samples = (
        _vectorized_bootstrap_summaries(
            mechanism_correct,
            mechanism_signal_jsd,
            seed_weights=seed_weights,
            signal_weights_by_class=signal_weights,
        )
    )
    baseline_rate_samples, baseline_jsd_samples = _vectorized_bootstrap_summaries(
        baseline_correct,
        baseline_signal_jsd,
        seed_weights=seed_weights,
        signal_weights_by_class=signal_weights,
    )
    worst_ba_samples = np.min(mechanism_rate_samples, axis=1) - np.min(
        baseline_rate_samples, axis=1
    )
    jsd_reduction_samples = baseline_jsd_samples - mechanism_jsd_samples

    tail = (1.0 - confidence_level) / 2.0

    def interval(samples: NDArray[np.float64]) -> dict[str, float]:
        lower, upper = np.quantile(samples, [tail, 1.0 - tail])
        return {"lower": float(lower), "upper": float(upper)}

    worst_interval = interval(worst_ba_samples)
    jsd_interval = interval(jsd_reduction_samples)
    supported = bool(
        point_worst_ba > 0.0
        and point_jsd_reduction > 0.0
        and worst_interval["lower"] > 0.0
        and jsd_interval["lower"] > 0.0
    )
    refuted = bool(
        worst_interval["upper"] < 0.0 or jsd_interval["upper"] < 0.0
    )
    result: dict[str, Any] = {
        "estimands": {
            "worst_rate_balanced_accuracy_effect": point_worst_ba,
            "jsd_reduction_effect": point_jsd_reduction,
        },
        "percentile_intervals": {
            "worst_rate_balanced_accuracy_effect": worst_interval,
            "jsd_reduction_effect": jsd_interval,
        },
        "gate": {
            "worst_rate_balanced_accuracy_ci_lower_gt_zero": bool(
                worst_interval["lower"] > 0.0
            ),
            "jsd_reduction_ci_lower_gt_zero": bool(jsd_interval["lower"] > 0.0),
            "both_point_estimates_positive": bool(
                point_worst_ba > 0.0 and point_jsd_reduction > 0.0
            ),
            "c1_supported": supported,
            "decision": "supported" if supported else "refuted" if refuted else "inconclusive",
        },
        "bootstrap": {
            "generator": "numpy_Generator_PCG64",
            "seed": int(bootstrap_seed),
            "replicates": int(replicates),
            "confidence_level": float(confidence_level),
            "pairing": "identical_signal_and_seed_multiplicities_for_both_arms",
            "resampling": "signals_within_class_crossed_with_model_seeds",
            "rate_copies_independent": False,
        },
    }
    if include_samples:
        result["samples"] = {
            "worst_rate_balanced_accuracy_effect": worst_ba_samples.copy(),
            "jsd_reduction_effect": jsd_reduction_samples.copy(),
        }
    return result
