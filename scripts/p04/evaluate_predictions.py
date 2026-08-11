"""Evaluate P04 held-out prediction and routing safeguards.

Input is a non-pickled ``.npz`` archive with schema
``p04.predictions-input.v1``.  It requires scalar ``seed``, ``arm``, ``dataset``,
``partition_name='test'``, ``split_manifest_sha256`` and ``checkpoint_sha256``;
unique ``sample_ids[N]``; ``logits[N,C]``; integer ``labels[N]``;
``class_labels[C]``; and ``group_ids[N]``.  A four-expert MoE additionally
provides ``routing_weights[N,4]``.  Groups are CWRU recordings or XJTU bearings.

The canonical output is deterministic JSON and is opened exclusively so a
completed run artifact cannot be overwritten.  Metrics are calculated from raw
logits only; no temperature is fitted on test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


INPUT_SCHEMA = "p04.predictions-input.v1"
OUTPUT_SCHEMA = "p04.prediction-metrics.v1"
OUTPUT_SCHEMA_VERSION = "1.0.0"
ECE_BINS = 15


def _finite_array(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    result = array.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _integer_vector(value: Any, name: str, length: int | None = None) -> np.ndarray:
    numeric = _finite_array(value, name, ndim=1)
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain integers")
    result = numeric.astype(np.int64)
    if length is not None and result.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    return result


def _text_scalar(value: Any, name: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar string")
    result = str(array.item())
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _integer_scalar(value: Any, name: str) -> int:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar integer")
    raw = array.item()
    if isinstance(raw, (bool, np.bool_)) or int(raw) != raw:
        raise ValueError(f"{name} must be a scalar integer")
    return int(raw)


def _validate_sha256(value: Any, name: str) -> str:
    digest = _text_scalar(value, name).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} must be a 64-character SHA-256 digest")
    return digest


def group_equal_weights(group_ids: Any) -> np.ndarray:
    """Give every group equal mass and observations equal mass within group."""
    groups = np.asarray(group_ids)
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError("group_ids must be a non-empty one-dimensional array")
    normalized = groups.astype(str)
    if np.any(normalized == ""):
        raise ValueError("group_ids must be non-empty")
    unique, inverse, counts = np.unique(
        normalized, return_inverse=True, return_counts=True
    )
    return 1.0 / (unique.size * counts[inverse].astype(np.float64))


def softmax(logits: Any) -> np.ndarray:
    matrix = _finite_array(logits, "logits", ndim=2)
    maximum = matrix.max(axis=1, keepdims=True)
    exponential = np.exp(matrix - maximum)
    return exponential / exponential.sum(axis=1, keepdims=True)


def expected_calibration_error(
    confidence: Any,
    correctness: Any,
    weights: Any,
    *,
    bins: int = ECE_BINS,
) -> tuple[float, list[dict[str, Any]]]:
    """Exact frozen equal-width ECE, including 0/1 endpoints and empty bins."""
    if bins != ECE_BINS:
        raise ValueError("the frozen primary evaluator requires exactly 15 bins")
    confidence_array = _finite_array(confidence, "confidence", ndim=1)
    correctness_array = _finite_array(correctness, "correctness", ndim=1)
    weight_array = _finite_array(weights, "weights", ndim=1)
    if not (
        confidence_array.shape == correctness_array.shape == weight_array.shape
    ):
        raise ValueError("confidence, correctness, and weights must have equal shape")
    if np.any(confidence_array < 0.0) or np.any(confidence_array > 1.0):
        raise ValueError("confidence must lie in [0, 1]")
    if not np.isin(correctness_array, (0.0, 1.0)).all():
        raise ValueError("correctness must be binary")
    if np.any(weight_array < 0.0) or not np.isclose(
        weight_array.sum(), 1.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("weights must be non-negative and sum to one")

    bin_index = np.minimum(
        np.floor(confidence_array * bins).astype(np.int64), bins - 1
    )
    result = 0.0
    records: list[dict[str, Any]] = []
    for index in range(bins):
        selected = bin_index == index
        mass = float(weight_array[selected].sum())
        if mass == 0.0:
            accuracy = None
            mean_confidence = None
            contribution = 0.0
            count = 0
        else:
            accuracy = float(
                np.sum(weight_array[selected] * correctness_array[selected]) / mass
            )
            mean_confidence = float(
                np.sum(weight_array[selected] * confidence_array[selected]) / mass
            )
            contribution = mass * abs(accuracy - mean_confidence)
            result += contribution
            count = int(selected.sum())
        records.append(
            {
                "bin": index,
                "left": index / bins,
                "right": (index + 1) / bins,
                "right_closed": index == bins - 1,
                "count": count,
                "weight": mass,
                "accuracy": accuracy,
                "confidence": mean_confidence,
                "contribution": contribution,
            }
        )
    return float(result), records


def _classification_metrics(
    probabilities: np.ndarray,
    log_probabilities: np.ndarray,
    labels: np.ndarray,
    class_labels: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    predicted_indices = np.argmax(probabilities, axis=1)
    predicted = class_labels[predicted_indices]
    recalls: list[float] = []
    f1_scores: list[float] = []
    per_class: list[dict[str, Any]] = []
    for class_label in class_labels:
        actual_class = labels == class_label
        predicted_class = predicted == class_label
        support_weight = float(weights[actual_class].sum())
        if support_weight <= 0.0:
            raise ValueError(
                f"protocol-defined class {int(class_label)} has zero held-out support"
            )
        true_positive = float(weights[actual_class & predicted_class].sum())
        false_positive = float(weights[~actual_class & predicted_class].sum())
        false_negative = float(weights[actual_class & ~predicted_class].sum())
        recall = true_positive / (true_positive + false_negative)
        denominator = 2.0 * true_positive + false_positive + false_negative
        f1 = 0.0 if denominator == 0.0 else 2.0 * true_positive / denominator
        recalls.append(recall)
        f1_scores.append(f1)
        per_class.append(
            {
                "class_label": int(class_label),
                "true_count": int(actual_class.sum()),
                "predicted_count": int(predicted_class.sum()),
                "weighted_recall": recall,
                "weighted_f1": f1,
            }
        )
    confidence = probabilities.max(axis=1)
    correctness = (predicted == labels).astype(np.float64)
    ece, bins = expected_calibration_error(confidence, correctness, weights)

    label_to_column = {int(label): index for index, label in enumerate(class_labels)}
    target_columns = np.asarray([label_to_column[int(label)] for label in labels])
    selected_log_probabilities = log_probabilities[
        np.arange(labels.size), target_columns
    ]
    nll = float(np.sum(weights * -selected_log_probabilities))
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(labels.size), target_columns] = 1.0
    brier = float(np.sum(weights * np.sum((probabilities - one_hot) ** 2, axis=1)))
    return {
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "ece_15_equal_width": ece,
        "negative_log_likelihood": nll,
        "multiclass_brier_score": brier,
        "per_class": per_class,
        "ece_bins": bins,
        "argmax_tie_break": "ascending_logit_column_index",
    }


def collapse_safeguard(routing_weights: Any, weights: Any) -> dict[str, Any]:
    routing = _finite_array(routing_weights, "routing_weights", ndim=2)
    observation_weights = _finite_array(weights, "weights", ndim=1)
    if routing.shape[0] != observation_weights.size or routing.shape[1] != 4:
        raise ValueError("routing_weights must have shape [N, 4]")
    if np.any(routing < 0.0) or np.any(routing > 1.0):
        raise ValueError("routing_weights must lie in [0, 1]")
    if not np.allclose(routing.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-6):
        raise ValueError("routing_weights rows must sum to one within 1e-6")
    if observation_weights.shape != (routing.shape[0],) or not np.isclose(
        observation_weights.sum(), 1.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("weights must have shape [N] and sum to one")

    marginal_use = np.sum(observation_weights[:, None] * routing, axis=0)
    positive = marginal_use > 0.0
    entropy = float(
        -np.sum(marginal_use[positive] * np.log(marginal_use[positive])) / np.log(4.0)
    )
    collapsed = routing.max(axis=1) >= 0.98
    collapsed_fraction = float(observation_weights[collapsed].sum())
    failures: list[str] = []
    if collapsed_fraction > 0.20:
        failures.append("collapsed_window_fraction_above_0.20")
    if np.any(marginal_use < 0.05):
        failures.append("marginal_expert_usage_below_0.05")
    if np.any(marginal_use > 0.80):
        failures.append("marginal_expert_usage_above_0.80")
    if entropy < 0.60:
        failures.append("marginal_usage_entropy_below_0.60")
    return {
        "applicable": True,
        "group_equal_weighted": True,
        "marginal_expert_use": marginal_use.tolist(),
        "marginal_usage_entropy": entropy,
        "collapsed_window_fraction": collapsed_fraction,
        "collapsed_window_count": int(collapsed.sum()),
        "failed": bool(failures),
        "failure_reasons": failures,
        "thresholds": {
            "collapsed_if_max_weight_at_least": 0.98,
            "maximum_collapsed_fraction": 0.20,
            "minimum_marginal_use": 0.05,
            "maximum_marginal_use": 0.80,
            "minimum_marginal_usage_entropy": 0.60,
        },
    }


def evaluate_predictions(
    logits: Any,
    labels: Any,
    class_labels: Any,
    group_ids: Any,
    *,
    routing_weights: Any | None = None,
) -> dict[str, Any]:
    matrix = _finite_array(logits, "logits", ndim=2)
    if matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError("logits must have shape [N, C] with N > 0 and C >= 2")
    targets = _integer_vector(labels, "labels", matrix.shape[0])
    classes = _integer_vector(class_labels, "class_labels", matrix.shape[1])
    if np.unique(classes).size != classes.size:
        raise ValueError("class_labels must be unique")
    if not np.isin(targets, classes).all():
        raise ValueError("labels contain a value absent from class_labels")
    groups = np.asarray(group_ids)
    if groups.shape != (matrix.shape[0],):
        raise ValueError("group_ids must have one value per observation")

    maximum = matrix.max(axis=1, keepdims=True)
    shifted = matrix - maximum
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probabilities = np.exp(log_probabilities)
    primary_weights = group_equal_weights(groups)
    pooled_weights = np.full(matrix.shape[0], 1.0 / matrix.shape[0])
    primary = _classification_metrics(
        probabilities, log_probabilities, targets, classes, primary_weights
    )
    pooled = _classification_metrics(
        probabilities, log_probabilities, targets, classes, pooled_weights
    )
    if routing_weights is None:
        collapse = {"applicable": False, "reason": "routing_weights_not_provided"}
    else:
        collapse = collapse_safeguard(routing_weights, primary_weights)
    return {
        "schema_id": OUTPUT_SCHEMA,
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "primary_weighting": "equal_total_weight_per_group_then_equal_within_group",
        "group_equal": primary,
        "pooled_window_descriptive": pooled,
        "collapse_safeguard": collapse,
        "observation_count": matrix.shape[0],
        "group_count": int(np.unique(groups.astype(str)).size),
        "class_labels": classes.tolist(),
        "primary_logits": "raw",
        "temperature_scaling_fitted": False,
    }


def _load_input(path: Path) -> tuple[dict[str, np.ndarray], str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except Exception as exc:
        raise ValueError(f"could not read predictions NPZ {path}: {exc}") from exc
    required = {
        "schema_id",
        "seed",
        "arm",
        "dataset",
        "partition_name",
        "split_manifest_sha256",
        "checkpoint_sha256",
        "sample_ids",
        "logits",
        "labels",
        "class_labels",
        "group_ids",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError("prediction artifact is missing fields: " + ", ".join(missing))
    return arrays, digest


def run_evaluation(input_path: Path, output_path: Path) -> dict[str, Any]:
    arrays, input_sha256 = _load_input(input_path)
    schema = _text_scalar(arrays["schema_id"], "schema_id")
    if schema != INPUT_SCHEMA:
        raise ValueError(f"schema_id must be {INPUT_SCHEMA!r}, got {schema!r}")
    partition = _text_scalar(arrays["partition_name"], "partition_name")
    if partition != "test":
        raise ValueError("prediction metrics may only consume partition_name='test'")
    logits = np.asarray(arrays["logits"])
    if logits.ndim != 2:
        raise ValueError("logits must have shape [N, C]")
    count = logits.shape[0]
    sample_ids = np.asarray(arrays["sample_ids"])
    if sample_ids.shape != (count,):
        raise ValueError("sample_ids must have one value per observation")
    normalized_ids = sample_ids.astype(str)
    if np.any(normalized_ids == "") or np.unique(normalized_ids).size != count:
        raise ValueError("sample_ids must be non-empty and unique")

    result = evaluate_predictions(
        arrays["logits"],
        arrays["labels"],
        arrays["class_labels"],
        arrays["group_ids"],
        routing_weights=arrays.get("routing_weights"),
    )
    result["provenance"] = {
        "input_sha256": input_sha256,
        "seed": _integer_scalar(arrays["seed"], "seed"),
        "arm": _text_scalar(arrays["arm"], "arm"),
        "dataset": _text_scalar(arrays["dataset"], "dataset"),
        "partition_name": partition,
        "split_manifest_sha256": _validate_sha256(
            arrays["split_manifest_sha256"], "split_manifest_sha256"
        ),
        "checkpoint_sha256": _validate_sha256(
            arrays["checkpoint_sha256"], "checkpoint_sha256"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen P04 real-data prediction safeguards."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_evaluation(args.input, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "balanced_accuracy": result["group_equal"]["balanced_accuracy"],
                "macro_f1": result["group_equal"]["macro_f1"],
                "ece": result["group_equal"]["ece_15_equal_width"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "collapse_safeguard",
    "evaluate_predictions",
    "expected_calibration_error",
    "group_equal_weights",
    "run_evaluation",
    "softmax",
]
