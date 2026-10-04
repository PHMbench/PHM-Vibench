"""Source-condition development qualification from saved acquisition predictions.

Qualification is not independent generalization evidence. It uses the same
equal-group/equal-acquisition accuracy as the P01 evaluator, never pooled windows.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


def qualify_source_prediction_arrays(
    arrays: Mapping[str, Any], source_conditions: Sequence[str], classes: int,
    threshold: float = .8, *, predictor: str = "candidate",
) -> dict[str, Any]:
    """Reject malformed populations; report measured qualification failures.

    ``predictor`` explicitly chooses ``candidate_probs`` or ``raw_probs``. There
    is no fallback to a different predictor when the requested array is absent.
    """
    conditions = [str(value) for value in source_conditions]
    if not conditions or any(not value for value in conditions) or len(set(conditions)) != len(conditions):
        raise ValueError("Declare a nonempty, unique expected source-condition set.")
    if isinstance(classes, bool) or int(classes) != classes or classes < 2:
        raise ValueError("Qualification requires the declared closed-set class count.")
    classes = int(classes)
    if not math.isfinite(threshold) or not .8 <= threshold <= 1:
        raise ValueError("Source qualification threshold must lie in [0.8,1].")
    if predictor not in {"candidate", "raw"}:
        raise ValueError("Qualification predictor must be candidate or raw explicitly.")
    required = ("domains", "group_ids", "acquisition_ids", "labels", predictor + "_probs")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"Qualification prediction arrays lack {missing}.")
    domains, groups, acquisitions = [np.asarray(arrays[key]).astype(str)
                                     for key in required[:3]]
    labels = np.asarray(arrays["labels"])
    probabilities = np.asarray(arrays[predictor + "_probs"], dtype=float)
    count = len(labels) if labels.ndim == 1 else 0
    if not count or any(value.shape != (count,) for value in (domains, groups, acquisitions, labels)):
        raise ValueError("Qualification requires nonempty, aligned one-dimensional identities and labels.")
    if any(np.any(value == "") for value in (domains, groups, acquisitions)):
        raise ValueError("Qualification requires explicit condition, physical group and acquisition IDs.")
    if set(domains) != set(conditions):
        raise ValueError(f"Qualification source-condition coverage differs: expected {conditions}, observed {sorted(set(domains))}.")
    if probabilities.shape != (count, classes) or not np.isfinite(probabilities).all():
        raise ValueError("Qualification probabilities must be finite with the declared sample/class shape.")
    if np.any(probabilities < 0) or np.any(probabilities > 1) or not np.allclose(probabilities.sum(1), 1., rtol=1e-5, atol=1e-7):
        raise ValueError("Qualification requires normalized class probabilities; no repair is applied.")
    if labels.dtype.kind not in "iuf" or not np.isfinite(labels).all() or np.any(labels != labels.astype(int)) or np.any(labels < 0) or np.any(labels >= classes):
        raise ValueError("Qualification labels must be integer indices in the declared class space.")
    labels = labels.astype(int)
    if "window_ids" in arrays:
        windows = np.asarray(arrays["window_ids"]).astype(str)
        if windows.shape != (count,) or len(set(zip(domains, groups, acquisitions, windows))) != count:
            raise ValueError("Qualification window identities are misaligned or duplicated.")

    buckets: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    owners: dict[tuple[str, str], str] = {}
    for index, (condition, group, acquisition) in enumerate(zip(domains, groups, acquisitions)):
        if owners.setdefault((condition, acquisition), group) != group:
            raise ValueError("A qualification acquisition belongs to more than one physical group.")
        buckets[(condition, group, acquisition)].append(index)
    grouped: dict[str, dict[str, list[tuple[int, int]]]] = {
        condition: defaultdict(list) for condition in conditions
    }
    for (condition, group, _), indices in buckets.items():
        if len(set(labels[indices].tolist())) != 1:
            raise ValueError("Qualification requires constant-label acquisitions.")
        grouped[condition][group].append((int(labels[indices[0]]), int(probabilities[indices].mean(0).argmax())))

    rows = []
    for condition in conditions:
        group_rows = grouped[condition]
        matrix = np.zeros((classes, classes), dtype=float)
        support = np.zeros(classes, dtype=int)
        predicted = np.zeros(classes, dtype=int)
        for acquisition_rows in group_rows.values():
            weight = 1. / (len(group_rows) * len(acquisition_rows))
            for label, prediction in acquisition_rows:
                matrix[label, prediction] += weight
                support[label] += 1
                predicted[prediction] += 1
        truth_weight, predicted_weight = matrix.sum(1), matrix.sum(0)
        recall = np.divide(np.diag(matrix), truth_weight, out=np.zeros(classes), where=truth_weight > 0)
        denominator = truth_weight + predicted_weight
        macro_f1 = np.divide(2 * np.diag(matrix), denominator, out=np.zeros(classes), where=denominator > 0).mean()
        accuracy = float(np.trace(matrix))
        reasons = []
        if accuracy < threshold and not math.isclose(accuracy, threshold, rel_tol=0, abs_tol=1e-12):
            reasons.append("accuracy_below_threshold")
        if np.any(support == 0):
            reasons.append("missing_true_class_support")
        if np.any(recall == 0):
            reasons.append("zero_class_recall")
        rows.append(dict(domain=condition, predictor=predictor, accuracy=accuracy,
                         macro_f1=float(macro_f1), confusion_matrix=matrix.tolist(),
                         class_support=support.tolist(), weighted_class_support=truth_weight.tolist(),
                         predicted_class_support=predicted.tolist(),
                         weighted_predicted_class_support=predicted_weight.tolist(), class_recall=recall.tolist(),
                         groups=len(group_rows), acquisitions=int(support.sum()),
                         passed=not reasons, reasons=reasons))
    return dict(passed=all(row["passed"] for row in rows), threshold=threshold, predictor=predictor,
                expected_source_conditions=conditions, source_conditions=rows,
                reasons=[f"{row['domain']}: {reason}" for row in rows for reason in row["reasons"]],
                independent_test_guarantee=False,
                aggregation="equal acquisitions/physical group; equal groups/condition")


def record_training_failure(output: Path, error: BaseException) -> None:
    """Keep a direct trainer failure visible without masking its original error."""
    scope_path = output / "result_scope.json"
    try:
        scope = json.loads(scope_path.read_text()) if scope_path.is_file() else {}
    except (OSError, ValueError):
        scope = {}
    scope.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                 error_type=type(error).__name__, error=str(error))
    try:
        scope_path.write_text(json.dumps(scope, indent=2))
        (output / "failure.json").write_text(json.dumps(dict(
            status=scope["status"], error_type=type(error).__name__, error=str(error)), indent=2))
    except OSError as write_error:
        print(f"Could not save trainer failure to {output}: {write_error}", file=sys.stderr)
