"""Descriptive within-specimen operating-condition contrasts of frozen predictions.

This supplementary paired population never replaces the full target population.
Within each specimen, classes receive equal fixed weight in both conditions;
acquisitions are equal within class and windows are equal within acquisition.
No causal effect is estimated. Intervals condition on the fitted predictor and
resample whole specimens within their shared class-support strata.
"""
from __future__ import annotations

from collections import defaultdict
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.p01.analyze_d1 import acquisition_estimates, condition_metrics, group_estimates


METRICS = ("ce", "brier", "accuracy", "macro_f1")
ESTIMAND = "paired_specimen_equal_class"


def _class_standardized_groups(
    acquisitions: Sequence[Mapping[str, Any]], classes: int
) -> list[dict[str, Any]]:
    """Average acquisition summaries within class before averaging class terms."""
    by_specimen: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for label in range(classes):
        class_rows = [row for row in acquisitions if row["label"] == label]
        for row in group_estimates(class_rows, classes):
            by_specimen[(row["domain"], row["predictor"], row["unit_id"])].append(row)
    result = []
    for (domain, predictor, group), rows in sorted(by_specimen.items()):
        result.append(dict(
            domain=domain, predictor=predictor, unit_id=group,
            acquisitions=sum(row["acquisitions"] for row in rows),
            windows=sum(row["windows"] for row in rows),
            **{key: float(np.mean([row[key] for row in rows]))
               for key in ("ce", "brier", "A", "b", "repair", "damage", "change")},
            confusion_matrix=np.mean([row["confusion_matrix"] for row in rows], axis=0),
        ))
    return result


def paired_condition_rows(
    arrays: Mapping[str, np.ndarray],
    source_conditions: Sequence[str],
    target_condition: str,
    *,
    bootstraps: int = 2000,
    seed: int = 20260919,
    eligible_groups: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    """Compare A and B on identical specimens, separately for raw/candidate.

    ``arrays`` is one validated frozen window export. Optional ``eligible_groups``
    maps each source condition to its prospectively declared A/B specimen cohort.
    Estimates use the observed intersection of that cohort and equal weights over
    specimens, then common classes within specimen, then acquisitions within class,
    then windows within acquisition. Accuracy uses averaged acquisition probabilities;
    macro-F1 uses the resulting pooled weighted confusion matrix. The estimand is
    labeled separately from the unstandardized primary condition metrics.
    All-population counts remain visible; missing counts use the union of observed
    and declared groups.
    A missing class or changed within-specimen class support makes the contrast
    non-estimable instead of selectively dropping acquisitions or inventing zeros.
    """
    sources = [str(value) for value in source_conditions]
    target = str(target_condition)
    if not sources or len(set(sources)) != len(sources) or target in sources:
        raise ValueError("Declare unique source conditions distinct from the target condition")
    if isinstance(bootstraps, bool) or not isinstance(bootstraps, Integral) or bootstraps < 2:
        raise ValueError("At least two paired bootstrap draws are required")
    if eligible_groups is not None and set(eligible_groups) != set(sources):
        raise ValueError("Declare the paired specimen cohort for every source condition")

    predictions = dict(arrays)
    classes = predictions["raw_probs"].shape[1]
    labels = np.asarray(predictions["labels"])
    if classes < 2 or labels.ndim != 1 or labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels >= classes)):
        raise ValueError("Window labels must use the declared closed-set class order")
    count = len(labels)
    for key in ("domains", "group_ids", "acquisition_ids"):
        if np.asarray(predictions[key]).shape != (count,):
            raise ValueError(f"Frozen window identity has a different length: {key}")
    for predictor in ("raw", "candidate"):
        for suffix in ("probs", "log_probs"):
            values = np.asarray(predictions[f"{predictor}_{suffix}"])
            if values.shape != (count, classes) or not np.isfinite(values).all():
                raise ValueError("Frozen probabilities/log probabilities must be finite and aligned")
    # The shared aggregator expects a deployed slot. This private alias is never
    # exported or analyzed as an adopted function and does not alter the input.
    predictions["deployed_probs"] = predictions["candidate_probs"]
    predictions["deployed_log_probs"] = predictions["candidate_log_probs"]
    acquisitions = [row for row in acquisition_estimates(predictions) if row["predictor"] != "deployed"]
    groups = _class_standardized_groups(acquisitions, classes)
    support: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for row in acquisitions:
        if row["predictor"] == "raw":
            support[row["domain"]][row["unit_id"]].add(row["label"])

    result = []
    target_groups = set(support[target])
    for source in sources:
        source_groups = set(support[source])
        declared = set(map(str, eligible_groups[source])) if eligible_groups is not None else source_groups & target_groups
        paired = sorted(source_groups & target_groups & declared)
        universe = source_groups | target_groups | declared
        missing_source = len(universe - source_groups)
        missing_target = len(universe - target_groups)
        status = "estimable_with_unpaired_groups" if missing_source or missing_target else "estimable"
        if len(paired) < 2:
            status = "not_estimable_insufficient_pairs"
        elif any(support[source][group] != support[target][group] for group in paired):
            status = "not_estimable_class_support_mismatch"
        elif set().union(*(support[source][group] for group in paired)) != set(range(classes)):
            status = "not_estimable_missing_classes"

        counts = None
        if status.startswith("estimable"):
            strata: dict[tuple[int, ...], list[int]] = defaultdict(list)
            for index, group in enumerate(paired):
                strata[tuple(sorted(support[source][group]))].append(index)
            rng = np.random.default_rng(seed)
            counts = np.zeros((bootstraps, len(paired)), dtype=int)
            for signature in sorted(strata):
                columns = strata[signature]
                draws = rng.integers(len(columns), size=(bootstraps, len(columns)))
                for offset, column in enumerate(columns):
                    counts[:, column] = (draws == offset).sum(axis=1)
            # The first row is the equal-specimen point estimate; remaining rows
            # use the same whole-specimen bootstrap draws for A, B and predictors.
            counts = np.concatenate((np.ones((1, len(paired)), dtype=int), counts))

        for predictor in ("raw", "candidate"):
            differences = None
            if counts is not None:
                source_rows = [row for row in groups if row["domain"] == source and row["predictor"] == predictor and row["unit_id"] in paired]
                target_rows = [row for row in groups if row["domain"] == target and row["predictor"] == predictor and row["unit_id"] in paired]
                source_samples = condition_metrics(source_rows, paired, counts)
                target_samples = condition_metrics(target_rows, paired, counts)
                differences = {metric: target_samples[metric] - source_samples[metric] for metric in METRICS}
            for metric in METRICS:
                source_estimate = target_estimate = delta = lower = upper = None
                if counts is not None:
                    source_estimate = float(source_samples[metric][0])
                    target_estimate = float(target_samples[metric][0])
                    delta = target_estimate - source_estimate
                    lower, upper = map(float, np.quantile(differences[metric][1:], [0.025, 0.975]))
                result.append(dict(
                    predictor=predictor, source_condition=source, target_condition=target,
                    estimand=ESTIMAND,
                    metric=metric, source_all_groups=len(source_groups), target_all_groups=len(target_groups),
                    paired_groups=len(paired), missing_source_groups=missing_source,
                    missing_target_groups=missing_target, status=status,
                    source_estimate=source_estimate, target_estimate=target_estimate,
                    target_minus_source=delta, lower=lower, upper=upper,
                ))
    return result
