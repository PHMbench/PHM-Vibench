"""Constructed paired condition contrasts; no real specimens or model fitting."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.p01.condition_comparison import paired_condition_rows


def arrays(records):
    # Records: condition, physical group, label, probability assigned to truth.
    labels = np.asarray([row[2] for row in records], dtype=int)
    probabilities = np.asarray([
        [confidence, 1 - confidence] if label == 0 else [1 - confidence, confidence]
        for _, _, label, confidence in records
    ])
    return dict(
        labels=labels, domains=np.asarray([row[0] for row in records]),
        group_ids=np.asarray([row[1] for row in records]),
        acquisition_ids=np.asarray([f"a{i}" for i in range(len(records))]),
        raw_probs=probabilities.copy(), candidate_probs=probabilities.copy(),
        raw_log_probs=np.log(probabilities), candidate_log_probs=np.log(probabilities),
    )


def metric(rows, name, predictor="candidate"):
    return next(row for row in rows if row["metric"] == name and row["predictor"] == predictor)


def test_exact_paired_brier_delta_with_shared_multiplicities_and_declared_cohort():
    records = [(condition, f"g{i}", i % 2, confidence)
               for condition, confidence in [("A", 0.9), ("B", 0.6)] for i in range(4)]
    predictions = arrays(records)
    saved = {key: value.copy() for key, value in predictions.items()}
    result = paired_condition_rows(predictions, ["A"], "B", bootstraps=100,
                                   eligible_groups={"A": ["g0", "g1", "g2", "g3"]})
    row = metric(result, "brier")
    assert row["status"] == "estimable"
    assert row["estimand"] == "paired_specimen_equal_class"
    assert row["source_all_groups"] == row["target_all_groups"] == row["paired_groups"] == 4
    assert row["source_estimate"] == pytest.approx(0.02)
    assert row["target_estimate"] == pytest.approx(0.32)
    assert row["target_minus_source"] == row["lower"] == pytest.approx(0.3)
    assert row["upper"] == pytest.approx(0.3)
    assert metric(result, "brier", "raw")["lower"] == row["lower"]
    assert predictions.keys() == saved.keys()
    for key in saved:
        np.testing.assert_array_equal(predictions[key], saved[key])


def test_unpaired_ids_missing_declared_specimens_and_class_failure_remain_visible():
    records = [(condition, f"g{i}", i, 0.8) for condition in ("A", "B") for i in (0, 1)]
    records += [("A", "source_only", 0, 0.9), ("B", "target_only", 1, 0.6)]
    result = paired_condition_rows(arrays(records), ["A"], "B", bootstraps=20,
                                   eligible_groups={"A": ["g0", "g1", "missing_both"]})
    row = metric(result, "brier")
    assert row["status"] == "estimable_with_unpaired_groups"
    assert row["source_all_groups"] == row["target_all_groups"] == 3
    assert row["paired_groups"] == 2
    assert row["missing_source_groups"] == row["missing_target_groups"] == 2
    changed_support = list(records)
    changed_support[2] = ("B", "g0", 1, 0.8)
    rows = paired_condition_rows(arrays(changed_support), ["A"], "B", bootstraps=20)
    assert all(r["status"] == "not_estimable_class_support_mismatch" for r in rows)
    assert all(r["target_minus_source"] is None and r["lower"] is None for r in rows)
    incomplete_classes = [(condition, group, 0, 0.8) for condition in ("A", "B") for group in ("g0", "g1")]
    rows = paired_condition_rows(arrays(incomplete_classes), ["A"], "B", bootstraps=20)
    assert all(r["status"] == "not_estimable_missing_classes" and r["source_estimate"] is None for r in rows)


def test_multilabel_specimens_stay_whole_and_macro_f1_uses_pooled_confusion():
    records = []
    for condition in ("A", "B"):
        for group, labels in [("g0", [0, 1]), ("g1", [0, 1, 1, 1])]:
            for label in labels:
                confidence = 0.2 if condition == "A" and group == "g0" and label == 1 else 0.8
                records.append((condition, group, label, confidence))
    row = metric(paired_condition_rows(arrays(records), ["A"], "B", bootstraps=256), "macro_f1")
    assert row["paired_groups"] == 2
    # Equal class weights inside each group give pooled F1 11/15; averaging
    # individual specimen F1 values would incorrectly give 2/3.
    assert row["source_estimate"] == pytest.approx(11 / 15)
    assert row["target_estimate"] == pytest.approx(1.0)
    assert row["target_minus_source"] == pytest.approx(4 / 15)
    assert row["lower"] == pytest.approx(0.0)
    assert row["upper"] == pytest.approx(2 / 3)


def test_changed_acquisition_class_mixtures_create_no_standardized_condition_effect():
    records = []
    for condition, labels in [("A", [0, 0, 0, 1]), ("B", [0, 1, 1, 1])]:
        for group in ("g0", "g1"):
            for label in labels:
                records.append((condition, group, label, 0.9 if label == 0 else 0.2))
    rows = paired_condition_rows(arrays(records), ["A"], "B", bootstraps=100)
    brier = metric(rows, "brier")
    # Acquisition-weighted source/target Brier would be .335/.965 solely because
    # their class mixtures differ. Equal within-specimen class weights give .65.
    assert brier["source_estimate"] == pytest.approx(0.65)
    assert brier["target_estimate"] == pytest.approx(0.65)
    assert metric(rows, "accuracy")["source_estimate"] == pytest.approx(0.5)
    assert metric(rows, "macro_f1")["source_estimate"] == pytest.approx(1 / 3)
    for row in rows:
        assert row["estimand"] == "paired_specimen_equal_class"
        assert row["target_minus_source"] == pytest.approx(0, abs=1e-12)
        assert row["lower"] == pytest.approx(0, abs=1e-12)
        assert row["upper"] == pytest.approx(0, abs=1e-12)
