import numpy as np
import pytest

from src.explain_factory.p05_selective_risk import (
    equal_mass_ece,
    equal_group_window_weights,
    fit_logistic_risk_ranker,
    fit_validation_risk_bundle,
    frozen_threshold_metrics,
    operational_wording_gate,
    retrospective_selective_metrics,
    select_validation_threshold,
    trace_risk_features,
)


def test_equal_group_weights_ignore_unequal_window_counts():
    weights = equal_group_window_weights(["a", "a", "a", "b"])
    assert weights.mean() == pytest.approx(1.0)
    assert weights[:3].sum() == pytest.approx(weights[3])


def test_logistic_ranker_requires_both_error_outcomes():
    features = np.column_stack((np.arange(6), np.arange(6) ** 2)).astype(np.float64)
    with pytest.raises(ValueError, match="both outcomes"):
        fit_logistic_risk_ranker(
            features,
            np.zeros(6, dtype=np.int64),
            np.ones(6),
            name="trace",
        )


def test_validation_bundle_fits_all_registered_scores_without_test_inputs():
    logits = np.asarray(
        [
            [4.0, 0.0],
            [0.2, 1.0],
            [0.8, 0.7],
            [0.0, 3.0],
            [1.0, 1.1],
            [2.0, 0.1],
            [0.4, 1.5],
            [1.2, 0.9],
        ]
    )
    labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
    firing = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.6, 0.3, 0.1],
            [0.34, 0.33, 0.33],
            [0.1, 0.2, 0.7],
            [0.4, 0.4, 0.2],
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.5, 0.3, 0.2],
        ]
    )
    bundle = fit_validation_risk_bundle(
        sample_ids=[f"s{index}" for index in range(8)],
        groups=["g1"] * 4 + ["g2"] * 4,
        logits=logits,
        firing=firing,
        labels=labels,
    )

    assert set(bundle.thresholds) == {"trace", "R0", "R1", "R2", "R3"}
    assert 0.05 <= bundle.temperature <= 20.0
    assert all(np.isfinite(value) for value in bundle.thresholds.values())


def test_trace_risk_accepts_and_normalizes_float32_softmax_row_drift():
    logits = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    firing = np.asarray(
        [
            [0.10000001, 0.20000002, 0.70000005],
            [0.33333334, 0.33333334, 0.33333334],
        ],
        dtype=np.float32,
    )
    features = trace_risk_features(logits, firing)
    assert features.shape == (2, 3)
    assert np.isfinite(features).all()


def test_trace_risk_still_rejects_materially_unnormalized_firing():
    with pytest.raises(ValueError, match="sum to one"):
        trace_risk_features(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            np.asarray([[0.2, 0.2, 0.2]], dtype=np.float32),
        )


def test_validation_threshold_includes_all_exact_score_ties():
    scores = np.asarray([0.1, 0.2, 0.2, 0.9, 0.1, 0.2, 0.2, 0.9])
    groups = ["a"] * 4 + ["b"] * 4
    threshold = select_validation_threshold(scores, groups, target_coverage=0.60)
    assert threshold == pytest.approx(0.2)
    assert np.mean(scores[:4] <= threshold) == pytest.approx(0.75)


def test_retrospective_selection_uses_exact_count_and_stable_id_ties():
    result = retrospective_selective_metrics(
        sample_ids=["b", "a", "c", "d"],
        groups=["bearing"] * 4,
        scores=np.asarray([0.1, 0.1, 0.2, 0.3]),
        predictions=np.asarray([0, 0, 1, 1]),
        labels=np.asarray([0, 1, 1, 0]),
        coverages=(0.5,),
    )
    metrics = result["groups"]["bearing"]
    selected = metrics["coverages"]["0.5"]
    assert selected["accepted"] == 2
    assert selected["accepted_sample_ids"] == ["a", "b"]
    assert selected["risk"] == pytest.approx(0.5)
    assert np.isfinite(metrics["aurc"])


def test_frozen_threshold_keeps_ties_and_can_shift_test_coverage():
    result = frozen_threshold_metrics(
        groups=["a", "a", "a"],
        scores=np.asarray([0.1, 0.2, 0.2]),
        predictions=np.asarray([0, 0, 1]),
        labels=np.asarray([0, 1, 1]),
        threshold=0.2,
    )
    assert result["a"]["accepted"] == 3
    assert result["a"]["coverage"] == pytest.approx(1.0)


def test_operational_gate_is_descriptive_and_requires_all_five_bearings():
    groups = [f"b{index}" for index in range(5)]
    trace = {group: {"coverage": 0.90, "risk": 0.05} for group in groups}
    r0 = {group: {"coverage": 0.92, "risk": 0.10} for group in groups}
    r1 = {group: {"coverage": 0.88, "risk": 0.08} for group in groups}
    passed = operational_wording_gate(trace, r0, r1)
    assert passed == {"passed": True, "failures": [], "confirmatory_p_value": None}

    r1["b4"]["risk"] = 0.04
    failed = operational_wording_gate(trace, r0, r1)
    assert failed["passed"] is False
    assert failed["confirmatory_p_value"] is None


def test_equal_mass_ece_uses_fifteen_nonempty_nearly_equal_bins_per_bearing():
    sample_ids = [f"a-{index:02d}" for index in range(31)] + [
        f"b-{index:02d}" for index in range(30)
    ]
    groups = ["a"] * 31 + ["b"] * 30
    confidence = np.linspace(0.51, 0.99, len(sample_ids))
    labels = np.asarray([index % 2 for index in range(len(sample_ids))])
    predictions = labels.copy()
    result = equal_mass_ece(
        sample_ids=sample_ids,
        groups=groups,
        confidence=confidence,
        predictions=predictions,
        labels=labels,
    )

    assert set(result["groups"]) == {"a", "b"}
    assert [item["count"] for item in result["groups"]["a"]["bins"]].count(3) == 1
    assert all(item["count"] == 2 for item in result["groups"]["b"]["bins"])
    assert 0.0 <= result["equal_group_mean_ece"] <= 1.0


def test_equal_mass_ece_rejects_empty_bins():
    with pytest.raises(ValueError, match="fewer than 15"):
        equal_mass_ece(
            sample_ids=[f"s{index}" for index in range(14)],
            groups=["g"] * 14,
            confidence=np.full(14, 0.5),
            predictions=np.zeros(14, dtype=np.int64),
            labels=np.zeros(14, dtype=np.int64),
        )
