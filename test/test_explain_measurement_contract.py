from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from src.explain_factory.contract_cli import load_conformance_config, run_conformance_fixture
from src.explain_factory.measurement import (
    MEASUREMENT_SCHEMA_VERSION,
    AdapterIdentity,
    ArrayMeasurementAdapter,
    MeasurementContractError,
    MeasurementObject,
    SourceIdentity,
    array_sha256,
    assert_measurements_aligned,
)
from src.explain_factory.measurement_metrics import (
    METRIC_OBSERVATION_SCHEMA_VERSION,
    METRIC_REGISTRY,
    MetricInputError,
    MetricObservation,
    activation_ratio,
    aopc,
    assert_metric_compatible,
    attribution_order,
    deletion_at_fraction,
    deletion_score_curve,
    elapsed_time_ms,
    kendall_tau_b,
    metric_registry_sha256,
    pairwise_rank_reversal_rate,
    peak_memory_mib,
    spearman_attribution,
    topk_iou,
)


def _digest(character: str) -> str:
    return character * 64


def _source() -> SourceIdentity:
    return SourceIdentity(
        paper_id="P07",
        method_id="method-a",
        model_id="model-a",
        dataset_id="CWRU",
        split_id="test-v1",
        seed=0,
        hardware_id="cpu:test",
        score_id="fixed_target_probability.v1",
        source_artifact_sha256=_digest("1"),
        model_artifact_sha256=_digest("6"),
        config_sha256=_digest("2"),
        environment_sha256=_digest("3"),
        code_sha256=_digest("4"),
    )


def _adapter() -> ArrayMeasurementAdapter:
    return ArrayMeasurementAdapter(
        AdapterIdentity(
            adapter_id="array-reference",
            adapter_version="1.0.0",
            adapter_sha256=_digest("5"),
            input_kind="sibling_numeric_attribution",
            output_kind="temporal_attribution",
            capabilities=("dense_attribution", "deletion", "paired_stability", "topk_support"),
        )
    )


def _measurement(
    locator: str = "first/attribution.npy",
    *,
    coordinate_map_sha256: str | None = None,
) -> MeasurementObject:
    return _adapter().adapt(
        values=np.asarray([[0.9, 0.1, 0.4, 0.2], [0.7, 0.2, 0.3, 0.1]]),
        sample_ids=("sample-1", "sample-2"),
        axes=("sample", "time"),
        axis_units=("id", "index"),
        coordinate_map_sha256=coordinate_map_sha256 or _digest("8"),
        value_semantics="signed_target_attribution",
        locator=locator,
        source=_source(),
        target_id="class:3",
    )


def test_array_hash_and_measurement_identity_are_content_based() -> None:
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    assert array_sha256(values) == array_sha256(np.asfortranarray(values))

    first = _measurement("first/attribution.npy")
    moved = _measurement("moved/attribution.npy")
    assert first.measurement_id == moved.measurement_id
    assert_measurements_aligned(first, moved)
    assert first.to_manifest()["explanation"]["locator"] != moved.to_manifest()["explanation"]["locator"]

    round_trip = MeasurementObject.from_manifest(first.to_manifest())
    assert round_trip == first
    assert round_trip.schema_version == MEASUREMENT_SCHEMA_VERSION
    with pytest.raises(MeasurementContractError, match="coordinate_map_sha256"):
        assert_measurements_aligned(first, _measurement(coordinate_map_sha256=_digest("9")))


def test_measurement_rejects_tampering_nonfinite_values_and_unstable_sample_ids() -> None:
    manifest = _measurement().to_manifest()
    tampered = copy.deepcopy(manifest)
    tampered["explanation"]["sha256"] = _digest("0")
    with pytest.raises(MeasurementContractError, match="measurement_id"):
        MeasurementObject.from_manifest(tampered)

    with pytest.raises(MeasurementContractError, match="finite"):
        array_sha256(np.asarray([1.0, np.nan]))
    with pytest.raises(MeasurementContractError, match="unique"):
        _adapter().adapt(
            values=np.ones((2, 4)),
            sample_ids=("same", "same"),
            axes=("sample", "time"),
            axis_units=("id", "index"),
            coordinate_map_sha256=_digest("8"),
            value_semantics="signed_target_attribution",
            locator="fixture.npy",
            source=_source(),
            target_id="class:0",
        )
    with pytest.raises(MeasurementContractError, match="sample_ids length"):
        _adapter().adapt(
            values=np.ones((2, 4)),
            sample_ids=("only-one",),
            axes=("sample", "time"),
            axis_units=("id", "index"),
            coordinate_map_sha256=_digest("8"),
            value_semantics="signed_target_attribution",
            locator="fixture.npy",
            source=_source(),
            target_id="class:0",
        )


def test_deletion_curve_aopc_and_negative_drops_are_exact_and_unclipped() -> None:
    values = np.asarray([4.0, 3.0, 2.0, 1.0])
    original = values.copy()
    curve = deletion_score_curve(
        lambda item: float(np.sum(item)),
        values,
        np.asarray([0.9, 0.1, 0.4, 0.2]),
        deletion_fractions=(0.25, 0.5, 1.0),
        baseline=0.0,
    )
    assert np.array_equal(values, original)
    assert curve.deletion_counts == (1, 2, 4)
    assert curve.perturbed_scores == (6.0, 4.0, 0.0)
    assert deletion_at_fraction(curve, 0.5) == pytest.approx(6.0)
    assert aopc(curve) == pytest.approx(20.0 / 3.0)

    negative_curve = deletion_score_curve(
        lambda item: float(np.sum(item)),
        np.asarray([-4.0, 1.0]),
        np.asarray([2.0, 1.0]),
        deletion_fractions=(0.5,),
        baseline=0.0,
    )
    assert aopc(negative_curve) == pytest.approx(-4.0)


def test_deletion_curve_rejects_ambiguous_grid_shape_and_score() -> None:
    with pytest.raises(MetricInputError, match="duplicate"):
        deletion_score_curve(
            lambda item: float(np.sum(item)),
            np.ones(4),
            np.arange(4.0),
            deletion_fractions=(0.1, 0.2),
        )
    with pytest.raises(MetricInputError, match="identical shapes"):
        deletion_score_curve(
            lambda item: float(np.sum(item)),
            np.ones(4),
            np.ones(3),
            deletion_fractions=(0.5,),
        )
    with pytest.raises(MetricInputError, match="finite"):
        deletion_score_curve(
            lambda _item: float("nan"),
            np.ones(4),
            np.arange(4.0),
            deletion_fractions=(0.5,),
        )


def test_attribution_stability_and_topk_contract() -> None:
    first = np.asarray([0.9, 0.1, 0.4, 0.2])
    second = np.asarray([0.8, 0.2, 0.3, 0.1])
    assert spearman_attribution(first, second) == pytest.approx(0.8)
    assert spearman_attribution(first, first * 7.0) == pytest.approx(1.0)
    assert topk_iou(first, second, k=2) == pytest.approx(1.0)
    assert list(attribution_order(np.asarray([1.0, -1.0, 0.5]))) == [0, 1, 2]

    with pytest.raises(MetricInputError, match="constant"):
        spearman_attribution(np.ones(4), np.arange(4.0))
    with pytest.raises(MetricInputError, match=r"\[1, feature_count\)"):
        topk_iou(first, second, k=4)
    with pytest.raises(MetricInputError, match="degenerate magnitude tie"):
        topk_iou(np.asarray([1.0, 1.0, 0.0]), np.asarray([2.0, 1.0, 0.0]), k=1)


def test_rank_metrics_retain_ties_margins_and_undefined_states() -> None:
    first = [0.9, 0.7, 0.5]
    second = [0.6, 0.8, 0.4]
    assert kendall_tau_b(first, second) == pytest.approx(1.0 / 3.0)
    reversal = pairwise_rank_reversal_rate(first, second)
    assert reversal.rate == pytest.approx(1.0 / 3.0)
    assert (reversal.reversals, reversal.comparable_pairs, reversal.excluded_pairs) == (1, 3, 0)

    excluded = pairwise_rank_reversal_rate(
        first,
        second,
        practical_margin_first=0.15,
        practical_margin_second=0.25,
    )
    assert excluded.comparable_pairs == 1
    assert excluded.excluded_pairs == 2
    with pytest.raises(MetricInputError, match="no comparable"):
        pairwise_rank_reversal_rate([1.0, 1.0], [2.0, 2.0])
    with pytest.raises(MetricInputError, match="no pair is orderable"):
        kendall_tau_b([1.0, 1.0], [2.0, 2.0])


def test_observation_statuses_never_substitute_failure_values() -> None:
    measurement = _measurement()
    accepted = MetricObservation(
        schema_version=METRIC_OBSERVATION_SCHEMA_VERSION,
        measurement_ids=(measurement.measurement_id,),
        metric_id="faithfulness.aopc.v1",
        metric_registry_sha256=metric_registry_sha256(),
        protocol_sha256=_digest("7"),
        parameters={"grid": [0.25, 0.5, 1.0]},
        status="accepted",
        value=0.25,
    )
    assert MetricObservation.from_manifest(accepted.to_manifest()) == accepted

    retained_control = MetricObservation(
        schema_version=METRIC_OBSERVATION_SCHEMA_VERSION,
        measurement_ids=(measurement.measurement_id,),
        metric_id="faithfulness.aopc.v1",
        metric_registry_sha256=metric_registry_sha256(),
        protocol_sha256=_digest("7"),
        parameters={"control": "label_leaking"},
        status="control_violating",
        value=0.99,
        reason_code="LABEL_LEAKING_CONTROL",
    )
    assert retained_control.value == pytest.approx(0.99)
    with pytest.raises(MetricInputError, match="must not substitute"):
        MetricObservation(
            schema_version=METRIC_OBSERVATION_SCHEMA_VERSION,
            measurement_ids=(measurement.measurement_id,),
            metric_id="faithfulness.aopc.v1",
            metric_registry_sha256=metric_registry_sha256(),
            protocol_sha256=_digest("7"),
            parameters={},
            status="invalid",
            value=0.5,
            reason_code="INVALID_INPUT",
        )


def test_efficiency_and_coverage_functionals_are_explicit() -> None:
    assert elapsed_time_ms(1_000_000, 2_500_000) == pytest.approx(1.5)
    assert peak_memory_mib(1_049_600, baseline_bytes=1_024) == pytest.approx(1.0)
    assert activation_ratio([0.9, 0.1, 0.4, 0.2], threshold=0.25) == pytest.approx(0.5)
    with pytest.raises(MetricInputError):
        elapsed_time_ms(2, 1)
    with pytest.raises(MetricInputError):
        peak_memory_mib(10, baseline_bytes=11)
    with pytest.raises(MetricInputError):
        activation_ratio([1.0], threshold=-0.1)


def test_registry_has_no_overall_score_and_fixture_is_frozen_non_evidence() -> None:
    expected_metrics = {
        "faithfulness.deletion_at_fraction.v1",
        "faithfulness.aopc.v1",
        "stability.spearman_attribution.v1",
        "stability.topk_iou.v1",
        "ranking.kendall_tau_b.v1",
        "ranking.pairwise_reversal_rate.v1",
        "efficiency.elapsed_time_ms.v1",
        "efficiency.peak_memory_mib.v1",
        "coverage.activation_ratio.v1",
    }
    assert set(METRIC_REGISTRY) == expected_metrics
    assert all("overall" not in metric_id.lower() for metric_id in METRIC_REGISTRY)
    assert assert_metric_compatible(
        _measurement(), "faithfulness.deletion_at_fraction.v1"
    ).metric_id == "faithfulness.deletion_at_fraction.v1"

    limited_adapter = ArrayMeasurementAdapter(
        AdapterIdentity(
            adapter_id="limited",
            adapter_version="1.0.0",
            adapter_sha256=_digest("8"),
            input_kind="intrinsic_operator_weight",
            output_kind="operator_weight",
            capabilities=("dense_attribution",),
        )
    )
    limited = limited_adapter.adapt(
        values=np.ones((1, 4)),
        sample_ids=("sample-1",),
        axes=("sample", "operator"),
        axis_units=("id", "index"),
        coordinate_map_sha256=_digest("8"),
        value_semantics="unsigned_operator_weight",
        locator="operator.npy",
        source=_source(),
        target_id="class:3",
    )
    with pytest.raises(MetricInputError, match="lacks capabilities"):
        assert_metric_compatible(limited, "faithfulness.deletion_at_fraction.v1")

    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "p02_xfd_benchmark_toolkit"
        / "measurement_contract_v1.yaml"
    )
    report = run_conformance_fixture(load_conformance_config(config_path))
    assert report["status"] == "passed"
    assert report["evidence_eligible"] is False
    assert report["metric_count"] == 9
    assert report["measurement_id"] == "5fb0afb8e738a737ff8f0a27eede3cde6e909a7ea9d792551d4eac8aebbbc669"
    assert report["observation_id"] == "6f3e9546260d6f8e86e5a6184ce126043736c179b5c60f1a15038cdbf6d5b992"
