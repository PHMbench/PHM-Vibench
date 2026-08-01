"""Local conformance runner for the P02 measurement contract.

The bundled fixture is explicitly non-evidence. It checks deterministic object
identity, manifest round-tripping, registry identity, and exact metric values.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .measurement import (
    AdapterIdentity,
    ArrayMeasurementAdapter,
    MeasurementContractError,
    MeasurementObject,
    SourceIdentity,
)
from .measurement_metrics import (
    METRIC_REGISTRY,
    METRIC_OBSERVATION_SCHEMA_VERSION,
    MetricObservation,
    activation_ratio,
    aopc,
    assert_metric_compatible,
    deletion_at_fraction,
    deletion_score_curve,
    kendall_tau_b,
    metric_registry_sha256,
    pairwise_rank_reversal_rate,
    peak_memory_mib,
    spearman_attribution,
    topk_iou,
    elapsed_time_ms,
)


CONFIG_SCHEMA_VERSION = "p02.measurement-conformance.v1"


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MeasurementContractError(f"{field} must be a mapping")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MeasurementContractError(f"{field} must be a sequence")
    return value


def _exact_keys(payload: Mapping[str, Any], expected: set[str], field: str) -> None:
    missing = sorted(expected - set(payload))
    extra = sorted(set(payload) - expected)
    if missing or extra:
        raise MeasurementContractError(f"{field} keys mismatch; missing={missing}, extra={extra}")


def load_conformance_config(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = _mapping(payload, "config")
    _exact_keys(
        config,
        {
            "schema_version",
            "contract_mode",
            "evidence_eligible",
            "metric_registry_sha256",
            "measurement_fixture",
            "metric_fixture",
        },
        "config",
    )
    if config["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise MeasurementContractError(f"unsupported config schema: {config['schema_version']!r}")
    if config["contract_mode"] != "conformance_only" or config["evidence_eligible"] is not False:
        raise MeasurementContractError("bundled conformance config must be non-evidence")
    current_registry = metric_registry_sha256()
    if str(config["metric_registry_sha256"]).strip().lower() != current_registry:
        raise MeasurementContractError(
            "configured metric_registry_sha256 does not match the maintained runtime registry"
        )
    return config


def _assert_expected(actual: float, expected: Any, name: str, *, tolerance: float) -> None:
    expected_value = float(expected)
    if not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=tolerance):
        raise MeasurementContractError(
            f"conformance value mismatch for {name}: actual={actual}, expected={expected_value}"
        )


def run_conformance_fixture(config: Mapping[str, Any]) -> dict[str, Any]:
    measurement_fixture = _mapping(config["measurement_fixture"], "measurement_fixture")
    _exact_keys(
        measurement_fixture,
        {
            "adapter",
            "source",
            "values",
            "sample_ids",
            "axes",
            "axis_units",
            "coordinate_map_sha256",
            "value_semantics",
            "locator",
            "target_id",
        },
        "measurement_fixture",
    )
    adapter = ArrayMeasurementAdapter(
        AdapterIdentity.from_dict(_mapping(measurement_fixture["adapter"], "measurement_fixture.adapter"))
    )
    measurement = adapter.adapt(
        values=np.asarray(measurement_fixture["values"], dtype=np.float64),
        sample_ids=tuple(str(value) for value in _sequence(measurement_fixture["sample_ids"], "sample_ids")),
        axes=tuple(str(value) for value in _sequence(measurement_fixture["axes"], "axes")),
        axis_units=tuple(
            str(value) for value in _sequence(measurement_fixture["axis_units"], "axis_units")
        ),
        coordinate_map_sha256=str(measurement_fixture["coordinate_map_sha256"]),
        value_semantics=str(measurement_fixture["value_semantics"]),
        locator=str(measurement_fixture["locator"]),
        source=SourceIdentity.from_dict(_mapping(measurement_fixture["source"], "measurement_fixture.source")),
        target_id=str(measurement_fixture["target_id"]),
    )
    round_trip = MeasurementObject.from_manifest(measurement.to_manifest())
    if round_trip.measurement_id != measurement.measurement_id:
        raise MeasurementContractError("measurement manifest round-trip changed object identity")
    for metric_id in METRIC_REGISTRY:
        assert_metric_compatible(measurement, metric_id)

    metric_fixture = _mapping(config["metric_fixture"], "metric_fixture")
    _exact_keys(
        metric_fixture,
        {
            "input_values",
            "attributions_first",
            "attributions_second",
            "deletion_fractions",
            "deletion_query_fraction",
            "baseline",
            "top_k",
            "ranking_first",
            "ranking_second",
            "practical_margin_first",
            "practical_margin_second",
            "activation_threshold",
            "timing_start_ns",
            "timing_end_ns",
            "peak_memory_bytes",
            "baseline_memory_bytes",
            "protocol_sha256",
            "expected",
            "absolute_tolerance",
        },
        "metric_fixture",
    )

    input_values = np.asarray(metric_fixture["input_values"], dtype=np.float64)
    first = np.asarray(metric_fixture["attributions_first"], dtype=np.float64)
    second = np.asarray(metric_fixture["attributions_second"], dtype=np.float64)
    curve = deletion_score_curve(
        lambda values: float(np.sum(values)),
        input_values,
        first,
        deletion_fractions=metric_fixture["deletion_fractions"],
        baseline=float(metric_fixture["baseline"]),
    )
    reversal = pairwise_rank_reversal_rate(
        metric_fixture["ranking_first"],
        metric_fixture["ranking_second"],
        practical_margin_first=float(metric_fixture["practical_margin_first"]),
        practical_margin_second=float(metric_fixture["practical_margin_second"]),
    )
    actual = {
        "deletion_at_fraction": deletion_at_fraction(
            curve, float(metric_fixture["deletion_query_fraction"])
        ),
        "aopc": aopc(curve),
        "spearman_attribution": spearman_attribution(first, second),
        "topk_iou": topk_iou(first, second, k=int(metric_fixture["top_k"])),
        "kendall_tau_b": kendall_tau_b(
            metric_fixture["ranking_first"], metric_fixture["ranking_second"]
        ),
        "pairwise_reversal_rate": reversal.rate,
        "activation_ratio": activation_ratio(
            first, threshold=float(metric_fixture["activation_threshold"])
        ),
        "elapsed_time_ms": elapsed_time_ms(
            int(metric_fixture["timing_start_ns"]), int(metric_fixture["timing_end_ns"])
        ),
        "peak_memory_mib": peak_memory_mib(
            int(metric_fixture["peak_memory_bytes"]),
            baseline_bytes=int(metric_fixture["baseline_memory_bytes"]),
        ),
    }
    expected = _mapping(metric_fixture["expected"], "metric_fixture.expected")
    _exact_keys(expected, set(actual), "metric_fixture.expected")
    tolerance = float(metric_fixture["absolute_tolerance"])
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise MeasurementContractError("absolute_tolerance must be finite and non-negative")
    for name, value in actual.items():
        _assert_expected(value, expected[name], name, tolerance=tolerance)

    observation = MetricObservation(
        schema_version=METRIC_OBSERVATION_SCHEMA_VERSION,
        measurement_ids=(measurement.measurement_id,),
        metric_id="faithfulness.deletion_at_fraction.v1",
        metric_registry_sha256=metric_registry_sha256(),
        protocol_sha256=str(metric_fixture["protocol_sha256"]),
        parameters={"fraction": float(metric_fixture["deletion_query_fraction"])},
        status="accepted",
        value=actual["deletion_at_fraction"],
    )
    if MetricObservation.from_manifest(observation.to_manifest()).observation_id != observation.observation_id:
        raise MeasurementContractError("metric observation round-trip changed object identity")

    return {
        "status": "passed",
        "contract_mode": "conformance_only",
        "evidence_eligible": False,
        "measurement_id": measurement.measurement_id,
        "measurement_schema_version": measurement.schema_version,
        "metric_registry_sha256": metric_registry_sha256(),
        "metric_count": len(METRIC_REGISTRY),
        "observation_id": observation.observation_id,
        "values": actual,
        "rank_reversal_counts": {
            "reversals": reversal.reversals,
            "comparable_pairs": reversal.comparable_pairs,
            "excluded_pairs": reversal.excluded_pairs,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the P02 measurement contract fixture.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_conformance_config(args.config)
        report = run_conformance_fixture(config)
    except (OSError, yaml.YAMLError, MeasurementContractError) as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "contract_mode": "conformance_only",
                    "evidence_eligible": False,
                    "error": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
