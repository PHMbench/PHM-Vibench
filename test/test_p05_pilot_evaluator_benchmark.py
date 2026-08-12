from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import src.explain_factory.p05_pilot_evaluator_benchmark as benchmark_module
from src.explain_factory.p05_d03_noise_intervention import P05D03Result
from src.explain_factory.p05_intervention_runner import P05ActualInterventionResult
from src.explain_factory.p05_pilot_evaluator_benchmark import (
    create_p05_pilot_evaluator_benchmark,
    verify_p05_pilot_evaluator_benchmark,
)


CONFIG_HASH = "a" * 64
CHECKPOINT_HASH = "b" * 64
MODEL_HASH = "c" * 64
CENTRAL_HASH = "d" * 64
D03_HASH = "e" * 64


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    content = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _stable_ids() -> list[str]:
    return [f"record-{index:03d}:{index * 4}:{index * 4 + 4}" for index in range(256)]


def _central_result(
    *,
    ids: list[str] | None = None,
    metadata_updates: dict | None = None,
    timing_updates: dict | None = None,
) -> P05ActualInterventionResult:
    selected_ids = ids or _stable_ids()
    records, starts, ends = zip(
        *(sample_id.split(":") for sample_id in selected_ids),
        strict=True,
    )
    metadata = {
        "protocol": {"actual_forward_calls": 43},
        "provenance": {
            "dataset": "XJTU",
            "split": "validation",
            "model_seed": 20260801,
            "config_sha256": CONFIG_HASH,
            "checkpoint_sha256": CHECKPOINT_HASH,
            "model_sha256": MODEL_HASH,
        },
        "selection": {
            "benchmark_first_n": 256,
            "input_count": 4096,
            "kind": "first_n_after_stable_sample_id_sort",
            "selected_count": 256,
        },
    }
    if metadata_updates:
        for name, update in metadata_updates.items():
            metadata[name] = {**metadata[name], **update}
    timing = {
        "original_seconds": 0.5,
        "deletion_seconds": 1.5,
        "shuffle_seconds": 2.0,
        "total_seconds": 4.0,
        "device_type": "cuda",
        "performance_claim_allowed": False,
        "scope": "diagnostic_wall_clock_boundary_only",
    }
    if timing_updates:
        timing.update(timing_updates)
    return P05ActualInterventionResult(
        arrays={
            "sample_id": np.asarray(selected_ids),
            "record_id": np.asarray(records),
            "window_start": np.asarray(starts, dtype=np.int64),
            "window_end": np.asarray(ends, dtype=np.int64),
        },
        metadata=metadata,
        timing=timing,
        semantic_sha256=CENTRAL_HASH,
    )


def _d03_result(tmp_path: Path) -> tuple[P05D03Result, dict]:
    artifact_dir = tmp_path / "d03"
    artifact_dir.mkdir()
    arrays_path = artifact_dir / "d03_arrays.npz"
    np.savez(arrays_path, sample_id=np.asarray(_stable_ids()))
    manifest = {
        "content": {
            "npz_sha256": _sha256_file(arrays_path),
            "semantic_sha256": D03_HASH,
        },
        "execution": {
            "actual_forward_calls": 33,
            "budget_retained": None,
            "chunk_count": 1,
            "chunk_size": 256,
            "device_class": "cuda",
            "phase": "pilot_benchmark",
        },
        "input_binding": {
            "input_count": 4096,
            "selected_count": 256,
            "selection": "first_256_after_stable_sample_id_sort",
        },
        "protocol": {"total_noise_draws_per_sample": 32},
        "partition_coverage": {
            "coverage": "exact",
            "expected_sample_count": 4096,
            "expected_sample_id_semantic_sha256": "1" * 64,
            "observed_sample_count": 4096,
            "observed_sample_id_semantic_sha256": "1" * 64,
            "selected_sample_count": 256,
            "selected_sample_id_semantic_sha256": _canonical_sha256(_stable_ids()),
        },
        "provenance": {
            "dataset": "XJTU",
            "split": "validation",
            "model_seed": 20260801,
            "config_sha256": CONFIG_HASH,
            "checkpoint_sha256": CHECKPOINT_HASH,
            "device_uuid": "GPU-pilot-test",
            "model_sha256": MODEL_HASH,
            "physical_gpu_index": 1,
        },
        "sample_count": 256,
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = P05D03Result(
        artifact_dir=artifact_dir,
        arrays_path=arrays_path,
        manifest_path=manifest_path,
        semantic_sha256=D03_HASH,
        arrays_sha256=_sha256_file(arrays_path),
        manifest_sha256=_sha256_file(manifest_path),
        status="created",
        timing={
            "noise_forward_seconds": 6.0,
            "original_forward_seconds": 2.0,
            "total_seconds": 8.0,
            "performance_claim_allowed": False,
            "scope": "diagnostic_wall_clock_boundary_only",
        },
    )
    return result, manifest


def _patch_source_verifiers(monkeypatch, d03_manifest: dict) -> None:
    monkeypatch.setattr(
        benchmark_module,
        "verify_p05_actual_intervention_result",
        lambda result: None,
    )
    monkeypatch.setattr(
        benchmark_module.d03_module,
        "verify_p05_d03_artifact",
        lambda artifact_dir: d03_manifest,
    )


def test_benchmark_is_atomic_create_only_and_non_evidentiary(
    tmp_path, monkeypatch
) -> None:
    central = _central_result()
    d03, d03_manifest = _d03_result(tmp_path)
    _patch_source_verifiers(monkeypatch, d03_manifest)

    result = create_p05_pilot_evaluator_benchmark(
        tmp_path / "benchmark",
        central_result=central,
        d03_result=d03,
    )

    assert result.status == "created"
    assert result.manifest_sha256 == _sha256_file(result.manifest_path)
    manifest = verify_p05_pilot_evaluator_benchmark(result.package_dir)
    assert manifest["status"] == "engineering_non_evidence"
    assert manifest["conclusion_control"] == {
        "budget_decision": "not_performed",
        "claim_decisions": "forbidden",
        "paper_evidence": False,
        "performance_conclusion": "forbidden",
        "scientific_status": "unadjudicated",
    }
    assert manifest["purpose"] == {
        "budget_cap_gpu_hours": 168,
        "makes_budget_decision": False,
        "role": "observed_input_to_gpu_hour_budget_forecast_only",
    }
    central_timing = manifest["benchmarks"]["central_e1_e2"]
    d03_timing = manifest["benchmarks"]["d03"]
    assert central_timing["actual_forward_calls"] == 43
    assert central_timing["seconds_per_window"] == 4.0 / 256
    assert central_timing["source_semantic_sha256"] == CENTRAL_HASH
    assert central_timing["components"] == {
        "consequent_shuffles": {
            "forward_calls_per_window": 32,
            "seconds_per_forward_call_per_window": 2.0 / 256 / 32,
            "seconds_per_window": 2.0 / 256,
            "total_seconds": 2.0,
        },
        "original_trace": {
            "forward_calls_per_window": 1,
            "seconds_per_forward_call_per_window": 0.5 / 256,
            "seconds_per_window": 0.5 / 256,
            "total_seconds": 0.5,
        },
        "rule_deletions": {
            "forward_calls_per_window": 10,
            "seconds_per_forward_call_per_window": 1.5 / 256 / 10,
            "seconds_per_window": 1.5 / 256,
            "total_seconds": 1.5,
        },
    }
    assert d03_timing["actual_forward_calls"] == 33
    assert d03_timing["seconds_per_window"] == 8.0 / 256
    assert d03_timing["source_semantic_sha256"] == D03_HASH
    assert d03_timing["components"] == {
        "noise_draws": {
            "forward_calls_per_window": 32,
            "seconds_per_forward_call_per_window": 6.0 / 256 / 32,
            "seconds_per_window": 6.0 / 256,
            "total_seconds": 6.0,
        },
        "original_trace": {
            "forward_calls_per_window": 1,
            "seconds_per_forward_call_per_window": 2.0 / 256,
            "seconds_per_window": 2.0 / 256,
            "total_seconds": 2.0,
        },
    }
    assert manifest["shared_provenance"] == {
        "checkpoint_sha256": CHECKPOINT_HASH,
        "config_sha256": CONFIG_HASH,
        "device_uuid": "GPU-pilot-test",
        "model_sha256": MODEL_HASH,
        "physical_gpu_index": 1,
    }

    with pytest.raises(FileExistsError, match="create-only"):
        create_p05_pilot_evaluator_benchmark(
            result.package_dir,
            central_result=None,
            d03_result=None,
        )


@pytest.mark.parametrize(
    ("central_updates", "timing_updates", "d03_updates", "message"),
    [
        ({"selection": {"benchmark_first_n": None}}, None, None, "first-256"),
        ({"protocol": {"actual_forward_calls": 42}}, None, None, "43"),
        ({"provenance": {"split": "test"}}, None, None, "validation"),
        ({"provenance": {"model_seed": 42}}, None, None, "20260801"),
        ({"selection": {"input_count": 2048}}, None, None, "same validation"),
        (None, {"total_seconds": float("nan")}, None, "finite non-negative"),
        (None, None, {"execution": {"phase": "budget_retained_secondary"}}, "phase"),
        (None, None, {"execution": {"actual_forward_calls": 32}}, "33"),
        (None, None, {"protocol": {"total_noise_draws_per_sample": 31}}, "32"),
        (
            None,
            None,
            {
                "partition_coverage": {
                    "selected_sample_id_semantic_sha256": "0" * 64
                }
            },
            "partition coverage",
        ),
        (None, None, {"provenance": {"checkpoint_sha256": "f" * 64}}, "differ"),
    ],
)
def test_benchmark_fails_closed_on_protocol_or_provenance_drift(
    tmp_path,
    monkeypatch,
    central_updates,
    timing_updates,
    d03_updates,
    message,
) -> None:
    central = _central_result(
        metadata_updates=central_updates,
        timing_updates=timing_updates,
    )
    d03, manifest = _d03_result(tmp_path)
    if d03_updates:
        for name, update in d03_updates.items():
            manifest[name] = {**manifest[name], **update}
    _patch_source_verifiers(monkeypatch, manifest)

    with pytest.raises(ValueError, match=message):
        create_p05_pilot_evaluator_benchmark(
            tmp_path / "benchmark",
            central_result=central,
            d03_result=d03,
        )


def test_benchmark_requires_identical_complete_stable_ids(
    tmp_path, monkeypatch
) -> None:
    central_ids = _stable_ids()
    central_ids[-1] = "unexpected:1020:1024"
    central = _central_result(ids=central_ids)
    d03, manifest = _d03_result(tmp_path)
    _patch_source_verifiers(monkeypatch, manifest)

    with pytest.raises(ValueError, match="same 256 sample IDs"):
        create_p05_pilot_evaluator_benchmark(
            tmp_path / "different",
            central_result=central,
            d03_result=d03,
        )

    duplicate = _stable_ids()
    duplicate[-1] = duplicate[-2]
    with pytest.raises(ValueError, match="duplicate"):
        create_p05_pilot_evaluator_benchmark(
            tmp_path / "duplicate",
            central_result=_central_result(ids=duplicate),
            d03_result=d03,
        )


def test_benchmark_rejects_chunked_pilot_d03_forward_accounting(
    tmp_path, monkeypatch
) -> None:
    central = _central_result()
    d03, manifest = _d03_result(tmp_path)
    manifest["execution"].update(
        {"actual_forward_calls": 66, "chunk_count": 2, "chunk_size": 128}
    )
    _patch_source_verifiers(monkeypatch, manifest)

    with pytest.raises(ValueError, match="one exact batch of 256"):
        create_p05_pilot_evaluator_benchmark(
            tmp_path / "chunked",
            central_result=central,
            d03_result=d03,
        )


def test_benchmark_verifier_rejects_rehashed_conclusion_promotion(
    tmp_path, monkeypatch
) -> None:
    central = _central_result()
    d03, manifest = _d03_result(tmp_path)
    _patch_source_verifiers(monkeypatch, manifest)
    result = create_p05_pilot_evaluator_benchmark(
        tmp_path / "benchmark",
        central_result=central,
        d03_result=d03,
    )
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    payload["conclusion_control"]["performance_conclusion"] = "pass"
    semantic = {name: payload[name] for name in payload if name != "content"}
    payload["content"]["semantic_sha256"] = _canonical_sha256(semantic)
    result.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="conclusion control"):
        verify_p05_pilot_evaluator_benchmark(result.package_dir)


@pytest.mark.parametrize(
    "invalid_content",
    [
        '{"schema_name":"a","schema_name":"b"}',
        '{"schema_name":"a","value":NaN}',
        '{"schema_name":"a","value":Infinity}',
        '{"schema_name":"a","value":-Infinity}',
    ],
)
def test_benchmark_verifier_rejects_non_strict_json(
    tmp_path, invalid_content
) -> None:
    package = tmp_path / hashlib.sha256(invalid_content.encode()).hexdigest()[:8]
    package.mkdir()
    (package / "manifest.json").write_text(invalid_content, encoding="utf-8")

    with pytest.raises(ValueError, match="strict finite JSON"):
        verify_p05_pilot_evaluator_benchmark(package)


def test_benchmark_rejects_tampered_d03_result_binding(tmp_path, monkeypatch) -> None:
    central = _central_result()
    d03, manifest = _d03_result(tmp_path)
    _patch_source_verifiers(monkeypatch, manifest)

    with pytest.raises(ValueError, match="arrays SHA-256"):
        create_p05_pilot_evaluator_benchmark(
            tmp_path / "benchmark",
            central_result=central,
            d03_result=replace(d03, arrays_sha256="0" * 64),
        )
