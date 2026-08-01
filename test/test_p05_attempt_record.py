import json
from pathlib import Path

import pytest

import src.utils.p05_attempt_record as attempt_record
from src.utils.p05_attempt_record import (
    begin_p05_attempt,
    finish_p05_attempt,
    invalidate_p05_attempt,
)


HASHES = {
    "source_metadata_sha256": "01" * 32,
    "derived_metadata_sha256": "02" * 32,
    "signal_cache_manifest_sha256": "03" * 32,
    "split_manifest_sha256": "04" * 32,
    "config_snapshot_sha256": "05" * 32,
    "code_snapshot_sha256": "06" * 32,
    "normalization_sha256": "07" * 32,
    "train_weight_plan_sha256": "08" * 32,
    "validation_weight_plan_sha256": "09" * 32,
}
COMMAND = [
    "conda",
    "run",
    "-n",
    "LQ_signal",
    "python",
    "main.py",
    "--config",
    "job/config.yaml",
]
STARTED = "2026-08-01T00:00:00+00:00"
FINISHED = "2026-08-01T00:01:00+00:00"


def _begin(package: Path, **overrides):
    values = {
        "attempt_id": package.name,
        "arm_id": "P05-M",
        "phase": "decisive",
        "dataset_id": 2,
        "seed": 42,
        "command_argv": COMMAND,
        "working_directory": package.parent,
        "package_versions": {
            "numpy": "2.0.0",
            "python": "3.10.0",
            "pytorch_lightning": "2.0.0",
            "torch": "2.0.0",
        },
        "device_identity": {
            "cuda_visible_devices": "0",
            "gpu_uuid": "GPU-test",
            "physical_gpu_index": 0,
        },
        "provenance": HASHES,
        "started_at_utc": STARTED,
    }
    values.update(overrides)
    return begin_p05_attempt(package, **values)


def test_completed_attempt_is_immutable_hashed_and_claim_neutral(tmp_path) -> None:
    package = tmp_path / "attempt-001"
    started = _begin(package)
    completed = finish_p05_attempt(
        package,
        status="completed",
        output_artifact_sha256={"run_contract": "aa" * 32, "test_logits": "bb" * 32},
        finished_at_utc=FINISHED,
    )

    start = json.loads(started.start_path.read_text(encoding="utf-8"))
    terminal = json.loads(completed.terminal_path.read_text(encoding="utf-8"))
    assert start["attempt"]["status"] == "running"
    assert start["execution"]["command_argv"] == COMMAND
    assert terminal["terminal"] == {
        "claim_decision": "not_performed",
        "finished_at_utc": FINISHED,
        "status": "completed",
    }
    assert terminal["start_semantic_sha256"] == started.semantic_sha256
    assert set(package.iterdir()) == {
        package / "start.json",
        package / "terminal.json",
        package / "invalidations",
    }
    with pytest.raises(FileExistsError, match="already exists"):
        finish_p05_attempt(
            package,
            status="completed",
            output_artifact_sha256={"run_contract": "aa" * 32},
            finished_at_utc=FINISHED,
        )


def test_failed_attempt_is_retained_and_retry_requires_identical_contract(tmp_path) -> None:
    failed_package = tmp_path / "attempt-failed"
    failed = _begin(failed_package)
    finish_p05_attempt(
        failed_package,
        status="failed",
        missing_outputs={"checkpoint": "training terminated before checkpoint selection"},
        failure_category="infrastructure",
        failure_type="RuntimeError",
        failure_message="CUDA worker exited",
        finished_at_utc=FINISHED,
    )

    retry_package = tmp_path / "attempt-retry"
    retried = _begin(
        retry_package,
        retry_of_package=failed_package,
        retry_reason="same-config infrastructure retry",
    )
    retry = json.loads(retried.start_path.read_text(encoding="utf-8"))["retry"]
    assert retry["retry_of_start_semantic_sha256"] == failed.semantic_sha256

    drifted = dict(HASHES)
    drifted["config_snapshot_sha256"] = "ff" * 32
    with pytest.raises(ValueError, match="must preserve"):
        _begin(
            tmp_path / "attempt-bad-retry",
            provenance=drifted,
            retry_of_package=failed_package,
            retry_reason="not actually identical",
        )

    scientific_package = tmp_path / "attempt-scientific-failure"
    _begin(scientific_package)
    finish_p05_attempt(
        scientific_package,
        status="failed",
        failure_category="scientific",
        failure_type="FloatingPointError",
        failure_message="non-finite evaluator output",
        finished_at_utc=FINISHED,
    )
    with pytest.raises(ValueError, match="infrastructure-classified"):
        _begin(
            tmp_path / "attempt-invalid-retry-category",
            retry_of_package=scientific_package,
            retry_reason="scientific failures are not infrastructure retries",
        )


def test_missing_start_provenance_must_be_explicit(tmp_path) -> None:
    missing = dict(HASHES)
    missing["signal_cache_manifest_sha256"] = None
    with pytest.raises(ValueError, match="requires an unavailable reason"):
        _begin(tmp_path / "attempt-missing", provenance=missing)

    result = _begin(
        tmp_path / "attempt-preflight",
        provenance=missing,
        unavailable_reasons={
            "signal_cache_manifest_sha256": "cache preflight failed before hash resolution"
        },
    )
    manifest = json.loads(result.start_path.read_text(encoding="utf-8"))
    assert manifest["provenance"]["signal_cache_manifest_sha256"] is None


def test_invalidation_appends_without_rewriting_attempt(tmp_path) -> None:
    package = tmp_path / "attempt-invalidated"
    started = _begin(package)
    completed = finish_p05_attempt(
        package,
        status="completed",
        output_artifact_sha256={"test_logits": "bb" * 32},
        finished_at_utc=FINISHED,
    )
    start_bytes = started.start_path.read_bytes()
    terminal_bytes = completed.terminal_path.read_bytes()

    invalidation = invalidate_p05_attempt(
        package,
        invalidation_id="code-change-001",
        reason="scientific inference code changed",
        changed_code_sha256="cc" * 32,
        affected_output_names=["test_logits"],
        invalidated_at_utc="2026-08-02T00:00:00+00:00",
    )
    payload = json.loads(invalidation.invalidation_path.read_text(encoding="utf-8"))
    assert payload["invalidation"]["claim_use_allowed"] is False
    assert payload["terminal_semantic_sha256"] == completed.semantic_sha256
    assert started.start_path.read_bytes() == start_bytes
    assert completed.terminal_path.read_bytes() == terminal_bytes
    with pytest.raises(FileExistsError, match="already exists"):
        invalidate_p05_attempt(
            package,
            invalidation_id="code-change-001",
            reason="duplicate",
            changed_code_sha256="cc" * 32,
            affected_output_names=["test_logits"],
        )


def test_tampering_and_unknown_invalidation_output_fail_closed(tmp_path) -> None:
    package = tmp_path / "attempt-tampered"
    started = _begin(package)
    finish_p05_attempt(
        package,
        status="completed",
        output_artifact_sha256={"test_logits": "bb" * 32},
        finished_at_utc=FINISHED,
    )
    with pytest.raises(ValueError, match="absent from terminal"):
        invalidate_p05_attempt(
            package,
            invalidation_id="bad-output",
            reason="test",
            changed_code_sha256="cc" * 32,
            affected_output_names=["not-recorded"],
        )

    payload = json.loads(started.start_path.read_text(encoding="utf-8"))
    payload["attempt"]["seed"] = 123
    started.start_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic hash mismatch"):
        finish_p05_attempt(
            package,
            status="failed",
            failure_category="provenance",
            failure_type="ValueError",
            failure_message="tampered",
        )


def test_atomic_terminal_write_failure_leaves_no_partial_record(tmp_path, monkeypatch) -> None:
    package = tmp_path / "attempt-write-failure"
    _begin(package)

    def fail_rename(source, target):
        raise RuntimeError("synthetic terminal install failure")

    monkeypatch.setattr(attempt_record, "_rename_noreplace", fail_rename)
    with pytest.raises(RuntimeError, match="synthetic terminal install failure"):
        finish_p05_attempt(
            package,
            status="failed",
            failure_category="implementation",
            failure_type="RuntimeError",
            failure_message="synthetic",
            finished_at_utc=FINISHED,
        )
    assert not (package / "terminal.json").exists()
    assert not list(package.glob(".terminal.json.*.tmp"))
