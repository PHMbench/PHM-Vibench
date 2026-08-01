from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.p01_collect_attempts import main as collect_main
from src.utils.p01_attempt_ledger import (
    AttemptCell,
    BUDGET_CONTRACT,
    collect_attempt_ledger,
    write_collection,
)
from test.test_p01_statistics import _write_prediction


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell(*, arm: str = "FULL", fold: int = 0, seed: int = 42) -> dict:
    return {
        "protocol_id": "P01-G040-v1",
        "dataset_key": "CWRU",
        "dataset_slug": "cwru",
        "dataset_id": 1,
        "arm_id": arm,
        "outer_fold": fold,
        "training_seed": seed,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("protocol_id", "P01-G040-unregistered", "outside the frozen protocol"),
        ("arm_id", "UNREGISTERED", "not registered"),
    ],
)
def test_attempt_cell_rejects_unregistered_identity(
    field: str, value: str, message: str
) -> None:
    cell = _cell()
    cell[field] = value
    with pytest.raises(ValueError, match=message):
        AttemptCell.from_mapping(cell, "cell")


def _canonical_prediction(
    root: Path, run_dir: Path, cell: dict, attempt_id: int
) -> Path:
    staged = _write_prediction(
        root / f"staging-{cell['arm_id']}-{attempt_id}",
        arm=cell["arm_id"],
        seed=cell["training_seed"],
        fold=cell["outer_fold"],
    )
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    staged.parent.rename(run_dir)
    artifacts = run_dir / "artifacts"
    artifacts.mkdir()
    moves = {
        "predictions.npz": artifacts / "predictions.npz",
        "predictions.manifest.json": artifacts / "predictions.manifest.json",
        "best_checkpoint.manifest.json": artifacts / "best_checkpoint.manifest.json",
        "data_snapshot.manifest.json": artifacts / "data_snapshot.manifest.json",
        "trainer_metrics.manifest.json": artifacts / "trainer_metrics.manifest.json",
    }
    for source_name, target in moves.items():
        (run_dir / source_name).rename(target)

    checkpoint = run_dir / "best.ckpt"
    checkpoint_manifest = artifacts / "best_checkpoint.manifest.json"
    checkpoint_payload = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
    checkpoint_payload["path"] = str(checkpoint.resolve())
    checkpoint_manifest.write_text(
        json.dumps(checkpoint_payload, sort_keys=True), encoding="utf-8"
    )
    metrics = run_dir / "logs" / "version_0" / "metrics.csv"
    metrics_manifest = artifacts / "trainer_metrics.manifest.json"
    metrics_payload = json.loads(metrics_manifest.read_text(encoding="utf-8"))
    metrics_payload["metrics_path"] = str(metrics.resolve())
    metrics_manifest.write_text(
        json.dumps(metrics_payload, sort_keys=True), encoding="utf-8"
    )

    invocation = run_dir / "invocation.json"
    invocation_payload = json.loads(invocation.read_text(encoding="utf-8"))
    invocation_payload["paper"]["attempt_id"] = attempt_id
    invocation.write_text(
        json.dumps(invocation_payload, sort_keys=True), encoding="utf-8"
    )
    data_snapshot = artifacts / "data_snapshot.manifest.json"
    data_snapshot_payload = json.loads(data_snapshot.read_text(encoding="utf-8"))
    data_snapshot_payload["paper"]["attempt_id"] = attempt_id
    data_snapshot_payload["invocation_sha256"] = _sha(invocation)
    data_snapshot.write_text(
        json.dumps(data_snapshot_payload, sort_keys=True), encoding="utf-8"
    )

    prediction = artifacts / "predictions.npz"
    prediction_manifest = artifacts / "predictions.manifest.json"
    payload = json.loads(prediction_manifest.read_text(encoding="utf-8"))
    provenance = payload["provenance"]
    provenance.update(
        {
            "attempt_id": attempt_id,
            "config_snapshot_path": str((run_dir / "config_snapshot.yaml").resolve()),
            "invocation_path": str(invocation.resolve()),
            "invocation_sha256": _sha(invocation),
            "best_checkpoint_manifest_path": str(checkpoint_manifest.resolve()),
            "best_checkpoint_manifest_sha256": _sha(checkpoint_manifest),
            "checkpoint_path": str(checkpoint.resolve()),
            "split_manifest_path": str((run_dir / "split.json").resolve()),
            "data_snapshot_manifest_path": str(
                data_snapshot.resolve()
            ),
            "data_snapshot_manifest_sha256": _sha(data_snapshot),
            "trainer_metrics_manifest_path": str(metrics_manifest.resolve()),
            "trainer_metrics_manifest_sha256": _sha(metrics_manifest),
            "trainer_metrics_path": str(metrics.resolve()),
        }
    )
    prediction_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return prediction


def _write_attempt(
    root: Path,
    cell: dict,
    *,
    attempt_id: int,
    status: str,
    attempted_epochs: int,
) -> dict:
    run_dir = (
        root
        / cell["protocol_id"]
        / cell["dataset_slug"]
        / cell["arm_id"]
        / f"fold_{cell['outer_fold']}"
        / f"seed_{cell['training_seed']}"
        / f"attempt_{attempt_id}"
    )
    prediction = None
    if status == "succeeded":
        prediction = _canonical_prediction(root, run_dir, cell, attempt_id)
    else:
        run_dir.mkdir(parents=True)
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(exist_ok=True)
    failure_class = {
        "infrastructure_failed": "infrastructure",
        "algorithm_failed": "algorithm",
    }.get(status)
    terminal = {
        "schema_version": 1,
        "cell": cell,
        "attempt_id": attempt_id,
        "status": status,
        "attempted_epochs": attempted_epochs,
        "failure": (
            None
            if failure_class is None
            else {"class": failure_class, "reason": f"synthetic {failure_class} failure"}
        ),
    }
    terminal_path = artifacts / "terminal_status.manifest.json"
    terminal_path.write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "cell": cell,
        "attempt_id": attempt_id,
        "status": status,
        "attempted_epochs": attempted_epochs,
        "run_dir": str(run_dir.resolve()),
        "terminal_status_path": str(terminal_path.resolve()),
        "terminal_status_sha256": _sha(terminal_path),
    }
    if prediction is not None:
        prediction_manifest = prediction.with_suffix(".manifest.json")
        manifest.update(
            {
                "prediction_path": str(prediction.resolve()),
                "prediction_sha256": _sha(prediction),
                "prediction_manifest_path": str(prediction_manifest.resolve()),
                "prediction_manifest_sha256": _sha(prediction_manifest),
            }
        )
    manifest_path = run_dir / "attempt.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "cell": cell,
        "attempt_id": attempt_id,
        "status": status,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha(manifest_path),
    }


def _write_ledger(path: Path, cells: list[dict], attempts: list[dict]) -> tuple[Path, str]:
    attempt_root = Path(attempts[0]["manifest_path"]).parents[6]
    payload = {
        "schema_version": 1,
        "kind": "p01_attempt_ledger",
        "protocol_id": "P01-G040-v1",
        "attempt_root": str(attempt_root.resolve()),
        "budget": BUDGET_CONTRACT,
        "expected_cells": cells,
        "attempts": attempts,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path, _sha(path)


def test_collector_selects_attempt_zero_and_cli_writes_once(tmp_path: Path) -> None:
    cell = _cell()
    attempt = _write_attempt(
        tmp_path / "runs", cell, attempt_id=0, status="succeeded", attempted_epochs=50
    )
    ledger, ledger_sha = _write_ledger(tmp_path / "ledger.json", [cell], [attempt])
    output = tmp_path / "collection.json"
    assert collect_main(
        ["--ledger", str(ledger), "--ledger-sha256", ledger_sha, "--output", str(output)]
    ) == 0
    collected = json.loads(output.read_text(encoding="utf-8"))
    assert collected["selected_attempts"][0]["attempt_id"] == 0
    assert collected["scorer_predictions"] == [attempt["manifest_path"].replace(
        "attempt.manifest.json", "artifacts/predictions.npz"
    )]
    assert collected["failed_attempts"] == []
    with pytest.raises(FileExistsError):
        write_collection(output, collected)


def test_infrastructure_failure_allows_one_retry_and_is_preserved(tmp_path: Path) -> None:
    cell = _cell()
    attempt_zero = _write_attempt(
        tmp_path / "runs", cell, attempt_id=0,
        status="infrastructure_failed", attempted_epochs=3,
    )
    attempt_one = _write_attempt(
        tmp_path / "runs", cell, attempt_id=1,
        status="succeeded", attempted_epochs=50,
    )
    ledger, ledger_sha = _write_ledger(
        tmp_path / "ledger.json", [cell], [attempt_zero, attempt_one]
    )
    collected = collect_attempt_ledger(ledger, ledger_sha)
    assert collected["selected_attempts"][0]["attempt_id"] == 1
    assert collected["budget"]["retry_attempts"] == 1
    assert collected["budget"]["attempted_epochs"] == 53
    assert collected["failed_attempts"][0]["status"] == "infrastructure_failed"
    assert collected["failed_attempts"][0]["failure"]["class"] == "infrastructure"


def test_algorithm_failure_cannot_be_retried(tmp_path: Path) -> None:
    cell = _cell()
    attempts = [
        _write_attempt(
            tmp_path / "runs", cell, attempt_id=0,
            status="algorithm_failed", attempted_epochs=50,
        ),
        _write_attempt(
            tmp_path / "runs", cell, attempt_id=1,
            status="succeeded", attempted_epochs=50,
        ),
    ]
    ledger, ledger_sha = _write_ledger(tmp_path / "ledger.json", [cell], attempts)
    with pytest.raises(ValueError, match="only after attempt 0 infrastructure_failed"):
        collect_attempt_ledger(ledger, ledger_sha)


def test_collector_rejects_duplicate_valid_and_missing_cells(tmp_path: Path) -> None:
    cell = _cell()
    success_zero = _write_attempt(
        tmp_path / "runs", cell, attempt_id=0, status="succeeded", attempted_epochs=50
    )
    success_one = _write_attempt(
        tmp_path / "runs", cell, attempt_id=1, status="succeeded", attempted_epochs=50
    )
    ledger, ledger_sha = _write_ledger(
        tmp_path / "duplicate.json", [cell], [success_zero, success_one]
    )
    with pytest.raises(ValueError, match="duplicate terminal-valid"):
        collect_attempt_ledger(ledger, ledger_sha)

    missing_cell = _cell(arm="B4-GATTN")
    missing_root_success = _write_attempt(
        tmp_path / "missing-runs", cell, attempt_id=0,
        status="succeeded", attempted_epochs=50,
    )
    ledger, ledger_sha = _write_ledger(
        tmp_path / "missing.json", [cell, missing_cell], [missing_root_success]
    )
    with pytest.raises(ValueError, match="missing mandatory attempt 0"):
        collect_attempt_ledger(ledger, ledger_sha)


def test_collector_rejects_status_and_hash_drift(tmp_path: Path) -> None:
    cell = _cell()
    attempt = _write_attempt(
        tmp_path / "runs", cell, attempt_id=0, status="succeeded", attempted_epochs=50
    )
    drifted = dict(attempt)
    drifted["status"] = "algorithm_failed"
    ledger, ledger_sha = _write_ledger(tmp_path / "status.json", [cell], [drifted])
    with pytest.raises(ValueError, match="status or attempt_id drift"):
        collect_attempt_ledger(ledger, ledger_sha)

    ledger, ledger_sha = _write_ledger(tmp_path / "hash.json", [cell], [attempt])
    Path(attempt["manifest_path"]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Attempt manifest hash mismatch"):
        collect_attempt_ledger(ledger, ledger_sha)
