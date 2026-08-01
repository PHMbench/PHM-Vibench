"""Fail-closed validation and collection of P01 training attempts."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.utils.p01_statistics import (
    DATASET_IDENTIFIERS,
    DATASET_OUTER_FOLDS,
    TRAINING_SEEDS,
    load_prediction_artifact,
)


MAX_RETRY_ATTEMPTS = 24
MAX_TOTAL_ATTEMPTS = 267
MAX_PRIMARY_CELLS = MAX_TOTAL_ATTEMPTS - MAX_RETRY_ATTEMPTS
MAX_EPOCHS_PER_ATTEMPT = 50
MAX_ATTEMPTED_EPOCHS = 13_350
PROTOCOL_ID = "P01-G040-v1"
REGISTERED_ARMS = frozenset(
    {
        "FULL",
        "B1-1D",
        "B2-2D",
        "B3-CONCAT",
        "B4-GATTN",
        "B5-NCE",
        "TRAIN-MISPAIR",
        "A-NO-ALIGN",
        "A-NO-PRIVATE-IND",
        "A-NO-REC",
        "A-NO-VAR",
        "A-SHARED-ONLY",
        "S-SHARED-ONLY-CAPACITY",
    }
)
TERMINAL_STATUSES = frozenset(
    {"succeeded", "infrastructure_failed", "algorithm_failed"}
)
BUDGET_CONTRACT = {
    "max_retry_attempts": MAX_RETRY_ATTEMPTS,
    "max_total_attempts": MAX_TOTAL_ATTEMPTS,
    "max_epochs_per_attempt": MAX_EPOCHS_PER_ATTEMPT,
    "max_attempted_epochs": MAX_ATTEMPTED_EPOCHS,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    rendered = str(value)
    if len(rendered) != 64 or any(c not in "0123456789abcdef" for c in rendered):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return rendered


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _verified_file(raw_path: Any, raw_hash: Any, label: str) -> Path:
    path = Path(str(raw_path))
    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is absent: {path}")
    expected = _require_sha256(raw_hash, f"{label} SHA-256")
    if _sha256_file(path) != expected:
        raise ValueError(f"{label} hash mismatch: {path}")
    return path


def _safe_component(value: Any, field: str) -> str:
    rendered = str(value)
    if not rendered or rendered in {".", ".."} or "/" in rendered or "\\" in rendered:
        raise ValueError(f"{field} must be a non-empty path component")
    return rendered


@dataclass(frozen=True, order=True)
class AttemptCell:
    protocol_id: str
    dataset_key: str
    dataset_slug: str
    dataset_id: int
    arm_id: str
    outer_fold: int
    training_seed: int

    @classmethod
    def from_mapping(cls, value: Any, field: str) -> "AttemptCell":
        if not isinstance(value, Mapping):
            raise ValueError(f"{field} must be an object")
        required = {
            "protocol_id", "dataset_key", "dataset_slug", "dataset_id",
            "arm_id", "outer_fold", "training_seed",
        }
        if set(value) != required:
            raise ValueError(f"{field} fields must be exactly {sorted(required)}")
        protocol_id = _safe_component(value["protocol_id"], f"{field}.protocol_id")
        dataset_key = str(value["dataset_key"])
        dataset_slug = _safe_component(value["dataset_slug"], f"{field}.dataset_slug")
        dataset_id = _strict_int(value["dataset_id"], f"{field}.dataset_id")
        arm_id = _safe_component(value["arm_id"], f"{field}.arm_id")
        outer_fold = _strict_int(value["outer_fold"], f"{field}.outer_fold")
        training_seed = _strict_int(value["training_seed"], f"{field}.training_seed")
        if protocol_id != PROTOCOL_ID:
            raise ValueError(f"{field}.protocol_id is outside the frozen protocol")
        if arm_id not in REGISTERED_ARMS:
            raise ValueError(f"{field}.arm_id is not registered by the frozen protocol")
        if dataset_key not in DATASET_IDENTIFIERS:
            raise ValueError(f"{field}.dataset_key is unsupported")
        if DATASET_IDENTIFIERS[dataset_key] != (dataset_slug, dataset_id):
            raise ValueError(f"{field} dataset key/slug/id binding drift")
        if outer_fold not in range(DATASET_OUTER_FOLDS[dataset_key]):
            raise ValueError(f"{field}.outer_fold is outside the frozen dataset grid")
        if training_seed not in TRAINING_SEEDS:
            raise ValueError(f"{field}.training_seed is outside the frozen seed grid")
        return cls(
            protocol_id, dataset_key, dataset_slug, dataset_id, arm_id,
            outer_fold, training_seed,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "dataset_key": self.dataset_key,
            "dataset_slug": self.dataset_slug,
            "dataset_id": self.dataset_id,
            "arm_id": self.arm_id,
            "outer_fold": self.outer_fold,
            "training_seed": self.training_seed,
        }

    def directory_suffix(self, attempt_id: int) -> str:
        return (
            f"/{self.protocol_id}/{self.dataset_slug}/{self.arm_id}/"
            f"fold_{self.outer_fold}/seed_{self.training_seed}/attempt_{attempt_id}"
        )


def _validate_budget(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != set(BUDGET_CONTRACT):
        raise ValueError("Attempt ledger budget fields have drifted")
    for field, expected in BUDGET_CONTRACT.items():
        observed = _strict_int(value[field], f"budget.{field}")
        if observed != expected:
            raise ValueError(f"budget.{field} must equal {expected}")


def _load_attempt(
    reference: Mapping[str, Any], expected_cells: set[AttemptCell]
) -> dict[str, Any]:
    required_reference = {
        "cell", "attempt_id", "status", "manifest_path", "manifest_sha256"
    }
    if set(reference) != required_reference:
        raise ValueError("Attempt reference fields have drifted")
    cell = AttemptCell.from_mapping(reference["cell"], "attempt.cell")
    if cell not in expected_cells:
        raise ValueError(f"Attempt references an unexpected cell: {cell}")
    attempt_id = _strict_int(reference["attempt_id"], "attempt.attempt_id")
    if attempt_id not in {0, 1}:
        raise ValueError("attempt_id must be integer 0 or 1")
    status = str(reference["status"])
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"Attempt has non-terminal or unknown status: {status!r}")
    manifest_path = _verified_file(
        reference["manifest_path"], reference["manifest_sha256"],
        "Attempt manifest",
    )
    manifest = _json_object(manifest_path, "Attempt manifest")
    required_manifest = {
        "schema_version", "cell", "attempt_id", "status", "attempted_epochs",
        "run_dir", "terminal_status_path", "terminal_status_sha256",
    }
    missing = required_manifest - set(manifest)
    if missing:
        raise ValueError(f"Attempt manifest lacks fields: {sorted(missing)}")
    if manifest.get("schema_version") != 1:
        raise ValueError("Attempt manifest schema_version must be 1")
    manifest_cell = AttemptCell.from_mapping(manifest["cell"], "manifest.cell")
    if manifest_cell != cell:
        raise ValueError("Attempt manifest cell binding drift")
    manifest_attempt = _strict_int(manifest["attempt_id"], "manifest.attempt_id")
    manifest_status = str(manifest["status"])
    if (manifest_attempt, manifest_status) != (attempt_id, status):
        raise ValueError("Attempt status or attempt_id drift between ledger and manifest")
    attempted_epochs = _strict_int(
        manifest["attempted_epochs"], "manifest.attempted_epochs"
    )
    if attempted_epochs not in range(MAX_EPOCHS_PER_ATTEMPT + 1):
        raise ValueError("attempted_epochs must be between 0 and 50")

    run_dir = Path(str(manifest["run_dir"]))
    if not run_dir.is_absolute():
        raise ValueError("Attempt run_dir must be absolute")
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Attempt run directory is absent: {run_dir}")
    if not ("/" + run_dir.as_posix().strip("/")).endswith(
        cell.directory_suffix(attempt_id)
    ):
        raise ValueError("Attempt run directory identity drift")
    if manifest_path != (run_dir / "attempt.manifest.json").resolve():
        raise ValueError("Attempt manifest is outside its canonical run directory")

    terminal_path = _verified_file(
        manifest["terminal_status_path"], manifest["terminal_status_sha256"],
        "Terminal-status manifest",
    )
    if terminal_path != (run_dir / "artifacts" / "terminal_status.manifest.json").resolve():
        raise ValueError("Terminal-status manifest path drift")
    terminal = _json_object(terminal_path, "Terminal-status manifest")
    terminal_cell = AttemptCell.from_mapping(terminal.get("cell"), "terminal.cell")
    terminal_attempt = _strict_int(terminal.get("attempt_id"), "terminal.attempt_id")
    terminal_epochs = _strict_int(
        terminal.get("attempted_epochs"), "terminal.attempted_epochs"
    )
    if (
        terminal.get("schema_version") != 1
        or terminal_cell != cell
        or terminal_attempt != attempt_id
        or str(terminal.get("status")) != status
        or terminal_epochs != attempted_epochs
    ):
        raise ValueError("Terminal status drift from the bound attempt manifest")
    failure = terminal.get("failure")
    if status == "succeeded":
        if failure not in (None, {}):
            raise ValueError("Succeeded attempt cannot carry failure classification")
    else:
        expected_class = "infrastructure" if status == "infrastructure_failed" else "algorithm"
        if (
            not isinstance(failure, Mapping)
            or failure.get("class") != expected_class
            or not str(failure.get("reason", "")).strip()
        ):
            raise ValueError("Failed attempt lacks a matching failure classification")

    normalized: dict[str, Any] = {
        "cell": cell.as_dict(),
        "attempt_id": attempt_id,
        "status": status,
        "attempted_epochs": attempted_epochs,
        "run_dir": str(run_dir),
        "attempt_manifest_path": str(manifest_path),
        "attempt_manifest_sha256": _sha256_file(manifest_path),
        "terminal_status_path": str(terminal_path),
        "terminal_status_sha256": _sha256_file(terminal_path),
        "failure": None if status == "succeeded" else dict(failure),
    }
    canonical_prediction = run_dir / "artifacts" / "predictions.npz"
    canonical_prediction_manifest = canonical_prediction.with_suffix(".manifest.json")
    if status == "succeeded":
        for field in (
            "prediction_path", "prediction_sha256", "prediction_manifest_path",
            "prediction_manifest_sha256",
        ):
            if field not in manifest:
                raise ValueError(f"Succeeded attempt manifest lacks {field}")
        prediction_path = _verified_file(
            manifest["prediction_path"], manifest["prediction_sha256"],
            "Prediction artifact",
        )
        prediction_manifest_path = _verified_file(
            manifest["prediction_manifest_path"],
            manifest["prediction_manifest_sha256"],
            "Prediction manifest",
        )
        if prediction_path != canonical_prediction.resolve():
            raise ValueError("Prediction artifact path drift")
        if prediction_manifest_path != canonical_prediction_manifest.resolve():
            raise ValueError("Prediction manifest path drift")
        artifact = load_prediction_artifact(prediction_path)
        artifact_identity = (
            artifact.protocol_id, artifact.dataset_key, artifact.dataset_slug,
            artifact.dataset_id, artifact.arm_id, artifact.outer_fold,
            artifact.training_seed,
        )
        cell_identity = (
            cell.protocol_id, cell.dataset_key, cell.dataset_slug,
            cell.dataset_id, cell.arm_id, cell.outer_fold, cell.training_seed,
        )
        if artifact_identity != cell_identity or artifact.attempt_id != attempt_id:
            raise ValueError("Prediction provenance does not match its attempt cell")
        normalized.update(
            {
                "prediction_path": str(prediction_path),
                "prediction_sha256": _sha256_file(prediction_path),
                "prediction_manifest_path": str(prediction_manifest_path),
                "prediction_manifest_sha256": _sha256_file(prediction_manifest_path),
            }
        )
    elif (
        any(field.startswith("prediction_") for field in manifest)
        or canonical_prediction.exists()
        or canonical_prediction_manifest.exists()
    ):
        raise ValueError("Failed attempt conflicts with prediction artifacts")
    return normalized


def collect_attempt_ledger(
    ledger_path: str | Path, expected_ledger_sha256: str
) -> dict[str, Any]:
    """Validate an immutable ledger and select one successful attempt per cell."""

    source_path = Path(ledger_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Attempt ledger is absent: {source_path}")
    expected_hash = _require_sha256(expected_ledger_sha256, "Attempt ledger SHA-256")
    if _sha256_file(source_path) != expected_hash:
        raise ValueError("Attempt ledger hash mismatch")
    ledger = _json_object(source_path, "Attempt ledger")
    required = {
        "schema_version", "kind", "protocol_id", "attempt_root", "budget",
        "expected_cells", "attempts",
    }
    if set(ledger) != required or ledger.get("schema_version") != 1:
        raise ValueError("Attempt ledger schema fields have drifted")
    if ledger.get("kind") != "p01_attempt_ledger":
        raise ValueError("Attempt ledger kind is invalid")
    protocol_id = _safe_component(ledger["protocol_id"], "ledger.protocol_id")
    if protocol_id != PROTOCOL_ID:
        raise ValueError("Attempt ledger protocol_id is outside the frozen protocol")
    attempt_root = Path(str(ledger["attempt_root"]))
    if not attempt_root.is_absolute():
        raise ValueError("Attempt ledger attempt_root must be absolute")
    if attempt_root.is_symlink() or not attempt_root.is_dir():
        raise ValueError("Attempt ledger attempt_root must be a real directory")
    attempt_root = attempt_root.resolve()
    _validate_budget(ledger["budget"])
    if not isinstance(ledger["expected_cells"], list) or not ledger["expected_cells"]:
        raise ValueError("Attempt ledger expected_cells must be a non-empty list")
    cells = [
        AttemptCell.from_mapping(value, f"expected_cells[{index}]")
        for index, value in enumerate(ledger["expected_cells"])
    ]
    if any(cell.protocol_id != protocol_id for cell in cells):
        raise ValueError("Expected cell protocol_id drifts from ledger protocol_id")
    if len(set(cells)) != len(cells):
        raise ValueError("Attempt ledger contains duplicate expected cells")
    if len(cells) > MAX_PRIMARY_CELLS:
        raise ValueError(f"Attempt ledger exceeds {MAX_PRIMARY_CELLS} primary cells")
    if not isinstance(ledger["attempts"], list):
        raise ValueError("Attempt ledger attempts must be a list")
    records = [
        _load_attempt(reference, set(cells))
        for reference in ledger["attempts"]
        if isinstance(reference, Mapping)
    ]
    if len(records) != len(ledger["attempts"]):
        raise ValueError("Every attempt ledger entry must be an object")
    if len(records) > MAX_TOTAL_ATTEMPTS:
        raise ValueError("Attempt ledger exceeds the 267-attempt budget")
    retry_count = sum(record["attempt_id"] == 1 for record in records)
    if retry_count > MAX_RETRY_ATTEMPTS:
        raise ValueError("Attempt ledger exceeds the 24-attempt retry reserve")
    attempted_epochs = sum(record["attempted_epochs"] for record in records)
    if attempted_epochs > MAX_ATTEMPTED_EPOCHS:
        raise ValueError("Attempt ledger exceeds the 13,350 attempted-epoch budget")

    listed_run_dirs = {Path(record["run_dir"]).resolve() for record in records}
    if any(not path.is_relative_to(attempt_root) for path in listed_run_dirs):
        raise ValueError("Attempt run directory is outside the ledger attempt_root")
    discovered_run_dirs: set[Path] = set()
    for candidate in attempt_root.rglob("attempt_*"):
        if not candidate.is_dir():
            continue
        if candidate.is_symlink():
            raise ValueError("Symlinked attempt directories are forbidden")
        discovered_run_dirs.add(candidate.resolve())
    if discovered_run_dirs != listed_run_dirs:
        raise ValueError("Attempt directory inventory drifts from the frozen ledger")

    by_key: dict[tuple[AttemptCell, int], dict[str, Any]] = {}
    by_cell: dict[AttemptCell, list[dict[str, Any]]] = {cell: [] for cell in cells}
    for record in records:
        cell = AttemptCell.from_mapping(record["cell"], "record.cell")
        key = (cell, record["attempt_id"])
        if key in by_key:
            raise ValueError("Attempt ledger contains duplicate cell/attempt_id records")
        by_key[key] = record
        by_cell[cell].append(record)

    selected: list[dict[str, Any]] = []
    for cell in sorted(cells):
        attempts = sorted(by_cell[cell], key=lambda item: item["attempt_id"])
        attempt_zero = next(
            (item for item in attempts if item["attempt_id"] == 0), None
        )
        if attempt_zero is None:
            raise ValueError(f"Cell is missing mandatory attempt 0: {cell}")
        successes = [item for item in attempts if item["status"] == "succeeded"]
        if len(successes) > 1:
            raise ValueError(f"Cell has duplicate terminal-valid attempts: {cell}")
        attempt_one = next(
            (item for item in attempts if item["attempt_id"] == 1), None
        )
        if attempt_one is not None and attempt_zero["status"] != "infrastructure_failed":
            raise ValueError(
                "Attempt 1 is permitted only after attempt 0 infrastructure_failed"
            )
        if len(successes) != 1:
            raise ValueError(f"Cell lacks exactly one terminal-valid attempt: {cell}")
        selected.append(successes[0])

    ordered_records = sorted(
        records,
        key=lambda item: (
            AttemptCell.from_mapping(item["cell"], "record.cell"),
            item["attempt_id"],
        ),
    )
    failures = [item for item in ordered_records if item["status"] != "succeeded"]
    selected_entries = [
        {
            "cell": item["cell"],
            "attempt_id": item["attempt_id"],
            "attempt_manifest_path": item["attempt_manifest_path"],
            "attempt_manifest_sha256": item["attempt_manifest_sha256"],
            "prediction_path": item["prediction_path"],
            "prediction_sha256": item["prediction_sha256"],
            "prediction_manifest_path": item["prediction_manifest_path"],
            "prediction_manifest_sha256": item["prediction_manifest_sha256"],
        }
        for item in selected
    ]
    return {
        "schema_version": 1,
        "kind": "p01_scorer_attempt_collection",
        "source_ledger_path": str(source_path),
        "source_ledger_sha256": expected_hash,
        "attempt_root": str(attempt_root),
        "budget": {
            **BUDGET_CONTRACT,
            "primary_cells": len(cells),
            "total_attempts": len(records),
            "retry_attempts": retry_count,
            "attempted_epochs": attempted_epochs,
        },
        "selected_attempts": selected_entries,
        "scorer_predictions": [item["prediction_path"] for item in selected_entries],
        "failed_attempts": failures,
        "all_attempts": ordered_records,
    }


def write_collection(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Atomically write a collector result once and return its SHA-256."""

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite attempt collection: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    try:
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_file(target)
