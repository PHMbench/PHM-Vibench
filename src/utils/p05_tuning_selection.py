"""Fail-closed, create-only learning-rate selection for P05 tuning runs.

The selector consumes exactly the sixteen validation-only candidate manifests
registered by G040.  It never opens a test artifact and it deliberately stops
at an unadjudicated selection record: no paper claim or sign-test decision is
made here.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


SCHEMA_NAME = "p05.tuning_selection"
SCHEMA_VERSION = 1
CANDIDATE_SCHEMA_NAME = "p05.tuning_validation_candidate"
CANDIDATE_SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
PROTOCOL_BUNDLE_SHA256 = (
    "8d01361c39a778d437ce235ad1e8d3877313f128d6593fbb74812a4b237a1654"
)

ARMS = ("P05-M", "P05-B0", "P05-B1", "P05-B3")
DATASETS = (("CWRU", 1), ("XJTU", 2))
LEARNING_RATES = (Decimal("0.001"), Decimal("0.0003"))
_LEARNING_RATE_TOKENS = {
    Decimal("0.001"): "LR1E3",
    Decimal("0.0003"): "LR3E4",
}
TUNING_SEED = 20260801
TIE_TOLERANCE = Decimal("0.0001")

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_JOB_ID_PATTERN = re.compile(r"^P05-TUNE-[A-Z0-9-]+$")
_PROVENANCE_KEYS = frozenset(
    {
        "source_metadata_sha256",
        "derived_metadata_sha256",
        "signal_cache_manifest_sha256",
        "split_manifest_sha256",
        "normalization_sha256",
        "train_weight_plan_sha256",
        "validation_weight_plan_sha256",
    }
)
_CANDIDATE_KEYS = frozenset(
    {
        "schema_name",
        "schema_version",
        "paper_id",
        "protocol_bundle_sha256",
        "source_matrix_sha256",
        "job",
        "execution",
        "validation",
        "artifacts",
        "provenance",
        "content",
    }
)


@dataclass(frozen=True)
class P05TuningSelectionResult:
    """Paths and hashes for one immutable selection package."""

    package_dir: Path
    manifest_path: Path
    semantic_sha256: str
    manifest_sha256: str
    status: str


@dataclass(frozen=True)
class _Candidate:
    source_path: Path
    source_manifest_sha256: str
    source_semantic_sha256: str
    source_matrix_sha256: str
    job_id: str
    arm_id: str
    dataset: str
    dataset_id: int
    learning_rate: Decimal
    checkpoint_epoch: int
    epochs_completed: int
    val_loss: Decimal
    val_f1_macro: Decimal
    config_sha256: str
    code_sha256: str
    run_contract_sha256: str
    checkpoint_sha256: str
    provenance: dict[str, str]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_hash(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, *, description: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a regular non-symlink file: {path}")
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {description}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain one JSON object: {path}")
    return value, payload


def _verify_semantic_manifest(
    path: Path,
    *,
    description: str,
    expected_schema_name: str,
    expected_semantic_sha256: str | None = None,
) -> tuple[dict[str, Any], str, str]:
    manifest, payload = _load_json(path, description=description)
    if manifest.get("schema_name") != expected_schema_name:
        raise ValueError(
            f"{description} schema_name must be {expected_schema_name!r}: {path}"
        )
    content = manifest.get("content")
    if not isinstance(content, dict) or set(content) != {"semantic_sha256"}:
        raise ValueError(f"{description} content hash is invalid: {path}")
    recorded = _required_hash(
        content["semantic_sha256"],
        name=f"{description}.content.semantic_sha256",
    )
    semantic = {key: value for key, value in manifest.items() if key != "content"}
    actual = _sha256_bytes(_canonical_json_bytes(semantic))
    if recorded != actual:
        raise ValueError(f"{description} semantic hash mismatch: {path}")
    if expected_semantic_sha256 is not None:
        expected = _required_hash(
            expected_semantic_sha256,
            name=f"expected {description} semantic SHA-256",
        )
        if actual != expected:
            raise ValueError(f"{description} does not match its registered hash: {path}")
    return manifest, actual, _sha256_bytes(payload)


def _exact_keys(value: Any, expected: frozenset[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = dict(value)
    if set(result) != expected:
        missing = sorted(expected - set(result))
        unexpected = sorted(set(result) - expected, key=str)
        raise ValueError(
            f"{name} fields do not match the frozen contract: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return result


def _exact(value: Any, expected: Any, *, name: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{name} must be exactly {expected!r}")


def _decimal_metric(
    value: Any,
    *,
    name: str,
    minimum: Decimal,
    maximum: Decimal | None = None,
) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a JSON number")
    try:
        converted = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not converted.is_finite() or converted < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    if maximum is not None and converted > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return converted


def _artifact_path(candidate_path: Path, raw_path: Any, *, name: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        raise ValueError(f"{name}.path must be non-empty path text")
    path = Path(raw_path)
    if not path.is_absolute():
        path = candidate_path.parent / path
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{name}.path is unavailable: {path}") from exc
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"{name}.path must resolve to a regular non-symlink file")
    return resolved


def _direct_artifact(
    candidate_path: Path,
    value: Any,
    *,
    name: str,
) -> tuple[Path, str]:
    artifact = _exact_keys(value, frozenset({"path", "sha256"}), name=name)
    path = _artifact_path(candidate_path, artifact["path"], name=name)
    registered = _required_hash(artifact["sha256"], name=f"{name}.sha256")
    if _sha256_file(path) != registered:
        raise ValueError(f"{name} file hash mismatch: {path}")
    return path, registered


def _semantic_artifact(
    candidate_path: Path,
    value: Any,
    *,
    name: str,
    schema_name: str,
) -> tuple[Path, str, dict[str, Any]]:
    artifact = _exact_keys(
        value,
        frozenset({"path", "semantic_sha256"}),
        name=name,
    )
    path = _artifact_path(candidate_path, artifact["path"], name=name)
    expected = _required_hash(
        artifact["semantic_sha256"],
        name=f"{name}.semantic_sha256",
    )
    manifest, semantic, _ = _verify_semantic_manifest(
        path,
        description=name,
        expected_schema_name=schema_name,
        expected_semantic_sha256=expected,
    )
    return path, semantic, manifest


def _validate_job(value: Any) -> tuple[str, str, str, int, Decimal]:
    job = _exact_keys(
        value,
        frozenset(
            {
                "job_id",
                "phase",
                "arm_id",
                "dataset",
                "dataset_id",
                "seed",
                "learning_rate",
            }
        ),
        name="candidate.job",
    )
    job_id = job["job_id"]
    if not isinstance(job_id, str) or _JOB_ID_PATTERN.fullmatch(job_id) is None:
        raise ValueError("candidate.job.job_id must be a safe P05-TUNE-* identifier")
    _exact(job["phase"], "tuning", name="candidate.job.phase")
    arm_id = job["arm_id"]
    if arm_id not in ARMS:
        raise ValueError(f"candidate.job.arm_id must be one of {list(ARMS)}")
    dataset = job["dataset"]
    dataset_map = dict(DATASETS)
    if dataset not in dataset_map:
        raise ValueError(f"candidate.job.dataset must be one of {list(dataset_map)}")
    dataset_id = job["dataset_id"]
    _exact(dataset_id, dataset_map[dataset], name="candidate.job.dataset_id")
    _exact(job["seed"], TUNING_SEED, name="candidate.job.seed")
    learning_rate = _decimal_metric(
        job["learning_rate"],
        name="candidate.job.learning_rate",
        minimum=Decimal("0"),
    )
    if learning_rate not in LEARNING_RATES:
        raise ValueError(
            "candidate.job.learning_rate must be exactly 1e-3 or 3e-4"
        )
    expected_job_id = (
        f"P05-TUNE-{arm_id[4:]}-{dataset}-{_LEARNING_RATE_TOKENS[learning_rate]}"
    )
    if job_id != expected_job_id:
        raise ValueError(
            "candidate.job.job_id conflicts with its arm/dataset/learning-rate cell: "
            f"expected {expected_job_id!r}"
        )
    return job_id, arm_id, dataset, dataset_id, learning_rate


def _validate_execution(value: Any) -> tuple[int, int]:
    execution = _exact_keys(
        value,
        frozenset(
            {
                "status",
                "stage",
                "evidence_eligible",
                "claim_decision",
                "data_roles_constructed",
                "test_access_count",
                "max_epochs",
                "patience",
                "epochs_completed",
                "checkpoint_monitor",
                "checkpoint_mode",
                "save_top_k",
                "selected_checkpoint_count",
            }
        ),
        name="candidate.execution",
    )
    expected = {
        "status": "completed",
        "stage": "fit_validate_only",
        "evidence_eligible": False,
        "claim_decision": "not_performed",
        "data_roles_constructed": ["train", "validation"],
        "test_access_count": 0,
        "max_epochs": 60,
        "patience": 10,
        "checkpoint_monitor": "val_loss",
        "checkpoint_mode": "min",
        "save_top_k": 1,
        "selected_checkpoint_count": 1,
    }
    for name, frozen in expected.items():
        _exact(execution[name], frozen, name=f"candidate.execution.{name}")
    epochs_completed = execution["epochs_completed"]
    if type(epochs_completed) is not int or not 1 <= epochs_completed <= 60:
        raise ValueError("candidate.execution.epochs_completed must be in [1, 60]")
    return epochs_completed, int(execution["test_access_count"])


def _validate_validation(
    value: Any,
    *,
    epochs_completed: int,
) -> tuple[int, Decimal, Decimal]:
    validation = _exact_keys(
        value,
        frozenset(
            {
                "partition",
                "checkpoint_epoch",
                "val_loss",
                "val_f1_macro",
                "loss_definition",
                "macro_f1_construction",
                "weighting",
                "zero_division",
            }
        ),
        name="candidate.validation",
    )
    exact = {
        "partition": "validation",
        "loss_definition": "group_equal_weighted_cross_entropy",
        "macro_f1_construction": "one_epoch_level_weighted_confusion_matrix",
        "weighting": "equal_group_then_equal_window",
        "zero_division": 0,
    }
    for name, frozen in exact.items():
        _exact(validation[name], frozen, name=f"candidate.validation.{name}")
    checkpoint_epoch = validation["checkpoint_epoch"]
    if type(checkpoint_epoch) is not int or not 0 <= checkpoint_epoch < epochs_completed:
        raise ValueError(
            "candidate.validation.checkpoint_epoch must identify a completed epoch"
        )
    val_loss = _decimal_metric(
        validation["val_loss"],
        name="candidate.validation.val_loss",
        minimum=Decimal("0"),
    )
    val_f1 = _decimal_metric(
        validation["val_f1_macro"],
        name="candidate.validation.val_f1_macro",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    return checkpoint_epoch, val_loss, val_f1


def _validate_provenance(value: Any) -> dict[str, str]:
    provenance = _exact_keys(value, _PROVENANCE_KEYS, name="candidate.provenance")
    return {
        key: _required_hash(provenance[key], name=f"candidate.provenance.{key}")
        for key in sorted(_PROVENANCE_KEYS)
    }


def _cross_check_run_contract(
    manifest: Mapping[str, Any],
    *,
    dataset_id: int,
    config_sha256: str,
    code_sha256: str,
    checkpoint_sha256: str,
    provenance: Mapping[str, str],
) -> None:
    _exact(manifest.get("schema_version"), 1, name="run_contract.schema_version")
    _exact(manifest.get("paper_id"), "P05", name="run_contract.paper_id")
    _exact(manifest.get("dataset_id"), dataset_id, name="run_contract.dataset_id")
    hashes = manifest.get("provenance")
    if not isinstance(hashes, Mapping):
        raise ValueError("run_contract.provenance must be a mapping")
    expected_hashes = {
        "config_sha256": config_sha256,
        "code_sha256": code_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    for key, expected in expected_hashes.items():
        observed = _required_hash(hashes.get(key), name=f"run_contract.provenance.{key}")
        if observed != expected:
            raise ValueError(f"run contract {key} conflicts with the tuning candidate")
    normalization = manifest.get("normalization_plan")
    if not isinstance(normalization, Mapping):
        raise ValueError("run_contract.normalization_plan must be a mapping")
    if _required_hash(
        normalization.get("sha256"),
        name="run_contract.normalization_plan.sha256",
    ) != provenance["normalization_sha256"]:
        raise ValueError("run contract normalization hash conflicts with provenance")
    weight_plans = manifest.get("weight_plans")
    if not isinstance(weight_plans, Mapping):
        raise ValueError("run_contract.weight_plans must be a mapping")
    for role, key in (
        ("train", "train_weight_plan_sha256"),
        ("validation", "validation_weight_plan_sha256"),
    ):
        plan = weight_plans.get(role)
        if not isinstance(plan, Mapping):
            raise ValueError(f"run_contract.weight_plans.{role} must be a mapping")
        observed = _required_hash(
            plan.get("sha256"),
            name=f"run_contract.weight_plans.{role}.sha256",
        )
        if observed != provenance[key]:
            raise ValueError(f"run contract {role} weight hash conflicts with provenance")


def _load_candidate(path_input: str | Path) -> _Candidate:
    path = Path(os.path.abspath(os.fspath(path_input)))
    manifest, semantic_sha256, manifest_sha256 = _verify_semantic_manifest(
        path,
        description="P05 tuning validation candidate",
        expected_schema_name=CANDIDATE_SCHEMA_NAME,
    )
    _exact_keys(manifest, _CANDIDATE_KEYS, name="candidate manifest")
    _exact(manifest["schema_version"], CANDIDATE_SCHEMA_VERSION, name="candidate.schema_version")
    _exact(manifest["paper_id"], "P05", name="candidate.paper_id")
    protocol_hash = _required_hash(
        manifest["protocol_bundle_sha256"],
        name="candidate.protocol_bundle_sha256",
    )
    if protocol_hash != PROTOCOL_BUNDLE_SHA256:
        raise ValueError("candidate is not bound to the approved P05-G040 bundle")
    source_matrix_sha256 = _required_hash(
        manifest["source_matrix_sha256"],
        name="candidate.source_matrix_sha256",
    )

    job_id, arm_id, dataset, dataset_id, learning_rate = _validate_job(manifest["job"])
    epochs_completed, _ = _validate_execution(manifest["execution"])
    checkpoint_epoch, val_loss, val_f1 = _validate_validation(
        manifest["validation"],
        epochs_completed=epochs_completed,
    )
    provenance = _validate_provenance(manifest["provenance"])

    artifacts = _exact_keys(
        manifest["artifacts"],
        frozenset({"config_snapshot", "code_snapshot", "run_contract", "checkpoint"}),
        name="candidate.artifacts",
    )
    _, config_sha256 = _direct_artifact(
        path,
        artifacts["config_snapshot"],
        name="candidate.artifacts.config_snapshot",
    )
    _, checkpoint_sha256 = _direct_artifact(
        path,
        artifacts["checkpoint"],
        name="candidate.artifacts.checkpoint",
    )
    _, code_sha256, code_manifest = _semantic_artifact(
        path,
        artifacts["code_snapshot"],
        name="candidate.artifacts.code_snapshot",
        schema_name="p05.code_snapshot",
    )
    _exact(code_manifest.get("schema_version"), 1, name="code_snapshot.schema_version")
    _exact(code_manifest.get("paper_id"), "P05", name="code_snapshot.paper_id")
    _, run_contract_sha256, run_contract = _semantic_artifact(
        path,
        artifacts["run_contract"],
        name="candidate.artifacts.run_contract",
        schema_name="p05.run_artifact_bundle",
    )
    _cross_check_run_contract(
        run_contract,
        dataset_id=dataset_id,
        config_sha256=config_sha256,
        code_sha256=code_sha256,
        checkpoint_sha256=checkpoint_sha256,
        provenance=provenance,
    )
    return _Candidate(
        source_path=path,
        source_manifest_sha256=manifest_sha256,
        source_semantic_sha256=semantic_sha256,
        source_matrix_sha256=source_matrix_sha256,
        job_id=job_id,
        arm_id=arm_id,
        dataset=dataset,
        dataset_id=dataset_id,
        learning_rate=learning_rate,
        checkpoint_epoch=checkpoint_epoch,
        epochs_completed=epochs_completed,
        val_loss=val_loss,
        val_f1_macro=val_f1,
        config_sha256=config_sha256,
        code_sha256=code_sha256,
        run_contract_sha256=run_contract_sha256,
        checkpoint_sha256=checkpoint_sha256,
        provenance=provenance,
    )


def _validate_complete_grid(candidates: Sequence[_Candidate]) -> str:
    if len(candidates) != len(ARMS) * len(DATASETS) * len(LEARNING_RATES):
        raise ValueError("P05 tuning selection requires exactly 16 candidate manifests")
    job_ids = [candidate.job_id for candidate in candidates]
    if len(set(job_ids)) != len(job_ids):
        raise ValueError("P05 tuning candidate job IDs must be unique")
    keys = [
        (candidate.arm_id, candidate.dataset, candidate.learning_rate)
        for candidate in candidates
    ]
    expected = {
        (arm, dataset, learning_rate)
        for arm in ARMS
        for dataset, _ in DATASETS
        for learning_rate in LEARNING_RATES
    }
    if set(keys) != expected or len(set(keys)) != len(keys):
        missing = sorted(expected - set(keys), key=lambda item: tuple(map(str, item)))
        duplicates = sorted(
            {key for key in keys if keys.count(key) > 1},
            key=lambda item: tuple(map(str, item)),
        )
        unexpected = sorted(set(keys) - expected, key=lambda item: tuple(map(str, item)))
        raise ValueError(
            "P05 tuning candidate grid is incomplete or duplicated: "
            f"missing={missing}, duplicates={duplicates}, unexpected={unexpected}"
        )
    matrix_hashes = {candidate.source_matrix_sha256 for candidate in candidates}
    if len(matrix_hashes) != 1:
        raise ValueError("all tuning candidates must bind the same source matrix hash")
    code_hashes = {candidate.code_sha256 for candidate in candidates}
    if len(code_hashes) != 1:
        raise ValueError("all tuning candidates must use one identical code snapshot")
    global_provenance = (
        "source_metadata_sha256",
        "derived_metadata_sha256",
        "signal_cache_manifest_sha256",
    )
    for key in global_provenance:
        if len({candidate.provenance[key] for candidate in candidates}) != 1:
            raise ValueError(f"all tuning candidates must share provenance {key}")
    dataset_provenance = (
        "split_manifest_sha256",
        "normalization_sha256",
        "train_weight_plan_sha256",
        "validation_weight_plan_sha256",
    )
    for dataset, _ in DATASETS:
        subset = [candidate for candidate in candidates if candidate.dataset == dataset]
        for key in dataset_provenance:
            if len({candidate.provenance[key] for candidate in subset}) != 1:
                raise ValueError(
                    f"all {dataset} tuning candidates must share provenance {key}"
                )
    return next(iter(matrix_hashes))


def _candidate_sort_key(candidate: _Candidate) -> tuple[int, int, Decimal]:
    dataset_order = {name: index for index, (name, _) in enumerate(DATASETS)}
    arm_order = {name: index for index, name in enumerate(ARMS)}
    return (
        dataset_order[candidate.dataset],
        arm_order[candidate.arm_id],
        candidate.learning_rate,
    )


def _select(candidates: Sequence[_Candidate]) -> tuple[_Candidate, str]:
    if len(candidates) != 2 or {item.learning_rate for item in candidates} != set(
        LEARNING_RATES
    ):
        raise ValueError("each arm/dataset cell must contain the two registered rates")
    best_f1 = max(candidate.val_f1_macro for candidate in candidates)
    tied = [
        candidate
        for candidate in candidates
        if best_f1 - candidate.val_f1_macro <= TIE_TOLERANCE
    ]
    if len(tied) == 1:
        return tied[0], "validation_macro_f1_max"
    best_loss = min(candidate.val_loss for candidate in tied)
    loss_winners = [candidate for candidate in tied if candidate.val_loss == best_loss]
    if len(loss_winners) == 1:
        return loss_winners[0], "macro_f1_tie_within_1e-4_then_validation_loss_min"
    winner = min(loss_winners, key=lambda candidate: candidate.learning_rate)
    return winner, "macro_f1_tie_then_validation_loss_tie_then_lower_learning_rate"


def _float(value: Decimal) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("selection contains a non-finite numeric value")
    return converted


def _candidate_row(candidate: _Candidate) -> dict[str, Any]:
    return {
        "job_id": candidate.job_id,
        "arm_id": candidate.arm_id,
        "dataset": candidate.dataset,
        "dataset_id": candidate.dataset_id,
        "seed": TUNING_SEED,
        "learning_rate": _float(candidate.learning_rate),
        "epochs_completed": candidate.epochs_completed,
        "checkpoint_epoch": candidate.checkpoint_epoch,
        "val_loss": _float(candidate.val_loss),
        "val_f1_macro": _float(candidate.val_f1_macro),
        "config_sha256": candidate.config_sha256,
        "code_sha256": candidate.code_sha256,
        "run_contract_sha256": candidate.run_contract_sha256,
        "checkpoint_sha256": candidate.checkpoint_sha256,
        "provenance": candidate.provenance,
        "source_candidate_path": str(candidate.source_path),
        "source_candidate_semantic_sha256": candidate.source_semantic_sha256,
        "source_candidate_manifest_sha256": candidate.source_manifest_sha256,
    }


def _semantic_manifest(candidates: Sequence[_Candidate]) -> dict[str, Any]:
    source_matrix_sha256 = _validate_complete_grid(candidates)
    ordered = sorted(candidates, key=_candidate_sort_key)
    candidate_rows = [_candidate_row(candidate) for candidate in ordered]
    selections: list[dict[str, Any]] = []
    selection_index: dict[str, dict[str, Any]] = {}
    for dataset, dataset_id in DATASETS:
        for arm_id in ARMS:
            cell = [
                candidate
                for candidate in ordered
                if candidate.dataset == dataset and candidate.arm_id == arm_id
            ]
            winner, reason = _select(cell)
            row_index = len(selections)
            selection_id = f"P05-TUNING-SELECTION-{dataset}-{arm_id[4:]}"
            row = {
                "selection_id": selection_id,
                "arm_id": arm_id,
                "dataset": dataset,
                "dataset_id": dataset_id,
                "selected_learning_rate": _float(winner.learning_rate),
                "selected_job_id": winner.job_id,
                "selected_checkpoint_epoch": winner.checkpoint_epoch,
                "selected_val_f1_macro": _float(winner.val_f1_macro),
                "selected_val_loss": _float(winner.val_loss),
                "selection_reason": reason,
                "selected_config_sha256": winner.config_sha256,
                "selected_code_sha256": winner.code_sha256,
                "selected_run_contract_sha256": winner.run_contract_sha256,
                "selected_checkpoint_sha256": winner.checkpoint_sha256,
                "source_candidate_semantic_sha256": winner.source_semantic_sha256,
            }
            selections.append(row)
            selection_index[f"{dataset}/{arm_id}"] = {
                "row_index": row_index,
                "selection_id": selection_id,
                "selected_learning_rate": row["selected_learning_rate"],
                "selected_job_id": winner.job_id,
                "selected_checkpoint_sha256": winner.checkpoint_sha256,
                "selected_run_contract_sha256": winner.run_contract_sha256,
            }
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "paper_id": "P05",
        "phase": "tuning_selection",
        "status": "computed_unadjudicated",
        "claim_decision": "not_performed",
        "evidence_eligible": False,
        "test_access": "forbidden_and_not_performed",
        "protocol_bundle_sha256": PROTOCOL_BUNDLE_SHA256,
        "source_matrix_sha256": source_matrix_sha256,
        "protocol": {
            "candidate_count": 16,
            "arms": list(ARMS),
            "datasets": [name for name, _ in DATASETS],
            "learning_rates": [_float(value) for value in LEARNING_RATES],
            "seed": TUNING_SEED,
            "candidate_checkpoint": "minimum_validation_loss",
            "primary": "exact_epoch_validation_macro_f1_at_candidate_checkpoint_max",
            "tie_tolerance": _float(TIE_TOLERANCE),
            "first_tie_break": "validation_loss_min",
            "second_tie_break": "lower_learning_rate",
        },
        "candidates": candidate_rows,
        "selections": selections,
        "selection_index": selection_index,
    }


def _result(
    target: Path,
    manifest: Mapping[str, Any],
    *,
    status: str,
) -> P05TuningSelectionResult:
    manifest_path = target / MANIFEST_NAME
    return P05TuningSelectionResult(
        package_dir=target,
        manifest_path=manifest_path,
        semantic_sha256=str(manifest["content"]["semantic_sha256"]),
        manifest_sha256=_sha256_file(manifest_path),
        status=status,
    )


def _reuse_existing(
    target: Path,
    semantic_manifest: Mapping[str, Any],
) -> P05TuningSelectionResult:
    if target.is_symlink() or not target.is_dir():
        raise FileExistsError(f"invalid existing P05 tuning selection target: {target}")
    entries = {entry.name: entry for entry in target.iterdir()}
    if set(entries) != {MANIFEST_NAME}:
        raise FileExistsError(f"incomplete existing P05 tuning selection: {target}")
    manifest_path = entries[MANIFEST_NAME]
    try:
        manifest, _, _ = _verify_semantic_manifest(
            manifest_path,
            description="existing P05 tuning selection",
            expected_schema_name=SCHEMA_NAME,
        )
    except (TypeError, ValueError) as exc:
        raise FileExistsError(
            f"existing P05 tuning selection manifest is invalid: {target}"
        ) from exc
    if set(manifest) != set(semantic_manifest) | {"content"}:
        raise FileExistsError(f"existing P05 tuning selection schema conflicts: {target}")
    existing_semantic = {key: value for key, value in manifest.items() if key != "content"}
    if _canonical_json_bytes(existing_semantic) != _canonical_json_bytes(semantic_manifest):
        raise FileExistsError(f"existing P05 tuning selection content conflicts: {target}")
    return _result(target, manifest, status="reused")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic create-only export requires Linux renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), str(target))
    raise OSError(error_number, os.strerror(error_number), str(target))


def _write_new(
    target: Path,
    semantic_manifest: Mapping[str, Any],
) -> P05TuningSelectionResult:
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"P05 tuning selection parent must be a real directory: {parent}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(parent))
    )
    try:
        semantic_hash = _sha256_bytes(_canonical_json_bytes(semantic_manifest))
        manifest = {
            **semantic_manifest,
            "content": {"semantic_sha256": semantic_hash},
        }
        manifest_path = temporary / MANIFEST_NAME
        with manifest_path.open("xb") as handle:
            handle.write(_pretty_json_bytes(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary)
        try:
            _rename_directory_noreplace(temporary, target)
        except FileExistsError:
            return _reuse_existing(target, semantic_manifest)
        _fsync_directory(parent)
        return _result(target, manifest, status="created")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def export_p05_tuning_selection(
    package_dir: str | Path,
    *,
    candidate_manifest_paths: Sequence[str | Path],
) -> P05TuningSelectionResult:
    """Validate sixteen tuning candidates and create one selection manifest.

    Inputs are validation-only, completed tuning-candidate manifests.  The
    function verifies their self-hashes and referenced config, code snapshot,
    run contract, and checkpoint artifacts before applying the frozen selector.
    """

    if isinstance(candidate_manifest_paths, (str, bytes, Path)):
        raise TypeError("candidate_manifest_paths must be a sequence of paths")
    candidates = [_load_candidate(path) for path in candidate_manifest_paths]
    semantic_manifest = _semantic_manifest(candidates)
    target = Path(os.path.abspath(os.fspath(package_dir)))
    if target.is_symlink():
        raise FileExistsError(f"refusing P05 tuning selection through symlink: {target}")
    if target.exists():
        return _reuse_existing(target, semantic_manifest)
    return _write_new(target, semantic_manifest)


__all__ = ["P05TuningSelectionResult", "export_p05_tuning_selection"]
