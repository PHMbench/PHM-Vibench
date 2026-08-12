"""Fail-closed collection gate for the frozen P05 decisive artifacts.

This module deliberately stops before statistical adjudication.  It proves
that a supplied collection contains only immutable, completed, mutually
consistent decisive artifacts.  A complete collection is labelled
``computed_unadjudicated``; it never emits a claim decision or a p-value.

The frozen protocol has two different counts which must not be conflated:

* 60 central GPU jobs is the *budget* count (4 pilot + 16 tuning + 40
  decisive neural jobs).
* 52 is the decisive execution-artifact count collected here (40 neural,
  10 P05-B2, and two dataset-only deterministic P05-B4 fits).

P05-B4 is never repeated or relabelled on the five neural model-seed axis.
Its attempt seed is the frozen FCM initialisation seed 20260801, while its
normalised ``model_seed`` remains ``None``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_NAME = "p05.decisive_collection_gate"
SCHEMA_VERSION = 1

NEURAL_ARMS = ("P05-M", "P05-B0", "P05-B1", "P05-B3")
DATASETS = (("CWRU", 1), ("XJTU", 2))
DECISIVE_SEEDS = (42, 123, 456, 789, 1024)
B4_FCM_INITIALISATION_SEED = 20260801

CENTRAL_GPU_BUDGET_JOB_COUNT = 60
DECISIVE_NEURAL_ARTIFACT_COUNT = 40
DECISIVE_B2_ARTIFACT_COUNT = 10
DECISIVE_B4_ARTIFACT_COUNT = 2
DECISIVE_EXECUTION_ARTIFACT_COUNT = 52

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_DIR = REPO_ROOT / "configs" / "experiments" / "p05" / "protocol"
NEURAL_MATRIX_PATH = PROTOCOL_DIR / "neural_decisive_matrix_p05_v1.yaml"
CPU_MATRIX_PATH = PROTOCOL_DIR / "cpu_arm_matrix_p05_v1.yaml"

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_DESCRIPTOR_KEYS = frozenset(
    {
        "job_id",
        "attempt_package_dir",
        "materialized_manifest_path",
        "run_manifest_path",
        "result_path",
        "evidence_manifest_path",
    }
)
_ATTEMPT_PROVENANCE_FIELDS = frozenset(
    {
        "source_metadata_sha256",
        "derived_metadata_sha256",
        "signal_cache_manifest_sha256",
        "split_manifest_sha256",
        "config_snapshot_sha256",
        "code_snapshot_sha256",
        "normalization_sha256",
        "train_weight_plan_sha256",
        "validation_weight_plan_sha256",
    }
)
_LEAKAGE_KEYS = frozenset(
    {
        "checkpoint_selection_split",
        "coefficient_fit_split",
        "feature_selection_split",
        "fit_split",
        "hyperparameter_selection_split",
        "normalization_fit_split",
        "selection_split",
        "threshold_fit_split",
        "threshold_selection_split",
        "tuning_split",
    }
)


@dataclass(frozen=True)
class _ExpectedJob:
    job_id: str
    arm_id: str
    dataset: str
    dataset_id: int
    model_seed: int | None
    attempt_seed: int
    phase: str
    kind: str


@dataclass(frozen=True)
class _VerifiedManifest:
    path: Path
    value: dict[str, Any]
    semantic_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class P05DecisiveCollectionResult:
    """Normalised semantic manifest returned by the collection gate."""

    manifest: dict[str, Any]
    semantic_sha256: str
    status: str
    collected_job_count: int
    missing_job_ids: tuple[str, ...]


def _expected_jobs() -> tuple[_ExpectedJob, ...]:
    jobs: list[_ExpectedJob] = []
    for arm in NEURAL_ARMS:
        short_arm = arm[4:]
        for dataset, dataset_id in DATASETS:
            for seed in DECISIVE_SEEDS:
                jobs.append(
                    _ExpectedJob(
                        job_id=f"P05-DEC-{short_arm}-{dataset}-S{seed}",
                        arm_id=arm,
                        dataset=dataset,
                        dataset_id=dataset_id,
                        model_seed=seed,
                        attempt_seed=seed,
                        phase="decisive",
                        kind="neural",
                    )
                )
    for dataset, dataset_id in DATASETS:
        for seed in DECISIVE_SEEDS:
            jobs.append(
                _ExpectedJob(
                    job_id=f"P05-CPU-B2-{dataset}-S{seed}",
                    arm_id="P05-B2",
                    dataset=dataset,
                    dataset_id=dataset_id,
                    model_seed=seed,
                    attempt_seed=seed,
                    phase="cpu_baseline",
                    kind="b2",
                )
            )
    for dataset, dataset_id in DATASETS:
        jobs.append(
            _ExpectedJob(
                job_id=f"P05-CPU-B4-{dataset}",
                arm_id="P05-B4",
                dataset=dataset,
                dataset_id=dataset_id,
                model_seed=None,
                attempt_seed=B4_FCM_INITIALISATION_SEED,
                phase="cpu_baseline",
                kind="b4",
            )
        )
    if len(jobs) != DECISIVE_EXECUTION_ARTIFACT_COUNT:
        raise AssertionError("internal P05 decisive job registry has the wrong size")
    return tuple(jobs)


EXPECTED_JOBS = _expected_jobs()
EXPECTED_JOB_IDS = tuple(job.job_id for job in EXPECTED_JOBS)
_EXPECTED_BY_ID = {job.job_id: job for job in EXPECTED_JOBS}


def expected_p05_decisive_job_ids() -> tuple[str, ...]:
    """Return the exact ordered 52-artifact decisive execution registry."""

    return EXPECTED_JOB_IDS


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _required_real_file(value: str | Path, *, name: str) -> Path:
    path = Path(os.path.abspath(os.fspath(value)))
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a real file: {path}")
    return path


def _required_real_dir(value: str | Path, *, name: str) -> Path:
    path = Path(os.path.abspath(os.fspath(value)))
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{name} must be a real directory: {path}")
    return path


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: Any, *, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite numeric value is forbidden at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite(item, path=f"{path}[{index}]")


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {name}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object: {path}")
    _reject_nonfinite(value)
    return value


def _reject_test_leakage(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).strip().lower()
            if key in _LEAKAGE_KEYS:
                if isinstance(item, str) and "test" in item.strip().lower():
                    raise ValueError(f"test-fitted decision is forbidden at {path}.{raw_key}")
                if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                    if any(isinstance(part, str) and "test" in part.lower() for part in item):
                        raise ValueError(
                            f"test-fitted decision is forbidden at {path}.{raw_key}"
                        )
            _reject_test_leakage(item, path=f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_test_leakage(item, path=f"{path}[{index}]")


def _verify_npz(path: Path, *, name: str) -> None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ValueError(f"{name} contains duplicate array names")
            for array_name in archive.files:
                array = np.asarray(archive[array_name])
                if array.dtype.hasobject:
                    raise ValueError(f"{name} array {array_name!r} requires pickle")
                if array.dtype.kind in {"f", "c"} and not np.isfinite(array).all():
                    raise ValueError(f"{name} array {array_name!r} contains NaN or Inf")
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(name):
            raise
        raise ValueError(f"invalid {name}: {path}") from exc


def _verify_child_hash(path: Path, expected: Any, *, name: str) -> None:
    child = _required_real_file(path, name=name)
    recorded = _required_sha256(expected, name=f"{name} SHA-256")
    if _sha256_file(child) != recorded:
        raise ValueError(f"{name} SHA-256 mismatch: {child}")
    if child.suffix.lower() == ".npz":
        _verify_npz(child, name=name)


def _verify_standard_manifest(path: Path, *, name: str) -> _VerifiedManifest:
    value = _load_json(path, name=name)
    content = value.get("content")
    if not isinstance(content, dict) or "semantic_sha256" not in content:
        raise ValueError(f"{name} has no standard semantic content hash")
    recorded = _required_sha256(
        content["semantic_sha256"], name=f"{name} content.semantic_sha256"
    )
    semantic = {key: item for key, item in value.items() if key != "content"}
    if _sha256_bytes(_canonical_json_bytes(semantic)) != recorded:
        raise ValueError(f"{name} semantic SHA-256 mismatch")

    other_hashes = set(content) - {"semantic_sha256"}
    if other_hashes:
        schema_name = value.get("schema_name")
        if schema_name == "p05.window_predictions" and other_hashes == {"arrays_sha256"}:
            _verify_child_hash(
                path.parent / value.get("arrays_file", ""),
                content["arrays_sha256"],
                name=f"{name} arrays",
            )
        elif schema_name == "p05.c2_c3_evaluation_bundle" and other_hashes == {
            "arrays_sha256",
            "c3_sha256",
        }:
            outputs = value.get("outputs")
            if not isinstance(outputs, Mapping):
                raise ValueError(f"{name} outputs are missing")
            _verify_child_hash(
                path.parent / str(outputs.get("arrays_file", "")),
                content["arrays_sha256"],
                name=f"{name} arrays",
            )
            _verify_child_hash(
                path.parent / str(outputs.get("c3_file", "")),
                content["c3_sha256"],
                name=f"{name} C3 payload",
            )
        elif schema_name == "p05.b2_posthoc_fuzzy_surrogate" and other_hashes == {
            "checkpoint_sha256"
        }:
            checkpoint = value.get("checkpoint")
            if not isinstance(checkpoint, Mapping):
                raise ValueError(f"{name} checkpoint descriptor is missing")
            _verify_child_hash(
                path.parent / str(checkpoint.get("file", "")),
                content["checkpoint_sha256"],
                name=f"{name} checkpoint",
            )
        else:
            raise ValueError(
                f"{name} has unsupported content hashes for schema {schema_name!r}"
            )
    _reject_test_leakage(value)
    return _VerifiedManifest(path, value, recorded, _sha256_file(path))


def _verify_b4_manifest(path: Path, *, name: str) -> _VerifiedManifest:
    value = _load_json(path, name=name)
    if value.get("schema_name") != "p05.b4_classical_fuzzy":
        raise ValueError(f"{name} is not a P05-B4 artifact")
    recorded = _required_sha256(value.get("semantic_sha256"), name=f"{name} semantic_sha256")
    files = value.get("files")
    if not isinstance(files, dict) or set(files) != {"model.npz", "predictions.npz"}:
        raise ValueError(f"{name} file hash inventory is invalid")
    semantic = {
        key: item
        for key, item in value.items()
        if key not in {"semantic_sha256", "files"}
    }
    if _sha256_bytes(_canonical_json_bytes(semantic)) != recorded:
        raise ValueError(f"{name} semantic SHA-256 mismatch")
    for filename, expected_hash in files.items():
        _verify_child_hash(
            path.parent / filename,
            expected_hash,
            name=f"{name} {filename}",
        )
    _reject_test_leakage(value)
    return _VerifiedManifest(path, value, recorded, _sha256_file(path))


def _verify_manifest(path_value: str | Path, *, name: str) -> _VerifiedManifest:
    path = _required_real_file(path_value, name=name)
    value = _load_json(path, name=name)
    if value.get("schema_name") == "p05.b4_classical_fuzzy":
        return _verify_b4_manifest(path, name=name)
    return _verify_standard_manifest(path, name=name)


def _load_attempt(package_value: str | Path, expected: _ExpectedJob) -> tuple[
    _VerifiedManifest,
    _VerifiedManifest,
]:
    package = _required_real_dir(package_value, name="attempt package")
    entries = {entry.name: entry for entry in package.iterdir()}
    if set(entries) != {"start.json", "terminal.json", "invalidations"}:
        raise ValueError(f"attempt package is incomplete or has unexpected entries: {package}")
    invalidations = entries["invalidations"]
    if invalidations.is_symlink() or not invalidations.is_dir():
        raise ValueError("attempt invalidations entry must be a real directory")
    invalidation_entries = list(invalidations.iterdir())
    if invalidation_entries:
        raise ValueError(f"invalidated attempt is forbidden: {package}")

    start = _verify_standard_manifest(entries["start.json"], name="attempt start")
    terminal = _verify_standard_manifest(entries["terminal.json"], name="attempt terminal")
    start_value = start.value
    terminal_value = terminal.value
    if start_value.get("schema_name") != "p05.experiment_attempt" or start_value.get(
        "schema_version"
    ) != 1:
        raise ValueError("attempt start schema is unsupported")
    attempt = start_value.get("attempt")
    if not isinstance(attempt, dict):
        raise ValueError("attempt identity is missing")
    expected_identity = {
        "arm_id": expected.arm_id,
        "phase": expected.phase,
        "dataset_id": expected.dataset_id,
        "seed": expected.attempt_seed,
        "status": "running",
    }
    for key, required in expected_identity.items():
        if attempt.get(key) != required:
            raise ValueError(
                f"attempt {key} does not match {expected.job_id}: "
                f"observed={attempt.get(key)!r}, expected={required!r}"
            )
    attempt_id = attempt.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("attempt_id must be non-empty")

    provenance = start_value.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != _ATTEMPT_PROVENANCE_FIELDS:
        raise ValueError("attempt provenance field set is incomplete")
    normalised_provenance = {
        key: _required_sha256(value, name=f"attempt provenance {key}")
        for key, value in provenance.items()
    }
    if start_value.get("unavailable_reasons") != {}:
        raise ValueError("decisive collection forbids unavailable provenance")

    execution = start_value.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("attempt execution record is missing")
    _reject_test_leakage(execution)
    command = execution.get("command_argv")
    if not isinstance(command, list) or not command:
        raise ValueError("attempt command_argv is missing")
    if any(not isinstance(item, str) or not item for item in command):
        raise ValueError("attempt command_argv contains invalid values")

    if terminal_value.get("schema_name") != "p05.experiment_attempt" or terminal_value.get(
        "schema_version"
    ) != 1:
        raise ValueError("attempt terminal schema is unsupported")
    if terminal_value.get("attempt_id") != attempt_id:
        raise ValueError("attempt terminal refers to a different attempt_id")
    if terminal_value.get("start_semantic_sha256") != start.semantic_sha256:
        raise ValueError("attempt terminal start hash mismatch")
    terminal_state = terminal_value.get("terminal")
    if not isinstance(terminal_state, dict) or terminal_state.get("status") != "completed":
        raise ValueError("only a completed attempt may enter the decisive collection")
    if terminal_state.get("claim_decision") != "not_performed":
        raise ValueError("attempt performed a premature claim decision")
    if terminal_value.get("failure") is not None:
        raise ValueError("completed decisive attempt must not carry a failure")
    if terminal_value.get("missing_outputs") != {}:
        raise ValueError("completed decisive attempt has missing outputs")
    outputs = terminal_value.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("attempt terminal outputs are missing")
    for name, value in outputs.items():
        _required_sha256(value, name=f"attempt output {name}")

    # Keep the normalised hashes in memory for later cross-artifact checks.
    start_value["provenance"] = normalised_provenance
    return start, terminal


def _resolve_selection_path(materialized_path: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError("decisive tuning-selection path is invalid")
    path = Path(raw)
    if not path.is_absolute():
        path = materialized_path.parent / path
    return _required_real_file(path, name="tuning-selection manifest")


def _verify_materialized(
    path_value: str | Path,
    *,
    expected: _ExpectedJob,
    attempt_start: _VerifiedManifest,
    attempt_terminal: _VerifiedManifest,
) -> tuple[_VerifiedManifest, str | None]:
    manifest = _verify_manifest(path_value, name="materialized job manifest")
    value = manifest.value
    outputs = attempt_terminal.value["outputs"]
    if outputs.get("materialized_job") != manifest.semantic_sha256:
        raise ValueError("attempt materialized_job hash mismatch")
    for key, required in {
        "paper_id": "P05",
        "job_id": expected.job_id,
        "arm": expected.arm_id,
        "dataset": expected.dataset,
    }.items():
        if value.get(key) != required:
            raise ValueError(f"materialized {key} does not match {expected.job_id}")

    attempt_provenance = attempt_start.value["provenance"]
    selection_semantic: str | None = None
    if expected.kind == "neural":
        if value.get("schema_name") != "p05.materialized_neural_job":
            raise ValueError("neural decisive job has the wrong materialized schema")
        fixed = {
            "schema_version": 1,
            "protocol_id": "P05-G040-v3.2",
            "matrix_id": "P05-NEURAL-DECISIVE-v1",
            "stage": "decisive",
            "seed": expected.model_seed,
            "materialization_status": "created-not-executed",
            "execution_status": "not_started",
            "evidence_status": "unadjudicated",
            "claim_support": "forbidden_before_ledger_and_audit",
            "learning_rate_source": "bound_hash_verified_tuning_selection_manifest",
        }
        for key, required in fixed.items():
            if value.get(key) != required:
                raise ValueError(f"materialized neural {key} drifted")
        matrix_hash = _sha256_file(
            _required_real_file(NEURAL_MATRIX_PATH, name="canonical neural matrix")
        )
        if _required_sha256(value.get("matrix_sha256"), name="neural matrix_sha256") != matrix_hash:
            raise ValueError("materialized neural matrix hash drifted")
        config_name = value.get("config_file")
        if config_name != "config.yaml":
            raise ValueError("materialized neural config_file must be config.yaml")
        config_hash = _required_sha256(value.get("config_sha256"), name="config_sha256")
        _verify_child_hash(
            manifest.path.parent / config_name,
            config_hash,
            name="materialized neural config",
        )
        command = attempt_start.value["execution"]["command_argv"]
        config_flags = [index for index, item in enumerate(command) if item == "--config"]
        if len(config_flags) != 1 or config_flags[0] + 1 >= len(command):
            raise ValueError("neural attempt command must bind exactly one --config path")
        command_path = Path(command[config_flags[0] + 1])
        if not command_path.is_absolute():
            command_path = Path(attempt_start.value["execution"]["working_directory"]) / command_path
        command_path = _required_real_file(command_path, name="attempt command config")
        if command_path != _required_real_file(
            manifest.path.parent / config_name,
            name="materialized neural config",
        ):
            raise ValueError("attempt command does not point to the materialized config")
        selection_summary = value.get("tuning_selection")
        if not isinstance(selection_summary, dict):
            raise ValueError("decisive materialization lacks tuning-selection binding")
        if selection_summary.get("key") != f"{expected.dataset}/{expected.arm_id}":
            raise ValueError("decisive tuning-selection key does not match the job")
        selection_path = _resolve_selection_path(
            manifest.path, selection_summary.get("path")
        )
        direct = _required_sha256(
            selection_summary.get("sha256"), name="tuning-selection direct SHA-256"
        )
        if _sha256_file(selection_path) != direct:
            raise ValueError("tuning-selection direct SHA-256 mismatch")
        selection = _verify_standard_manifest(
            selection_path, name="tuning-selection manifest"
        )
        if selection.manifest_sha256 != direct:
            raise ValueError("tuning-selection manifest changed during verification")
        selection_value = selection.value
        if (
            selection_value.get("schema_name") != "p05.tuning_selection"
            or selection_value.get("phase") != "tuning_selection"
            or selection_value.get("status") != "computed_unadjudicated"
            or selection_value.get("claim_decision") != "not_performed"
            or selection_value.get("test_access") != "forbidden_and_not_performed"
        ):
            raise ValueError("tuning-selection scientific state is not frozen validation-only")
        selection_semantic = selection.semantic_sha256
    else:
        if value.get("schema_name") != "p05.materialized_cpu_arm_job_manifest":
            raise ValueError("CPU decisive job has the wrong materialized schema")
        fixed = {
            "schema_version": 1,
            "protocol_id": "P05-G040-v3.2",
            "matrix_id": "P05-CPU-ARMS-v1",
            "materialization_status": "created_not_executed",
            "execution_status": "not_started",
            "evidence_status": "unadjudicated",
            "claim_support": "forbidden_before_ledger_and_audit",
        }
        for key, required in fixed.items():
            if value.get(key) != required:
                raise ValueError(f"materialized CPU {key} drifted")
        matrix_hash = _sha256_file(
            _required_real_file(CPU_MATRIX_PATH, name="canonical CPU matrix")
        )
        if _required_sha256(value.get("matrix_sha256"), name="CPU matrix_sha256") != matrix_hash:
            raise ValueError("materialized CPU matrix hash drifted")
        job_name = value.get("job_file")
        if job_name != "job.yaml":
            raise ValueError("materialized CPU job_file must be job.yaml")
        job_hash = _required_sha256(value.get("job_sha256"), name="CPU job_sha256")
        _verify_child_hash(
            manifest.path.parent / job_name,
            job_hash,
            name="materialized CPU job",
        )
    return manifest, selection_semantic


def _verify_run_manifest(
    path_value: str | Path | None,
    *,
    expected: _ExpectedJob,
    attempt_start: _VerifiedManifest,
    attempt_terminal: _VerifiedManifest,
    materialized: _VerifiedManifest,
) -> _VerifiedManifest | None:
    if expected.kind != "neural":
        if path_value is not None:
            raise ValueError("CPU collection descriptors must not invent a neural run contract")
        return None
    if path_value is None:
        raise ValueError("neural decisive collection requires a run-contract manifest")
    manifest = _verify_manifest(path_value, name="run-contract manifest")
    value = manifest.value
    if value.get("schema_name") != "p05.run_artifact_bundle" or value.get(
        "schema_version"
    ) != 1:
        raise ValueError("neural run-contract schema is unsupported")
    if value.get("paper_id") != "P05" or value.get("dataset_id") != expected.dataset_id:
        raise ValueError("run-contract dataset identity mismatch")
    outputs = attempt_terminal.value["outputs"]
    if outputs.get("run_contract") != manifest.semantic_sha256:
        raise ValueError("attempt run_contract hash mismatch")
    provenance = value.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != {
        "checkpoint_sha256",
        "code_sha256",
        "config_sha256",
        "model_sha256",
    }:
        raise ValueError("run-contract provenance field set is invalid")
    provenance = {
        key: _required_sha256(item, name=f"run-contract provenance {key}")
        for key, item in provenance.items()
    }
    attempt_provenance = attempt_start.value["provenance"]
    if provenance["config_sha256"] != attempt_provenance["config_snapshot_sha256"]:
        raise ValueError("run-contract config hash differs from attempt provenance")
    if provenance["code_sha256"] != attempt_provenance["code_snapshot_sha256"]:
        raise ValueError("run-contract code hash differs from attempt provenance")
    if outputs.get("checkpoint") != provenance["checkpoint_sha256"]:
        raise ValueError("attempt checkpoint hash differs from run contract")
    normalization = value.get("normalization_plan")
    weights = value.get("weight_plans")
    if not isinstance(normalization, dict) or not isinstance(weights, dict):
        raise ValueError("run-contract normalization/weight provenance is missing")
    if _required_sha256(
        normalization.get("sha256"), name="run normalization_plan.sha256"
    ) != attempt_provenance["normalization_sha256"]:
        raise ValueError("run normalization hash differs from attempt provenance")
    for role, attempt_field in (
        ("train", "train_weight_plan_sha256"),
        ("validation", "validation_weight_plan_sha256"),
    ):
        plan = weights.get(role)
        if not isinstance(plan, dict):
            raise ValueError(f"run-contract {role} weight plan is missing")
        if _required_sha256(
            plan.get("sha256"), name=f"run {role} weight plan SHA-256"
        ) != attempt_provenance[attempt_field]:
            raise ValueError(
                f"run {role} weight hash differs from attempt provenance"
            )
    runtime = value.get("runtime_identity")
    if not isinstance(runtime, dict):
        raise ValueError("run-contract runtime identity is missing")
    if _canonical_json_bytes(runtime) != _canonical_json_bytes(
        attempt_start.value["execution"].get("device_identity")
    ):
        raise ValueError("run-contract runtime identity differs from attempt start")
    if runtime.get("physical_gpu_index") not in {0, 1}:
        raise ValueError("decisive run used a forbidden physical GPU")
    if runtime.get("gpu_uuid") != materialized.value.get("expected_gpu_uuid"):
        raise ValueError("run-contract GPU UUID differs from materialized binding")
    return manifest


def _load_result_csv(path_value: str | Path) -> tuple[Path, dict[str, str], str]:
    path = _required_real_file(path_value, name="neural result CSV")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"invalid neural result CSV: {path}") from exc
    if len(rows) != 2 or not rows[0] or len(rows[0]) != len(rows[1]):
        raise ValueError("neural result CSV must contain exactly one complete result row")
    header = rows[0]
    if len(header) != len(set(header)) or any(not name for name in header):
        raise ValueError("neural result CSV header is empty or duplicated")
    record = dict(zip(header, rows[1], strict=True))
    for key, raw in record.items():
        text = raw.strip()
        if not text:
            raise ValueError(f"neural result CSV field {key!r} is empty")
        lowered = text.lower()
        if lowered in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity"}:
            raise ValueError(f"neural result CSV field {key!r} is non-finite")
        try:
            numeric = float(text)
        except ValueError:
            numeric = None
        if numeric is not None and not math.isfinite(numeric):
            raise ValueError(f"neural result CSV field {key!r} is non-finite")
        if key.strip().lower() in _LEAKAGE_KEYS and "test" in lowered:
            raise ValueError(f"neural result CSV records test-fitted selection in {key!r}")
    return path, record, _sha256_file(path)


def _verify_prediction_manifest(
    manifest: _VerifiedManifest,
    *,
    expected: _ExpectedJob,
    checkpoint_sha256: str,
    config_sha256: str,
    code_sha256: str,
    run_contract_sha256: str | None,
) -> None:
    value = manifest.value
    if value.get("schema_name") != "p05.window_predictions" or value.get(
        "schema_version"
    ) != 1:
        raise ValueError("prediction artifact schema is unsupported")
    if value.get("evidence_status") != "unadjudicated":
        raise ValueError("prediction artifact has a premature evidence state")
    conclusion = value.get("conclusion_control")
    if not isinstance(conclusion, dict) or conclusion.get("claim_decisions") != "not_performed":
        raise ValueError("prediction artifact performed a premature claim decision")
    if conclusion.get("decisive") is not False or conclusion.get("status") != "unadjudicated":
        raise ValueError("prediction artifact conclusion control drifted")
    splits = value.get("splits")
    if not isinstance(splits, dict) or "test" not in splits:
        raise ValueError("prediction artifact lacks the immutable test export")
    test = splits["test"]
    if not isinstance(test, dict) or type(test.get("sample_count")) is not int or test[
        "sample_count"
    ] <= 0:
        raise ValueError("prediction artifact has no test samples")
    provenance = value.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("prediction artifact provenance is missing")
    required = {
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "code_sha256": code_sha256,
    }
    if run_contract_sha256 is not None:
        required["run_contract_sha256"] = run_contract_sha256
    for key, expected_hash in required.items():
        if _required_sha256(provenance.get(key), name=f"prediction provenance {key}") != expected_hash:
            raise ValueError(f"prediction provenance {key} mismatch for {expected.job_id}")


def _verify_evaluation_manifest(
    manifest: _VerifiedManifest,
    *,
    expected: _ExpectedJob,
    run_manifest: _VerifiedManifest,
) -> None:
    value = manifest.value
    if value.get("schema_name") != "p05.c2_c3_evaluation_bundle" or value.get(
        "schema_version"
    ) != 2:
        raise ValueError("P05-M evaluation bundle schema is unsupported")
    conclusion = value.get("conclusion_control")
    if not isinstance(conclusion, dict):
        raise ValueError("evaluation conclusion control is missing")
    required_conclusion = {
        "claim_decisions": "not_performed",
        "decisive": False,
        "status": "computed_unadjudicated",
        "predictive_cost_gate": "not_evaluated",
        "operational_wording_gate": "not_evaluated",
    }
    for key, required in required_conclusion.items():
        if conclusion.get(key) != required:
            raise ValueError(f"evaluation conclusion control {key} drifted")
    frozen = value.get("frozen_parameters")
    if not isinstance(frozen, dict) or frozen.get("dataset") != expected.dataset or frozen.get(
        "model_seed"
    ) != expected.model_seed:
        raise ValueError("evaluation dataset/model-seed binding mismatch")
    run_provenance = run_manifest.value["provenance"]
    inputs = value.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("evaluation inputs are missing")
    for role in ("validation_trace", "evaluation_trace"):
        trace = inputs.get(role)
        provenance = trace.get("provenance") if isinstance(trace, dict) else None
        if not isinstance(provenance, dict):
            raise ValueError(f"evaluation {role} provenance is missing")
        for key in ("checkpoint_sha256", "config_sha256", "model_sha256"):
            if provenance.get(key) != run_provenance[key]:
                raise ValueError(f"evaluation {role} {key} differs from the run contract")


def _verify_record(descriptor_value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(descriptor_value, Mapping):
        raise TypeError("each decisive descriptor must be a mapping")
    descriptor = dict(descriptor_value)
    if set(descriptor) != _DESCRIPTOR_KEYS:
        missing = sorted(_DESCRIPTOR_KEYS - set(descriptor))
        unexpected = sorted(set(descriptor) - _DESCRIPTOR_KEYS, key=str)
        raise ValueError(
            f"decisive descriptor field mismatch: missing={missing}, unexpected={unexpected}"
        )
    job_id = descriptor["job_id"]
    if not isinstance(job_id, str) or job_id not in _EXPECTED_BY_ID:
        raise ValueError(f"unregistered decisive job_id: {job_id!r}")
    expected = _EXPECTED_BY_ID[job_id]
    start, terminal = _load_attempt(descriptor["attempt_package_dir"], expected)
    outputs = terminal.value["outputs"]

    materialized, selection_semantic = _verify_materialized(
        descriptor["materialized_manifest_path"],
        expected=expected,
        attempt_start=start,
        attempt_terminal=terminal,
    )
    run = _verify_run_manifest(
        descriptor["run_manifest_path"],
        expected=expected,
        attempt_start=start,
        attempt_terminal=terminal,
        materialized=materialized,
    )
    evidence = _verify_manifest(
        descriptor["evidence_manifest_path"], name="prediction/evaluation manifest"
    )

    checkpoint_sha256: str
    result_hash: str
    if expected.kind == "neural":
        assert run is not None
        checkpoint_sha256 = run.value["provenance"]["checkpoint_sha256"]
        result_path, result_row, result_hash = _load_result_csv(descriptor["result_path"])
        if outputs.get("result") != result_hash:
            raise ValueError("attempt result CSV hash mismatch")
        required_result_links = {
            "materialized_job_id": expected.job_id,
            "materialized_job_semantic_sha256": materialized.semantic_sha256,
            "run_contract_semantic_sha256": run.semantic_sha256,
        }
        evidence_output_name: str
        evidence_result_column: str
        if expected.arm_id == "P05-M":
            evidence_output_name = "evaluation"
            evidence_result_column = "p05_evaluation_semantic_sha256"
            _verify_evaluation_manifest(evidence, expected=expected, run_manifest=run)
        else:
            evidence_output_name = "predictions"
            evidence_result_column = "p05_prediction_semantic_sha256"
            _verify_prediction_manifest(
                evidence,
                expected=expected,
                checkpoint_sha256=checkpoint_sha256,
                config_sha256=run.value["provenance"]["config_sha256"],
                code_sha256=run.value["provenance"]["code_sha256"],
                run_contract_sha256=run.semantic_sha256,
            )
        required_result_links[evidence_result_column] = evidence.semantic_sha256
        for key, required in required_result_links.items():
            if result_row.get(key) != str(required):
                raise ValueError(f"result CSV {key} does not bind {expected.job_id}")
        if outputs.get(evidence_output_name) != evidence.semantic_sha256:
            raise ValueError(f"attempt {evidence_output_name} hash mismatch")
        del result_path
    elif expected.kind == "b2":
        result = _verify_manifest(descriptor["result_path"], name="P05-B2 result manifest")
        if result.value.get("schema_name") != "p05.b2_posthoc_fuzzy_surrogate":
            raise ValueError("P05-B2 result schema is unsupported")
        if result.value.get("baseline_id") != "P05-B2" or result.value.get(
            "evidence_status"
        ) != "unadjudicated":
            raise ValueError("P05-B2 result identity/evidence state drifted")
        provenance = result.value.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("model_seed") != expected.model_seed:
            raise ValueError("P05-B2 result model seed mismatch")
        model = result.value.get("model")
        expected_classes = 4 if expected.dataset == "CWRU" else 2
        if not isinstance(model, dict) or model.get("num_classes") != expected_classes:
            raise ValueError("P05-B2 result dataset/class binding mismatch")
        checkpoint_sha256 = _required_sha256(
            result.value["content"].get("checkpoint_sha256"),
            name="P05-B2 checkpoint_sha256",
        )
        result_hash = result.semantic_sha256
        if outputs.get("result") != result_hash or outputs.get("checkpoint") != checkpoint_sha256:
            raise ValueError("P05-B2 attempt result/checkpoint hash mismatch")
        _verify_prediction_manifest(
            evidence,
            expected=expected,
            checkpoint_sha256=checkpoint_sha256,
            config_sha256=start.value["provenance"]["config_snapshot_sha256"],
            code_sha256=start.value["provenance"]["code_snapshot_sha256"],
            run_contract_sha256=None,
        )
        if outputs.get("predictions") != evidence.semantic_sha256:
            raise ValueError("P05-B2 attempt predictions hash mismatch")
    else:
        result = _verify_manifest(descriptor["result_path"], name="P05-B4 result manifest")
        if result.semantic_sha256 != evidence.semantic_sha256:
            raise ValueError("P05-B4 result and prediction package must be the same fit artifact")
        value = result.value
        if (
            value.get("baseline_id") != "P05-B4"
            or value.get("dataset_id") != expected.dataset_id
            or value.get("fit_id") != f"P05-B4-dataset-{expected.dataset_id}"
            or value.get("evidence_status") != "unadjudicated"
        ):
            raise ValueError("P05-B4 result identity/evidence state drifted")
        fit = value.get("fit_contract")
        if not isinstance(fit, dict) or fit.get("fits_per_dataset") != 1 or fit.get(
            "model_seed_repetition"
        ) != "forbidden_as_redundant_deterministic_fit":
            raise ValueError("P05-B4 result illegally introduces a model-seed axis")
        clustering = fit.get("clustering")
        if not isinstance(clustering, dict) or clustering.get(
            "initialization_seed"
        ) != B4_FCM_INITIALISATION_SEED:
            raise ValueError("P05-B4 FCM initialisation seed drifted")
        predictions = value.get("provenance", {}).get("predictions")
        if not isinstance(predictions, dict) or set(predictions) != {
            "train",
            "validation",
            "test",
        }:
            raise ValueError("P05-B4 must export train/validation/test exactly once")
        checkpoint_sha256 = _required_sha256(
            value["files"].get("model.npz"), name="P05-B4 model hash"
        )
        result_hash = result.semantic_sha256
        if outputs.get("result") != result_hash or outputs.get("predictions") != result_hash:
            raise ValueError("P05-B4 attempt result/predictions hash mismatch")

    required_outputs = {
        "code_snapshot",
        "config_snapshot",
        "materialized_job",
        "result",
    }
    if expected.kind == "neural":
        required_outputs.update({"all_results", "checkpoint", "run_contract"})
        if expected.arm_id == "P05-M":
            required_outputs.update(
                {
                    "diagnostics_test",
                    "diagnostics_val",
                    "evaluation",
                    "trace_test",
                    "trace_val",
                }
            )
        else:
            required_outputs.add("predictions")
    elif expected.kind == "b2":
        required_outputs.update({"checkpoint", "predictions"})
    else:
        required_outputs.add("predictions")
    missing_outputs = sorted(required_outputs - set(outputs))
    if missing_outputs:
        raise ValueError(
            f"completed attempt lacks required outputs for {expected.job_id}: {missing_outputs}"
        )
    if outputs.get("code_snapshot") != start.value["provenance"]["code_snapshot_sha256"]:
        raise ValueError("attempt code_snapshot output differs from provenance")
    if outputs.get("config_snapshot") != start.value["provenance"]["config_snapshot_sha256"]:
        raise ValueError("attempt config_snapshot output differs from provenance")

    run_semantic = run.semantic_sha256 if run is not None else None
    model_sha256 = run.value["provenance"]["model_sha256"] if run is not None else None
    b2_parent = None
    if expected.kind == "b2":
        b2_provenance = result.value["provenance"]
        b2_parent = {
            "checkpoint_sha256": _required_sha256(
                b2_provenance.get("b0_checkpoint_sha256"),
                name="P05-B2 parent checkpoint SHA-256",
            ),
            "run_contract_semantic_sha256": _required_sha256(
                b2_provenance.get("b0_run_artifact_semantic_sha256"),
                name="P05-B2 parent run-contract SHA-256",
            ),
        }
    return {
        "job_id": expected.job_id,
        "kind": expected.kind,
        "arm_id": expected.arm_id,
        "dataset": expected.dataset,
        "dataset_id": expected.dataset_id,
        "model_seed": expected.model_seed,
        "attempt_seed": expected.attempt_seed,
        "attempt_id": start.value["attempt"]["attempt_id"],
        "attempt_start_semantic_sha256": start.semantic_sha256,
        "attempt_terminal_semantic_sha256": terminal.semantic_sha256,
        "materialized_semantic_sha256": materialized.semantic_sha256,
        "tuning_selection_semantic_sha256": selection_semantic,
        "run_contract_semantic_sha256": run_semantic,
        "checkpoint_sha256": checkpoint_sha256,
        "model_sha256": model_sha256,
        "result_sha256": result_hash,
        "evidence_semantic_sha256": evidence.semantic_sha256,
        "provenance": dict(start.value["provenance"]),
        "b2_parent": b2_parent,
    }


def _verify_collection_consistency(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        return
    global_fields = (
        "source_metadata_sha256",
        "derived_metadata_sha256",
        "signal_cache_manifest_sha256",
        "code_snapshot_sha256",
    )
    for field in global_fields:
        values = {record["provenance"][field] for record in records}
        if len(values) != 1:
            raise ValueError(f"decisive collection has inconsistent global provenance {field}")
    per_dataset_fields = (
        "split_manifest_sha256",
        "normalization_sha256",
        "train_weight_plan_sha256",
        "validation_weight_plan_sha256",
    )
    for dataset, _dataset_id in DATASETS:
        subset = [record for record in records if record["dataset"] == dataset]
        for field in per_dataset_fields:
            values = {record["provenance"][field] for record in subset}
            if len(values) > 1:
                raise ValueError(
                    f"decisive collection has inconsistent {dataset} provenance {field}"
                )

    # B2 is bound to the corresponding completed B0 artifact.  Check the
    # dependency whenever both are present; a full collection necessarily
    # checks all ten bindings.
    index = {record["job_id"]: record for record in records}
    for record in records:
        if record["kind"] != "b2":
            continue
        parent_id = f"P05-DEC-B0-{record['dataset']}-S{record['model_seed']}"
        parent = index.get(parent_id)
        if parent is None:
            continue
        if record["b2_parent"] != {
            "checkpoint_sha256": parent["checkpoint_sha256"],
            "run_contract_semantic_sha256": parent["run_contract_semantic_sha256"],
        }:
            raise ValueError(f"P05-B2 parent binding differs from {parent_id}")

    for arm in NEURAL_ARMS:
        for dataset, _dataset_id in DATASETS:
            subset = [
                record
                for record in records
                if record["arm_id"] == arm and record["dataset"] == dataset
            ]
            selections = {
                record["tuning_selection_semantic_sha256"] for record in subset
            }
            if len(selections) > 1:
                raise ValueError(
                    f"decisive seeds do not share one frozen tuning selection for {dataset}/{arm}"
                )


def build_p05_decisive_collection_manifest(
    descriptors: Sequence[Mapping[str, Any]],
) -> P05DecisiveCollectionResult:
    """Validate supplied artifacts and return an unadjudicated semantic manifest.

    An empty or otherwise proper subset is represented as
    ``collection_incomplete``.  A malformed, duplicate, failed, invalidated,
    non-finite, leaked, or hash-inconsistent supplied record raises instead of
    being silently treated as missing.
    """

    if isinstance(descriptors, (str, bytes)) or not isinstance(descriptors, Sequence):
        raise TypeError("descriptors must be a sequence of mappings")
    normalised: list[dict[str, Any]] = []
    seen_job_ids: set[str] = set()
    seen_attempt_ids: set[str] = set()
    for descriptor in descriptors:
        record = _verify_record(descriptor)
        job_id = record["job_id"]
        if job_id in seen_job_ids:
            raise ValueError(f"duplicate decisive job descriptor: {job_id}")
        if record["attempt_id"] in seen_attempt_ids:
            raise ValueError(f"duplicate decisive attempt_id: {record['attempt_id']}")
        seen_job_ids.add(job_id)
        seen_attempt_ids.add(record["attempt_id"])
        normalised.append(record)

    normalised.sort(key=lambda record: EXPECTED_JOB_IDS.index(record["job_id"]))
    _verify_collection_consistency(normalised)
    missing = tuple(job_id for job_id in EXPECTED_JOB_IDS if job_id not in seen_job_ids)
    status = "computed_unadjudicated" if not missing else "collection_incomplete"
    semantic = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "paper_id": "P05",
        "protocol": {
            "protocol_id": "P05-G040-v3.2",
            "central_gpu_budget_job_count": CENTRAL_GPU_BUDGET_JOB_COUNT,
            "central_gpu_budget_components": {
                "pilot": 4,
                "tuning": 16,
                "decisive_neural": DECISIVE_NEURAL_ARTIFACT_COUNT,
            },
            "decisive_execution_artifact_count": DECISIVE_EXECUTION_ARTIFACT_COUNT,
            "decisive_execution_components": {
                "neural": DECISIVE_NEURAL_ARTIFACT_COUNT,
                "P05-B2": DECISIVE_B2_ARTIFACT_COUNT,
                "P05-B4": DECISIVE_B4_ARTIFACT_COUNT,
            },
            "b4_fit_axis": "dataset_only",
            "b4_model_seed_axis": "forbidden",
            "pilot_or_tuning_as_decisive_result": "forbidden",
        },
        "collection": {
            "status": status,
            "expected_job_count": DECISIVE_EXECUTION_ARTIFACT_COUNT,
            "collected_job_count": len(normalised),
            "expected_job_ids": list(EXPECTED_JOB_IDS),
            "missing_job_ids": list(missing),
            "records": normalised,
        },
        "conclusion_control": {
            "claim_decisions": "not_performed",
            "statistical_adjudication": "not_performed",
            "p_values": "not_computed",
            "evidence_status": status,
            "positive_claim_support": "forbidden_before_separate_registered_adjudication",
        },
    }
    semantic_hash = _sha256_bytes(_canonical_json_bytes(semantic))
    manifest = {**semantic, "content": {"semantic_sha256": semantic_hash}}
    return P05DecisiveCollectionResult(
        manifest=manifest,
        semantic_sha256=semantic_hash,
        status=status,
        collected_job_count=len(normalised),
        missing_job_ids=missing,
    )


__all__ = [
    "B4_FCM_INITIALISATION_SEED",
    "CENTRAL_GPU_BUDGET_JOB_COUNT",
    "DECISIVE_EXECUTION_ARTIFACT_COUNT",
    "EXPECTED_JOB_IDS",
    "P05DecisiveCollectionResult",
    "build_p05_decisive_collection_manifest",
    "expected_p05_decisive_job_ids",
]
