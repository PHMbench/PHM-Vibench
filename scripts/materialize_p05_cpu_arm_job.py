#!/usr/bin/env python3
"""Create one hash-bound, create-only P05 B2/B4 CPU job package."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shlex
import shutil
import tempfile
from collections.abc import Mapping
from itertools import product
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = (
    REPO_ROOT
    / "configs"
    / "experiments"
    / "p05"
    / "protocol"
    / "cpu_arm_matrix_p05_v1.yaml"
)
JOB_NAME = "job.yaml"
MANIFEST_NAME = "manifest.json"

REGISTERED_B0_SEEDS = (42, 123, 456, 789, 1024)
EXPECTED_DATASETS = {"CWRU": (1, 4), "XJTU": (2, 2)}
DATASET_ARTIFACTS = frozenset(
    {
        "train_weights",
        "validation_weights",
        "channel_normalization",
        "split_manifest",
        "signal_cache_manifest",
    }
)
B0_ARTIFACTS = frozenset({"checkpoint", "run_manifest", "predictions"})
COMMAND_PREFIX = ("conda", "run", "-n", "LQ_signal", "python")

_PLACEHOLDER_PATTERN = re.compile(r"^__REQUIRED_[A-Z0-9_]+__$")
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _load_yaml(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required P05 CPU matrix must be a real file: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("P05 CPU matrix must contain a mapping")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"bindings JSON contains forbidden constant {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"bindings JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_bindings(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"bindings JSON must be a real file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid bindings JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("bindings JSON must contain one object")
    if any(not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()):
        raise TypeError("bindings JSON keys and values must be strings")
    return value


def _artifact_template(value: Any, *, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name} must contain exactly path and sha256 placeholders")
    path_placeholder = value["path"]
    hash_placeholder = value["sha256"]
    if (
        not isinstance(path_placeholder, str)
        or _PLACEHOLDER_PATTERN.fullmatch(path_placeholder) is None
        or not path_placeholder.endswith("_PATH__")
    ):
        raise ValueError(f"{name}.path must be a required path placeholder")
    if (
        not isinstance(hash_placeholder, str)
        or _PLACEHOLDER_PATTERN.fullmatch(hash_placeholder) is None
        or not hash_placeholder.endswith("_SHA256__")
    ):
        raise ValueError(f"{name}.sha256 must be a required SHA-256 placeholder")
    if path_placeholder == hash_placeholder:
        raise ValueError(f"{name} path and SHA-256 placeholders must differ")
    return {"path": path_placeholder, "sha256": hash_placeholder}


def _validate_matrix(matrix: Mapping[str, Any]) -> None:
    expected_top = {
        "schema_version",
        "kind",
        "paper_id",
        "protocol_id",
        "matrix_id",
        "status",
        "budget",
        "runtime",
        "outputs",
        "arms",
        "datasets",
        "jobs",
    }
    if set(matrix) != expected_top:
        raise ValueError("P05 CPU matrix top-level schema drift")
    if matrix["schema_version"] != 1:
        raise ValueError("P05 CPU matrix schema_version must be 1")
    if matrix["kind"] != "p05_frozen_cpu_arm_execution_matrix":
        raise ValueError("P05 CPU matrix kind drift")
    if matrix["paper_id"] != "P05" or matrix["protocol_id"] != "P05-G040-v3.2":
        raise ValueError("P05 CPU matrix protocol identity drift")
    if matrix["matrix_id"] != "P05-CPU-ARMS-v1":
        raise ValueError("P05 CPU matrix ID drift")
    if matrix["status"] != "frozen_declarative_awaiting_artifact_hash_binding":
        raise ValueError("P05 CPU matrix must remain at the artifact-binding gate")

    budget = matrix["budget"]
    if not isinstance(budget, Mapping):
        raise ValueError("P05 CPU matrix budget must be a mapping")
    if budget.get("device_class") != "cpu":
        raise ValueError("P05 CPU arms must remain CPU-only")
    if budget.get("total_fit_ceiling") != 12:
        raise ValueError("P05 CPU total fit ceiling must equal 12")
    if budget.get("arm_fit_ceiling") != {"P05-B2": 10, "P05-B4": 2}:
        raise ValueError("P05 CPU arm fit ceilings must equal B2=10 and B4=2")
    if budget.get("ceiling_exceeded") != "hard_error":
        raise ValueError("P05 CPU fit ceiling must fail closed")

    runtime = matrix["runtime"]
    if not isinstance(runtime, Mapping):
        raise ValueError("P05 CPU runtime must be a mapping")
    if tuple(runtime.get("command_prefix", ())) != COMMAND_PREFIX:
        raise ValueError("P05 CPU commands must start with conda run -n LQ_signal python")
    if runtime.get("conda_environment") != "LQ_signal":
        raise ValueError("P05 CPU jobs require conda environment LQ_signal")
    if runtime.get("gpu_use") != "forbidden" or runtime.get("network_use") != "forbidden":
        raise ValueError("P05 CPU jobs must forbid GPU and network use")
    if runtime.get("output_collision") != "atomic_create_only_hard_error":
        raise ValueError("P05 CPU outputs must be strict create-only")

    outputs = matrix["outputs"]
    if not isinstance(outputs, Mapping):
        raise ValueError("P05 CPU outputs contract must be a mapping")
    if outputs.get("evidence_status") != "unadjudicated":
        raise ValueError("P05 CPU outputs must remain unadjudicated")
    if outputs.get("execution_status_on_materialization") != "not_started":
        raise ValueError("P05 CPU materialization cannot claim execution")
    if outputs.get("claim_support_before_ledger_and_audit") != "forbidden":
        raise ValueError("P05 CPU materialization cannot support claims")

    arms = matrix["arms"]
    if not isinstance(arms, Mapping) or set(arms) != {"P05-B2", "P05-B4"}:
        raise ValueError("P05 CPU matrix must contain exactly B2 and B4")
    b2 = arms["P05-B2"]
    b4 = arms["P05-B4"]
    if b2.get("fit_count") != 10:
        raise ValueError("P05-B2 must contain exactly ten fits")
    if tuple(b2.get("registered_b0_model_seeds", ())) != REGISTERED_B0_SEEDS:
        raise ValueError("P05-B2 registered B0 seed set drift")
    if set(b2.get("required_dataset_artifacts", ())) != DATASET_ARTIFACTS:
        raise ValueError("P05-B2 dataset dependency set drift")
    if set(b2.get("required_b0_artifacts", ())) != B0_ARTIFACTS:
        raise ValueError("P05-B2 B0 dependency set drift")
    if b4.get("fit_count") != 2 or b4.get("fits_per_dataset") != 1:
        raise ValueError("P05-B4 must contain one fit per dataset and two total")
    if b4.get("fit_axis") != "dataset_only":
        raise ValueError("P05-B4 fit axis must be dataset-only")
    if b4.get("model_seed_axis") != "forbidden":
        raise ValueError("P05-B4 model-seed repetition must be forbidden")
    if b4.get("model_seed_repetition") != (
        "forbidden_as_redundant_deterministic_fit"
    ):
        raise ValueError("P05-B4 deterministic repetition contract drift")
    if b4.get("deterministic_fcm_initialization_seed") != 20260801:
        raise ValueError("P05-B4 deterministic FCM seed drift")
    if set(b4.get("required_dataset_artifacts", ())) != (
        DATASET_ARTIFACTS - {"validation_weights"}
    ):
        raise ValueError("P05-B4 dataset dependency set drift")
    if b4.get("required_b0_artifacts") != []:
        raise ValueError("P05-B4 must not inherit B0 seed artifacts")

    datasets = matrix["datasets"]
    if not isinstance(datasets, Mapping) or set(datasets) != set(EXPECTED_DATASETS):
        raise ValueError("P05 CPU matrix dataset set drift")
    seen_dataset_placeholders: set[str] = set()
    for dataset_name, (dataset_id, num_classes) in EXPECTED_DATASETS.items():
        dataset = datasets[dataset_name]
        if dataset.get("dataset_id") != dataset_id:
            raise ValueError(f"{dataset_name} dataset ID drift")
        if dataset.get("num_classes") != num_classes:
            raise ValueError(f"{dataset_name} class count drift")
        artifacts = dataset.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != DATASET_ARTIFACTS:
            raise ValueError(f"{dataset_name} dependency set drift")
        for artifact_name, artifact in artifacts.items():
            template = _artifact_template(
                artifact,
                name=f"datasets.{dataset_name}.artifacts.{artifact_name}",
            )
            placeholders = set(template.values())
            if seen_dataset_placeholders.intersection(placeholders):
                raise ValueError("dataset artifact placeholders must be unique")
            seen_dataset_placeholders.update(placeholders)

    jobs = matrix["jobs"]
    if not isinstance(jobs, list) or len(jobs) != 12:
        raise ValueError("P05 CPU matrix must contain exactly 12 jobs")
    if any(not isinstance(job, Mapping) for job in jobs):
        raise ValueError("P05 CPU jobs must be mappings")
    job_ids = [job.get("id") for job in jobs]
    if any(not isinstance(job_id, str) or not job_id for job_id in job_ids):
        raise ValueError("P05 CPU job IDs must be non-empty strings")
    if len(set(job_ids)) != len(job_ids):
        raise ValueError("P05 CPU job IDs must be unique")
    output_dirs = [job.get("output_dir") for job in jobs]
    if any(not isinstance(path, str) or not path for path in output_dirs):
        raise ValueError("P05 CPU output directories must be non-empty strings")
    if len(set(output_dirs)) != len(output_dirs):
        raise ValueError("P05 CPU output directories must be unique")

    b2_jobs = [job for job in jobs if job.get("arm") == "P05-B2"]
    b4_jobs = [job for job in jobs if job.get("arm") == "P05-B4"]
    expected_b2 = set(product(EXPECTED_DATASETS, REGISTERED_B0_SEEDS))
    observed_b2 = {
        (job.get("dataset"), job.get("b0_model_seed")) for job in b2_jobs
    }
    if len(b2_jobs) != 10 or observed_b2 != expected_b2:
        raise ValueError("P05-B2 jobs must equal dataset x five registered B0 seeds")
    expected_b4 = set(EXPECTED_DATASETS)
    observed_b4 = {job.get("dataset") for job in b4_jobs}
    if len(b4_jobs) != 2 or observed_b4 != expected_b4:
        raise ValueError("P05-B4 jobs must equal one fit per dataset")

    seen_b0_placeholders: set[str] = set()
    for job in b2_jobs:
        if set(job) != {
            "id",
            "arm",
            "dataset",
            "b0_model_seed",
            "b0_artifacts",
            "output_dir",
            "materialize_command",
        }:
            raise ValueError("P05-B2 job schema drift")
        artifacts = job["b0_artifacts"]
        if not isinstance(artifacts, Mapping) or set(artifacts) != B0_ARTIFACTS:
            raise ValueError("P05-B2 job must bind checkpoint/run/predictions")
        for artifact_name, artifact in artifacts.items():
            template = _artifact_template(
                artifact,
                name=f"jobs.{job['id']}.b0_artifacts.{artifact_name}",
            )
            placeholders = set(template.values())
            if seen_b0_placeholders.intersection(placeholders):
                raise ValueError("B0 artifact placeholders must be unique per job")
            seen_b0_placeholders.update(placeholders)
        expected_id = (
            f"P05-CPU-B2-{job['dataset']}-S{job['b0_model_seed']}"
        )
        if job["id"] != expected_id:
            raise ValueError("P05-B2 job ID does not bind its dataset and B0 seed")

    for job in b4_jobs:
        if set(job) != {
            "id",
            "arm",
            "dataset",
            "fit_identity",
            "output_dir",
            "materialize_command",
        }:
            raise ValueError("P05-B4 job schema forbids model-seed fields")
        if job["fit_identity"] != "dataset_only_no_model_seed":
            raise ValueError("P05-B4 fit identity must exclude model seeds")
        if job["id"] != f"P05-CPU-B4-{job['dataset']}":
            raise ValueError("P05-B4 job ID must contain only its dataset axis")

    for job in jobs:
        command = job.get("materialize_command")
        if not isinstance(command, str):
            raise ValueError("every P05 CPU job needs a materialization command")
        try:
            command_tokens = tuple(shlex.split(command))
        except ValueError as exc:
            raise ValueError(f"invalid materialization command for {job['id']}") from exc
        expected_command = COMMAND_PREFIX + (
            "scripts/materialize_p05_cpu_arm_job.py",
            "--job-id",
            job["id"],
            "--bindings-json",
            "BINDINGS_JSON_PATH",
            "--output-package",
            "CREATE_ONLY_PACKAGE_PATH",
        )
        if command_tokens[: len(COMMAND_PREFIX)] != COMMAND_PREFIX:
            raise ValueError(
                "every P05 CPU command must start with conda run -n LQ_signal python"
            )
        if command_tokens != expected_command:
            raise ValueError(
                "P05 CPU materialization command must exactly bind the registered "
                "materializer, job ID, bindings path, and create-only package"
            )


def _selected_job(matrix: Mapping[str, Any], job_id: str) -> Mapping[str, Any]:
    jobs = [job for job in matrix["jobs"] if job["id"] == job_id]
    if len(jobs) != 1:
        raise ValueError(f"job_id must identify exactly one P05 CPU job: {job_id!r}")
    return jobs[0]


def _resolve_artifact(
    template: Mapping[str, Any],
    *,
    bindings: Mapping[str, str],
    name: str,
) -> dict[str, Any]:
    normalized = _artifact_template(template, name=name)
    path_placeholder = normalized["path"]
    hash_placeholder = normalized["sha256"]
    raw_path = bindings[path_placeholder]
    expected_hash = bindings[hash_placeholder]
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        raise ValueError(f"binding {path_placeholder} must be a non-empty path")
    if _PLACEHOLDER_PATTERN.fullmatch(raw_path) is not None:
        raise ValueError(f"binding {path_placeholder} remains an unresolved placeholder")
    if not isinstance(expected_hash, str) or _SHA256_PATTERN.fullmatch(expected_hash) is None:
        raise ValueError(f"binding {hash_placeholder} must be a hexadecimal SHA-256")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    if candidate.is_symlink():
        raise ValueError(f"bound artifact must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"bound artifact does not exist: {candidate}") from exc
    if not resolved.is_file():
        raise ValueError(f"bound artifact must be a file: {resolved}")
    observed_hash = _sha256_file(resolved)
    if observed_hash != expected_hash.lower():
        raise ValueError(f"bound artifact SHA-256 mismatch for {name}")
    return {
        "path": str(resolved),
        "sha256": observed_hash,
        "size_bytes": resolved.stat().st_size,
    }


def _resolve_job(
    matrix: Mapping[str, Any],
    job: Mapping[str, Any],
    bindings: Mapping[str, str],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    arm = matrix["arms"][job["arm"]]
    dataset = matrix["datasets"][job["dataset"]]
    required_dataset_artifacts = tuple(arm["required_dataset_artifacts"])
    templates: dict[str, Mapping[str, Any]] = {
        name: dataset["artifacts"][name] for name in required_dataset_artifacts
    }
    if job["arm"] == "P05-B2":
        for name in arm["required_b0_artifacts"]:
            templates[f"b0_{name}"] = job["b0_artifacts"][name]

    required_placeholders = tuple(
        sorted(
            placeholder
            for template in templates.values()
            for placeholder in _artifact_template(
                template,
                name="selected_job_artifact",
            ).values()
        )
    )
    if set(bindings) != set(required_placeholders):
        missing = sorted(set(required_placeholders) - set(bindings))
        unexpected = sorted(set(bindings) - set(required_placeholders))
        raise ValueError(
            "bindings must exactly replace selected-job placeholders; "
            f"missing={missing}, unexpected={unexpected}"
        )
    dependencies = {
        name: _resolve_artifact(
            template,
            bindings=bindings,
            name=name,
        )
        for name, template in sorted(templates.items())
    }

    resolved: dict[str, Any] = {
        "schema_name": "p05.materialized_cpu_arm_job",
        "schema_version": 1,
        "paper_id": "P05",
        "protocol_id": matrix["protocol_id"],
        "matrix_id": matrix["matrix_id"],
        "job_id": job["id"],
        "arm": job["arm"],
        "role": arm["role"],
        "dataset": job["dataset"],
        "dataset_id": dataset["dataset_id"],
        "num_classes": dataset["num_classes"],
        "device_class": "cpu",
        "implementation_callable": arm["implementation_callable"],
        "fit_axis": arm["fit_axis"],
        "dependencies": dependencies,
        "output": {
            "path": job["output_dir"],
            "kind": arm["output_kind"],
            "collision_policy": "atomic_create_only_hard_error",
            "execution_status": "not_started",
            "evidence_status": "unadjudicated",
            "claim_support": "forbidden_before_ledger_and_audit",
        },
        "runtime": {
            "conda_environment": "LQ_signal",
            "required_command_prefix": list(COMMAND_PREFIX),
            "gpu_use": "forbidden",
            "network_use": "forbidden",
        },
    }
    if job["arm"] == "P05-B2":
        resolved["corresponding_b0_model_seed"] = job["b0_model_seed"]
    else:
        resolved["fit_identity"] = "dataset_only_no_model_seed"
        resolved["deterministic_fcm_initialization_seed"] = 20260801
        resolved["model_seed_axis"] = "forbidden"
        resolved["model_seed_repetition"] = (
            "forbidden_as_redundant_deterministic_fit"
        )
    return resolved, required_placeholders


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
        raise RuntimeError("atomic P05 CPU materialization requires Linux renameat2")
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


def _write_bytes(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def materialize_p05_cpu_arm_job(
    *,
    job_id: str,
    bindings: Mapping[str, str],
    output_package: str | Path,
    matrix_path: str | Path = MATRIX_PATH,
) -> dict[str, Any]:
    """Bind one selected CPU job to verified artifacts without executing it."""

    target = Path(os.path.abspath(os.fspath(output_package)))
    if target.is_symlink() or target.exists():
        raise FileExistsError(f"P05 CPU materialization target already exists: {target}")
    if not isinstance(bindings, Mapping):
        raise TypeError("bindings must be a mapping")
    normalized_bindings = dict(bindings)
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in normalized_bindings.items()
    ):
        raise TypeError("bindings keys and values must be strings")

    matrix_file = Path(matrix_path).resolve(strict=True)
    matrix = _load_yaml(matrix_file)
    _validate_matrix(matrix)
    job = _selected_job(matrix, job_id)
    resolved_job, required_placeholders = _resolve_job(
        matrix,
        job,
        normalized_bindings,
    )
    job_bytes = yaml.safe_dump(
        resolved_job,
        sort_keys=False,
        allow_unicode=False,
    ).encode("utf-8")
    job_hash = _sha256_bytes(job_bytes)
    semantic_manifest = {
        "schema_name": "p05.materialized_cpu_arm_job_manifest",
        "schema_version": 1,
        "paper_id": "P05",
        "protocol_id": matrix["protocol_id"],
        "matrix_id": matrix["matrix_id"],
        "job_id": job_id,
        "arm": job["arm"],
        "dataset": job["dataset"],
        "matrix_sha256": _sha256_file(matrix_file),
        "bindings_sha256": _sha256_bytes(
            _canonical_json_bytes(normalized_bindings)
        ),
        "bound_placeholder_count": len(required_placeholders),
        "job_file": JOB_NAME,
        "job_sha256": job_hash,
        "materialization_status": "created_not_executed",
        "execution_status": "not_started",
        "evidence_status": "unadjudicated",
        "claim_support": "forbidden_before_ledger_and_audit",
        "output_collision": "atomic_create_only_hard_error",
    }
    semantic_hash = _sha256_bytes(_canonical_json_bytes(semantic_manifest))
    manifest = {
        **semantic_manifest,
        "content": {"semantic_sha256": semantic_hash},
    }
    manifest_bytes = _pretty_json_bytes(manifest)

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"P05 CPU materialization parent must be real: {parent}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=parent)
    )
    try:
        _write_bytes(temporary / JOB_NAME, job_bytes)
        _write_bytes(temporary / MANIFEST_NAME, manifest_bytes)
        _fsync_directory(temporary)
        _rename_directory_noreplace(temporary, target)
        _fsync_directory(parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "package_dir": str(target),
        "job_path": str(target / JOB_NAME),
        "job_sha256": job_hash,
        "manifest_path": str(target / MANIFEST_NAME),
        "manifest_sha256": _sha256_file(target / MANIFEST_NAME),
        "semantic_sha256": semantic_hash,
        "job_id": job_id,
        "arm": job["arm"],
        "dataset": job["dataset"],
        "status": "created_not_executed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--bindings-json", required=True)
    parser.add_argument("--output-package", required=True)
    parser.add_argument("--matrix-path", default=str(MATRIX_PATH))
    args = parser.parse_args()
    bindings_path = Path(args.bindings_json).resolve(strict=True)
    result = materialize_p05_cpu_arm_job(
        job_id=args.job_id,
        bindings=_load_bindings(bindings_path),
        output_package=args.output_package,
        matrix_path=args.matrix_path,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
