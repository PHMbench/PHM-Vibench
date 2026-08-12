#!/usr/bin/env python3
"""Create one launchable, hash-registered P05 pilot config package."""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config_schema import ExperimentConfig  # noqa: E402


MATRIX_PATH = REPO_ROOT / "configs/experiments/p05/protocol/pilot_matrix_p05_v1.yaml"
CONFIG_NAME = "config.yaml"
MANIFEST_NAME = "manifest.json"


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


def _load_yaml(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required P05 protocol YAML must be a real file: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"required P05 protocol YAML must contain a mapping: {path}")
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _required_gpu_uuid(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("GPU-")
        or len(value) > 128
        or any(character.isspace() or not character.isprintable() for character in value)
        or "REQUIRED" in value
    ):
        raise ValueError("gpu_uuid must be an observed printable NVIDIA GPU-* UUID")
    return value


def _resolve_job(
    matrix: Mapping[str, Any],
    launch_plan: Mapping[str, Any],
    *,
    job_id: str,
    gpu_uuid: str,
) -> tuple[dict[str, Any], int]:
    jobs = [job for job in matrix.get("jobs", []) if job.get("id") == job_id]
    if len(jobs) != 1:
        raise ValueError(f"job_id must identify exactly one frozen pilot job: {job_id!r}")
    job = jobs[0]
    arm = matrix["arms"][job["arm"]]
    dataset = matrix["datasets"][job["dataset"]]
    config = _deep_merge(matrix["common_config"], dataset["config"])
    config = _deep_merge(config, arm["config"])
    config = _deep_merge(config, job["config"])

    assignments = {
        item["job_id"]: int(item["physical_gpu_index"])
        for wave in launch_plan["execution_waves"]
        for item in wave["concurrent_jobs"]
    }
    if set(assignments) != {entry["id"] for entry in matrix["jobs"]}:
        raise ValueError("pilot launch plan does not cover the frozen job matrix exactly")
    physical_index = assignments[job_id]
    if physical_index not in {0, 1}:
        raise ValueError("pilot launch plan assigned a forbidden physical GPU")
    config["trainer"]["expected_gpu_uuid"] = _required_gpu_uuid(gpu_uuid)
    ExperimentConfig.model_validate(config, strict=True)
    return config, physical_index


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
        raise RuntimeError("atomic P05 pilot materialization requires Linux renameat2")
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


def materialize_p05_pilot_job(
    *,
    job_id: str,
    gpu_uuid: str,
    output_package: str | Path,
    matrix_path: str | Path = MATRIX_PATH,
) -> dict[str, Any]:
    """Materialize one frozen job, changing only the launch-time GPU UUID."""

    matrix_file = Path(matrix_path).resolve(strict=True)
    matrix = _load_yaml(matrix_file)
    if matrix.get("status") != "frozen_declarative" or matrix.get("evidence_eligible") is not False:
        raise ValueError("pilot matrix is not in the frozen non-evidence state")
    launch_relative = matrix["launch_gate"]["launch_plan_path"]
    launch_file = (REPO_ROOT / launch_relative).resolve(strict=True)
    launch_plan = _load_yaml(launch_file)
    if launch_plan.get("status") != "frozen_awaiting_physical_gpu_uuid_binding":
        raise ValueError("pilot launch plan is not frozen at the UUID-binding gate")
    config, physical_index = _resolve_job(
        matrix,
        launch_plan,
        job_id=job_id,
        gpu_uuid=gpu_uuid,
    )

    config_bytes = yaml.safe_dump(
        config,
        sort_keys=False,
        allow_unicode=False,
    ).encode("utf-8")
    config_hash = _sha256_bytes(config_bytes)
    semantic_manifest = {
        "schema_name": "p05.materialized_pilot_config",
        "schema_version": 1,
        "paper_id": "P05",
        "matrix_id": matrix["matrix_id"],
        "job_id": job_id,
        "physical_gpu_index": physical_index,
        "expected_gpu_uuid": gpu_uuid,
        "config_file": CONFIG_NAME,
        "config_sha256": config_hash,
        "matrix_sha256": _sha256_file(matrix_file),
        "launch_plan_sha256": _sha256_file(launch_file),
        "evidence_eligible": False,
        "claim_support": "forbidden",
        "scientific_overrides": "forbidden",
    }
    semantic_hash = _sha256_bytes(_canonical_json_bytes(semantic_manifest))
    manifest = {
        **semantic_manifest,
        "content": {"semantic_sha256": semantic_hash},
    }
    manifest_bytes = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    target = Path(os.path.abspath(os.fspath(output_package)))
    if target.is_symlink() or target.exists():
        raise FileExistsError(f"P05 pilot materialization target already exists: {target}")
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"P05 pilot materialization parent must be real: {parent}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(parent))
    )
    try:
        for name, payload in ((CONFIG_NAME, config_bytes), (MANIFEST_NAME, manifest_bytes)):
            with (temporary / name).open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        _fsync_directory(temporary)
        _rename_directory_noreplace(temporary, target)
        _fsync_directory(parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "package_dir": str(target),
        "config_path": str(target / CONFIG_NAME),
        "config_sha256": config_hash,
        "manifest_path": str(target / MANIFEST_NAME),
        "manifest_sha256": _sha256_file(target / MANIFEST_NAME),
        "semantic_sha256": semantic_hash,
        "job_id": job_id,
        "physical_gpu_index": physical_index,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output-package", required=True)
    parser.add_argument("--matrix-path", default=str(MATRIX_PATH))
    args = parser.parse_args()
    result = materialize_p05_pilot_job(
        job_id=args.job_id,
        gpu_uuid=args.gpu_uuid,
        output_package=args.output_package,
        matrix_path=args.matrix_path,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
