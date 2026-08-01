"""Strict create-only serialization for canonical P05 run artifacts."""

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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data_factory.p05_weighting import WeightPlan
from src.data_factory.protocol_transforms import ChannelStandardizationPlan


SCHEMA_NAME = "p05.run_artifact_bundle"
SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_WEIGHT_PLAN_KEYS = frozenset({"train", "val"})
_RUNTIME_FIELDS = frozenset(
    {
        "schema_version",
        "paper_id",
        "evidence_mode",
        "cuda_visible_devices",
        "physical_gpu_index",
        "gpu_uuid",
        "expected_gpu_uuid",
        "identity_source",
        "accelerator",
        "devices",
        "gpus",
        "strategy",
        "precision",
        "deterministic",
    }
)


@dataclass(frozen=True)
class P05RunArtifactExportResult:
    """Paths, hashes, and terminal state for one run-artifact export."""

    package_dir: Path
    manifest_path: Path
    semantic_sha256: str
    manifest_sha256: str
    status: str


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


def _required_sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _require_exact_int(value: Any, *, name: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _normalization_payload(plan: ChannelStandardizationPlan) -> dict[str, Any]:
    if not isinstance(plan, ChannelStandardizationPlan):
        raise TypeError("normalization_plan must be a ChannelStandardizationPlan")
    dataset_id = _require_exact_int(plan.dataset_id, name="normalization_plan.dataset_id")
    if dataset_id not in {1, 2}:
        raise ValueError("normalization_plan.dataset_id must be 1 or 2 for P05")

    if not isinstance(plan.channel_names, tuple) or not plan.channel_names:
        raise TypeError("normalization_plan.channel_names must be a non-empty tuple")
    channels = list(plan.channel_names)
    if any(not isinstance(name, str) or not name for name in channels):
        raise ValueError("normalization_plan.channel_names must contain non-empty strings")
    if len(channels) != len(set(channels)):
        raise ValueError("normalization_plan.channel_names must be unique")
    if not isinstance(plan.mean, tuple) or not isinstance(plan.std, tuple):
        raise TypeError("normalization_plan mean and std must be tuples")
    if len(plan.mean) != len(channels) or len(plan.std) != len(channels):
        raise ValueError("normalization_plan mean/std lengths must match channel_names")
    try:
        mean = [float(value) for value in plan.mean]
        std = [float(value) for value in plan.std]
    except (TypeError, ValueError) as exc:
        raise TypeError("normalization_plan mean/std must be real numeric values") from exc
    if not all(math.isfinite(value) for value in (*mean, *std)):
        raise ValueError("normalization_plan mean/std must be finite")
    if any(value < 1.0e-8 for value in std):
        raise ValueError("normalization_plan std values must be at least 1e-8")

    if not isinstance(plan.group_window_counts, Mapping) or not plan.group_window_counts:
        raise TypeError("normalization_plan.group_window_counts must be a non-empty mapping")
    group_counts: dict[str, int] = {}
    for group, count in plan.group_window_counts.items():
        if not isinstance(group, str) or not group:
            raise ValueError("normalization_plan group IDs must be non-empty strings")
        group_counts[group] = _require_exact_int(
            count,
            name=f"normalization_plan.group_window_counts[{group!r}]",
            minimum=1,
        )
    window_size = _require_exact_int(
        plan.window_size,
        name="normalization_plan.window_size",
        minimum=1,
    )
    window_count = _require_exact_int(
        plan.window_count,
        name="normalization_plan.window_count",
        minimum=1,
    )
    if window_count != sum(group_counts.values()):
        raise ValueError(
            "normalization_plan.window_count does not match group_window_counts"
        )

    contract = {
        "schema_version": 1,
        "paper_id": "P05",
        "dataset_id": dataset_id,
        "fit_role": "train",
        "method": "equal_group_equal_window_equal_point_population_standardization",
        "accumulator_dtype": "float64",
        "channel_names": channels,
        "mean": mean,
        "std": std,
        "group_window_counts": {
            group: group_counts[group] for group in sorted(group_counts)
        },
        "window_size": window_size,
        "window_count": window_count,
    }
    expected_hash = _sha256_bytes(_canonical_json_bytes(contract))
    observed_hash = _required_sha256(
        plan.sha256,
        name="normalization_plan.sha256",
    )
    if observed_hash != expected_hash:
        raise ValueError("normalization_plan source SHA-256 does not match its contract")
    return {**contract, "sha256": expected_hash}


def _record_id(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer record ID")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be an integer record ID") from exc
    if isinstance(value, float) and (not math.isfinite(value) or value != converted):
        raise TypeError(f"{name} must be an integer record ID")
    if isinstance(value, str) and value != str(converted):
        raise TypeError(f"{name} must use canonical integer text")
    return converted


def _weight_plan_payload(
    plan: WeightPlan,
    *,
    key: str,
    expected_role: str,
    dataset_id: int,
) -> dict[str, Any]:
    if not isinstance(plan, WeightPlan):
        raise TypeError(f"weight_plans[{key!r}] must be a WeightPlan")
    plan_dataset_id = _require_exact_int(
        plan.dataset_id,
        name=f"weight_plans[{key!r}].dataset_id",
    )
    if plan_dataset_id != dataset_id:
        raise ValueError(f"weight_plans[{key!r}] dataset_id conflicts with normalization")
    if plan.role != expected_role:
        raise ValueError(
            f"weight_plans[{key!r}].role must be {expected_role!r}, got {plan.role!r}"
        )
    windows_per_record = _require_exact_int(
        plan.windows_per_record,
        name=f"weight_plans[{key!r}].windows_per_record",
        minimum=1,
    )
    if not isinstance(plan.formula, str) or not plan.formula:
        raise ValueError(f"weight_plans[{key!r}].formula must be a non-empty string")
    if not isinstance(plan.record_weights, Mapping) or not plan.record_weights:
        raise TypeError(f"weight_plans[{key!r}].record_weights must be a non-empty mapping")

    converted_weights: dict[int, float] = {}
    for raw_record_id, raw_weight in plan.record_weights.items():
        record_id = _record_id(
            raw_record_id,
            name=f"weight_plans[{key!r}].record_weights key",
        )
        if record_id in converted_weights:
            raise ValueError(f"weight_plans[{key!r}] contains duplicate canonical record IDs")
        if isinstance(raw_weight, bool):
            raise TypeError(f"weight_plans[{key!r}] weights must be real numbers")
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"weight_plans[{key!r}] weights must be real numbers") from exc
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"weight_plans[{key!r}] weights must be finite and positive")
        converted_weights[record_id] = weight
    mean_weight = math.fsum(converted_weights.values()) / len(converted_weights)
    if not math.isclose(mean_weight, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"weight_plans[{key!r}] weights must have mean one")

    rows = [
        {"Id": record_id, "window_weight": converted_weights[record_id]}
        for record_id in sorted(converted_weights)
    ]
    contract = {
        "schema_version": 1,
        "paper_id": "P05",
        "dataset_id": plan_dataset_id,
        "role": expected_role,
        "windows_per_record": windows_per_record,
        "formula": plan.formula,
        "normalization": "mean_train_or_evaluation_window_weight_equals_one",
        "record_weights": rows,
    }
    expected_hash = _sha256_bytes(_canonical_json_bytes(contract))
    observed_hash = _required_sha256(
        plan.sha256,
        name=f"weight_plans[{key!r}].sha256",
    )
    if observed_hash != expected_hash:
        raise ValueError(
            f"weight_plans[{key!r}] source SHA-256 does not match its contract"
        )
    return {**contract, "sha256": expected_hash}


def _runtime_payload(runtime_identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(runtime_identity, Mapping):
        raise TypeError("runtime_identity must be a mapping")
    actual_fields = set(runtime_identity)
    if actual_fields != _RUNTIME_FIELDS:
        missing = sorted(_RUNTIME_FIELDS - actual_fields)
        unexpected = sorted(actual_fields - _RUNTIME_FIELDS, key=str)
        raise ValueError(
            "runtime_identity fields do not match the frozen P05 contract: "
            f"missing={missing}, unexpected={unexpected}"
        )
    runtime = dict(runtime_identity)
    exact_values = {
        "schema_version": 1,
        "paper_id": "P05",
        "evidence_mode": True,
        "identity_source": "nvidia-smi:index,uuid",
        "accelerator": "gpu",
        "devices": 1,
        "gpus": 1,
        "strategy": "auto",
        "precision": 32,
        "deterministic": True,
    }
    for name, expected in exact_values.items():
        value = runtime[name]
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"runtime_identity[{name!r}] must be exactly {expected!r}")
    visible = runtime["cuda_visible_devices"]
    physical_index = runtime["physical_gpu_index"]
    if type(visible) is not str or visible not in {"0", "1"}:
        raise ValueError("runtime_identity['cuda_visible_devices'] must be '0' or '1'")
    if type(physical_index) is not int or physical_index != int(visible):
        raise ValueError(
            "runtime_identity physical_gpu_index must match cuda_visible_devices"
        )
    gpu_uuid = runtime["gpu_uuid"]
    expected_gpu_uuid = runtime["expected_gpu_uuid"]
    if not isinstance(gpu_uuid, str) or not gpu_uuid:
        raise ValueError("runtime_identity['gpu_uuid'] must be a non-empty string")
    if not isinstance(expected_gpu_uuid, str) or expected_gpu_uuid != gpu_uuid:
        raise ValueError("runtime_identity expected_gpu_uuid must match gpu_uuid")
    _canonical_json_bytes(runtime)
    return runtime


def _semantic_manifest(
    normalization_plan: ChannelStandardizationPlan,
    weight_plans: Mapping[str, WeightPlan],
    runtime_identity: Mapping[str, Any],
    *,
    config_sha256: str,
    model_sha256: str,
    checkpoint_sha256: str,
    code_sha256: str,
) -> dict[str, Any]:
    if not isinstance(weight_plans, Mapping):
        raise TypeError("weight_plans must be a mapping")
    actual_plan_keys = set(weight_plans)
    if actual_plan_keys != _WEIGHT_PLAN_KEYS:
        missing = sorted(_WEIGHT_PLAN_KEYS - actual_plan_keys)
        unexpected = sorted(actual_plan_keys - _WEIGHT_PLAN_KEYS, key=str)
        raise ValueError(
            "weight_plans must contain exactly train and val: "
            f"missing={missing}, unexpected={unexpected}"
        )
    normalization = _normalization_payload(normalization_plan)
    dataset_id = int(normalization["dataset_id"])
    weights = {
        "train": _weight_plan_payload(
            weight_plans["train"],
            key="train",
            expected_role="train",
            dataset_id=dataset_id,
        ),
        "validation": _weight_plan_payload(
            weight_plans["val"],
            key="val",
            expected_role="validation",
            dataset_id=dataset_id,
        ),
    }
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "paper_id": "P05",
        "dataset_id": dataset_id,
        "normalization_plan": normalization,
        "weight_plans": weights,
        "runtime_identity": _runtime_payload(runtime_identity),
        "provenance": {
            "checkpoint_sha256": checkpoint_sha256,
            "code_sha256": code_sha256,
            "config_sha256": config_sha256,
            "model_sha256": model_sha256,
        },
    }


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise FileExistsError(f"existing P05 run-artifact manifest is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise FileExistsError(f"existing P05 run-artifact manifest is not an object: {path}")
    return value


def _result(
    target: Path,
    manifest: Mapping[str, Any],
    *,
    status: str,
) -> P05RunArtifactExportResult:
    manifest_path = target / MANIFEST_NAME
    return P05RunArtifactExportResult(
        package_dir=target,
        manifest_path=manifest_path,
        semantic_sha256=str(manifest["content"]["semantic_sha256"]),
        manifest_sha256=_sha256_file(manifest_path),
        status=status,
    )


def _reuse_existing(
    target: Path,
    semantic_manifest: Mapping[str, Any],
) -> P05RunArtifactExportResult:
    if target.is_symlink():
        raise FileExistsError(
            f"refusing create-only P05 run-artifact export through symlink: {target}"
        )
    if not target.is_dir():
        raise FileExistsError(
            f"P05 run-artifact target exists and is not a directory: {target}"
        )
    entries = {entry.name: entry for entry in target.iterdir()}
    if set(entries) != {MANIFEST_NAME}:
        raise FileExistsError(
            f"existing P05 run-artifact package has unexpected or incomplete content: {target}"
        )
    manifest_path = entries[MANIFEST_NAME]
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileExistsError(
            f"existing P05 run-artifact manifest is not a regular file: {manifest_path}"
        )
    manifest = _load_manifest(manifest_path)
    if set(manifest) != set(semantic_manifest) | {"content"}:
        raise FileExistsError(
            f"existing P05 run-artifact schema conflicts with proposed export: {target}"
        )
    content = manifest.get("content")
    if not isinstance(content, dict) or set(content) != {"semantic_sha256"}:
        raise FileExistsError(
            f"existing P05 run-artifact content hash is invalid: {target}"
        )
    try:
        recorded_hash = _required_sha256(
            content["semantic_sha256"],
            name="existing content.semantic_sha256",
        )
    except ValueError as exc:
        raise FileExistsError(
            f"existing P05 run-artifact content hash is invalid: {target}"
        ) from exc
    existing_semantic = {key: value for key, value in manifest.items() if key != "content"}
    actual_hash = _sha256_bytes(_canonical_json_bytes(existing_semantic))
    if recorded_hash != actual_hash:
        raise FileExistsError(
            f"existing P05 run-artifact semantic hash does not match its manifest: {target}"
        )
    if _canonical_json_bytes(existing_semantic) != _canonical_json_bytes(semantic_manifest):
        raise FileExistsError(
            f"existing P05 run-artifact content conflicts with proposed export: {target}"
        )
    return _result(target, manifest, status="reused")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically install a directory without an overwrite-capable fallback."""

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
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), str(target))
    raise OSError(error_number, os.strerror(error_number), str(target))


def _write_manifest_file(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _write_new_package(
    target: Path,
    semantic_manifest: Mapping[str, Any],
) -> P05RunArtifactExportResult:
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"P05 run-artifact parent must be a real directory: {parent}")
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(parent),
        )
    )
    try:
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
        _write_manifest_file(temporary / MANIFEST_NAME, manifest_bytes)
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


def export_p05_run_artifact_bundle(
    package_dir: str | Path,
    *,
    normalization_plan: ChannelStandardizationPlan,
    weight_plans: Mapping[str, WeightPlan],
    runtime_identity: Mapping[str, Any],
    config_sha256: str,
    model_sha256: str,
    checkpoint_sha256: str,
    code_sha256: str,
) -> P05RunArtifactExportResult:
    """Validate and atomically create or semantically reuse one P05 bundle.

    The input plans are already computed by the data protocol. Their canonical
    source contracts are reconstructed here and their stored digests must match
    before any filesystem mutation occurs.
    """

    hashes = {
        "config_sha256": _required_sha256(config_sha256, name="config_sha256"),
        "model_sha256": _required_sha256(model_sha256, name="model_sha256"),
        "checkpoint_sha256": _required_sha256(
            checkpoint_sha256,
            name="checkpoint_sha256",
        ),
        "code_sha256": _required_sha256(code_sha256, name="code_sha256"),
    }
    semantic_manifest = _semantic_manifest(
        normalization_plan,
        weight_plans,
        runtime_identity,
        **hashes,
    )
    target = Path(os.path.abspath(os.fspath(package_dir)))
    if target.is_symlink():
        raise FileExistsError(
            f"refusing create-only P05 run-artifact export through symlink: {target}"
        )
    if target.exists():
        return _reuse_existing(target, semantic_manifest)
    return _write_new_package(target, semantic_manifest)


__all__ = [
    "P05RunArtifactExportResult",
    "export_p05_run_artifact_bundle",
]
