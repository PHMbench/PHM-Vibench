"""Create-only P05 per-window feature/logit prediction artifacts."""

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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA_NAME = "p05.window_predictions"
SCHEMA_VERSION = 1
ARRAYS_NAME = "prediction_arrays.npz"
MANIFEST_NAME = "manifest.json"
SPLIT_ORDER = ("train", "val", "test")

_FLOAT32 = np.dtype("<f4")
_FLOAT64 = np.dtype("<f8")
_INT64 = np.dtype("<i8")
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_ARRAY_NAMES = {
    "split",
    "sample_id",
    "record_id",
    "group_id",
    "window_index",
    "window_start",
    "window_end",
    "y",
    "sample_weight",
    "reduced_features",
    "logits",
}
_IDENTIFIER_ARRAY_NAMES = {"split", "sample_id", "record_id", "group_id"}


@dataclass(frozen=True)
class P05PredictionBatch:
    """One prediction batch with complete immutable window provenance."""

    split: str
    sample_id: Sequence[str]
    record_id: Sequence[str]
    group_id: Sequence[str]
    window_index: Any
    window_start: Any
    window_end: Any
    y: Any
    sample_weight: Any
    reduced_features: Any
    logits: Any


@dataclass(frozen=True)
class P05PredictionExportResult:
    package_dir: Path
    arrays_path: Path
    manifest_path: Path
    semantic_sha256: str
    arrays_sha256: str
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


def _required_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _identifier_array(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a one-dimensional string sequence")
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    identifiers = raw.tolist()
    for item in identifiers:
        if not isinstance(item, str) or not item.strip() or "\x00" in item:
            raise ValueError(f"{name} entries must be non-empty strings without NUL bytes")
    width = max(len(item) for item in identifiers)
    return np.ascontiguousarray(np.asarray(identifiers, dtype=f"<U{width}"))


def _integer_array(value: Any, *, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        if value.dtype == torch.bool or value.dtype.is_floating_point or value.is_complex():
            raise TypeError(f"{name} must contain integers")
        array = value.detach().to(device="cpu").numpy()
    else:
        array = np.asarray(value)
        if array.dtype.kind not in {"i", "u"}:
            raise TypeError(f"{name} must contain integers")
    return np.ascontiguousarray(array, dtype=_INT64)


def _float32_array(value: Any, *, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        if value.dtype != torch.float32:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        array = value.detach().to(device="cpu").numpy()
    else:
        array = np.asarray(value)
        if array.dtype != np.dtype(np.float32):
            raise TypeError(f"{name} must be float32, got {array.dtype}")
    result = np.ascontiguousarray(array, dtype=_FLOAT32)
    if not np.isfinite(result).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return result


def _weight_array(value: Any, *, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        if value.dtype == torch.bool or not value.dtype.is_floating_point:
            raise TypeError(f"{name} must be floating point")
        array = value.detach().to(device="cpu", dtype=torch.float64).numpy()
    else:
        array = np.asarray(value)
        if array.dtype.kind != "f":
            raise TypeError(f"{name} must be floating point")
    result = np.ascontiguousarray(array, dtype=_FLOAT64)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError(f"{name} must contain finite positive weights")
    return result


def _expect_vector(array: np.ndarray, *, count: int, name: str) -> None:
    if tuple(array.shape) != (count,):
        raise ValueError(f"{name} must have shape ({count},), got {tuple(array.shape)}")


def _normalise_batch(
    batch: P05PredictionBatch,
    *,
    batch_index: int,
    expected_window_size: int,
) -> dict[str, np.ndarray]:
    if not isinstance(batch, P05PredictionBatch):
        raise TypeError("batches must contain P05PredictionBatch instances")
    if batch.split not in SPLIT_ORDER:
        raise ValueError(f"batch[{batch_index}].split must be one of {SPLIT_ORDER}")
    prefix = f"batch[{batch_index}]"
    sample_id = _identifier_array(batch.sample_id, name=f"{prefix}.sample_id")
    count = int(sample_id.shape[0])
    record_id = _identifier_array(batch.record_id, name=f"{prefix}.record_id")
    group_id = _identifier_array(batch.group_id, name=f"{prefix}.group_id")
    window_index = _integer_array(batch.window_index, name=f"{prefix}.window_index")
    window_start = _integer_array(batch.window_start, name=f"{prefix}.window_start")
    window_end = _integer_array(batch.window_end, name=f"{prefix}.window_end")
    targets = _integer_array(batch.y, name=f"{prefix}.y")
    weights = _weight_array(batch.sample_weight, name=f"{prefix}.sample_weight")
    for name, array in (
        ("record_id", record_id),
        ("group_id", group_id),
        ("window_index", window_index),
        ("window_start", window_start),
        ("window_end", window_end),
        ("y", targets),
        ("sample_weight", weights),
    ):
        _expect_vector(array, count=count, name=f"{prefix}.{name}")

    features = _float32_array(
        batch.reduced_features,
        name=f"{prefix}.reduced_features",
    )
    logits = _float32_array(batch.logits, name=f"{prefix}.logits")
    if tuple(features.shape) != (count, 8):
        raise ValueError(
            f"{prefix}.reduced_features must have shape ({count},8), "
            f"got {tuple(features.shape)}"
        )
    if logits.ndim != 2 or logits.shape[0] != count or logits.shape[1] not in {2, 4}:
        raise ValueError(f"{prefix}.logits must have shape (batch,2) or (batch,4)")
    if (targets < 0).any() or (targets >= logits.shape[1]).any():
        raise ValueError(f"{prefix}.y contains an out-of-range class index")
    if (window_index < 0).any() or (window_start < 0).any():
        raise ValueError(f"{prefix} window indices and starts must be non-negative")
    if not np.all(window_end - window_start == expected_window_size):
        raise ValueError(
            f"{prefix} every window must have size {expected_window_size}"
        )
    for index in range(count):
        expected_id = f"{record_id[index]}:{window_start[index]}:{window_end[index]}"
        if sample_id[index] != expected_id:
            raise ValueError(
                f"{prefix}.sample_id[{index}] must equal {expected_id!r}, "
                f"got {sample_id[index]!r}"
            )

    split_width = len(batch.split)
    return {
        "split": np.full(count, batch.split, dtype=f"<U{split_width}"),
        "sample_id": sample_id,
        "record_id": record_id,
        "group_id": group_id,
        "window_index": window_index,
        "window_start": window_start,
        "window_end": window_end,
        "y": targets,
        "sample_weight": weights,
        "reduced_features": features,
        "logits": logits,
    }


def _normalise_expected_records(
    expected_record_ids_by_split: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    if not isinstance(expected_record_ids_by_split, Mapping):
        raise TypeError("expected_record_ids_by_split must be a mapping")
    if set(expected_record_ids_by_split) != set(SPLIT_ORDER):
        raise ValueError(f"expected_record_ids_by_split must have keys {SPLIT_ORDER}")
    result: dict[str, tuple[str, ...]] = {}
    all_records: set[str] = set()
    for split in SPLIT_ORDER:
        array = _identifier_array(
            expected_record_ids_by_split[split],
            name=f"expected_record_ids_by_split[{split!r}]",
        )
        records = tuple(sorted(array.tolist()))
        if len(set(records)) != len(records):
            raise ValueError(f"expected {split} record IDs must be unique")
        overlap = all_records.intersection(records)
        if overlap:
            raise ValueError(
                f"expected record IDs overlap across splits: {sorted(overlap)!r}"
            )
        all_records.update(records)
        result[split] = records
    return result


def _validate_complete_records(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_records: Mapping[str, tuple[str, ...]],
    expected_windows_per_record: int,
) -> None:
    seen_groups: dict[str, str] = {}
    for split in SPLIT_ORDER:
        split_indices = np.flatnonzero(arrays["split"] == split)
        if split_indices.size == 0:
            raise ValueError(f"prediction export split {split!r} is empty")
        observed_records = set(arrays["record_id"][split_indices].tolist())
        if observed_records != set(expected_records[split]):
            missing = sorted(set(expected_records[split]) - observed_records)
            extra = sorted(observed_records - set(expected_records[split]))
            raise ValueError(
                f"{split} record coverage mismatch: missing={missing!r}, extra={extra!r}"
            )
        split_groups = set(arrays["group_id"][split_indices].tolist())
        for group_id in split_groups:
            previous = seen_groups.setdefault(group_id, split)
            if previous != split:
                raise ValueError(
                    f"group_id {group_id!r} overlaps splits {previous!r} and {split!r}"
                )

        split_weights = arrays["sample_weight"][split_indices]
        if not math.isclose(
            math.fsum(float(value) for value in split_weights) / len(split_weights),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"{split} sample weights must have mean one")

        for record_id in expected_records[split]:
            record_indices = split_indices[
                arrays["record_id"][split_indices] == record_id
            ]
            if len(record_indices) != expected_windows_per_record:
                raise ValueError(
                    f"{split} record {record_id!r} must contain exactly "
                    f"{expected_windows_per_record} windows"
                )
            order = record_indices[
                np.argsort(arrays["window_index"][record_indices], kind="stable")
            ]
            expected_indices = np.arange(expected_windows_per_record, dtype=_INT64)
            if not np.array_equal(arrays["window_index"][order], expected_indices):
                raise ValueError(
                    f"{split} record {record_id!r} has incomplete or duplicate window_index"
                )
            if len(set(arrays["group_id"][order].tolist())) != 1:
                raise ValueError(f"{split} record {record_id!r} maps to multiple groups")
            if len(set(arrays["y"][order].tolist())) != 1:
                raise ValueError(f"{split} record {record_id!r} maps to multiple labels")
            if not np.all(arrays["sample_weight"][order] == arrays["sample_weight"][order][0]):
                raise ValueError(f"{split} record {record_id!r} has inconsistent weights")
            starts = arrays["window_start"][order]
            ends = arrays["window_end"][order]
            if not np.all(starts[1:] >= ends[:-1]):
                raise ValueError(f"{split} record {record_id!r} contains overlapping windows")


def _collect_arrays(
    batches: Iterable[P05PredictionBatch],
    *,
    expected_records: Mapping[str, tuple[str, ...]],
    expected_windows_per_record: int,
    expected_window_size: int,
) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {name: [] for name in _ARRAY_NAMES}
    sample_ids_seen: set[str] = set()
    class_count: int | None = None
    batch_count = 0
    for batch_index, batch in enumerate(batches):
        normalised = _normalise_batch(
            batch,
            batch_index=batch_index,
            expected_window_size=expected_window_size,
        )
        observed_classes = int(normalised["logits"].shape[1])
        if class_count is None:
            class_count = observed_classes
        elif observed_classes != class_count:
            raise ValueError("prediction logit class count changed across batches")
        for sample_id in normalised["sample_id"].tolist():
            if sample_id in sample_ids_seen:
                raise ValueError(f"duplicate sample_id across prediction export: {sample_id!r}")
            sample_ids_seen.add(sample_id)
        for name, array in normalised.items():
            collected[name].append(array)
        batch_count += 1
    if batch_count == 0:
        raise ValueError("at least one non-empty P05PredictionBatch is required")

    arrays = {
        name: np.ascontiguousarray(np.concatenate(pieces, axis=0))
        for name, pieces in collected.items()
    }
    order = np.asarray(
        sorted(
            range(len(arrays["sample_id"])),
            key=lambda index: (
                SPLIT_ORDER.index(str(arrays["split"][index])),
                str(arrays["sample_id"][index]),
            ),
        ),
        dtype=_INT64,
    )
    arrays = {name: np.ascontiguousarray(array[order]) for name, array in arrays.items()}
    if set(arrays) != _ARRAY_NAMES:
        raise AssertionError("internal prediction array schema mismatch")
    _validate_complete_records(
        arrays,
        expected_records=expected_records,
        expected_windows_per_record=expected_windows_per_record,
    )
    return arrays


def _array_sha256(array: np.ndarray) -> str:
    descriptor = _canonical_json_bytes(
        {"dtype": array.dtype.str, "shape": [int(value) for value in array.shape]}
    )
    return _sha256_bytes(descriptor + b"\0" + array.tobytes(order="C"))


def _array_descriptors(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "dtype": array.dtype.str,
            "sha256": _array_sha256(array),
            "shape": [int(value) for value in array.shape],
        }
        for name, array in sorted(arrays.items())
    }


def _split_descriptors(
    arrays: Mapping[str, np.ndarray],
    expected_records: Mapping[str, tuple[str, ...]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for split in SPLIT_ORDER:
        indices = np.flatnonzero(arrays["split"] == split)
        result[split] = {
            "group_count": len(set(arrays["group_id"][indices].tolist())),
            "record_count": len(expected_records[split]),
            "record_ids_sha256": _sha256_bytes(
                _canonical_json_bytes(list(expected_records[split]))
            ),
            "sample_count": int(len(indices)),
            "sample_ids_sha256": _sha256_bytes(
                _canonical_json_bytes(arrays["sample_id"][indices].tolist())
            ),
        }
    return result


def _semantic_manifest(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_records: Mapping[str, tuple[str, ...]],
    expected_windows_per_record: int,
    expected_window_size: int,
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "arrays": _array_descriptors(arrays),
        "arrays_file": ARRAYS_NAME,
        "conclusion_control": {
            "claim_decisions": "not_performed",
            "decisive": False,
            "status": "unadjudicated",
        },
        "contract": {
            "feature_count": 8,
            "features_and_logits_dtype": _FLOAT32.str,
            "ordering": "train_then_val_then_test_each_by_stable_sample_id",
            "record_coverage": "exact_expected_record_ids_and_windows",
            "split_order": list(SPLIT_ORDER),
            "window_non_overlap": "required_within_record",
            "window_size": expected_window_size,
            "windows_per_record": expected_windows_per_record,
        },
        "evidence_status": "unadjudicated",
        "format": {
            "container": "numpy.npz",
            "identifier_arrays": sorted(_IDENTIFIER_ARRAY_NAMES),
            "load_allow_pickle": False,
            "object_arrays": False,
        },
        "provenance": dict(provenance),
        "sample_count": int(arrays["sample_id"].shape[0]),
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "splits": _split_descriptors(arrays, expected_records),
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ValueError("prediction NPZ contains duplicate array names")
            arrays = {
                name: np.array(archive[name], copy=True, order="C")
                for name in archive.files
            }
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid create-only prediction NPZ: {path}") from exc
    if set(arrays) != _ARRAY_NAMES:
        raise ValueError("prediction NPZ has an unexpected array schema")
    for name, array in arrays.items():
        if array.dtype.hasobject:
            raise ValueError(f"prediction NPZ array {name!r} requires pickle")
    return arrays


def _arrays_equal(
    existing: Mapping[str, np.ndarray],
    proposed: Mapping[str, np.ndarray],
) -> bool:
    if set(existing) != set(proposed):
        return False
    return all(
        existing[name].dtype == proposed[name].dtype
        and existing[name].shape == proposed[name].shape
        and np.array_equal(existing[name], proposed[name])
        for name in proposed
    )


def _result(
    target: Path,
    manifest: Mapping[str, Any],
    *,
    status: str,
) -> P05PredictionExportResult:
    manifest_path = target / MANIFEST_NAME
    return P05PredictionExportResult(
        package_dir=target,
        arrays_path=target / ARRAYS_NAME,
        manifest_path=manifest_path,
        semantic_sha256=str(manifest["content"]["semantic_sha256"]),
        arrays_sha256=str(manifest["content"]["arrays_sha256"]),
        manifest_sha256=_sha256_file(manifest_path),
        status=status,
    )


def _reuse_existing(
    target: Path,
    arrays: Mapping[str, np.ndarray],
    semantic_manifest: Mapping[str, Any],
) -> P05PredictionExportResult:
    if target.is_symlink() or not target.is_dir():
        raise FileExistsError(
            f"prediction export target conflicts with create-only output: {target}"
        )
    entries = {entry.name: entry for entry in target.iterdir()}
    if set(entries) != {ARRAYS_NAME, MANIFEST_NAME}:
        raise FileExistsError(f"existing prediction package is incomplete or unexpected: {target}")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries.values()):
        raise FileExistsError(f"existing prediction package contains a non-file entry: {target}")
    try:
        manifest = json.loads(entries[MANIFEST_NAME].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FileExistsError(f"existing prediction manifest is invalid: {target}") from exc
    if not isinstance(manifest, dict) or set(manifest) != set(semantic_manifest) | {"content"}:
        raise FileExistsError(f"existing prediction manifest schema conflicts: {target}")
    content = manifest.get("content")
    if not isinstance(content, dict) or set(content) != {"arrays_sha256", "semantic_sha256"}:
        raise FileExistsError(f"existing prediction content hashes are invalid: {target}")
    existing_arrays = _load_npz(entries[ARRAYS_NAME])
    if _sha256_file(entries[ARRAYS_NAME]) != content["arrays_sha256"]:
        raise FileExistsError(f"existing prediction NPZ hash does not match its manifest: {target}")
    existing_semantic = {name: value for name, value in manifest.items() if name != "content"}
    if _sha256_bytes(_canonical_json_bytes(existing_semantic)) != content["semantic_sha256"]:
        raise FileExistsError(f"existing prediction semantic hash is invalid: {target}")
    if existing_semantic.get("arrays") != _array_descriptors(existing_arrays):
        raise FileExistsError(f"existing prediction array hashes are invalid: {target}")
    if _canonical_json_bytes(existing_semantic) != _canonical_json_bytes(semantic_manifest):
        raise FileExistsError(f"existing prediction provenance or contract conflicts: {target}")
    if not _arrays_equal(existing_arrays, arrays):
        raise FileExistsError(f"existing prediction arrays conflict with proposed export: {target}")
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
        raise RuntimeError("atomic create-only prediction export requires Linux renameat2")
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


def _write_new_package(
    target: Path,
    arrays: Mapping[str, np.ndarray],
    semantic_manifest: Mapping[str, Any],
) -> P05PredictionExportResult:
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"prediction export parent must be a real directory: {parent}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(parent))
    )
    try:
        arrays_path = temporary / ARRAYS_NAME
        with arrays_path.open("wb") as handle:
            np.savez(handle, **{name: arrays[name] for name in sorted(arrays)})
            handle.flush()
            os.fsync(handle.fileno())
        arrays_sha256 = _sha256_file(arrays_path)
        semantic_sha256 = _sha256_bytes(_canonical_json_bytes(semantic_manifest))
        manifest = {
            **semantic_manifest,
            "content": {
                "arrays_sha256": arrays_sha256,
                "semantic_sha256": semantic_sha256,
            },
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
        manifest_path = temporary / MANIFEST_NAME
        with manifest_path.open("wb") as handle:
            handle.write(manifest_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary)
        try:
            _rename_directory_noreplace(temporary, target)
        except FileExistsError:
            return _reuse_existing(target, arrays, semantic_manifest)
        _fsync_directory(parent)
        return _result(target, manifest, status="created")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def export_p05_prediction_package(
    package_dir: str | Path,
    batches: Iterable[P05PredictionBatch],
    *,
    expected_record_ids_by_split: Mapping[str, Sequence[str]],
    expected_windows_per_record: int,
    expected_window_size: int,
    config_sha256: str,
    code_sha256: str,
    checkpoint_sha256: str,
    model_sha256: str,
    run_contract_sha256: str,
) -> P05PredictionExportResult:
    """Validate and create or semantically reuse one prediction package."""

    windows_per_record = _positive_int(
        expected_windows_per_record,
        name="expected_windows_per_record",
    )
    window_size = _positive_int(expected_window_size, name="expected_window_size")
    expected_records = _normalise_expected_records(expected_record_ids_by_split)
    provenance = {
        "checkpoint_sha256": _required_sha256(
            checkpoint_sha256,
            name="checkpoint_sha256",
        ),
        "code_sha256": _required_sha256(code_sha256, name="code_sha256"),
        "config_sha256": _required_sha256(config_sha256, name="config_sha256"),
        "model_sha256": _required_sha256(model_sha256, name="model_sha256"),
        "run_contract_sha256": _required_sha256(
            run_contract_sha256,
            name="run_contract_sha256",
        ),
    }
    arrays = _collect_arrays(
        batches,
        expected_records=expected_records,
        expected_windows_per_record=windows_per_record,
        expected_window_size=window_size,
    )
    semantic_manifest = _semantic_manifest(
        arrays,
        expected_records=expected_records,
        expected_windows_per_record=windows_per_record,
        expected_window_size=window_size,
        provenance=provenance,
    )
    target = Path(os.path.abspath(os.fspath(package_dir)))
    if target.is_symlink():
        raise FileExistsError(f"refusing create-only prediction export through symlink: {target}")
    if target.exists():
        return _reuse_existing(target, arrays, semantic_manifest)
    return _write_new_package(target, arrays, semantic_manifest)


__all__ = [
    "ARRAYS_NAME",
    "MANIFEST_NAME",
    "P05PredictionBatch",
    "P05PredictionExportResult",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SPLIT_ORDER",
    "export_p05_prediction_package",
]
