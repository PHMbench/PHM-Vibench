"""Create-only, auditable NPZ exports for complete P05 fuzzy traces."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
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


SCHEMA_NAME = "p05.complete_fuzzy_trace"
SCHEMA_VERSION = 1
NPZ_NAME = "trace_arrays.npz"
MANIFEST_NAME = "manifest.json"
DEFAULT_RECONSTRUCTION_ATOL = 1.0e-6
DEFAULT_RECONSTRUCTION_RTOL = 1.0e-6

_FLOAT64 = np.dtype("<f8")
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")

_BATCH_TRACE_FIELDS = (
    "reduced_features",
    "membership_values",
    "antecedent_memberships",
    "log_rule_firing",
    "rule_firing",
    "normalized_rule_firing",
    "rule_contributions",
    "fuzzy_logits",
    "rule_mask",
)
_SHARED_TRACE_FIELDS = (
    "centers",
    "widths",
    "antecedent_probabilities",
    "rule_consequents",
    "consequent_permutation",
)
_BATCH_ARRAY_NAMES = {
    "sample_id",
    "record_id",
    "group_id",
    "window_start",
    "window_end",
    "y",
    "logits",
    "non_fuzzy_logits",
    *(f"trace_{name}" for name in _BATCH_TRACE_FIELDS),
}
_SHARED_ARRAY_NAMES = {
    "fuzzy_scale",
    *(f"trace_{name}" for name in _SHARED_TRACE_FIELDS),
}
_IDENTIFIER_ARRAY_NAMES = {"sample_id", "record_id", "group_id"}


@dataclass(frozen=True)
class P05TraceBatch:
    """One batch of complete fuzzy traces plus stable sample provenance."""

    sample_id: Sequence[str]
    record_id: Sequence[str]
    group_id: Sequence[str]
    window_start: Any
    window_end: Any
    y: Any
    logits: Any
    non_fuzzy_logits: Any
    fuzzy_scale: Any
    fuzzy_trace: Any


@dataclass(frozen=True)
class P05TraceExportResult:
    package_dir: Path
    npz_path: Path
    manifest_path: Path
    semantic_sha256: str
    npz_sha256: str
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


def _required_field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise ValueError(f"complete FuzzyTrace is missing field {name!r}")
        return value[name]
    if not hasattr(value, name):
        raise ValueError(f"complete FuzzyTrace is missing field {name!r}")
    return getattr(value, name)


def _numeric_array(value: Any, *, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        if value.is_complex():
            raise ValueError(f"{name} must be real-valued")
        array = value.detach().to(device="cpu", dtype=torch.float64).numpy()
    else:
        array = np.asarray(value)
        if array.dtype.kind not in {"b", "i", "u", "f"}:
            raise ValueError(f"{name} must be a real numeric array")
        array = array.astype(_FLOAT64, copy=False)
    array = np.ascontiguousarray(array, dtype=_FLOAT64)
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return array


def _identifier_array(value: Sequence[str], *, name: str) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a one-dimensional sequence of strings")
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must have shape (batch,), got {tuple(raw.shape)}")
    identifiers = raw.tolist()
    if not identifiers:
        raise ValueError(f"{name} must not be empty")
    for item in identifiers:
        if not isinstance(item, str) or not item.strip() or "\x00" in item:
            raise ValueError(f"{name} entries must be non-empty strings without NUL bytes")
    width = max(len(item) for item in identifiers)
    return np.ascontiguousarray(np.asarray(identifiers, dtype=f"<U{width}"))


def _expect_shape(array: np.ndarray, shape: tuple[int, ...], *, name: str) -> None:
    if tuple(array.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(array.shape)}")


def _assert_integer_values(array: np.ndarray, *, name: str) -> None:
    if not np.array_equal(array, np.round(array)):
        raise ValueError(f"{name} must contain integer values")


def _assert_per_sample_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    sample_ids: np.ndarray,
    identity: str,
    atol: float,
    rtol: float,
) -> None:
    close = np.isclose(actual, expected, atol=atol, rtol=rtol)
    per_sample = close.reshape(close.shape[0], -1).all(axis=1)
    if per_sample.all():
        return
    index = int(np.flatnonzero(~per_sample)[0])
    max_error = float(np.max(np.abs(actual[index] - expected[index])))
    raise ValueError(
        f"trace reconstruction {identity} failed for sample_id={sample_ids[index]!r}: "
        f"max_abs_error={max_error:.12g}, atol={atol:.3g}, rtol={rtol:.3g}"
    )


def _normalise_batch(
    batch: P05TraceBatch,
    *,
    batch_index: int,
    atol: float,
    rtol: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    prefix = f"batch[{batch_index}]"
    sample_id = _identifier_array(batch.sample_id, name=f"{prefix}.sample_id")
    batch_size = int(sample_id.shape[0])
    record_id = _identifier_array(batch.record_id, name=f"{prefix}.record_id")
    group_id = _identifier_array(batch.group_id, name=f"{prefix}.group_id")
    _expect_shape(record_id, (batch_size,), name=f"{prefix}.record_id")
    _expect_shape(group_id, (batch_size,), name=f"{prefix}.group_id")

    window_start = _numeric_array(batch.window_start, name=f"{prefix}.window_start")
    window_end = _numeric_array(batch.window_end, name=f"{prefix}.window_end")
    targets = _numeric_array(batch.y, name=f"{prefix}.y")
    for name, value in (
        ("window_start", window_start),
        ("window_end", window_end),
        ("y", targets),
    ):
        _expect_shape(value, (batch_size,), name=f"{prefix}.{name}")
        _assert_integer_values(value, name=f"{prefix}.{name}")
    if (window_start < 0).any() or (window_end <= window_start).any():
        raise ValueError(f"{prefix} requires 0 <= window_start < window_end")

    logits = _numeric_array(batch.logits, name=f"{prefix}.logits")
    non_fuzzy_logits = _numeric_array(
        batch.non_fuzzy_logits,
        name=f"{prefix}.non_fuzzy_logits",
    )
    if logits.ndim != 2 or logits.shape[0] != batch_size or logits.shape[1] < 2:
        raise ValueError(f"{prefix}.logits must have shape (batch, classes>=2)")
    _expect_shape(
        non_fuzzy_logits,
        tuple(logits.shape),
        name=f"{prefix}.non_fuzzy_logits",
    )
    num_classes = int(logits.shape[1])
    if (targets < 0).any() or (targets >= num_classes).any():
        raise ValueError(f"{prefix}.y contains an out-of-range class index")

    scale_array = _numeric_array(batch.fuzzy_scale, name=f"{prefix}.fuzzy_scale")
    if scale_array.size != 1:
        raise ValueError(f"{prefix}.fuzzy_scale must be scalar")
    fuzzy_scale = np.asarray(scale_array.reshape(()), dtype=_FLOAT64)

    trace = batch.fuzzy_trace
    trace_arrays = {
        name: _numeric_array(
            _required_field(trace, name),
            name=f"{prefix}.fuzzy_trace.{name}",
        )
        for name in (*_BATCH_TRACE_FIELDS, *_SHARED_TRACE_FIELDS)
    }

    reduced = trace_arrays["reduced_features"]
    membership = trace_arrays["membership_values"]
    if reduced.ndim != 2 or reduced.shape[0] != batch_size or reduced.shape[1] < 1:
        raise ValueError(f"{prefix}.fuzzy_trace.reduced_features has an invalid batch shape")
    if membership.ndim != 3 or membership.shape[:2] != reduced.shape:
        raise ValueError(f"{prefix}.fuzzy_trace.membership_values has an invalid shape")
    num_features = int(reduced.shape[1])
    num_memberships = int(membership.shape[2])
    if num_memberships < 1:
        raise ValueError(f"{prefix} must contain at least one membership function")

    normalized_firing = trace_arrays["normalized_rule_firing"]
    if normalized_firing.ndim != 2 or normalized_firing.shape[0] != batch_size:
        raise ValueError(f"{prefix}.fuzzy_trace.normalized_rule_firing has an invalid shape")
    num_rules = int(normalized_firing.shape[1])
    if num_rules < 1:
        raise ValueError(f"{prefix} must contain at least one rule")

    expected_shapes = {
        "antecedent_memberships": (batch_size, num_rules, num_features),
        "log_rule_firing": (batch_size, num_rules),
        "rule_firing": (batch_size, num_rules),
        "normalized_rule_firing": (batch_size, num_rules),
        "rule_contributions": (batch_size, num_rules, num_classes),
        "fuzzy_logits": (batch_size, num_classes),
        "rule_mask": (batch_size, num_rules),
        "centers": (num_features, num_memberships),
        "widths": (num_features, num_memberships),
        "antecedent_probabilities": (num_rules, num_features, num_memberships),
        "rule_consequents": (num_rules, num_classes),
        "consequent_permutation": (num_rules,),
    }
    for name, shape in expected_shapes.items():
        _expect_shape(
            trace_arrays[name],
            shape,
            name=f"{prefix}.fuzzy_trace.{name}",
        )

    rule_mask = trace_arrays["rule_mask"]
    if not np.isin(rule_mask, (0.0, 1.0)).all():
        raise ValueError(f"{prefix}.fuzzy_trace.rule_mask must contain only 0/1")
    permutation = trace_arrays["consequent_permutation"]
    _assert_integer_values(permutation, name=f"{prefix}.fuzzy_trace.consequent_permutation")
    if set(permutation.astype(np.int64).tolist()) != set(range(num_rules)):
        raise ValueError(
            f"{prefix}.fuzzy_trace.consequent_permutation must contain each rule exactly once"
        )

    fuzzy_logits = trace_arrays["fuzzy_logits"]
    reconstructed_fuzzy = trace_arrays["rule_contributions"].sum(axis=1)
    _assert_per_sample_close(
        fuzzy_logits,
        reconstructed_fuzzy,
        sample_ids=sample_id,
        identity="fuzzy_logits == sum(rule_contributions)",
        atol=atol,
        rtol=rtol,
    )
    reconstructed_logits = non_fuzzy_logits + float(fuzzy_scale) * fuzzy_logits
    _assert_per_sample_close(
        logits,
        reconstructed_logits,
        sample_ids=sample_id,
        identity="logits == non_fuzzy_logits + fuzzy_scale * fuzzy_logits",
        atol=atol,
        rtol=rtol,
    )

    batch_arrays = {
        "sample_id": sample_id,
        "record_id": record_id,
        "group_id": group_id,
        "window_start": window_start,
        "window_end": window_end,
        "y": targets,
        "logits": logits,
        "non_fuzzy_logits": non_fuzzy_logits,
        **{f"trace_{name}": trace_arrays[name] for name in _BATCH_TRACE_FIELDS},
    }
    shared_arrays = {
        "fuzzy_scale": fuzzy_scale,
        **{f"trace_{name}": trace_arrays[name] for name in _SHARED_TRACE_FIELDS},
    }
    return batch_arrays, shared_arrays


def _collect_arrays(
    batches: Iterable[P05TraceBatch],
    *,
    atol: float,
    rtol: float,
) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {name: [] for name in _BATCH_ARRAY_NAMES}
    shared_reference: dict[str, np.ndarray] | None = None
    sample_ids_seen: set[str] = set()
    batch_count = 0

    for batch_index, batch in enumerate(batches):
        if not isinstance(batch, P05TraceBatch):
            raise TypeError("batches must contain P05TraceBatch instances")
        batch_arrays, shared_arrays = _normalise_batch(
            batch,
            batch_index=batch_index,
            atol=atol,
            rtol=rtol,
        )
        for sample_id in batch_arrays["sample_id"].tolist():
            if sample_id in sample_ids_seen:
                raise ValueError(f"duplicate sample_id across trace export: {sample_id!r}")
            sample_ids_seen.add(sample_id)
        for name, value in batch_arrays.items():
            collected[name].append(value)

        if shared_reference is None:
            shared_reference = {name: value.copy() for name, value in shared_arrays.items()}
        else:
            for name, value in shared_arrays.items():
                reference = shared_reference[name]
                if reference.dtype != value.dtype or reference.shape != value.shape:
                    raise ValueError(f"shared trace field {name!r} changed shape or dtype")
                if not np.array_equal(reference, value):
                    raise ValueError(f"shared trace field {name!r} changed across batches")
        batch_count += 1

    if batch_count == 0 or shared_reference is None:
        raise ValueError("at least one non-empty P05TraceBatch is required")

    arrays: dict[str, np.ndarray] = dict(shared_reference)
    for name, pieces in collected.items():
        concatenated = np.concatenate(pieces, axis=0)
        if name in _IDENTIFIER_ARRAY_NAMES:
            arrays[name] = np.ascontiguousarray(concatenated)
        else:
            arrays[name] = np.ascontiguousarray(concatenated, dtype=_FLOAT64)
    if set(arrays) != _BATCH_ARRAY_NAMES | _SHARED_ARRAY_NAMES:
        raise AssertionError("internal trace schema field mismatch")
    return arrays


def _array_sha256(array: np.ndarray) -> str:
    metadata = _canonical_json_bytes(
        {"dtype": array.dtype.str, "shape": [int(size) for size in array.shape]}
    )
    return _sha256_bytes(metadata + b"\0" + array.tobytes(order="C"))


def _array_descriptors(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "batch_axis": name in _BATCH_ARRAY_NAMES,
            "dtype": array.dtype.str,
            "sha256": _array_sha256(array),
            "shape": [int(size) for size in array.shape],
        }
        for name, array in sorted(arrays.items())
    }


def _semantic_manifest(
    arrays: Mapping[str, np.ndarray],
    *,
    config_sha256: str,
    checkpoint_sha256: str,
    model_sha256: str,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    return {
        "arrays": _array_descriptors(arrays),
        "format": {
            "container": "numpy.npz",
            "identifier_arrays": sorted(_IDENTIFIER_ARRAY_NAMES),
            "load_allow_pickle": False,
            "numeric_dtype": _FLOAT64.str,
            "object_arrays": False,
        },
        "npz_file": NPZ_NAME,
        "provenance": {
            "checkpoint_sha256": checkpoint_sha256,
            "config_sha256": config_sha256,
            "model_sha256": model_sha256,
        },
        "reconstruction": {
            "atol": atol,
            "identity": "logits == non_fuzzy_logits + fuzzy_scale * fuzzy_logits",
            "rtol": rtol,
        },
        "sample_count": int(arrays["sample_id"].shape[0]),
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ValueError("trace NPZ contains duplicate array names")
            # ``np.ascontiguousarray`` promotes a scalar from shape ``()`` to
            # ``(1,)``. A direct ordered copy preserves the semantic shape of
            # shared scalars such as ``fuzzy_scale``.
            arrays = {
                name: np.array(archive[name], copy=True, order="C")
                for name in archive.files
            }
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid create-only trace NPZ: {path}") from exc
    for name, array in arrays.items():
        if array.dtype.hasobject:
            raise ValueError(f"trace NPZ array {name!r} requires pickle")
    return arrays


def _arrays_equal(
    existing: Mapping[str, np.ndarray],
    proposed: Mapping[str, np.ndarray],
) -> bool:
    if set(existing) != set(proposed):
        return False
    for name in proposed:
        left = existing[name]
        right = proposed[name]
        if left.dtype != right.dtype or left.shape != right.shape:
            return False
        if not np.array_equal(left, right):
            return False
    return True


def _result(
    target: Path,
    manifest: Mapping[str, Any],
    *,
    status: str,
) -> P05TraceExportResult:
    manifest_path = target / MANIFEST_NAME
    return P05TraceExportResult(
        package_dir=target,
        npz_path=target / NPZ_NAME,
        manifest_path=manifest_path,
        semantic_sha256=str(manifest["content"]["semantic_sha256"]),
        npz_sha256=str(manifest["content"]["npz_sha256"]),
        manifest_sha256=_sha256_file(manifest_path),
        status=status,
    )


def _reuse_existing(
    target: Path,
    arrays: Mapping[str, np.ndarray],
    semantic_manifest: Mapping[str, Any],
) -> P05TraceExportResult:
    if target.is_symlink():
        raise FileExistsError(f"refusing create-only export through symlink: {target}")
    if not target.is_dir():
        raise FileExistsError(f"trace export target already exists and is not a directory: {target}")
    entries = {entry.name: entry for entry in target.iterdir()}
    if set(entries) != {NPZ_NAME, MANIFEST_NAME}:
        raise FileExistsError(f"existing trace package has unexpected or incomplete content: {target}")
    if any(entry.is_symlink() for entry in entries.values()):
        raise FileExistsError(f"existing trace package contains a symlink: {target}")
    if not all(entry.is_file() for entry in entries.values()):
        raise FileExistsError(f"existing trace package contains a non-file entry: {target}")

    manifest_path = entries[MANIFEST_NAME]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FileExistsError(f"existing trace manifest is invalid: {manifest_path}") from exc
    if not isinstance(manifest, dict) or set(manifest) != set(semantic_manifest) | {"content"}:
        raise FileExistsError(f"existing trace manifest schema conflicts with proposed export: {target}")
    content = manifest.get("content")
    if not isinstance(content, dict) or set(content) != {"npz_sha256", "semantic_sha256"}:
        raise FileExistsError(f"existing trace manifest content hashes are invalid: {target}")

    existing_arrays = _load_npz(entries[NPZ_NAME])
    actual_npz_sha256 = _sha256_file(entries[NPZ_NAME])
    if actual_npz_sha256 != content["npz_sha256"]:
        raise FileExistsError(f"existing trace NPZ hash does not match its manifest: {target}")
    existing_semantic = {name: value for name, value in manifest.items() if name != "content"}
    existing_semantic_sha256 = _sha256_bytes(_canonical_json_bytes(existing_semantic))
    if existing_semantic_sha256 != content["semantic_sha256"]:
        raise FileExistsError(f"existing trace semantic hash does not match its manifest: {target}")
    if existing_semantic.get("arrays") != _array_descriptors(existing_arrays):
        raise FileExistsError(f"existing trace array hashes do not match its manifest: {target}")
    if _canonical_json_bytes(existing_semantic) != _canonical_json_bytes(semantic_manifest):
        raise FileExistsError(f"existing trace provenance or schema conflicts with proposed export: {target}")
    if not _arrays_equal(existing_arrays, arrays):
        raise FileExistsError(f"existing trace arrays conflict with proposed export: {target}")
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
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(target),
        rename_noreplace,
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
) -> P05TraceExportResult:
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"trace export parent must be a real directory: {parent}")
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(parent),
        )
    )
    try:
        npz_path = temporary / NPZ_NAME
        with npz_path.open("wb") as handle:
            np.savez(handle, **{name: arrays[name] for name in sorted(arrays)})
            handle.flush()
            os.fsync(handle.fileno())
        npz_sha256 = _sha256_file(npz_path)
        semantic_sha256 = _sha256_bytes(_canonical_json_bytes(semantic_manifest))
        manifest = {
            **semantic_manifest,
            "content": {
                "npz_sha256": npz_sha256,
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


def export_p05_trace_package(
    package_dir: str | Path,
    batches: Iterable[P05TraceBatch],
    *,
    config_sha256: str,
    checkpoint_sha256: str,
    model_sha256: str,
    reconstruction_atol: float = DEFAULT_RECONSTRUCTION_ATOL,
    reconstruction_rtol: float = DEFAULT_RECONSTRUCTION_RTOL,
) -> P05TraceExportResult:
    """Validate and atomically create or semantically reuse one trace package.

    NPZ container bytes are hashed as actually written but are not assumed to
    be reproducible across NumPy/ZIP implementations. Idempotency compares the
    normalized arrays, schema, tolerances, and provenance exactly.
    """

    config_hash = _required_sha256(config_sha256, name="config_sha256")
    checkpoint_hash = _required_sha256(checkpoint_sha256, name="checkpoint_sha256")
    model_hash = _required_sha256(model_sha256, name="model_sha256")
    atol = float(reconstruction_atol)
    rtol = float(reconstruction_rtol)
    if not np.isfinite([atol, rtol]).all() or atol < 0 or rtol < 0:
        raise ValueError("reconstruction tolerances must be finite and non-negative")

    arrays = _collect_arrays(batches, atol=atol, rtol=rtol)
    semantic_manifest = _semantic_manifest(
        arrays,
        config_sha256=config_hash,
        checkpoint_sha256=checkpoint_hash,
        model_sha256=model_hash,
        atol=atol,
        rtol=rtol,
    )
    target = Path(os.path.abspath(os.fspath(package_dir)))
    if target.is_symlink():
        raise FileExistsError(f"refusing create-only export through symlink: {target}")
    if target.exists():
        return _reuse_existing(target, arrays, semantic_manifest)
    return _write_new_package(target, arrays, semantic_manifest)


__all__ = [
    "DEFAULT_RECONSTRUCTION_ATOL",
    "DEFAULT_RECONSTRUCTION_RTOL",
    "P05TraceBatch",
    "P05TraceExportResult",
    "export_p05_trace_package",
]
