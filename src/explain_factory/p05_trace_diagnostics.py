"""Create-only mandatory P05-D01/D02 diagnostics from complete trace exports."""

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_NAME = "p05.d01_d02_trace_diagnostics"
SCHEMA_VERSION = 1
ARRAYS_NAME = "diagnostic_arrays.npz"
MANIFEST_NAME = "manifest.json"

TRACE_SCHEMA_NAME = "p05.complete_fuzzy_trace"
TRACE_SCHEMA_VERSION = 1
TRACE_ARRAYS_NAME = "trace_arrays.npz"
TRACE_MANIFEST_NAME = "manifest.json"

NORMALIZATION_ATOL = 1.0e-6
FIRING_THRESHOLD = 0.10

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TRACE_IDENTIFIER_ARRAYS = {"sample_id", "record_id", "group_id"}
_TRACE_BATCH_ARRAYS = {
    "sample_id",
    "record_id",
    "group_id",
    "window_start",
    "window_end",
    "y",
    "logits",
    "non_fuzzy_logits",
    "trace_reduced_features",
    "trace_membership_values",
    "trace_antecedent_memberships",
    "trace_log_rule_firing",
    "trace_rule_firing",
    "trace_normalized_rule_firing",
    "trace_rule_contributions",
    "trace_fuzzy_logits",
    "trace_rule_mask",
}
_TRACE_SHARED_ARRAYS = {
    "fuzzy_scale",
    "trace_centers",
    "trace_widths",
    "trace_antecedent_probabilities",
    "trace_rule_consequents",
    "trace_consequent_permutation",
}
_TRACE_REQUIRED_ARRAYS = _TRACE_BATCH_ARRAYS | _TRACE_SHARED_ARRAYS
_OUTPUT_IDENTIFIER_ARRAYS = {"sample_id", "record_id", "group_id"}


@dataclass(frozen=True)
class P05TraceDiagnosticsResult:
    artifact_dir: Path
    arrays_path: Path
    manifest_path: Path
    semantic_sha256: str
    arrays_sha256: str
    manifest_sha256: str
    status: str


@dataclass(frozen=True)
class _VerifiedTrace:
    arrays: Mapping[str, np.ndarray]
    semantic_sha256: str
    npz_sha256: str
    manifest_sha256: str
    provenance: Mapping[str, str]


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


def _required_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    normalized = value.lower()
    if _SHA256.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return normalized


def _array_sha256(array: np.ndarray) -> str:
    descriptor = _canonical_json_bytes(
        {"dtype": array.dtype.str, "shape": [int(size) for size in array.shape]}
    )
    return _sha256_bytes(descriptor + b"\0" + array.tobytes(order="C"))


def _output_descriptors(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "dtype": array.dtype.str,
            "sha256": _array_sha256(array),
            "shape": [int(size) for size in array.shape],
        }
        for name, array in sorted(arrays.items())
    }


def _load_npz(path: Path, *, identity: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ValueError(f"{identity} contains duplicate array names")
            arrays = {
                name: np.array(archive[name], copy=True, order="C")
                for name in archive.files
            }
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid {identity}: {path}") from exc
    for name, array in arrays.items():
        if array.dtype.hasobject:
            raise ValueError(f"{identity} array {name!r} requires pickle")
    return arrays


def _verify_source_trace(
    package_dir: str | Path,
    *,
    expected_trace_semantic_sha256: str,
    expected_config_sha256: str,
    expected_checkpoint_sha256: str,
    expected_model_sha256: str,
) -> _VerifiedTrace:
    expected_hashes = {
        "trace_semantic_sha256": _required_sha256(
            expected_trace_semantic_sha256,
            name="expected_trace_semantic_sha256",
        ),
        "config_sha256": _required_sha256(
            expected_config_sha256,
            name="expected_config_sha256",
        ),
        "checkpoint_sha256": _required_sha256(
            expected_checkpoint_sha256,
            name="expected_checkpoint_sha256",
        ),
        "model_sha256": _required_sha256(
            expected_model_sha256,
            name="expected_model_sha256",
        ),
    }
    target = Path(os.path.abspath(os.fspath(package_dir)))
    if target.is_symlink() or not target.is_dir():
        raise FileNotFoundError(
            f"trace package must be a real existing directory: {target}"
        )
    entries = {entry.name: entry for entry in target.iterdir()}
    if set(entries) != {TRACE_ARRAYS_NAME, TRACE_MANIFEST_NAME}:
        raise ValueError(f"trace package has unexpected or incomplete content: {target}")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries.values()):
        raise ValueError(f"trace package entries must be real files: {target}")

    manifest_path = entries[TRACE_MANIFEST_NAME]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid trace manifest: {manifest_path}") from exc
    expected_manifest_keys = {
        "arrays",
        "content",
        "format",
        "npz_file",
        "provenance",
        "reconstruction",
        "sample_count",
        "schema_name",
        "schema_version",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_keys:
        raise ValueError("trace manifest schema is incomplete or unexpected")
    if (
        manifest["schema_name"] != TRACE_SCHEMA_NAME
        or manifest["schema_version"] != TRACE_SCHEMA_VERSION
        or manifest["npz_file"] != TRACE_ARRAYS_NAME
    ):
        raise ValueError("trace manifest schema identity is unsupported")
    if manifest["format"] != {
        "container": "numpy.npz",
        "identifier_arrays": ["group_id", "record_id", "sample_id"],
        "load_allow_pickle": False,
        "numeric_dtype": "<f8",
        "object_arrays": False,
    }:
        raise ValueError("trace manifest format contract is unsupported")

    content = manifest["content"]
    if not isinstance(content, dict) or set(content) != {
        "npz_sha256",
        "semantic_sha256",
    }:
        raise ValueError("trace content hash block is invalid")
    recorded_npz_hash = _required_sha256(
        content["npz_sha256"], name="trace content.npz_sha256"
    )
    recorded_semantic_hash = _required_sha256(
        content["semantic_sha256"], name="trace content.semantic_sha256"
    )
    if recorded_npz_hash != _sha256_file(entries[TRACE_ARRAYS_NAME]):
        raise ValueError("trace NPZ hash does not match its manifest")
    semantic_manifest = {
        name: value for name, value in manifest.items() if name != "content"
    }
    if _sha256_bytes(_canonical_json_bytes(semantic_manifest)) != recorded_semantic_hash:
        raise ValueError("trace semantic hash does not match its manifest")
    if recorded_semantic_hash != expected_hashes["trace_semantic_sha256"]:
        raise ValueError("trace semantic hash does not match the expected binding")

    provenance = manifest["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {
        "checkpoint_sha256",
        "config_sha256",
        "model_sha256",
    }:
        raise ValueError("trace provenance hash block is invalid")
    normalized_provenance = {
        name: _required_sha256(value, name=f"trace provenance.{name}")
        for name, value in provenance.items()
    }
    for name in ("config_sha256", "checkpoint_sha256", "model_sha256"):
        if normalized_provenance[name] != expected_hashes[name]:
            raise ValueError(f"trace {name} does not match the expected binding")

    arrays = _load_npz(entries[TRACE_ARRAYS_NAME], identity="trace NPZ")
    if set(arrays) != _TRACE_REQUIRED_ARRAYS:
        missing = sorted(_TRACE_REQUIRED_ARRAYS - set(arrays))
        unexpected = sorted(set(arrays) - _TRACE_REQUIRED_ARRAYS)
        raise ValueError(
            "trace array schema differs from complete FuzzyTrace: "
            f"missing={missing}, unexpected={unexpected}"
        )
    descriptors = manifest["arrays"]
    if not isinstance(descriptors, dict) or set(descriptors) != set(arrays):
        raise ValueError("trace array inventory does not match its manifest")
    for name, array in arrays.items():
        descriptor = descriptors[name]
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "batch_axis",
            "dtype",
            "sha256",
            "shape",
        }:
            raise ValueError(f"trace descriptor for {name!r} is invalid")
        expected_batch_axis = name in _TRACE_BATCH_ARRAYS
        if descriptor["batch_axis"] is not expected_batch_axis:
            raise ValueError(f"trace batch-axis descriptor mismatch for {name!r}")
        if (
            descriptor["dtype"] != array.dtype.str
            or descriptor["shape"] != [int(size) for size in array.shape]
            or descriptor["sha256"] != _array_sha256(array)
        ):
            raise ValueError(f"trace descriptor or array hash mismatch for {name!r}")
        if name in _TRACE_IDENTIFIER_ARRAYS:
            if array.dtype.kind != "U":
                raise ValueError(f"trace identifier {name!r} must be Unicode")
        elif array.dtype != np.dtype("<f8"):
            raise ValueError(f"trace numeric array {name!r} must use little-endian float64")
        elif not np.isfinite(array).all():
            raise FloatingPointError(f"trace array {name!r} is non-finite")

    sample_count = int(arrays["sample_id"].shape[0])
    if (
        type(manifest["sample_count"]) is not int
        or manifest["sample_count"] != sample_count
        or sample_count <= 0
    ):
        raise ValueError("trace sample count is invalid")
    for name in _TRACE_IDENTIFIER_ARRAYS:
        values = arrays[name]
        if values.shape != (sample_count,):
            raise ValueError(f"trace {name} must be a sample vector")
        if any(not str(value).strip() or "\x00" in str(value) for value in values):
            raise ValueError(f"trace {name} contains an invalid stable identifier")
    if len(set(arrays["sample_id"].tolist())) != sample_count:
        raise ValueError("trace sample_id values must be unique")
    for name in _TRACE_BATCH_ARRAYS - _TRACE_IDENTIFIER_ARRAYS:
        if arrays[name].ndim == 0 or arrays[name].shape[0] != sample_count:
            raise ValueError(f"trace batch axis mismatch for {name!r}")

    labels = arrays["y"]
    logits = arrays["logits"]
    if labels.shape != (sample_count,) or not np.equal(labels, np.round(labels)).all():
        raise ValueError("trace labels must be an integer sample vector")
    if logits.ndim != 2 or logits.shape[0] != sample_count or logits.shape[1] < 2:
        raise ValueError("trace logits must have shape (samples, classes>=2)")
    class_count = int(logits.shape[1])
    labels_i64 = labels.astype(np.int64)
    if np.any(labels_i64 < 0) or np.any(labels_i64 >= class_count):
        raise ValueError("trace labels contain an out-of-range protocol class")
    observed_classes = set(labels_i64.tolist())
    expected_classes = set(range(class_count))
    if observed_classes != expected_classes:
        missing = sorted(expected_classes - observed_classes)
        raise ValueError(f"trace is missing protocol classes: {missing}")

    firing = arrays["trace_normalized_rule_firing"]
    log_firing = arrays["trace_log_rule_firing"]
    if firing.ndim != 2 or firing.shape[0] != sample_count or firing.shape[1] < 3:
        raise ValueError(
            "trace normalized firing must have shape (samples, rules>=3)"
        )
    if log_firing.shape != firing.shape:
        raise ValueError("trace log firing shape differs from normalized firing")
    if np.any(firing < 0.0):
        raise ValueError("trace normalized firing contains a negative mass")
    row_sums = firing.sum(axis=1, dtype=np.float64)
    if np.any(np.abs(row_sums - 1.0) > NORMALIZATION_ATOL):
        raise ValueError("trace normalized firing rows do not sum to one")
    shifted = log_firing - np.max(log_firing, axis=1, keepdims=True)
    expected_firing = np.exp(shifted)
    expected_firing /= expected_firing.sum(axis=1, keepdims=True)
    if not np.allclose(
        firing,
        expected_firing,
        atol=NORMALIZATION_ATOL,
        rtol=NORMALIZATION_ATOL,
    ):
        raise ValueError("trace normalized firing differs from log-firing softmax")
    rules = int(firing.shape[1])
    if arrays["trace_rule_mask"].shape != (sample_count, rules) or not np.array_equal(
        arrays["trace_rule_mask"], np.ones((sample_count, rules), dtype=np.float64)
    ):
        raise ValueError("trace diagnostics require every rule to be active")
    if arrays["trace_consequent_permutation"].shape != (rules,) or not np.array_equal(
        arrays["trace_consequent_permutation"],
        np.arange(rules, dtype=np.float64),
    ):
        raise ValueError("trace diagnostics require canonical rule-index order")

    reconstruction = manifest["reconstruction"]
    if not isinstance(reconstruction, dict) or set(reconstruction) != {
        "atol",
        "identity",
        "rtol",
    }:
        raise ValueError("trace reconstruction contract is invalid")
    if reconstruction["identity"] != (
        "logits == non_fuzzy_logits + fuzzy_scale * fuzzy_logits"
    ):
        raise ValueError("trace reconstruction identity is unsupported")
    for name in ("atol", "rtol"):
        value = reconstruction[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"trace reconstruction {name} is invalid")

    return _VerifiedTrace(
        arrays=arrays,
        semantic_sha256=recorded_semantic_hash,
        npz_sha256=recorded_npz_hash,
        manifest_sha256=_sha256_file(manifest_path),
        provenance=normalized_provenance,
    )


def _coverage_masks(
    firing: np.ndarray,
    top_rule: np.ndarray,
    top3_rules: np.ndarray,
    selection: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rules = int(firing.shape[1])
    top_mask = np.zeros(rules, dtype=np.bool_)
    top_mask[np.unique(top_rule[selection])] = True
    threshold_mask = np.any(firing[selection] > FIRING_THRESHOLD, axis=0)
    top3_mask = np.zeros(rules, dtype=np.bool_)
    top3_mask[np.unique(top3_rules[selection])] = True
    return top_mask, threshold_mask, top3_mask


def _diagnostic_arrays(trace: _VerifiedTrace) -> dict[str, np.ndarray]:
    source = trace.arrays
    labels = source["y"].astype(np.int64)
    class_count = int(source["logits"].shape[1])
    firing = np.asarray(source["trace_normalized_rule_firing"], dtype=np.float64)
    firing = firing / firing.sum(axis=1, keepdims=True, dtype=np.float64)
    sample_count, rule_count = firing.shape

    ranked_rules = np.argsort(-firing, axis=1, kind="stable")
    top_rule = ranked_rules[:, 0]
    top3_rules = ranked_rules[:, :3]
    ranked_firing = np.take_along_axis(firing, ranked_rules, axis=1)
    positive = firing > 0.0
    entropy_terms = np.zeros_like(firing)
    entropy_terms[positive] = firing[positive] * np.log(firing[positive])
    effective_rule_count = np.exp(-entropy_terms.sum(axis=1))
    pairwise = np.abs(firing[:, :, None] - firing[:, None, :])
    gini = pairwise.sum(axis=(1, 2)) / (2.0 * rule_count)

    overall_selection = np.ones(sample_count, dtype=np.bool_)
    overall_masks = _coverage_masks(
        firing,
        top_rule,
        top3_rules,
        overall_selection,
    )
    by_class_masks = [
        _coverage_masks(
            firing,
            top_rule,
            top3_rules,
            labels == class_index,
        )
        for class_index in range(class_count)
    ]
    mask_names = (
        "ever_top_ranked",
        "firing_gt_0_10",
        "appearing_top3",
    )
    arrays: dict[str, np.ndarray] = {
        "sample_id": np.asarray(source["sample_id"]),
        "record_id": np.asarray(source["record_id"]),
        "group_id": np.asarray(source["group_id"]),
        "window_start": np.asarray(source["window_start"], dtype="<i8"),
        "window_end": np.asarray(source["window_end"], dtype="<i8"),
        "protocol_class": np.asarray(labels, dtype="<i8"),
        "protocol_class_index": np.arange(class_count, dtype="<i8"),
        "protocol_class_sample_count": np.bincount(
            labels,
            minlength=class_count,
        ).astype("<i8"),
        "rule_index": np.arange(rule_count, dtype="<i8"),
        "d01_effective_rule_count": np.asarray(
            effective_rule_count,
            dtype="<f8",
        ),
        "d01_top1_mass": np.asarray(ranked_firing[:, 0], dtype="<f8"),
        "d01_top3_mass": np.asarray(ranked_firing[:, :3].sum(axis=1), dtype="<f8"),
        "d01_gini": np.asarray(gini, dtype="<f8"),
        "d01_top_rule_index": np.asarray(top_rule, dtype="<i8"),
        "d01_top3_rule_indices": np.asarray(top3_rules, dtype="<i8"),
    }
    for metric_index, name in enumerate(mask_names):
        overall = np.asarray(overall_masks[metric_index], dtype=np.bool_)
        by_class = np.asarray(
            [masks[metric_index] for masks in by_class_masks],
            dtype=np.bool_,
        )
        arrays[f"d02_overall_{name}_rule_mask"] = overall
        arrays[f"d02_overall_{name}_coverage"] = np.asarray(
            overall.mean(dtype=np.float64),
            dtype="<f8",
        )
        arrays[f"d02_by_class_{name}_rule_mask"] = by_class
        arrays[f"d02_by_class_{name}_coverage"] = np.asarray(
            by_class.mean(axis=1, dtype=np.float64),
            dtype="<f8",
        )
    normalized = {
        name: np.array(value, copy=True, order="C")
        for name, value in sorted(arrays.items())
    }
    for name, array in normalized.items():
        if array.dtype.hasobject:
            raise AssertionError(f"diagnostic array {name!r} cannot contain objects")
        if name in _OUTPUT_IDENTIFIER_ARRAYS:
            if array.dtype.kind != "U":
                raise AssertionError(f"diagnostic identifier {name!r} is not Unicode")
        elif array.dtype.kind in {"b", "i", "u", "f"}:
            if not np.isfinite(array).all():
                raise FloatingPointError(f"diagnostic array {name!r} is non-finite")
        else:
            raise AssertionError(f"diagnostic array {name!r} has unsupported dtype")
    return normalized


def _semantic_manifest(
    trace: _VerifiedTrace,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    return {
        "arrays": _output_descriptors(arrays),
        "conclusion_control": {
            "claim_decisions": "not_performed",
            "confirmatory_sign_tests": "not_performed",
            "scientific_status": "computed_unadjudicated",
            "scope": "mandatory_P05_D01_D02_trace_diagnostics_only",
        },
        "format": {
            "container": "numpy.npz",
            "identifier_arrays": sorted(_OUTPUT_IDENTIFIER_ARRAYS),
            "load_allow_pickle": False,
            "object_arrays": False,
        },
        "npz_file": ARRAYS_NAME,
        "protocol": {
            "P05-D01": {
                "effective_rule_count": (
                    "exp(-sum_r p_r * natural_log(p_r)); zero-mass terms are zero"
                ),
                "gini": "sum_r sum_s abs(p_r-p_s) / (2 * rule_count)",
                "input": "trace_normalized_rule_firing",
                "top1_mass": "largest normalized firing mass",
                "top3_mass": "sum of three largest normalized firing masses",
            },
            "P05-D02": {
                "coverage_denominator": "rule_count",
                "groupings": ["overall", "protocol_class_label"],
                "metrics": [
                    "ever_top_ranked",
                    "firing_gt_0_10",
                    "appearing_top3",
                ],
                "threshold": FIRING_THRESHOLD,
                "threshold_operator": ">",
            },
            "class_domain": "all integer logits columns 0..C-1; missing class is error",
            "firing_row_sum_atol": NORMALIZATION_ATOL,
            "metric_precision": "float64 after row renormalization",
            "tie_break": "descending firing, then lower rule index",
        },
        "protocol_class_count": int(arrays["protocol_class_index"].shape[0]),
        "rule_count": int(arrays["rule_index"].shape[0]),
        "sample_count": int(arrays["sample_id"].shape[0]),
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "source_trace": {
            "checkpoint_sha256": trace.provenance["checkpoint_sha256"],
            "config_sha256": trace.provenance["config_sha256"],
            "manifest_sha256": trace.manifest_sha256,
            "model_sha256": trace.provenance["model_sha256"],
            "npz_sha256": trace.npz_sha256,
            "schema_name": TRACE_SCHEMA_NAME,
            "schema_version": TRACE_SCHEMA_VERSION,
            "semantic_sha256": trace.semantic_sha256,
        },
    }


def _arrays_equal(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
) -> bool:
    if set(left) != set(right):
        return False
    return all(
        left[name].dtype == right[name].dtype
        and left[name].shape == right[name].shape
        and np.array_equal(left[name], right[name])
        for name in right
    )


def _result(
    target: Path,
    manifest: Mapping[str, Any],
    *,
    status: str,
) -> P05TraceDiagnosticsResult:
    manifest_path = target / MANIFEST_NAME
    return P05TraceDiagnosticsResult(
        artifact_dir=target,
        arrays_path=target / ARRAYS_NAME,
        manifest_path=manifest_path,
        semantic_sha256=str(manifest["content"]["semantic_sha256"]),
        arrays_sha256=str(manifest["content"]["npz_sha256"]),
        manifest_sha256=_sha256_file(manifest_path),
        status=status,
    )


def _reuse_existing(
    target: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    semantic_manifest: Mapping[str, Any],
) -> P05TraceDiagnosticsResult:
    if target.is_symlink():
        raise FileExistsError(f"refusing create-only artifact through symlink: {target}")
    if not target.is_dir():
        raise FileExistsError(
            f"diagnostic artifact target exists and is not a directory: {target}"
        )
    entries = {entry.name: entry for entry in target.iterdir()}
    if set(entries) != {ARRAYS_NAME, MANIFEST_NAME}:
        raise FileExistsError(
            f"existing diagnostic artifact has unexpected or incomplete content: {target}"
        )
    if any(entry.is_symlink() or not entry.is_file() for entry in entries.values()):
        raise FileExistsError(
            f"existing diagnostic artifact entries must be real files: {target}"
        )
    try:
        manifest = json.loads(entries[MANIFEST_NAME].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FileExistsError(
            f"existing diagnostic manifest is invalid: {entries[MANIFEST_NAME]}"
        ) from exc
    if not isinstance(manifest, dict) or set(manifest) != set(semantic_manifest) | {
        "content"
    }:
        raise FileExistsError("existing diagnostic manifest schema conflicts")
    content = manifest["content"]
    if not isinstance(content, dict) or set(content) != {
        "npz_sha256",
        "semantic_sha256",
    }:
        raise FileExistsError("existing diagnostic content hashes are invalid")
    existing_arrays = _load_npz(entries[ARRAYS_NAME], identity="diagnostic NPZ")
    if _sha256_file(entries[ARRAYS_NAME]) != content["npz_sha256"]:
        raise FileExistsError("existing diagnostic NPZ hash differs from its manifest")
    existing_semantic = {
        name: value for name, value in manifest.items() if name != "content"
    }
    if _sha256_bytes(_canonical_json_bytes(existing_semantic)) != content[
        "semantic_sha256"
    ]:
        raise FileExistsError("existing diagnostic semantic hash is invalid")
    if existing_semantic.get("arrays") != _output_descriptors(existing_arrays):
        raise FileExistsError("existing diagnostic array descriptors are invalid")
    if _canonical_json_bytes(existing_semantic) != _canonical_json_bytes(
        semantic_manifest
    ) or not _arrays_equal(existing_arrays, arrays):
        raise FileExistsError("existing diagnostic artifact conflicts with proposed output")
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
        raise RuntimeError("atomic create-only artifact requires Linux renameat2")
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


def _write_new(
    target: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    semantic_manifest: Mapping[str, Any],
) -> P05TraceDiagnosticsResult:
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"diagnostic artifact parent must be a real directory: {parent}")
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(parent),
        )
    )
    try:
        arrays_path = temporary / ARRAYS_NAME
        with arrays_path.open("wb") as handle:
            np.savez(handle, **{name: arrays[name] for name in sorted(arrays)})
            handle.flush()
            os.fsync(handle.fileno())
        manifest = {
            **semantic_manifest,
            "content": {
                "npz_sha256": _sha256_file(arrays_path),
                "semantic_sha256": _sha256_bytes(
                    _canonical_json_bytes(semantic_manifest)
                ),
            },
        }
        manifest_path = temporary / MANIFEST_NAME
        with manifest_path.open("wb") as handle:
            handle.write(_pretty_json_bytes(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary)
        try:
            _rename_directory_noreplace(temporary, target)
        except FileExistsError:
            return _reuse_existing(
                target,
                arrays=arrays,
                semantic_manifest=semantic_manifest,
            )
        _fsync_directory(parent)
        return _result(target, manifest, status="created")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def create_p05_d01_d02_trace_diagnostics(
    artifact_dir: str | Path,
    *,
    trace_package: str | Path,
    expected_trace_semantic_sha256: str,
    expected_config_sha256: str,
    expected_checkpoint_sha256: str,
    expected_model_sha256: str,
) -> P05TraceDiagnosticsResult:
    """Create or exactly reuse hash-bound, unadjudicated D01/D02 diagnostics."""

    trace = _verify_source_trace(
        trace_package,
        expected_trace_semantic_sha256=expected_trace_semantic_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_model_sha256=expected_model_sha256,
    )
    arrays = _diagnostic_arrays(trace)
    semantic_manifest = _semantic_manifest(trace, arrays)
    target = Path(os.path.abspath(os.fspath(artifact_dir)))
    if target.is_symlink():
        raise FileExistsError(f"refusing create-only artifact through symlink: {target}")
    if target.exists():
        return _reuse_existing(
            target,
            arrays=arrays,
            semantic_manifest=semantic_manifest,
        )
    return _write_new(
        target,
        arrays=arrays,
        semantic_manifest=semantic_manifest,
    )


__all__ = [
    "P05TraceDiagnosticsResult",
    "create_p05_d01_d02_trace_diagnostics",
]
