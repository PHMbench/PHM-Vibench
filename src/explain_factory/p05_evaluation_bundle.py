"""Create-only coordinator for unadjudicated P05 C2/C3 evaluation bundles."""

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
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau

from .p05_intervention_eval import (
    evaluate_rule_interventions,
    natural_log_jsd,
)
from .p05_intervention_runner import (
    P05ActualInterventionResult,
    verify_p05_actual_intervention_result,
)
from .p05_selective_risk import (
    ValidationRiskBundle,
    equal_mass_ece,
    fit_validation_risk_bundle,
    frozen_threshold_metrics,
    retrospective_selective_metrics,
    score_risk_methods,
)


SCHEMA_NAME = "p05.c2_c3_evaluation_bundle"
SCHEMA_VERSION = 2
ARRAYS_NAME = "evaluation_arrays.npz"
C3_NAME = "c3_retrospective.json"
MANIFEST_NAME = "manifest.json"
TRACE_SCHEMA_NAME = "p05.complete_fuzzy_trace"
TRACE_SCHEMA_VERSION = 1
TRACE_ARRAYS_NAME = "trace_arrays.npz"
TRACE_MANIFEST_NAME = "manifest.json"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REGISTERED_SEEDS = {42, 123, 456, 789, 1024}
_FROZEN_COVERAGES = (0.70, 0.80, 0.90, 0.95)
_REQUIRED_TRACE_ARRAYS = {
    "sample_id",
    "record_id",
    "group_id",
    "window_start",
    "window_end",
    "y",
    "logits",
    "non_fuzzy_logits",
    "fuzzy_scale",
    "trace_reduced_features",
    "trace_membership_values",
    "trace_centers",
    "trace_widths",
    "trace_antecedent_probabilities",
    "trace_antecedent_memberships",
    "trace_log_rule_firing",
    "trace_rule_firing",
    "trace_normalized_rule_firing",
    "trace_rule_consequents",
    "trace_rule_contributions",
    "trace_fuzzy_logits",
    "trace_rule_mask",
    "trace_consequent_permutation",
}
_SHARED_TRACE_ARRAYS = {
    "fuzzy_scale",
    "trace_centers",
    "trace_widths",
    "trace_antecedent_probabilities",
    "trace_rule_consequents",
    "trace_consequent_permutation",
}


@dataclass(frozen=True)
class P05EvaluationFrozenParameters:
    """Frozen per-run binding required to coordinate C2 and C3."""

    dataset: str
    model_seed: int
    validation_trace_semantic_sha256: str
    evaluation_trace_semantic_sha256: str
    validation_split: str = "validation"
    evaluation_split: str = "test"
    reconstruction_atol: float = 1.0e-6
    reconstruction_rtol: float = 1.0e-6
    shuffle_count: int = 32
    rule_count: int = 10
    fuzzy_scale: float = 0.5
    target_coverage: float = 0.90
    coverages: tuple[float, ...] = _FROZEN_COVERAGES
    ece_bins: int = 15


@dataclass(frozen=True)
class P05EvaluationBundleResult:
    bundle_dir: Path
    arrays_path: Path
    c3_path: Path
    manifest_path: Path
    semantic_sha256: str
    arrays_sha256: str
    c3_sha256: str
    manifest_sha256: str
    status: str


@dataclass(frozen=True)
class _VerifiedTracePackage:
    package_dir: Path
    arrays: Mapping[str, np.ndarray]
    manifest: Mapping[str, Any]
    manifest_sha256: str
    semantic_sha256: str
    npz_sha256: str


@dataclass(frozen=True)
class _VerifiedActualInterventions:
    arrays: Mapping[str, np.ndarray]
    input_payload: Mapping[str, Any]


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
    metadata = _canonical_json_bytes(
        {"dtype": array.dtype.str, "shape": [int(size) for size in array.shape]}
    )
    return _sha256_bytes(metadata + b"\0" + array.tobytes(order="C"))


def _array_descriptors(
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
        if array.dtype.kind in {"b", "i", "u", "f"} and not np.isfinite(array).all():
            raise FloatingPointError(f"{identity} array {name!r} is non-finite")
    return arrays


def _verify_trace_package(package_dir: str | Path) -> _VerifiedTracePackage:
    target = Path(os.path.abspath(os.fspath(package_dir)))
    if target.is_symlink() or not target.is_dir():
        raise FileNotFoundError(f"trace package must be a real existing directory: {target}")
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
    if not isinstance(manifest, dict):
        raise ValueError("trace manifest must be a JSON object")
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
    if set(manifest) != expected_manifest_keys:
        raise ValueError("trace manifest schema is incomplete or unexpected")
    if (
        manifest.get("schema_name") != TRACE_SCHEMA_NAME
        or manifest.get("schema_version") != TRACE_SCHEMA_VERSION
        or manifest.get("npz_file") != TRACE_ARRAYS_NAME
    ):
        raise ValueError("trace manifest schema identity is unsupported")
    if manifest.get("format") != {
        "container": "numpy.npz",
        "identifier_arrays": ["group_id", "record_id", "sample_id"],
        "load_allow_pickle": False,
        "numeric_dtype": "<f8",
        "object_arrays": False,
    }:
        raise ValueError("trace manifest format contract is unsupported")
    reconstruction = manifest.get("reconstruction")
    if not isinstance(reconstruction, dict) or set(reconstruction) != {
        "atol",
        "identity",
        "rtol",
    }:
        raise ValueError("trace reconstruction contract is invalid")
    if reconstruction.get("identity") != (
        "logits == non_fuzzy_logits + fuzzy_scale * fuzzy_logits"
    ):
        raise ValueError("trace reconstruction identity is unsupported")
    for name in ("atol", "rtol"):
        value = reconstruction.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"trace reconstruction {name} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"trace reconstruction {name} must be finite and non-negative")

    content = manifest.get("content")
    if not isinstance(content, dict) or set(content) != {
        "npz_sha256",
        "semantic_sha256",
    }:
        raise ValueError("trace manifest content hash block is invalid")
    recorded_npz_hash = _required_sha256(
        content.get("npz_sha256"),
        name="trace content.npz_sha256",
    )
    recorded_semantic_hash = _required_sha256(
        content.get("semantic_sha256"),
        name="trace content.semantic_sha256",
    )
    actual_npz_hash = _sha256_file(entries[TRACE_ARRAYS_NAME])
    if actual_npz_hash != recorded_npz_hash:
        raise ValueError("trace NPZ hash does not match its manifest")
    semantic_manifest = {
        name: value for name, value in manifest.items() if name != "content"
    }
    if _sha256_bytes(_canonical_json_bytes(semantic_manifest)) != recorded_semantic_hash:
        raise ValueError("trace semantic hash does not match its manifest")

    arrays = _load_npz(entries[TRACE_ARRAYS_NAME], identity="trace NPZ")
    descriptors = manifest.get("arrays")
    if not isinstance(descriptors, dict) or set(descriptors) != set(arrays):
        raise ValueError("trace array inventory does not match its manifest")
    for name, array in arrays.items():
        descriptor = descriptors.get(name)
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "batch_axis",
            "dtype",
            "sha256",
            "shape",
        }:
            raise ValueError(f"trace descriptor for {name!r} is invalid")
        if descriptor.get("dtype") != array.dtype.str:
            raise ValueError(f"trace dtype descriptor mismatch for {name!r}")
        if descriptor.get("shape") != [int(size) for size in array.shape]:
            raise ValueError(f"trace shape descriptor mismatch for {name!r}")
        if descriptor.get("sha256") != _array_sha256(array):
            raise ValueError(f"trace array hash mismatch for {name!r}")
    if set(arrays) != _REQUIRED_TRACE_ARRAYS:
        missing = sorted(_REQUIRED_TRACE_ARRAYS - set(arrays))
        unexpected = sorted(set(arrays) - _REQUIRED_TRACE_ARRAYS)
        raise ValueError(
            "trace package array schema differs from complete FuzzyTrace: "
            f"missing={missing}, unexpected={unexpected}"
        )

    sample_count = int(arrays["sample_id"].shape[0])
    if (
        type(manifest.get("sample_count")) is not int
        or manifest.get("sample_count") != sample_count
        or sample_count <= 0
    ):
        raise ValueError("trace sample count is invalid")
    for name in ("sample_id", "record_id", "group_id"):
        values = arrays[name]
        if values.shape != (sample_count,) or values.dtype.kind != "U":
            raise ValueError(f"trace {name} must be a Unicode sample vector")
        if any(not str(value) for value in values.tolist()):
            raise ValueError(f"trace {name} contains an empty identifier")
    if len(set(arrays["sample_id"].tolist())) != sample_count:
        raise ValueError("trace sample_id values must be unique")
    for name in (
        "window_start",
        "window_end",
        "y",
        "logits",
        "non_fuzzy_logits",
        "trace_log_rule_firing",
        "trace_normalized_rule_firing",
        "trace_rule_contributions",
    ):
        if arrays[name].shape[0] != sample_count:
            raise ValueError(f"trace batch axis mismatch for {name!r}")
    if arrays["fuzzy_scale"].size != 1:
        raise ValueError("trace fuzzy_scale must be scalar")
    labels = arrays["y"]
    if labels.shape != (sample_count,) or not np.equal(labels, np.round(labels)).all():
        raise ValueError("trace labels must be an integer sample vector")
    window_start = arrays["window_start"]
    window_end = arrays["window_end"]
    if (
        window_start.shape != (sample_count,)
        or window_end.shape != (sample_count,)
        or not np.equal(window_start, np.round(window_start)).all()
        or not np.equal(window_end, np.round(window_end)).all()
        or np.any(window_start < 0)
        or np.any(window_end <= window_start)
    ):
        raise ValueError("trace window bounds are invalid")
    logits = arrays["logits"]
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("trace logits must have at least two classes")
    if arrays["non_fuzzy_logits"].shape != logits.shape:
        raise ValueError("trace non-fuzzy logits shape differs from logits")
    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("trace labels contain an out-of-range class index")

    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != {
        "checkpoint_sha256",
        "config_sha256",
        "model_sha256",
    }:
        raise ValueError("trace provenance hash block is invalid")
    for name, value in provenance.items():
        _required_sha256(value, name=f"trace provenance.{name}")

    return _VerifiedTracePackage(
        package_dir=target,
        arrays=arrays,
        manifest=manifest,
        manifest_sha256=_sha256_file(manifest_path),
        semantic_sha256=recorded_semantic_hash,
        npz_sha256=recorded_npz_hash,
    )


def _validate_trace_semantics(
    trace: _VerifiedTracePackage,
    frozen: P05EvaluationFrozenParameters,
    *,
    role: str,
) -> None:
    arrays = trace.arrays
    reconstruction = trace.manifest["reconstruction"]
    if not math.isclose(
        float(reconstruction["atol"]),
        frozen.reconstruction_atol,
        rel_tol=0.0,
        abs_tol=0.0,
    ) or not math.isclose(
        float(reconstruction["rtol"]),
        frozen.reconstruction_rtol,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(f"{role} trace reconstruction tolerances are not frozen")

    logits = arrays["logits"]
    expected_classes = 2 if frozen.dataset == "XJTU" else 4
    if logits.shape[1] != expected_classes:
        raise ValueError(
            f"{role} trace class count does not match frozen dataset {frozen.dataset}"
        )
    log_firing = arrays["trace_log_rule_firing"]
    normalized_firing = arrays["trace_normalized_rule_firing"]
    consequents = arrays["trace_rule_consequents"]
    contributions = arrays["trace_rule_contributions"]
    sample_count = logits.shape[0]
    rules = frozen.rule_count
    if (
        log_firing.shape != (sample_count, rules)
        or arrays["trace_rule_firing"].shape != (sample_count, rules)
        or normalized_firing.shape != (sample_count, rules)
        or consequents.shape != (rules, expected_classes)
        or contributions.shape != (sample_count, rules, expected_classes)
        or arrays["trace_fuzzy_logits"].shape != logits.shape
        or arrays["trace_rule_mask"].shape != (sample_count, rules)
        or arrays["trace_consequent_permutation"].shape != (rules,)
    ):
        raise ValueError(f"{role} trace tensor shapes violate the frozen model contract")
    if not np.array_equal(arrays["trace_rule_mask"], np.ones((sample_count, rules))):
        raise ValueError(f"{role} trace must retain every rule before C2 evaluation")
    if not np.array_equal(
        arrays["trace_consequent_permutation"],
        np.arange(rules, dtype=np.float64),
    ):
        raise ValueError(f"{role} trace consequents must be in their original order")

    shifted = log_firing - np.max(log_firing, axis=1, keepdims=True)
    expected_rule_firing = np.exp(log_firing)
    expected_normalized = np.exp(shifted)
    expected_normalized /= expected_normalized.sum(axis=1, keepdims=True)
    atol = frozen.reconstruction_atol
    rtol = frozen.reconstruction_rtol
    if not np.allclose(
        arrays["trace_rule_firing"], expected_rule_firing, atol=atol, rtol=rtol
    ):
        raise ValueError(f"{role} trace rule firing does not reconstruct from log firing")
    if not np.allclose(
        normalized_firing, expected_normalized, atol=atol, rtol=rtol
    ):
        raise ValueError(f"{role} trace normalized firing does not reconstruct")
    expected_contributions = normalized_firing[:, :, None] * consequents[None, :, :]
    if not np.allclose(contributions, expected_contributions, atol=atol, rtol=rtol):
        raise ValueError(f"{role} trace contributions do not equal firing times consequent")
    expected_fuzzy_logits = contributions.sum(axis=1)
    if not np.allclose(
        arrays["trace_fuzzy_logits"], expected_fuzzy_logits, atol=atol, rtol=rtol
    ):
        raise ValueError(f"{role} trace fuzzy logits do not reconstruct")
    scale = float(np.asarray(arrays["fuzzy_scale"]).reshape(()))
    if not math.isclose(scale, frozen.fuzzy_scale, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(f"{role} trace fuzzy scale does not match the frozen model")
    expected_logits = arrays["non_fuzzy_logits"] + scale * expected_fuzzy_logits
    if not np.allclose(logits, expected_logits, atol=atol, rtol=rtol):
        raise ValueError(f"{role} trace logits do not reconstruct")


def _normalized_firing_from_log(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Recompute protocol firing in float64 after trace semantics are verified."""

    log_firing = np.asarray(arrays["trace_log_rule_firing"], dtype=np.float64)
    shifted = log_firing - np.max(log_firing, axis=1, keepdims=True)
    firing = np.exp(shifted)
    firing /= firing.sum(axis=1, keepdims=True)
    return firing


def _validate_frozen_parameters(
    frozen: P05EvaluationFrozenParameters,
) -> dict[str, Any]:
    if not isinstance(frozen, P05EvaluationFrozenParameters):
        raise TypeError("frozen must be P05EvaluationFrozenParameters")
    if frozen.dataset not in {"CWRU", "XJTU"}:
        raise ValueError("frozen dataset must be CWRU or XJTU")
    if type(frozen.model_seed) is not int or frozen.model_seed not in _REGISTERED_SEEDS:
        raise ValueError("frozen model_seed is not a registered decisive seed")
    if frozen.validation_split != "validation" or frozen.evaluation_split != "test":
        raise ValueError("frozen splits must be validation and test")
    validation_hash = _required_sha256(
        frozen.validation_trace_semantic_sha256,
        name="frozen validation trace semantic hash",
    )
    evaluation_hash = _required_sha256(
        frozen.evaluation_trace_semantic_sha256,
        name="frozen evaluation trace semantic hash",
    )
    if not math.isclose(frozen.reconstruction_atol, 1.0e-6, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("frozen reconstruction_atol must equal 1e-6")
    if not math.isclose(frozen.reconstruction_rtol, 1.0e-6, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("frozen reconstruction_rtol must equal 1e-6")
    if frozen.shuffle_count != 32:
        raise ValueError("frozen shuffle_count must equal 32")
    if type(frozen.rule_count) is not int or frozen.rule_count != 10:
        raise ValueError("frozen rule_count must equal 10")
    if not math.isclose(frozen.fuzzy_scale, 0.5, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("frozen fuzzy_scale must equal 0.5")
    if not math.isclose(frozen.target_coverage, 0.90, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("frozen target_coverage must equal 0.90")
    coverages = tuple(float(value) for value in frozen.coverages)
    if coverages != _FROZEN_COVERAGES:
        raise ValueError("frozen coverages must equal (0.70, 0.80, 0.90, 0.95)")
    if type(frozen.ece_bins) is not int or frozen.ece_bins != 15:
        raise ValueError("frozen ece_bins must equal 15")
    return {
        "coverages": list(coverages),
        "dataset": frozen.dataset,
        "ece_bins": frozen.ece_bins,
        "evaluation_split": frozen.evaluation_split,
        "evaluation_trace_semantic_sha256": evaluation_hash,
        "fuzzy_scale": frozen.fuzzy_scale,
        "model_seed": frozen.model_seed,
        "reconstruction_atol": frozen.reconstruction_atol,
        "reconstruction_rtol": frozen.reconstruction_rtol,
        "shuffle_count": frozen.shuffle_count,
        "rule_count": frozen.rule_count,
        "target_coverage": frozen.target_coverage,
        "validation_split": frozen.validation_split,
        "validation_trace_semantic_sha256": validation_hash,
    }


def _collect_actual_interventions(
    results: Sequence[P05ActualInterventionResult],
    *,
    evaluation: _VerifiedTracePackage,
    frozen: P05EvaluationFrozenParameters,
) -> _VerifiedActualInterventions:
    if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        raise TypeError("actual_intervention_results must be a sequence")
    if not results:
        raise ValueError("actual same-checkpoint intervention results are required")

    evaluation_ids = evaluation.arrays["sample_id"].tolist()
    evaluation_index = {
        str(sample_id): index for index, sample_id in enumerate(evaluation_ids)
    }
    locations: dict[str, tuple[Mapping[str, np.ndarray], int]] = {}
    chunk_payloads: list[dict[str, Any]] = []
    actual_names: set[str] | None = None
    trace_provenance = evaluation.manifest["provenance"]
    for chunk_index, result in enumerate(results):
        verify_p05_actual_intervention_result(result)
        metadata = result.metadata
        provenance = metadata["provenance"]
        selection = metadata["selection"]
        if (
            selection["benchmark_first_n"] is not None
            or selection["kind"] != "all_after_stable_sample_id_sort"
            or selection["input_count"] != selection["selected_count"]
        ):
            raise ValueError(
                "evaluation bundle requires complete actual forwards, not a benchmark prefix"
            )
        if (
            provenance["dataset"] != frozen.dataset
            or provenance["split"] != frozen.evaluation_split
            or provenance["model_seed"] != frozen.model_seed
        ):
            raise ValueError(
                f"actual intervention chunk {chunk_index} provenance differs from frozen run"
            )
        for name in ("checkpoint_sha256", "config_sha256", "model_sha256"):
            if provenance[name] != trace_provenance[name]:
                raise ValueError(
                    f"actual intervention chunk {chunk_index} {name} differs from test trace"
                )

        arrays = result.arrays
        names = {name for name in arrays if name.startswith("actual_")}
        if actual_names is None:
            actual_names = names
        elif names != actual_names:
            raise ValueError("actual intervention chunks expose different array schemas")
        chunk_ids = arrays["sample_id"].tolist()
        for local_index, sample_id in enumerate(chunk_ids):
            sample_id = str(sample_id)
            if sample_id not in evaluation_index:
                raise ValueError(
                    f"actual intervention sample_id is absent from test trace: {sample_id!r}"
                )
            if sample_id in locations:
                raise ValueError(
                    f"actual intervention sample_id is duplicated across chunks: {sample_id!r}"
                )
            trace_index = evaluation_index[sample_id]
            for name in _REQUIRED_TRACE_ARRAYS - _SHARED_TRACE_ARRAYS:
                if not np.array_equal(
                    arrays[name][local_index],
                    evaluation.arrays[name][trace_index],
                ):
                    raise ValueError(
                        f"actual original forward differs from test trace for "
                        f"sample_id={sample_id!r}, field={name!r}"
                    )
            locations[sample_id] = (arrays, local_index)
        for name in _SHARED_TRACE_ARRAYS:
            if not np.array_equal(arrays[name], evaluation.arrays[name]):
                raise ValueError(
                    f"actual intervention chunk {chunk_index} shared field {name!r} "
                    "differs from test trace"
                )
        chunk_payloads.append(
            {
                "first_sample_id": str(chunk_ids[0]),
                "last_sample_id": str(chunk_ids[-1]),
                "model_state_sha256": metadata["model_state"]["before_sha256"],
                "sample_count": len(chunk_ids),
                "sample_ids_sha256": _sha256_bytes(
                    _canonical_json_bytes([str(value) for value in chunk_ids])
                ),
                "semantic_sha256": result.semantic_sha256,
            }
        )
    if set(locations) != set(evaluation_ids):
        missing = sorted(set(evaluation_ids) - set(locations))
        raise ValueError(
            "actual intervention results do not cover every test trace sample: "
            f"missing={missing[:5]}, missing_count={len(missing)}"
        )
    assert actual_names is not None
    aligned = {
        name: np.ascontiguousarray(
            np.stack(
                [locations[str(sample_id)][0][name][locations[str(sample_id)][1]] for sample_id in evaluation_ids],
                axis=0,
            )
        )
        for name in sorted(actual_names)
    }
    return _VerifiedActualInterventions(
        arrays=aligned,
        input_payload={
            "chunk_count": len(chunk_payloads),
            "chunks": sorted(
                chunk_payloads,
                key=lambda value: (value["first_sample_id"], value["semantic_sha256"]),
            ),
            "sample_count": len(evaluation_ids),
            "source": "p05.actual_same_checkpoint_forward",
            "timing_in_semantic_hash": False,
        },
    )


def _endpoint_arrays(
    results: list[Mapping[str, Any]],
    *,
    endpoint: str,
) -> dict[str, np.ndarray]:
    prefix = f"c2_{endpoint}"
    scalar_float_fields = (
        "matched_distance_median",
        "matched_distance_max",
        "top_deletion_jsd",
        "matched_pool_mean_jsd",
        "unmatched_non_top_mean_jsd",
        "d_top",
        "tau_attribution_jsd",
        "tau_firing_jsd",
        "tau_consequent_jsd",
        "d_rank",
    )
    return {
        f"{prefix}_top_rule": np.asarray(
            [result[endpoint]["top_rule"] for result in results],
            dtype="<i8",
        ),
        f"{prefix}_matched_rules": np.asarray(
            [result[endpoint]["matched_rules"] for result in results],
            dtype="<i8",
        ),
        f"{prefix}_matched_distances": np.asarray(
            [result[endpoint]["matched_distances"] for result in results],
            dtype="<f8",
        ),
        **{
            f"{prefix}_{field}": np.asarray(
                [result[endpoint][field] for result in results],
                dtype="<f8",
            )
            for field in scalar_float_fields
        },
    }


def _softmax_vector(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values)
    probability = np.exp(shifted)
    probability /= probability.sum(dtype=np.float64)
    return probability


def _tau_b(left: np.ndarray, right: np.ndarray) -> float:
    value = float(kendalltau(left, right, variant="b", nan_policy="propagate").statistic)
    return value if math.isfinite(value) else 0.0


def _actual_endpoint(
    structural: Mapping[str, Any],
    *,
    attribution: np.ndarray,
    firing: np.ndarray,
    consequent: np.ndarray,
    deletion_jsd: np.ndarray,
) -> dict[str, Any]:
    top_rule = int(structural["top_rule"])
    if top_rule != int(np.argmax(attribution)):
        raise AssertionError("offline structural matcher changed the top attribution rule")
    matched = [int(value) for value in structural["matched_rules"]]
    candidates = [rule for rule in range(len(attribution)) if rule != top_rule]
    matched_mean = float(np.mean(deletion_jsd[matched]))
    tau_attribution = _tau_b(attribution, deletion_jsd)
    tau_firing = _tau_b(firing, deletion_jsd)
    tau_consequent = _tau_b(consequent, deletion_jsd)
    return {
        "top_rule": top_rule,
        "matched_rules": matched,
        "matched_distances": [
            float(value) for value in structural["matched_distances"]
        ],
        "matched_distance_median": float(structural["matched_distance_median"]),
        "matched_distance_max": float(structural["matched_distance_max"]),
        "top_deletion_jsd": float(deletion_jsd[top_rule]),
        "matched_pool_mean_jsd": matched_mean,
        "unmatched_non_top_mean_jsd": float(np.mean(deletion_jsd[candidates])),
        "d_top": float(deletion_jsd[top_rule] - matched_mean),
        "tau_attribution_jsd": tau_attribution,
        "tau_firing_jsd": tau_firing,
        "tau_consequent_jsd": tau_consequent,
        "d_rank": float(
            tau_attribution - max(0.0, tau_firing, tau_consequent)
        ),
    }


def _evaluate_c2(
    trace: _VerifiedTracePackage,
    frozen: P05EvaluationFrozenParameters,
    actual: _VerifiedActualInterventions,
) -> dict[str, np.ndarray]:
    source = trace.arrays
    actual_arrays = actual.arrays
    sample_ids = source["sample_id"].tolist()
    scale = float(np.asarray(source["fuzzy_scale"]).reshape(()))
    results = []
    deletion_disagreement = np.empty(len(sample_ids), dtype="<f8")
    shuffle_disagreement = np.empty(len(sample_ids), dtype="<f8")
    for index, sample_id in enumerate(sample_ids):
        structural = evaluate_rule_interventions(
            dataset=frozen.dataset,
            split=frozen.evaluation_split,
            model_seed=frozen.model_seed,
            sample_id=str(sample_id),
            logits=source["logits"][index],
            non_fuzzy_logits=source["non_fuzzy_logits"][index],
            fuzzy_scale=scale,
            log_rule_firing=source["trace_log_rule_firing"][index],
            rule_consequents=source["trace_rule_consequents"],
            rule_contributions=source["trace_rule_contributions"][index],
        )
        actual_deletion_logits = actual_arrays["actual_deletion_logits"][index]
        deletion_disagreement[index] = float(
            np.max(np.abs(actual_deletion_logits - structural["deletion_logits"]))
        )
        if not np.allclose(
            actual_deletion_logits,
            structural["deletion_logits"],
            atol=frozen.reconstruction_atol,
            rtol=frozen.reconstruction_rtol,
        ):
            raise ValueError(
                f"actual deletion forward differs from registered semantics for "
                f"sample_id={sample_id!r}"
            )
        original_probability = _softmax_vector(source["logits"][index])
        deletion_jsd = np.asarray(
            [
                natural_log_jsd(
                    original_probability,
                    _softmax_vector(logits),
                )
                for logits in actual_deletion_logits
            ],
            dtype="<f8",
        )

        firing = source["trace_normalized_rule_firing"][index]
        consequents = source["trace_rule_consequents"]
        contributions = source["trace_rule_contributions"][index]
        predicted_class = int(np.argmax(source["logits"][index]))
        attribution = np.abs(scale * contributions[:, predicted_class])
        consequent = np.abs(scale * consequents[:, predicted_class])
        primary = _actual_endpoint(
            structural["primary_reference_class"],
            attribution=attribution,
            firing=firing,
            consequent=consequent,
            deletion_jsd=deletion_jsd,
        )
        full_vector = _actual_endpoint(
            structural["full_vector_sensitivity"],
            attribution=np.linalg.norm(scale * contributions, axis=1),
            firing=firing,
            consequent=np.linalg.norm(scale * consequents, axis=1),
            deletion_jsd=deletion_jsd,
        )

        actual_permutations = actual_arrays["actual_shuffle_permutations"][index]
        actual_seed = int(actual_arrays["actual_shuffle_seed"][index])
        if actual_seed != structural["shuffle"]["seed"] or not np.array_equal(
            actual_permutations,
            structural["shuffle"]["permutations"],
        ):
            raise ValueError(
                f"actual shuffle registration differs for sample_id={sample_id!r}"
            )
        expected_shuffle_logits = np.asarray(
            [
                source["non_fuzzy_logits"][index]
                + scale * (firing @ consequents[permutation])
                for permutation in actual_permutations
            ],
            dtype="<f8",
        )
        actual_shuffle_logits = actual_arrays["actual_shuffle_logits"][index]
        shuffle_disagreement[index] = float(
            np.max(np.abs(actual_shuffle_logits - expected_shuffle_logits))
        )
        if not np.allclose(
            actual_shuffle_logits,
            expected_shuffle_logits,
            atol=frozen.reconstruction_atol,
            rtol=frozen.reconstruction_rtol,
        ):
            raise ValueError(
                f"actual shuffle forward differs from registered semantics for "
                f"sample_id={sample_id!r}"
            )
        shuffle_jsd = np.asarray(
            [
                natural_log_jsd(
                    original_probability,
                    _softmax_vector(logits),
                )
                for logits in actual_shuffle_logits
            ],
            dtype="<f8",
        )
        original_fuzzy_vector = scale * contributions.sum(axis=0)
        shuffled_fuzzy_vectors = scale * actual_arrays[
            "actual_shuffle_rule_contributions"
        ][index].sum(axis=1)
        shuffle_l1 = np.abs(
            shuffled_fuzzy_vectors - original_fuzzy_vector[None, :]
        ).sum(axis=1)
        results.append(
            {
                "predicted_class": predicted_class,
                "rule_count": frozen.rule_count,
                "deletion_logits": actual_deletion_logits,
                "deletion_jsd": deletion_jsd,
                "primary_reference_class": primary,
                "full_vector_sensitivity": full_vector,
                "shuffle": {
                    "seed": actual_seed,
                    "permutations": actual_permutations,
                    "predictive_jsd": shuffle_jsd,
                    "predictive_jsd_mean": float(np.mean(shuffle_jsd)),
                    "fuzzy_class_vector_l1_change": shuffle_l1,
                    "membership_invariant": bool(
                        actual_arrays[
                            "actual_shuffle_membership_invariant_pass"
                        ][index].all()
                    ),
                    "antecedent_invariant": bool(
                        actual_arrays[
                            "actual_shuffle_antecedent_invariant_pass"
                        ][index].all()
                    ),
                    "firing_invariant": bool(
                        actual_arrays[
                            "actual_shuffle_firing_invariant_pass"
                        ][index].all()
                    ),
                },
            }
        )
    arrays = {
        "sample_id": np.asarray(source["sample_id"]),
        "record_id": np.asarray(source["record_id"]),
        "group_id": np.asarray(source["group_id"]),
        "window_start": np.asarray(source["window_start"], dtype="<i8"),
        "window_end": np.asarray(source["window_end"], dtype="<i8"),
        "label": np.asarray(source["y"], dtype="<i8"),
        "c2_predicted_class": np.asarray(
            [result["predicted_class"] for result in results], dtype="<i8"
        ),
        "c2_rule_count": np.asarray(
            [result["rule_count"] for result in results], dtype="<i8"
        ),
        "c2_deletion_logits": np.asarray(
            [result["deletion_logits"] for result in results], dtype="<f8"
        ),
        "c2_deletion_jsd": np.asarray(
            [result["deletion_jsd"] for result in results], dtype="<f8"
        ),
        "c2_actual_deletion_invariant_max_abs": np.asarray(
            actual_arrays["actual_deletion_invariant_max_abs"], dtype="<f8"
        ),
        "c2_actual_deletion_membership_invariant_pass": np.asarray(
            actual_arrays["actual_deletion_membership_invariant_pass"],
            dtype=np.bool_,
        ),
        "c2_actual_deletion_antecedent_invariant_pass": np.asarray(
            actual_arrays["actual_deletion_antecedent_invariant_pass"],
            dtype=np.bool_,
        ),
        "c2_actual_deletion_firing_invariant_pass": np.asarray(
            actual_arrays["actual_deletion_firing_invariant_pass"],
            dtype=np.bool_,
        ),
        "c2_actual_vs_offline_deletion_logits_max_abs": deletion_disagreement,
        "c2_shuffle_seed": np.asarray(
            [result["shuffle"]["seed"] for result in results], dtype="<u8"
        ),
        "c2_shuffle_permutations": np.asarray(
            [result["shuffle"]["permutations"] for result in results], dtype="<i8"
        ),
        "c2_shuffle_predictive_jsd": np.asarray(
            [result["shuffle"]["predictive_jsd"] for result in results], dtype="<f8"
        ),
        "c2_shuffle_predictive_jsd_mean": np.asarray(
            [result["shuffle"]["predictive_jsd_mean"] for result in results],
            dtype="<f8",
        ),
        "c2_shuffle_fuzzy_vector_l1": np.asarray(
            [result["shuffle"]["fuzzy_class_vector_l1_change"] for result in results],
            dtype="<f8",
        ),
        "c2_shuffle_membership_invariant": np.asarray(
            [result["shuffle"]["membership_invariant"] for result in results],
            dtype=np.bool_,
        ),
        "c2_shuffle_antecedent_invariant": np.asarray(
            [result["shuffle"]["antecedent_invariant"] for result in results],
            dtype=np.bool_,
        ),
        "c2_shuffle_firing_invariant": np.asarray(
            [result["shuffle"]["firing_invariant"] for result in results],
            dtype=np.bool_,
        ),
        "c2_actual_shuffle_invariant_max_abs": np.asarray(
            actual_arrays["actual_shuffle_invariant_max_abs"], dtype="<f8"
        ),
        "c2_actual_vs_offline_shuffle_logits_max_abs": shuffle_disagreement,
        "c2_actual_forward_bound": np.ones(len(sample_ids), dtype=np.bool_),
        **_endpoint_arrays(results, endpoint="primary_reference_class"),
        **_endpoint_arrays(results, endpoint="full_vector_sensitivity"),
    }
    if arrays["c2_shuffle_permutations"].shape[1] != frozen.shuffle_count:
        raise AssertionError("C2 evaluator did not emit the frozen shuffle count")
    return {name: np.ascontiguousarray(value) for name, value in arrays.items()}


def _ranker_payload(bundle: ValidationRiskBundle) -> dict[str, Any]:
    def ranker(value: Any) -> dict[str, Any]:
        return {
            "coefficient": list(value.coefficient),
            "feature_mean": list(value.feature_mean),
            "feature_std": list(value.feature_std),
            "intercept": value.intercept,
            "name": value.name,
        }

    return {
        "temperature": bundle.temperature,
        "thresholds": dict(bundle.thresholds),
        "trace_free_ranker": ranker(bundle.trace_free_ranker),
        "trace_ranker": ranker(bundle.trace_ranker),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _json_safe(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _evaluate_c3(
    validation: _VerifiedTracePackage,
    evaluation: _VerifiedTracePackage,
    frozen: P05EvaluationFrozenParameters,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    validation_arrays = validation.arrays
    evaluation_arrays = evaluation.arrays
    # The exported float32 firing is already checked against log-firing above.
    # Recomputing here avoids carrying float32 row-sum roundoff into the strict
    # float64 selective-risk feature contract.
    validation_firing = _normalized_firing_from_log(validation_arrays)
    evaluation_firing = _normalized_firing_from_log(evaluation_arrays)
    risk_bundle = fit_validation_risk_bundle(
        sample_ids=validation_arrays["sample_id"].tolist(),
        groups=validation_arrays["group_id"].tolist(),
        logits=validation_arrays["logits"],
        firing=validation_firing,
        labels=validation_arrays["y"],
    )
    scores = score_risk_methods(
        risk_bundle,
        evaluation_arrays["logits"],
        evaluation_firing,
    )
    predictions = np.asarray(evaluation_arrays["logits"]).argmax(axis=1)
    sample_ids = evaluation_arrays["sample_id"].tolist()
    groups = evaluation_arrays["group_id"].tolist()
    labels = np.asarray(evaluation_arrays["y"], dtype=np.int64)
    methods = {}
    for method in ("trace", "R0", "R1", "R2", "R3"):
        methods[method] = {
            "frozen_validation_threshold": frozen_threshold_metrics(
                groups=groups,
                scores=scores[method],
                predictions=predictions,
                labels=labels,
                threshold=float(risk_bundle.thresholds[method]),
            ),
            "retrospective_matched_count": retrospective_selective_metrics(
                sample_ids=sample_ids,
                groups=groups,
                scores=scores[method],
                predictions=predictions,
                labels=labels,
                coverages=frozen.coverages,
            ),
        }
    c3_payload = _json_safe(
        {
            "claim_decisions": "not_performed",
            "decisive": False,
            "evaluation": {
                "ece": equal_mass_ece(
                    sample_ids=sample_ids,
                    groups=groups,
                    confidence=scores["confidence"],
                    predictions=predictions,
                    labels=labels,
                    bins=frozen.ece_bins,
                ),
                "methods": methods,
            },
            "interpretation": {
                "confirmatory_sign_tests": "not_performed",
                "cross_seed_aggregation": "not_performed",
                "inference": "not_performed",
                "operational_wording_gate": {
                    "evaluated": False,
                    "reason": "requires separate predictive-cost and five-bearing adjudication",
                },
                "predictive_cost_gate": {
                    "evaluated": False,
                    "reason": "P05-B1 predictions are not an input to this per-seed bundle",
                },
                "scope": "computed_unadjudicated_retrospective_metrics",
                "undefined_floats": "encoded_as_null",
            },
            "schema_name": "p05.c3_retrospective_metrics",
            "schema_version": 1,
            "validation_fit": _ranker_payload(risk_bundle),
        }
    )
    score_arrays = {
        "c3_prediction": np.asarray(predictions, dtype="<i8"),
        "c3_confidence": np.asarray(scores["confidence"], dtype="<f8"),
        **{
            f"c3_score_{method}": np.asarray(scores[method], dtype="<f8")
            for method in ("trace", "R0", "R1", "R2", "R3")
        },
    }
    return score_arrays, c3_payload


def _trace_input_payload(
    package: _VerifiedTracePackage,
    *,
    role: str,
) -> dict[str, Any]:
    return {
        "manifest_sha256": package.manifest_sha256,
        "npz_sha256": package.npz_sha256,
        "package_role": role,
        "provenance": dict(package.manifest["provenance"]),
        "sample_count": int(package.manifest["sample_count"]),
        "semantic_sha256": package.semantic_sha256,
    }


def _semantic_manifest(
    *,
    validation: _VerifiedTracePackage,
    evaluation: _VerifiedTracePackage,
    actual: _VerifiedActualInterventions,
    frozen: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    c3_payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "conclusion_control": {
            "c2_intervention_source": "actual_same_checkpoint_forwards",
            "claim_decisions": "not_performed",
            "decisive": False,
            "operational_wording_gate": "not_evaluated",
            "predictive_cost_gate": "not_evaluated",
            "status": "computed_unadjudicated",
        },
        "frozen_parameters": dict(frozen),
        "inputs": {
            "actual_interventions": dict(actual.input_payload),
            "evaluation_trace": _trace_input_payload(evaluation, role="test"),
            "validation_trace": _trace_input_payload(validation, role="validation"),
        },
        "outputs": {
            "arrays": _array_descriptors(arrays),
            "arrays_file": ARRAYS_NAME,
            "c3_file": C3_NAME,
            "c3_semantic_sha256": _sha256_bytes(_canonical_json_bytes(c3_payload)),
        },
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
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
        for name in left
    )


def _result(
    target: Path,
    manifest: Mapping[str, Any],
    *,
    status: str,
) -> P05EvaluationBundleResult:
    return P05EvaluationBundleResult(
        bundle_dir=target,
        arrays_path=target / ARRAYS_NAME,
        c3_path=target / C3_NAME,
        manifest_path=target / MANIFEST_NAME,
        semantic_sha256=str(manifest["content"]["semantic_sha256"]),
        arrays_sha256=str(manifest["content"]["arrays_sha256"]),
        c3_sha256=str(manifest["content"]["c3_sha256"]),
        manifest_sha256=_sha256_file(target / MANIFEST_NAME),
        status=status,
    )


def _reuse_existing(
    target: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    c3_payload: Mapping[str, Any],
    semantic_manifest: Mapping[str, Any],
) -> P05EvaluationBundleResult:
    if target.is_symlink() or not target.is_dir():
        raise FileExistsError(f"evaluation bundle target conflicts: {target}")
    entries = {entry.name: entry for entry in target.iterdir()}
    if set(entries) != {ARRAYS_NAME, C3_NAME, MANIFEST_NAME}:
        raise FileExistsError(f"existing evaluation bundle is incomplete or unexpected: {target}")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries.values()):
        raise FileExistsError(f"existing evaluation bundle contains a non-file entry: {target}")
    try:
        manifest = json.loads(entries[MANIFEST_NAME].read_text(encoding="utf-8"))
        existing_c3 = json.loads(entries[C3_NAME].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FileExistsError(f"existing evaluation bundle JSON is invalid: {target}") from exc
    if not isinstance(manifest, dict) or set(manifest) != set(semantic_manifest) | {"content"}:
        raise FileExistsError(f"existing evaluation manifest schema conflicts: {target}")
    content = manifest.get("content")
    if not isinstance(content, dict) or set(content) != {
        "arrays_sha256",
        "c3_sha256",
        "semantic_sha256",
    }:
        raise FileExistsError(f"existing evaluation content hashes are invalid: {target}")
    for name, file_name in (
        ("arrays_sha256", ARRAYS_NAME),
        ("c3_sha256", C3_NAME),
    ):
        try:
            recorded_hash = _required_sha256(content.get(name), name=f"content.{name}")
        except ValueError as exc:
            raise FileExistsError(
                f"existing evaluation {name} is not a SHA-256: {target}"
            ) from exc
        if _sha256_file(entries[file_name]) != recorded_hash:
            raise FileExistsError(f"existing evaluation {name} does not match: {target}")
    existing_semantic = {
        name: value for name, value in manifest.items() if name != "content"
    }
    try:
        recorded_semantic_hash = _required_sha256(
            content.get("semantic_sha256"),
            name="content.semantic_sha256",
        )
    except ValueError as exc:
        raise FileExistsError(
            f"existing evaluation semantic hash is not a SHA-256: {target}"
        ) from exc
    if (
        _sha256_bytes(_canonical_json_bytes(existing_semantic))
        != recorded_semantic_hash
    ):
        raise FileExistsError(f"existing evaluation semantic hash is invalid: {target}")
    if _canonical_json_bytes(existing_semantic) != _canonical_json_bytes(semantic_manifest):
        raise FileExistsError(f"existing evaluation provenance or parameters conflict: {target}")
    existing_arrays = _load_npz(entries[ARRAYS_NAME], identity="evaluation NPZ")
    if not _arrays_equal(existing_arrays, arrays):
        raise FileExistsError(f"existing evaluation arrays conflict: {target}")
    if _canonical_json_bytes(existing_c3) != _canonical_json_bytes(c3_payload):
        raise FileExistsError(f"existing C3 metrics conflict: {target}")
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
        raise RuntimeError("atomic create-only evaluation requires Linux renameat2")
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


def _write_new_bundle(
    target: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    c3_payload: Mapping[str, Any],
    semantic_manifest: Mapping[str, Any],
) -> P05EvaluationBundleResult:
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"evaluation bundle parent must be a real directory: {parent}")
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
        c3_path = temporary / C3_NAME
        with c3_path.open("wb") as handle:
            handle.write(_pretty_json_bytes(c3_payload))
            handle.flush()
            os.fsync(handle.fileno())
        semantic_sha256 = _sha256_bytes(_canonical_json_bytes(semantic_manifest))
        manifest = {
            **semantic_manifest,
            "content": {
                "arrays_sha256": _sha256_file(arrays_path),
                "c3_sha256": _sha256_file(c3_path),
                "semantic_sha256": semantic_sha256,
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
                c3_payload=c3_payload,
                semantic_manifest=semantic_manifest,
            )
        _fsync_directory(parent)
        return _result(target, manifest, status="created")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def create_p05_c2_c3_evaluation_bundle(
    bundle_dir: str | Path,
    *,
    validation_trace_package: str | Path,
    evaluation_trace_package: str | Path,
    actual_intervention_results: Sequence[P05ActualInterventionResult],
    frozen: P05EvaluationFrozenParameters,
) -> P05EvaluationBundleResult:
    """Create or exactly reuse an unadjudicated C2/C3 evaluation bundle.

    C2 effects are computed from actual same-checkpoint deletion/shuffle
    forwards. Offline algebra is retained only as a fail-closed semantic
    cross-check. This coordinator does not aggregate across seeds, run a sign
    test, compare the B1 predictive-cost bound, call the operational wording
    gate, or decide claims.
    """

    frozen_payload = _validate_frozen_parameters(frozen)
    validation = _verify_trace_package(validation_trace_package)
    evaluation = _verify_trace_package(evaluation_trace_package)
    if validation.package_dir == evaluation.package_dir:
        raise ValueError("validation and evaluation trace packages must be distinct")
    if validation.semantic_sha256 != frozen_payload[
        "validation_trace_semantic_sha256"
    ]:
        raise ValueError("validation trace semantic hash does not match frozen input")
    if evaluation.semantic_sha256 != frozen_payload[
        "evaluation_trace_semantic_sha256"
    ]:
        raise ValueError("evaluation trace semantic hash does not match frozen input")
    _validate_trace_semantics(validation, frozen, role="validation")
    _validate_trace_semantics(evaluation, frozen, role="evaluation")
    validation_provenance = validation.manifest["provenance"]
    evaluation_provenance = evaluation.manifest["provenance"]
    for name in ("checkpoint_sha256", "config_sha256", "model_sha256"):
        if validation_provenance[name] != evaluation_provenance[name]:
            raise ValueError(f"validation/evaluation trace {name} values differ")
    for name in (
        "fuzzy_scale",
        "trace_centers",
        "trace_widths",
        "trace_antecedent_probabilities",
        "trace_rule_consequents",
        "trace_consequent_permutation",
    ):
        if not np.array_equal(validation.arrays[name], evaluation.arrays[name]):
            raise ValueError(f"validation/evaluation shared trace array {name!r} differs")
    validation_ids = set(validation.arrays["sample_id"].tolist())
    evaluation_ids = set(evaluation.arrays["sample_id"].tolist())
    if validation_ids & evaluation_ids:
        raise ValueError("validation and evaluation sample_id values overlap")
    validation_groups = set(validation.arrays["group_id"].tolist())
    evaluation_groups = set(evaluation.arrays["group_id"].tolist())
    if validation_groups & evaluation_groups:
        raise ValueError("validation and evaluation group_id values overlap")

    actual = _collect_actual_interventions(
        actual_intervention_results,
        evaluation=evaluation,
        frozen=frozen,
    )
    arrays = _evaluate_c2(evaluation, frozen, actual)
    c3_arrays, c3_payload = _evaluate_c3(validation, evaluation, frozen)
    arrays.update(c3_arrays)
    arrays = {
        name: np.ascontiguousarray(value)
        for name, value in sorted(arrays.items())
    }
    if any(
        array.dtype.kind in {"b", "i", "u", "f"}
        and not np.isfinite(array).all()
        for array in arrays.values()
    ):
        raise FloatingPointError("evaluation arrays contain non-finite values")
    semantic_manifest = _semantic_manifest(
        validation=validation,
        evaluation=evaluation,
        actual=actual,
        frozen=frozen_payload,
        arrays=arrays,
        c3_payload=c3_payload,
    )

    target = Path(os.path.abspath(os.fspath(bundle_dir)))
    for input_path in (validation.package_dir, evaluation.package_dir):
        if target == input_path or target.is_relative_to(input_path):
            raise ValueError("evaluation bundle cannot be created inside a trace package")
    if target.is_symlink():
        raise FileExistsError(f"refusing create-only evaluation through symlink: {target}")
    if target.exists():
        return _reuse_existing(
            target,
            arrays=arrays,
            c3_payload=c3_payload,
            semantic_manifest=semantic_manifest,
        )
    return _write_new_bundle(
        target,
        arrays=arrays,
        c3_payload=c3_payload,
        semantic_manifest=semantic_manifest,
    )


__all__ = [
    "P05EvaluationBundleResult",
    "P05EvaluationFrozenParameters",
    "create_p05_c2_c3_evaluation_bundle",
]
