"""Create-only timing summary for the non-evidentiary P05 pilot evaluators."""

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
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from . import p05_d03_noise_intervention as d03_module
from .p05_intervention_runner import (
    P05ActualInterventionResult,
    verify_p05_actual_intervention_result,
)


SCHEMA_NAME = "p05.pilot_evaluator_benchmark"
SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
PILOT_MODEL_SEED = 20260801
BENCHMARK_SAMPLE_COUNT = 256
CENTRAL_FORWARD_CALLS = 43
D03_FORWARD_CALLS = 33
GPU_HOUR_BUDGET_CAP = 168

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID = re.compile(r"^GPU-[!-~]+$")


@dataclass(frozen=True)
class P05PilotEvaluatorBenchmarkResult:
    """Location and hashes for one immutable engineering-only timing summary."""

    package_dir: Path
    manifest_path: Path
    semantic_sha256: str
    manifest_sha256: str
    status: str


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


def _required_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _required_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _nonnegative_seconds(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be a finite non-negative duration")
    return float(value)


def _positive_integer(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _strict_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            "pilot evaluator benchmark manifest is not strict finite JSON"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("pilot evaluator benchmark manifest must be a JSON object")
    return value


def _sample_ids(value: Any, *, name: str) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.shape != (BENCHMARK_SAMPLE_COUNT,) or array.dtype.kind != "U":
        raise ValueError(
            f"{name} must contain exactly {BENCHMARK_SAMPLE_COUNT} Unicode IDs"
        )
    identifiers = tuple(array.tolist())
    if any(not item or "\x00" in item for item in identifiers):
        raise ValueError(f"{name} contains an invalid stable sample ID")
    if len(set(identifiers)) != BENCHMARK_SAMPLE_COUNT:
        raise ValueError(f"{name} contains duplicate stable sample IDs")
    if tuple(sorted(identifiers)) != identifiers:
        raise ValueError(f"{name} must be in stable sample-ID sort order")
    return identifiers


def _central_stable_sample_ids(result: P05ActualInterventionResult) -> tuple[str, ...]:
    identifiers = _sample_ids(result.arrays["sample_id"], name="central")
    record_ids = np.asarray(result.arrays["record_id"])
    starts = np.asarray(result.arrays["window_start"])
    ends = np.asarray(result.arrays["window_end"])
    if (
        record_ids.shape != (BENCHMARK_SAMPLE_COUNT,)
        or record_ids.dtype.kind != "U"
        or starts.shape != (BENCHMARK_SAMPLE_COUNT,)
        or ends.shape != (BENCHMARK_SAMPLE_COUNT,)
        or starts.dtype.kind not in {"i", "u"}
        or ends.dtype.kind not in {"i", "u"}
    ):
        raise ValueError("central stable-ID component arrays are invalid")
    expected = tuple(
        f"{record_id}:{int(start)}:{int(end)}"
        for record_id, start, end in zip(record_ids, starts, ends, strict=True)
    )
    if identifiers != expected:
        raise ValueError("central sample IDs do not match record and window boundaries")
    return identifiers


def _sample_id_semantic_sha256(sample_ids: Sequence[str]) -> str:
    return _sha256_bytes(_canonical_json_bytes(list(sample_ids)))


def _validate_central_source(
    result: P05ActualInterventionResult,
) -> tuple[Mapping[str, Any], tuple[str, ...], Mapping[str, float], int]:
    if not isinstance(result, P05ActualInterventionResult):
        raise TypeError("central_result must be P05ActualInterventionResult")
    verify_p05_actual_intervention_result(result)

    metadata = _required_mapping(result.metadata, name="central metadata")
    provenance = _required_mapping(
        metadata.get("provenance"), name="central provenance"
    )
    selection = _required_mapping(metadata.get("selection"), name="central selection")
    protocol = _required_mapping(metadata.get("protocol"), name="central protocol")
    if (
        selection.get("benchmark_first_n") != BENCHMARK_SAMPLE_COUNT
        or selection.get("selected_count") != BENCHMARK_SAMPLE_COUNT
        or selection.get("kind") != "first_n_after_stable_sample_id_sort"
    ):
        raise ValueError("central source is not the frozen first-256 pilot benchmark")
    input_count = selection.get("input_count")
    if type(input_count) is not int or input_count < BENCHMARK_SAMPLE_COUNT:
        raise ValueError("central source input_count is incomplete")
    if protocol.get("actual_forward_calls") != CENTRAL_FORWARD_CALLS:
        raise ValueError("central source must record exactly 43 actual forward calls")
    if provenance.get("dataset") not in {"CWRU", "XJTU"}:
        raise ValueError("central source dataset must be CWRU or XJTU")
    if provenance.get("split") != "validation":
        raise ValueError("central pilot benchmark must use the validation split")
    if provenance.get("model_seed") != PILOT_MODEL_SEED:
        raise ValueError("central pilot benchmark must use model seed 20260801")

    timing = _required_mapping(result.timing, name="central timing")
    if (
        timing.get("device_type") != "cuda"
        or timing.get("performance_claim_allowed") is not False
        or timing.get("scope") != "diagnostic_wall_clock_boundary_only"
    ):
        raise ValueError("central timing is not a CUDA diagnostic-only boundary")
    original = _nonnegative_seconds(
        timing.get("original_seconds"), name="central original_seconds"
    )
    deletion = _nonnegative_seconds(
        timing.get("deletion_seconds"), name="central deletion_seconds"
    )
    shuffle = _nonnegative_seconds(
        timing.get("shuffle_seconds"), name="central shuffle_seconds"
    )
    total = _nonnegative_seconds(
        timing.get("total_seconds"), name="central total_seconds"
    )
    if not math.isclose(total, original + deletion + shuffle, rel_tol=1.0e-12):
        raise ValueError("central total_seconds differs from its timing components")
    return (
        provenance,
        _central_stable_sample_ids(result),
        {
            "original_trace": original,
            "rule_deletions": deletion,
            "consequent_shuffles": shuffle,
            "total": total,
        },
        input_count,
    )


def _absolute_path(value: Any, *, name: str) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(value)))
    except TypeError as exc:
        raise TypeError(f"{name} must be path-like") from exc


def _validate_d03_result_paths(result: d03_module.P05D03Result) -> Path:
    target = _absolute_path(result.artifact_dir, name="d03_result.artifact_dir")
    arrays_path = _absolute_path(result.arrays_path, name="d03_result.arrays_path")
    manifest_path = _absolute_path(
        result.manifest_path, name="d03_result.manifest_path"
    )
    if target.is_symlink() or not target.is_dir():
        raise FileNotFoundError(f"D03 source must be a real directory: {target}")
    expected_arrays_path = target / d03_module.ARRAYS_NAME
    expected_manifest_path = target / d03_module.MANIFEST_NAME
    if arrays_path != expected_arrays_path or manifest_path != expected_manifest_path:
        raise ValueError("D03 result paths do not bind to its artifact directory")
    if any(
        path.is_symlink() or not path.is_file()
        for path in (arrays_path, manifest_path)
    ):
        raise ValueError("D03 result paths must identify real files")
    if _sha256_file(arrays_path) != _required_sha256(
        result.arrays_sha256, name="d03_result.arrays_sha256"
    ):
        raise ValueError("D03 arrays SHA-256 differs from the result object")
    if _sha256_file(manifest_path) != _required_sha256(
        result.manifest_sha256, name="d03_result.manifest_sha256"
    ):
        raise ValueError("D03 manifest SHA-256 differs from the result object")
    return target


def _validate_d03_source(
    result: d03_module.P05D03Result,
) -> tuple[
    Mapping[str, Any],
    tuple[str, ...],
    Mapping[str, float],
    int,
    int,
    int,
]:
    if not isinstance(result, d03_module.P05D03Result):
        raise TypeError("d03_result must be P05D03Result")
    if result.status != "created":
        raise ValueError("D03 source must have immutable created status")
    target = _validate_d03_result_paths(result)
    manifest = d03_module.verify_p05_d03_artifact(target)
    if not isinstance(manifest, Mapping):
        raise ValueError("verified D03 manifest must be a mapping")
    content = _required_mapping(manifest.get("content"), name="D03 content")
    if _required_sha256(
        result.semantic_sha256, name="d03_result.semantic_sha256"
    ) != _required_sha256(
        content.get("semantic_sha256"), name="D03 content.semantic_sha256"
    ):
        raise ValueError("D03 semantic SHA-256 differs from its verified manifest")
    if result.arrays_sha256 != _required_sha256(
        content.get("npz_sha256"), name="D03 content.npz_sha256"
    ):
        raise ValueError("D03 arrays SHA-256 differs from its verified manifest")

    provenance = _required_mapping(manifest.get("provenance"), name="D03 provenance")
    execution = _required_mapping(manifest.get("execution"), name="D03 execution")
    input_binding = _required_mapping(
        manifest.get("input_binding"), name="D03 input binding"
    )
    partition_coverage = _required_mapping(
        manifest.get("partition_coverage"), name="D03 partition coverage"
    )
    if manifest.get("sample_count") != BENCHMARK_SAMPLE_COUNT:
        raise ValueError("D03 pilot benchmark must contain exactly 256 samples")
    if (
        input_binding.get("selected_count") != BENCHMARK_SAMPLE_COUNT
        or input_binding.get("selection")
        != "first_256_after_stable_sample_id_sort"
    ):
        raise ValueError("D03 source is not the frozen first-256 pilot benchmark")
    input_count = input_binding.get("input_count")
    if type(input_count) is not int or input_count < BENCHMARK_SAMPLE_COUNT:
        raise ValueError("D03 source input_count is incomplete")
    if execution.get("phase") != "pilot_benchmark":
        raise ValueError("D03 source phase must be pilot_benchmark")
    if execution.get("budget_retained") is not None:
        raise ValueError("D03 pilot source must precede the budget decision")
    if execution.get("device_class") != "cuda":
        raise ValueError("D03 pilot benchmark must use CUDA")
    if provenance.get("dataset") not in {"CWRU", "XJTU"}:
        raise ValueError("D03 source dataset must be CWRU or XJTU")
    if provenance.get("split") != "validation":
        raise ValueError("D03 pilot benchmark must use the validation split")
    if provenance.get("model_seed") != PILOT_MODEL_SEED:
        raise ValueError("D03 pilot benchmark must use model seed 20260801")

    protocol = _required_mapping(manifest.get("protocol"), name="D03 protocol")
    if protocol.get("total_noise_draws_per_sample") != 32:
        raise ValueError("D03 source must preserve all 32 registered noise draws")
    chunk_count = execution.get("chunk_count", 1)
    chunk_count = _positive_integer(chunk_count, name="D03 chunk_count")
    chunk_size = _positive_integer(
        execution.get("chunk_size", BENCHMARK_SAMPLE_COUNT),
        name="D03 chunk_size",
    )
    if chunk_count != 1 or chunk_size != BENCHMARK_SAMPLE_COUNT:
        raise ValueError("D03 pilot benchmark must execute one exact batch of 256")
    actual_forward_calls = _positive_integer(
        execution.get("actual_forward_calls"), name="D03 actual_forward_calls"
    )
    if actual_forward_calls != D03_FORWARD_CALLS:
        raise ValueError("D03 pilot benchmark must record exactly 33 forward calls")

    timing = _required_mapping(result.timing, name="D03 timing")
    if (
        timing.get("performance_claim_allowed") is not False
        or timing.get("scope") != "diagnostic_wall_clock_boundary_only"
    ):
        raise ValueError("D03 timing is not a diagnostic-only boundary")
    original = _nonnegative_seconds(
        timing.get("original_forward_seconds"),
        name="D03 original_forward_seconds",
    )
    noise = _nonnegative_seconds(
        timing.get("noise_forward_seconds"), name="D03 noise_forward_seconds"
    )
    total = _nonnegative_seconds(timing.get("total_seconds"), name="D03 total_seconds")
    if not math.isclose(total, original + noise, rel_tol=1.0e-12):
        raise ValueError("D03 total_seconds differs from its timing components")

    try:
        with np.load(result.arrays_path, allow_pickle=False) as archive:
            identifiers = np.array(archive["sample_id"], copy=True)
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError("D03 source does not expose safe sample_id arrays") from exc
    stable_ids = _sample_ids(identifiers, name="D03")
    stable_id_sha256 = _sample_id_semantic_sha256(stable_ids)
    if (
        partition_coverage.get("coverage") != "exact"
        or partition_coverage.get("expected_sample_count") != input_count
        or partition_coverage.get("observed_sample_count") != input_count
        or partition_coverage.get("selected_sample_count")
        != BENCHMARK_SAMPLE_COUNT
        or partition_coverage.get("expected_sample_id_semantic_sha256")
        != partition_coverage.get("observed_sample_id_semantic_sha256")
        or partition_coverage.get("selected_sample_id_semantic_sha256")
        != stable_id_sha256
    ):
        raise ValueError("D03 partition coverage does not bind the complete selection")
    return (
        provenance,
        stable_ids,
        {"original_trace": original, "noise_draws": noise, "total": total},
        chunk_count,
        actual_forward_calls,
        input_count,
    )


def _shared_provenance(
    central: Mapping[str, Any],
    d03: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name in ("config_sha256", "checkpoint_sha256", "model_sha256"):
        central_value = _required_sha256(
            central.get(name), name=f"central provenance {name}"
        )
        d03_value = _required_sha256(d03.get(name), name=f"D03 provenance {name}")
        if central_value != d03_value:
            raise ValueError(f"central and D03 provenance differ for {name}")
        payload[name] = central_value
    if central.get("dataset") != d03.get("dataset"):
        raise ValueError("central and D03 pilot sources must use the same dataset")
    physical_gpu_index = d03.get("physical_gpu_index")
    device_uuid = d03.get("device_uuid")
    if (
        type(physical_gpu_index) is not int
        or physical_gpu_index not in {0, 1}
        or not isinstance(device_uuid, str)
        or _GPU_UUID.fullmatch(device_uuid) is None
    ):
        raise ValueError("D03 pilot provenance has no valid blocked GPU identity")
    payload.update(
        {
            "device_uuid": device_uuid,
            "physical_gpu_index": physical_gpu_index,
        }
    )
    return payload


def _component_timing(
    *,
    total_seconds: float,
    forward_calls_per_window: int,
) -> dict[str, Any]:
    calls = _positive_integer(
        forward_calls_per_window,
        name="component forward_calls_per_window",
    )
    seconds = _nonnegative_seconds(total_seconds, name="component total_seconds")
    seconds_per_window = seconds / BENCHMARK_SAMPLE_COUNT
    return {
        "forward_calls_per_window": calls,
        "seconds_per_forward_call_per_window": seconds_per_window / calls,
        "seconds_per_window": seconds_per_window,
        "total_seconds": seconds,
    }


def _semantic_manifest(
    *,
    central_result: P05ActualInterventionResult,
    d03_result: d03_module.P05D03Result,
) -> dict[str, Any]:
    (
        central_provenance,
        central_ids,
        central_timing,
        central_input_count,
    ) = _validate_central_source(central_result)
    (
        d03_provenance,
        d03_ids,
        d03_timing,
        d03_chunk_count,
        d03_forward_calls,
        d03_input_count,
    ) = _validate_d03_source(d03_result)
    if central_ids != d03_ids:
        raise ValueError(
            "central and D03 pilot sources must bind the same 256 sample IDs"
        )
    if central_input_count != d03_input_count:
        raise ValueError(
            "central and D03 pilot sources must cover the same validation partition"
        )
    shared = _shared_provenance(central_provenance, d03_provenance)
    return {
        "benchmarks": {
            "central_e1_e2": {
                "actual_forward_calls": CENTRAL_FORWARD_CALLS,
                "components": {
                    "consequent_shuffles": _component_timing(
                        total_seconds=central_timing["consequent_shuffles"],
                        forward_calls_per_window=32,
                    ),
                    "original_trace": _component_timing(
                        total_seconds=central_timing["original_trace"],
                        forward_calls_per_window=1,
                    ),
                    "rule_deletions": _component_timing(
                        total_seconds=central_timing["rule_deletions"],
                        forward_calls_per_window=10,
                    ),
                },
                "seconds_per_window": (
                    central_timing["total"] / BENCHMARK_SAMPLE_COUNT
                ),
                "source_semantic_sha256": _required_sha256(
                    central_result.semantic_sha256,
                    name="central_result.semantic_sha256",
                ),
                "total_seconds": central_timing["total"],
            },
            "d03": {
                "actual_forward_calls": d03_forward_calls,
                "chunk_count": d03_chunk_count,
                "components": {
                    "noise_draws": _component_timing(
                        total_seconds=d03_timing["noise_draws"],
                        forward_calls_per_window=32,
                    ),
                    "original_trace": _component_timing(
                        total_seconds=d03_timing["original_trace"],
                        forward_calls_per_window=1,
                    ),
                },
                "seconds_per_window": d03_timing["total"] / BENCHMARK_SAMPLE_COUNT,
                "source_semantic_sha256": _required_sha256(
                    d03_result.semantic_sha256,
                    name="d03_result.semantic_sha256",
                ),
                "total_seconds": d03_timing["total"],
            },
        },
        "conclusion_control": {
            "budget_decision": "not_performed",
            "claim_decisions": "forbidden",
            "paper_evidence": False,
            "performance_conclusion": "forbidden",
            "scientific_status": "unadjudicated",
        },
        "purpose": {
            "budget_cap_gpu_hours": GPU_HOUR_BUDGET_CAP,
            "makes_budget_decision": False,
            "role": "observed_input_to_gpu_hour_budget_forecast_only",
        },
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "dataset": central_provenance["dataset"],
            "model_seed": PILOT_MODEL_SEED,
            "partition_sample_count": central_input_count,
            "sample_count": BENCHMARK_SAMPLE_COUNT,
            "sample_id_semantic_sha256": _sample_id_semantic_sha256(central_ids),
            "selection": "first_256_after_stable_sample_id_sort",
            "split": "validation",
        },
        "shared_provenance": shared,
        "status": "engineering_non_evidence",
    }


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
        raise RuntimeError(
            "atomic create-only pilot evaluator export requires renameat2"
        )
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


def _assert_create_only_target(target: Path) -> None:
    if target.is_symlink() or target.exists():
        raise FileExistsError(
            f"P05 pilot evaluator benchmark target is create-only: {target}"
        )


def create_p05_pilot_evaluator_benchmark(
    package_dir: str | Path,
    *,
    central_result: P05ActualInterventionResult,
    d03_result: d03_module.P05D03Result,
) -> P05PilotEvaluatorBenchmarkResult:
    """Validate paired timings and atomically publish a non-evidence manifest."""

    target = _absolute_path(package_dir, name="package_dir")
    _assert_create_only_target(target)
    semantic_manifest = _semantic_manifest(
        central_result=central_result,
        d03_result=d03_result,
    )
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"pilot evaluator parent must be a real directory: {parent}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(parent))
    )
    try:
        semantic_sha256 = _sha256_bytes(_canonical_json_bytes(semantic_manifest))
        manifest = {
            **semantic_manifest,
            "content": {"semantic_sha256": semantic_sha256},
        }
        manifest_path = temporary / MANIFEST_NAME
        with manifest_path.open("xb") as handle:
            handle.write(_pretty_json_bytes(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary)
        _rename_directory_noreplace(temporary, target)
        _fsync_directory(parent)
        installed_manifest = target / MANIFEST_NAME
        return P05PilotEvaluatorBenchmarkResult(
            package_dir=target,
            manifest_path=installed_manifest,
            semantic_sha256=semantic_sha256,
            manifest_sha256=_sha256_file(installed_manifest),
            status="created",
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_p05_pilot_evaluator_benchmark(
    package_dir: str | Path,
) -> Mapping[str, Any]:
    """Verify the standalone manifest without making a budget or claim decision."""

    target = _absolute_path(package_dir, name="package_dir")
    if target.is_symlink() or not target.is_dir():
        raise FileNotFoundError(
            f"pilot evaluator benchmark must be a real directory: {target}"
        )
    entries = {entry.name: entry for entry in target.iterdir()}
    if set(entries) != {MANIFEST_NAME}:
        raise ValueError(
            "pilot evaluator benchmark has unexpected or incomplete content"
        )
    manifest_path = entries[MANIFEST_NAME]
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("pilot evaluator benchmark manifest must be a real file")
    manifest = _strict_json_object(manifest_path)
    expected_keys = {
        "benchmarks",
        "conclusion_control",
        "content",
        "purpose",
        "schema_name",
        "schema_version",
        "scope",
        "shared_provenance",
        "status",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise ValueError("pilot evaluator benchmark schema is incomplete or unexpected")
    if (
        manifest.get("schema_name") != SCHEMA_NAME
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "engineering_non_evidence"
    ):
        raise ValueError(
            "pilot evaluator benchmark schema identity or status is invalid"
        )
    content = _required_mapping(manifest["content"], name="content")
    if set(content) != {"semantic_sha256"}:
        raise ValueError("pilot evaluator benchmark content block is invalid")
    semantic_manifest = {name: manifest[name] for name in manifest if name != "content"}
    if _required_sha256(
        content["semantic_sha256"], name="content.semantic_sha256"
    ) != _sha256_bytes(_canonical_json_bytes(semantic_manifest)):
        raise ValueError("pilot evaluator benchmark semantic SHA-256 does not match")
    if manifest["conclusion_control"] != {
        "budget_decision": "not_performed",
        "claim_decisions": "forbidden",
        "paper_evidence": False,
        "performance_conclusion": "forbidden",
        "scientific_status": "unadjudicated",
    }:
        raise ValueError("pilot evaluator benchmark conclusion control is invalid")
    if manifest["purpose"] != {
        "budget_cap_gpu_hours": GPU_HOUR_BUDGET_CAP,
        "makes_budget_decision": False,
        "role": "observed_input_to_gpu_hour_budget_forecast_only",
    }:
        raise ValueError("pilot evaluator benchmark purpose exceeds its allowed scope")

    scope = _required_mapping(manifest["scope"], name="scope")
    if (
        set(scope)
        != {
            "dataset",
            "model_seed",
            "partition_sample_count",
            "sample_count",
            "sample_id_semantic_sha256",
            "selection",
            "split",
        }
        or scope["dataset"] not in {"CWRU", "XJTU"}
        or scope["model_seed"] != PILOT_MODEL_SEED
        or type(scope["partition_sample_count"]) is not int
        or scope["partition_sample_count"] < BENCHMARK_SAMPLE_COUNT
        or scope["sample_count"] != BENCHMARK_SAMPLE_COUNT
        or scope["selection"] != "first_256_after_stable_sample_id_sort"
        or scope["split"] != "validation"
    ):
        raise ValueError("pilot evaluator benchmark scope is invalid")
    _required_sha256(
        scope["sample_id_semantic_sha256"],
        name="scope.sample_id_semantic_sha256",
    )
    shared = _required_mapping(manifest["shared_provenance"], name="shared provenance")
    if set(shared) != {
        "checkpoint_sha256",
        "config_sha256",
        "device_uuid",
        "model_sha256",
        "physical_gpu_index",
    }:
        raise ValueError("pilot evaluator benchmark shared provenance is invalid")
    for name in ("checkpoint_sha256", "config_sha256", "model_sha256"):
        _required_sha256(shared[name], name=f"shared_provenance.{name}")
    if (
        type(shared["physical_gpu_index"]) is not int
        or shared["physical_gpu_index"] not in {0, 1}
        or not isinstance(shared["device_uuid"], str)
        or _GPU_UUID.fullmatch(shared["device_uuid"]) is None
    ):
        raise ValueError("pilot evaluator benchmark GPU provenance is invalid")

    benchmarks = _required_mapping(manifest["benchmarks"], name="benchmarks")
    if set(benchmarks) != {"central_e1_e2", "d03"}:
        raise ValueError("pilot evaluator benchmark sources are incomplete")
    central = _required_mapping(benchmarks["central_e1_e2"], name="central benchmark")
    d03 = _required_mapping(benchmarks["d03"], name="D03 benchmark")
    if set(central) != {
        "actual_forward_calls",
        "components",
        "seconds_per_window",
        "source_semantic_sha256",
        "total_seconds",
    } or central["actual_forward_calls"] != CENTRAL_FORWARD_CALLS:
        raise ValueError("central evaluator benchmark contract is invalid")
    if set(d03) != {
        "actual_forward_calls",
        "chunk_count",
        "components",
        "seconds_per_window",
        "source_semantic_sha256",
        "total_seconds",
    }:
        raise ValueError("D03 evaluator benchmark contract is invalid")
    chunk_count = _positive_integer(d03["chunk_count"], name="D03 chunk_count")
    if chunk_count != 1 or d03["actual_forward_calls"] != D03_FORWARD_CALLS:
        raise ValueError("D03 evaluator forward counts are invalid")

    expected_components = {
        "central": {
            "consequent_shuffles": 32,
            "original_trace": 1,
            "rule_deletions": 10,
        },
        "D03": {"noise_draws": 32, "original_trace": 1},
    }
    for name, payload in (("central", central), ("D03", d03)):
        total = _nonnegative_seconds(
            payload["total_seconds"], name=f"{name} total_seconds"
        )
        per_window = _nonnegative_seconds(
            payload["seconds_per_window"], name=f"{name} seconds_per_window"
        )
        if not math.isclose(
            per_window,
            total / BENCHMARK_SAMPLE_COUNT,
            rel_tol=1.0e-12,
        ):
            raise ValueError(f"{name} seconds_per_window is inconsistent")
        _required_sha256(
            payload["source_semantic_sha256"],
            name=f"{name} source_semantic_sha256",
        )
        components = _required_mapping(
            payload["components"], name=f"{name} components"
        )
        if set(components) != set(expected_components[name]):
            raise ValueError(f"{name} evaluator components are incomplete")
        component_total = 0.0
        component_calls = 0
        for component_name, expected_calls in expected_components[name].items():
            component = _required_mapping(
                components[component_name],
                name=f"{name}.{component_name}",
            )
            if set(component) != {
                "forward_calls_per_window",
                "seconds_per_forward_call_per_window",
                "seconds_per_window",
                "total_seconds",
            } or component["forward_calls_per_window"] != expected_calls:
                raise ValueError(f"{name}.{component_name} timing contract is invalid")
            component_seconds = _nonnegative_seconds(
                component["total_seconds"],
                name=f"{name}.{component_name}.total_seconds",
            )
            component_per_window = _nonnegative_seconds(
                component["seconds_per_window"],
                name=f"{name}.{component_name}.seconds_per_window",
            )
            component_unit = _nonnegative_seconds(
                component["seconds_per_forward_call_per_window"],
                name=(
                    f"{name}.{component_name}."
                    "seconds_per_forward_call_per_window"
                ),
            )
            if not math.isclose(
                component_per_window,
                component_seconds / BENCHMARK_SAMPLE_COUNT,
                rel_tol=1.0e-12,
            ) or not math.isclose(
                component_unit,
                component_per_window / expected_calls,
                rel_tol=1.0e-12,
            ):
                raise ValueError(f"{name}.{component_name} timing formula is invalid")
            component_total += component_seconds
            component_calls += expected_calls
        if not math.isclose(component_total, total, rel_tol=1.0e-12):
            raise ValueError(f"{name} component timings do not sum to total_seconds")
        if component_calls != payload["actual_forward_calls"]:
            raise ValueError(f"{name} component forward calls do not sum to total")
    return MappingProxyType(manifest)


__all__ = [
    "P05PilotEvaluatorBenchmarkResult",
    "create_p05_pilot_evaluator_benchmark",
    "verify_p05_pilot_evaluator_benchmark",
]
