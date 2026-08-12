"""Actual, create-only P05-D03 same-checkpoint AWGN interventions."""

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
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch

from .p05_trace_runner import model_state_sha256


SCHEMA_NAME = "p05.d03_actual_noise_intervention"
SCHEMA_VERSION = 2
ARRAYS_NAME = "d03_arrays.npz"
MANIFEST_NAME = "manifest.json"

SNR_LEVELS_DB = (30, 20)
DRAWS_PER_LEVEL = 16
TOTAL_NOISE_DRAWS = len(SNR_LEVELS_DB) * DRAWS_PER_LEVEL
DEFAULT_CHUNK_SIZE = 256
MAX_CHUNK_SIZE = 256
RULE_COUNT = 10
FUZZY_FEATURE_COUNT = 8
FUZZY_SCALE = 0.5
RECONSTRUCTION_ATOL = 1.0e-6
RECONSTRUCTION_RTOL = 1.0e-6
REGISTERED_FINAL_SEEDS = frozenset({42, 123, 456, 789, 1024})
REGISTERED_PARTITION_WINDOW_COUNTS = {
    ("CWRU", "validation"): 19 * 16,
    ("CWRU", "test"): 23 * 16,
    ("XJTU", "validation"): 1317 * 4,
    ("XJTU", "test"): 6647 * 4,
}

_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_IDENTIFIER_ARRAYS = {"sample_id", "record_id", "group_id"}
_HASH_ARRAYS = {"input_window_sha256", "noise_sha256"}
_ARRAY_NAMES = {
    "sample_id",
    "record_id",
    "group_id",
    "window_start",
    "window_end",
    "y",
    "snr_db",
    "draw_index",
    "rule_index",
    "noise_seed",
    "noise_sha256",
    "input_window_sha256",
    "signal_power",
    "target_noise_std",
    "realized_noise_power",
    "realized_snr_db",
    "original_logits",
    "original_prediction",
    "original_normalized_rule_firing",
    "original_reference_class_attribution",
    "noisy_logits",
    "noisy_normalized_rule_firing",
    "noisy_reference_class_attribution",
    "top_rule_agreement",
    "top3_jaccard",
    "firing_vector_jsd",
    "attribution_rank_tau",
    "prediction_agreement",
}
_SHARED_ARRAYS = {"snr_db", "draw_index", "rule_index"}
_BATCH_ARRAYS = _ARRAY_NAMES - _SHARED_ARRAYS


@dataclass(frozen=True)
class P05D03Provenance:
    """Immutable run/data/device binding for one D03 artifact."""

    dataset: str
    split: str
    model_seed: int
    config_sha256: str
    code_sha256: str
    checkpoint_sha256: str
    model_sha256: str
    run_contract_sha256: str
    source_metadata_sha256: str
    derived_metadata_sha256: str
    cache_manifest_sha256: str
    split_manifest_sha256: str
    normalization_sha256: str
    physical_gpu_index: int | None = None
    device_uuid: str | None = None


@dataclass(frozen=True)
class P05D03Result:
    artifact_dir: Path
    arrays_path: Path
    manifest_path: Path
    semantic_sha256: str
    arrays_sha256: str
    manifest_sha256: str
    status: str
    timing: Mapping[str, Any]


@dataclass(frozen=True)
class _PreparedBatch:
    x: torch.Tensor
    y: np.ndarray
    sample_id: np.ndarray
    record_id: np.ndarray
    group_id: np.ndarray
    window_start: np.ndarray
    window_end: np.ndarray
    input_count: int


@dataclass(frozen=True)
class _ExecutionResult:
    arrays: Mapping[str, np.ndarray]
    timing: Mapping[str, float]
    chunk_count: int


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


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON numeric constant is forbidden: {value}")


def _strict_json_load(path: Path) -> Any:
    try:
        payload = path.read_text(encoding="utf-8")
        return json.loads(
            payload,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid strict JSON document: {path}") from exc


def _required_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _array_sha256(array: np.ndarray) -> str:
    descriptor = _canonical_json_bytes(
        {"dtype": array.dtype.str, "shape": [int(size) for size in array.shape]}
    )
    return _sha256_bytes(descriptor + b"\0" + array.tobytes(order="C"))


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


def _hash_strings(array: np.ndarray) -> list[str]:
    try:
        return [value.decode("ascii") for value in array.reshape(-1).tolist()]
    except (AttributeError, UnicodeDecodeError) as exc:
        raise ValueError("D03 hash array must contain fixed-width ASCII bytes") from exc


def p05_d03_noise_seed(
    *,
    dataset: str,
    split: str,
    model_seed: int,
    sample_id: str,
    snr_db: int,
) -> int:
    """Return the frozen first-eight-byte SHA-256 PCG64 seed."""

    if dataset not in {"CWRU", "XJTU"}:
        raise ValueError("dataset must be CWRU or XJTU")
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if type(model_seed) is not int or not 0 <= model_seed < 2**63:
        raise ValueError("model_seed must be a non-negative signed 64-bit integer")
    if not isinstance(sample_id, str) or not sample_id.strip() or "\x00" in sample_id:
        raise ValueError("sample_id must be a non-empty string without NUL")
    if type(snr_db) is not int or snr_db not in SNR_LEVELS_DB:
        raise ValueError(f"snr_db must be one of {SNR_LEVELS_DB}")
    payload = (
        f"P05-stability|{dataset}|{split}|{model_seed}|{sample_id}|{snr_db}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _validate_provenance(
    value: P05D03Provenance,
    *,
    require_cuda: bool,
) -> dict[str, Any]:
    if not isinstance(value, P05D03Provenance):
        raise TypeError("provenance must be P05D03Provenance")
    if value.dataset not in {"CWRU", "XJTU"}:
        raise ValueError("provenance dataset must be CWRU or XJTU")
    if value.split not in {"validation", "test"}:
        raise ValueError("provenance split must be validation or test")
    if type(value.model_seed) is not int or not 0 <= value.model_seed < 2**63:
        raise ValueError("provenance model_seed must be a non-negative integer")
    hash_names = (
        "cache_manifest_sha256",
        "checkpoint_sha256",
        "code_sha256",
        "config_sha256",
        "derived_metadata_sha256",
        "model_sha256",
        "normalization_sha256",
        "run_contract_sha256",
        "source_metadata_sha256",
        "split_manifest_sha256",
    )
    payload: dict[str, Any] = {
        name: _required_sha256(getattr(value, name), name=f"provenance {name}")
        for name in hash_names
    }
    payload.update(
        {
            "dataset": value.dataset,
            "device_uuid": value.device_uuid,
            "model_seed": value.model_seed,
            "physical_gpu_index": value.physical_gpu_index,
            "split": value.split,
        }
    )
    if require_cuda:
        if type(value.physical_gpu_index) is not int or value.physical_gpu_index not in {
            0,
            1,
        }:
            raise ValueError("CUDA D03 requires physical_gpu_index 0 or 1")
        if (
            not isinstance(value.device_uuid, str)
            or not value.device_uuid.startswith("GPU-")
            or any(character.isspace() for character in value.device_uuid)
        ):
            raise ValueError("CUDA D03 requires a recorded NVIDIA GPU UUID")
    elif value.physical_gpu_index is not None or value.device_uuid is not None:
        raise ValueError("CPU-only tests must not record a physical GPU identity")
    return payload


def _validate_phase(
    provenance: P05D03Provenance,
    *,
    phase: str,
    budget_retained: bool | None,
    benchmark_first_n: int | None,
) -> None:
    if phase == "pilot_benchmark":
        if provenance.split != "validation" or provenance.model_seed != 20260801:
            raise ValueError(
                "pilot_benchmark requires validation and model seed 20260801"
            )
        if benchmark_first_n != 256:
            raise ValueError("pilot_benchmark requires benchmark_first_n=256")
        if budget_retained is not None:
            raise ValueError("pilot_benchmark has no post-pilot budget decision")
        return
    if phase != "budget_retained_secondary":
        raise ValueError(
            "phase must be pilot_benchmark or budget_retained_secondary"
        )
    if provenance.split != "test" or provenance.model_seed not in REGISTERED_FINAL_SEEDS:
        raise ValueError(
            "budget_retained_secondary requires test and a registered final seed"
        )
    if budget_retained is not True:
        raise RuntimeError("P05-D03 is forbidden unless the locked budget gate retains it")
    if benchmark_first_n is not None:
        raise ValueError("decisive D03 may not truncate the registered test partition")


def _identifier_vector(value: Any, *, name: str, count: int) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"batch {name} must be a string sequence")
    raw = np.asarray(value)
    if raw.shape != (count,):
        raise ValueError(f"batch {name} must have shape ({count},)")
    values = raw.tolist()
    if any(
        not isinstance(item, str) or not item.strip() or "\x00" in item
        for item in values
    ):
        raise ValueError(f"batch {name} values must be non-empty strings without NUL")
    return np.ascontiguousarray(
        np.asarray(values, dtype=f"<U{max(len(item) for item in values)}")
    )


def _integer_vector(value: Any, *, name: str, count: int) -> np.ndarray:
    if torch.is_tensor(value):
        if value.dtype == torch.bool or value.dtype.is_floating_point:
            raise TypeError(f"batch {name} must contain integers")
        raw = value.detach().to(device="cpu").numpy()
    else:
        raw = np.asarray(value)
        if raw.dtype.kind not in {"i", "u"}:
            raise TypeError(f"batch {name} must contain integers")
    if raw.shape != (count,):
        raise ValueError(f"batch {name} must have shape ({count},)")
    return np.ascontiguousarray(raw, dtype="<i8")


def _prepare_batch(
    batch: Mapping[str, Any],
    *,
    expected_window_size: int,
    benchmark_first_n: int | None,
) -> _PreparedBatch:
    if not isinstance(batch, Mapping):
        raise TypeError("batch must be a mapping")
    required = {
        "x",
        "y",
        "sample_id",
        "record_id",
        "group_id",
        "window_start",
        "window_end",
    }
    if set(batch) != required:
        raise ValueError(
            "batch fields differ from the D03 contract: "
            f"missing={sorted(required - set(batch))}, "
            f"unexpected={sorted(set(batch) - required)}"
        )
    x = batch["x"]
    if not torch.is_tensor(x) or x.dtype != torch.float32:
        raise TypeError("batch x must be a float32 tensor")
    if x.ndim != 3 or tuple(x.shape[1:]) != (expected_window_size, 2):
        raise ValueError(
            f"batch x must have shape (batch,{expected_window_size},2), "
            f"got {tuple(x.shape)}"
        )
    if not torch.isfinite(x).all():
        raise FloatingPointError("batch x contains non-finite values")
    count = int(x.shape[0])
    if count <= 0:
        raise ValueError("batch must contain at least one sample")
    if benchmark_first_n is not None and (
        type(benchmark_first_n) is not int or benchmark_first_n <= 0
    ):
        raise ValueError("benchmark_first_n must be a positive integer or None")

    sample_id = _identifier_vector(batch["sample_id"], name="sample_id", count=count)
    record_id = _identifier_vector(batch["record_id"], name="record_id", count=count)
    group_id = _identifier_vector(batch["group_id"], name="group_id", count=count)
    y = _integer_vector(batch["y"], name="y", count=count)
    window_start = _integer_vector(
        batch["window_start"], name="window_start", count=count
    )
    window_end = _integer_vector(batch["window_end"], name="window_end", count=count)
    if len(set(sample_id.tolist())) != count:
        raise ValueError("batch sample_id values must be unique")
    if np.any(window_start < 0) or np.any(window_end - window_start != expected_window_size):
        raise ValueError("batch windows must have the exact registered window size")
    for index in range(count):
        expected_id = f"{record_id[index]}:{window_start[index]}:{window_end[index]}"
        if sample_id[index] != expected_id:
            raise ValueError(
                f"batch sample_id[{index}] must equal {expected_id!r}, "
                f"got {sample_id[index]!r}"
            )

    order = np.argsort(sample_id, kind="stable")
    if benchmark_first_n is not None:
        order = order[:benchmark_first_n]
    if benchmark_first_n == 256 and len(order) != 256:
        raise ValueError("pilot benchmark input must contain at least 256 unique windows")
    torch_order = torch.as_tensor(order, dtype=torch.long, device=x.device)
    selected_x = x.index_select(0, torch_order).detach().to(device="cpu").contiguous()
    return _PreparedBatch(
        x=selected_x,
        y=np.ascontiguousarray(y[order]),
        sample_id=np.ascontiguousarray(sample_id[order]),
        record_id=np.ascontiguousarray(record_id[order]),
        group_id=np.ascontiguousarray(group_id[order]),
        window_start=np.ascontiguousarray(window_start[order]),
        window_end=np.ascontiguousarray(window_end[order]),
        input_count=count,
    )


def _normalize_expected_sample_ids(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("expected_sample_ids must be a string sequence")
    identifiers = list(value)
    if not identifiers:
        raise ValueError("expected_sample_ids cannot be empty")
    if any(
        not isinstance(item, str) or not item.strip() or "\x00" in item
        for item in identifiers
    ):
        raise ValueError(
            "expected_sample_ids must contain non-empty strings without NUL"
        )
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("expected_sample_ids must be unique")
    return tuple(sorted(identifiers))


def _validate_chunk_size(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= MAX_CHUNK_SIZE:
        raise ValueError(
            f"chunk_size must be an integer within [1, {MAX_CHUNK_SIZE}]"
        )
    return value


def _validate_phase_chunk_size(*, phase: str, chunk_size: int) -> None:
    if phase == "pilot_benchmark" and chunk_size != 256:
        raise ValueError(
            "pilot_benchmark requires one exact batched evaluator chunk_size=256"
        )


def _validate_registered_partition_count(
    *,
    provenance: P05D03Provenance,
    expected_sample_ids: tuple[str, ...],
    require_cuda: bool,
) -> None:
    if not require_cuda:
        return
    expected_count = REGISTERED_PARTITION_WINDOW_COUNTS[
        (provenance.dataset, provenance.split)
    ]
    if len(expected_sample_ids) != expected_count:
        raise ValueError(
            "CUDA D03 expected_sample_ids count differs from the frozen partition: "
            f"expected={expected_count}, got={len(expected_sample_ids)}"
        )


def _slice_prepared(
    prepared: _PreparedBatch,
    indices: np.ndarray,
) -> _PreparedBatch:
    if indices.ndim != 1 or indices.dtype.kind not in {"i", "u"}:
        raise TypeError("prepared indices must be a one-dimensional integer array")
    torch_indices = torch.as_tensor(indices, dtype=torch.long)
    return _PreparedBatch(
        x=prepared.x.index_select(0, torch_indices),
        y=np.ascontiguousarray(prepared.y[indices]),
        sample_id=np.ascontiguousarray(prepared.sample_id[indices]),
        record_id=np.ascontiguousarray(prepared.record_id[indices]),
        group_id=np.ascontiguousarray(prepared.group_id[indices]),
        window_start=np.ascontiguousarray(prepared.window_start[indices]),
        window_end=np.ascontiguousarray(prepared.window_end[indices]),
        input_count=len(indices),
    )


def _prepared_chunks(
    prepared: _PreparedBatch,
    *,
    chunk_size: int,
) -> Iterable[_PreparedBatch]:
    for start in range(0, len(prepared.sample_id), chunk_size):
        stop = min(start + chunk_size, len(prepared.sample_id))
        yield _slice_prepared(
            prepared,
            np.arange(start, stop, dtype=np.int64),
        )


def _prepared_from_rows(rows: list[tuple[Any, ...]]) -> _PreparedBatch:
    if not rows:
        raise ValueError("cannot construct an empty D03 chunk")
    count = len(rows)
    sample_ids = [str(row[2]) for row in rows]
    record_ids = [str(row[3]) for row in rows]
    group_ids = [str(row[4]) for row in rows]
    return _PreparedBatch(
        x=torch.stack([row[0] for row in rows], dim=0).contiguous(),
        y=np.asarray([row[1] for row in rows], dtype="<i8"),
        sample_id=_identifier_vector(
            sample_ids,
            name="pending sample_id",
            count=count,
        ),
        record_id=_identifier_vector(
            record_ids,
            name="pending record_id",
            count=count,
        ),
        group_id=_identifier_vector(
            group_ids,
            name="pending group_id",
            count=count,
        ),
        window_start=np.asarray([row[5] for row in rows], dtype="<i8"),
        window_end=np.asarray([row[6] for row in rows], dtype="<i8"),
        input_count=count,
    )


def _network_device(network: torch.nn.Module) -> torch.device:
    devices = {tensor.device for tensor in network.parameters()}
    devices.update(tensor.device for tensor in network.buffers())
    if not devices:
        raise ValueError("P05-D03 network exposes no parameter or buffer device")
    if len(devices) != 1:
        raise ValueError(f"P05-D03 network spans multiple devices: {devices}")
    return next(iter(devices))


def _require_float32_model(network: torch.nn.Module) -> None:
    for name, tensor in (*network.named_parameters(), *network.named_buffers()):
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise TypeError(f"P05-D03 model tensor {name!r} must be float32")


def _validate_output(
    output: Any,
    *,
    sample_count: int,
    class_count: int,
    identity: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = getattr(output, "logits", None)
    non_fuzzy = getattr(output, "non_fuzzy_logits", None)
    trace = getattr(output, "fuzzy_trace", None)
    if trace is None:
        raise ValueError(f"{identity} output is missing fuzzy_trace")
    for name, value, shape in (
        ("logits", logits, (sample_count, class_count)),
        ("non_fuzzy_logits", non_fuzzy, (sample_count, class_count)),
        (
            "fuzzy_trace.reduced_features",
            getattr(trace, "reduced_features", None),
            (sample_count, FUZZY_FEATURE_COUNT),
        ),
        (
            "fuzzy_trace.normalized_rule_firing",
            getattr(trace, "normalized_rule_firing", None),
            (sample_count, RULE_COUNT),
        ),
        (
            "fuzzy_trace.rule_contributions",
            getattr(trace, "rule_contributions", None),
            (sample_count, RULE_COUNT, class_count),
        ),
        (
            "fuzzy_trace.fuzzy_logits",
            getattr(trace, "fuzzy_logits", None),
            (sample_count, class_count),
        ),
    ):
        if not torch.is_tensor(value) or value.dtype != torch.float32:
            raise TypeError(f"{identity} {name} must be a float32 tensor")
        if tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"{identity} {name} has an invalid shape or value")
    scale = getattr(output, "fuzzy_scale", None)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError(f"{identity} fuzzy_scale must be numeric")
    if not math.isclose(float(scale), FUZZY_SCALE, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(f"{identity} fuzzy_scale must equal the frozen value 0.5")

    firing = trace.normalized_rule_firing
    contributions = trace.rule_contributions
    if torch.any(firing < 0.0) or not torch.allclose(
        firing.sum(dim=1),
        torch.ones(sample_count, dtype=torch.float32, device=firing.device),
        atol=RECONSTRUCTION_ATOL,
        rtol=RECONSTRUCTION_RTOL,
    ):
        raise ValueError(f"{identity} normalized firing is invalid")
    if not torch.allclose(
        trace.fuzzy_logits,
        contributions.sum(dim=1),
        atol=RECONSTRUCTION_ATOL,
        rtol=RECONSTRUCTION_RTOL,
    ):
        raise ValueError(f"{identity} fuzzy logits do not reconstruct")
    if not torch.allclose(
        logits,
        non_fuzzy + FUZZY_SCALE * trace.fuzzy_logits,
        atol=RECONSTRUCTION_ATOL,
        rtol=RECONSTRUCTION_RTOL,
    ):
        raise ValueError(f"{identity} total logits do not reconstruct")
    mask = getattr(trace, "rule_mask", None)
    if (
        not torch.is_tensor(mask)
        or mask.dtype != torch.bool
        or tuple(mask.shape) != (sample_count, RULE_COUNT)
        or not bool(mask.all())
    ):
        raise ValueError(f"{identity} must retain every rule")
    permutation = getattr(trace, "consequent_permutation", None)
    expected_permutation = torch.arange(RULE_COUNT, device=logits.device)
    if not torch.is_tensor(permutation) or not torch.equal(
        permutation.to(dtype=torch.long), expected_permutation
    ):
        raise ValueError(f"{identity} must retain the identity consequent order")
    return logits, firing, contributions


def _to_numpy_float32(value: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(
        value.detach().to(device="cpu").numpy(), dtype=np.dtype("<f4")
    )


def _attribution(
    contributions: np.ndarray,
    reference_class: np.ndarray,
) -> np.ndarray:
    sample = np.arange(contributions.shape[0])[:, None]
    rules = np.arange(contributions.shape[1])[None, :]
    classes = reference_class[:, None]
    return np.ascontiguousarray(
        np.abs(FUZZY_SCALE * contributions[sample, rules, classes]),
        dtype="<f4",
    )


def _rule_order(values: np.ndarray) -> np.ndarray:
    rule_index = np.arange(values.shape[0], dtype=np.int64)
    return np.lexsort((rule_index, -values.astype(np.float64, copy=False)))


def _rank_tau(clean: np.ndarray, noisy: np.ndarray) -> float:
    clean_order = _rule_order(clean)
    noisy_order = _rule_order(noisy)
    clean_position = np.empty(RULE_COUNT, dtype=np.int64)
    noisy_position = np.empty(RULE_COUNT, dtype=np.int64)
    clean_position[clean_order] = np.arange(RULE_COUNT)
    noisy_position[noisy_order] = np.arange(RULE_COUNT)
    concordant = 0
    discordant = 0
    for left in range(RULE_COUNT - 1):
        for right in range(left + 1, RULE_COUNT):
            clean_sign = int(clean_position[left] < clean_position[right])
            noisy_sign = int(noisy_position[left] < noisy_position[right])
            if clean_sign == noisy_sign:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / float(concordant + discordant)


def _natural_log_jsd(left: np.ndarray, right: np.ndarray) -> float:
    p = left.astype(np.float64, copy=False)
    q = right.astype(np.float64, copy=False)
    if np.any(p < 0.0) or np.any(q < 0.0):
        raise ValueError("firing distributions cannot contain negative mass")
    p_sum = float(p.sum())
    q_sum = float(q.sum())
    if p_sum <= 0.0 or q_sum <= 0.0:
        raise ValueError("firing distributions must have positive mass")
    p = p / p_sum
    q = q / q_sum
    midpoint = 0.5 * (p + q)

    def _kl(value: np.ndarray) -> float:
        positive = value > 0.0
        return float(np.sum(value[positive] * np.log(value[positive] / midpoint[positive])))

    return 0.5 * (_kl(p) + _kl(q))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_arrays(
    *,
    network: torch.nn.Module,
    prepared: _PreparedBatch,
    provenance: P05D03Provenance,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    sample_count = len(prepared.sample_id)
    class_count = 2 if provenance.dataset == "XJTU" else 4
    x_numpy = np.ascontiguousarray(prepared.x.numpy(), dtype="<f4")
    signal_power = np.mean(
        np.square(x_numpy.astype(np.float64)), axis=(1, 2), dtype=np.float64
    )
    if not np.isfinite(signal_power).all() or np.any(signal_power <= 0.0):
        raise ValueError("every standardized window must have positive finite signal power")
    target_noise_std = np.sqrt(
        signal_power[:, None]
        / np.power(10.0, np.asarray(SNR_LEVELS_DB, dtype=np.float64)[None, :] / 10.0)
    )
    if not np.isfinite(target_noise_std).all() or np.any(target_noise_std <= 0.0):
        raise ValueError("registered AWGN standard deviation is invalid")

    noise_seed = np.empty((sample_count, len(SNR_LEVELS_DB)), dtype="<u8")
    input_window_sha256 = np.asarray(
        [_array_sha256(x_numpy[index]) for index in range(sample_count)],
        dtype="S64",
    )
    generators: list[list[np.random.Generator]] = []
    for sample_index, sample_id in enumerate(prepared.sample_id.tolist()):
        row: list[np.random.Generator] = []
        for snr_index, snr_db in enumerate(SNR_LEVELS_DB):
            seed = p05_d03_noise_seed(
                dataset=provenance.dataset,
                split=provenance.split,
                model_seed=provenance.model_seed,
                sample_id=sample_id,
                snr_db=snr_db,
            )
            noise_seed[sample_index, snr_index] = seed
            row.append(np.random.Generator(np.random.PCG64(seed)))
        generators.append(row)
    if len(set(noise_seed.reshape(-1).tolist())) != noise_seed.size:
        raise RuntimeError("registered D03 seed collision detected")

    x_device = prepared.x.to(device=device, dtype=torch.float32, non_blocking=False)
    _synchronize(device)
    started_ns = time.perf_counter_ns()
    with torch.no_grad():
        original = network.forward_with_fuzzy_trace(x_device)
        original_logits_t, original_firing_t, original_contributions_t = _validate_output(
            original,
            sample_count=sample_count,
            class_count=class_count,
            identity="D03 original",
        )
    _synchronize(device)
    original_seconds = (time.perf_counter_ns() - started_ns) / 1.0e9

    original_logits = _to_numpy_float32(original_logits_t)
    original_firing = _to_numpy_float32(original_firing_t)
    original_contributions = _to_numpy_float32(original_contributions_t)
    reference_class = np.argmax(original_logits, axis=1).astype("<i8", copy=False)
    if np.any(prepared.y < 0) or np.any(prepared.y >= class_count):
        raise ValueError("batch y contains an out-of-range protocol class")
    original_attribution = _attribution(original_contributions, reference_class)
    clean_orders = np.stack([_rule_order(row) for row in original_attribution])

    levels = len(SNR_LEVELS_DB)
    shape_prefix = (sample_count, levels, DRAWS_PER_LEVEL)
    noisy_logits = np.empty((*shape_prefix, class_count), dtype="<f4")
    noisy_firing = np.empty((*shape_prefix, RULE_COUNT), dtype="<f4")
    noisy_attribution = np.empty((*shape_prefix, RULE_COUNT), dtype="<f4")
    noise_sha256 = np.empty(shape_prefix, dtype="S64")
    realized_noise_power = np.empty(shape_prefix, dtype="<f8")
    realized_snr_db = np.empty(shape_prefix, dtype="<f8")
    top_rule_agreement = np.empty(shape_prefix, dtype=np.bool_)
    top3_jaccard = np.empty(shape_prefix, dtype="<f8")
    firing_jsd = np.empty(shape_prefix, dtype="<f8")
    attribution_tau = np.empty(shape_prefix, dtype="<f8")
    prediction_agreement = np.empty(shape_prefix, dtype=np.bool_)

    _synchronize(device)
    noise_started_ns = time.perf_counter_ns()
    for snr_index, _snr_db in enumerate(SNR_LEVELS_DB):
        for draw_index in range(DRAWS_PER_LEVEL):
            noisy_x = np.empty_like(x_numpy)
            for sample_index in range(sample_count):
                standard_normal = generators[sample_index][snr_index].standard_normal(
                    x_numpy.shape[1:], dtype=np.float32
                )
                noise = np.multiply(
                    standard_normal,
                    np.float32(target_noise_std[sample_index, snr_index]),
                    dtype=np.float32,
                )
                noisy_x[sample_index] = np.add(
                    x_numpy[sample_index], noise, dtype=np.float32
                )
                noise_sha256[sample_index, snr_index, draw_index] = _array_sha256(
                    np.ascontiguousarray(noise, dtype="<f4")
                )
                measured_power = float(
                    np.mean(np.square(noise.astype(np.float64)), dtype=np.float64)
                )
                if not math.isfinite(measured_power) or measured_power <= 0.0:
                    raise FloatingPointError("realized D03 noise power is invalid")
                realized_noise_power[sample_index, snr_index, draw_index] = measured_power
                realized_snr_db[sample_index, snr_index, draw_index] = 10.0 * math.log10(
                    float(signal_power[sample_index]) / measured_power
                )

            noisy_tensor = torch.from_numpy(noisy_x).to(
                device=device, dtype=torch.float32, non_blocking=False
            )
            with torch.no_grad():
                noisy_output = network.forward_with_fuzzy_trace(noisy_tensor)
                logits_t, firing_t, contributions_t = _validate_output(
                    noisy_output,
                    sample_count=sample_count,
                    class_count=class_count,
                    identity=f"D03 snr={SNR_LEVELS_DB[snr_index]} draw={draw_index}",
                )
            logits = _to_numpy_float32(logits_t)
            firing = _to_numpy_float32(firing_t)
            contributions = _to_numpy_float32(contributions_t)
            attribution = _attribution(contributions, reference_class)
            noisy_logits[:, snr_index, draw_index] = logits
            noisy_firing[:, snr_index, draw_index] = firing
            noisy_attribution[:, snr_index, draw_index] = attribution
            predictions = np.argmax(logits, axis=1)
            prediction_agreement[:, snr_index, draw_index] = (
                predictions == reference_class
            )
            for sample_index in range(sample_count):
                noisy_order = _rule_order(attribution[sample_index])
                clean_order = clean_orders[sample_index]
                top_rule_agreement[sample_index, snr_index, draw_index] = (
                    noisy_order[0] == clean_order[0]
                )
                intersection = len(set(noisy_order[:3]) & set(clean_order[:3]))
                top3_jaccard[sample_index, snr_index, draw_index] = intersection / (
                    6 - intersection
                )
                firing_jsd[sample_index, snr_index, draw_index] = _natural_log_jsd(
                    original_firing[sample_index], firing[sample_index]
                )
                attribution_tau[sample_index, snr_index, draw_index] = _rank_tau(
                    original_attribution[sample_index], attribution[sample_index]
                )
    _synchronize(device)
    noise_seconds = (time.perf_counter_ns() - noise_started_ns) / 1.0e9

    for sample_index in range(sample_count):
        for snr_index in range(levels):
            hashes = noise_sha256[sample_index, snr_index].tolist()
            if len(set(hashes)) != DRAWS_PER_LEVEL:
                raise RuntimeError("D03 noise draws are not unique within a seed stream")

    arrays: dict[str, np.ndarray] = {
        "sample_id": prepared.sample_id,
        "record_id": prepared.record_id,
        "group_id": prepared.group_id,
        "window_start": prepared.window_start,
        "window_end": prepared.window_end,
        "y": prepared.y,
        "snr_db": np.asarray(SNR_LEVELS_DB, dtype="<i8"),
        "draw_index": np.arange(DRAWS_PER_LEVEL, dtype="<i8"),
        "rule_index": np.arange(RULE_COUNT, dtype="<i8"),
        "noise_seed": noise_seed,
        "noise_sha256": noise_sha256,
        "input_window_sha256": input_window_sha256,
        "signal_power": np.ascontiguousarray(signal_power, dtype="<f8"),
        "target_noise_std": np.ascontiguousarray(target_noise_std, dtype="<f8"),
        "realized_noise_power": realized_noise_power,
        "realized_snr_db": realized_snr_db,
        "original_logits": original_logits,
        "original_prediction": reference_class,
        "original_normalized_rule_firing": original_firing,
        "original_reference_class_attribution": original_attribution,
        "noisy_logits": noisy_logits,
        "noisy_normalized_rule_firing": noisy_firing,
        "noisy_reference_class_attribution": noisy_attribution,
        "top_rule_agreement": top_rule_agreement,
        "top3_jaccard": top3_jaccard,
        "firing_vector_jsd": firing_jsd,
        "attribution_rank_tau": attribution_tau,
        "prediction_agreement": prediction_agreement,
    }
    normalized = {
        name: np.ascontiguousarray(value)
        for name, value in sorted(arrays.items())
    }
    return normalized, {
        "noise_forward_seconds": noise_seconds,
        "original_forward_seconds": original_seconds,
        "total_seconds": original_seconds + noise_seconds,
    }


def _execute_chunks(
    *,
    network: torch.nn.Module,
    chunks: Iterable[_PreparedBatch],
    provenance: P05D03Provenance,
    device: torch.device,
) -> _ExecutionResult:
    batch_arrays: dict[str, list[np.ndarray]] = {
        name: [] for name in sorted(_BATCH_ARRAYS)
    }
    shared_arrays: dict[str, np.ndarray] | None = None
    timing = {
        "noise_forward_seconds": 0.0,
        "original_forward_seconds": 0.0,
        "total_seconds": 0.0,
    }
    chunk_count = 0
    for prepared in chunks:
        if len(prepared.sample_id) == 0:
            raise ValueError("D03 execution received an empty chunk")
        arrays, chunk_timing = _build_arrays(
            network=network,
            prepared=prepared,
            provenance=provenance,
            device=device,
        )
        chunk_count += 1
        for name in timing:
            timing[name] += float(chunk_timing[name])
        if shared_arrays is None:
            shared_arrays = {
                name: np.array(arrays[name], copy=True, order="C")
                for name in sorted(_SHARED_ARRAYS)
            }
        else:
            for name in _SHARED_ARRAYS:
                if not np.array_equal(shared_arrays[name], arrays[name]):
                    raise RuntimeError(f"D03 shared array {name!r} changed by chunk")
        for name in _BATCH_ARRAYS:
            batch_arrays[name].append(arrays[name])
    if chunk_count == 0 or shared_arrays is None:
        raise ValueError("D03 execution selected no windows")

    combined = {
        name: np.ascontiguousarray(np.concatenate(parts, axis=0))
        for name, parts in batch_arrays.items()
    }
    combined.update(shared_arrays)
    order = np.argsort(combined["sample_id"], kind="stable")
    if len(set(combined["sample_id"].tolist())) != len(order):
        raise ValueError("D03 selected sample IDs are not unique across chunks")
    for name in _BATCH_ARRAYS:
        combined[name] = np.ascontiguousarray(combined[name][order])
    flattened_seeds = combined["noise_seed"].reshape(-1).tolist()
    if len(set(flattened_seeds)) != len(flattened_seeds):
        raise RuntimeError("registered D03 seed collision detected across chunks")
    return _ExecutionResult(
        arrays=MappingProxyType(dict(sorted(combined.items()))),
        timing=MappingProxyType(timing),
        chunk_count=chunk_count,
    )


def _sample_id_semantic_sha256(sample_ids: Sequence[str]) -> str:
    return _sha256_bytes(_canonical_json_bytes(list(sample_ids)))


def _input_binding(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_window_size: int,
    input_count: int,
    phase: str,
) -> dict[str, Any]:
    window_hashes = _hash_strings(arrays["input_window_sha256"])
    sample_ids = arrays["sample_id"].tolist()
    return {
        "input_count": input_count,
        "selected_count": len(sample_ids),
        "selected_float32_windows": {
            "dtype": "<f4",
            "per_window_sha256_array": "input_window_sha256",
            "sample_id_and_window_sha256_semantic_sha256": _sha256_bytes(
                _canonical_json_bytes(
                    [
                        {"sample_id": sample_id, "sha256": window_hash}
                        for sample_id, window_hash in zip(
                            sample_ids, window_hashes, strict=True
                        )
                    ]
                )
            ),
            "shape": [len(sample_ids), expected_window_size, 2],
        },
        "selection": (
            "first_256_after_stable_sample_id_sort"
            if phase == "pilot_benchmark"
            else "all_after_stable_sample_id_sort"
        ),
    }


def _partition_coverage(
    *,
    expected_sample_ids: tuple[str, ...],
    observed_sample_ids: Sequence[str],
    selected_sample_ids: Sequence[str],
    phase: str,
) -> dict[str, Any]:
    observed = tuple(sorted(observed_sample_ids))
    if len(observed) != len(set(observed)):
        raise ValueError("D03 loader emitted duplicate sample IDs")
    expected_set = set(expected_sample_ids)
    observed_set = set(observed)
    if observed_set != expected_set:
        missing = sorted(expected_set - observed_set)
        unexpected = sorted(observed_set - expected_set)
        raise ValueError(
            "D03 loader coverage differs from expected_sample_ids: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    selected = tuple(selected_sample_ids)
    expected_selected = (
        expected_sample_ids[:256]
        if phase == "pilot_benchmark"
        else expected_sample_ids
    )
    if selected != expected_selected:
        raise ValueError("D03 selected sample IDs differ from the frozen phase selection")
    return {
        "coverage": "exact",
        "expected_sample_count": len(expected_sample_ids),
        "expected_sample_id_semantic_sha256": _sample_id_semantic_sha256(
            expected_sample_ids
        ),
        "observed_sample_count": len(observed),
        "observed_sample_id_semantic_sha256": _sample_id_semantic_sha256(observed),
        "selected_sample_count": len(selected),
        "selected_sample_id_semantic_sha256": _sample_id_semantic_sha256(selected),
    }


def _protocol_payload() -> dict[str, Any]:
    return {
        "id": "P05-D03",
        "attribution": (
            "abs(0.5 * rule_contribution_to_unperturbed_argmax_class)"
        ),
        "attribution_rank_tau": (
            "Kendall_tau_b_on_total_orders_after_lower_rule_index_tie_break"
        ),
        "awgn": (
            "noise_std=sqrt(mean_float64(standardized_window_float32^2)/"
            "10^(snr_db/10)); PCG64 standard_normal float32; addition float32"
        ),
        "draws_per_level": DRAWS_PER_LEVEL,
        "firing_vector_jsd": "natural_log_Jensen_Shannon_divergence",
        "independent_draws": (
            "successive_nonoverlapping_standard_normal_draws_from_one_"
            "PCG64_stream_per_sample_and_snr"
        ),
        "metrics": [
            "top_rule_agreement",
            "top3_Jaccard",
            "firing_vector_JSD",
            "attribution_rank_tau",
            "prediction_agreement",
        ],
        "prediction_tie_break": "lower_class_index_via_argmax",
        "seed": (
            "unsigned_big_endian_first_8_SHA256_bytes_of_"
            "P05-stability|dataset|split|model_seed|sample_id|snr"
        ),
        "snr_db": list(SNR_LEVELS_DB),
        "top_rule_tie_break": "lower_rule_index",
        "top3_jaccard": "set_intersection_size_over_set_union_size",
        "total_noise_draws_per_sample": TOTAL_NOISE_DRAWS,
    }


def _semantic_manifest(
    *,
    arrays: Mapping[str, np.ndarray],
    provenance: Mapping[str, Any],
    input_binding: Mapping[str, Any],
    partition_coverage: Mapping[str, Any],
    state_sha256: str,
    phase: str,
    require_cuda: bool,
    budget_retained: bool | None,
    chunk_size: int,
    chunk_count: int,
) -> dict[str, Any]:
    return {
        "arrays": _array_descriptors(arrays),
        "conclusion_control": {
            "claim_decisions": "not_performed",
            "confirmatory_sign_tests": "not_performed",
            "performance_claim": False,
            "scientific_status": "computed_unadjudicated",
            "scope": "budget_conditional_secondary_P05_D03_only",
        },
        "execution": {
            "actual_forward_calls": chunk_count * (1 + TOTAL_NOISE_DRAWS),
            "budget_retained": budget_retained,
            "chunk_count": chunk_count,
            "chunk_size": chunk_size,
            "device_class": "cuda" if require_cuda else "cpu_test_only",
            "phase": phase,
        },
        "format": {
            "container": "numpy.npz",
            "hash_arrays": sorted(_HASH_ARRAYS),
            "identifier_arrays": sorted(_IDENTIFIER_ARRAYS),
            "load_allow_pickle": False,
            "model_and_noise_dtype": "float32",
            "object_arrays": False,
            "offline_metric_dtype": "float64",
        },
        "input_binding": dict(input_binding),
        "model_state": {
            "after_sha256": state_sha256,
            "before_sha256": state_sha256,
            "unchanged": True,
        },
        "npz_file": ARRAYS_NAME,
        "partition_coverage": dict(partition_coverage),
        "protocol": _protocol_payload(),
        "provenance": dict(provenance),
        "sample_count": int(arrays["sample_id"].shape[0]),
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
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
        raise RuntimeError("atomic create-only D03 artifact requires Linux renameat2")
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


def _write_package(
    target: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    semantic_manifest: Mapping[str, Any],
    timing: Mapping[str, float],
) -> P05D03Result:
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"D03 artifact parent must be a real directory: {parent}")
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
                "npz_sha256": arrays_sha256,
                "semantic_sha256": semantic_sha256,
            },
        }
        manifest_path = temporary / MANIFEST_NAME
        with manifest_path.open("wb") as handle:
            handle.write(_pretty_json_bytes(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary)
        _rename_directory_noreplace(temporary, target)
        _fsync_directory(parent)
        frozen_timing = MappingProxyType(
            {
                **timing,
                "performance_claim_allowed": False,
                "scope": "diagnostic_wall_clock_boundary_only",
            }
        )
        return P05D03Result(
            artifact_dir=target,
            arrays_path=target / ARRAYS_NAME,
            manifest_path=target / MANIFEST_NAME,
            semantic_sha256=semantic_sha256,
            arrays_sha256=arrays_sha256,
            manifest_sha256=_sha256_file(target / MANIFEST_NAME),
            status="created",
            timing=frozen_timing,
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_p05_d03_artifact(artifact_dir: str | Path) -> Mapping[str, Any]:
    """Verify package hashes, schema, arrays, and unadjudicated status."""

    target = Path(os.path.abspath(os.fspath(artifact_dir)))
    if target.is_symlink() or not target.is_dir():
        raise FileNotFoundError(f"D03 artifact must be a real directory: {target}")
    entries = {entry.name: entry for entry in target.iterdir()}
    if set(entries) != {ARRAYS_NAME, MANIFEST_NAME}:
        raise ValueError("D03 artifact has incomplete or unexpected content")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries.values()):
        raise ValueError("D03 artifact entries must be real files")
    manifest = _strict_json_load(entries[MANIFEST_NAME])
    expected_manifest_keys = {
        "arrays",
        "conclusion_control",
        "content",
        "execution",
        "format",
        "input_binding",
        "model_state",
        "npz_file",
        "partition_coverage",
        "protocol",
        "provenance",
        "sample_count",
        "schema_name",
        "schema_version",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_keys:
        raise ValueError("D03 manifest schema is incomplete or unexpected")
    if manifest.get("schema_name") != SCHEMA_NAME or manifest.get(
        "schema_version"
    ) != SCHEMA_VERSION:
        raise ValueError("D03 schema identity is unsupported")
    if manifest.get("npz_file") != ARRAYS_NAME:
        raise ValueError("D03 NPZ filename differs from the schema")
    content = manifest.get("content")
    if not isinstance(content, dict) or set(content) != {
        "npz_sha256",
        "semantic_sha256",
    }:
        raise ValueError("D03 content hash block is invalid")
    if _required_sha256(content["npz_sha256"], name="content.npz_sha256") != (
        _sha256_file(entries[ARRAYS_NAME])
    ):
        raise ValueError("D03 NPZ hash does not match its manifest")
    semantic_manifest = {
        name: value for name, value in manifest.items() if name != "content"
    }
    if _required_sha256(
        content["semantic_sha256"], name="content.semantic_sha256"
    ) != _sha256_bytes(_canonical_json_bytes(semantic_manifest)):
        raise ValueError("D03 semantic hash does not match its manifest")
    try:
        with np.load(entries[ARRAYS_NAME], allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True, order="C")
                for name in archive.files
            }
    except (OSError, ValueError) as exc:
        raise ValueError("invalid D03 NPZ") from exc
    if set(arrays) != _ARRAY_NAMES:
        raise ValueError("D03 array schema is incomplete or unexpected")
    if manifest.get("arrays") != _array_descriptors(arrays):
        raise ValueError("D03 array descriptors or hashes differ")
    for name, array in arrays.items():
        if array.dtype.hasobject:
            raise ValueError(f"D03 array {name!r} requires pickle")
        if name in _IDENTIFIER_ARRAYS:
            if array.dtype.kind != "U":
                raise ValueError(f"D03 identifier array {name!r} must be Unicode")
        elif name in _HASH_ARRAYS:
            if array.dtype != np.dtype("S64"):
                raise ValueError(f"D03 hash array {name!r} must use fixed ASCII S64")
        elif array.dtype.kind in {"b", "i", "u", "f"}:
            if not np.isfinite(array).all():
                raise FloatingPointError(f"D03 array {name!r} is non-finite")
        else:
            raise ValueError(f"D03 array {name!r} has an unsupported dtype")
    raw_sample_count = manifest.get("sample_count")
    if type(raw_sample_count) is not int or raw_sample_count <= 0:
        raise ValueError("D03 sample count is invalid")
    sample_count = raw_sample_count
    if arrays["sample_id"].shape != (sample_count,):
        raise ValueError("D03 sample count is invalid")
    if arrays["sample_id"].tolist() != sorted(arrays["sample_id"].tolist()):
        raise ValueError("D03 sample IDs are not stably sorted")
    if len(set(arrays["sample_id"].tolist())) != sample_count:
        raise ValueError("D03 sample IDs are not unique")
    for name in _IDENTIFIER_ARRAYS:
        if any(
            not str(value).strip() or "\x00" in str(value)
            for value in arrays[name].tolist()
        ):
            raise ValueError(f"D03 stable identifier {name!r} is invalid")
    provenance = manifest.get("provenance")
    required_provenance = {
        "cache_manifest_sha256",
        "checkpoint_sha256",
        "code_sha256",
        "config_sha256",
        "dataset",
        "derived_metadata_sha256",
        "device_uuid",
        "model_seed",
        "model_sha256",
        "normalization_sha256",
        "physical_gpu_index",
        "run_contract_sha256",
        "source_metadata_sha256",
        "split",
        "split_manifest_sha256",
    }
    if not isinstance(provenance, dict) or set(provenance) != required_provenance:
        raise ValueError("D03 provenance schema is incomplete or unexpected")
    for name in required_provenance - {
        "dataset",
        "device_uuid",
        "model_seed",
        "physical_gpu_index",
        "split",
    }:
        _required_sha256(provenance[name], name=f"provenance {name}")
    if provenance["dataset"] not in {"CWRU", "XJTU"}:
        raise ValueError("D03 provenance dataset is invalid")
    if provenance["split"] not in {"validation", "test"}:
        raise ValueError("D03 provenance split is invalid")
    if type(provenance["model_seed"]) is not int or not 0 <= provenance[
        "model_seed"
    ] < 2**63:
        raise ValueError("D03 provenance model seed is invalid")

    class_count = 2 if provenance["dataset"] == "XJTU" else 4
    expected_prefix = (sample_count, len(SNR_LEVELS_DB), DRAWS_PER_LEVEL)
    expected_shapes = {
        "sample_id": (sample_count,),
        "record_id": (sample_count,),
        "group_id": (sample_count,),
        "window_start": (sample_count,),
        "window_end": (sample_count,),
        "y": (sample_count,),
        "snr_db": (len(SNR_LEVELS_DB),),
        "draw_index": (DRAWS_PER_LEVEL,),
        "rule_index": (RULE_COUNT,),
        "noise_seed": (sample_count, len(SNR_LEVELS_DB)),
        "noise_sha256": expected_prefix,
        "input_window_sha256": (sample_count,),
        "signal_power": (sample_count,),
        "target_noise_std": (sample_count, len(SNR_LEVELS_DB)),
        "realized_noise_power": expected_prefix,
        "realized_snr_db": expected_prefix,
        "original_logits": (sample_count, class_count),
        "original_prediction": (sample_count,),
        "original_normalized_rule_firing": (sample_count, RULE_COUNT),
        "original_reference_class_attribution": (sample_count, RULE_COUNT),
        "noisy_logits": (*expected_prefix, class_count),
        "noisy_normalized_rule_firing": (*expected_prefix, RULE_COUNT),
        "noisy_reference_class_attribution": (*expected_prefix, RULE_COUNT),
        "top_rule_agreement": expected_prefix,
        "top3_jaccard": expected_prefix,
        "firing_vector_jsd": expected_prefix,
        "attribution_rank_tau": expected_prefix,
        "prediction_agreement": expected_prefix,
    }
    expected_dtypes = {
        **{name: np.dtype("<i8") for name in (
            "window_start",
            "window_end",
            "y",
            "snr_db",
            "draw_index",
            "rule_index",
            "original_prediction",
        )},
        "noise_seed": np.dtype("<u8"),
        **{name: np.dtype("<f4") for name in (
            "original_logits",
            "original_normalized_rule_firing",
            "original_reference_class_attribution",
            "noisy_logits",
            "noisy_normalized_rule_firing",
            "noisy_reference_class_attribution",
        )},
        **{name: np.dtype("<f8") for name in (
            "signal_power",
            "target_noise_std",
            "realized_noise_power",
            "realized_snr_db",
            "top3_jaccard",
            "firing_vector_jsd",
            "attribution_rank_tau",
        )},
        "top_rule_agreement": np.dtype(np.bool_),
        "prediction_agreement": np.dtype(np.bool_),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"D03 array {name!r} must have shape {shape}")
    for name, dtype in expected_dtypes.items():
        if arrays[name].dtype != dtype:
            raise ValueError(f"D03 array {name!r} must have dtype {dtype.str}")
    if not np.array_equal(arrays["snr_db"], np.asarray(SNR_LEVELS_DB)):
        raise ValueError("D03 SNR levels differ from the frozen protocol")
    if not np.array_equal(arrays["draw_index"], np.arange(DRAWS_PER_LEVEL)):
        raise ValueError("D03 draw indices differ from the frozen protocol")
    if not np.array_equal(arrays["rule_index"], np.arange(RULE_COUNT)):
        raise ValueError("D03 rule indices differ from the frozen protocol")
    input_binding = manifest.get("input_binding")
    if not isinstance(input_binding, dict) or set(input_binding) != {
        "input_count",
        "selected_count",
        "selected_float32_windows",
        "selection",
    }:
        raise ValueError("D03 input binding is invalid")
    window_binding = input_binding["selected_float32_windows"]
    if not isinstance(window_binding, dict) or set(window_binding) != {
        "dtype",
        "per_window_sha256_array",
        "sample_id_and_window_sha256_semantic_sha256",
        "shape",
    }:
        raise ValueError("D03 standardized-window binding is invalid")
    if (
        window_binding["dtype"] != "<f4"
        or not isinstance(window_binding["shape"], list)
        or len(window_binding["shape"]) != 3
        or window_binding["shape"][0] != sample_count
        or window_binding["shape"][2] != 2
        or window_binding["per_window_sha256_array"]
        != "input_window_sha256"
    ):
        raise ValueError("D03 standardized-window shape binding is invalid")
    _required_sha256(
        window_binding["sample_id_and_window_sha256_semantic_sha256"],
        name="input_binding selected windows semantic sha256",
    )
    window_size = window_binding["shape"][1]
    if type(window_size) is not int or window_size <= 0:
        raise ValueError("D03 bound window size is invalid")
    if input_binding["selected_count"] != sample_count:
        raise ValueError("D03 selected count differs from the output arrays")
    if type(input_binding["input_count"]) is not int or input_binding[
        "input_count"
    ] < sample_count:
        raise ValueError("D03 input count is invalid")
    expected_input_semantic = _sha256_bytes(
        _canonical_json_bytes(
            [
                {"sample_id": sample_id, "sha256": window_hash}
                for sample_id, window_hash in zip(
                    arrays["sample_id"].tolist(),
                    _hash_strings(arrays["input_window_sha256"]),
                    strict=True,
                )
            ]
        )
    )
    if (
        window_binding["sample_id_and_window_sha256_semantic_sha256"]
        != expected_input_semantic
    ):
        raise ValueError("D03 selected-window semantic binding is inconsistent")
    for index in range(sample_count):
        expected_id = (
            f"{arrays['record_id'][index]}:{arrays['window_start'][index]}:"
            f"{arrays['window_end'][index]}"
        )
        if arrays["sample_id"][index] != expected_id:
            raise ValueError("D03 stable sample ID binding is invalid")
    if np.any(arrays["window_start"] < 0) or np.any(
        arrays["window_end"] - arrays["window_start"] != window_size
    ):
        raise ValueError("D03 window boundaries differ from the bound window size")
    if np.any(arrays["y"] < 0) or np.any(arrays["y"] >= class_count):
        raise ValueError("D03 labels contain an out-of-range class")

    execution = manifest.get("execution")
    if not isinstance(execution, dict) or set(execution) != {
        "actual_forward_calls",
        "budget_retained",
        "chunk_count",
        "chunk_size",
        "device_class",
        "phase",
    }:
        raise ValueError("D03 execution block is invalid")
    if (
        type(execution["chunk_size"]) is not int
        or not 1 <= execution["chunk_size"] <= MAX_CHUNK_SIZE
        or type(execution["chunk_count"]) is not int
        or execution["chunk_count"] <= 0
    ):
        raise ValueError("D03 bounded chunk execution metadata is invalid")
    expected_chunk_count = math.ceil(sample_count / execution["chunk_size"])
    if execution["chunk_count"] != expected_chunk_count:
        raise ValueError("D03 chunk count differs from selected sample count")
    if execution["actual_forward_calls"] != execution["chunk_count"] * (
        1 + TOTAL_NOISE_DRAWS
    ):
        raise ValueError("D03 actual forward count differs from the frozen protocol")
    if execution["phase"] == "pilot_benchmark":
        if (
            execution["budget_retained"] is not None
            or provenance["split"] != "validation"
            or provenance["model_seed"] != 20260801
            or input_binding["selection"]
            != "first_256_after_stable_sample_id_sort"
            or sample_count != 256
            or execution["chunk_size"] != 256
            or execution["chunk_count"] != 1
            or execution["actual_forward_calls"] != 1 + TOTAL_NOISE_DRAWS
        ):
            raise ValueError("D03 pilot execution binding is invalid")
    elif execution["phase"] == "budget_retained_secondary":
        if (
            execution["budget_retained"] is not True
            or provenance["split"] != "test"
            or provenance["model_seed"] not in REGISTERED_FINAL_SEEDS
            or input_binding["selection"] != "all_after_stable_sample_id_sort"
        ):
            raise ValueError("D03 secondary execution binding is invalid")
    else:
        raise ValueError("D03 execution phase is invalid")
    if execution["device_class"] == "cuda":
        if (
            provenance["physical_gpu_index"] not in {0, 1}
            or not isinstance(provenance["device_uuid"], str)
            or not provenance["device_uuid"].startswith("GPU-")
            or any(character.isspace() for character in provenance["device_uuid"])
        ):
            raise ValueError("D03 CUDA device provenance is invalid")
    elif execution["device_class"] == "cpu_test_only":
        if (
            provenance["physical_gpu_index"] is not None
            or provenance["device_uuid"] is not None
        ):
            raise ValueError("D03 CPU-test artifact falsely records a GPU")
    else:
        raise ValueError("D03 device class is invalid")

    coverage = manifest.get("partition_coverage")
    if not isinstance(coverage, dict) or set(coverage) != {
        "coverage",
        "expected_sample_count",
        "expected_sample_id_semantic_sha256",
        "observed_sample_count",
        "observed_sample_id_semantic_sha256",
        "selected_sample_count",
        "selected_sample_id_semantic_sha256",
    }:
        raise ValueError("D03 partition coverage block is invalid")
    if (
        coverage["coverage"] != "exact"
        or type(coverage["expected_sample_count"]) is not int
        or coverage["expected_sample_count"] <= 0
        or coverage["observed_sample_count"]
        != coverage["expected_sample_count"]
        or coverage["observed_sample_count"] != input_binding["input_count"]
        or coverage["selected_sample_count"] != sample_count
    ):
        raise ValueError("D03 partition coverage counts are inconsistent")
    if execution["device_class"] == "cuda" and coverage[
        "expected_sample_count"
    ] != REGISTERED_PARTITION_WINDOW_COUNTS[
        (provenance["dataset"], provenance["split"])
    ]:
        raise ValueError("D03 CUDA artifact does not cover the frozen partition size")
    for name in (
        "expected_sample_id_semantic_sha256",
        "observed_sample_id_semantic_sha256",
        "selected_sample_id_semantic_sha256",
    ):
        _required_sha256(coverage[name], name=f"partition_coverage {name}")
    if (
        coverage["expected_sample_id_semantic_sha256"]
        != coverage["observed_sample_id_semantic_sha256"]
        or coverage["selected_sample_id_semantic_sha256"]
        != _sample_id_semantic_sha256(arrays["sample_id"].tolist())
    ):
        raise ValueError("D03 partition sample-ID hashes are inconsistent")
    if execution["phase"] == "pilot_benchmark":
        if coverage["selected_sample_count"] != 256:
            raise ValueError("D03 pilot must select exactly 256 stable-ID windows")
    elif (
        coverage["selected_sample_count"] != coverage["expected_sample_count"]
        or coverage["selected_sample_id_semantic_sha256"]
        != coverage["expected_sample_id_semantic_sha256"]
    ):
        raise ValueError("D03 decisive artifact is not full-partition complete")

    model_state = manifest.get("model_state")
    if not isinstance(model_state, dict) or set(model_state) != {
        "after_sha256",
        "before_sha256",
        "unchanged",
    }:
        raise ValueError("D03 model-state block is invalid")
    if (
        model_state["unchanged"] is not True
        or model_state["before_sha256"] != model_state["after_sha256"]
        or model_state["before_sha256"] != provenance["model_sha256"]
    ):
        raise ValueError("D03 model-state hashes are not unchanged and bound")

    if manifest.get("protocol") != _protocol_payload():
        raise ValueError("D03 protocol metadata differs from the frozen definition")
    if manifest.get("format") != {
        "container": "numpy.npz",
        "hash_arrays": sorted(_HASH_ARRAYS),
        "identifier_arrays": sorted(_IDENTIFIER_ARRAYS),
        "load_allow_pickle": False,
        "model_and_noise_dtype": "float32",
        "object_arrays": False,
        "offline_metric_dtype": "float64",
    }:
        raise ValueError("D03 format metadata is invalid")

    if np.any(arrays["signal_power"] <= 0.0) or np.any(
        arrays["realized_noise_power"] <= 0.0
    ):
        raise ValueError("D03 signal and realized noise powers must be positive")
    if any(
        _SHA256.fullmatch(value) is None
        for value in _hash_strings(arrays["input_window_sha256"])
    ):
        raise ValueError("D03 input-window hashes are invalid")
    expected_std = np.sqrt(
        arrays["signal_power"][:, None]
        / np.power(10.0, arrays["snr_db"][None, :] / 10.0)
    )
    if not np.allclose(
        arrays["target_noise_std"], expected_std, atol=0.0, rtol=1.0e-14
    ):
        raise ValueError("D03 target noise standard deviation is inconsistent")
    expected_realized_snr = 10.0 * np.log10(
        arrays["signal_power"][:, None, None]
        / arrays["realized_noise_power"]
    )
    if not np.allclose(
        arrays["realized_snr_db"], expected_realized_snr, atol=1.0e-12, rtol=0.0
    ):
        raise ValueError("D03 realized SNR is inconsistent with recorded power")
    flattened_seeds = arrays["noise_seed"].reshape(-1).tolist()
    if len(set(flattened_seeds)) != len(flattened_seeds):
        raise ValueError("D03 noise seeds are not globally unique")
    for sample_index, sample_id in enumerate(arrays["sample_id"].tolist()):
        for snr_index, snr_db in enumerate(SNR_LEVELS_DB):
            expected_seed = p05_d03_noise_seed(
                dataset=provenance["dataset"],
                split=provenance["split"],
                model_seed=provenance["model_seed"],
                sample_id=sample_id,
                snr_db=snr_db,
            )
            if int(arrays["noise_seed"][sample_index, snr_index]) != expected_seed:
                raise ValueError("D03 noise seed differs from the frozen derivation")
            hashes = _hash_strings(
                arrays["noise_sha256"][sample_index, snr_index]
            )
            if len(set(hashes)) != DRAWS_PER_LEVEL or any(
                _SHA256.fullmatch(value) is None for value in hashes
            ):
                raise ValueError("D03 noise draw hashes are invalid or duplicated")

    original_firing = arrays["original_normalized_rule_firing"]
    noisy_firing = arrays["noisy_normalized_rule_firing"]
    if np.any(original_firing < 0.0) or np.any(noisy_firing < 0.0):
        raise ValueError("D03 normalized firing contains negative mass")
    if not np.allclose(
        original_firing.sum(axis=1),
        1.0,
        atol=RECONSTRUCTION_ATOL,
        rtol=0.0,
    ):
        raise ValueError("D03 original firing rows do not sum to one")
    if not np.allclose(
        noisy_firing.sum(axis=-1),
        1.0,
        atol=RECONSTRUCTION_ATOL,
        rtol=0.0,
    ):
        raise ValueError("D03 noisy firing rows do not sum to one")
    expected_prediction = np.argmax(arrays["original_logits"], axis=1)
    if not np.array_equal(arrays["original_prediction"], expected_prediction):
        raise ValueError("D03 original predictions differ from logits")
    if np.any((arrays["top3_jaccard"] < 0.0) | (arrays["top3_jaccard"] > 1.0)):
        raise ValueError("D03 top-3 Jaccard is outside [0,1]")
    if np.any(arrays["firing_vector_jsd"] < 0.0):
        raise ValueError("D03 firing-vector JSD is negative")
    if np.any(
        (arrays["attribution_rank_tau"] < -1.0)
        | (arrays["attribution_rank_tau"] > 1.0)
    ):
        raise ValueError("D03 attribution-rank tau is outside [-1,1]")
    for sample_index in range(sample_count):
        clean_attribution = arrays["original_reference_class_attribution"][
            sample_index
        ]
        clean_order = _rule_order(clean_attribution)
        for snr_index in range(len(SNR_LEVELS_DB)):
            for draw_index in range(DRAWS_PER_LEVEL):
                noisy_attribution = arrays["noisy_reference_class_attribution"][
                    sample_index, snr_index, draw_index
                ]
                noisy_order = _rule_order(noisy_attribution)
                expected_top = clean_order[0] == noisy_order[0]
                intersection = len(set(clean_order[:3]) & set(noisy_order[:3]))
                expected_jaccard = intersection / (6 - intersection)
                expected_jsd = _natural_log_jsd(
                    original_firing[sample_index],
                    noisy_firing[sample_index, snr_index, draw_index],
                )
                expected_tau = _rank_tau(clean_attribution, noisy_attribution)
                expected_agreement = (
                    np.argmax(
                        arrays["noisy_logits"][sample_index, snr_index, draw_index]
                    )
                    == expected_prediction[sample_index]
                )
                if arrays["top_rule_agreement"][
                    sample_index, snr_index, draw_index
                ] != expected_top:
                    raise ValueError("D03 top-rule agreement is inconsistent")
                if not math.isclose(
                    float(arrays["top3_jaccard"][sample_index, snr_index, draw_index]),
                    expected_jaccard,
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                ):
                    raise ValueError("D03 top-3 Jaccard is inconsistent")
                if not math.isclose(
                    float(arrays["firing_vector_jsd"][sample_index, snr_index, draw_index]),
                    expected_jsd,
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                ):
                    raise ValueError("D03 firing-vector JSD is inconsistent")
                if not math.isclose(
                    float(arrays["attribution_rank_tau"][sample_index, snr_index, draw_index]),
                    expected_tau,
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                ):
                    raise ValueError("D03 attribution-rank tau is inconsistent")
                if arrays["prediction_agreement"][
                    sample_index, snr_index, draw_index
                ] != expected_agreement:
                    raise ValueError("D03 prediction agreement is inconsistent")
    if manifest.get("conclusion_control") != {
        "claim_decisions": "not_performed",
        "confirmatory_sign_tests": "not_performed",
        "performance_claim": False,
        "scientific_status": "computed_unadjudicated",
        "scope": "budget_conditional_secondary_P05_D03_only",
    }:
        raise ValueError("D03 artifact contains an invalid conclusion control")
    return MappingProxyType(manifest)


def _network_preflight(
    *,
    artifact_dir: str | Path,
    network: torch.nn.Module,
    provenance: P05D03Provenance,
    require_cuda: bool,
) -> tuple[Path, torch.device, str, dict[str, Any]]:
    if not isinstance(network, torch.nn.Module):
        raise TypeError("network must be a torch.nn.Module")
    if type(require_cuda) is not bool:
        raise TypeError("require_cuda must be a boolean")
    provenance_payload = _validate_provenance(provenance, require_cuda=require_cuda)
    target = Path(os.path.abspath(os.fspath(artifact_dir)))
    if target.is_symlink() or target.exists():
        raise FileExistsError(f"D03 artifact target is create-only: {target}")
    device = _network_device(network)
    if require_cuda and device.type != "cuda":
        raise RuntimeError("P05-D03 evidence execution requires a CUDA model")
    if not require_cuda and device.type != "cpu":
        raise RuntimeError("non-CUDA D03 execution is restricted to CPU tests")
    _require_float32_model(network)
    state_before = model_state_sha256(network)
    if state_before != provenance_payload["model_sha256"]:
        raise ValueError("P05-D03 model state does not match provenance model_sha256")
    return target, device, state_before, provenance_payload


def _execute_and_publish(
    *,
    target: Path,
    network: torch.nn.Module,
    chunks: Iterable[_PreparedBatch],
    provenance: P05D03Provenance,
    provenance_payload: Mapping[str, Any],
    device: torch.device,
    state_before: str,
    phase: str,
    budget_retained: bool | None,
    require_cuda: bool,
    chunk_size: int,
    expected_window_size: int,
    expected_sample_ids: tuple[str, ...],
    observed_sample_ids: list[str],
) -> P05D03Result:
    was_training = bool(network.training)
    network.eval()
    try:
        execution = _execute_chunks(
            network=network,
            chunks=chunks,
            provenance=provenance,
            device=device,
        )
    finally:
        network.train(was_training)
    state_after = model_state_sha256(network)
    if state_after != state_before:
        raise RuntimeError("P05-D03 forwards mutated the checkpoint/model state")

    selected_sample_ids = execution.arrays["sample_id"].tolist()
    coverage = _partition_coverage(
        expected_sample_ids=expected_sample_ids,
        observed_sample_ids=observed_sample_ids,
        selected_sample_ids=selected_sample_ids,
        phase=phase,
    )
    input_binding = _input_binding(
        execution.arrays,
        expected_window_size=expected_window_size,
        input_count=len(observed_sample_ids),
        phase=phase,
    )
    semantic_manifest = _semantic_manifest(
        arrays=execution.arrays,
        provenance=provenance_payload,
        input_binding=input_binding,
        partition_coverage=coverage,
        state_sha256=state_before,
        phase=phase,
        require_cuda=require_cuda,
        budget_retained=budget_retained,
        chunk_size=chunk_size,
        chunk_count=execution.chunk_count,
    )
    result = _write_package(
        target,
        arrays=execution.arrays,
        semantic_manifest=semantic_manifest,
        timing=execution.timing,
    )
    verify_p05_d03_artifact(result.artifact_dir)
    return result


def run_p05_d03_noise_interventions(
    artifact_dir: str | Path,
    *,
    network: torch.nn.Module,
    batch: Mapping[str, Any],
    provenance: P05D03Provenance,
    expected_sample_ids: Sequence[str],
    phase: str,
    budget_retained: bool | None,
    expected_window_size: int = 4096,
    require_cuda: bool = True,
    benchmark_first_n: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> P05D03Result:
    """Run D03 from one complete in-memory partition using bounded chunks.

    ``expected_sample_ids`` must enumerate the complete registered partition;
    a single loader batch cannot therefore be relabeled as a full-partition
    artifact. Prefer :func:`run_p05_d03_noise_interventions_from_loader` for
    decisive partitions so the standardized windows remain bounded in memory.
    """

    if type(expected_window_size) is not int or expected_window_size <= 0:
        raise ValueError("expected_window_size must be a positive integer")
    bounded_chunk_size = _validate_chunk_size(chunk_size)
    _validate_phase(
        provenance,
        phase=phase,
        budget_retained=budget_retained,
        benchmark_first_n=benchmark_first_n,
    )
    _validate_phase_chunk_size(phase=phase, chunk_size=bounded_chunk_size)
    expected = _normalize_expected_sample_ids(expected_sample_ids)
    _validate_registered_partition_count(
        provenance=provenance,
        expected_sample_ids=expected,
        require_cuda=require_cuda,
    )
    if phase == "pilot_benchmark" and len(expected) < 256:
        raise ValueError("pilot benchmark partition must contain at least 256 windows")
    target, device, state_before, checked_provenance = _network_preflight(
        artifact_dir=artifact_dir,
        network=network,
        provenance=provenance,
        require_cuda=require_cuda,
    )
    provenance_payload = checked_provenance
    complete = _prepare_batch(
        batch,
        expected_window_size=expected_window_size,
        benchmark_first_n=None,
    )
    observed = complete.sample_id.tolist()
    if tuple(observed) != expected:
        missing = sorted(set(expected) - set(observed))
        unexpected = sorted(set(observed) - set(expected))
        raise ValueError(
            "D03 batch coverage differs from expected_sample_ids: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    selected_count = 256 if phase == "pilot_benchmark" else len(complete.sample_id)
    selected = _slice_prepared(
        complete,
        np.arange(selected_count, dtype=np.int64),
    )
    return _execute_and_publish(
        target=target,
        network=network,
        chunks=_prepared_chunks(selected, chunk_size=bounded_chunk_size),
        provenance=provenance,
        provenance_payload=provenance_payload,
        device=device,
        state_before=state_before,
        phase=phase,
        budget_retained=budget_retained,
        require_cuda=require_cuda,
        chunk_size=bounded_chunk_size,
        expected_window_size=expected_window_size,
        expected_sample_ids=expected,
        observed_sample_ids=observed,
    )


def run_p05_d03_noise_interventions_from_loader(
    artifact_dir: str | Path,
    *,
    network: torch.nn.Module,
    batches: Iterable[Mapping[str, Any]],
    provenance: P05D03Provenance,
    expected_sample_ids: Sequence[str],
    phase: str,
    budget_retained: bool | None,
    expected_window_size: int = 4096,
    require_cuda: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> P05D03Result:
    """Stream a complete partition and publish one sorted aggregate artifact."""

    if isinstance(batches, (str, bytes)) or not isinstance(batches, Iterable):
        raise TypeError("batches must be an iterable of batch mappings")
    if type(expected_window_size) is not int or expected_window_size <= 0:
        raise ValueError("expected_window_size must be a positive integer")
    bounded_chunk_size = _validate_chunk_size(chunk_size)
    benchmark_first_n = 256 if phase == "pilot_benchmark" else None
    _validate_phase(
        provenance,
        phase=phase,
        budget_retained=budget_retained,
        benchmark_first_n=benchmark_first_n,
    )
    _validate_phase_chunk_size(phase=phase, chunk_size=bounded_chunk_size)
    expected = _normalize_expected_sample_ids(expected_sample_ids)
    _validate_registered_partition_count(
        provenance=provenance,
        expected_sample_ids=expected,
        require_cuda=require_cuda,
    )
    if phase == "pilot_benchmark" and len(expected) < 256:
        raise ValueError("pilot benchmark partition must contain at least 256 windows")
    selected_set = set(expected[:256] if phase == "pilot_benchmark" else expected)
    expected_set = set(expected)
    target, device, state_before, checked_provenance = _network_preflight(
        artifact_dir=artifact_dir,
        network=network,
        provenance=provenance,
        require_cuda=require_cuda,
    )
    provenance_payload = checked_provenance

    observed: list[str] = []
    observed_set: set[str] = set()

    def _stream_chunks() -> Iterable[_PreparedBatch]:
        pending: list[tuple[Any, ...]] = []
        for batch_index, batch in enumerate(batches):
            if not isinstance(batch, Mapping):
                raise TypeError(f"D03 loader batch[{batch_index}] must be a mapping")
            required_fields = {
                "x",
                "y",
                "sample_id",
                "record_id",
                "group_id",
                "window_start",
                "window_end",
            }
            missing = sorted(required_fields - set(batch))
            if missing:
                raise ValueError(
                    f"D03 loader batch[{batch_index}] is missing fields {missing}"
                )
            stable_batch = {name: batch[name] for name in required_fields}
            prepared = _prepare_batch(
                stable_batch,
                expected_window_size=expected_window_size,
                benchmark_first_n=None,
            )
            for index, sample_id in enumerate(prepared.sample_id.tolist()):
                if sample_id in observed_set:
                    raise ValueError(
                        f"D03 loader emitted duplicate sample_id {sample_id!r}"
                    )
                if sample_id not in expected_set:
                    raise ValueError(
                        f"D03 loader batch[{batch_index}] emitted unexpected "
                        f"sample_id {sample_id!r}"
                    )
                observed_set.add(sample_id)
                observed.append(sample_id)
                if sample_id not in selected_set:
                    continue
                pending.append(
                    (
                        prepared.x[index],
                        int(prepared.y[index]),
                        sample_id,
                        str(prepared.record_id[index]),
                        str(prepared.group_id[index]),
                        int(prepared.window_start[index]),
                        int(prepared.window_end[index]),
                    )
                )
                if len(pending) == bounded_chunk_size:
                    yield _prepared_from_rows(pending)
                    pending = []
        if pending:
            yield _prepared_from_rows(pending)

    return _execute_and_publish(
        target=target,
        network=network,
        chunks=_stream_chunks(),
        provenance=provenance,
        provenance_payload=provenance_payload,
        device=device,
        state_before=state_before,
        phase=phase,
        budget_retained=budget_retained,
        require_cuda=require_cuda,
        chunk_size=bounded_chunk_size,
        expected_window_size=expected_window_size,
        expected_sample_ids=expected,
        observed_sample_ids=observed,
    )


__all__ = [
    "ARRAYS_NAME",
    "DEFAULT_CHUNK_SIZE",
    "DRAWS_PER_LEVEL",
    "MANIFEST_NAME",
    "MAX_CHUNK_SIZE",
    "P05D03Provenance",
    "P05D03Result",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SNR_LEVELS_DB",
    "TOTAL_NOISE_DRAWS",
    "p05_d03_noise_seed",
    "run_p05_d03_noise_interventions",
    "run_p05_d03_noise_interventions_from_loader",
    "verify_p05_d03_artifact",
]
