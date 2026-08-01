"""Actual same-checkpoint forward runner for the frozen P05 C2 interventions."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .p05_intervention_eval import (
    generate_unique_nonidentity_permutations,
    shuffle_seed,
)
from .p05_trace_export import P05TraceBatch
from .p05_trace_runner import model_state_sha256


RECONSTRUCTION_ATOL = 1.0e-6
RECONSTRUCTION_RTOL = 1.0e-6
RULE_COUNT = 10
SHUFFLE_COUNT = 32
PILOT_BENCHMARK_SAMPLE_COUNT = 256
PILOT_MODEL_SEED = 20260801
REGISTERED_VALIDATION_WINDOW_COUNTS = {
    "CWRU": 19 * 16,
    "XJTU": 1317 * 4,
}

_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_TRACE_FLOAT_FIELDS = (
    "reduced_features",
    "membership_values",
    "centers",
    "widths",
    "antecedent_probabilities",
    "antecedent_memberships",
    "log_rule_firing",
    "rule_firing",
    "normalized_rule_firing",
    "rule_consequents",
    "rule_contributions",
    "fuzzy_logits",
)
_BATCH_INVARIANT_FIELDS = (
    "reduced_features",
    "membership_values",
    "antecedent_memberships",
    "log_rule_firing",
    "rule_firing",
)
_SHARED_INVARIANT_FIELDS = (
    "centers",
    "widths",
    "antecedent_probabilities",
)
_IDENTIFIER_ARRAYS = {"sample_id", "record_id", "group_id"}
_ACTUAL_ARRAYS = {
    "actual_deletion_logits",
    "actual_deletion_normalized_rule_firing",
    "actual_deletion_rule_contributions",
    "actual_deletion_invariant_max_abs",
    "actual_deletion_membership_invariant_pass",
    "actual_deletion_antecedent_invariant_pass",
    "actual_deletion_firing_invariant_pass",
    "actual_shuffle_seed",
    "actual_shuffle_permutations",
    "actual_shuffle_logits",
    "actual_shuffle_rule_contributions",
    "actual_shuffle_invariant_max_abs",
    "actual_shuffle_membership_invariant_pass",
    "actual_shuffle_antecedent_invariant_pass",
    "actual_shuffle_firing_invariant_pass",
}
_ORIGINAL_ARRAYS = {
    "sample_id",
    "record_id",
    "group_id",
    "window_start",
    "window_end",
    "y",
    "logits",
    "non_fuzzy_logits",
    "fuzzy_scale",
    *(f"trace_{name}" for name in _TRACE_FLOAT_FIELDS),
    "trace_rule_mask",
    "trace_consequent_permutation",
}


@dataclass(frozen=True)
class P05InterventionProvenance:
    dataset: str
    split: str
    model_seed: int
    config_sha256: str
    checkpoint_sha256: str
    model_sha256: str


@dataclass(frozen=True)
class P05ActualInterventionResult:
    """Validated arrays and non-evidentiary timing from actual model forwards."""

    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]
    timing: Mapping[str, Any]
    semantic_sha256: str

    def c2_evaluator_kwargs(self, sample_index: int) -> dict[str, Any]:
        """Return exactly the original inputs consumed by the offline C2 evaluator."""

        count = int(self.arrays["sample_id"].shape[0])
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise TypeError("sample_index must be an integer")
        if not 0 <= sample_index < count:
            raise IndexError(f"sample_index must be within [0, {count})")
        provenance = self.metadata["provenance"]
        return {
            "dataset": provenance["dataset"],
            "split": provenance["split"],
            "model_seed": provenance["model_seed"],
            "sample_id": str(self.arrays["sample_id"][sample_index]),
            "logits": self.arrays["logits"][sample_index],
            "non_fuzzy_logits": self.arrays["non_fuzzy_logits"][sample_index],
            "fuzzy_scale": float(np.asarray(self.arrays["fuzzy_scale"]).reshape(())),
            "log_rule_firing": self.arrays["trace_log_rule_firing"][sample_index],
            "rule_consequents": self.arrays["trace_rule_consequents"],
            "rule_contributions": self.arrays["trace_rule_contributions"][sample_index],
        }

    def as_trace_batch(self) -> P05TraceBatch:
        """Return the original forward as an export-ready complete trace batch."""

        trace = {
            name: self.arrays[f"trace_{name}"]
            for name in (
                *_TRACE_FLOAT_FIELDS,
                "rule_mask",
                "consequent_permutation",
            )
        }
        return P05TraceBatch(
            sample_id=self.arrays["sample_id"].tolist(),
            record_id=self.arrays["record_id"].tolist(),
            group_id=self.arrays["group_id"].tolist(),
            window_start=self.arrays["window_start"],
            window_end=self.arrays["window_end"],
            y=self.arrays["y"],
            logits=self.arrays["logits"],
            non_fuzzy_logits=self.arrays["non_fuzzy_logits"],
            fuzzy_scale=self.arrays["fuzzy_scale"],
            fuzzy_trace=trace,
        )


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
    benchmark_first_n: int | None


def _required_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _validate_provenance(value: P05InterventionProvenance) -> dict[str, Any]:
    if not isinstance(value, P05InterventionProvenance):
        raise TypeError("provenance must be P05InterventionProvenance")
    if value.dataset not in {"CWRU", "XJTU"}:
        raise ValueError("provenance dataset must be CWRU or XJTU")
    if value.split not in {"validation", "test"}:
        raise ValueError("provenance split must be validation or test")
    if (
        isinstance(value.model_seed, bool)
        or not isinstance(value.model_seed, int)
        or not 0 <= value.model_seed < 2**63
    ):
        raise ValueError("provenance model_seed must be a non-negative signed 64-bit integer")
    return {
        "checkpoint_sha256": _required_sha256(
            value.checkpoint_sha256, name="provenance checkpoint_sha256"
        ),
        "config_sha256": _required_sha256(
            value.config_sha256, name="provenance config_sha256"
        ),
        "dataset": value.dataset,
        "model_seed": value.model_seed,
        "model_sha256": _required_sha256(
            value.model_sha256, name="provenance model_sha256"
        ),
        "split": value.split,
    }


def _network_device(network: torch.nn.Module) -> torch.device:
    devices = {tensor.device for tensor in network.parameters()}
    devices.update(tensor.device for tensor in network.buffers())
    if not devices:
        raise ValueError("P05 intervention network exposes no parameter or buffer device")
    if len(devices) != 1:
        raise ValueError(
            "P05 intervention network spans multiple devices: "
            f"{sorted(map(str, devices))}"
        )
    return next(iter(devices))


def _identifier_vector(value: Any, *, name: str, count: int) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"batch {name} must be a sequence of strings")
    raw = np.asarray(value)
    if raw.shape != (count,):
        raise ValueError(f"batch {name} must have shape ({count},)")
    identifiers = raw.tolist()
    if any(
        not isinstance(item, str) or not item.strip() or "\x00" in item
        for item in identifiers
    ):
        raise ValueError(f"batch {name} values must be non-empty strings without NUL")
    width = max(len(item) for item in identifiers)
    return np.asarray(identifiers, dtype=f"<U{width}")


def _integer_vector(value: Any, *, name: str, count: int) -> np.ndarray:
    if torch.is_tensor(value):
        raw = value.detach().to(device="cpu").numpy()
    else:
        raw = np.asarray(value)
    if raw.shape != (count,) or raw.dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError(f"batch {name} must be a numeric vector with shape ({count},)")
    numeric = raw.astype(np.float64, copy=False)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.round(numeric)).all():
        raise ValueError(f"batch {name} must contain finite integer values")
    return numeric.astype(np.int64, copy=False)


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
            "batch fields differ from the stable intervention contract: "
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
    count = int(x.shape[0])
    if count <= 0:
        raise ValueError("batch must contain at least one sample")
    if benchmark_first_n is not None and (
        isinstance(benchmark_first_n, bool)
        or not isinstance(benchmark_first_n, int)
        or benchmark_first_n <= 0
    ):
        raise ValueError("benchmark_first_n must be a positive integer or None")

    sample_id = _identifier_vector(batch["sample_id"], name="sample_id", count=count)
    if len(set(sample_id.tolist())) != count:
        raise ValueError("batch sample_id values must be unique")
    record_id = _identifier_vector(batch["record_id"], name="record_id", count=count)
    group_id = _identifier_vector(batch["group_id"], name="group_id", count=count)
    y = _integer_vector(batch["y"], name="y", count=count)
    window_start = _integer_vector(
        batch["window_start"], name="window_start", count=count
    )
    window_end = _integer_vector(batch["window_end"], name="window_end", count=count)
    if np.any(window_start < 0) or np.any(window_end <= window_start):
        raise ValueError("batch requires 0 <= window_start < window_end")

    order = np.argsort(sample_id, kind="stable")
    if benchmark_first_n is not None:
        order = order[:benchmark_first_n]
    torch_order = torch.as_tensor(order, dtype=torch.long, device=x.device)
    return _PreparedBatch(
        x=x.index_select(0, torch_order),
        y=np.ascontiguousarray(y[order]),
        sample_id=np.ascontiguousarray(sample_id[order]),
        record_id=np.ascontiguousarray(record_id[order]),
        group_id=np.ascontiguousarray(group_id[order]),
        window_start=np.ascontiguousarray(window_start[order]),
        window_end=np.ascontiguousarray(window_end[order]),
        input_count=count,
        benchmark_first_n=benchmark_first_n,
    )


def _trace(output: Any, *, identity: str) -> Any:
    trace = getattr(output, "fuzzy_trace", None)
    if trace is None:
        raise ValueError(f"{identity} output is missing fuzzy_trace")
    return trace


def _validate_output(
    output: Any,
    *,
    identity: str,
    sample_ids: np.ndarray,
    expected_classes: int | None = None,
    allow_batched_consequents: bool = False,
) -> tuple[int, int]:
    count = len(sample_ids)
    for name in ("logits", "non_fuzzy_logits"):
        value = getattr(output, name, None)
        if not torch.is_tensor(value) or value.dtype != torch.float32:
            raise TypeError(f"{identity} output.{name} must be a float32 tensor")
        if value.ndim != 2 or value.shape[0] != count or not torch.isfinite(value).all():
            raise ValueError(f"{identity} output.{name} has an invalid batch shape or value")
    logits = output.logits
    classes = int(logits.shape[1])
    if classes < 2 or output.non_fuzzy_logits.shape != logits.shape:
        raise ValueError(f"{identity} output logits have an invalid class shape")
    if expected_classes is not None and classes != expected_classes:
        raise ValueError(f"{identity} output class count changed across interventions")
    scale = getattr(output, "fuzzy_scale", None)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError(f"{identity} fuzzy_scale must be numeric")
    if not math.isfinite(float(scale)):
        raise ValueError(f"{identity} fuzzy_scale must be finite")

    trace = _trace(output, identity=identity)
    for name in _TRACE_FLOAT_FIELDS:
        value = getattr(trace, name, None)
        if not torch.is_tensor(value) or value.dtype != torch.float32:
            raise TypeError(f"{identity} fuzzy_trace.{name} must be float32")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{identity} fuzzy_trace.{name} is non-finite")
    reduced = trace.reduced_features
    membership = trace.membership_values
    normalized = trace.normalized_rule_firing
    if reduced.ndim != 2 or reduced.shape[0] != count or reduced.shape[1] < 1:
        raise ValueError(f"{identity} reduced_features shape is invalid")
    features = int(reduced.shape[1])
    if membership.ndim != 3 or membership.shape[:2] != reduced.shape:
        raise ValueError(f"{identity} membership_values shape is invalid")
    memberships = int(membership.shape[2])
    if normalized.ndim != 2 or normalized.shape[0] != count:
        raise ValueError(f"{identity} normalized_rule_firing shape is invalid")
    rules = int(normalized.shape[1])
    consequent_shape = (
        (count, rules, classes)
        if allow_batched_consequents
        else (rules, classes)
    )
    shapes = {
        "centers": (features, memberships),
        "widths": (features, memberships),
        "antecedent_probabilities": (rules, features, memberships),
        "antecedent_memberships": (count, rules, features),
        "log_rule_firing": (count, rules),
        "rule_firing": (count, rules),
        "rule_consequents": consequent_shape,
        "rule_contributions": (count, rules, classes),
        "fuzzy_logits": (count, classes),
    }
    for name, shape in shapes.items():
        if tuple(getattr(trace, name).shape) != shape:
            raise ValueError(f"{identity} fuzzy_trace.{name} must have shape {shape}")
    if not torch.is_tensor(trace.rule_mask) or trace.rule_mask.dtype != torch.bool:
        raise TypeError(f"{identity} fuzzy_trace.rule_mask must be boolean")
    if tuple(trace.rule_mask.shape) != (count, rules):
        raise ValueError(f"{identity} fuzzy_trace.rule_mask shape is invalid")
    permutation = trace.consequent_permutation
    expected_permutation_shape = (
        (count, rules) if allow_batched_consequents else (rules,)
    )
    if (
        not torch.is_tensor(permutation)
        or permutation.dtype == torch.bool
        or permutation.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
        or tuple(permutation.shape) != expected_permutation_shape
    ):
        raise TypeError(
            f"{identity} consequent_permutation must have integer shape "
            f"{expected_permutation_shape}"
        )
    expected_rules = torch.arange(
        rules,
        dtype=torch.long,
        device=permutation.device,
    )
    expected_rows = (
        expected_rules
        if permutation.ndim == 1
        else expected_rules.unsqueeze(0).expand(count, -1)
    )
    if not torch.equal(
        permutation.to(torch.long).sort(dim=-1).values,
        expected_rows,
    ):
        raise ValueError(f"{identity} consequent_permutation rows are invalid")

    _assert_per_sample_close(
        trace.rule_firing,
        trace.log_rule_firing.exp(),
        sample_ids=sample_ids,
        identity=f"{identity} raw firing reconstruction",
    )
    expected_fuzzy = trace.rule_contributions.sum(dim=1)
    _assert_per_sample_close(
        trace.fuzzy_logits,
        expected_fuzzy,
        sample_ids=sample_ids,
        identity=f"{identity} fuzzy-logit reconstruction",
    )
    reconstructed = output.non_fuzzy_logits + float(scale) * expected_fuzzy
    _assert_per_sample_close(
        output.logits,
        reconstructed,
        sample_ids=sample_ids,
        identity=f"{identity} total-logit reconstruction",
    )
    return rules, classes


def _assert_per_sample_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    sample_ids: np.ndarray,
    identity: str,
) -> np.ndarray:
    if actual.shape != expected.shape:
        raise ValueError(
            f"{identity} shape changed: actual={tuple(actual.shape)}, "
            f"expected={tuple(expected.shape)}"
        )
    close = torch.isclose(
        actual,
        expected,
        atol=RECONSTRUCTION_ATOL,
        rtol=RECONSTRUCTION_RTOL,
    )
    per_sample = close.reshape(len(sample_ids), -1).all(dim=1)
    differences = (actual - expected).abs().reshape(len(sample_ids), -1)
    maxima = differences.max(dim=1).values.detach().to(device="cpu", dtype=torch.float64)
    if not bool(per_sample.all()):
        index = int(torch.nonzero(~per_sample, as_tuple=False)[0].item())
        raise ValueError(
            f"{identity} invariant failed for sample_id={sample_ids[index]!r}: "
            f"max_abs_error={float(maxima[index]):.12g}, "
            f"atol={RECONSTRUCTION_ATOL:.3g}, rtol={RECONSTRUCTION_RTOL:.3g}"
        )
    return maxima.numpy()


def _assert_shared_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    identity: str,
) -> float:
    if actual.shape != expected.shape or not torch.allclose(
        actual,
        expected,
        atol=RECONSTRUCTION_ATOL,
        rtol=RECONSTRUCTION_RTOL,
    ):
        maximum = (
            float((actual - expected).abs().max().detach().cpu())
            if actual.shape == expected.shape
            else float("inf")
        )
        raise ValueError(
            f"{identity} shared invariant failed: max_abs_error={maximum:.12g}"
        )
    return float((actual - expected).abs().max().detach().cpu())


def _invariant_residual(
    candidate: Any,
    reference: Any,
    *,
    sample_ids: np.ndarray,
    identity: str,
    include_normalized_firing: bool,
) -> np.ndarray:
    candidate_trace = _trace(candidate, identity=identity)
    reference_trace = _trace(reference, identity=f"{identity} reference")
    residual = _assert_per_sample_close(
        candidate.non_fuzzy_logits,
        reference.non_fuzzy_logits,
        sample_ids=sample_ids,
        identity=f"{identity} non-fuzzy logits",
    )
    fields = list(_BATCH_INVARIANT_FIELDS)
    if include_normalized_firing:
        fields.append("normalized_rule_firing")
    for name in fields:
        residual = np.maximum(
            residual,
            _assert_per_sample_close(
                getattr(candidate_trace, name),
                getattr(reference_trace, name),
                sample_ids=sample_ids,
                identity=f"{identity} {name}",
            ),
        )
    for name in _SHARED_INVARIANT_FIELDS:
        shared = _assert_shared_close(
            getattr(candidate_trace, name),
            getattr(reference_trace, name),
            identity=f"{identity} {name}",
        )
        residual = np.maximum(residual, shared)
    if not math.isclose(
        float(candidate.fuzzy_scale),
        float(reference.fuzzy_scale),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(f"{identity} fuzzy_scale invariant failed")
    return residual


def _to_numpy(value: torch.Tensor, *, dtype: np.dtype[Any]) -> np.ndarray:
    return np.ascontiguousarray(value.detach().to(device="cpu").numpy(), dtype=dtype)


def _original_arrays(output: Any, batch: _PreparedBatch) -> dict[str, np.ndarray]:
    trace = output.fuzzy_trace
    arrays: dict[str, np.ndarray] = {
        "sample_id": batch.sample_id,
        "record_id": batch.record_id,
        "group_id": batch.group_id,
        "window_start": batch.window_start,
        "window_end": batch.window_end,
        "y": batch.y,
        "logits": _to_numpy(output.logits, dtype=np.dtype("<f8")),
        "non_fuzzy_logits": _to_numpy(
            output.non_fuzzy_logits, dtype=np.dtype("<f8")
        ),
        "fuzzy_scale": np.asarray(float(output.fuzzy_scale), dtype="<f8"),
    }
    for name in _TRACE_FLOAT_FIELDS:
        arrays[f"trace_{name}"] = _to_numpy(
            getattr(trace, name), dtype=np.dtype("<f8")
        )
    arrays["trace_rule_mask"] = _to_numpy(
        trace.rule_mask, dtype=np.dtype(np.bool_)
    )
    arrays["trace_consequent_permutation"] = _to_numpy(
        trace.consequent_permutation, dtype=np.dtype("<i8")
    )
    return arrays


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_start(device: torch.device) -> int:
    _synchronize(device)
    return time.perf_counter_ns()


def _timed_end(device: torch.device, started_ns: int) -> float:
    _synchronize(device)
    return (time.perf_counter_ns() - started_ns) / 1.0e9


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _array_sha256(array: np.ndarray) -> str:
    descriptor = _canonical_json_bytes(
        {"dtype": array.dtype.str, "shape": [int(size) for size in array.shape]}
    )
    return hashlib.sha256(descriptor + b"\0" + array.tobytes(order="C")).hexdigest()


def _freeze_arrays(arrays: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    frozen: dict[str, np.ndarray] = {}
    for name, value in sorted(arrays.items()):
        array = np.array(value, copy=True, order="C")
        array.setflags(write=False)
        frozen[name] = array
    return MappingProxyType(frozen)


def verify_p05_actual_intervention_result(
    result: P05ActualInterventionResult,
) -> None:
    """Fail closed unless a runner result is intact and protocol-complete."""

    if not isinstance(result, P05ActualInterventionResult):
        raise TypeError("actual intervention input must be P05ActualInterventionResult")
    arrays = result.arrays
    if not isinstance(arrays, Mapping) or set(arrays) != _ORIGINAL_ARRAYS | _ACTUAL_ARRAYS:
        raise ValueError("actual intervention array schema is incomplete or unexpected")
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray) or array.flags.writeable:
            raise ValueError(f"actual intervention array {name!r} must be read-only NumPy")
        if array.dtype.hasobject:
            raise ValueError(f"actual intervention array {name!r} cannot contain objects")
        if name in _IDENTIFIER_ARRAYS:
            if array.dtype.kind != "U":
                raise ValueError(f"actual intervention identifier {name!r} must be Unicode")
        elif array.dtype.kind in {"b", "i", "u", "f"}:
            if not np.isfinite(array).all():
                raise FloatingPointError(f"actual intervention array {name!r} is non-finite")
        else:
            raise ValueError(f"actual intervention array {name!r} has unsupported dtype")

    sample_ids = arrays["sample_id"]
    if sample_ids.ndim != 1 or len(sample_ids) == 0:
        raise ValueError("actual intervention sample_id must be a non-empty vector")
    sample_count = len(sample_ids)
    if sample_ids.tolist() != sorted(sample_ids.tolist()):
        raise ValueError("actual intervention sample_id values must be stably sorted")
    if len(set(sample_ids.tolist())) != sample_count:
        raise ValueError("actual intervention sample_id values must be unique")
    for name in ("record_id", "group_id", "window_start", "window_end", "y"):
        if arrays[name].shape != (sample_count,):
            raise ValueError(f"actual intervention {name!r} batch shape differs")
    logits = arrays["logits"]
    if logits.ndim != 2 or logits.shape[0] != sample_count or logits.shape[1] < 2:
        raise ValueError("actual intervention logits shape is invalid")
    classes = int(logits.shape[1])
    expected_shapes = {
        "actual_deletion_logits": (sample_count, RULE_COUNT, classes),
        "actual_deletion_normalized_rule_firing": (
            sample_count,
            RULE_COUNT,
            RULE_COUNT,
        ),
        "actual_deletion_rule_contributions": (
            sample_count,
            RULE_COUNT,
            RULE_COUNT,
            classes,
        ),
        "actual_deletion_invariant_max_abs": (sample_count, RULE_COUNT),
        "actual_deletion_membership_invariant_pass": (sample_count, RULE_COUNT),
        "actual_deletion_antecedent_invariant_pass": (sample_count, RULE_COUNT),
        "actual_deletion_firing_invariant_pass": (sample_count, RULE_COUNT),
        "actual_shuffle_seed": (sample_count,),
        "actual_shuffle_permutations": (sample_count, SHUFFLE_COUNT, RULE_COUNT),
        "actual_shuffle_logits": (sample_count, SHUFFLE_COUNT, classes),
        "actual_shuffle_rule_contributions": (
            sample_count,
            SHUFFLE_COUNT,
            RULE_COUNT,
            classes,
        ),
        "actual_shuffle_invariant_max_abs": (sample_count, SHUFFLE_COUNT),
        "actual_shuffle_membership_invariant_pass": (
            sample_count,
            SHUFFLE_COUNT,
        ),
        "actual_shuffle_antecedent_invariant_pass": (
            sample_count,
            SHUFFLE_COUNT,
        ),
        "actual_shuffle_firing_invariant_pass": (sample_count, SHUFFLE_COUNT),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"actual intervention {name!r} must have shape {shape}")
    for name in (
        "actual_deletion_membership_invariant_pass",
        "actual_deletion_antecedent_invariant_pass",
        "actual_deletion_firing_invariant_pass",
        "actual_shuffle_membership_invariant_pass",
        "actual_shuffle_antecedent_invariant_pass",
        "actual_shuffle_firing_invariant_pass",
    ):
        if arrays[name].dtype != np.dtype(np.bool_) or not arrays[name].all():
            raise ValueError(f"actual forward invariant checks did not all pass: {name}")
    for name in (
        "actual_deletion_invariant_max_abs",
        "actual_shuffle_invariant_max_abs",
    ):
        if np.any(arrays[name] < 0.0):
            raise ValueError(f"actual invariant residual {name!r} cannot be negative")

    metadata = result.metadata
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "arrays",
        "conclusion_control",
        "model_state",
        "protocol",
        "provenance",
        "selection",
    }:
        raise ValueError("actual intervention metadata schema is incomplete or unexpected")
    descriptors = metadata["arrays"]
    expected_descriptors = {
        name: {
            "dtype": array.dtype.str,
            "shape": [int(size) for size in array.shape],
            "sha256": _array_sha256(array),
        }
        for name, array in arrays.items()
    }
    if descriptors != expected_descriptors:
        raise ValueError("actual intervention array descriptors or hashes differ")
    if metadata["conclusion_control"] != {
        "claim_decision": "not_performed",
        "performance_claim": False,
        "scope": "actual_forward_outputs_and_invariant_checks_only",
    }:
        raise ValueError("actual intervention conclusion control is invalid")
    provenance = metadata["provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "checkpoint_sha256",
        "config_sha256",
        "dataset",
        "model_seed",
        "model_sha256",
        "split",
    }:
        raise ValueError("actual intervention provenance schema is invalid")
    for name in ("checkpoint_sha256", "config_sha256", "model_sha256"):
        _required_sha256(provenance[name], name=f"actual provenance {name}")
    if provenance["dataset"] not in {"CWRU", "XJTU"} or provenance["split"] not in {
        "validation",
        "test",
    }:
        raise ValueError("actual intervention dataset or split is invalid")
    if type(provenance["model_seed"]) is not int or not 0 <= provenance["model_seed"] < 2**63:
        raise ValueError("actual intervention model seed is invalid")
    model_state = metadata["model_state"]
    if not isinstance(model_state, Mapping) or set(model_state) != {
        "after_sha256",
        "before_sha256",
        "unchanged",
    }:
        raise ValueError("actual intervention model-state block is invalid")
    if (
        model_state["unchanged"] is not True
        or model_state["before_sha256"] != model_state["after_sha256"]
        or model_state["before_sha256"] != provenance["model_sha256"]
    ):
        raise ValueError("actual intervention model state is not provenance-bound")
    protocol = metadata["protocol"]
    if protocol != {
        "actual_forward_calls": 1 + RULE_COUNT + SHUFFLE_COUNT,
        "deletion_count": RULE_COUNT,
        "reconstruction_atol": RECONSTRUCTION_ATOL,
        "reconstruction_rtol": RECONSTRUCTION_RTOL,
        "rule_count": RULE_COUNT,
        "shuffle_count_per_sample": SHUFFLE_COUNT,
        "shuffle_execution": (
            "one_batched_forward_per_shuffle_index_with_per_sample_permutations"
        ),
    }:
        raise ValueError("actual intervention protocol metadata is not frozen")
    selection = metadata["selection"]
    if not isinstance(selection, Mapping) or set(selection) != {
        "benchmark_first_n",
        "input_count",
        "kind",
        "selected_count",
    }:
        raise ValueError("actual intervention selection metadata is invalid")
    if selection["selected_count"] != sample_count:
        raise ValueError("actual intervention selected_count differs from arrays")
    expected_semantic = hashlib.sha256(
        _canonical_json_bytes(dict(metadata))
    ).hexdigest()
    if result.semantic_sha256 != expected_semantic:
        raise ValueError("actual intervention semantic SHA-256 differs from metadata")
    timing = result.timing
    if not isinstance(timing, Mapping) or set(timing) != {
        "deletion_seconds",
        "device_type",
        "original_seconds",
        "performance_claim_allowed",
        "scope",
        "shuffle_seconds",
        "total_seconds",
    }:
        raise ValueError("actual intervention timing schema is invalid")
    if timing["performance_claim_allowed"] is not False or timing["scope"] != (
        "diagnostic_wall_clock_boundary_only"
    ):
        raise ValueError("actual intervention timing may not support a performance claim")
    seconds = [
        timing["original_seconds"],
        timing["deletion_seconds"],
        timing["shuffle_seconds"],
        timing["total_seconds"],
    ]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in seconds
    ):
        raise ValueError("actual intervention timing contains an invalid duration")


def run_p05_same_checkpoint_interventions(
    *,
    network: torch.nn.Module,
    batch: Mapping[str, Any],
    provenance: P05InterventionProvenance,
    expected_window_size: int = 4096,
    require_cuda: bool = True,
    benchmark_first_n: int | None = None,
) -> P05ActualInterventionResult:
    """Execute original, exhaustive deletion, and registered shuffle forwards.

    ``benchmark_first_n=256`` selects the first 256 unique sample IDs after a
    label-free stable sort. Timing is diagnostic only and is deliberately
    excluded from the deterministic semantic hash and all claim decisions.
    """

    if not isinstance(network, torch.nn.Module):
        raise TypeError("network must be a torch.nn.Module")
    if type(expected_window_size) is not int or expected_window_size <= 0:
        raise ValueError("expected_window_size must be a positive integer")
    if type(require_cuda) is not bool:
        raise TypeError("require_cuda must be a boolean")
    provenance_payload = _validate_provenance(provenance)
    prepared = _prepare_batch(
        batch,
        expected_window_size=expected_window_size,
        benchmark_first_n=benchmark_first_n,
    )
    device = _network_device(network)
    if require_cuda and device.type != "cuda":
        raise RuntimeError("P05 evidence intervention runner requires a CUDA model")
    state_before = model_state_sha256(network)
    if state_before != provenance_payload["model_sha256"]:
        raise ValueError(
            "P05 intervention model state does not match provenance model_sha256"
        )

    x = prepared.x.to(device=device, dtype=torch.float32, non_blocking=False)
    sample_ids = prepared.sample_id
    sample_count = len(sample_ids)
    was_training = bool(network.training)
    network.eval()
    try:
        with torch.no_grad():
            original_started = _timed_start(device)
            original = network.forward_with_fuzzy_trace(x)
            original_seconds = _timed_end(device, original_started)
            rules, classes = _validate_output(
                original,
                identity="original",
                sample_ids=sample_ids,
            )
            if rules != RULE_COUNT:
                raise ValueError(
                    f"P05 C2 requires exactly {RULE_COUNT} rules, got {rules}"
                )
            expected_classes = 2 if provenance.dataset == "XJTU" else 4
            if classes != expected_classes:
                raise ValueError(
                    f"P05 {provenance.dataset} requires {expected_classes} classes, "
                    f"got {classes}"
                )
            if np.any(prepared.y < 0) or np.any(prepared.y >= classes):
                raise ValueError("batch y contains an out-of-range class index")
            identity_permutation = torch.arange(rules, device=device)
            if not bool(original.fuzzy_trace.rule_mask.all()) or not torch.equal(
                original.fuzzy_trace.consequent_permutation,
                identity_permutation,
            ):
                raise ValueError(
                    "original trace must retain all rules and preserve consequent order"
                )
            expected_original_normalized = torch.softmax(
                original.fuzzy_trace.log_rule_firing,
                dim=1,
            )
            _assert_per_sample_close(
                original.fuzzy_trace.normalized_rule_firing,
                expected_original_normalized,
                sample_ids=sample_ids,
                identity="original normalized firing",
            )
            _assert_per_sample_close(
                original.fuzzy_trace.rule_contributions,
                expected_original_normalized.unsqueeze(-1)
                * original.fuzzy_trace.rule_consequents.unsqueeze(0),
                sample_ids=sample_ids,
                identity="original rule contributions",
            )

            deletion_logits = np.empty(
                (sample_count, rules, classes), dtype=np.dtype("<f8")
            )
            deletion_contributions = np.empty(
                (sample_count, rules, rules, classes), dtype=np.dtype("<f8")
            )
            deletion_normalized = np.empty(
                (sample_count, rules, rules), dtype=np.dtype("<f8")
            )
            deletion_invariant_max_abs = np.empty(
                (sample_count, rules), dtype=np.dtype("<f8")
            )
            deletion_started = _timed_start(device)
            for deleted_rule in range(rules):
                mask = torch.ones(rules, dtype=torch.bool, device=device)
                mask[deleted_rule] = False
                deleted = network.forward_with_fuzzy_trace(x, rule_mask=mask)
                _validate_output(
                    deleted,
                    identity=f"deletion[{deleted_rule}]",
                    sample_ids=sample_ids,
                    expected_classes=classes,
                )
                deletion_invariant_max_abs[:, deleted_rule] = _invariant_residual(
                    deleted,
                    original,
                    sample_ids=sample_ids,
                    identity=f"deletion[{deleted_rule}]",
                    include_normalized_firing=False,
                )
                expected_mask = mask.unsqueeze(0).expand(sample_count, -1)
                if not torch.equal(deleted.fuzzy_trace.rule_mask, expected_mask):
                    raise ValueError(f"deletion[{deleted_rule}] rule mask was not applied")
                if not torch.equal(
                    deleted.fuzzy_trace.consequent_permutation,
                    identity_permutation,
                ):
                    raise ValueError(
                        f"deletion[{deleted_rule}] changed consequent ordering"
                    )
                _assert_shared_close(
                    deleted.fuzzy_trace.rule_consequents,
                    original.fuzzy_trace.rule_consequents,
                    identity=f"deletion[{deleted_rule}] rule consequents",
                )
                masked_log_firing = original.fuzzy_trace.log_rule_firing.masked_fill(
                    ~expected_mask,
                    -torch.inf,
                )
                expected_normalized = torch.softmax(masked_log_firing, dim=1)
                _assert_per_sample_close(
                    deleted.fuzzy_trace.normalized_rule_firing,
                    expected_normalized,
                    sample_ids=sample_ids,
                    identity=f"deletion[{deleted_rule}] normalized firing",
                )
                _assert_per_sample_close(
                    deleted.fuzzy_trace.rule_contributions,
                    expected_normalized.unsqueeze(-1)
                    * original.fuzzy_trace.rule_consequents.unsqueeze(0),
                    sample_ids=sample_ids,
                    identity=f"deletion[{deleted_rule}] rule contributions",
                )
                deletion_logits[:, deleted_rule] = _to_numpy(
                    deleted.logits, dtype=np.dtype("<f8")
                )
                deletion_contributions[:, deleted_rule] = _to_numpy(
                    deleted.fuzzy_trace.rule_contributions,
                    dtype=np.dtype("<f8"),
                )
                deletion_normalized[:, deleted_rule] = _to_numpy(
                    deleted.fuzzy_trace.normalized_rule_firing,
                    dtype=np.dtype("<f8"),
                )
            deletion_seconds = _timed_end(device, deletion_started)

            shuffle_permutations = np.empty(
                (sample_count, SHUFFLE_COUNT, rules), dtype=np.dtype("<i8")
            )
            shuffle_seeds = np.empty(sample_count, dtype=np.dtype("<u8"))
            shuffle_logits = np.empty(
                (sample_count, SHUFFLE_COUNT, classes), dtype=np.dtype("<f8")
            )
            shuffle_contributions = np.empty(
                (sample_count, SHUFFLE_COUNT, rules, classes), dtype=np.dtype("<f8")
            )
            shuffle_invariant_max_abs = np.empty(
                (sample_count, SHUFFLE_COUNT), dtype=np.dtype("<f8")
            )
            shuffle_started = _timed_start(device)
            for sample_index, sample_id in enumerate(sample_ids.tolist()):
                registered_shuffle_seed = shuffle_seed(
                    dataset=provenance.dataset,
                    split=provenance.split,
                    model_seed=provenance.model_seed,
                    sample_id=sample_id,
                )
                shuffle_seeds[sample_index] = registered_shuffle_seed
                permutations = generate_unique_nonidentity_permutations(
                    rules,
                    seed=registered_shuffle_seed,
                    count=SHUFFLE_COUNT,
                )
                shuffle_permutations[sample_index] = permutations

            for shuffle_index in range(SHUFFLE_COUNT):
                permutation = torch.as_tensor(
                    shuffle_permutations[:, shuffle_index, :],
                    dtype=torch.long,
                    device=device,
                )
                shuffled = network.forward_with_fuzzy_trace(
                    x,
                    consequent_permutation=permutation,
                )
                identity_name = f"shuffle_batch[{shuffle_index}]"
                _validate_output(
                    shuffled,
                    identity=identity_name,
                    sample_ids=sample_ids,
                    expected_classes=classes,
                    allow_batched_consequents=True,
                )
                shuffle_invariant_max_abs[:, shuffle_index] = _invariant_residual(
                    shuffled,
                    original,
                    sample_ids=sample_ids,
                    identity=identity_name,
                    include_normalized_firing=True,
                )
                if not bool(shuffled.fuzzy_trace.rule_mask.all()):
                    raise ValueError(f"{identity_name} changed the rule mask")
                if not torch.equal(
                    shuffled.fuzzy_trace.consequent_permutation,
                    permutation,
                ):
                    raise ValueError(
                        f"{identity_name} did not apply the registered permutations"
                    )
                expected_consequents = original.fuzzy_trace.rule_consequents[
                    permutation
                ]
                _assert_per_sample_close(
                    shuffled.fuzzy_trace.rule_consequents,
                    expected_consequents,
                    sample_ids=sample_ids,
                    identity=f"{identity_name} rule consequents",
                )
                expected_contributions = (
                    original.fuzzy_trace.normalized_rule_firing.unsqueeze(-1)
                    * expected_consequents
                )
                _assert_per_sample_close(
                    shuffled.fuzzy_trace.rule_contributions,
                    expected_contributions,
                    sample_ids=sample_ids,
                    identity=f"{identity_name} rule contributions",
                )
                shuffle_logits[:, shuffle_index] = _to_numpy(
                    shuffled.logits,
                    dtype=np.dtype("<f8"),
                )
                shuffle_contributions[:, shuffle_index] = _to_numpy(
                    shuffled.fuzzy_trace.rule_contributions,
                    dtype=np.dtype("<f8"),
                )
            shuffle_seconds = _timed_end(device, shuffle_started)
    finally:
        network.train(was_training)

    state_after = model_state_sha256(network)
    if state_after != state_before:
        raise RuntimeError("P05 actual intervention forwards mutated the checkpoint state")

    arrays = _original_arrays(original, prepared)
    arrays.update(
        {
            "actual_deletion_logits": deletion_logits,
            "actual_deletion_normalized_rule_firing": deletion_normalized,
            "actual_deletion_rule_contributions": deletion_contributions,
            "actual_deletion_invariant_max_abs": deletion_invariant_max_abs,
            "actual_deletion_membership_invariant_pass": np.ones(
                (sample_count, rules), dtype=np.bool_
            ),
            "actual_deletion_antecedent_invariant_pass": np.ones(
                (sample_count, rules), dtype=np.bool_
            ),
            "actual_deletion_firing_invariant_pass": np.ones(
                (sample_count, rules), dtype=np.bool_
            ),
            "actual_shuffle_permutations": shuffle_permutations,
            "actual_shuffle_seed": shuffle_seeds,
            "actual_shuffle_logits": shuffle_logits,
            "actual_shuffle_rule_contributions": shuffle_contributions,
            "actual_shuffle_invariant_max_abs": shuffle_invariant_max_abs,
            "actual_shuffle_membership_invariant_pass": np.ones(
                (sample_count, SHUFFLE_COUNT), dtype=np.bool_
            ),
            "actual_shuffle_antecedent_invariant_pass": np.ones(
                (sample_count, SHUFFLE_COUNT), dtype=np.bool_
            ),
            "actual_shuffle_firing_invariant_pass": np.ones(
                (sample_count, SHUFFLE_COUNT), dtype=np.bool_
            ),
        }
    )
    frozen_arrays = _freeze_arrays(arrays)
    selection = {
        "benchmark_first_n": benchmark_first_n,
        "input_count": prepared.input_count,
        "kind": (
            "first_n_after_stable_sample_id_sort"
            if benchmark_first_n is not None
            else "all_after_stable_sample_id_sort"
        ),
        "selected_count": sample_count,
    }
    descriptors = {
        name: {
            "dtype": array.dtype.str,
            "shape": [int(size) for size in array.shape],
            "sha256": _array_sha256(array),
        }
        for name, array in frozen_arrays.items()
    }
    metadata = {
        "arrays": descriptors,
        "conclusion_control": {
            "claim_decision": "not_performed",
            "performance_claim": False,
            "scope": "actual_forward_outputs_and_invariant_checks_only",
        },
        "model_state": {
            "after_sha256": state_after,
            "before_sha256": state_before,
            "unchanged": True,
        },
        "protocol": {
            "actual_forward_calls": 1 + RULE_COUNT + SHUFFLE_COUNT,
            "deletion_count": RULE_COUNT,
            "reconstruction_atol": RECONSTRUCTION_ATOL,
            "reconstruction_rtol": RECONSTRUCTION_RTOL,
            "rule_count": RULE_COUNT,
            "shuffle_count_per_sample": SHUFFLE_COUNT,
            "shuffle_execution": (
                "one_batched_forward_per_shuffle_index_with_per_sample_permutations"
            ),
        },
        "provenance": provenance_payload,
        "selection": selection,
    }
    semantic_sha256 = hashlib.sha256(_canonical_json_bytes(metadata)).hexdigest()
    timing = MappingProxyType(
        {
            "deletion_seconds": deletion_seconds,
            "device_type": device.type,
            "original_seconds": original_seconds,
            "performance_claim_allowed": False,
            "scope": "diagnostic_wall_clock_boundary_only",
            "shuffle_seconds": shuffle_seconds,
            "total_seconds": original_seconds + deletion_seconds + shuffle_seconds,
        }
    )
    return P05ActualInterventionResult(
        arrays=frozen_arrays,
        metadata=MappingProxyType(metadata),
        timing=timing,
        semantic_sha256=semantic_sha256,
    )


def _normalized_expected_sample_ids(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("expected_sample_ids must be a string sequence")
    identifiers = list(value)
    if len(identifiers) < PILOT_BENCHMARK_SAMPLE_COUNT:
        raise ValueError("pilot validation partition must contain at least 256 windows")
    if any(
        not isinstance(item, str) or not item.strip() or "\x00" in item
        for item in identifiers
    ):
        raise ValueError("expected_sample_ids contains an invalid stable ID")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("expected_sample_ids must be unique")
    return tuple(sorted(identifiers))


def _bind_pilot_partition_selection(
    result: P05ActualInterventionResult,
    *,
    input_count: int,
) -> P05ActualInterventionResult:
    metadata = dict(result.metadata)
    metadata["selection"] = {
        "benchmark_first_n": PILOT_BENCHMARK_SAMPLE_COUNT,
        "input_count": input_count,
        "kind": "first_n_after_stable_sample_id_sort",
        "selected_count": PILOT_BENCHMARK_SAMPLE_COUNT,
    }
    semantic_sha256 = hashlib.sha256(_canonical_json_bytes(metadata)).hexdigest()
    rebound = P05ActualInterventionResult(
        arrays=result.arrays,
        metadata=MappingProxyType(metadata),
        timing=result.timing,
        semantic_sha256=semantic_sha256,
    )
    verify_p05_actual_intervention_result(rebound)
    return rebound


def run_p05_pilot_interventions_from_loader(
    *,
    network: torch.nn.Module,
    batches: Iterable[Mapping[str, Any]],
    provenance: P05InterventionProvenance,
    expected_sample_ids: Sequence[str],
    expected_window_size: int = 4096,
    require_cuda: bool = True,
) -> P05ActualInterventionResult:
    """Scan a complete validation loader, then benchmark its first 256 IDs.

    Full stable-ID coverage is checked before any model forward.  Only the
    registered first 256 standardized windows are retained in host memory and
    they are evaluated together, preserving the frozen 43-forward pilot
    boundary while recording the complete validation-partition count.
    """

    if isinstance(batches, (str, bytes)) or not isinstance(batches, Iterable):
        raise TypeError("batches must be an iterable of batch mappings")
    if not isinstance(provenance, P05InterventionProvenance):
        raise TypeError("provenance must be P05InterventionProvenance")
    if provenance.split != "validation" or provenance.model_seed != PILOT_MODEL_SEED:
        raise ValueError(
            "pilot loader benchmark requires validation and model seed 20260801"
        )
    expected = _normalized_expected_sample_ids(expected_sample_ids)
    if type(require_cuda) is not bool:
        raise TypeError("require_cuda must be a boolean")
    if require_cuda and len(expected) != REGISTERED_VALIDATION_WINDOW_COUNTS.get(
        provenance.dataset
    ):
        raise ValueError(
            "pilot expected_sample_ids count differs from the frozen validation partition"
        )

    required_fields = {
        "x",
        "y",
        "sample_id",
        "record_id",
        "group_id",
        "window_start",
        "window_end",
    }
    expected_set = set(expected)
    selected_ids = expected[:PILOT_BENCHMARK_SAMPLE_COUNT]
    selected_set = set(selected_ids)
    observed: set[str] = set()
    rows: dict[str, tuple[Any, ...]] = {}
    for batch_index, batch in enumerate(batches):
        if not isinstance(batch, Mapping):
            raise TypeError(f"pilot loader batch[{batch_index}] must be a mapping")
        missing = sorted(required_fields - set(batch))
        if missing:
            raise ValueError(f"pilot loader batch[{batch_index}] is missing {missing}")
        prepared = _prepare_batch(
            {name: batch[name] for name in required_fields},
            expected_window_size=expected_window_size,
            benchmark_first_n=None,
        )
        for index, sample_id in enumerate(prepared.sample_id.tolist()):
            stable_id = (
                f"{prepared.record_id[index]}:{int(prepared.window_start[index])}:"
                f"{int(prepared.window_end[index])}"
            )
            if sample_id != stable_id:
                raise ValueError(
                    f"pilot loader sample_id {sample_id!r} must equal {stable_id!r}"
                )
            if sample_id in observed:
                raise ValueError(f"pilot loader emitted duplicate sample_id {sample_id!r}")
            if sample_id not in expected_set:
                raise ValueError(f"pilot loader emitted unexpected sample_id {sample_id!r}")
            observed.add(sample_id)
            if sample_id in selected_set:
                rows[sample_id] = (
                    prepared.x[index].detach().to(device="cpu").clone(),
                    int(prepared.y[index]),
                    str(prepared.record_id[index]),
                    str(prepared.group_id[index]),
                    int(prepared.window_start[index]),
                    int(prepared.window_end[index]),
                )
    if observed != expected_set:
        missing = sorted(expected_set - observed)
        raise ValueError(
            "pilot loader coverage differs from expected_sample_ids: "
            f"missing={missing[:5]}"
        )
    if set(rows) != selected_set:
        raise RuntimeError("pilot loader did not retain the frozen first 256 windows")

    ordered_rows = [rows[sample_id] for sample_id in selected_ids]
    selected_batch = {
        "x": torch.stack([row[0] for row in ordered_rows], dim=0).contiguous(),
        "y": np.asarray([row[1] for row in ordered_rows], dtype="<i8"),
        "sample_id": list(selected_ids),
        "record_id": [row[2] for row in ordered_rows],
        "group_id": [row[3] for row in ordered_rows],
        "window_start": np.asarray([row[4] for row in ordered_rows], dtype="<i8"),
        "window_end": np.asarray([row[5] for row in ordered_rows], dtype="<i8"),
    }
    result = run_p05_same_checkpoint_interventions(
        network=network,
        batch=selected_batch,
        provenance=provenance,
        expected_window_size=expected_window_size,
        require_cuda=require_cuda,
        benchmark_first_n=None,
    )
    if tuple(result.arrays["sample_id"].tolist()) != selected_ids:
        raise RuntimeError("pilot evaluator selected IDs differ from the frozen prefix")
    return _bind_pilot_partition_selection(result, input_count=len(expected))


__all__ = [
    "PILOT_BENCHMARK_SAMPLE_COUNT",
    "P05ActualInterventionResult",
    "P05InterventionProvenance",
    "run_p05_pilot_interventions_from_loader",
    "run_p05_same_checkpoint_interventions",
    "verify_p05_actual_intervention_result",
]
