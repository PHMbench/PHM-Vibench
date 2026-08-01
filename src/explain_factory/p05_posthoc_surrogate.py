"""Frozen CPU trainer and artifact contract for the P05-B2 surrogate."""

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
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data_factory.p05_weighting import WeightPlan
from src.model_factory.X_model.UXFD.fuzzy import FuzzyConfig, FuzzyReasoner


SCHEMA_NAME = "p05.b2_posthoc_fuzzy_surrogate"
SCHEMA_VERSION = 1
CHECKPOINT_NAME = "best_model.npz"
MANIFEST_NAME = "manifest.json"

REGISTERED_SEEDS = frozenset({42, 123, 456, 789, 1024})
INPUT_FEATURES = 8
MEMBERSHIP_FUNCTIONS = 3
RULES = 10
LOGIT_SCALE = 1.0
ANTECEDENT_TEMPERATURE = 1.0
MIN_WIDTH = 1.0e-4
FIRING_EPSILON = 1.0e-12
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 15
MIN_DELTA = 0.0
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_EXPECTED_CLASSES = {1: 4, 2: 2}
_EXPECTED_PARAMETER_COUNT = {2: 324, 4: 344}
_EXPECTED_WINDOWS_PER_RECORD = {1: 16, 2: 4}
_EXPECTED_FORMULAS = {
    (1, "train"): "1/(4*n_recordings_in_class*16)",
    (2, "train"): "1/(10*n_records_in_bearing_class_cell*4)",
    (1, "validation"): "1/(n_groups*n_windows_in_group)",
    (2, "validation"): "1/(n_groups*n_windows_in_group)",
}


@dataclass(frozen=True)
class P05B2FrozenSplit:
    """Frozen B0 inputs for one split; labels are deliberately absent."""

    sample_ids: Sequence[str]
    record_ids: Sequence[int]
    group_ids: Sequence[str]
    features: torch.Tensor
    b0_logits: torch.Tensor
    weight_plan: WeightPlan


@dataclass(frozen=True)
class P05B2TrainingResult:
    """Terminal state and selected-checkpoint provenance for one B2 fit."""

    package_dir: Path
    checkpoint_path: Path
    manifest_path: Path
    best_epoch: int
    best_validation_mse: float
    epochs_ran: int
    stopped_early: bool
    semantic_sha256: str
    checkpoint_sha256: str
    manifest_sha256: str
    status: str


@dataclass(frozen=True)
class _PreparedSplit:
    role: str
    dataset_id: int
    sample_ids: tuple[str, ...]
    record_ids: tuple[int, ...]
    group_ids: tuple[str, ...]
    features: torch.Tensor
    logits: torch.Tensor
    weights: torch.Tensor
    provenance: Mapping[str, Any]


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
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    metadata = _canonical_json_bytes(
        {
            "dtype": contiguous.dtype.str,
            "shape": [int(size) for size in contiguous.shape],
        }
    )
    return _sha256_bytes(metadata + b"\0" + contiguous.tobytes(order="C"))


def _array_descriptor(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": contiguous.dtype.str,
        "shape": [int(size) for size in contiguous.shape],
        "sha256": _array_sha256(contiguous),
    }


def _tensor_descriptor(tensor: torch.Tensor) -> dict[str, Any]:
    return _array_descriptor(tensor.detach().cpu().contiguous().numpy())


def _record_id(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must contain integer record IDs")
    return int(value)


def _weight_plan_contract(
    plan: WeightPlan,
    *,
    expected_role: str,
) -> tuple[dict[str, Any], dict[int, float]]:
    if not isinstance(plan, WeightPlan):
        raise TypeError("split weight_plan must be a WeightPlan")
    if type(plan.dataset_id) is not int or plan.dataset_id not in _EXPECTED_CLASSES:
        raise ValueError("weight_plan dataset_id must be registered P05 dataset 1 or 2")
    if plan.role != expected_role:
        raise ValueError(
            f"weight_plan role must be {expected_role!r}, got {plan.role!r}"
        )
    expected_windows = _EXPECTED_WINDOWS_PER_RECORD[plan.dataset_id]
    if type(plan.windows_per_record) is not int or plan.windows_per_record != expected_windows:
        raise ValueError(
            f"P05 dataset {plan.dataset_id} requires {expected_windows} windows per record"
        )
    expected_formula = _EXPECTED_FORMULAS[(plan.dataset_id, expected_role)]
    if plan.formula != expected_formula:
        raise ValueError(
            f"weight_plan formula for {expected_role} must be {expected_formula!r}"
        )
    if not isinstance(plan.record_weights, Mapping) or not plan.record_weights:
        raise TypeError("weight_plan record_weights must be a non-empty mapping")

    weights: dict[int, float] = {}
    for raw_record_id, raw_weight in plan.record_weights.items():
        record_id = _record_id(raw_record_id, name="weight_plan.record_weights")
        if record_id in weights:
            raise ValueError("weight_plan contains duplicate canonical record IDs")
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, Real):
            raise TypeError("weight_plan weights must be real numbers")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("weight_plan weights must be finite and positive")
        weights[record_id] = weight
    if not math.isclose(
        math.fsum(weights.values()) / len(weights),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("weight_plan record weights must have mean one")

    rows = [
        {"Id": record_id, "window_weight": weights[record_id]}
        for record_id in sorted(weights)
    ]
    contract = {
        "schema_version": 1,
        "paper_id": "P05",
        "dataset_id": plan.dataset_id,
        "role": expected_role,
        "windows_per_record": plan.windows_per_record,
        "formula": plan.formula,
        "normalization": "mean_train_or_evaluation_window_weight_equals_one",
        "record_weights": rows,
    }
    expected_hash = _sha256_bytes(_canonical_json_bytes(contract))
    observed_hash = _required_sha256(plan.sha256, name="weight_plan.sha256")
    if observed_hash != expected_hash:
        raise ValueError("weight_plan source SHA-256 does not match its contract")
    return {**contract, "sha256": expected_hash}, weights


def _string_vector(
    values: Sequence[str],
    *,
    name: str,
    count: int,
    unique: bool,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings")
    normalized = tuple(values)
    if len(normalized) != count:
        raise ValueError(f"{name} length must equal the split sample count")
    if any(not isinstance(value, str) or not value or "\x00" in value for value in normalized):
        raise ValueError(f"{name} must contain non-empty strings without NUL bytes")
    if unique and len(set(normalized)) != count:
        raise ValueError(f"{name} must be unique")
    return normalized


def _frozen_tensor(
    value: Any,
    *,
    name: str,
    columns: int | None,
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must remain on CPU for P05-B2")
    if value.dtype != torch.float32:
        raise ValueError(f"{name} must use frozen float32 precision")
    if value.requires_grad:
        raise ValueError(f"{name} must be a detached frozen B0 tensor")
    if value.ndim != 2 or value.shape[0] <= 0:
        raise ValueError(f"{name} must have shape (windows, columns)")
    if columns is not None and int(value.shape[1]) != columns:
        raise ValueError(f"{name} must have exactly {columns} columns")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains non-finite values")
    return value.detach().contiguous().clone()


def _prepare_split(split: P05B2FrozenSplit, *, role: str) -> _PreparedSplit:
    if not isinstance(split, P05B2FrozenSplit):
        raise TypeError(f"{role} must be a P05B2FrozenSplit")
    expected_role = "train" if role == "train" else "validation"
    plan_contract, record_weights = _weight_plan_contract(
        split.weight_plan,
        expected_role=expected_role,
    )
    features = _frozen_tensor(
        split.features,
        name=f"{role}.features",
        columns=INPUT_FEATURES,
    )
    logits = _frozen_tensor(
        split.b0_logits,
        name=f"{role}.b0_logits",
        columns=None,
    )
    sample_count = int(features.shape[0])
    if logits.shape[0] != sample_count:
        raise ValueError(f"{role} features and B0 logits must have the same windows")
    expected_classes = _EXPECTED_CLASSES[split.weight_plan.dataset_id]
    if int(logits.shape[1]) != expected_classes:
        raise ValueError(
            f"P05 dataset {split.weight_plan.dataset_id} requires K={expected_classes} logits"
        )
    sample_ids = _string_vector(
        split.sample_ids,
        name=f"{role}.sample_ids",
        count=sample_count,
        unique=True,
    )
    group_ids = _string_vector(
        split.group_ids,
        name=f"{role}.group_ids",
        count=sample_count,
        unique=False,
    )
    if isinstance(split.record_ids, (str, bytes)):
        raise TypeError(f"{role}.record_ids must be an integer sequence")
    record_ids = tuple(
        _record_id(value, name=f"{role}.record_ids") for value in split.record_ids
    )
    if len(record_ids) != sample_count:
        raise ValueError(f"{role}.record_ids length must equal the split sample count")

    record_counts = Counter(record_ids)
    if set(record_counts) != set(record_weights):
        raise ValueError(f"{role} record IDs do not exactly cover the registered weight plan")
    windows_per_record = split.weight_plan.windows_per_record
    if any(count != windows_per_record for count in record_counts.values()):
        raise ValueError(
            f"{role} must contain exactly {windows_per_record} windows for every record"
        )
    record_groups: dict[int, str] = {}
    for record_id, group_id in zip(record_ids, group_ids):
        previous = record_groups.setdefault(record_id, group_id)
        if previous != group_id:
            raise ValueError(f"{role} record {record_id} maps to multiple groups")

    weight_values = [record_weights[record_id] for record_id in record_ids]
    if not math.isclose(
        math.fsum(weight_values) / len(weight_values),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"{role} per-window registered weights must have mean one")
    if role == "validation":
        group_counts = Counter(group_ids)
        group_count = len(group_counts)
        for index, (group_id, observed) in enumerate(zip(group_ids, weight_values)):
            expected = sample_count / (group_count * group_counts[group_id])
            if not math.isclose(observed, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
                raise ValueError(
                    "validation weights are not equal-group/equal-window at "
                    f"sample {sample_ids[index]!r}"
                )
    weights = torch.tensor(weight_values, dtype=torch.float32, device="cpu")
    provenance = {
        "sample_count": sample_count,
        "sample_ids_sha256": _sha256_bytes(_canonical_json_bytes(list(sample_ids))),
        "record_ids_sha256": _sha256_bytes(_canonical_json_bytes(list(record_ids))),
        "group_ids_sha256": _sha256_bytes(_canonical_json_bytes(list(group_ids))),
        "features": _tensor_descriptor(features),
        "b0_logits": _tensor_descriptor(logits),
        "sample_weights": _tensor_descriptor(weights),
        "weight_plan": plan_contract,
    }
    return _PreparedSplit(
        role=role,
        dataset_id=split.weight_plan.dataset_id,
        sample_ids=sample_ids,
        record_ids=record_ids,
        group_ids=group_ids,
        features=features,
        logits=logits,
        weights=weights,
        provenance=provenance,
    )


def p05_b2_weighted_logit_mse(
    prediction: torch.Tensor,
    b0_logits: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    """Return the weighted mean of per-window mean-K squared logit error."""

    if not all(torch.is_tensor(value) for value in (prediction, b0_logits, sample_weights)):
        raise TypeError("prediction, B0 logits, and sample weights must be tensors")
    if prediction.device.type != "cpu" or b0_logits.device.type != "cpu":
        raise ValueError("P05-B2 logit MSE is CPU-only")
    if sample_weights.device.type != "cpu":
        raise ValueError("P05-B2 sample weights must remain on CPU")
    if prediction.dtype != torch.float32 or b0_logits.dtype != torch.float32:
        raise ValueError("P05-B2 prediction and target logits must be float32")
    if sample_weights.dtype != torch.float32:
        raise ValueError("P05-B2 sample weights must be float32")
    if prediction.ndim != 2 or prediction.shape != b0_logits.shape:
        raise ValueError("prediction and frozen B0 logits must have identical (windows, K) shape")
    if sample_weights.shape != (prediction.shape[0],):
        raise ValueError("sample weights must have shape (windows,)")
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(b0_logits).all()):
        raise FloatingPointError("P05-B2 logits must be finite")
    if not bool(torch.isfinite(sample_weights).all()) or not bool((sample_weights > 0).all()):
        raise ValueError("P05-B2 sample weights must be finite and positive")
    per_window_mean_k = (prediction - b0_logits).square().mean(dim=1)
    return (sample_weights * per_window_mean_k).sum() / sample_weights.sum()


def _evaluate(model: FuzzyReasoner, split: _PreparedSplit) -> float:
    model.eval()
    with torch.no_grad():
        prediction = model(split.features)
        loss = p05_b2_weighted_logit_mse(prediction, split.logits, split.weights)
    value = float(loss.item())
    if not math.isfinite(value):
        raise FloatingPointError(f"{split.role} weighted logit MSE is non-finite")
    return value


def _new_reasoner(num_classes: int) -> FuzzyReasoner:
    cfg = FuzzyConfig(
        num_fuzzy_features=INPUT_FEATURES,
        num_membership_functions=MEMBERSHIP_FUNCTIONS,
        num_rules=RULES,
        logit_scale=LOGIT_SCALE,
        antecedent_temperature=ANTECEDENT_TEMPERATURE,
        min_width=MIN_WIDTH,
        firing_epsilon=FIRING_EPSILON,
    )
    model = FuzzyReasoner(dim_in=INPUT_FEATURES, num_classes=num_classes, cfg=cfg)
    model.to(device=torch.device("cpu"), dtype=torch.float32)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != _EXPECTED_PARAMETER_COUNT[num_classes]:
        raise RuntimeError(
            "P05-B2 FuzzyReasoner parameter contract drift: "
            f"expected {_EXPECTED_PARAMETER_COUNT[num_classes]}, got {parameter_count}"
        )
    return model


def _clone_state(model: FuzzyReasoner) -> dict[str, torch.Tensor]:
    state = {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }
    if not state or any(value.dtype != torch.float32 for value in state.values()):
        raise RuntimeError("P05-B2 checkpoint must contain only float32 model tensors")
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise FloatingPointError("P05-B2 checkpoint contains non-finite parameters")
    return state


def _state_arrays(state: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        name: np.ascontiguousarray(value.detach().cpu().numpy(), dtype="<f4")
        for name, value in sorted(state.items())
    }


def _assert_create_only_target(target: Path) -> None:
    if target.is_symlink():
        raise FileExistsError(f"refusing P05-B2 export through symlink: {target}")
    if target.exists():
        raise FileExistsError(f"P05-B2 artifact conflicts with existing target: {target}")


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
        raise RuntimeError("atomic create-only P05-B2 export requires Linux renameat2")
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
        raise FileExistsError(
            error_number,
            "P05-B2 artifact conflicts with existing target",
            str(target),
        )
    raise OSError(error_number, os.strerror(error_number), str(target))


def _write_checkpoint(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as handle:
        np.savez(handle, **{name: arrays[name] for name in sorted(arrays)})
        handle.flush()
        os.fsync(handle.fileno())


def _write_manifest(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _write_artifact(
    target: Path,
    *,
    state: Mapping[str, torch.Tensor],
    semantic_manifest: Mapping[str, Any],
) -> P05B2TrainingResult:
    _assert_create_only_target(target)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"P05-B2 artifact parent must be a real directory: {parent}")
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(parent),
        )
    )
    try:
        arrays = _state_arrays(state)
        checkpoint_path = temporary / CHECKPOINT_NAME
        _write_checkpoint(checkpoint_path, arrays)
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        semantic_sha256 = _sha256_bytes(_canonical_json_bytes(semantic_manifest))
        manifest = {
            **semantic_manifest,
            "content": {
                "checkpoint_sha256": checkpoint_sha256,
                "semantic_sha256": semantic_sha256,
            },
        }
        _write_manifest(temporary / MANIFEST_NAME, _pretty_json_bytes(manifest))
        _fsync_directory(temporary)
        _rename_directory_noreplace(temporary, target)
        _fsync_directory(parent)
        manifest_path = target / MANIFEST_NAME
        selection = manifest["selection"]
        return P05B2TrainingResult(
            package_dir=target,
            checkpoint_path=target / CHECKPOINT_NAME,
            manifest_path=manifest_path,
            best_epoch=int(selection["best_epoch"]),
            best_validation_mse=float(selection["best_validation_mse"]),
            epochs_ran=int(selection["epochs_ran"]),
            stopped_early=bool(selection["stopped_early"]),
            semantic_sha256=semantic_sha256,
            checkpoint_sha256=checkpoint_sha256,
            manifest_sha256=_sha256_file(manifest_path),
            status="created",
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def train_p05_b2_posthoc_surrogate(
    package_dir: str | Path,
    *,
    train: P05B2FrozenSplit,
    validation: P05B2FrozenSplit,
    model_seed: int,
    b0_checkpoint_sha256: str,
    b0_run_artifact_semantic_sha256: str,
) -> P05B2TrainingResult:
    """Fit and create one strict P05-B2 selected-checkpoint artifact."""

    target = Path(os.path.abspath(os.fspath(package_dir)))
    _assert_create_only_target(target)
    if type(model_seed) is not int or model_seed not in REGISTERED_SEEDS:
        raise ValueError("model_seed must equal the corresponding registered B0 seed")
    b0_checkpoint_hash = _required_sha256(
        b0_checkpoint_sha256,
        name="b0_checkpoint_sha256",
    )
    b0_run_hash = _required_sha256(
        b0_run_artifact_semantic_sha256,
        name="b0_run_artifact_semantic_sha256",
    )
    prepared_train = _prepare_split(train, role="train")
    prepared_validation = _prepare_split(validation, role="validation")
    if prepared_train.dataset_id != prepared_validation.dataset_id:
        raise ValueError("train and validation must come from the same P05 dataset")
    if prepared_train.logits.shape[1] != prepared_validation.logits.shape[1]:
        raise ValueError("train and validation frozen B0 logits must have the same K")
    if set(prepared_train.sample_ids) & set(prepared_validation.sample_ids):
        raise ValueError("train and validation sample IDs must be disjoint")
    if set(prepared_train.record_ids) & set(prepared_validation.record_ids):
        raise ValueError("train and validation record IDs must be disjoint")
    if set(prepared_train.group_ids) & set(prepared_validation.group_ids):
        raise ValueError("train and validation groups must be disjoint")

    previous_determinism = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(model_seed)
            model = _new_reasoner(int(prepared_train.logits.shape[1]))
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                betas=(0.9, 0.999),
                eps=1.0e-8,
                amsgrad=False,
            )
            shuffle_generator = torch.Generator(device="cpu")
            shuffle_generator.manual_seed(model_seed)
            best_validation = math.inf
            best_epoch = 0
            best_state: dict[str, torch.Tensor] | None = None
            patience_count = 0
            history: list[dict[str, Any]] = []

            for epoch_index in range(MAX_EPOCHS):
                model.train()
                order = torch.randperm(
                    prepared_train.features.shape[0],
                    generator=shuffle_generator,
                    device="cpu",
                )
                for start in range(0, int(order.numel()), BATCH_SIZE):
                    indices = order[start : start + BATCH_SIZE]
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(prepared_train.features.index_select(0, indices))
                    loss = p05_b2_weighted_logit_mse(
                        prediction,
                        prepared_train.logits.index_select(0, indices),
                        prepared_train.weights.index_select(0, indices),
                    )
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("P05-B2 training loss became non-finite")
                    loss.backward()
                    optimizer.step()
                if any(
                    not bool(torch.isfinite(parameter).all())
                    for parameter in model.parameters()
                ):
                    raise FloatingPointError("P05-B2 model parameters became non-finite")

                train_mse = _evaluate(model, prepared_train)
                validation_mse = _evaluate(model, prepared_validation)
                improved = validation_mse < best_validation - MIN_DELTA
                if improved:
                    best_validation = validation_mse
                    best_epoch = epoch_index + 1
                    best_state = _clone_state(model)
                    patience_count = 0
                else:
                    patience_count += 1
                history.append(
                    {
                        "epoch": epoch_index + 1,
                        "train_weighted_logit_mse": train_mse,
                        "validation_equal_group_equal_window_weighted_logit_mse": (
                            validation_mse
                        ),
                        "strict_validation_improvement": improved,
                        "patience_count": patience_count,
                    }
                )
                if patience_count >= PATIENCE:
                    break
    finally:
        torch.use_deterministic_algorithms(previous_determinism)

    if best_state is None or best_epoch <= 0 or not history:
        raise RuntimeError("P05-B2 training produced no validation-selected checkpoint")
    recorded_validation = [
        float(row["validation_equal_group_equal_window_weighted_logit_mse"])
        for row in history
    ]
    strict_minimum = min(recorded_validation)
    strict_minimum_epoch = recorded_validation.index(strict_minimum) + 1
    if strict_minimum != best_validation or strict_minimum_epoch != best_epoch:
        raise RuntimeError("P05-B2 selected checkpoint is not the strict minimum validation MSE")
    model.load_state_dict(best_state, strict=True)
    selected_validation = _evaluate(model, prepared_validation)
    if selected_validation != best_validation:
        raise RuntimeError("P05-B2 restored checkpoint does not reproduce selected validation MSE")

    state_arrays = _state_arrays(best_state)
    epochs_ran = len(history)
    stopped_early = epochs_ran < MAX_EPOCHS
    semantic_manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "paper_id": "P05",
        "baseline_id": "P05-B2",
        "role": "posthoc_fuzzy_surrogate",
        "evidence_status": "unadjudicated",
        "device": "cpu",
        "precision": "float32",
        "model": {
            "implementation": (
                "src.model_factory.X_model.UXFD.fuzzy.FuzzyReasoner"
            ),
            "input_features": INPUT_FEATURES,
            "membership_functions": MEMBERSHIP_FUNCTIONS,
            "rules": RULES,
            "num_classes": int(prepared_train.logits.shape[1]),
            "logit_scale": LOGIT_SCALE,
            "antecedent_temperature": ANTECEDENT_TEMPERATURE,
            "min_width": MIN_WIDTH,
            "firing_epsilon": FIRING_EPSILON,
            "residual_logits": "none",
            "parameter_count": sum(value.numel() for value in best_state.values()),
        },
        "training_contract": {
            "target": "frozen_B0_K_logits",
            "label_usage": "forbidden_for_target_training_and_selection",
            "train_reduction": (
                "sum(registered_window_weight * per_window_mean_K_logit_squared_error)"
                "/sum(registered_window_weight)"
            ),
            "validation_reduction": "equal_group_equal_window_weighted_logit_MSE",
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "adam_betas": [0.9, 0.999],
            "adam_eps": 1.0e-8,
            "adam_amsgrad": False,
            "batch_size": BATCH_SIZE,
            "shuffle": "torch_cpu_generator_seeded_by_corresponding_B0_seed",
            "drop_last": False,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "patience_rule": "validation_epochs_without_strictly_lower_MSE",
            "selector": "strict_minimum_validation_logit_mse",
            "complexity_search": "none",
        },
        "provenance": {
            "model_seed": model_seed,
            "b0_checkpoint_sha256": b0_checkpoint_hash,
            "b0_run_artifact_semantic_sha256": b0_run_hash,
            "software": {
                "numpy": np.__version__,
                "torch": str(torch.__version__),
            },
            "train": prepared_train.provenance,
            "validation": prepared_validation.provenance,
        },
        "selection": {
            "best_epoch": best_epoch,
            "best_validation_mse": best_validation,
            "epochs_ran": epochs_ran,
            "stopped_early": stopped_early,
            "stop_reason": (
                "patience_exhausted" if stopped_early else "max_epochs_reached"
            ),
            "history": history,
        },
        "checkpoint": {
            "file": CHECKPOINT_NAME,
            "format": "numpy.npz_no_pickle",
            "state_tensors": {
                name: _array_descriptor(array)
                for name, array in sorted(state_arrays.items())
            },
        },
    }
    return _write_artifact(
        target,
        state=best_state,
        semantic_manifest=semantic_manifest,
    )


def load_p05_b2_surrogate_checkpoint(package_dir: str | Path) -> FuzzyReasoner:
    """Verify and load a strict P05-B2 NPZ checkpoint on CPU."""

    target = Path(os.path.abspath(os.fspath(package_dir)))
    if target.is_symlink() or not target.is_dir():
        raise FileNotFoundError(f"P05-B2 package must be a real directory: {target}")
    entries = {entry.name: entry for entry in target.iterdir()}
    if set(entries) != {CHECKPOINT_NAME, MANIFEST_NAME}:
        raise ValueError("P05-B2 package has unexpected or incomplete content")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries.values()):
        raise ValueError("P05-B2 package entries must be real files")
    try:
        manifest = json.loads(entries[MANIFEST_NAME].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("P05-B2 manifest is invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_name") != SCHEMA_NAME:
        raise ValueError("P05-B2 manifest schema is unsupported")
    content = manifest.get("content")
    if not isinstance(content, dict) or set(content) != {
        "checkpoint_sha256",
        "semantic_sha256",
    }:
        raise ValueError("P05-B2 manifest content hashes are invalid")
    checkpoint_hash = _required_sha256(
        content["checkpoint_sha256"],
        name="content.checkpoint_sha256",
    )
    semantic_hash = _required_sha256(
        content["semantic_sha256"],
        name="content.semantic_sha256",
    )
    if checkpoint_hash != _sha256_file(entries[CHECKPOINT_NAME]):
        raise ValueError("P05-B2 checkpoint hash does not match its manifest")
    semantic_manifest = {key: value for key, value in manifest.items() if key != "content"}
    if semantic_hash != _sha256_bytes(_canonical_json_bytes(semantic_manifest)):
        raise ValueError("P05-B2 semantic hash does not match its manifest")
    model_contract = manifest.get("model")
    if not isinstance(model_contract, dict):
        raise ValueError("P05-B2 model contract is missing")
    if (
        model_contract.get("input_features") != INPUT_FEATURES
        or model_contract.get("membership_functions") != MEMBERSHIP_FUNCTIONS
        or model_contract.get("rules") != RULES
        or model_contract.get("logit_scale") != LOGIT_SCALE
        or model_contract.get("antecedent_temperature") != ANTECEDENT_TEMPERATURE
        or model_contract.get("min_width") != MIN_WIDTH
        or model_contract.get("firing_epsilon") != FIRING_EPSILON
        or model_contract.get("residual_logits") != "none"
    ):
        raise ValueError("P05-B2 frozen model contract drifted")
    num_classes = model_contract.get("num_classes")
    if type(num_classes) is not int or num_classes not in {2, 4}:
        raise ValueError("P05-B2 num_classes must be 2 or 4")
    try:
        with np.load(entries[CHECKPOINT_NAME], allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True, order="C")
                for name in archive.files
            }
    except (OSError, ValueError) as exc:
        raise ValueError("P05-B2 checkpoint NPZ is invalid") from exc
    descriptors = manifest.get("checkpoint", {}).get("state_tensors")
    if not isinstance(descriptors, dict) or set(descriptors) != set(arrays):
        raise ValueError("P05-B2 checkpoint state inventory differs from its manifest")
    for name, array in arrays.items():
        if array.dtype != np.dtype("<f4") or not np.isfinite(array).all():
            raise ValueError(f"P05-B2 checkpoint tensor {name!r} is not finite float32")
        if descriptors[name] != _array_descriptor(array):
            raise ValueError(f"P05-B2 checkpoint tensor {name!r} descriptor mismatch")
    model = _new_reasoner(num_classes)
    expected_state = model.state_dict()
    if set(arrays) != set(expected_state):
        raise ValueError("P05-B2 checkpoint state keys do not match FuzzyReasoner")
    state = {name: torch.from_numpy(arrays[name].copy()) for name in arrays}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


__all__ = [
    "P05B2FrozenSplit",
    "P05B2TrainingResult",
    "load_p05_b2_surrogate_checkpoint",
    "p05_b2_weighted_logit_mse",
    "train_p05_b2_posthoc_surrogate",
]
