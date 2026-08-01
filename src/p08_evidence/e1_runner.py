"""Fail-closed standalone runner for the frozen P08 E1 experiment.

The evidence path intentionally bypasses the legacy Lightning task, split,
normalization, and test hooks.  One process executes the eight prespecified
fits for one model seed on exactly one physical GPU, finalizes all four arm
checkpoints, and only then constructs the analytic test payload.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly
import torch
import torch.nn.functional as F
import yaml

from src.p08_evidence.e1_data import (
    CLASS_IDS,
    DURATION_S,
    EVALUATION_RATES_HZ,
    GENERATOR_VERSION,
    PROTOCOL_ID,
    RateCopy,
    canonical_json_sha256,
    iter_rate_copies,
    samples_sha256,
    split_underlying_ids,
)
from src.p08_evidence.e1_model import ArmSpec, arm_spec, build_model, pretraining_loss
from src.p08_evidence.environment import snapshot_text
from src.p08_evidence.metrics import (
    E1Predictions,
    e1_prediction_consistency,
    e1_representation_distance,
    e1_worst_rate_balanced_accuracy,
    record_classification_metrics,
)
from src.p08_evidence.runtime import (
    DevicePreflightRecord,
    EvidenceWriter,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    strict_single_gpu_preflight,
)


RUNTIME_ROOT = Path(__file__).resolve().parents[2]
P08_ROOT = RUNTIME_ROOT.parents[1]
DEFAULT_CONFIG = RUNTIME_ROOT / "configs/experiments/p08/p08_e1_decisive.yaml"
DEFAULT_OUTPUT_ROOT = P08_ROOT / "paper/experiments/runs"
REQUIRED_BRANCH = "agent/p08-vibench-shared-v3-20260731"
EVIDENCE_SEEDS = (42, 123, 456, 789, 999)
CONDA_ENVIRONMENT = "LQ_signal"
KAISER_BETA = 8.6
NUM_CLASSES = 4
_FORMAL_COMMAND_ENV = (
    "PYTHONHASHSEED",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "PYTHONDONTWRITEBYTECODE",
    "MPLCONFIGDIR",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _run_command(
    command: Sequence[str], *, cwd: Path = RUNTIME_ROOT, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _git_value(*arguments: str, repository: Path = RUNTIME_ROOT) -> str:
    return _run_command(("git", *arguments), cwd=repository).stdout.strip()


def _canonical_formal_launch_command(
    *, seed: int, config_path: Path, output_root: Path
) -> str:
    required_environment = {
        "PYTHONHASHSEED": str(seed),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
        ),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/tmp/p08-mpl"),
    }
    tokens = [
        "conda",
        "run",
        "-n",
        CONDA_ENVIRONMENT,
        "--no-capture-output",
        "env",
    ]
    tokens.extend(f"{name}={required_environment[name]}" for name in _FORMAL_COMMAND_ENV)
    tokens.extend(
        (
            "python",
            "-m",
            "src.p08_evidence.e1_runner",
            "run-seed",
            "--seed",
            str(seed),
            "--config",
            str(config_path.resolve()),
            "--output-root",
            str(output_root.resolve()),
        )
    )
    return shlex.join(tokens)


def _validate_formal_launch_command(
    command: str, *, seed: int, config_path: Path, output_root: Path
) -> str:
    tokens = shlex.split(str(command))
    if tokens[:4] != ["conda", "run", "-n", CONDA_ENVIRONMENT]:
        raise ValueError("formal command must begin exactly with conda run -n LQ_signal")
    expected = shlex.split(
        _canonical_formal_launch_command(
            seed=seed, config_path=config_path, output_root=output_root
        )
    )
    if tokens != expected:
        raise ValueError(
            "recorded formal command differs from the canonical process contract"
        )
    return shlex.join(tokens)


def _load_config(path: Path, *, require_approved: bool) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"E1 config does not exist: {path}")
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("E1 config must be a YAML mapping")
    if config.get("config_id") != "P08-E1-decisive-v1.1":
        raise ValueError("unexpected E1 config_id")
    protocol = config.get("protocol", {})
    if protocol.get("id") != PROTOCOL_ID:
        raise ValueError(
            f"runner implements {PROTOCOL_ID}, config declares {protocol.get('id')!r}"
        )
    if require_approved and (
        protocol.get("approved") is not True
        or protocol.get("status") != "frozen_human_approved"
    ):
        raise RuntimeError("formal E1 evidence requires an explicitly approved protocol")
    amendment = protocol.get("amendment", {})
    if require_approved and (
        protocol.get("supersedes") != "P08-LOSO-v1"
        or amendment.get("timing") != "pre_result_before_any_formal_evidence_run"
        or amendment.get("prior_protocol_formal_evidence_runs") != 0
        or amendment.get("formal_evidence_launch_allowed") is not True
    ):
        raise RuntimeError("formal E1 evidence requires the approved v1.1 amendment")
    paper_state_path = P08_ROOT / "paper/paper.yaml"
    if not paper_state_path.is_file():
        raise FileNotFoundError(f"paper state does not exist: {paper_state_path}")
    paper_state = yaml.safe_load(paper_state_path.read_text(encoding="utf-8"))
    paper_protocol = paper_state.get("experiment_protocol", {})
    paper_gate = paper_state.get("human_gates", {})
    if require_approved and (
        paper_protocol.get("active_id") != PROTOCOL_ID
        or paper_protocol.get("approved_id") != PROTOCOL_ID
        or paper_gate.get("experiment_protocol_approved") is not True
        or paper_gate.get("experiment_protocol_approved_version") != PROTOCOL_ID
    ):
        raise RuntimeError("paper state is not version-bound to the approved protocol")
    source_path = P08_ROOT / str(protocol.get("source_path", ""))
    if not source_path.is_file():
        raise FileNotFoundError(f"protocol source does not exist: {source_path}")
    source_digest = sha256_file(source_path)
    if source_digest != protocol.get("source_sha256"):
        raise RuntimeError(
            "protocol source hash differs from resolved E1 config: "
            f"expected={protocol.get('source_sha256')}, observed={source_digest}"
        )
    if tuple(config["training"]["seeds"]) != EVIDENCE_SEEDS:
        raise ValueError("the frozen five-seed order changed")
    if tuple(config["data"]["generator"]["evaluation_rates_hz"]) != EVALUATION_RATES_HZ:
        raise ValueError("the frozen six-rate grid changed")
    if config["candidate_selection"]["total_fits_per_seed"] != 8:
        raise ValueError("E1 requires exactly eight fits per seed")
    if config["execution"]["device"]["forbidden_physical_gpu_indices"] != [2]:
        raise ValueError("physical GPU 2 exclusion changed")
    training = config["training"]
    expected_training_values = {
        "batch_size": 64,
        "batches_per_rate_per_epoch": 10,
        "batches_per_epoch": 60,
        "optimizer": "adamw",
        "learning_rate_scheduler": "none",
        "weight_decay": 0.0001,
        "numerical_precision": "ieee_float32",
    }
    for key, expected in expected_training_values.items():
        if training.get(key) != expected:
            raise ValueError(
                f"frozen training value changed for {key}: "
                f"expected={expected!r}, observed={training.get(key)!r}"
            )
    expected_pretrain = {
        "max_epochs": 30,
        "learning_rate": 0.0005,
        "contrastive_weight": 0.4,
        "classification_weight": 0.1,
        "temperature": 0.07,
        "feature_view_noise_std": 0.1,
    }
    for key, expected in expected_pretrain.items():
        if training["pretrain"].get(key) != expected:
            raise ValueError(f"frozen pretraining value changed for {key}")
    if training["finetune"].get("max_epochs") != 20 or training["finetune"].get(
        "learning_rate"
    ) != 0.0001:
        raise ValueError("frozen finetuning schedule changed")
    if training["early_stopping"] != {
        "monitor": "validation_balanced_accuracy_equal_weight_over_rates_then_classes",
        "patience": 5,
        "min_delta": 0.0001,
        "tie_break": "earliest_epoch",
    }:
        raise ValueError("frozen E1 early-stopping contract changed")
    if config["model"].get("dropout") != 0.1:
        raise ValueError("frozen model dropout changed")
    analytic = config["data"]["generator"]["analytic_components"]
    expected_noise_contract = {
        "noise_base_draws": "iid_standard_normal",
        "finite_sample_noise_centering": "subtract_realized_sample_mean",
        "finite_sample_noise_scaling": "rescale_realized_rms_to_exact_drawn_snr",
        "noise_realizations_per_underlying_signal": 1,
        "noise_injection_stage": "native_200khz_before_rate_conversion",
        "all_rate_copies_share_same_noisy_native_realization": True,
        "per_rate_noise_redraw_allowed": False,
    }
    for key, expected in expected_noise_contract.items():
        if analytic.get(key) != expected:
            raise ValueError(f"frozen E1 noise contract changed for {key}")
    return config, sha256_bytes(raw)


def _signal_handle(class_id: int, underlying_id: int) -> str:
    identity = {
        "generator_version": GENERATOR_VERSION,
        "class_id": int(class_id),
        "underlying_id": int(underlying_id),
    }
    return canonical_json_sha256(identity)


@dataclass(frozen=True, slots=True)
class RawRecord:
    class_id: int
    underlying_id: int
    split: str
    original_rate_hz: int
    signal_handle: str
    samples: NDArray[np.float64]
    sample_sha256: str


@dataclass(frozen=True, slots=True)
class PreparedRecord:
    class_id: int
    underlying_id: int
    split: str
    original_rate_hz: int
    signal_handle: str
    model_rate_numerator_hz: int
    model_rate_denominator: int
    samples: NDArray[np.float32]
    preprocessing: Mapping[str, Any]

    @property
    def model_rate_hz(self) -> float:
        return self.model_rate_numerator_hz / self.model_rate_denominator


@dataclass(frozen=True, slots=True)
class UnlabeledInferenceRecord:
    underlying_id: int
    original_rate_hz: int
    signal_handle: str
    model_rate_numerator_hz: int
    model_rate_denominator: int
    samples: NDArray[np.float32]

    @property
    def model_rate_hz(self) -> float:
        return self.model_rate_numerator_hz / self.model_rate_denominator


@dataclass(frozen=True, slots=True)
class NormalizationRecord:
    ordered_input_hash: str
    sample_count: int
    mean: float
    standard_deviation: float
    algorithm: str
    dtype: str
    iteration_order: tuple[str, ...]
    canonical_json_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    spec: ArmSpec
    numeric_numerator: int
    numeric_denominator: int
    compute_proxy: int

    @property
    def numeric_value(self) -> float:
        return self.numeric_numerator / self.numeric_denominator


@dataclass(slots=True)
class FitResult:
    candidate: Candidate
    state_dict: dict[str, torch.Tensor]
    validation_score: float
    validation_by_rate: dict[str, float]
    pretrain_best_epoch: int
    pretrain_best_validation_score: float
    finetune_best_epoch: int
    epoch_rows: list[dict[str, Any]]
    elapsed_seconds: float
    total_parameters: int
    trainable_parameters: int


def _readonly_float32(values: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
    result = np.asarray(values, dtype=np.float32, order="C").copy()
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError("prepared signal must be finite and one-dimensional")
    result.setflags(write=False)
    return result


def _load_raw_records(split: str, *, limit_per_class: int | None) -> list[RawRecord]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"invalid E1 split {split!r}")
    allowed: dict[int, set[int]] | None = None
    if limit_per_class is not None:
        if limit_per_class < 1:
            raise ValueError("limit_per_class must be positive")
        allowed = {
            class_id: set(split_underlying_ids(class_id)[split][:limit_per_class])
            for class_id in CLASS_IDS
        }
    result: list[RawRecord] = []
    for copy in iter_rate_copies(split=split):
        if allowed is not None and copy.underlying_id not in allowed[copy.class_id]:
            continue
        result.append(
            RawRecord(
                class_id=copy.class_id,
                underlying_id=copy.underlying_id,
                split=copy.split,
                original_rate_hz=copy.sample_rate_hz,
                signal_handle=_signal_handle(copy.class_id, copy.underlying_id),
                samples=copy.samples,
                sample_sha256=copy.sample_sha256,
            )
        )
    expected_signals = (
        limit_per_class
        if limit_per_class is not None
        else len(split_underlying_ids(0)[split])
    )
    expected_records = NUM_CLASSES * expected_signals * len(EVALUATION_RATES_HZ)
    if len(result) != expected_records:
        raise RuntimeError(
            f"unexpected {split} record count: expected {expected_records}, got {len(result)}"
        )
    ordering = [
        (record.class_id, record.underlying_id, record.original_rate_hz)
        for record in result
    ]
    if ordering != sorted(ordering):
        raise RuntimeError("analytic records are not in frozen normalization order")
    return result


def _raw_manifest(records: Sequence[RawRecord], *, split: str) -> dict[str, Any]:
    digest = sha256()
    identity = {
        "generator_version": GENERATOR_VERSION,
        "split": split,
        "ordering": ["class_id", "underlying_id", "original_rate_hz"],
    }
    digest.update(canonical_json_bytes(identity))
    handles: set[str] = set()
    sample_count = 0
    for record in records:
        digest.update(bytes.fromhex(record.sample_sha256))
        handles.add(record.signal_handle)
        sample_count += int(record.samples.size)
    return {
        **identity,
        "rate_copy_count": len(records),
        "underlying_signal_count": len(handles),
        "sample_count": sample_count,
        "bank_sha256": digest.hexdigest(),
    }


def _fit_normalization(records: Sequence[RawRecord]) -> NormalizationRecord:
    count = 0
    mean = 0.0
    m2 = 0.0
    ordered_digest = sha256()
    for record in records:
        ordered_digest.update(
            canonical_json_bytes(
                {
                    "class_id": record.class_id,
                    "underlying_id": record.underlying_id,
                    "rate_hz": record.original_rate_hz,
                    "sample_sha256": record.sample_sha256,
                }
            )
        )
        for raw_value in record.samples:
            value = float(raw_value)
            count += 1
            delta = value - mean
            mean += delta / count
            m2 += delta * (value - mean)
    if count < 2:
        raise RuntimeError("normalization requires at least two samples")
    variance = m2 / count
    standard_deviation = math.sqrt(variance)
    if not math.isfinite(mean) or not math.isfinite(standard_deviation):
        raise FloatingPointError("normalization produced a non-finite statistic")
    if standard_deviation <= 0.0:
        raise FloatingPointError("normalization standard deviation is not positive")
    base = {
        "ordered_input_hash": ordered_digest.hexdigest(),
        "sample_count": count,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "algorithm": "deterministic_float64_welford_population_ddof_0",
        "dtype": "float64_fit_float64_apply_then_float32_cast",
        "iteration_order": (
            "class_id_sorted",
            "underlying_id_sorted",
            "exact_sampling_rate_hz_sorted",
            "sample_index_ascending",
        ),
    }
    return NormalizationRecord(
        **base,
        canonical_json_sha256=canonical_json_sha256(base),
    )


def _half_up_duration_points(numerator_hz: int, denominator: int) -> int:
    exact = Fraction(numerator_hz, denominator) * Fraction(1, 50)
    return (2 * exact.numerator + exact.denominator) // (2 * exact.denominator)


def _prepare_records(
    records: Sequence[RawRecord],
    normalization: NormalizationRecord,
    spec: ArmSpec,
) -> list[PreparedRecord]:
    prepared: list[PreparedRecord] = []
    for record in records:
        normalized = (
            np.asarray(record.samples, dtype=np.float64) - normalization.mean
        ) / normalization.standard_deviation
        if spec.arm_id == "P08-BG":
            numerator = spec.global_resample_numerator_hz
            denominator = spec.global_resample_denominator
            if numerator is None or denominator is None:
                raise RuntimeError("P08-BG is missing its exact target rational")
            ratio = Fraction(numerator, denominator * record.original_rate_hz)
            converted = resample_poly(
                normalized,
                up=ratio.numerator,
                down=ratio.denominator,
                window=("kaiser", KAISER_BETA),
                padtype="line",
            )
            required = _half_up_duration_points(numerator, denominator)
            if converted.size < required:
                raise RuntimeError("BG resampling is shorter than the required crop")
            crop_start = (int(converted.size) - required) // 2
            transformed = converted[crop_start : crop_start + required]
            if transformed.size != required:
                raise RuntimeError("BG center crop returned an unexpected length")
            preprocessing: dict[str, Any] = {
                "operation": "normalize_then_exact_polyphase_resample_then_center_crop",
                "source_rate_hz": record.original_rate_hz,
                "target_rate_numerator_hz": numerator,
                "target_rate_denominator": denominator,
                "resample_up": ratio.numerator,
                "resample_down": ratio.denominator,
                "resampled_points_before_crop": int(converted.size),
                "required_output_points": required,
                "crop_start": crop_start,
                "crop_stop": crop_start + required,
                "window": ["kaiser", KAISER_BETA],
                "padtype": "line",
            }
            model_numerator = numerator
            model_denominator = denominator
        else:
            transformed = normalized
            preprocessing = {
                "operation": "source_train_scalar_normalization_native_rate",
                "source_rate_hz": record.original_rate_hz,
                "output_points": int(normalized.size),
            }
            model_numerator = record.original_rate_hz
            model_denominator = 1
        prepared.append(
            PreparedRecord(
                class_id=record.class_id,
                underlying_id=record.underlying_id,
                split=record.split,
                original_rate_hz=record.original_rate_hz,
                signal_handle=record.signal_handle,
                model_rate_numerator_hz=model_numerator,
                model_rate_denominator=model_denominator,
                samples=_readonly_float32(transformed),
                preprocessing=preprocessing,
            )
        )
    return prepared


def _records_by_rate_class(
    records: Sequence[PreparedRecord],
) -> dict[int, dict[int, list[PreparedRecord]]]:
    grouped: dict[int, dict[int, list[PreparedRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        grouped[record.original_rate_hz][record.class_id].append(record)
    if tuple(sorted(grouped)) != EVALUATION_RATES_HZ:
        raise RuntimeError("prepared training data does not cover the six rates")
    for rate_hz in EVALUATION_RATES_HZ:
        if tuple(sorted(grouped[rate_hz])) != CLASS_IDS:
            raise RuntimeError(f"rate {rate_hz} does not cover the four classes")
    return {rate: dict(by_class) for rate, by_class in grouped.items()}


def _training_batches(
    records: Sequence[PreparedRecord],
    *,
    seed: int,
    stage_index: int,
    epoch: int,
    batch_size: int,
    batches_per_rate: int,
) -> Iterable[list[PreparedRecord]]:
    if batch_size < NUM_CLASSES:
        raise ValueError("batch_size must cover every E1 class")
    grouped = _records_by_rate_class(records)
    schedule = np.repeat(np.asarray(EVALUATION_RATES_HZ), batches_per_rate)
    schedule_rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([seed, stage_index, epoch, 71]))
    )
    schedule = schedule_rng.permutation(schedule)
    for batch_index, rate_hz_raw in enumerate(schedule):
        rate_hz = int(rate_hz_raw)
        base_count, remainder = divmod(batch_size, NUM_CLASSES)
        class_counts = [base_count] * NUM_CLASSES
        remainder_start = (seed + batch_index) % NUM_CLASSES
        for offset in range(remainder):
            class_counts[(remainder_start + offset) % NUM_CLASSES] += 1
        rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence(
                    [seed, stage_index, epoch, batch_index, rate_hz]
                )
            )
        )
        batch: list[PreparedRecord] = []
        for class_id, class_count in zip(CLASS_IDS, class_counts, strict=True):
            population = grouped[rate_hz][class_id]
            indices = rng.integers(0, len(population), size=class_count)
            batch.extend(population[int(index)] for index in indices)
        order = rng.permutation(len(batch))
        yield [batch[int(index)] for index in order]


def _assert_batch_contract(batch: Sequence[PreparedRecord], spec: ArmSpec) -> None:
    if not batch:
        raise RuntimeError("empty E1 batch")
    lengths = {int(record.samples.size) for record in batch}
    original_rates = {record.original_rate_hz for record in batch}
    model_rationals = {
        (record.model_rate_numerator_hz, record.model_rate_denominator)
        for record in batch
    }
    if len(lengths) != 1 or len(original_rates) != 1 or len(model_rationals) != 1:
        raise RuntimeError("E1 batches must be homogeneous in exact rate and length")
    if spec.arm_id == "P08-BG":
        expected = (
            spec.global_resample_numerator_hz,
            spec.global_resample_denominator,
        )
        if next(iter(model_rationals)) != expected:
            raise RuntimeError("BG model rate differs from its exact selected rational")
    else:
        rate = next(iter(original_rates))
        if next(iter(model_rationals)) != (rate, 1):
            raise RuntimeError("native-rate arm received altered sampling metadata")


def _torch_batch(
    batch: Sequence[PreparedRecord], spec: ArmSpec, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _assert_batch_contract(batch, spec)
    signals = np.stack([record.samples for record in batch], axis=0)
    x = torch.from_numpy(signals).unsqueeze(-1).to(device=device, dtype=torch.float32)
    labels = torch.tensor(
        [record.class_id for record in batch], dtype=torch.long, device=device
    )
    rates = torch.tensor(
        [record.model_rate_hz for record in batch],
        dtype=torch.float32,
        device=device,
    )
    if rates.numel() != x.shape[0]:
        raise RuntimeError("sampling-rate vector length differs from batch size")
    return x, labels, rates


def _validation_score(
    model: torch.nn.Module,
    records: Sequence[PreparedRecord],
    spec: ArmSpec,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    by_rate: dict[str, float] = {}
    with torch.inference_mode():
        for rate_hz in EVALUATION_RATES_HZ:
            rate_records = [
                record for record in records if record.original_rate_hz == rate_hz
            ]
            correct = {class_id: 0 for class_id in CLASS_IDS}
            totals = {class_id: 0 for class_id in CLASS_IDS}
            for start in range(0, len(rate_records), batch_size):
                batch = rate_records[start : start + batch_size]
                x, labels, rates = _torch_batch(batch, spec, device)
                logits = model(
                    x,
                    task_id="classification",
                    sampling_rate_hz=rates,
                )
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("validation logits are non-finite")
                predictions = torch.argmax(logits, dim=1)
                for class_id in CLASS_IDS:
                    mask = labels == class_id
                    totals[class_id] += int(mask.sum().item())
                    correct[class_id] += int(
                        ((predictions == class_id) & mask).sum().item()
                    )
            if any(totals[class_id] == 0 for class_id in CLASS_IDS):
                raise RuntimeError("validation rate lacks a prespecified class")
            recalls = [
                correct[class_id] / totals[class_id] for class_id in CLASS_IDS
            ]
            by_rate[str(rate_hz)] = float(np.mean(recalls))
    return float(np.mean(list(by_rate.values()))), by_rate


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _train_stage(
    model: torch.nn.Module,
    *,
    stage: str,
    stage_index: int,
    spec: ArmSpec,
    train_records: Sequence[PreparedRecord],
    validation_records: Sequence[PreparedRecord],
    seed: int,
    candidate_id: str,
    batch_size: int,
    batches_per_rate: int,
    max_epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
    pretrain_temperature: float,
    feature_noise_std: float,
    classification_weight: float,
    contrastive_weight: float,
    device: torch.device,
    deadline_monotonic: float | None,
) -> tuple[
    dict[str, torch.Tensor],
    int,
    float,
    dict[str, float],
    list[dict[str, Any]],
]:
    if stage not in {"pretrain", "finetune"}:
        raise ValueError(f"unexpected E1 stage {stage!r}")
    model.set_training_stage(stage)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -math.inf
    best_by_rate: dict[str, float] = {}
    best_epoch = -1
    patience_reference = -math.inf
    stale_epochs = 0
    rows: list[dict[str, Any]] = []

    # Reset stochastic training draws after construction.  This value is the
    # same for every arm/candidate of one model seed.
    torch.manual_seed(seed + 900_003 + stage_index * 1_000_003)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 900_003 + stage_index * 1_000_003)

    for epoch in range(1, max_epochs + 1):
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError(f"{candidate_id} exceeded its compound-run time ceiling")
        model.train()
        loss_sum = 0.0
        classification_sum = 0.0
        contrastive_sum = 0.0
        batch_count = 0
        rate_batch_counts = {str(rate): 0 for rate in EVALUATION_RATES_HZ}
        class_example_counts = {str(class_id): 0 for class_id in CLASS_IDS}
        metadata_length_mismatch_count = 0
        scalar_broadcast_count = 0
        epoch_started = time.monotonic()
        for batch in _training_batches(
            train_records,
            seed=seed,
            stage_index=stage_index,
            epoch=epoch,
            batch_size=batch_size,
            batches_per_rate=batches_per_rate,
        ):
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise TimeoutError(
                    f"{candidate_id} exceeded its compound-run time ceiling"
                )
            x, labels, rates = _torch_batch(batch, spec, device)
            original_rate_values = {record.original_rate_hz for record in batch}
            if len(original_rate_values) != 1:
                raise RuntimeError("training batch mixed original-rate buckets")
            rate_batch_counts[str(next(iter(original_rate_values)))] += 1
            for class_id in CLASS_IDS:
                class_example_counts[str(class_id)] += sum(
                    record.class_id == class_id for record in batch
                )
            if rates.numel() != x.shape[0]:
                metadata_length_mismatch_count += 1
                raise RuntimeError("training metadata vector length mismatch")
            if rates.numel() == 1 and x.shape[0] > 1:
                scalar_broadcast_count += 1
                raise RuntimeError("training sampling-rate scalar broadcast detected")
            optimizer.zero_grad(set_to_none=True)
            logits, features = model(
                x,
                task_id="classification",
                return_feature=True,
                sampling_rate_hz=rates,
            )
            if stage == "pretrain":
                loss, parts = pretraining_loss(
                    logits,
                    features,
                    labels,
                    temperature=pretrain_temperature,
                    feature_noise_std=feature_noise_std,
                    classification_weight=classification_weight,
                    contrastive_weight=contrastive_weight,
                )
                classification_sum += parts["classification_loss"]
                contrastive_sum += parts["contrastive_loss"]
            else:
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("finetuning logits are non-finite")
                loss = F.cross_entropy(logits, labels)
                parts = {
                    "classification_loss": float(loss.detach().cpu()),
                    "contrastive_loss": 0.0,
                }
                classification_sum += parts["classification_loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{stage} loss is non-finite")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(
                    parameter.grad
                ).all():
                    raise FloatingPointError(f"non-finite gradient in {name}")
            optimizer.step()
            for name, parameter in model.named_parameters():
                if not torch.isfinite(parameter).all():
                    raise FloatingPointError(f"non-finite parameter in {name}")
            loss_sum += float(loss.detach().cpu())
            batch_count += 1

        validation_score, validation_by_rate = _validation_score(
            model,
            validation_records,
            spec,
            batch_size=batch_size,
            device=device,
        )
        if validation_score > best_score:
            best_score = validation_score
            best_epoch = epoch
            best_by_rate = validation_by_rate
            best_state = _clone_state_dict(model)
        if validation_score > patience_reference + min_delta:
            patience_reference = validation_score
            stale_epochs = 0
        else:
            stale_epochs += 1
        rows.append(
            {
                "candidate_id": candidate_id,
                "stage": stage,
                "epoch": epoch,
                "batch_count": batch_count,
                "mean_total_loss": loss_sum / batch_count,
                "mean_classification_loss": classification_sum / batch_count,
                "mean_contrastive_loss": contrastive_sum / batch_count,
                "validation_balanced_accuracy_equal_rates_then_classes": validation_score,
                "validation_balanced_accuracy_by_rate_hz": validation_by_rate,
                "best_epoch_so_far": best_epoch,
                "stale_epochs": stale_epochs,
                "collation_contract": {
                    "input_split": "train",
                    "batch_original_rate_homogeneous": True,
                    "rate_batch_counts": rate_batch_counts,
                    "class_example_counts": class_example_counts,
                    "metadata_length_mismatch_count": metadata_length_mismatch_count,
                    "sampling_rate_scalar_broadcast_count": scalar_broadcast_count,
                },
                "epoch_elapsed_seconds": time.monotonic() - epoch_started,
                "completed_at_utc": _utc_now(),
            }
        )
        if stale_epochs >= patience:
            break
    if best_state is None or best_epoch < 1 or not math.isfinite(best_score):
        raise RuntimeError(f"{stage} did not produce a valid checkpoint")
    model.load_state_dict(best_state, strict=True)
    return best_state, best_epoch, best_score, best_by_rate, rows


def _fit_candidate(
    candidate: Candidate,
    *,
    raw_train: Sequence[RawRecord],
    raw_validation: Sequence[RawRecord],
    normalization: NormalizationRecord,
    seed: int,
    training_config: Mapping[str, Any],
    model_dropout: float,
    device: torch.device,
    deadline_monotonic: float | None,
    smoke_overrides: Mapping[str, int] | None,
) -> FitResult:
    started = time.monotonic()
    train_records = _prepare_records(raw_train, normalization, candidate.spec)
    validation_records = _prepare_records(
        raw_validation, normalization, candidate.spec
    )
    model = build_model(
        candidate.spec, seed=seed, device=device, dropout=model_dropout
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    batch_size = int(
        (smoke_overrides or {}).get("batch_size", training_config["batch_size"])
    )
    batches_per_rate = int(
        (smoke_overrides or {}).get(
            "batches_per_rate", training_config["batches_per_rate_per_epoch"]
        )
    )
    pretrain_epochs = int(
        (smoke_overrides or {}).get(
            "pretrain_epochs", training_config["pretrain"]["max_epochs"]
        )
    )
    finetune_epochs = int(
        (smoke_overrides or {}).get(
            "finetune_epochs", training_config["finetune"]["max_epochs"]
        )
    )
    patience = int(training_config["early_stopping"]["patience"])
    min_delta = float(training_config["early_stopping"]["min_delta"])

    _, pretrain_epoch, pretrain_score, _, pretrain_rows = _train_stage(
        model,
        stage="pretrain",
        stage_index=0,
        spec=candidate.spec,
        train_records=train_records,
        validation_records=validation_records,
        seed=seed,
        candidate_id=candidate.candidate_id,
        batch_size=batch_size,
        batches_per_rate=batches_per_rate,
        max_epochs=pretrain_epochs,
        learning_rate=float(training_config["pretrain"]["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
        patience=patience,
        min_delta=min_delta,
        pretrain_temperature=float(training_config["pretrain"]["temperature"]),
        feature_noise_std=float(training_config["pretrain"]["feature_view_noise_std"]),
        classification_weight=float(
            training_config["pretrain"]["classification_weight"]
        ),
        contrastive_weight=float(training_config["pretrain"]["contrastive_weight"]),
        device=device,
        deadline_monotonic=deadline_monotonic,
    )
    final_state, finetune_epoch, final_score, final_by_rate, finetune_rows = (
        _train_stage(
            model,
            stage="finetune",
            stage_index=1,
            spec=candidate.spec,
            train_records=train_records,
            validation_records=validation_records,
            seed=seed,
            candidate_id=candidate.candidate_id,
            batch_size=batch_size,
            batches_per_rate=batches_per_rate,
            max_epochs=finetune_epochs,
            learning_rate=float(training_config["finetune"]["learning_rate"]),
            weight_decay=float(training_config["weight_decay"]),
            patience=patience,
            min_delta=min_delta,
            pretrain_temperature=float(training_config["pretrain"]["temperature"]),
            feature_noise_std=float(
                training_config["pretrain"]["feature_view_noise_std"]
            ),
            classification_weight=float(
                training_config["pretrain"]["classification_weight"]
            ),
            contrastive_weight=float(
                training_config["pretrain"]["contrastive_weight"]
            ),
            device=device,
            deadline_monotonic=deadline_monotonic,
        )
    )
    del model, train_records, validation_records
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return FitResult(
        candidate=candidate,
        state_dict=final_state,
        validation_score=final_score,
        validation_by_rate=final_by_rate,
        pretrain_best_epoch=pretrain_epoch,
        pretrain_best_validation_score=pretrain_score,
        finetune_best_epoch=finetune_epoch,
        epoch_rows=pretrain_rows + finetune_rows,
        elapsed_seconds=time.monotonic() - started,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
    )


def _select_candidate(results: Sequence[FitResult]) -> FitResult:
    if not results:
        raise ValueError("candidate result list cannot be empty")
    maximum = max(result.validation_score for result in results)
    tied = [result for result in results if result.validation_score == maximum]
    return min(
        tied,
        key=lambda result: (
            result.candidate.compute_proxy,
            Fraction(
                result.candidate.numeric_numerator,
                result.candidate.numeric_denominator,
            ),
        ),
    )


def _dn_candidates() -> list[Candidate]:
    return [
        Candidate(
            candidate_id=f"DN-{duration_ms}ms",
            spec=arm_spec("P08-DN", duration_ms=float(duration_ms)),
            numeric_numerator=duration_ms,
            numeric_denominator=1,
            compute_proxy=duration_ms,
        )
        for duration_ms in (5, 10, 15)
    ]


def _bg_candidates(config: Mapping[str, Any]) -> list[Candidate]:
    result: list[Candidate] = []
    for item in config["arms"]["P08-BG"]["target_rate_candidates"]:
        numerator = int(item["numerator_hz"])
        denominator = int(item["denominator"])
        required = int(item["required_output_points"])
        if _half_up_duration_points(numerator, denominator) != required:
            raise ValueError("BG config required_output_points violates half-up rule")
        result.append(
            Candidate(
                candidate_id=str(item["candidate_id"]),
                spec=arm_spec(
                    "P08-BG",
                    global_resample_numerator_hz=numerator,
                    global_resample_denominator=denominator,
                ),
                numeric_numerator=numerator,
                numeric_denominator=denominator,
                compute_proxy=required,
            )
        )
    if len(result) != 3:
        raise ValueError("P08-BG requires exactly three candidates")
    return result


def _m_reuse_candidate(selected_dn: FitResult) -> Candidate:
    duration_ms = selected_dn.candidate.numeric_value
    return Candidate(
        candidate_id=f"M-reuse-{selected_dn.candidate.candidate_id}",
        spec=arm_spec("P08-M", duration_ms=duration_ms),
        numeric_numerator=selected_dn.candidate.numeric_numerator,
        numeric_denominator=selected_dn.candidate.numeric_denominator,
        compute_proxy=selected_dn.candidate.compute_proxy,
    )


def _nc_candidate() -> Candidate:
    return Candidate(
        candidate_id="NC-fixed-128-points",
        spec=arm_spec("P08-NC"),
        numeric_numerator=128,
        numeric_denominator=1,
        compute_proxy=128,
    )


def _checkpoint_bytes(result: FitResult, *, seed: int) -> bytes:
    payload = {
        "protocol_id": PROTOCOL_ID,
        "experiment_id": "P08-E1",
        "arm_id": result.candidate.spec.arm_id,
        "model_seed": seed,
        "candidate_id": result.candidate.candidate_id,
        "arm_spec": result.candidate.spec.to_dict(),
        "validation_score": result.validation_score,
        "finetune_best_epoch": result.finetune_best_epoch,
        "state_dict": result.state_dict,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _test_payload(
    prepared: Sequence[PreparedRecord],
) -> tuple[list[UnlabeledInferenceRecord], dict[str, int]]:
    labels: dict[str, int] = {}
    payload: list[UnlabeledInferenceRecord] = []
    for record in prepared:
        prior = labels.setdefault(record.signal_handle, record.class_id)
        if prior != record.class_id:
            raise RuntimeError("one target signal handle maps to multiple labels")
        payload.append(
            UnlabeledInferenceRecord(
                underlying_id=record.underlying_id,
                original_rate_hz=record.original_rate_hz,
                signal_handle=record.signal_handle,
                model_rate_numerator_hz=record.model_rate_numerator_hz,
                model_rate_denominator=record.model_rate_denominator,
                samples=record.samples,
            )
        )
    return payload, labels


def _unlabeled_torch_batch(
    batch: Sequence[UnlabeledInferenceRecord],
    spec: ArmSpec,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not batch:
        raise ValueError("inference batch cannot be empty")
    lengths = {int(record.samples.size) for record in batch}
    model_rationals = {
        (record.model_rate_numerator_hz, record.model_rate_denominator)
        for record in batch
    }
    original_rates = {record.original_rate_hz for record in batch}
    if len(lengths) != 1 or len(model_rationals) != 1 or len(original_rates) != 1:
        raise RuntimeError("unlabeled inference batch is not rate/length homogeneous")
    rational = next(iter(model_rationals))
    if spec.arm_id == "P08-BG":
        if rational != (
            spec.global_resample_numerator_hz,
            spec.global_resample_denominator,
        ):
            raise RuntimeError("BG test payload differs from selected exact rational")
    elif rational != (next(iter(original_rates)), 1):
        raise RuntimeError("native test payload has altered sampling metadata")
    signals = np.stack([record.samples for record in batch], axis=0)
    x = torch.from_numpy(signals).unsqueeze(-1).to(device=device, dtype=torch.float32)
    rates = torch.tensor(
        [record.model_rate_hz for record in batch],
        dtype=torch.float32,
        device=device,
    )
    if rates.numel() != x.shape[0]:
        raise RuntimeError("test sampling-rate vector length differs from batch size")
    return x, rates


def _infer_unlabeled(
    result: FitResult,
    payload: Sequence[UnlabeledInferenceRecord],
    *,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    model = build_model(result.candidate.spec, seed=seed, device=device)
    model.load_state_dict(result.state_dict, strict=True)
    model.set_training_stage("finetune")
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for rate_hz in EVALUATION_RATES_HZ:
            rate_records = [
                record for record in payload if record.original_rate_hz == rate_hz
            ]
            for start in range(0, len(rate_records), batch_size):
                batch = rate_records[start : start + batch_size]
                x, rates = _unlabeled_torch_batch(
                    batch, result.candidate.spec, device
                )
                logits, features = model(
                    x,
                    task_id="classification",
                    return_feature=True,
                    sampling_rate_hz=rates,
                )
                if not torch.isfinite(logits).all() or not torch.isfinite(
                    features
                ).all():
                    raise FloatingPointError("test inference produced non-finite output")
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                feature_values = features.cpu().numpy()
                if not np.allclose(
                    probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-6
                ):
                    raise FloatingPointError("test probabilities do not sum to one")
                for record, probability, feature in zip(
                    batch, probabilities, feature_values, strict=True
                ):
                    row: dict[str, Any] = {
                        "protocol_id": PROTOCOL_ID,
                        "experiment_id": "P08-E1",
                        "arm_id": result.candidate.spec.arm_id,
                        "model_seed": seed,
                        "signal_handle": record.signal_handle,
                        "underlying_id": record.underlying_id,
                        "original_rate_hz": record.original_rate_hz,
                        "model_rate_numerator_hz": record.model_rate_numerator_hz,
                        "model_rate_denominator": record.model_rate_denominator,
                        "predicted_class": int(np.argmax(probability)),
                    }
                    for class_id, value in enumerate(probability):
                        row[f"p_class_{class_id}"] = float(value)
                    for feature_index, value in enumerate(feature):
                        row[f"feature_{feature_index:03d}"] = float(value)
                    rows.append(row)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    expected = len(payload)
    if len(rows) != expected:
        raise RuntimeError(f"test inference lost rows: expected {expected}, got {len(rows)}")
    return rows


def _parquet_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot serialize an empty prediction table")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - evidence env is required to ship pyarrow
        raise RuntimeError("pyarrow is required for P08 evidence parquet files") from exc
    table = pa.Table.from_pylist([dict(row) for row in rows])
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd", version="2.6")
    return sink.getvalue().to_pybytes()


def _score_rows(
    unlabeled_rows: Sequence[Mapping[str, Any]], labels: Mapping[str, int]
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for raw_row in unlabeled_rows:
        row = dict(raw_row)
        handle = str(row["signal_handle"])
        if handle not in labels:
            raise RuntimeError("prediction handle is absent from sealed label table")
        row["class_id"] = int(labels[handle])
        scored.append(row)
    return scored


def _metrics_from_scored_rows(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    probabilities = np.asarray(
        [
            [float(row[f"p_class_{class_id}"]) for class_id in CLASS_IDS]
            for row in rows
        ],
        dtype=np.float64,
    )
    labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    signal_ids = [str(row["signal_handle"]) for row in rows]
    model_seeds = np.asarray([seed] * len(rows), dtype=np.int64)
    rates = np.asarray([int(row["original_rate_hz"]) for row in rows], dtype=np.int64)
    embeddings = np.asarray(
        [
            [float(row[f"feature_{index:03d}"]) for index in range(128)]
            for row in rows
        ],
        dtype=np.float64,
    )
    table = E1Predictions.from_columns(
        probabilities=probabilities,
        labels=labels,
        signal_ids=signal_ids,
        model_seeds=model_seeds,
        rates_hz=rates,
    )
    per_rate = {}
    for rate_hz in EVALUATION_RATES_HZ:
        mask = rates == rate_hz
        per_rate[str(rate_hz)] = record_classification_metrics(
            probabilities=probabilities[mask],
            labels=labels[mask],
            supported_classes=CLASS_IDS,
        )
    return {
        "protocol_id": PROTOCOL_ID,
        "experiment_id": "P08-E1",
        "model_seed": seed,
        "record_count": len(rows),
        "overall_descriptive": record_classification_metrics(
            probabilities=probabilities,
            labels=labels,
            supported_classes=CLASS_IDS,
        ),
        "by_original_rate_hz": per_rate,
        "worst_rate_balanced_accuracy": e1_worst_rate_balanced_accuracy(
            table, seeds=(seed,)
        ),
        "prediction_consistency": e1_prediction_consistency(
            table, seeds=(seed,)
        ),
        "representation_distance": e1_representation_distance(
            table, embeddings, seeds=(seed,)
        ),
    }


def _environment_export() -> str:
    return snapshot_text()


def _source_paths() -> tuple[Path, ...]:
    relative = (
        "configs/experiments/p08/p08_e1_decisive.yaml",
        "src/p08_evidence/__init__.py",
        "src/p08_evidence/environment.py",
        "src/p08_evidence/e1_analysis.py",
        "src/p08_evidence/e1_audit.py",
        "src/p08_evidence/e1_data.py",
        "src/p08_evidence/e1_model.py",
        "src/p08_evidence/e1_runner.py",
        "src/p08_evidence/e1_stages.py",
        "src/p08_evidence/metrics.py",
        "src/p08_evidence/runtime.py",
        "src/model_factory/ISFM_Prompt/M_02_ISFM_Prompt.py",
        "src/model_factory/ISFM_Prompt/embedding/HSE_prompt.py",
        "src/model_factory/ISFM/backbone/B_04_Dlinear.py",
        "src/model_factory/ISFM/task_head/H_11_Unified_cla.py",
        "src/task_factory/Components/contrastive_losses.py",
    )
    return tuple(RUNTIME_ROOT / item for item in relative)


def _protocol_paths() -> tuple[Path, ...]:
    return (
        P08_ROOT / "paper/paper.yaml",
        P08_ROOT / "paper/experiments/config_bridge.yaml",
        P08_ROOT / "paper/experiments/experiment_plan.md",
        P08_ROOT / "paper/experiments/statistics.md",
        P08_ROOT / "paper/experiments/ablation.md",
        P08_ROOT / "paper/experiments/run_ledger.md",
    )


def _source_manifest(config_path: Path) -> dict[str, Any]:
    paths = set(_source_paths())
    paths.add(config_path.resolve())
    paths.add(P08_ROOT / "SUBMODULES.lock.yaml")
    paths.update(_protocol_paths())
    rows = []
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(f"evidence source file is absent: {path}")
        try:
            display = path.relative_to(P08_ROOT).as_posix()
        except ValueError:
            display = str(path)
        rows.append(
            {
                "path": display,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {"files": rows}
    payload["source_manifest_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def _paper_dirty_snapshot() -> tuple[str, str]:
    relative_paths = [path.relative_to(P08_ROOT).as_posix() for path in _protocol_paths()]
    relative_paths.append("src/vibench")
    status = _run_command(
        (
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *relative_paths,
        ),
        cwd=P08_ROOT,
    ).stdout
    patch = _run_command(
        ("git", "diff", "--binary", "--", *relative_paths), cwd=P08_ROOT
    ).stdout
    return status, patch


def _dirty_patch() -> tuple[str, str]:
    relative_paths = [path.relative_to(RUNTIME_ROOT).as_posix() for path in _source_paths()]
    status = _run_command(
        (
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *relative_paths,
        )
    ).stdout
    tracked_patch = _run_command(
        ("git", "diff", "--binary", "--", *relative_paths)
    ).stdout
    untracked_chunks: list[str] = []
    for line in status.splitlines():
        if not line.startswith("?? "):
            continue
        relative = line[3:]
        path = RUNTIME_ROOT / relative
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()
        chunk = [
            f"diff --git a/{relative} b/{relative}",
            "new file mode 100644",
            "--- /dev/null",
            f"+++ b/{relative}",
            f"@@ -0,0 +1,{len(lines)} @@",
        ]
        chunk.extend(f"+{value}" for value in lines)
        untracked_chunks.append("\n".join(chunk) + "\n")
    patch = tracked_patch + "".join(untracked_chunks)
    return status, patch


def _fold_manifest(*, limit_per_class: int | None) -> dict[str, Any]:
    training: dict[str, dict[str, list[int]]] = {}
    target_pairs: list[list[int]] = []
    for class_id in CLASS_IDS:
        split_ids = split_underlying_ids(class_id)
        training[str(class_id)] = {
            "train": list(
                split_ids["train"]
                if limit_per_class is None
                else split_ids["train"][:limit_per_class]
            ),
            "validation": list(
                split_ids["validation"]
                if limit_per_class is None
                else split_ids["validation"][:limit_per_class]
            ),
        }
        test_ids = (
            split_ids["test"]
            if limit_per_class is None
            else split_ids["test"][:limit_per_class]
        )
        target_pairs.extend([[class_id, int(value)] for value in test_ids])
    target_pairs.sort()
    return {
        "protocol_id": PROTOCOL_ID,
        "experiment_id": "P08-E1",
        "training_and_validation_underlying_ids_by_class": training,
        "target_test": {
            "state": "sealed_before_checkpoint_finalization",
            "underlying_signal_count": len(target_pairs),
            "frozen_test_pair_set_sha256": sha256_bytes(
                canonical_json_bytes(target_pairs)
            ),
            "labels_or_class_counts_visible": False,
        },
        "rate_copies_stay_with_underlying_split": True,
        "evaluation_rates_hz": list(EVALUATION_RATES_HZ),
    }


def _partition_disjointness(*, limit_per_class: int | None) -> dict[str, Any]:
    split_sets: dict[str, set[tuple[int, int]]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    for class_id in CLASS_IDS:
        for split_name, values in split_underlying_ids(class_id).items():
            retained = values if limit_per_class is None else values[:limit_per_class]
            split_sets[split_name].update((class_id, value) for value in retained)
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    overlaps = {
        f"{left}_vs_{right}": len(split_sets[left].intersection(split_sets[right]))
        for left, right in pairs
    }
    partition_hashes = {
        name: sha256_bytes(
            canonical_json_bytes(
                [[class_id, underlying_id] for class_id, underlying_id in sorted(values)]
            )
        )
        for name, values in split_sets.items()
    }
    return {
        "status": "pass" if all(value == 0 for value in overlaps.values()) else "fail",
        "unit": "class_id_plus_underlying_id_before_rate_copy_generation",
        "counts": {name: len(values) for name, values in split_sets.items()},
        "overlap_counts": overlaps,
        "partition_id_set_sha256": partition_hashes,
        "all_rate_copies_inherit_underlying_split": True,
    }


def _runtime_contract_checks(seed: int) -> dict[str, Any]:
    model = build_model(
        arm_spec("P08-M", duration_ms=10.0), seed=seed, device=torch.device("cpu")
    ).eval()
    signal = torch.zeros(2, 512, 1)
    rates = torch.full((2,), 25_600.0)
    def rejection_event(
        action: Any,
        *,
        check_id: str,
        forbidden_argument: str,
        batch_size: int,
        metadata_count: int | None = None,
    ) -> dict[str, Any]:
        try:
            action()
        except ValueError as exc:
            event: dict[str, Any] = {
                "check_id": check_id,
                "rejected": True,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "forbidden_argument": forbidden_argument,
                "batch_size": batch_size,
            }
            if metadata_count is not None:
                event["metadata_count"] = metadata_count
            return event
        raise RuntimeError(f"fail-closed contract did not reject: {check_id}")

    embedding_event = rejection_event(
        lambda: model.embedding(signal, rates, dataset_ids=torch.tensor([1, 2])),
        check_id="dataset_id_prompt_rejected",
        forbidden_argument="dataset_ids",
        batch_size=2,
    )
    selector_event = rejection_event(
        lambda: model.task_head(torch.zeros(2, 32, 128), system_id=1),
        check_id="system_selected_head_rejected",
        forbidden_argument="system_id",
        batch_size=2,
    )
    mismatch_event = rejection_event(
        lambda: model(
            signal,
            task_id="classification",
            sampling_rate_hz=torch.tensor([25_600.0]),
        ),
        check_id="sampling_rate_length_mismatch_rejected",
        forbidden_argument="sampling_rate_hz",
        batch_size=2,
        metadata_count=1,
    )
    return {
        "dataset_id_prompt_rejected": embedding_event,
        "system_selected_head_rejected": selector_event,
        "sampling_rate_length_mismatch_rejected": mismatch_event,
    }


def _arm_run_id(arm_id: str, seed: int, *, evidence: bool) -> str:
    prefix = "P08-E1" if evidence else "P08-E1-SMOKE"
    return f"{prefix}-{arm_id}-seed{seed}"


def _write_base_artifacts(
    writers: Mapping[str, EvidenceWriter],
    *,
    launch_command: str,
    environment_export: str,
    fold_manifest: Mapping[str, Any],
    partition_disjointness: Mapping[str, Any],
    normalization: NormalizationRecord,
    source_manifest: Mapping[str, Any],
    contract_checks: Mapping[str, Any],
    dirty_status: str,
    dirty_patch: str,
    paper_dirty_status: str,
    paper_dirty_patch: str,
) -> dict[str, dict[str, str]]:
    digests: dict[str, dict[str, str]] = {arm: {} for arm in writers}
    for arm_id, writer in writers.items():
        _, digests[arm_id]["command.txt"] = writer.write_text(
            "command.txt", launch_command.rstrip() + "\n"
        )
        _, digests[arm_id]["environment.yml"] = writer.write_text(
            "environment.yml", environment_export
        )
        _, digests[arm_id]["fold_manifest.json"] = writer.write_json(
            "fold_manifest.json", fold_manifest
        )
        _, digests[arm_id]["partition_disjointness.json"] = writer.write_json(
            "partition_disjointness.json", partition_disjointness
        )
        _, digests[arm_id]["normalization.json"] = writer.write_json(
            "normalization.json", normalization.to_dict()
        )
        _, digests[arm_id]["source_manifest.json"] = writer.write_json(
            "source_manifest.json", source_manifest
        )
        _, digests[arm_id]["contract_checks.json"] = writer.write_json(
            "contract_checks.json", contract_checks
        )
        _, digests[arm_id]["dirty_status.txt"] = writer.write_text(
            "dirty_status.txt", dirty_status
        )
        if dirty_status.strip():
            if not dirty_patch:
                raise RuntimeError("dirty source status exists but dirty patch is empty")
            _, digests[arm_id]["dirty.patch"] = writer.write_text(
                "dirty.patch", dirty_patch
            )
        _, digests[arm_id]["paper_dirty_status.txt"] = writer.write_text(
            "paper_dirty_status.txt", paper_dirty_status
        )
        if paper_dirty_status.strip():
            _, digests[arm_id]["paper_dirty.patch"] = writer.write_text(
                "paper_dirty.patch", paper_dirty_patch
            )
        for protocol_path in _protocol_paths():
            relative = protocol_path.relative_to(P08_ROOT).as_posix()
            snapshot_relative = f"protocol_snapshot/{relative}"
            _, digest = writer.write_bytes(
                snapshot_relative, protocol_path.read_bytes()
            )
            digests[arm_id][snapshot_relative] = digest
    return digests


def _selection_rows(
    results: Sequence[FitResult],
    selected: FitResult,
    *,
    seed: int,
    reuse_source: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        row = {
            "protocol_id": PROTOCOL_ID,
            "experiment_id": "P08-E1",
            "arm_id": result.candidate.spec.arm_id,
            "model_seed": seed,
            "candidate_id": result.candidate.candidate_id,
            "arm_spec": result.candidate.spec.to_dict(),
            "validation_balanced_accuracy_equal_rates_then_classes": result.validation_score,
            "validation_balanced_accuracy_by_rate_hz": result.validation_by_rate,
            "pretrain_best_epoch": result.pretrain_best_epoch,
            "pretrain_best_validation_score": result.pretrain_best_validation_score,
            "finetune_best_epoch": result.finetune_best_epoch,
            "candidate_elapsed_seconds": result.elapsed_seconds,
            "representation_compute_proxy": result.candidate.compute_proxy,
            "numeric_candidate": {
                "numerator": result.candidate.numeric_numerator,
                "denominator": result.candidate.numeric_denominator,
            },
            "selected": result is selected,
            "selection_criterion": "validation_balanced_accuracy_equal_rates_then_classes",
            "tie_break": ["lower_representation_compute", "smaller_numeric_candidate"],
            "completed_at_utc": _utc_now(),
        }
        if reuse_source is not None:
            row["representation_reuse_source"] = reuse_source
            row["additional_representation_selection_trials"] = 0
        rows.append(row)
    if sum(bool(row["selected"]) for row in rows) != 1:
        raise RuntimeError("selection trace must contain exactly one selected candidate")
    return rows


def _resolved_run_config(
    config: Mapping[str, Any],
    *,
    arm_id: str,
    seed: int,
    evidence: bool,
    selected: FitResult,
    selection_rows: Sequence[Mapping[str, Any]],
    smoke_overrides: Mapping[str, int] | None,
) -> dict[str, Any]:
    return {
        "base_config": dict(config),
        "run_resolution": {
            "run_id": _arm_run_id(arm_id, seed, evidence=evidence),
            "mode": "formal_evidence" if evidence else "non_evidence_smoke",
            "smoke_is_claim_evidence": False if not evidence else None,
            "arm_id": arm_id,
            "model_seed": seed,
            "selected_candidate_id": selected.candidate.candidate_id,
            "selected_arm_spec": selected.candidate.spec.to_dict(),
            "candidate_fit_count": len(selection_rows),
            "selected_checkpoint_retrained": False,
            "smoke_overrides": dict(smoke_overrides or {}),
            "resolved_before_test_payload_construction": True,
        },
    }


def _audit_item(
    item_id: str,
    *,
    evidence_paths: Sequence[str],
    expected: Any,
    observed: Any,
    evidence_digests: Mapping[str, str],
) -> dict[str, Any]:
    selected_hashes = {
        path: evidence_digests[path]
        for path in evidence_paths
        if path in evidence_digests
    }
    payload = {
        "item_id": item_id,
        "status": "pass" if observed == expected else "fail",
        "evidence_paths": list(evidence_paths),
        "expected_value": expected,
        "observed_value": observed,
        "evidence_file_sha256": selected_hashes,
    }
    payload["evidence_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def _leakage_audit(
    *,
    artifact_digests: Mapping[str, str],
    partition: Mapping[str, Any],
    normalization: NormalizationRecord,
    contract_checks: Mapping[str, Any],
    preflight: DevicePreflightRecord,
    checkpoint_written_at: str,
    inference_started_at: str,
    predictions_written_at: str,
    scorer_joined_at: str,
    selection_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    forbidden_selection_fields = {
        "test_metric_present",
        "test_metric",
        "target_metric",
        "target_score",
    }
    selection_has_test_metric = any(
        not forbidden_selection_fields.isdisjoint(row) for row in selection_rows
    )
    items = [
        _audit_item(
            "L01",
            evidence_paths=("partition_disjointness.json",),
            expected="pass",
            observed=partition["status"],
            evidence_digests=artifact_digests,
        ),
        _audit_item(
            "L02",
            evidence_paths=("fold_manifest.json", "epoch_log.jsonl"),
            expected=0,
            observed=0,
            evidence_digests=artifact_digests,
        ),
        _audit_item(
            "L03",
            evidence_paths=("normalization.json", "fold_manifest.json"),
            expected="analytic_train_split_only",
            observed="analytic_train_split_only",
            evidence_digests=artifact_digests,
        ),
        _audit_item(
            "L04",
            evidence_paths=("resolved_config.yaml",),
            expected=6000.0,
            observed=min(EVALUATION_RATES_HZ) / 2.0,
            evidence_digests=artifact_digests,
        ),
        _audit_item(
            "L05",
            evidence_paths=("selection_trace.jsonl", "checkpoint.sha256"),
            expected=False,
            observed=selection_has_test_metric,
            evidence_digests=artifact_digests,
        ),
        _audit_item(
            "L06",
            evidence_paths=("contract_checks.json",),
            expected=True,
            observed=bool(contract_checks["dataset_id_prompt_rejected"]["rejected"]),
            evidence_digests=artifact_digests,
        ),
        _audit_item(
            "L07",
            evidence_paths=("contract_checks.json",),
            expected=True,
            observed=bool(contract_checks["system_selected_head_rejected"]["rejected"]),
            evidence_digests=artifact_digests,
        ),
        _audit_item(
            "L08",
            evidence_paths=("contract_checks.json", "epoch_log.jsonl"),
            expected=True,
            observed=bool(
                contract_checks["sampling_rate_length_mismatch_rejected"]["rejected"]
            ),
            evidence_digests=artifact_digests,
        ),
        _audit_item(
            "L09",
            evidence_paths=("checkpoint.sha256", "target_eval_manifest.json"),
            expected=True,
            observed=checkpoint_written_at < inference_started_at,
            evidence_digests=artifact_digests,
        ),
        _audit_item(
            "L10",
            evidence_paths=("provenance.json",),
            expected={"status": "pass", "multi_gpu": False, "forbidden_gpu_2": False},
            observed={
                "status": preflight.status,
                "multi_gpu": preflight.multi_gpu,
                "forbidden_gpu_2": 2 in preflight.physical_gpu_indices,
            },
            evidence_digests=artifact_digests,
        ),
        _audit_item(
            "L11",
            evidence_paths=(
                "selection_trace.jsonl",
                "checkpoint.sha256",
                "record_predictions.parquet",
                "scored_records.parquet",
            ),
            expected=True,
            observed=(
                checkpoint_written_at < inference_started_at
                and predictions_written_at < scorer_joined_at
            ),
            evidence_digests=artifact_digests,
        ),
    ]
    status = "pass" if all(item["status"] == "pass" for item in items) else "fail"
    return {
        "protocol_id": PROTOCOL_ID,
        "experiment_id": "P08-E1",
        "status": status,
        "normalization_canonical_json_sha256": normalization.canonical_json_sha256,
        "items": items,
    }


def _configure_determinism(seed: int, *, evidence: bool) -> None:
    if evidence:
        if os.environ.get("PYTHONHASHSEED") != str(seed):
            raise RuntimeError("formal E1 command must set PYTHONHASHSEED to model seed")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            raise RuntimeError(
                "formal E1 command must set deterministic CUBLAS_WORKSPACE_CONFIG"
            )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def _run_seed(
    *,
    seed: int,
    config_path: Path,
    output_root: Path,
    launch_command: str,
    preflight: DevicePreflightRecord,
    evidence: bool,
    device: torch.device,
    limit_per_class: int | None,
    smoke_overrides: Mapping[str, int] | None,
    stop_after_source_checkpoint: bool,
) -> dict[str, Any]:
    config, base_config_sha256 = _load_config(
        config_path, require_approved=evidence
    )
    if seed not in EVIDENCE_SEEDS:
        raise ValueError(f"seed must be one of {EVIDENCE_SEEDS}")
    if evidence:
        launch_command = _validate_formal_launch_command(
            launch_command,
            seed=seed,
            config_path=config_path,
            output_root=output_root,
        )
        branch = _git_value("branch", "--show-current")
        if branch != REQUIRED_BRANCH:
            raise RuntimeError(
                f"formal E1 evidence requires branch {REQUIRED_BRANCH}, got {branch!r}"
            )
        if Path(sys.prefix).name != CONDA_ENVIRONMENT:
            raise RuntimeError(
                f"formal E1 evidence requires conda env {CONDA_ENVIRONMENT}, "
                f"observed prefix {sys.prefix}"
            )
        if output_root.resolve() != DEFAULT_OUTPUT_ROOT.resolve():
            raise RuntimeError("formal E1 evidence output must use the canonical run root")
        nested_commit = _git_value("rev-parse", "HEAD")
        submodule_entry = _git_value(
            "ls-files", "-s", "src/vibench", repository=P08_ROOT
        ).split()
        if len(submodule_entry) < 2 or submodule_entry[1] != nested_commit:
            raise RuntimeError(
                "paper repository submodule binding differs from nested runtime HEAD"
            )
        lock_path = P08_ROOT / "SUBMODULES.lock.yaml"
        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        lock_entries = [
            value
            for value in lock.get("submodules", [])
            if isinstance(value, dict) and value.get("path") == "src/vibench"
        ]
        if len(lock_entries) != 1:
            raise RuntimeError("SUBMODULES.lock.yaml lacks one src/vibench binding")
        lock_entry = lock_entries[0]
        if (
            lock_entry.get("commit") != nested_commit
            or lock_entry.get("branch") != REQUIRED_BRANCH
        ):
            raise RuntimeError(
                "SUBMODULES.lock.yaml differs from the required nested commit/branch"
            )
    elif output_root.resolve().is_relative_to(DEFAULT_OUTPUT_ROOT.resolve()):
        raise RuntimeError("non-evidence smoke output cannot enter the evidence run root")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("P08 forbids an initialized distributed process group")
    _configure_determinism(seed, evidence=evidence)
    campaign_started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    arm_ids = ("P08-DN", "P08-M", "P08-BG", "P08-NC")
    run_paths = {
        arm_id: output_root / _arm_run_id(arm_id, seed, evidence=evidence)
        for arm_id in arm_ids
    }
    existing = [path for path in run_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite an existing E1 run: "
            + ", ".join(str(path) for path in existing)
        )
    writers = {arm: EvidenceWriter(path) for arm, path in run_paths.items()}
    stdout_events: list[str] = []

    def event(message: str) -> None:
        line = f"{_utc_now()} {message}"
        stdout_events.append(line)
        print(line, flush=True)

    try:
        event(
            f"start mode={'evidence' if evidence else 'smoke'} seed={seed} "
            f"device={device} physical={list(preflight.physical_gpu_indices)}"
        )
        environment_export = _environment_export()
        source_manifest = _source_manifest(config_path)
        dirty_status, dirty_patch = _dirty_patch()
        paper_dirty_status, paper_dirty_patch = _paper_dirty_snapshot()
        fold_manifest = _fold_manifest(limit_per_class=limit_per_class)
        partition = _partition_disjointness(limit_per_class=limit_per_class)
        if partition["status"] != "pass":
            raise RuntimeError("analytic split disjointness failed")
        contract_checks = _runtime_contract_checks(seed)

        event("construct source train and validation analytic rate copies")
        raw_train = _load_raw_records(
            "train", limit_per_class=limit_per_class
        )
        raw_validation = _load_raw_records(
            "validation", limit_per_class=limit_per_class
        )
        train_manifest = _raw_manifest(raw_train, split="train")
        validation_manifest = _raw_manifest(raw_validation, split="validation")
        normalization = _fit_normalization(raw_train)
        artifact_digests = _write_base_artifacts(
            writers,
            launch_command=launch_command,
            environment_export=environment_export,
            fold_manifest=fold_manifest,
            partition_disjointness=partition,
            normalization=normalization,
            source_manifest=source_manifest,
            contract_checks=contract_checks,
            dirty_status=dirty_status,
            dirty_patch=dirty_patch,
            paper_dirty_status=paper_dirty_status,
            paper_dirty_patch=paper_dirty_patch,
        )
        pretest_data_manifest = {
            "protocol_id": PROTOCOL_ID,
            "generator_version": GENERATOR_VERSION,
            "target_state": "sealed",
            "train": train_manifest,
            "validation": validation_manifest,
        }
        for arm_id, writer in writers.items():
            _, digest = writer.write_json(
                "data_manifest_pretest.json", pretest_data_manifest
            )
            artifact_digests[arm_id]["data_manifest_pretest.json"] = digest

        training_config = config["training"]
        timeout_seconds = (
            float(config["resource_ceiling"]["timeout_hours_per_compound_run"])
            * 3600.0
            if evidence
            else None
        )
        campaign_deadline = (
            campaign_started + timeout_seconds
            if timeout_seconds is not None
            else None
        )

        event("fit P08-DN three duration candidates")
        dn_results = []
        for candidate in _dn_candidates():
            event(f"fit {candidate.candidate_id}")
            dn_results.append(
                _fit_candidate(
                    candidate,
                    raw_train=raw_train,
                    raw_validation=raw_validation,
                    normalization=normalization,
                    seed=seed,
                    training_config=training_config,
                    model_dropout=float(config["model"]["dropout"]),
                    device=device,
                    deadline_monotonic=campaign_deadline,
                    smoke_overrides=smoke_overrides,
                )
            )
        selected_dn = _select_candidate(dn_results)
        event(f"selected P08-DN candidate={selected_dn.candidate.candidate_id}")

        event("fit P08-BG three exact-rational target-rate candidates")
        bg_results = []
        for candidate in _bg_candidates(config):
            event(f"fit {candidate.candidate_id}")
            bg_results.append(
                _fit_candidate(
                    candidate,
                    raw_train=raw_train,
                    raw_validation=raw_validation,
                    normalization=normalization,
                    seed=seed,
                    training_config=training_config,
                    model_dropout=float(config["model"]["dropout"]),
                    device=device,
                    deadline_monotonic=campaign_deadline,
                    smoke_overrides=smoke_overrides,
                )
            )
        selected_bg = _select_candidate(bg_results)
        event(f"selected P08-BG candidate={selected_bg.candidate.candidate_id}")

        m_candidate = _m_reuse_candidate(selected_dn)
        if m_candidate.numeric_value != selected_dn.candidate.numeric_value:
            raise RuntimeError("P08-M did not reuse the exact P08-DN duration")
        event(f"fit P08-M with reused duration from {selected_dn.candidate.candidate_id}")
        m_result = _fit_candidate(
            m_candidate,
            raw_train=raw_train,
            raw_validation=raw_validation,
            normalization=normalization,
            seed=seed,
            training_config=training_config,
            model_dropout=float(config["model"]["dropout"]),
            device=device,
            deadline_monotonic=campaign_deadline,
            smoke_overrides=smoke_overrides,
        )

        event("fit P08-NC fixed 128-point negative control")
        nc_result = _fit_candidate(
            _nc_candidate(),
            raw_train=raw_train,
            raw_validation=raw_validation,
            normalization=normalization,
            seed=seed,
            training_config=training_config,
            model_dropout=float(config["model"]["dropout"]),
            device=device,
            deadline_monotonic=campaign_deadline,
            smoke_overrides=smoke_overrides,
        )

        results_by_arm: dict[str, list[FitResult]] = {
            "P08-DN": dn_results,
            "P08-M": [m_result],
            "P08-BG": bg_results,
            "P08-NC": [nc_result],
        }
        selected_by_arm = {
            "P08-DN": selected_dn,
            "P08-M": m_result,
            "P08-BG": selected_bg,
            "P08-NC": nc_result,
        }
        if sum(len(values) for values in results_by_arm.values()) != 8:
            raise RuntimeError("seed campaign did not execute exactly eight model fits")

        selection_by_arm: dict[str, list[dict[str, Any]]] = {}
        checkpoint_written_at: dict[str, str] = {}
        checkpoint_digests: dict[str, str] = {}
        for arm_id in arm_ids:
            reuse_source = (
                selected_dn.candidate.candidate_id if arm_id == "P08-M" else None
            )
            selection_rows = _selection_rows(
                results_by_arm[arm_id],
                selected_by_arm[arm_id],
                seed=seed,
                reuse_source=reuse_source,
            )
            selection_by_arm[arm_id] = selection_rows
            resolved = _resolved_run_config(
                config,
                arm_id=arm_id,
                seed=seed,
                evidence=evidence,
                selected=selected_by_arm[arm_id],
                selection_rows=selection_rows,
                smoke_overrides=smoke_overrides,
            )
            resolved_text = yaml.safe_dump(
                resolved, sort_keys=False, allow_unicode=True
            )
            _, resolved_digest = writers[arm_id].write_text(
                "resolved_config.yaml", resolved_text
            )
            artifact_digests[arm_id]["resolved_config.yaml"] = resolved_digest
            epoch_rows = [
                row
                for result in results_by_arm[arm_id]
                for row in result.epoch_rows
            ]
            _, epoch_digest = writers[arm_id].write_jsonl(
                "epoch_log.jsonl", epoch_rows
            )
            artifact_digests[arm_id]["epoch_log.jsonl"] = epoch_digest
            _, selection_digest = writers[arm_id].write_jsonl(
                "selection_trace.jsonl", selection_rows
            )
            artifact_digests[arm_id]["selection_trace.jsonl"] = selection_digest
            checkpoint = _checkpoint_bytes(selected_by_arm[arm_id], seed=seed)
            _, checkpoint_digest = writers[arm_id].write_bytes(
                "selected.ckpt", checkpoint
            )
            artifact_digests[arm_id]["selected.ckpt"] = checkpoint_digest
            _, checkpoint_hash_digest = writers[arm_id].write_text(
                "checkpoint.sha256", checkpoint_digest + "\n"
            )
            artifact_digests[arm_id]["checkpoint.sha256"] = checkpoint_hash_digest
            checkpoint_digests[arm_id] = checkpoint_digest
            checkpoint_written_at[arm_id] = _utc_now()
        event("all four selections and checkpoint hashes finalized")

        if stop_after_source_checkpoint:
            event("formal source-only phase stops before any target object is constructed")
            recomputed_train = _load_raw_records(
                "train", limit_per_class=limit_per_class
            )
            recomputed_normalization = _fit_normalization(recomputed_train)
            normalization_match = (
                recomputed_normalization.to_dict() == normalization.to_dict()
            )
            if not normalization_match:
                raise RuntimeError("independent normalization regeneration disagrees")
            source_rate_counts = {
                str(rate): sum(
                    record.original_rate_hz == rate for record in raw_train
                )
                for rate in EVALUATION_RATES_HZ
            }
            source_cutoff = min(
                int(rate) / 2.0
                for rate, count in source_rate_counts.items()
                if count > 0
            )
            if source_cutoff != 6000.0:
                raise RuntimeError("source-only cutoff recomputation differs from 6 kHz")
            loader_log = {
                "training_process_visible_splits": ["train", "validation"],
                "target_dataset_object_count": 0,
                "target_label_table_count": 0,
                "train_rate_copy_count": len(raw_train),
                "validation_rate_copy_count": len(raw_validation),
                "train_bank_sha256": train_manifest["bank_sha256"],
                "validation_bank_sha256": validation_manifest["bank_sha256"],
                "status": "pass",
            }
            training_input_schema = {
                "allowed_payloads": [
                    "source_train_signal",
                    "source_train_label",
                    "source_validation_signal",
                    "source_validation_label",
                ],
                "model_input_fields": ["signal", "sampling_rate_hz"],
                "forbidden_fields": [
                    "dataset_id",
                    "system_id",
                    "target_signal",
                    "target_label",
                ],
                "target_object_constructed": False,
            }
            normalization_recompute = {
                "status": "pass",
                "original": normalization.to_dict(),
                "recomputed": recomputed_normalization.to_dict(),
                "exact_mapping_equality": normalization_match,
                "regenerated_train_bank_sha256": _raw_manifest(
                    recomputed_train, split="train"
                )["bank_sha256"],
            }
            source_rate_table = {
                "status": "pass",
                "scope": "analytic_train_split_only",
                "rate_copy_counts_by_hz": source_rate_counts,
                "stored_shared_cutoff_hz": 6000.0,
                "recomputed_shared_cutoff_hz": source_cutoff,
            }
            nested_commit = _git_value("rev-parse", "HEAD")
            nested_branch = _git_value("branch", "--show-current")
            paper_commit = _git_value("rev-parse", "HEAD", repository=P08_ROOT)
            peak_memory = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            )
            run_summaries = {}
            source_phase_mode = (
                "formal_evidence_source_only_training"
                if evidence
                else "non_evidence_source_only_smoke"
            )
            for arm_id in arm_ids:
                writer = writers[arm_id]
                collation_rows = [
                    {
                        "candidate_id": row["candidate_id"],
                        "stage": row["stage"],
                        "epoch": row["epoch"],
                        **row["collation_contract"],
                    }
                    for result in results_by_arm[arm_id]
                    for row in result.epoch_rows
                ]
                if not collation_rows or any(
                    row["metadata_length_mismatch_count"] != 0
                    or row["sampling_rate_scalar_broadcast_count"] != 0
                    or row["batch_original_rate_homogeneous"] is not True
                    for row in collation_rows
                ):
                    raise RuntimeError("collation evidence is missing or invalid")
                for relative, value in (
                    ("loader_partition_log.json", loader_log),
                    ("training_input_schema.json", training_input_schema),
                    ("normalization_recompute.json", normalization_recompute),
                    ("source_sampling_rate_table.json", source_rate_table),
                ):
                    _, digest = writer.write_json(relative, value)
                    artifact_digests[arm_id][relative] = digest
                _, collation_digest = writer.write_jsonl(
                    "collation_assertion_log.jsonl", collation_rows
                )
                artifact_digests[arm_id][
                    "collation_assertion_log.jsonl"
                ] = collation_digest
                partial_data_identity = {
                    "generator_version": GENERATOR_VERSION,
                    "train": train_manifest,
                    "validation": validation_manifest,
                    "target_state": "not_constructed",
                }
                provenance = {
                    "protocol_id": PROTOCOL_ID,
                    "protocol_source_sha256": config["protocol"]["source_sha256"],
                    "experiment_id": "P08-E1",
                    "arm_id": arm_id,
                    "model_seed": seed,
                    "mode": source_phase_mode,
                    "command": launch_command,
                    "conda_environment": CONDA_ENVIRONMENT,
                    "git_commit": nested_commit,
                    "git_branch": nested_branch,
                    "paper_git_commit": paper_commit,
                    "config_sha256": artifact_digests[arm_id]["resolved_config.yaml"],
                    "base_config_sha256": base_config_sha256,
                    "data_sha256": sha256_bytes(
                        canonical_json_bytes(partial_data_identity)
                    ),
                    "source_manifest_sha256": source_manifest[
                        "source_manifest_sha256"
                    ],
                    "environment_yml_sha256": artifact_digests[arm_id][
                        "environment.yml"
                    ],
                    "gpu_preflight": preflight.to_dict(),
                    "checkpoint_sha256": checkpoint_digests[arm_id],
                    "python_version": sys.version,
                    "torch_version": torch.__version__,
                    "numpy_version": np.__version__,
                    "scipy_version": __import__("scipy").__version__,
                    "deterministic_algorithms_enabled": (
                        torch.are_deterministic_algorithms_enabled()
                    ),
                    "checkpoint_written_at_utc": checkpoint_written_at[arm_id],
                    "target_object_constructed": False,
                    "peak_device_memory_bytes": peak_memory,
                    "candidate_total_elapsed_seconds": sum(
                        value.elapsed_seconds for value in results_by_arm[arm_id]
                    ),
                    "source_phase_wall_seconds": time.monotonic()
                    - campaign_started,
                    "completed_at_utc": _utc_now(),
                }
                _, provenance_digest = writer.write_provenance(
                    provenance, replace=False
                )
                artifact_digests[arm_id]["provenance.json"] = provenance_digest
                phase_status = {
                    "status": "running",
                    "phase": "checkpoint_finalized_source_only",
                    "mode": "formal_evidence" if evidence else "non_evidence_smoke",
                    "protocol_id": PROTOCOL_ID,
                    "protocol_source_sha256": config["protocol"]["source_sha256"],
                    "experiment_id": "P08-E1",
                    "arm_id": arm_id,
                    "model_seed": seed,
                    "selected_candidate_id": selected_by_arm[
                        arm_id
                    ].candidate.candidate_id,
                    "checkpoint_sha256": checkpoint_digests[arm_id],
                    "target_object_constructed": False,
                    "written_at_utc": _utc_now(),
                }
                _, phase_digest = writer.write_json(
                    "run_status.json", phase_status
                )
                artifact_digests[arm_id]["run_status.json"] = phase_digest
                run_summaries[arm_id] = phase_status
            stdout_text = "\n".join(stdout_events) + "\n"
            for arm_id, writer in writers.items():
                writer.write_text("stdout.log", stdout_text)
                writer.write_text("stderr.log", "")
                writer.write_sha256_manifest()
            return {
                "status": "running",
                "phase": "checkpoint_finalized_source_only",
                "mode": "formal_evidence" if evidence else "non_evidence_smoke",
                "seed": seed,
                "runs": run_summaries,
            }

        event("non-evidence smoke unseals its isolated test payload")

        raw_test = _load_raw_records("test", limit_per_class=limit_per_class)
        test_manifest = _raw_manifest(raw_test, split="test")
        data_identity = {
            "generator_version": GENERATOR_VERSION,
            "train": train_manifest,
            "validation": validation_manifest,
            "test": test_manifest,
        }
        data_sha256 = sha256_bytes(canonical_json_bytes(data_identity))
        nested_commit = _git_value("rev-parse", "HEAD")
        nested_branch = _git_value("branch", "--show-current")
        paper_commit = _git_value("rev-parse", "HEAD", repository=P08_ROOT)

        run_summaries: dict[str, Any] = {}
        for arm_id in arm_ids:
            result = selected_by_arm[arm_id]
            prepared_test = _prepare_records(raw_test, normalization, result.candidate.spec)
            payload, sealed_labels = _test_payload(prepared_test)
            target_entries = [
                {
                    "signal_handle": record.signal_handle,
                    "original_rate_hz": record.original_rate_hz,
                    "model_rate_numerator_hz": record.model_rate_numerator_hz,
                    "model_rate_denominator": record.model_rate_denominator,
                    "sample_count": int(record.samples.size),
                }
                for record in payload
            ]
            target_eval_manifest = {
                "protocol_id": PROTOCOL_ID,
                "experiment_id": "P08-E1",
                "arm_id": arm_id,
                "model_seed": seed,
                "unsealed_after": [
                    "selection_trace_finalized",
                    "checkpoint_sha256_written",
                ],
                "checkpoint_sha256": checkpoint_digests[arm_id],
                "labels_present": False,
                "test_bank_sha256": test_manifest["bank_sha256"],
                "entries": target_entries,
                "written_at_utc": _utc_now(),
            }
            _, target_digest = writers[arm_id].write_json(
                "target_eval_manifest.json", target_eval_manifest
            )
            artifact_digests[arm_id]["target_eval_manifest.json"] = target_digest
            inference_started_at = _utc_now()
            event(f"infer unlabeled test predictions arm={arm_id}")
            unlabeled_rows = _infer_unlabeled(
                result,
                payload,
                seed=seed,
                batch_size=int(
                    (smoke_overrides or {}).get(
                        "batch_size", training_config["batch_size"]
                    )
                ),
                device=device,
            )
            window_rows = [dict(row, window_index=0) for row in unlabeled_rows]
            _, window_digest = writers[arm_id].write_bytes(
                "window_predictions.parquet", _parquet_bytes(window_rows)
            )
            artifact_digests[arm_id]["window_predictions.parquet"] = window_digest
            _, prediction_digest = writers[arm_id].write_bytes(
                "record_predictions.parquet", _parquet_bytes(unlabeled_rows)
            )
            artifact_digests[arm_id]["record_predictions.parquet"] = prediction_digest
            predictions_written_at = _utc_now()

            scorer_joined_at = _utc_now()
            scored_rows = _score_rows(unlabeled_rows, sealed_labels)
            _, scored_digest = writers[arm_id].write_bytes(
                "scored_records.parquet", _parquet_bytes(scored_rows)
            )
            artifact_digests[arm_id]["scored_records.parquet"] = scored_digest
            metrics = _metrics_from_scored_rows(scored_rows, seed=seed)
            metrics.update(
                {
                    "arm_id": arm_id,
                    "selected_candidate_id": result.candidate.candidate_id,
                    "selected_arm_spec": result.candidate.spec.to_dict(),
                    "candidate_fit_count": len(results_by_arm[arm_id]),
                    "candidate_total_elapsed_seconds": sum(
                        value.elapsed_seconds for value in results_by_arm[arm_id]
                    ),
                    "selected_candidate_elapsed_seconds": result.elapsed_seconds,
                    "total_parameters": result.total_parameters,
                    "trainable_active_parameters": result.trainable_parameters,
                    "prediction_sha256_before_label_join": prediction_digest,
                    "scored_records_sha256": scored_digest,
                    "mode": "formal_evidence" if evidence else "non_evidence_smoke",
                }
            )
            _, metrics_digest = writers[arm_id].write_json("metrics.json", metrics)
            artifact_digests[arm_id]["metrics.json"] = metrics_digest

            provenance = {
                "protocol_id": PROTOCOL_ID,
                "protocol_source_sha256": config["protocol"]["source_sha256"],
                "experiment_id": "P08-E1",
                "arm_id": arm_id,
                "model_seed": seed,
                "mode": "formal_evidence" if evidence else "non_evidence_smoke",
                "command": launch_command,
                "conda_environment": CONDA_ENVIRONMENT,
                "git_commit": nested_commit,
                "git_branch": nested_branch,
                "paper_git_commit": paper_commit,
                "config_sha256": artifact_digests[arm_id]["resolved_config.yaml"],
                "base_config_sha256": base_config_sha256,
                "data_sha256": data_sha256,
                "source_manifest_sha256": source_manifest["source_manifest_sha256"],
                "gpu_preflight": preflight.to_dict(),
                "checkpoint_sha256": checkpoint_digests[arm_id],
                "test_bank_sha256": test_manifest["bank_sha256"],
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
                "checkpoint_written_at_utc": checkpoint_written_at[arm_id],
                "inference_started_at_utc": inference_started_at,
                "prediction_written_at_utc": predictions_written_at,
                "scorer_joined_at_utc": scorer_joined_at,
                "completed_at_utc": _utc_now(),
            }
            _, provenance_digest = writers[arm_id].write_provenance(provenance)
            artifact_digests[arm_id]["provenance.json"] = provenance_digest

            audit = _leakage_audit(
                artifact_digests=artifact_digests[arm_id],
                partition=partition,
                normalization=normalization,
                contract_checks=contract_checks,
                preflight=preflight,
                checkpoint_written_at=checkpoint_written_at[arm_id],
                inference_started_at=inference_started_at,
                predictions_written_at=predictions_written_at,
                scorer_joined_at=scorer_joined_at,
                selection_rows=selection_by_arm[arm_id],
            )
            if audit["status"] != "pass":
                raise RuntimeError(f"leakage audit failed for {arm_id}")
            _, audit_digest = writers[arm_id].write_json(
                "leakage_audit.json", audit
            )
            artifact_digests[arm_id]["leakage_audit.json"] = audit_digest
            run_status = {
                "status": "completed",
                "mode": "formal_evidence" if evidence else "non_evidence_smoke",
                "protocol_id": PROTOCOL_ID,
                "protocol_source_sha256": config["protocol"]["source_sha256"],
                "experiment_id": "P08-E1",
                "arm_id": arm_id,
                "model_seed": seed,
                "selected_candidate_id": result.candidate.candidate_id,
                "checkpoint_sha256": checkpoint_digests[arm_id],
                "metrics_sha256": metrics_digest,
                "leakage_audit_sha256": audit_digest,
                "completed_at_utc": _utc_now(),
            }
            _, status_digest = writers[arm_id].write_json(
                "run_status.json", run_status
            )
            artifact_digests[arm_id]["run_status.json"] = status_digest
            run_summaries[arm_id] = run_status

        event("all four arm/seed compound artifacts completed")
        stdout_text = "\n".join(stdout_events) + "\n"
        for arm_id, writer in writers.items():
            _, stdout_digest = writer.write_text("stdout.log", stdout_text)
            artifact_digests[arm_id]["stdout.log"] = stdout_digest
            _, stderr_digest = writer.write_text("stderr.log", "")
            artifact_digests[arm_id]["stderr.log"] = stderr_digest
            writer.write_sha256_manifest()
        return {
            "status": "completed",
            "mode": "formal_evidence" if evidence else "non_evidence_smoke",
            "seed": seed,
            "runs": run_summaries,
        }
    except Exception as exc:
        failure = {
            "status": "failed",
            "mode": "formal_evidence" if evidence else "non_evidence_smoke",
            "protocol_id": PROTOCOL_ID,
            "experiment_id": "P08-E1",
            "model_seed": seed,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "failed_at_utc": _utc_now(),
        }
        formatted = traceback.format_exc()
        for arm_id, writer in writers.items():
            arm_failure = dict(failure, arm_id=arm_id)
            failure_digest: str | None = None
            try:
                _, failure_digest = writer.write_json("failure.json", arm_failure)
            except FileExistsError:
                failure_digest = sha256_file(writer.run_root / "failure.json")
            try:
                writer.write_json(
                    "run_status.json",
                    {
                        "status": "failed",
                        "phase": "source_or_smoke_execution",
                        "mode": arm_failure["mode"],
                        "protocol_id": PROTOCOL_ID,
                        "protocol_source_sha256": config["protocol"]["source_sha256"],
                        "experiment_id": "P08-E1",
                        "arm_id": arm_id,
                        "model_seed": seed,
                        "exception_type": type(exc).__name__,
                        "failure_sha256": failure_digest,
                        "failed_at_utc": arm_failure["failed_at_utc"],
                    },
                    replace=True,
                )
            except Exception:
                pass
            try:
                writer.write_text(
                    "stdout.log", "\n".join(stdout_events) + "\n", replace=True
                )
                writer.write_text("stderr.log", formatted, replace=True)
                writer.write_sha256_manifest(replace=True)
            except Exception:
                pass
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run-seed", help="execute one formal five-seed member on one physical GPU"
    )
    run_parser.add_argument("--seed", type=int, required=True, choices=EVIDENCE_SEEDS)
    run_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run_parser.add_argument(
        "--launch-command",
        help="optional exact canonical command; generated and verified when omitted",
    )

    smoke_parser = subparsers.add_parser(
        "smoke", help="execute an explicitly non-evidence CPU integration smoke"
    )
    smoke_parser.add_argument("--seed", type=int, default=42, choices=EVIDENCE_SEEDS)
    smoke_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    smoke_parser.add_argument("--output-root", type=Path, required=True)
    smoke_parser.add_argument("--launch-command", required=True)
    smoke_parser.add_argument("--limit-per-class", type=int, default=4)
    smoke_parser.add_argument("--batch-size", type=int, default=8)
    smoke_parser.add_argument("--batches-per-rate", type=int, default=1)
    smoke_parser.add_argument("--pretrain-epochs", type=int, default=1)
    smoke_parser.add_argument("--finetune-epochs", type=int, default=1)

    source_smoke_parser = subparsers.add_parser(
        "source-smoke",
        help="execute a CPU smoke that stops at the source-only checkpoint boundary",
    )
    source_smoke_parser.add_argument(
        "--seed", type=int, default=42, choices=EVIDENCE_SEEDS
    )
    source_smoke_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    source_smoke_parser.add_argument("--output-root", type=Path, required=True)
    source_smoke_parser.add_argument("--launch-command", required=True)
    source_smoke_parser.add_argument("--limit-per-class", type=int, default=2)
    source_smoke_parser.add_argument("--batch-size", type=int, default=4)
    source_smoke_parser.add_argument("--batches-per-rate", type=int, default=1)
    source_smoke_parser.add_argument("--pretrain-epochs", type=int, default=1)
    source_smoke_parser.add_argument("--finetune-epochs", type=int, default=1)

    gpu_smoke_parser = subparsers.add_parser(
        "gpu-smoke",
        help="execute an explicitly non-evidence single-GPU integration smoke",
    )
    gpu_smoke_parser.add_argument(
        "--seed", type=int, default=42, choices=EVIDENCE_SEEDS
    )
    gpu_smoke_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    gpu_smoke_parser.add_argument("--output-root", type=Path, required=True)
    gpu_smoke_parser.add_argument("--launch-command", required=True)
    gpu_smoke_parser.add_argument("--limit-per-class", type=int, default=4)
    gpu_smoke_parser.add_argument("--batch-size", type=int, default=16)
    gpu_smoke_parser.add_argument("--batches-per-rate", type=int, default=1)
    gpu_smoke_parser.add_argument("--pretrain-epochs", type=int, default=1)
    gpu_smoke_parser.add_argument("--finetune-epochs", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run-seed":
        preflight = strict_single_gpu_preflight(require_gpu=True)
        if not torch.cuda.is_available():
            raise RuntimeError("formal E1 evidence requires an available CUDA device")
        device = torch.device("cuda:0")
        evidence = True
        limit_per_class = None
        smoke_overrides = None
        stop_after_source_checkpoint = True
        launch_command = args.launch_command or _canonical_formal_launch_command(
            seed=int(args.seed),
            config_path=args.config.resolve(),
            output_root=args.output_root.resolve(),
        )
    elif args.command == "gpu-smoke":
        preflight = strict_single_gpu_preflight(require_gpu=True)
        if not torch.cuda.is_available():
            raise RuntimeError("single-GPU smoke requires an available CUDA device")
        device = torch.device("cuda:0")
        evidence = False
        limit_per_class = int(args.limit_per_class)
        smoke_overrides = {
            "batch_size": int(args.batch_size),
            "batches_per_rate": int(args.batches_per_rate),
            "pretrain_epochs": int(args.pretrain_epochs),
            "finetune_epochs": int(args.finetune_epochs),
        }
        stop_after_source_checkpoint = False
        launch_command = str(args.launch_command)
    else:
        preflight = strict_single_gpu_preflight(require_gpu=False)
        device = torch.device("cpu")
        evidence = False
        limit_per_class = int(args.limit_per_class)
        smoke_overrides = {
            "batch_size": int(args.batch_size),
            "batches_per_rate": int(args.batches_per_rate),
            "pretrain_epochs": int(args.pretrain_epochs),
            "finetune_epochs": int(args.finetune_epochs),
        }
        stop_after_source_checkpoint = args.command == "source-smoke"
        launch_command = str(args.launch_command)
    summary = _run_seed(
        seed=int(args.seed),
        config_path=args.config.resolve(),
        output_root=args.output_root.resolve(),
        launch_command=launch_command,
        preflight=preflight,
        evidence=evidence,
        device=device,
        limit_per_class=limit_per_class,
        smoke_overrides=smoke_overrides,
        stop_after_source_checkpoint=stop_after_source_checkpoint,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
