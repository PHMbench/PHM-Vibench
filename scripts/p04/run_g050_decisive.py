#!/usr/bin/env python3
"""Run the exact P04-G050 CWRU pilot without the retired sealed pipeline.

This direct entry point owns one frozen path only:

    CWRU12K-DE-v1 -> P0/P1/P2 x seeds 20/21/22 -> blind matching
    -> recovered-role knockout -> frozen decision.

It deliberately does not provide fallback data discovery, automatic metadata
repair, CPU fallback, a sweep interface, or compatibility with the older
FULL/HOMO/RAND synthetic protocol.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import openpyxl
import scipy
import torch
import torch.nn.functional as F
import yaml
from scipy.io import loadmat, whosmat
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model_factory.MoE.M_04_RoleConstrainedMoE import Model  # noqa: E402


ARMS = ("P0", "P1", "P2")
SEEDS = (20, 21, 22)
ROLE_IDS = (0, 1, 2, 3)
PARTITIONS = ("train", "validation", "P_match", "P_eval")
G050_CONFIG_PATH = REPO_ROOT / "configs/experiments/p04/g050_decisive.yaml"


@dataclass(frozen=True)
class Source:
    file_id: int
    metadata_id: str
    source_id: str
    label: int
    domain: int
    partition: str
    sample_rate_hz: float
    rotation_speed_rpm: float
    load_hp: float
    raw_rpm: float | None
    signal_length: int
    signal_key: str
    window_starts: tuple[int, ...]


@dataclass
class PartitionData:
    x: torch.Tensor
    y: torch.Tensor
    file_id: torch.Tensor
    source_id: list[str]
    sample_rate_hz: torch.Tensor
    rotation_speed_rpm: torch.Tensor
    load_hp: torch.Tensor

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def physical(self, indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "sample_rate_hz": self.sample_rate_hz[indices].to(device),
            "rotation_speed_rpm": self.rotation_speed_rpm[indices].to(device),
            "load_hp": self.load_hp[indices].to(device),
        }


@dataclass
class AdmittedData:
    partitions: dict[str, PartitionData]
    sources: list[Source]
    normalization_mean: float
    normalization_std: float
    contract: dict[str, Any]


def _plain(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _plain(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON artifacts cannot contain NaN or infinity")
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _plain(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config does not exist: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("G050 config must be one YAML mapping")
    if tuple(config["protocol"]["arms"]) != ARMS:
        raise ValueError(f"frozen arms must be {ARMS}")
    if tuple(int(seed) for seed in config["protocol"]["seeds"]) != SEEDS:
        raise ValueError(f"frozen seeds must be {SEEDS}")
    if config["protocol"]["contract_id"] != "P04-CWRU12K-DE-v1":
        raise ValueError("unknown data contract")
    if config["data"]["signal_key_fallback"] != "forbidden":
        raise ValueError("signal-key fallback must remain forbidden")
    if config["data"]["normalization"]["per_window_refit"] != "forbidden":
        raise ValueError("per-window normalization must remain forbidden")
    if int(config["training"]["visible_physical_gpu"]) != 5:
        raise ValueError("this authorized run is frozen to physical GPU 5")
    if config["training"]["cublas_workspace_config"] != ":4096:8":
        raise ValueError("deterministic cuBLAS workspace contract changed")
    speed = config["data"].get("speed", {})
    if speed.get("mode") != "nominal_by_domain":
        raise ValueError("speed.mode must explicitly remain nominal_by_domain")
    if speed.get("raw_rpm_use") != "diagnostic_only":
        raise ValueError("raw RPM must remain diagnostic_only under this contract")
    match_probes = list(config["probes"]["match"])
    eval_probes = list(config["probes"]["eval"])
    controls = list(config["probes"]["controls"])
    match_ids = [str(probe["id"]) for probe in match_probes]
    eval_ids = [str(probe["id"]) for probe in eval_probes]
    control_ids = [str(probe["id"]) for probe in controls]
    if len(set(match_ids)) != len(match_ids) or len(set(eval_ids)) != len(eval_ids):
        raise ValueError("probe IDs must be unique within P_match and P_eval")
    if set(match_ids) & set(eval_ids):
        raise ValueError("P_match and P_eval probe IDs must be disjoint")
    if (set(match_ids) | set(eval_ids)) & set(control_ids):
        raise ValueError("control probe IDs must be disjoint from role probe IDs")
    for name, probes in (("P_match", match_probes), ("P_eval", eval_probes)):
        role_counts = {
            role: sum(int(probe["role_id"]) == role for probe in probes)
            for role in ROLE_IDS
        }
        if role_counts != {role: 2 for role in ROLE_IDS}:
            raise ValueError(f"{name} must contain exactly two probes per role")
    def transform_spec(probe: Mapping[str, Any]) -> str:
        return json.dumps(
            {key: value for key, value in probe.items() if key not in {"id", "role_id"}},
            sort_keys=True,
        )
    if {transform_spec(probe) for probe in match_probes} & {
        transform_spec(probe) for probe in eval_probes
    }:
        raise ValueError("P_match and P_eval contain an identical transform specification")
    return config


def _require_gpu5() -> torch.device:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "5":
        raise RuntimeError(
            "set CUDA_VISIBLE_DEVICES=5 exactly; automatic GPU selection is forbidden"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launch to preserve the "
            "frozen deterministic CUDA contract"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 5 is unavailable; CPU fallback is forbidden")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"exactly one visible GPU is required, observed {torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    return device


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime_git_provenance() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        return completed.stdout.strip()

    changed = git("status", "--porcelain=v1").splitlines()
    return {
        "root": str(REPO_ROOT),
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(changed),
        "changed_paths": changed,
    }


def _run_semantic_gate_tests() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "test/test_p04_g050_semantics.py",
        "test/test_p04_g050_data_contract.py",
        "test/test_p04_role_constrained_moe.py",
        "test/test_p04_g050_runner.py",
        "test/test_per_sample_metadata.py",
        "test/test_transform_truth.py",
    ]
    started = _utc_now()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )
    combined = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    matches = re.findall(r"(\d+) passed", combined)
    pass_count = int(matches[-1]) if matches else 0
    return {
        "status": "passed" if completed.returncode == 0 and pass_count > 0 else "failed",
        "started_at": started,
        "completed_at": _utc_now(),
        "command": command,
        "exit_code": completed.returncode,
        "observed_pass_count": pass_count,
        "output_tail": combined.splitlines()[-20:],
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)


def _metadata_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"metadata workbook does not exist: {path}")
    sheet = openpyxl.load_workbook(path, read_only=True, data_only=True)["Sheet1"]
    iterator = sheet.iter_rows(values_only=True)
    headers = next(iterator)
    if any(value is None for value in headers):
        raise ValueError("metadata workbook contains empty header cells")
    names = [str(value) for value in headers]
    required = {
        "Id",
        "Dataset_id",
        "Name",
        "File",
        "Label",
        "Domain_id",
        "Sample_rate",
        "Sample_lenth",
    }
    missing = required - set(names)
    if missing:
        raise ValueError("metadata workbook is missing: " + ", ".join(sorted(missing)))
    rows: dict[str, dict[str, Any]] = {}
    for values in iterator:
        row = dict(zip(names, values))
        filename = str(row.get("File", ""))
        if filename in rows:
            raise ValueError(f"metadata contains duplicate File row: {filename}")
        rows[filename] = row
    return rows


def _exact_scalar(value: Any, field: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


def _window_starts(length: int, window: int, count: int) -> tuple[int, ...]:
    if count < 2 or length < window * count:
        raise ValueError(
            f"source length {length} cannot provide {count} non-overlapping windows of {window}"
        )
    starts = tuple(math.floor(index * (length - window) / (count - 1)) for index in range(count))
    if len(set(starts)) != count:
        raise ValueError("window start formula produced duplicate starts")
    if starts[0] != 0 or starts[-1] + window != length:
        raise ValueError("window start formula no longer spans the exact recording boundary")
    if any(right - left < window for left, right in zip(starts, starts[1:])):
        raise ValueError("window contract produced overlapping windows")
    return starts


def _load_admitted_data(config: Mapping[str, Any], raw_root_override: Path | None) -> AdmittedData:
    data_config = config["data"]
    raw_root = (raw_root_override or Path(data_config["raw_root"])).resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw CWRU root does not exist: {raw_root}")
    metadata_path = Path(data_config["metadata_path"])
    if not metadata_path.is_absolute():
        metadata_path = (REPO_ROOT / metadata_path).resolve()
    rows = _metadata_rows(metadata_path)
    expected_sources = list(data_config["sources"])
    if len(expected_sources) != 16:
        raise ValueError("CWRU contract must contain exactly 16 sources")
    file_ids = [int(entry["file_id"]) for entry in expected_sources]
    if len(set(file_ids)) != 16:
        raise ValueError("CWRU source allow-list contains duplicate files")

    window_length = int(data_config["windowing"]["length_samples"])
    windows_per_source = int(data_config["windowing"]["windows_per_source"])
    expected_rate = int(data_config["sample_rate_hz"])
    label_map = {int(key): str(value) for key, value in data_config["label_map"].items()}
    if set(label_map) != set(ROLE_IDS):
        raise ValueError("label map must resolve exactly labels 0, 1, 2, 3")
    rpm_by_domain = {
        int(key): float(value)
        for key, value in data_config["nominal_rpm_by_domain"].items()
    }
    partition_by_domain = {
        int(key): str(value)
        for key, value in data_config["partition_by_domain"].items()
    }
    if set(partition_by_domain.values()) != set(PARTITIONS):
        raise ValueError("domain partition map must define train/validation/P_match/P_eval")

    source_records: list[Source] = []
    partition_windows: dict[str, list[np.ndarray]] = {name: [] for name in PARTITIONS}
    partition_labels: dict[str, list[int]] = {name: [] for name in PARTITIONS}
    partition_files: dict[str, list[int]] = {name: [] for name in PARTITIONS}
    partition_source_ids: dict[str, list[str]] = {name: [] for name in PARTITIONS}
    partition_rates: dict[str, list[float]] = {name: [] for name in PARTITIONS}
    partition_speeds: dict[str, list[float]] = {name: [] for name in PARTITIONS}
    partition_loads: dict[str, list[float]] = {name: [] for name in PARTITIONS}

    for declared in expected_sources:
        file_id = int(declared["file_id"])
        label = int(declared["label"])
        domain = int(declared["domain"])
        filename = f"{file_id}.mat"
        if filename not in rows:
            raise ValueError(f"metadata row is missing for admitted source {filename}")
        row = rows[filename]
        observed = {
            "Dataset_id": int(row["Dataset_id"]),
            "Name": str(row["Name"]),
            "Label": int(row["Label"]),
            "Domain_id": int(row["Domain_id"]),
            "Sample_rate": int(row["Sample_rate"]),
        }
        expected = {
            "Dataset_id": int(data_config["dataset_id"]),
            "Name": str(data_config["dataset_name"]),
            "Label": label,
            "Domain_id": domain,
            "Sample_rate": expected_rate,
        }
        if observed != expected:
            raise ValueError(
                f"metadata contract mismatch for {filename}: expected {expected}, observed {observed}"
            )
        path = raw_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"admitted raw source is missing: {path}")
        signal_key = str(data_config["signal_key_format"]).format(file_id=file_id)
        raw_rpm_key = f"X{file_id:03d}RPM"
        variable_shapes = {name: shape for name, shape, _ in whosmat(path)}
        if signal_key not in variable_shapes:
            raise ValueError(
                f"exact signal variable {signal_key!r} is absent from {filename}; fallback is forbidden"
            )
        loaded = loadmat(path, variable_names=[signal_key, raw_rpm_key])
        signal = np.asarray(loaded[signal_key], dtype=np.float32).squeeze()
        if signal.ndim != 1:
            raise ValueError(f"{filename}:{signal_key} must resolve to one DE vector")
        expected_length = int(row["Sample_lenth"])
        if int(signal.shape[0]) != expected_length:
            raise ValueError(
                f"{filename} length mismatch: metadata={expected_length}, signal={signal.shape[0]}"
            )
        if not np.isfinite(signal).all():
            raise ValueError(f"{filename} contains non-finite DE values")
        raw_rpm: float | None = None
        if raw_rpm_key in loaded:
            rpm_array = np.asarray(loaded[raw_rpm_key]).squeeze()
            if rpm_array.size != 1:
                raise ValueError(f"{filename}:{raw_rpm_key} must be one scalar")
            raw_rpm = _exact_scalar(rpm_array.item(), f"{filename} raw RPM")
            if raw_rpm <= 0.0:
                raise ValueError(f"{filename} raw RPM must be positive")
        elif file_id not in {98, 99}:
            raise ValueError(f"{filename} unexpectedly lacks raw RPM")

        partition = partition_by_domain[domain]
        nominal_rpm = rpm_by_domain[domain]
        starts = _window_starts(expected_length, window_length, windows_per_source)
        source_id = f"cwru12k-de-f{file_id:03d}"
        source_records.append(
            Source(
                file_id=file_id,
                metadata_id=str(row["Id"]),
                source_id=source_id,
                label=label,
                domain=domain,
                partition=partition,
                sample_rate_hz=float(expected_rate),
                rotation_speed_rpm=nominal_rpm,
                load_hp=float(domain),
                raw_rpm=raw_rpm,
                signal_length=expected_length,
                signal_key=signal_key,
                window_starts=starts,
            )
        )
        for start in starts:
            partition_windows[partition].append(signal[start : start + window_length, None])
            partition_labels[partition].append(label)
            partition_files[partition].append(file_id)
            partition_source_ids[partition].append(source_id)
            partition_rates[partition].append(float(expected_rate))
            partition_speeds[partition].append(nominal_rpm)
            partition_loads[partition].append(float(domain))

    partition_source_sets = {
        name: {source.source_id for source in source_records if source.partition == name}
        for name in PARTITIONS
    }
    for left_index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[left_index + 1 :]:
            overlap = partition_source_sets[left] & partition_source_sets[right]
            if overlap:
                raise ValueError(f"physical source leakage between {left} and {right}: {overlap}")
    for partition in PARTITIONS:
        sources = [source for source in source_records if source.partition == partition]
        if len(sources) != 4 or {source.label for source in sources} != set(ROLE_IDS):
            raise ValueError(f"partition {partition} must contain one source per class")
        if len(partition_windows[partition]) != 116:
            raise ValueError(f"partition {partition} must contain exactly 116 windows")

    train_array = np.stack(partition_windows["train"], axis=0).astype(np.float64)
    normalization_mean = float(train_array.mean())
    normalization_std = float(train_array.std())
    if not math.isfinite(normalization_mean) or not math.isfinite(normalization_std):
        raise ValueError("training normalization statistics are non-finite")
    if normalization_std <= 0.0:
        raise ValueError("training normalization standard deviation must be positive")

    partitions: dict[str, PartitionData] = {}
    for partition in PARTITIONS:
        raw = np.stack(partition_windows[partition], axis=0)
        normalized = ((raw - normalization_mean) / normalization_std).astype(np.float32)
        partitions[partition] = PartitionData(
            x=torch.from_numpy(normalized),
            y=torch.tensor(partition_labels[partition], dtype=torch.long),
            file_id=torch.tensor(partition_files[partition], dtype=torch.long),
            source_id=list(partition_source_ids[partition]),
            sample_rate_hz=torch.tensor(partition_rates[partition], dtype=torch.float32),
            rotation_speed_rpm=torch.tensor(partition_speeds[partition], dtype=torch.float32),
            load_hp=torch.tensor(partition_loads[partition], dtype=torch.float32),
        )

    contract = {
        "contract_id": config["protocol"]["contract_id"],
        "raw_root": str(raw_root),
        "metadata_path": str(metadata_path),
        "dataset_id": data_config["dataset_id"],
        "dataset_name": data_config["dataset_name"],
        "source_selection": data_config["selection"],
        "independent_unit": config["protocol"]["independent_unit"],
        "label_map": label_map,
        "sample_rate_hz": expected_rate,
        "speed_mode": data_config["speed"]["mode"],
        "raw_rpm_use": data_config["speed"]["raw_rpm_use"],
        "channel": data_config["channel"],
        "signal_key_fallback": "forbidden",
        "split_before_windowing": True,
        "source_sets": {key: sorted(value) for key, value in partition_source_sets.items()},
        "window_length_samples": window_length,
        "windows_per_source": windows_per_source,
        "windows_are_independent_replicates": False,
        "normalization": {
            "method": data_config["normalization"]["method"],
            "fit_partition": "train",
            "mean": normalization_mean,
            "std": normalization_std,
            "per_window_refit": False,
        },
        "sources": [source.__dict__ for source in source_records],
    }
    return AdmittedData(
        partitions=partitions,
        sources=source_records,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        contract=contract,
    )


def _model_args(config: Mapping[str, Any], arm: str) -> SimpleNamespace:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    model = dict(config["model"])
    model.update(
        {
            "scientific_arm": arm,
            "router_mode": "learned_only" if arm == "P1" else "learned_prior",
            "expert_representation_mode": (
                "homogeneous_raw" if arm == "P1" else "role_constrained"
            ),
            "semantic_alignment": list(config["protocol"]["semantic_alignment"][arm]),
            "role_prior_strength": 0.5,
            "role_prior_max": 1.0,
            "role_prior_permutation": [0, 1, 2, 3],
            "role_prior_assignment": "unspecified",
            "entropy_floor": 0.25,
        }
    )
    return SimpleNamespace(**model)


def _checkpoint_contract(
    config: Mapping[str, Any], arm: str, seed: int, epochs_requested: int
) -> dict[str, Any]:
    return {
        "contract_id": config["protocol"]["contract_id"],
        "experiment_id": config["protocol"]["experiment_id"],
        "arm": arm,
        "seed": int(seed),
        "epochs_requested": int(epochs_requested),
        "run_kind": (
            "smoke"
            if int(epochs_requested) == int(config["training"]["smoke_epochs"])
            else "pilot"
        ),
        "slot_permutation": list(
            config["protocol"]["slot_permutation_by_seed"][seed]
        ),
        "model": vars(_model_args(config, arm)),
        "training": {
            key: config["training"][key]
            for key in (
                "optimizer",
                "learning_rate",
                "weight_decay",
                "batch_size",
                "checkpoint_rule",
                "device",
                "visible_physical_gpu",
                "cublas_workspace_config",
            )
        },
        "data": {
            "sample_rate_hz": config["data"]["sample_rate_hz"],
            "channel": config["data"]["channel"],
            "windowing": config["data"]["windowing"],
            "normalization": config["data"]["normalization"],
            "speed": config["data"]["speed"],
        },
    }


def _batch_indices(length: int, batch_size: int, *, shuffle_seed: int | None = None) -> Iterable[torch.Tensor]:
    if shuffle_seed is None:
        order = torch.arange(length)
    else:
        generator = torch.Generator().manual_seed(shuffle_seed)
        order = torch.randperm(length, generator=generator)
    for start in range(0, length, batch_size):
        yield order[start : start + batch_size]


def _forward_batch(
    model: Model,
    data: PartitionData,
    indices: torch.Tensor,
    device: torch.device,
    *,
    diagnostics: bool = False,
) -> Any:
    return model(
        data.x[indices].to(device),
        file_id=data.file_id[indices].to(device),
        physical_metadata=data.physical(indices, device),
        return_diagnostics=diagnostics,
    )


def _fit_compatibility_statistics(
    model: Model,
    train: PartitionData,
    batch_size: int,
    device: torch.device,
) -> dict[str, list[float]]:
    if model.scientific_arm not in {"P0", "P2"}:
        return {"mean": [0.0] * 4, "std": [1.0] * 4, "fit_partition": "not_used_by_P1"}
    values = []
    model.eval()
    with torch.no_grad():
        for indices in _batch_indices(len(train), batch_size):
            cues = model.compatibility_cues(
                train.x[indices].to(device),
                file_id=train.file_id[indices].to(device),
                physical_metadata=train.physical(indices, device),
            )
            values.append(cues.cpu())
    all_cues = torch.cat(values, dim=0)
    mean = all_cues.mean(dim=0)
    std = all_cues.std(dim=0, unbiased=False)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 1e-8):
        raise RuntimeError("train-only compatibility cue statistics are degenerate")
    model.set_compatibility_statistics(mean, std)
    return {"mean": mean.tolist(), "std": std.tolist(), "fit_partition": "train"}


def _classification_metrics(labels: np.ndarray, probabilities: np.ndarray, source_ids: Sequence[str]) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    accuracy = float((predictions == labels).mean())
    f1_values = []
    for label in ROLE_IDS:
        true_positive = int(np.sum((predictions == label) & (labels == label)))
        false_positive = int(np.sum((predictions == label) & (labels != label)))
        false_negative = int(np.sum((predictions != label) & (labels == label)))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    confidence = probabilities.max(axis=1)
    correctness = (predictions == labels).astype(np.float64)
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(correctness[mask].mean()) - float(confidence[mask].mean()))
    source_predictions = []
    source_labels = []
    per_source = {}
    for source_id in sorted(set(source_ids)):
        mask = np.asarray([value == source_id for value in source_ids])
        mean_probability = probabilities[mask].mean(axis=0)
        source_prediction = int(mean_probability.argmax())
        source_label_values = np.unique(labels[mask])
        if source_label_values.size != 1:
            raise RuntimeError(f"source {source_id} has ambiguous labels")
        source_label = int(source_label_values.item())
        source_predictions.append(source_prediction)
        source_labels.append(source_label)
        per_source[source_id] = {
            "label": source_label,
            "prediction": source_prediction,
            "mean_probability": mean_probability.tolist(),
        }
    source_predictions_array = np.asarray(source_predictions)
    source_labels_array = np.asarray(source_labels)
    source_probabilities = np.asarray(
        [per_source[source_id]["mean_probability"] for source_id in sorted(per_source)]
    )
    source_f1 = []
    for label in ROLE_IDS:
        tp = int(np.sum((source_predictions_array == label) & (source_labels_array == label)))
        fp = int(np.sum((source_predictions_array == label) & (source_labels_array != label)))
        fn = int(np.sum((source_predictions_array != label) & (source_labels_array == label)))
        denominator = 2 * tp + fp + fn
        source_f1.append(0.0 if denominator == 0 else 2 * tp / denominator)
    source_confidence = source_probabilities.max(axis=1)
    source_correctness = (source_predictions_array == source_labels_array).astype(np.float64)
    source_ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (source_confidence >= lower) & (
            source_confidence < upper if upper < 1.0 else source_confidence <= upper
        )
        if np.any(mask):
            source_ece += float(mask.mean()) * abs(
                float(source_correctness[mask].mean())
                - float(source_confidence[mask].mean())
            )
    return {
        "window_accuracy_descriptive": accuracy,
        "window_macro_f1_descriptive": float(np.mean(f1_values)),
        "window_calibration_ece_descriptive": ece,
        "source_macro_f1": float(np.mean(source_f1)),
        "source_calibration_ece_descriptive": source_ece,
        "source_calibration_unit_count": len(per_source),
        "source_count": len(per_source),
        "per_source": per_source,
    }


def _evaluate_partition(
    model: Model,
    data: PartitionData,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    losses = []
    probabilities = []
    labels = []
    routing = []
    routing_entropy = []
    with torch.no_grad():
        for indices in _batch_indices(len(data), batch_size):
            logits, diagnostics = _forward_batch(model, data, indices, device, diagnostics=True)
            losses.append(F.cross_entropy(logits, data.y[indices].to(device), reduction="none").cpu())
            probabilities.append(F.softmax(logits, dim=-1).cpu())
            labels.append(data.y[indices])
            routing.append(diagnostics["routing_weights"].cpu())
            routing_entropy.append(diagnostics["routing_entropy"].cpu())
    probability = torch.cat(probabilities).numpy()
    label = torch.cat(labels).numpy()
    usage = torch.cat(routing).mean(dim=0)
    effective_experts = float(torch.exp(-(usage * usage.clamp_min(1e-8).log()).sum()).item())
    metrics = _classification_metrics(label, probability, data.source_id)
    metrics.update(
        {
            "cross_entropy": float(torch.cat(losses).mean().item()),
            "mean_routing_usage": usage.tolist(),
            "mean_normalized_routing_entropy": float(
                torch.cat(routing_entropy).mean().item()
            ),
            "effective_expert_count": effective_experts,
            "maximum_expert_usage": float(usage.max().item()),
            "collapsed": bool(effective_experts < 1.5 or float(usage.max()) > 0.80),
        }
    )
    return metrics


def _checkpoint_payload(
    model: Model,
    *,
    config: Mapping[str, Any],
    arm: str,
    seed: int,
    slot_permutation: Sequence[int],
    epoch: int,
    validation_loss: float,
    epochs_requested: int,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "seed": seed,
        "slot_permutation": list(slot_permutation),
        "epoch": epoch,
        "validation_loss": validation_loss,
        "contract": _checkpoint_contract(config, arm, seed, epochs_requested),
        "model_state": copy.deepcopy(model.state_dict()),
    }


def _load_checkpoint_strict(
    path: Path,
    config: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    epochs_requested: int,
    device: torch.device,
) -> Model:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_permutation = list(config["protocol"]["slot_permutation_by_seed"][seed])
    if payload.get("arm") != arm or int(payload.get("seed", -1)) != seed:
        raise ValueError("checkpoint arm/seed identity does not match requested run")
    if list(payload.get("slot_permutation", [])) != expected_permutation:
        raise ValueError("checkpoint slot permutation does not match frozen protocol")
    expected_contract = _checkpoint_contract(config, arm, seed, epochs_requested)
    if payload.get("contract") != expected_contract:
        raise ValueError("checkpoint scientific contract does not match the resolved config")
    if "model_state" not in payload:
        raise ValueError("checkpoint is missing model_state")
    _seed_everything(seed)
    model = Model(_model_args(config, arm)).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model


def _benchmark_latency(
    model: Model,
    data: PartitionData,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    indices = next(iter(_batch_indices(len(data), batch_size)))
    x = data.x[indices].to(device)
    file_id = data.file_id[indices].to(device)
    physical = data.physical(indices, device)
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(x, file_id=file_id, physical_metadata=physical)
        torch.cuda.synchronize()
        samples = []
        for _ in range(20):
            start = time.perf_counter()
            model(x, file_id=file_id, physical_metadata=physical)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "batch_size": int(indices.numel()),
        "median_batch_latency_ms": float(np.median(samples)),
        "active_experts_per_sample": 4,
        "dense_soft_mixture": True,
    }


def _train_run(
    config: Mapping[str, Any],
    data: AdmittedData,
    *,
    arm: str,
    seed: int,
    epochs: int,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[Model, dict[str, Any]]:
    _seed_everything(seed)
    slot_permutation = list(config["protocol"]["slot_permutation_by_seed"][seed])
    model = Model(_model_args(config, arm))
    model.permute_slots_(slot_permutation)
    model.to(device)
    batch_size = int(config["training"]["batch_size"])
    compatibility = _fit_compatibility_statistics(
        model, data.partitions["train"], batch_size, device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    best_payload: dict[str, Any] | None = None
    best_loss = math.inf
    history = []
    started = time.perf_counter()
    train = data.partitions["train"]
    validation = data.partitions["validation"]
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for indices in _batch_indices(len(train), batch_size, shuffle_seed=seed * 10_000 + epoch):
            optimizer.zero_grad(set_to_none=True)
            logits = _forward_batch(model, train, indices, device)
            task_loss = F.cross_entropy(logits, train.y[indices].to(device))
            auxiliary = model.consume_auxiliary_losses()
            if set(auxiliary) != {"moe_load_balance"}:
                raise RuntimeError(f"unexpected decisive objective terms: {sorted(auxiliary)}")
            total = task_loss + sum(auxiliary.values())
            if model.consume_auxiliary_losses():
                raise RuntimeError("model objective terms were not consumed exactly once")
            if not torch.isfinite(total):
                raise RuntimeError("training objective became non-finite")
            total.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"non-finite gradient in {name}")
            optimizer.step()
            train_losses.append(float(total.detach().cpu()))
        validation_metrics = _evaluate_partition(model, validation, batch_size, device)
        validation_loss = float(validation_metrics["cross_entropy"])
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": float(np.mean(train_losses)),
                "validation_cross_entropy": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_payload = _checkpoint_payload(
                model,
                config=config,
                arm=arm,
                seed=seed,
                slot_permutation=slot_permutation,
                epoch=epoch,
                validation_loss=validation_loss,
                epochs_requested=epochs,
            )
    if best_payload is None:
        raise RuntimeError("checkpoint rule did not select any epoch")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_payload, checkpoint_path)
    model = _load_checkpoint_strict(
        checkpoint_path,
        config,
        arm=arm,
        seed=seed,
        epochs_requested=epochs,
        device=device,
    )
    partition_metrics = {
        partition: _evaluate_partition(
            model, data.partitions[partition], batch_size, device
        )
        for partition in PARTITIONS
    }
    latency = _benchmark_latency(model, data.partitions["P_eval"], batch_size, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    run = {
        "arm": arm,
        "seed": seed,
        "slot_permutation": slot_permutation,
        "semantic_alignment_by_slot": model.semantic_alignment.tolist(),
        "slot_to_structure": model.slot_to_structure.tolist(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_rule": config["training"]["checkpoint_rule"],
        "selected_epoch": int(best_payload["epoch"]),
        "selected_validation_loss": float(best_payload["validation_loss"]),
        "epochs_executed": epochs,
        "history": history,
        "compatibility_standardization": compatibility,
        "partition_metrics": partition_metrics,
        "parameters": parameter_count,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "active_compute": latency,
        "elapsed_seconds": time.perf_counter() - started,
        "status": (
            "collapsed" if partition_metrics["P_eval"]["collapsed"] else "completed"
        ),
        "failure_reason": None,
    }
    return model, run


def _scaled_perturbation(x: torch.Tensor, perturbation: torch.Tensor, relative_rms: float) -> torch.Tensor:
    centered = perturbation - perturbation.mean(dim=1, keepdim=True)
    perturbation_rms = centered.square().mean(dim=1, keepdim=True).add(1e-8).sqrt()
    signal_rms = x.square().mean(dim=1, keepdim=True).add(1e-8).sqrt()
    return centered / perturbation_rms * (relative_rms * signal_rms)


def _apply_probe(
    x: torch.Tensor,
    physical: Mapping[str, torch.Tensor],
    probe: Mapping[str, Any],
    *,
    relative_rms: float,
    batch_offset: int,
) -> torch.Tensor:
    transform = str(probe["transform"])
    if transform == "time_roll":
        return x.roll(shifts=int(probe["shift_samples"]), dims=1)
    if transform == "sign_invert":
        return -x
    batch, length, _ = x.shape
    sample_rate = physical["sample_rate_hz"].to(x)[:, None, None]
    rotation_hz = physical["rotation_speed_rpm"].to(x)[:, None, None] / 60.0
    sample_index = torch.arange(length, dtype=x.dtype, device=x.device)[None, :, None]
    time_axis = sample_index / sample_rate
    if transform == "low_order":
        frequency = float(probe["order"]) * rotation_hz
        perturbation = torch.sin(2.0 * torch.pi * frequency * time_axis)
    elif transform == "harmonic_comb":
        perturbation = torch.zeros_like(x)
        orders = [float(value) for value in probe["orders"]]
        for order_index, order in enumerate(orders):
            phase = order_index * torch.pi / 5.0
            perturbation = perturbation + torch.sin(
                2.0 * torch.pi * order * rotation_hz * time_axis + phase
            ) / len(orders)
    elif transform == "periodic_impulse":
        repetition = float(probe["order"]) * rotation_hz
        phase = torch.remainder(time_axis * repetition, 1.0)
        envelope = torch.exp(-phase / 0.045)
        carrier = torch.sin(2.0 * torch.pi * 45.0 * rotation_hz * time_axis)
        perturbation = envelope * carrier
    elif transform == "broadband":
        generator = torch.Generator(device=x.device)
        generator.manual_seed(int(probe["noise_seed"]) * 100_000 + batch_offset)
        perturbation = torch.randn(
            (batch, length, 1), dtype=x.dtype, device=x.device, generator=generator
        )
    else:
        raise ValueError(f"unknown probe transform: {transform}")
    return x + _scaled_perturbation(x, perturbation, relative_rms)


def _probe_signatures(
    model: Model,
    data: PartitionData,
    probes: Sequence[Mapping[str, Any]],
    *,
    relative_rms: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    output: dict[str, Any] = {}
    with torch.no_grad():
        for probe in probes:
            signatures = []
            sources: list[str] = []
            offset = 0
            for indices in _batch_indices(len(data), batch_size):
                x = data.x[indices].to(device)
                physical = data.physical(indices, device)
                transformed = _apply_probe(
                    x,
                    physical,
                    probe,
                    relative_rms=relative_rms,
                    batch_offset=offset,
                )
                q = model.probe_response_signature(
                    x,
                    transformed,
                    file_id=data.file_id[indices].to(device),
                    physical_metadata=physical,
                )
                signatures.append(q.cpu())
                sources.extend(data.source_id[int(index)] for index in indices)
                offset += int(indices.numel())
            all_q = torch.cat(signatures, dim=0)
            by_source = {}
            for source_id in sorted(set(sources)):
                mask = torch.tensor([value == source_id for value in sources])
                by_source[source_id] = all_q[mask].mean(dim=0).tolist()
            output[str(probe["id"])] = {
                "role_id": probe.get("role_id"),
                "transform": probe["transform"],
                "by_source": by_source,
                "mean": all_q.mean(dim=0).tolist(),
            }
    return output


def _signature_array(
    payload: Mapping[str, Any],
    probes: Sequence[Mapping[str, Any]],
    source_ids: Sequence[str],
) -> np.ndarray:
    values = []
    for probe in probes:
        record = payload[str(probe["id"])]
        values.append(np.asarray([record["by_source"][source] for source in source_ids]))
    # [probe, source, slot, component] -> [slot, probe, source, component]
    return np.stack(values, axis=0).transpose(2, 0, 1, 3)


def _cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1, keepdims=True).clip(min=1e-12)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True).clip(min=1e-12)
    return (left / left_norm) @ (right / right_norm).T


def _hungarian(similarity: np.ndarray) -> list[int]:
    rows, columns = linear_sum_assignment(-similarity)
    if list(rows) != list(ROLE_IDS):
        raise RuntimeError("Hungarian solver returned incomplete left assignment")
    assignment = [-1] * 4
    for row, column in zip(rows, columns):
        assignment[int(row)] = int(column)
    if sorted(assignment) != list(ROLE_IDS):
        raise RuntimeError("Hungarian solver returned a non-bijective assignment")
    return assignment


def _delta_rid(p0: float, p1: float, p2: float, p0_null_mean: float) -> float:
    values = np.asarray([p0, p1, p2, p0_null_mean], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("RID estimands must be finite")
    return float(p0 - max(p1, p2, p0_null_mean))


def _knockout_contrast(slot_effects: Sequence[float], target_slot: int) -> float:
    effects = np.asarray(slot_effects, dtype=np.float64)
    if effects.shape != (4,) or not np.isfinite(effects).all():
        raise ValueError("knockout effects must contain four finite slot values")
    if int(target_slot) not in ROLE_IDS:
        raise ValueError("target_slot must be in [0, 1, 2, 3]")
    mismatched = [slot for slot in ROLE_IDS if slot != int(target_slot)]
    return float(effects[int(target_slot)] - effects[mismatched].mean())


def _role_to_slot(
    standardized_match: np.ndarray,
    probes: Sequence[Mapping[str, Any]],
) -> dict[int, int]:
    # standardized_match: [slot, probe, source, component]
    response_strength = np.linalg.norm(standardized_match, axis=-1).mean(axis=2)
    scores = np.zeros((4, 4), dtype=np.float64)
    roles = np.asarray([int(probe["role_id"]) for probe in probes])
    for role in ROLE_IDS:
        positive = response_strength[:, roles == role].mean(axis=1)
        negative = response_strength[:, roles != role].mean(axis=1)
        scores[role] = positive - negative
    role_rows, slot_columns = linear_sum_assignment(-scores)
    mapping = {int(role): int(slot) for role, slot in zip(role_rows, slot_columns)}
    if set(mapping) != set(ROLE_IDS) or set(mapping.values()) != set(ROLE_IDS):
        raise RuntimeError("recovered role map is not a complete bijection")
    return mapping


def _aggregate_matching(
    config: Mapping[str, Any],
    run_records: Mapping[tuple[str, int], Mapping[str, Any]],
    data: AdmittedData,
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[int, int]]]:
    match_probes = list(config["probes"]["match"])
    eval_probes = list(config["probes"]["eval"])
    match_sources = sorted(set(data.partitions["P_match"].source_id))
    eval_sources = sorted(set(data.partitions["P_eval"].source_id))
    role_maps: dict[tuple[str, int], dict[int, int]] = {}
    arms_output = {}
    all_null_values = []
    for arm in ARMS:
        match = np.stack(
            [
                _signature_array(run_records[(arm, seed)]["probe_match"], match_probes, match_sources)
                for seed in SEEDS
            ],
            axis=0,
        )
        evaluation = np.stack(
            [
                _signature_array(run_records[(arm, seed)]["probe_eval"], eval_probes, eval_sources)
                for seed in SEEDS
            ],
            axis=0,
        )
        component_mean = match.mean(axis=(0, 1, 2, 3))
        component_std = match.std(axis=(0, 1, 2, 3))
        if np.any(~np.isfinite(component_mean)) or np.any(component_std <= 1e-12):
            raise RuntimeError(f"{arm} P_match response standardization is degenerate")
        standardized_match = (match - component_mean) / component_std
        standardized_eval = (evaluation - component_mean) / component_std
        for seed_index, seed in enumerate(SEEDS):
            role_maps[(arm, seed)] = _role_to_slot(
                standardized_match[seed_index], match_probes
            )

        pair_records = []
        random_null = []
        probe_null = []
        for left_index, left_seed in enumerate(SEEDS):
            for right_index in range(left_index + 1, len(SEEDS)):
                right_seed = SEEDS[right_index]
                left_match = standardized_match[left_index].reshape(4, -1)
                right_match = standardized_match[right_index].reshape(4, -1)
                match_similarity = _cosine_matrix(left_match, right_match)
                assignment = _hungarian(match_similarity)
                left_eval = standardized_eval[left_index].reshape(4, -1)
                right_eval = standardized_eval[right_index].reshape(4, -1)
                eval_similarity = _cosine_matrix(left_eval, right_eval)
                held_out = float(np.mean([eval_similarity[slot, assignment[slot]] for slot in ROLE_IDS]))
                per_slot = [float(eval_similarity[slot, assignment[slot]]) for slot in ROLE_IDS]
                rng = np.random.default_rng(24_040_000 + left_seed * 100 + right_seed)
                pair_random = []
                pair_probe = []
                for _ in range(int(config["probes"]["null_repetitions"])):
                    random_assignment = rng.permutation(4)
                    pair_random.append(
                        float(np.mean([eval_similarity[slot, random_assignment[slot]] for slot in ROLE_IDS]))
                    )
                    permuted_probe_blocks = rng.permutation(len(match_probes))
                    shuffled = standardized_match[right_index][:, permuted_probe_blocks, :, :].reshape(4, -1)
                    shuffled_assignment = _hungarian(_cosine_matrix(left_match, shuffled))
                    pair_probe.append(
                        float(np.mean([eval_similarity[slot, shuffled_assignment[slot]] for slot in ROLE_IDS]))
                    )
                random_null.extend(pair_random)
                probe_null.extend(pair_probe)
                pair_records.append(
                    {
                        "left_seed": left_seed,
                        "right_seed": right_seed,
                        "assignment_left_slot_to_right_slot": assignment,
                        "match_similarity": match_similarity.tolist(),
                        "held_out_similarity": held_out,
                        "per_left_slot_held_out_similarity": per_slot,
                        "random_assignment_null_mean": float(np.mean(pair_random)),
                        "probe_label_permutation_null_mean": float(np.mean(pair_probe)),
                    }
                )
        arm_null = random_null + probe_null
        all_null_values.extend(arm_null)
        arms_output[arm] = {
            "component_standardization": {
                "fit_partition": "P_match",
                "mean": component_mean.tolist(),
                "std": component_std.tolist(),
            },
            "role_to_slot_from_P_match": {
                str(seed): role_maps[(arm, seed)] for seed in SEEDS
            },
            "seed_pairs": pair_records,
            "RID_HO": float(np.mean([record["held_out_similarity"] for record in pair_records])),
            "random_assignment_null_mean": float(np.mean(random_null)),
            "probe_label_permutation_null_mean": float(np.mean(probe_null)),
            "combined_null_mean": float(np.mean(arm_null)),
        }
    p0 = arms_output["P0"]["RID_HO"]
    p0_null_mean = float(arms_output["P0"]["combined_null_mean"])
    strongest_control = max(
        arms_output["P1"]["RID_HO"],
        arms_output["P2"]["RID_HO"],
        p0_null_mean,
    )
    output = {
        "blinding": {
            "expert_names_hidden": True,
            "structural_labels_hidden": True,
            "task_loss_used_for_matching": False,
        },
        "fit_partition": "P_match",
        "evaluation_partition": "P_eval",
        "arms": arms_output,
        "P0_null_mean": p0_null_mean,
        "all_arms_null_diagnostic_mean": float(np.mean(all_null_values)),
        "delta_rid": _delta_rid(
            float(p0),
            float(arms_output["P1"]["RID_HO"]),
            float(arms_output["P2"]["RID_HO"]),
            p0_null_mean,
        ),
        "strongest_rid_control": float(strongest_control),
    }
    return output, role_maps


def _interventions(
    model: Model,
    data: PartitionData,
    probes: Sequence[Mapping[str, Any]],
    role_to_slot: Mapping[int, int],
    *,
    relative_rms: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    records = []
    with torch.no_grad():
        for probe in probes:
            role = int(probe["role_id"])
            target_slot = int(role_to_slot[role])
            by_source: dict[str, list[list[float]]] = {}
            offset = 0
            for indices in _batch_indices(len(data), batch_size):
                x = data.x[indices].to(device)
                y = data.y[indices].to(device)
                physical = data.physical(indices, device)
                transformed = _apply_probe(
                    x,
                    physical,
                    probe,
                    relative_rms=relative_rms,
                    batch_offset=offset,
                )
                effects = model.deletion_effects(
                    transformed,
                    file_id=data.file_id[indices].to(device),
                    physical_metadata=physical,
                )
                baseline_loss = F.cross_entropy(
                    effects["baseline_logits"], y, reduction="none"
                )
                deleted = effects["deleted_logits"]
                deletion_loss = torch.stack(
                    [
                        F.cross_entropy(deleted[:, slot, :], y, reduction="none")
                        for slot in ROLE_IDS
                    ],
                    dim=1,
                )
                delta = (deletion_loss - baseline_loss[:, None]).cpu().tolist()
                for local_index, global_index in enumerate(indices.tolist()):
                    source = data.source_id[global_index]
                    by_source.setdefault(source, []).append(delta[local_index])
                offset += int(indices.numel())
            source_effects = {
                source: np.asarray(values, dtype=np.float64).mean(axis=0).tolist()
                for source, values in by_source.items()
            }
            mean_effect = np.asarray(list(source_effects.values())).mean(axis=0)
            mismatched = [slot for slot in ROLE_IDS if slot != target_slot]
            contrast = _knockout_contrast(mean_effect, target_slot)
            records.append(
                {
                    "probe_id": probe["id"],
                    "role_id": role,
                    "recovered_target_slot": target_slot,
                    "slot_knockout_effect_by_source": source_effects,
                    "mean_slot_knockout_effect": mean_effect.tolist(),
                    "targeted_effect": float(mean_effect[target_slot]),
                    "mismatched_mean_effect": float(mean_effect[mismatched].mean()),
                    "targeted_minus_mismatched": contrast,
                }
            )
    return {
        "role_to_slot": dict(role_to_slot),
        "records": records,
        "delta_int_run": float(np.mean([record["targeted_minus_mismatched"] for record in records])),
    }


def _decision(
    matching: Mapping[str, Any],
    interventions: Mapping[str, Any],
    runs: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    arms = matching["arms"]
    p0_rid = float(arms["P0"]["RID_HO"])
    rid_controls = {
        "P1": float(arms["P1"]["RID_HO"]),
        "P2": float(arms["P2"]["RID_HO"]),
        "null": float(matching["P0_null_mean"]),
    }
    delta_int = float(interventions["delta_int"])
    pair_advantages = []
    for pair_index in range(3):
        p0_pair = float(arms["P0"]["seed_pairs"][pair_index]["held_out_similarity"])
        control_pair = max(
            float(arms["P1"]["seed_pairs"][pair_index]["held_out_similarity"]),
            float(arms["P2"]["seed_pairs"][pair_index]["held_out_similarity"]),
            float(arms["P0"]["seed_pairs"][pair_index]["random_assignment_null_mean"]),
            float(arms["P0"]["seed_pairs"][pair_index]["probe_label_permutation_null_mean"]),
        )
        pair_advantages.append(p0_pair - control_pair)
    p0_diagnostics = [runs[("P0", seed)]["partition_metrics"]["P_eval"] for seed in SEEDS]
    no_collapse = all(not bool(value["collapsed"]) for value in p0_diagnostics)
    no_dominance = all(float(value["maximum_expert_usage"]) <= 0.80 for value in p0_diagnostics)
    rid_exceeds_all = all(p0_rid > value for value in rid_controls.values())
    intervention_valid = delta_int > 0.0
    multi_pair_support = sum(value > 0.0 for value in pair_advantages) >= 2
    if rid_exceeds_all and intervention_valid and multi_pair_support and no_collapse and no_dominance:
        decision = "continue"
        rationale = "P0 exceeds all RID controls and recovered-role knockout exceeds mismatches without collapse."
    elif rid_exceeds_all and no_collapse and no_dominance:
        decision = "reposition"
        rationale = "Routing is non-collapsed, but functional-role recovery or intervention validity does not clear every control."
    else:
        decision = "stop_or_merge"
        rationale = "The primary P0 advantage is absent or explained by controls, dominance, or collapse."
    return {
        "decision": decision,
        "rationale": rationale,
        "checks": {
            "P0_RID_exceeds_P1_P2_and_null": rid_exceeds_all,
            "targeted_knockout_exceeds_mismatched": intervention_valid,
            "at_least_two_seed_pairs_support_primary_advantage": multi_pair_support,
            "no_P0_collapse": no_collapse,
            "no_P0_dominant_expert": no_dominance,
        },
        "pair_advantages": pair_advantages,
        "rid_controls": rid_controls,
        "delta_rid": matching["delta_rid"],
        "delta_int": delta_int,
    }


def _probe_spec(config: Mapping[str, Any], data: AdmittedData) -> dict[str, Any]:
    return {
        "declared_in_versioned_config_before_execution": True,
        "P_match_and_P_eval_disjoint": True,
        "probe_IDs_and_exact_transform_specs_validated_disjoint": True,
        "P_match_sources": sorted(set(data.partitions["P_match"].source_id)),
        "P_eval_sources": sorted(set(data.partitions["P_eval"].source_id)),
        "sample_rate_required": True,
        "speed_required": True,
        "load_consumed": True,
        "speed_load_variation": "P_match uses domain/load 2 at 1750 RPM; P_eval uses domain/load 3 at 1730 RPM",
        "matched_relative_rms": config["probes"]["relative_rms"],
        "match": config["probes"]["match"],
        "eval": config["probes"]["eval"],
        "irrelevant_controls": config["probes"]["controls"],
        "irrelevant_control_use": "descriptive specificity diagnostic on P_eval only; never fitted or thresholded",
        "response_signature": [
            "delta_routing_mass",
            "delta_feature_l2_norm",
            "one_minus_feature_cosine",
        ],
        "standardization": config["probes"]["match_standardization"],
        "matching": config["probes"]["matching"],
    }


def _semantic_gate_payload(observed: Mapping[str, Any]) -> dict[str, Any]:
    if observed.get("status") != "passed":
        return {
            "status": "failed_before_pilot",
            "observed_test_execution": observed,
            "gates": {},
        }
    return {
        "status": "passed_before_pilot",
        "observed_test_execution": observed,
        "gates": {
            "sampling_rate_sensitive_physical_interpretation": "passed",
            "consistent_slot_permutation_invariance": "passed",
            "inconsistent_router_expert_permutation_failure": "passed",
            "fixed_physical_path_gradient_connectivity": "passed",
            "softmax_l1_disabled_value_and_gradient": "passed",
            "single_objective_accounting": "passed",
            "P2_semantic_map_only": "passed",
            "ambiguity_fail_fast": "passed",
            "recovered_role_intervention": "passed",
            "source_disjoint_after_windowing": "passed",
        },
        "trainable_physical_terms": [],
        "fixed_physical_transforms_are_structural_preprocessing": True,
        "physical_loss_weight": 0.0,
        "entropy_floor_or_rescue_loss": False,
    }


def _run_smoke(
    config: Mapping[str, Any],
    data: AdmittedData,
    output_root: Path,
    device: torch.device,
) -> None:
    smoke_root = output_root / "smoke"
    checkpoint = smoke_root / "p0_seed20_smoke.pt"
    model, record = _train_run(
        config,
        data,
        arm="P0",
        seed=20,
        epochs=int(config["training"]["smoke_epochs"]),
        checkpoint_path=checkpoint,
        device=device,
    )
    batch_size = int(config["training"]["batch_size"])
    relative_rms = float(config["probes"]["relative_rms"])
    match_payload = _probe_signatures(
        model,
        data.partitions["P_match"],
        config["probes"]["match"],
        relative_rms=relative_rms,
        batch_size=batch_size,
        device=device,
    )
    control_payload = _probe_signatures(
        model,
        data.partitions["P_eval"],
        config["probes"]["controls"],
        relative_rms=relative_rms,
        batch_size=batch_size,
        device=device,
    )
    match_sources = sorted(set(data.partitions["P_match"].source_id))
    match_array = _signature_array(match_payload, config["probes"]["match"], match_sources)
    component_mean = match_array.mean(axis=(0, 1, 2))
    component_std = match_array.std(axis=(0, 1, 2))
    if np.any(component_std <= 1e-12):
        raise RuntimeError("smoke P_match response standardization is degenerate")
    role_map = _role_to_slot(
        (match_array - component_mean) / component_std,
        config["probes"]["match"],
    )
    intervention = _interventions(
        model,
        data.partitions["P_eval"],
        config["probes"]["eval"],
        role_map,
        relative_rms=relative_rms,
        batch_size=batch_size,
        device=device,
    )
    portable_record = dict(record)
    portable_record["checkpoint"] = str(
        Path(record["checkpoint"]).relative_to(output_root)
    )
    _write_json(
        smoke_root / "smoke.json",
        {
            "paper_evidence": False,
            "purpose": "loading/metadata/gradient/checkpoint/permutation/probe/intervention plumbing",
            "run": portable_record,
            "recovered_role_to_slot": role_map,
            "irrelevant_control_signatures": control_payload,
            "intervention_plumbing": intervention,
            "status": "passed",
        },
    )


def _run_pilot(
    config: Mapping[str, Any],
    data: AdmittedData,
    output_root: Path,
    device: torch.device,
    execution_context: dict[str, Any],
) -> None:
    batch_size = int(config["training"]["batch_size"])
    relative_rms = float(config["probes"]["relative_rms"])
    run_records: dict[tuple[str, int], dict[str, Any]] = {}
    for arm in ARMS:
        for seed in SEEDS:
            execution_context["execution_stage"] = f"train:{arm}:seed{seed}"
            checkpoint = output_root / "checkpoints" / f"{arm.lower()}_seed{seed}.pt"
            key = (arm, seed)
            try:
                model, record = _train_run(
                    config,
                    data,
                    arm=arm,
                    seed=seed,
                    epochs=int(config["training"]["epochs"]),
                    checkpoint_path=checkpoint,
                    device=device,
                )
                record["probe_match"] = _probe_signatures(
                    model,
                    data.partitions["P_match"],
                    config["probes"]["match"],
                    relative_rms=relative_rms,
                    batch_size=batch_size,
                    device=device,
                )
                record["probe_eval"] = _probe_signatures(
                    model,
                    data.partitions["P_eval"],
                    config["probes"]["eval"],
                    relative_rms=relative_rms,
                    batch_size=batch_size,
                    device=device,
                )
                record["probe_controls"] = _probe_signatures(
                    model,
                    data.partitions["P_eval"],
                    config["probes"]["controls"],
                    relative_rms=relative_rms,
                    batch_size=batch_size,
                    device=device,
                )
                run_records[key] = record
                print(f"completed {arm} seed={seed} status={record['status']}", flush=True)
                del model
                torch.cuda.empty_cache()
            except Exception as exc:
                run_records[key] = {
                    "arm": arm,
                    "seed": seed,
                    "slot_permutation": list(config["protocol"]["slot_permutation_by_seed"][seed]),
                    "status": "failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "checkpoint": str(checkpoint),
                }
                print(f"failed {arm} seed={seed}: {type(exc).__name__}: {exc}", flush=True)

    missing = [key for key, value in run_records.items() if "probe_match" not in value]
    run_summaries = [
        dict({
            key: item
            for key, item in value.items()
            if key not in {"probe_match", "probe_eval", "probe_controls"}
        }, checkpoint=str(Path(value["checkpoint"]).relative_to(output_root)))
        for value in run_records.values()
    ]
    run_index = {
        "experiment_id": config["protocol"]["experiment_id"],
        "execution_status": "evaluation_pending" if not missing else "failed",
        "execution": execution_context,
        "arm_seed_matrix_complete": len(run_records) == 9,
        "all_runs_evaluable": not missing,
        "runs": run_summaries,
    }
    execution_context["execution_stage"] = "write_run_index"
    _write_json(output_root / "run_index.json", run_index)
    if missing:
        execution_context["execution_stage"] = "train_matrix_validation"
        raise RuntimeError(f"pilot contains non-evaluable runs; no seed substitution allowed: {missing}")

    execution_context["execution_stage"] = "matching"
    matching, role_maps = _aggregate_matching(config, run_records, data)
    intervention_runs = {}
    for arm in ARMS:
        for seed in SEEDS:
            execution_context["execution_stage"] = f"intervention:{arm}:seed{seed}"
            model = _load_checkpoint_strict(
                Path(run_records[(arm, seed)]["checkpoint"]),
                config,
                arm=arm,
                seed=seed,
                epochs_requested=int(config["training"]["epochs"]),
                device=device,
            )
            intervention_runs[f"{arm}_seed{seed}"] = _interventions(
                model,
                data.partitions["P_eval"],
                config["probes"]["eval"],
                role_maps[(arm, seed)],
                relative_rms=relative_rms,
                batch_size=batch_size,
                device=device,
            )
            del model
            torch.cuda.empty_cache()
    p0_delta_int = float(
        np.mean([intervention_runs[f"P0_seed{seed}"]["delta_int_run"] for seed in SEEDS])
    )
    interventions = {
        "fit_target_from": "P_match response signatures only",
        "evaluation_partition": "P_eval",
        "task_loss_used_for_matching": False,
        "runs": intervention_runs,
        "delta_int": p0_delta_int,
    }
    execution_context["execution_stage"] = "decision"
    decision = _decision(matching, interventions, run_records)
    metrics = {
        "delta_rid": matching["delta_rid"],
        "delta_int": interventions["delta_int"],
        "decision": decision,
        "arm_seed_metrics": {
            f"{arm}_seed{seed}": {
                "status": run_records[(arm, seed)]["status"],
                "partition_metrics": run_records[(arm, seed)]["partition_metrics"],
                "parameters": run_records[(arm, seed)]["parameters"],
                "active_compute": run_records[(arm, seed)]["active_compute"],
            }
            for arm in ARMS
            for seed in SEEDS
        },
        "irrelevant_control_response_norms": {
            f"{arm}_seed{seed}": {
                probe_id: float(np.linalg.norm(np.asarray(record["mean"]), axis=-1).mean())
                for probe_id, record in run_records[(arm, seed)]["probe_controls"].items()
            }
            for arm in ARMS
            for seed in SEEDS
        },
    }
    execution_context["execution_stage"] = "write_outputs"
    _write_json(output_root / "role_matching.json", matching)
    _write_json(output_root / "interventions.json", interventions)
    _write_json(output_root / "metrics.json", metrics)
    # Full response signatures remain machine-readable but separate from the run index.
    _write_json(
        output_root / "response_signatures.json",
        {
            f"{arm}_seed{seed}": {
                "P_match": run_records[(arm, seed)]["probe_match"],
                "P_eval": run_records[(arm, seed)]["probe_eval"],
                "irrelevant_controls_P_eval": run_records[(arm, seed)]["probe_controls"],
            }
            for arm in ARMS
            for seed in SEEDS
        },
    )
    completed_at = _utc_now()
    completed_execution = dict(execution_context)
    completed_execution.update(
        {
            "execution_status": "complete",
            "completed_at": completed_at,
        }
    )
    run_index.update(
        {
            "execution_status": "complete",
            "completed_at": completed_at,
            "execution": completed_execution,
            "delta_rid": matching["delta_rid"],
            "delta_int": interventions["delta_int"],
            "decision": decision["decision"],
        }
    )
    _write_json(output_root / "run_index.json", run_index)


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument(
        "--mode", choices=("smoke", "pilot", "all"), required=True
    )
    return parser.parse_args(arguments)


def _record_terminal_failure(
    output_root: Path,
    config: Mapping[str, Any],
    execution_context: Mapping[str, Any],
    error: Exception,
) -> None:
    """Persist one truthful terminal state without discarding completed runs."""

    index_path = output_root / "run_index.json"
    previous: dict[str, Any] = {}
    previous_read_error: str | None = None
    if index_path.is_file():
        try:
            loaded = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous = loaded
            else:
                previous_read_error = "existing run_index.json is not a JSON object"
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            previous_read_error = f"{type(exc).__name__}: {exc}"

    terminal_context = dict(execution_context)
    prior_status = str(terminal_context.get("execution_status", ""))
    terminal_status = prior_status if prior_status.startswith("failed") else "failed"
    terminal_context.update(
        {
            "execution_status": terminal_status,
            "completed_at": _utc_now(),
        }
    )
    failure = {
        "stage": terminal_context.get("execution_stage", "unknown"),
        "type": type(error).__name__,
        "message": str(error),
    }
    if previous_read_error is not None:
        failure["previous_run_index_error"] = previous_read_error
    previous.update(
        {
            "experiment_id": config["protocol"]["experiment_id"],
            "execution_status": terminal_status,
            "completed_at": terminal_context["completed_at"],
            "execution": terminal_context,
            "failure": failure,
        }
    )
    previous.setdefault("runs", [])
    _write_json(index_path, previous)


def main() -> int:
    args = _parse_args()
    config_path = G050_CONFIG_PATH.resolve()
    config = _load_config(config_path)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"output root is non-empty and cannot be reused: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    execution_context: dict[str, Any] = {
        "execution_status": "running",
        "started_at": _utc_now(),
        "command": [sys.executable, *sys.argv],
        "config_path": str(config_path),
        "resolved_config_path": "resolved_config.yaml",
        "artifact_root_at_execution": str(output_root),
        "artifact_paths_relative_to_result_root": True,
        "execution_stage": "runtime_preflight",
    }
    _write_json(
        output_root / "run_index.json",
        {
            "experiment_id": config["protocol"]["experiment_id"],
            "execution_status": "running",
            "execution": execution_context,
            "runs": [],
        },
    )
    try:
        device = _require_gpu5()
        runtime_provenance = _runtime_git_provenance()
        execution_context["runtime_git"] = runtime_provenance
        if runtime_provenance["dirty"]:
            raise RuntimeError(
                "evidence-bearing execution requires a clean versioned runtime; "
                f"changed paths: {runtime_provenance['changed_paths']}"
            )
        (output_root / "resolved_config.yaml").write_text(
            yaml.safe_dump(_plain(config), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        execution_context["execution_stage"] = "semantic_gate"
        observed_gate = _run_semantic_gate_tests()
        gate_payload = _semantic_gate_payload(observed_gate)
        _write_json(output_root / "semantic_gate.json", gate_payload)
        execution_context["semantic_gate_execution"] = observed_gate
        if observed_gate["status"] != "passed":
            execution_context["execution_status"] = "failed_semantic_gate"
            raise RuntimeError(
                "targeted semantic gate tests failed; pilot was not launched"
            )

        execution_context["execution_stage"] = "load_data"
        data = _load_admitted_data(config, args.raw_root)
        _write_json(output_root / "data_contract.json", data.contract)
        (output_root / "probe_spec.yaml").write_text(
            yaml.safe_dump(
                _plain(_probe_spec(config, data)), sort_keys=False, allow_unicode=True
            ),
            encoding="utf-8",
        )
        _write_json(
            output_root / "environment.json",
            {
                "python": sys.executable,
                "python_version": sys.version,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "openpyxl": openpyxl.__version__,
                "pyyaml": yaml.__version__,
                "cuda_available": torch.cuda.is_available(),
                "visible_device_count": torch.cuda.device_count(),
                "visible_device_name": torch.cuda.get_device_name(0),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "requested_physical_gpu": 5,
                "multi_gpu": False,
                "ddp": False,
                "data_parallel": False,
                "cpu_fallback": False,
                "runtime_git": runtime_provenance,
            },
        )
        if args.mode in {"smoke", "all"}:
            execution_context["execution_stage"] = "smoke"
            _run_smoke(config, data, output_root, device)
        if args.mode in {"pilot", "all"}:
            execution_context["execution_stage"] = "pilot"
            _run_pilot(config, data, output_root, device, execution_context)
        elif args.mode == "smoke":
            execution_context["execution_status"] = "smoke_complete"
            execution_context["completed_at"] = _utc_now()
            _write_json(
                output_root / "run_index.json",
                {
                    "experiment_id": config["protocol"]["experiment_id"],
                    "execution_status": "smoke_complete",
                    "completed_at": execution_context["completed_at"],
                    "execution": execution_context,
                    "runs": [],
                },
            )
    except Exception as exc:
        try:
            _record_terminal_failure(output_root, config, execution_context, exc)
        except Exception as state_error:
            raise RuntimeError(
                "G050 failed and its terminal state could not be written: "
                f"original={type(exc).__name__}: {exc}"
            ) from state_error
        raise
    print(f"G050 {args.mode} complete: {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
