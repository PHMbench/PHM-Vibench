#!/usr/bin/env python3
"""Run the P09-G050 real-data mechanism experiment.

This is deliberately smaller than the claim experiment. It uses the frozen
P09-GFS-V1 taxonomy, six outer target systems, five seeds, record-disjoint
support/query pools, and three paired arms. Its output may support, refute, or
leave the central mechanism inconclusive; it cannot by itself establish C1/C2.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.task_factory.task.GFS.reliability_conditioned import (
    SupportReliabilityConditioner,
)


@dataclass(frozen=True)
class SourceClassRole:
    head_records: tuple[int, ...]
    support_record: int
    query_record: int


@dataclass(frozen=True)
class FrozenHead:
    scaler: StandardScaler
    weights: torch.Tensor
    bias: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def as_int_mapping(value: Mapping[Any, Any]) -> dict[int, int]:
    return {int(key): int(mapped) for key, mapped in value.items()}


def feature_vector(window: np.ndarray) -> np.ndarray:
    signal = np.asarray(window, dtype=np.float64).reshape(window.shape[0], -1)
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    standardized = (signal - mean) / (std + 1.0e-8)

    abs_mean = np.mean(np.abs(standardized), axis=0)
    peak = np.max(np.abs(standardized), axis=0)
    skew = np.mean(standardized**3, axis=0)
    kurtosis = np.mean(standardized**4, axis=0) - 3.0
    zero_crossing = np.mean(
        np.signbit(standardized[1:]) != np.signbit(standardized[:-1]), axis=0
    )
    autocorrelation = np.mean(standardized[1:] * standardized[:-1], axis=0)

    spectrum = np.fft.rfft(standardized, axis=0)
    power = np.abs(spectrum) ** 2
    power = power / (power.sum(axis=0, keepdims=True) + 1.0e-12)
    frequencies = np.linspace(0.0, 1.0, power.shape[0], dtype=np.float64)[:, None]
    centroid = np.sum(frequencies * power, axis=0)
    entropy = -np.sum(power * np.log(power + 1.0e-12), axis=0) / math.log(power.shape[0])
    band_edges = np.linspace(0, power.shape[0], 5, dtype=int)
    bands = [
        power[band_edges[index] : band_edges[index + 1]].sum(axis=0)
        for index in range(4)
    ]

    channel_features = np.stack(
        [
            abs_mean,
            peak,
            skew,
            kurtosis,
            zero_crossing,
            autocorrelation,
            centroid,
            entropy,
            *bands,
        ],
        axis=1,
    )
    aggregated = np.concatenate(
        (channel_features.mean(axis=0), channel_features.max(axis=0)), axis=0
    )
    return np.nan_to_num(aggregated, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def evenly_spaced_starts(length: int, window_size: int, count: int) -> np.ndarray:
    if length < window_size:
        raise ValueError(f"record length {length} is smaller than window size {window_size}")
    if count <= 0:
        raise ValueError("candidate window count must be positive")
    starts = np.rint(np.linspace(0, length - window_size, count)).astype(np.int64)
    if np.unique(starts).size != count:
        raise ValueError("record cannot provide the requested distinct candidate windows")
    return starts


def build_feature_cache(
    cache_path: Path,
    records: Mapping[int, Mapping[int, Sequence[int]]],
    *,
    window_size: int,
    candidate_windows: int,
) -> tuple[dict[int, np.ndarray], dict[int, list[int]]]:
    feature_cache: dict[int, np.ndarray] = {}
    start_cache: dict[int, list[int]] = {}
    record_ids = sorted(
        {
            int(record_id)
            for system_records in records.values()
            for class_records in system_records.values()
            for record_id in class_records
        }
    )
    with h5py.File(cache_path, "r") as handle:
        for position, record_id in enumerate(record_ids, start=1):
            key = str(record_id)
            if key not in handle:
                raise KeyError(f"cache is missing metadata Id {record_id}")
            dataset = handle[key]
            starts = evenly_spaced_starts(
                int(dataset.shape[0]), window_size, candidate_windows
            )
            vectors = [
                feature_vector(np.asarray(dataset[start : start + window_size]))
                for start in starts
            ]
            feature_cache[record_id] = np.stack(vectors)
            start_cache[record_id] = [int(value) for value in starts]
            if position % 50 == 0 or position == len(record_ids):
                print(f"feature_cache {position}/{len(record_ids)}", flush=True)
    return feature_cache, start_cache


def load_records(
    metadata_path: Path,
    system_ids: Sequence[int],
    canonical_maps: Mapping[int, Mapping[int, int]],
) -> tuple[pd.DataFrame, dict[int, dict[int, list[int]]]]:
    metadata = pd.read_excel(metadata_path)
    records: dict[int, dict[int, list[int]]] = {}
    for system_id in system_ids:
        class_map = canonical_maps[system_id]
        mapped_label = metadata["Label"].map(
            lambda value: pd.notna(value) and int(value) in class_map
        )
        selected = metadata[
            (metadata["Dataset_id"] == system_id)
            & mapped_label
        ].copy()
        selected["canonical_label"] = selected["Label"].map(
            lambda value: class_map[int(value)]
        )
        records[system_id] = {}
        for class_id in range(4):
            ids = sorted(int(value) for value in selected.loc[
                selected["canonical_label"] == class_id, "Id"
            ].tolist())
            if len(ids) < 3:
                raise ValueError(
                    f"system {system_id}, canonical class {class_id} has fewer than three records"
                )
            records[system_id][class_id] = ids
    return metadata, records


def candidate_keys(record_ids: Iterable[int], candidate_windows: int) -> list[tuple[int, int]]:
    return [
        (int(record_id), window_index)
        for record_id in record_ids
        for window_index in range(candidate_windows)
    ]


def sample_keys(
    record_ids: Sequence[int],
    count: int,
    candidate_windows: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    candidates = candidate_keys(record_ids, candidate_windows)
    if len(candidates) < count:
        raise ValueError(f"requested {count} samples from a pool of {len(candidates)}")
    indices = rng.choice(len(candidates), size=count, replace=False)
    return [candidates[int(index)] for index in indices]


def vectors_for_keys(
    keys: Sequence[tuple[int, int]],
    feature_cache: Mapping[int, np.ndarray],
    scaler: StandardScaler,
) -> np.ndarray:
    matrix = np.stack([feature_cache[record_id][window] for record_id, window in keys])
    return scaler.transform(matrix).astype(np.float32)


def split_source_roles(
    records: Mapping[int, Mapping[int, Sequence[int]]],
    source_systems: Sequence[int],
    seed: int,
    max_head_records: int,
) -> dict[int, dict[int, SourceClassRole]]:
    roles: dict[int, dict[int, SourceClassRole]] = {}
    for system_id in source_systems:
        roles[system_id] = {}
        for class_id in (0, 1):
            rng = np.random.default_rng(np.random.SeedSequence([seed, system_id, class_id, 11]))
            permutation = [int(value) for value in rng.permutation(records[system_id][class_id])]
            head_candidates = permutation[:-2]
            if not head_candidates:
                raise ValueError("source role split requires at least three records per class")
            head = tuple(head_candidates[:max_head_records])
            roles[system_id][class_id] = SourceClassRole(
                head_records=head,
                support_record=permutation[-2],
                query_record=permutation[-1],
            )
    return roles


def fit_frozen_head(
    roles: Mapping[int, Mapping[int, SourceClassRole]],
    feature_cache: Mapping[int, np.ndarray],
    *,
    seed: int,
    logistic_c: float,
    max_iter: int,
) -> FrozenHead:
    matrices: list[np.ndarray] = []
    labels: list[int] = []
    for system_roles in roles.values():
        for class_id in (0, 1):
            for record_id in system_roles[class_id].head_records:
                matrix = feature_cache[record_id]
                matrices.append(matrix)
                labels.extend([class_id] * matrix.shape[0])
    features = np.concatenate(matrices, axis=0)
    label_array = np.asarray(labels, dtype=np.int64)
    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    classifier = LogisticRegression(
        C=logistic_c,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=seed,
        solver="lbfgs",
    ).fit(scaled, label_array)
    if classifier.classes_.tolist() != [0, 1] or classifier.coef_.shape[0] != 1:
        raise RuntimeError("binary source head did not resolve canonical classes [0, 1]")
    coefficient = classifier.coef_[0].astype(np.float32)
    intercept = float(classifier.intercept_[0])
    weights = torch.from_numpy(np.stack((-0.5 * coefficient, 0.5 * coefficient)))
    bias = torch.tensor([-0.5 * intercept, 0.5 * intercept], dtype=torch.float32)
    return FrozenHead(scaler=scaler, weights=weights, bias=bias)


def train_conditioner(
    roles: Mapping[int, Mapping[int, SourceClassRole]],
    source_systems: Sequence[int],
    feature_cache: Mapping[int, np.ndarray],
    head: FrozenHead,
    config: Mapping[str, Any],
    *,
    seed: int,
    candidate_windows: int,
) -> tuple[SupportReliabilityConditioner, float]:
    torch.manual_seed(seed)
    feature_dim = int(head.weights.shape[1])
    module = SupportReliabilityConditioner(
        feature_dim=feature_dim,
        adapter_rank=min(int(config["adapter_rank"]), feature_dim),
    )
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=float(config["controller_learning_rate"]),
        weight_decay=float(config["controller_weight_decay"]),
    )
    rng = np.random.default_rng(np.random.SeedSequence([seed, 991]))
    final_loss = math.nan
    module.train()
    for step in range(int(config["controller_steps"])):
        system_id = int(source_systems[step % len(source_systems)])
        novel_class = int(rng.integers(0, 2))
        base_class = 1 - novel_class
        novel_role = roles[system_id][novel_class]
        base_role = roles[system_id][base_class]
        support_keys = sample_keys(
            [novel_role.support_record],
            int(config["controller_support"]),
            candidate_windows,
            rng,
        )
        novel_query_keys = sample_keys(
            [novel_role.query_record],
            int(config["controller_query_per_class"]),
            candidate_windows,
            rng,
        )
        base_query_keys = sample_keys(
            [base_role.query_record],
            int(config["controller_query_per_class"]),
            candidate_windows,
            rng,
        )
        support = torch.from_numpy(vectors_for_keys(support_keys, feature_cache, head.scaler))
        support_labels = torch.full(
            (support.shape[0],), novel_class, dtype=torch.long
        )
        query = torch.from_numpy(
            vectors_for_keys(base_query_keys + novel_query_keys, feature_cache, head.scaler)
        )
        base_weights = head.weights[[base_class]]
        base_bias = head.bias[[base_class]]
        condition = module.condition(
            support, support_labels, base_weights, [novel_class]
        )
        prediction = module.predict(
            query,
            base_weights,
            [base_class],
            condition,
            base_bias=base_bias,
        )
        targets = torch.cat(
            (
                torch.zeros(len(base_query_keys), dtype=torch.long),
                torch.ones(len(novel_query_keys), dtype=torch.long),
            )
        )
        loss = F.cross_entropy(prediction["joint_logits"], targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    module.eval()
    if not math.isfinite(final_loss):
        raise RuntimeError("controller training produced a non-finite loss")
    return module, final_loss


def split_target_records(
    system_records: Mapping[int, Sequence[int]], seed: int, system_id: int
) -> dict[int, dict[str, list[int]]]:
    result: dict[int, dict[str, list[int]]] = {}
    for class_id in range(4):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, system_id, class_id, 23])
        )
        permutation = [int(value) for value in rng.permutation(system_records[class_id])]
        adaptation_count = max(1, len(permutation) // 3)
        adaptation = permutation[:adaptation_count]
        query = permutation[adaptation_count:]
        if not adaptation or not query or set(adaptation) & set(query):
            raise RuntimeError("target record split is empty or overlapping")
        result[class_id] = {"adaptation": adaptation, "query": query}
    return result


def mean_prototype_prediction(
    query: torch.Tensor,
    support: torch.Tensor,
    support_labels: torch.Tensor,
    base_weights: torch.Tensor,
    base_bias: torch.Tensor,
    novel_ids: Sequence[int],
    eps: float = 1.0e-8,
) -> dict[str, torch.Tensor]:
    prototypes = torch.stack(
        [support[support_labels == class_id].mean(dim=0) for class_id in novel_ids]
    )
    novel_weights = F.normalize(prototypes, dim=1) * torch.linalg.vector_norm(
        base_weights, dim=1
    ).median().clamp_min(eps)
    base_logits = query @ base_weights.transpose(0, 1) + base_bias
    novel_logits = query @ novel_weights.transpose(0, 1)
    joint_logits = torch.cat((base_logits, novel_logits), dim=1)
    probabilities = torch.softmax(joint_logits, dim=1)
    confidence, prediction_index = probabilities.max(dim=1)
    return {
        "base_logits": base_logits,
        "joint_logits": joint_logits,
        "probabilities": probabilities,
        "confidence": confidence,
        "accepted": torch.ones_like(confidence, dtype=torch.bool),
        "prediction_index": prediction_index,
    }


def cleared_control_prediction(
    query: torch.Tensor,
    support: torch.Tensor,
    support_labels: torch.Tensor,
    base_weights: torch.Tensor,
    base_bias: torch.Tensor,
    novel_ids: Sequence[int],
    eps: float = 1.0e-8,
) -> dict[str, torch.Tensor]:
    means = [support[support_labels == class_id].mean(dim=0) for class_id in novel_ids]
    normalized = F.normalize(torch.stack(means), dim=1)
    frozen_scale = torch.median(torch.linalg.vector_norm(base_weights, dim=1)).clamp_min(eps)
    base_logits = query @ base_weights.transpose(0, 1) + base_bias
    novel_logits = query @ (normalized * frozen_scale).transpose(0, 1)
    logits = torch.cat([base_logits, novel_logits], dim=1)
    probabilities = torch.softmax(logits, dim=1)
    confidence, prediction_index = torch.max(probabilities, dim=1)
    return {
        "base_logits": base_logits,
        "joint_logits": logits,
        "probabilities": probabilities,
        "confidence": confidence,
        "accepted": torch.ones(confidence.shape, dtype=torch.bool),
        "prediction_index": prediction_index,
    }


def harmonic_mean(base_accuracy: float, novel_accuracy: float) -> float:
    denominator = base_accuracy + novel_accuracy
    return 0.0 if denominator == 0.0 else 2.0 * base_accuracy * novel_accuracy / denominator


def equal_mass_ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    order = np.argsort(confidence, kind="mergesort")
    chunks = np.array_split(order, min(bins, len(order)))
    return float(
        sum(
            len(chunk)
            / len(order)
            * abs(float(correct[chunk].mean()) - float(confidence[chunk].mean()))
            for chunk in chunks
            if len(chunk)
        )
    )


def normalized_aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    order = np.argsort(-confidence, kind="mergesort")
    errors = (~correct[order]).astype(np.float64)
    coverage = np.arange(1, len(order) + 1, dtype=np.float64) / len(order)
    risk = np.cumsum(errors) / np.arange(1, len(order) + 1)
    mask = coverage >= 0.5
    selected_coverage = np.concatenate(([0.5], coverage[mask]))
    risk_at_half = float(np.interp(0.5, coverage, risk))
    selected_risk = np.concatenate(([risk_at_half], risk[mask]))
    return float(np.trapz(selected_risk, selected_coverage) / 0.5)


def prediction_metrics(
    prediction: Mapping[str, torch.Tensor], targets: np.ndarray
) -> dict[str, float]:
    probabilities = prediction["probabilities"].detach().cpu().numpy()
    predicted = prediction["prediction_index"].detach().cpu().numpy()
    confidence = prediction["confidence"].detach().cpu().numpy()
    accepted = prediction["accepted"].detach().cpu().numpy().astype(bool)
    correct = predicted == targets
    base_mask = targets < 2
    novel_mask = ~base_mask
    base_accuracy = float(correct[base_mask].mean())
    novel_accuracy = float(correct[novel_mask].mean())
    selected_probability = probabilities[np.arange(len(targets)), targets]
    return {
        "base_accuracy": base_accuracy,
        "novel_accuracy": novel_accuracy,
        "harmonic_mean": harmonic_mean(base_accuracy, novel_accuracy),
        "joint_accuracy": float(correct.mean()),
        "nll": float(-np.log(selected_probability + 1.0e-12).mean()),
        "ece": equal_mass_ece(confidence, correct),
        "aurc": normalized_aurc(confidence, correct),
        "coverage": float(accepted.mean()),
        "selective_risk": float((~correct[accepted]).mean()) if accepted.any() else 1.0,
    }


def hierarchical_interval(
    table: pd.DataFrame,
    *,
    state: str,
    metric: str,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    selected = table[table["state"] == state]
    pivot = selected.pivot_table(
        index=["target_system", "seed"], columns="arm", values=metric, aggfunc="mean"
    )
    delta = (pivot["P"] - pivot["B0"]).rename("delta").reset_index()
    targets = sorted(delta["target_system"].unique().tolist())
    by_target = {
        target: delta.loc[delta["target_system"] == target, "delta"].to_numpy()
        for target in targets
    }
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled_targets = rng.choice(targets, size=len(targets), replace=True)
        values = []
        for target in sampled_targets:
            seed_values = by_target[int(target)]
            values.append(float(rng.choice(seed_values, size=len(seed_values), replace=True).mean()))
        bootstrap[draw] = float(np.mean(values))
    observed = float(delta.groupby("target_system")["delta"].mean().mean())
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return observed, float(lower), float(upper)


def keys_to_json(keys: Sequence[tuple[int, int]], starts: Mapping[int, Sequence[int]]) -> list[list[int]]:
    return [
        [int(record_id), int(window_index), int(starts[record_id][window_index])]
        for record_id, window_index in keys
    ]


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    design_cfg = config["design"]
    source_cfg = config["source_fit"]
    decision_cfg = config["decision"]
    execution_cfg = config["execution"]

    metadata_path = Path(data_cfg["metadata_path"])
    cache_path = Path(data_cfg["cache_path"])
    if sha256_file(metadata_path) != str(data_cfg["metadata_sha256"]):
        raise RuntimeError("metadata SHA-256 does not match the frozen config")
    output_dir = Path(execution_cfg["output_dir"]).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    system_ids = [int(value) for value in data_cfg["target_system_ids"]]
    canonical_maps = {
        int(system_id): as_int_mapping(class_map)
        for system_id, class_map in data_cfg["canonical_maps"].items()
    }
    _, records = load_records(metadata_path, system_ids, canonical_maps)
    feature_cache, start_cache = build_feature_cache(
        cache_path,
        records,
        window_size=int(data_cfg["window_size"]),
        candidate_windows=int(data_cfg["candidate_windows_per_record"]),
    )
    feature_dim = int(next(iter(feature_cache.values())).shape[1])
    if any(matrix.shape[1] != feature_dim for matrix in feature_cache.values()):
        raise RuntimeError("feature extractor produced inconsistent dimensions")

    episode_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    controller_losses: list[float] = []
    maximum_negative_difference = 0.0
    maximum_base_logit_difference = 0.0
    distinct_episode_count = 0
    evaluated_episode_count = 0
    overlap_count = 0
    adapted_parameters = 0

    predictions_path = output_dir / "predictions.csv.gz"
    with gzip.open(predictions_path, "wt", encoding="utf-8", newline="") as prediction_handle:
        prediction_writer = csv.writer(prediction_handle)
        prediction_writer.writerow(
            [
                "target_system",
                "seed",
                "state",
                "episode",
                "arm",
                "query_index",
                "target",
                "prediction",
                "confidence",
                "accepted",
            ]
        )
        for seed in [int(value) for value in design_cfg["seeds"]]:
            for target_system in system_ids:
                source_systems = [value for value in system_ids if value != target_system]
                roles = split_source_roles(
                    records,
                    source_systems,
                    seed,
                    int(data_cfg["max_head_records_per_system_class"]),
                )
                head = fit_frozen_head(
                    roles,
                    feature_cache,
                    seed=seed,
                    logistic_c=float(source_cfg["logistic_c"]),
                    max_iter=int(source_cfg["logistic_max_iter"]),
                )
                conditioner, controller_loss = train_conditioner(
                    roles,
                    source_systems,
                    feature_cache,
                    head,
                    source_cfg,
                    seed=seed,
                    candidate_windows=int(data_cfg["candidate_windows_per_record"]),
                )
                controller_losses.append(controller_loss)
                adapted_parameters = conditioner.trainable_parameter_count
                target_split = split_target_records(records[target_system], seed, target_system)

                for episode in range(int(design_cfg["episodes_per_state"])):
                    episode_rng = np.random.default_rng(
                        np.random.SeedSequence([seed, target_system, episode, 41])
                    )
                    support_keys_by_class = {
                        class_id: sample_keys(
                            target_split[class_id]["adaptation"],
                            int(design_cfg["k_shot"]),
                            int(data_cfg["candidate_windows_per_record"]),
                            episode_rng,
                        )
                        for class_id in (2, 3)
                    }
                    query_keys_by_class = {
                        class_id: sample_keys(
                            target_split[class_id]["query"],
                            int(design_cfg["query_per_class"]),
                            int(data_cfg["candidate_windows_per_record"]),
                            episode_rng,
                        )
                        for class_id in range(4)
                    }
                    original_support_keys = support_keys_by_class[2] + support_keys_by_class[3]
                    support_labels_np = np.asarray(
                        [2] * int(design_cfg["k_shot"])
                        + [3] * int(design_cfg["k_shot"]),
                        dtype=np.int64,
                    )
                    query_keys = sum((query_keys_by_class[class_id] for class_id in range(4)), [])
                    targets = np.concatenate(
                        [
                            np.full(int(design_cfg["query_per_class"]), class_id, dtype=np.int64)
                            for class_id in range(4)
                        ]
                    )

                    for state_index, state in enumerate(design_cfg["support_states"]):
                        effective_support_keys = list(original_support_keys)
                        corruption_mask = np.zeros(len(effective_support_keys), dtype=bool)
                        if state == "outlier":
                            corruption_rng = np.random.default_rng(
                                np.random.SeedSequence(
                                    [seed, target_system, episode, state_index, 53]
                                )
                            )
                            corruption_mask = corruption_rng.random(len(effective_support_keys)) < float(
                                design_cfg["outlier_replacement_probability"]
                            )
                            contamination_records = (
                                target_split[0]["adaptation"]
                                + target_split[1]["adaptation"]
                            )
                            replacement_keys = sample_keys(
                                contamination_records,
                                int(corruption_mask.sum()),
                                int(data_cfg["candidate_windows_per_record"]),
                                corruption_rng,
                            ) if corruption_mask.any() else []
                            replacement_index = 0
                            for index, corrupted in enumerate(corruption_mask):
                                if corrupted:
                                    effective_support_keys[index] = replacement_keys[replacement_index]
                                    replacement_index += 1
                        elif state != "clean":
                            raise ValueError(f"unsupported decisive state: {state}")

                        support_records = {record for record, _ in effective_support_keys}
                        query_records = {record for record, _ in query_keys}
                        overlap = support_records & query_records
                        overlap_count += len(overlap)
                        if overlap:
                            raise RuntimeError(f"support/query record overlap: {sorted(overlap)}")

                        support = torch.from_numpy(
                            vectors_for_keys(
                                effective_support_keys, feature_cache, head.scaler
                            )
                        )
                        support_labels = torch.from_numpy(support_labels_np)
                        query = torch.from_numpy(
                            vectors_for_keys(query_keys, feature_cache, head.scaler)
                        )
                        with torch.no_grad():
                            baseline = mean_prototype_prediction(
                                query,
                                support,
                                support_labels,
                                head.weights,
                                head.bias,
                                (2, 3),
                            )
                            negative = cleared_control_prediction(
                                query,
                                support,
                                support_labels,
                                head.weights,
                                head.bias,
                                (2, 3),
                            )
                            condition = conditioner.condition(
                                support,
                                support_labels,
                                head.weights,
                                (2, 3),
                            )
                            proposed = conditioner.predict(
                                query,
                                head.weights,
                                (0, 1),
                                condition,
                                base_bias=head.bias,
                            )

                        negative_difference = float(
                            torch.max(
                                torch.abs(
                                    baseline["joint_logits"] - negative["joint_logits"]
                                )
                            )
                        )
                        base_difference = max(
                            float(torch.max(torch.abs(proposed["base_logits"] - baseline["base_logits"]))),
                            float(torch.max(torch.abs(negative["base_logits"] - baseline["base_logits"]))),
                        )
                        proposed_difference = float(
                            torch.max(
                                torch.abs(
                                    proposed["joint_logits"] - baseline["joint_logits"]
                                )
                            )
                        )
                        maximum_negative_difference = max(
                            maximum_negative_difference, negative_difference
                        )
                        maximum_base_logit_difference = max(
                            maximum_base_logit_difference, base_difference
                        )
                        distinct_episode_count += int(
                            proposed_difference > float(decision_cfg["distinct_logit_tolerance"])
                        )
                        evaluated_episode_count += 1

                        predictions = {"B0": baseline, "P": proposed, "A7": negative}
                        for arm, prediction in predictions.items():
                            metrics = prediction_metrics(prediction, targets)
                            episode_rows.append(
                                {
                                    "target_system": target_system,
                                    "seed": seed,
                                    "state": state,
                                    "episode": episode,
                                    "arm": arm,
                                    **metrics,
                                    "realized_outlier_rate": float(corruption_mask.mean()),
                                    "adapter_gate": float(condition.adapter_gate) if arm == "P" else 0.0,
                                    "mean_reliability": float(condition.reliability.mean()) if arm == "P" else 1.0,
                                }
                            )
                            predicted = prediction["prediction_index"].cpu().numpy()
                            confidence = prediction["confidence"].cpu().numpy()
                            accepted = prediction["accepted"].cpu().numpy()
                            for query_index in range(len(targets)):
                                prediction_writer.writerow(
                                    [
                                        target_system,
                                        seed,
                                        state,
                                        episode,
                                        arm,
                                        query_index,
                                        int(targets[query_index]),
                                        int(predicted[query_index]),
                                        float(confidence[query_index]),
                                        bool(accepted[query_index]),
                                    ]
                                )

                        manifest_rows.append(
                            {
                                "target_system": target_system,
                                "seed": seed,
                                "state": state,
                                "episode": episode,
                                "support_original": keys_to_json(original_support_keys, start_cache),
                                "support_effective": keys_to_json(effective_support_keys, start_cache),
                                "support_labels": support_labels_np.tolist(),
                                "query": keys_to_json(query_keys, start_cache),
                                "query_labels": targets.tolist(),
                                "corruption_mask": corruption_mask.astype(int).tolist(),
                            }
                        )
                print(
                    f"completed target={target_system} seed={seed} controller_loss={controller_loss:.6f}",
                    flush=True,
                )

    episode_table = pd.DataFrame(episode_rows)
    episode_path = output_dir / "episode_metrics.csv"
    episode_table.to_csv(episode_path, index=False)
    manifest_path = output_dir / "episode_manifest.json"
    json_dump(
        manifest_path,
        {
            "schema_version": 1,
            "protocol_id": config["protocol_id"],
            "experiment_id": config["experiment_id"],
            "metadata_sha256": data_cfg["metadata_sha256"],
            "rows": manifest_rows,
        },
    )
    resolved_config_path = output_dir / "resolved_config.yaml"
    shutil.copy2(config_path, resolved_config_path)

    clean_mean, clean_lower, clean_upper = hierarchical_interval(
        episode_table,
        state="clean",
        metric="harmonic_mean",
        draws=int(decision_cfg["bootstrap_draws"]),
        seed=int(decision_cfg["bootstrap_seed"]),
    )
    outlier_mean, outlier_lower, outlier_upper = hierarchical_interval(
        episode_table,
        state="outlier",
        metric="harmonic_mean",
        draws=int(decision_cfg["bootstrap_draws"]),
        seed=int(decision_cfg["bootstrap_seed"]) + 1,
    )
    base_mean, base_lower, base_upper = hierarchical_interval(
        episode_table.assign(state="all"),
        state="all",
        metric="base_accuracy",
        draws=int(decision_cfg["bootstrap_draws"]),
        seed=int(decision_cfg["bootstrap_seed"]) + 2,
    )

    distinct_fraction = distinct_episode_count / evaluated_episode_count
    execution_gates = {
        "support_query_record_overlap_zero": overlap_count == 0,
        "negative_control_matches_B0": maximum_negative_difference
        <= float(decision_cfg["identity_tolerance"]),
        "base_logits_identical_across_arms": maximum_base_logit_difference
        <= float(decision_cfg["identity_tolerance"]),
        "proposed_path_distinct_fraction_at_least_0_95": distinct_fraction >= 0.95,
        "all_metrics_finite": bool(
            np.isfinite(
                episode_table.select_dtypes(include=[np.number]).to_numpy()
            ).all()
        ),
    }
    execution_pass = all(execution_gates.values())
    supported = (
        execution_pass
        and outlier_lower > 0.0
        and clean_mean >= float(decision_cfg["clean_harmonic_noninferiority_margin"])
        and base_mean >= float(decision_cfg["base_accuracy_noninferiority_margin"])
    )
    refuted = execution_pass and (
        outlier_upper < 0.0
        or base_mean < float(decision_cfg["base_accuracy_noninferiority_margin"])
    )
    outcome = "supported" if supported else "refuted" if refuted else "inconclusive"

    grouped = (
        episode_table.groupby(["state", "arm"], sort=True)[
            ["base_accuracy", "novel_accuracy", "harmonic_mean", "ece", "aurc"]
        ]
        .mean()
        .reset_index()
    )
    artifact_hashes = {
        "episode_metrics.csv": sha256_file(episode_path),
        "episode_manifest.json": sha256_file(manifest_path),
        "predictions.csv.gz": sha256_file(predictions_path),
        "resolved_config.yaml": sha256_file(resolved_config_path),
    }
    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "protocol_id": config["protocol_id"],
        "status": "completed",
        "outcome": outcome,
        "evidence_tier": config["evidence_tier"],
        "claim_eligible": False,
        "targets": len(system_ids),
        "seeds": len(design_cfg["seeds"]),
        "episodes_per_state": int(design_cfg["episodes_per_state"]),
        "states": list(design_cfg["support_states"]),
        "arms": config["arms"],
        "feature_dim": feature_dim,
        "adapted_parameters": adapted_parameters,
        "execution": {
            "conda_environment": execution_cfg["conda_environment"],
            "physical_gpu_indices": execution_cfg["physical_gpu_indices"],
            "multi_gpu": execution_cfg["multi_gpu"],
            "device": execution_cfg["device"],
        },
        "execution_gates": execution_gates,
        "diagnostics": {
            "maximum_negative_control_logit_difference": maximum_negative_difference,
            "maximum_base_logit_difference": maximum_base_logit_difference,
            "proposed_distinct_episode_fraction": distinct_fraction,
            "mean_final_controller_loss": float(np.mean(controller_losses)),
            "mean_realized_outlier_rate": float(
                episode_table.loc[
                    episode_table["state"] == "outlier", "realized_outlier_rate"
                ].mean()
            ),
        },
        "paired_deltas_P_minus_B0": {
            "clean_harmonic_mean": {
                "mean": clean_mean,
                "ci95": [clean_lower, clean_upper],
            },
            "outlier_harmonic_mean": {
                "mean": outlier_mean,
                "ci95": [outlier_lower, outlier_upper],
            },
            "all_state_base_accuracy": {
                "mean": base_mean,
                "ci95": [base_lower, base_upper],
            },
        },
        "arm_state_means": grouped.to_dict(orient="records"),
        "artifact_sha256": artifact_hashes,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "h5py": h5py.__version__,
        },
    }
    summary_path = output_dir / "summary.json"
    json_dump(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
