#!/usr/bin/env python3
"""Freeze the P09-G060 record split, episode manifest, and raw window bank.

This command performs no model fitting and computes no target outcome.  It is
the pre-execution authority for every G060 arm.  Raw metadata ``Id`` is the
leakage group, and the adaptation/query record split is deliberately invariant
across all five experiment seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class Record:
    record_id: int
    system_id: int
    canonical_label: int
    domain_id: str
    sample_rate: float


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


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def as_int_mapping(value: Mapping[Any, Any]) -> dict[int, int]:
    return {int(key): int(mapped) for key, mapped in value.items()}


def stable_seed(*values: int) -> int:
    state = np.random.SeedSequence([int(value) for value in values]).generate_state(
        1, dtype=np.uint32
    )
    return int(state[0])


def evenly_spaced_starts(length: int, window_size: int, count: int) -> np.ndarray:
    if length < window_size:
        raise ValueError(f"record length {length} is smaller than {window_size}")
    if count <= 0:
        raise ValueError("candidate_windows_per_record must be positive")
    starts = np.rint(np.linspace(0, length - window_size, count)).astype(np.int64)
    if np.unique(starts).size != count:
        raise ValueError(
            f"record length {length} cannot provide {count} distinct starts"
        )
    return starts


def standardize_window(window: np.ndarray, epsilon: float) -> np.ndarray:
    value = np.asarray(window, dtype=np.float32)
    if value.ndim == 1:
        value = value[:, None]
    elif value.ndim != 2:
        value = value.reshape(value.shape[0], -1)
    if not np.isfinite(value).all():
        raise ValueError("raw window contains non-finite values")
    mean = value.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    std = value.std(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    standardized = (value - mean) / (std + float(epsilon))
    if not np.isfinite(standardized).all():
        raise ValueError("standardized window contains non-finite values")
    return standardized.astype(np.float32, copy=False)


def _domain_string(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def load_records(
    metadata_path: Path,
    system_ids: Sequence[int],
    canonical_maps: Mapping[int, Mapping[int, int]],
) -> tuple[dict[int, dict[int, list[Record]]], dict[int, Record]]:
    metadata = pd.read_excel(metadata_path)
    required = {"Id", "Dataset_id", "Label", "Domain_id", "Sample_rate"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")

    by_system: dict[int, dict[int, list[Record]]] = {}
    by_id: dict[int, Record] = {}
    for system_id in system_ids:
        class_map = canonical_maps[system_id]
        selected = metadata[
            (metadata["Dataset_id"] == system_id)
            & metadata["Label"].notna()
            & metadata["Label"].map(
                lambda value: pd.notna(value) and int(value) in class_map
            )
        ].copy()
        selected["canonical_label"] = selected["Label"].map(
            lambda value: class_map[int(value)]
        )
        by_system[system_id] = {}
        for class_id in range(4):
            rows = selected[selected["canonical_label"] == class_id]
            records: list[Record] = []
            for row in rows.itertuples(index=False):
                record_id = int(row.Id)
                if record_id in by_id:
                    raise ValueError(f"metadata Id is not unique: {record_id}")
                if pd.isna(row.Sample_rate) or float(row.Sample_rate) <= 0:
                    raise ValueError(f"record {record_id} has invalid sample rate")
                record = Record(
                    record_id=record_id,
                    system_id=system_id,
                    canonical_label=class_id,
                    domain_id=_domain_string(row.Domain_id),
                    sample_rate=float(row.Sample_rate),
                )
                records.append(record)
                by_id[record_id] = record
            records.sort(key=lambda item: item.record_id)
            if len(records) < 3:
                raise ValueError(
                    f"system {system_id} class {class_id} has fewer than three records"
                )
            by_system[system_id][class_id] = records
    return by_system, by_id


def domain_blocked_split(
    records: Sequence[Record], *, seed: int
) -> tuple[list[Record], list[Record]]:
    """Select one third of records across domains without using experiment seed."""
    if len(records) < 3:
        raise ValueError("record split requires at least three records")
    desired = min(len(records) - 1, max(1, len(records) // 3))
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[Record]] = {}
    for record in records:
        buckets.setdefault(record.domain_id, []).append(record)
    for key in buckets:
        permutation = rng.permutation(len(buckets[key]))
        buckets[key] = [buckets[key][int(index)] for index in permutation]
    domain_keys = list(buckets)
    rng.shuffle(domain_keys)
    adaptation: list[Record] = []
    while len(adaptation) < desired:
        progressed = False
        for domain in domain_keys:
            if buckets[domain] and len(adaptation) < desired:
                adaptation.append(buckets[domain].pop())
                progressed = True
        if not progressed:
            raise RuntimeError("domain-blocked split exhausted before desired size")
    adaptation_ids = {record.record_id for record in adaptation}
    query = [record for record in records if record.record_id not in adaptation_ids]
    if not adaptation or not query:
        raise RuntimeError("record split produced an empty role")
    if adaptation_ids & {record.record_id for record in query}:
        raise RuntimeError("record split overlaps")
    return sorted(adaptation, key=lambda item: item.record_id), query


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
    if count < 0 or len(candidates) < count:
        raise ValueError(f"cannot sample {count} keys from pool {len(candidates)}")
    if count == 0:
        return []
    indices = rng.choice(len(candidates), size=count, replace=False)
    return [candidates[int(index)] for index in indices]


def keys_to_json(
    keys: Sequence[tuple[int, int]], starts: Mapping[int, Sequence[int]]
) -> list[list[int]]:
    return [
        [int(record_id), int(window_index), int(starts[record_id][window_index])]
        for record_id, window_index in keys
    ]


def build_episode_core(
    *,
    target_system: int,
    seed: int,
    episode: int,
    split: Mapping[int, Mapping[str, Sequence[int]]],
    starts: Mapping[int, Sequence[int]],
    split_seed: int,
    candidate_windows: int,
    max_k: int,
    query_per_class: int,
) -> dict[str, Any]:
    rng_seed = stable_seed(split_seed, target_system, seed, episode, 1701)
    rng = np.random.default_rng(rng_seed)
    support = {
        str(class_id): keys_to_json(
            sample_keys(
                split[class_id]["adaptation"], max_k, candidate_windows, rng
            ),
            starts,
        )
        for class_id in (2, 3)
    }
    query = {
        str(class_id): keys_to_json(
            sample_keys(
                split[class_id]["query"], query_per_class, candidate_windows, rng
            ),
            starts,
        )
        for class_id in range(4)
    }
    support_records = {item[0] for values in support.values() for item in values}
    query_records = {item[0] for values in query.values() for item in values}
    if support_records & query_records:
        raise RuntimeError("episode core has support/query raw-record overlap")
    return {
        "core_id": f"t{target_system}-s{seed}-e{episode}",
        "target_system": target_system,
        "seed": seed,
        "episode": episode,
        "rng_seed": rng_seed,
        "support_max": support,
        "query": query,
        "query_labels": [
            class_id for class_id in range(4) for _ in range(query_per_class)
        ],
    }


def _selected_support(core: Mapping[str, Any], counts: Mapping[int, int]) -> list[list[int]]:
    return [
        item
        for class_id in (2, 3)
        for item in core["support_max"][str(class_id)][: counts[class_id]]
    ]


def build_cell(
    *,
    core: Mapping[str, Any],
    state: str,
    k_shot: int,
    split: Mapping[int, Mapping[str, Sequence[int]]],
    starts: Mapping[int, Sequence[int]],
    split_seed: int,
    state_index: int,
    candidate_windows: int,
    label_swap_probability: float,
    outlier_probability: float,
    imbalance_ratio: int,
) -> dict[str, Any]:
    target = int(core["target_system"])
    seed = int(core["seed"])
    episode = int(core["episode"])
    rng_seed = stable_seed(
        split_seed, target, seed, k_shot, state_index, episode, 1907
    )
    rng = np.random.default_rng(rng_seed)
    if state == "imbalance":
        minority = max(1, math.floor(k_shot / imbalance_ratio))
        majority_class = 2 if episode % 2 == 0 else 3
        counts = {
            2: k_shot if majority_class == 2 else minority,
            3: k_shot if majority_class == 3 else minority,
        }
    else:
        majority_class = None
        counts = {2: k_shot, 3: k_shot}

    selected = _selected_support(core, counts)
    original_labels = [
        class_id for class_id in (2, 3) for _ in range(counts[class_id])
    ]
    support_labels = list(original_labels)
    corruption_mask = [0] * len(selected)
    label_swap_pairs: list[int] = []
    replacement_keys: list[list[int]] = []

    if state == "label_noise":
        if counts[2] != counts[3]:
            raise RuntimeError("paired label swap requires balanced support")
        pair_mask = rng.random(counts[2]) < label_swap_probability
        for pair_index, swap in enumerate(pair_mask):
            if swap:
                other = counts[2] + pair_index
                support_labels[pair_index], support_labels[other] = (
                    support_labels[other],
                    support_labels[pair_index],
                )
                corruption_mask[pair_index] = 1
                corruption_mask[other] = 1
                label_swap_pairs.append(pair_index)
    elif state == "outlier":
        mask = rng.random(len(selected)) < outlier_probability
        contamination_records = list(split[0]["adaptation"]) + list(
            split[1]["adaptation"]
        )
        replacements = sample_keys(
            contamination_records,
            int(mask.sum()),
            candidate_windows,
            rng,
        )
        replacement_index = 0
        for position, replace_item in enumerate(mask):
            if replace_item:
                key = keys_to_json([replacements[replacement_index]], starts)[0]
                replacement_keys.append([position, *key])
                corruption_mask[position] = 1
                replacement_index += 1
    elif state not in {"clean", "imbalance"}:
        raise ValueError(f"unknown support state: {state}")

    query_records = {
        item[0] for values in core["query"].values() for item in values
    }
    effective_records = {item[0] for item in selected}
    effective_records.update(item[1] for item in replacement_keys)
    if effective_records & query_records:
        raise RuntimeError("cell has support/contamination and query record overlap")
    realized = float(sum(corruption_mask) / len(corruption_mask))
    return {
        "cell_id": f"{core['core_id']}-k{k_shot}-{state}",
        "core_id": core["core_id"],
        "target_system": target,
        "seed": seed,
        "episode": episode,
        "k_shot": k_shot,
        "support_state": state,
        "rng_seed": rng_seed,
        "run_order_seed": stable_seed(rng_seed, 23),
        "support_counts": {"2": counts[2], "3": counts[3]},
        "support_original_labels": original_labels,
        "support_labels": support_labels,
        "corruption_mask": corruption_mask,
        "label_swap_pairs": label_swap_pairs,
        "outlier_replacement_keys": replacement_keys,
        "majority_class": majority_class,
        "realized_corruption_rate": realized,
    }


def expected_cell_count(
    targets: int,
    seeds: int,
    episodes: int,
    k_values: Sequence[int],
    imbalance_values: Sequence[int],
) -> int:
    valid_k_state = 3 * len(k_values) + len(imbalance_values)
    return targets * seeds * episodes * valid_k_state


def _record_payload(record: Record) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "domain_id": record.domain_id,
        "sample_rate": record.sample_rate,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    window_cfg = config["windowing"]
    manifest_cfg = config["manifest"]
    output_cfg = config["outputs"]

    metadata_path = Path(data_cfg["metadata_path"])
    cache_path = Path(data_cfg["cache_path"])
    if sha256_file(metadata_path) != str(data_cfg["metadata_sha256"]):
        raise RuntimeError("metadata SHA-256 differs from the frozen config")
    output_paths = {name: Path(value).resolve() for name, value in output_cfg.items()}
    for path in output_paths.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite protocol artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    system_ids = [int(value) for value in data_cfg["target_system_ids"]]
    canonical_maps = {
        int(system_id): as_int_mapping(class_map)
        for system_id, class_map in data_cfg["canonical_maps"].items()
    }
    records, record_by_id = load_records(metadata_path, system_ids, canonical_maps)
    split_seed = int(manifest_cfg["split_seed"])
    splits: dict[int, dict[int, dict[str, list[int]]]] = {}
    split_payload: dict[str, Any] = {}
    for system_id in system_ids:
        splits[system_id] = {}
        split_payload[str(system_id)] = {}
        for class_id in range(4):
            adaptation, query = domain_blocked_split(
                records[system_id][class_id],
                seed=stable_seed(split_seed, system_id, class_id, 101),
            )
            splits[system_id][class_id] = {
                "adaptation": [item.record_id for item in adaptation],
                "query": [item.record_id for item in query],
            }
            split_payload[str(system_id)][str(class_id)] = {
                "adaptation": [_record_payload(item) for item in adaptation],
                "query": [_record_payload(item) for item in query],
            }

    window_size = int(window_cfg["window_size"])
    candidate_windows = int(window_cfg["candidate_windows_per_record"])
    epsilon = float(window_cfg["epsilon"])
    starts: dict[int, list[int]] = {}
    bank_path = output_paths["window_bank_path"]
    with h5py.File(cache_path, "r") as source, h5py.File(bank_path, "w") as bank:
        bank.attrs["schema_version"] = 1
        bank.attrs["protocol_id"] = config["protocol_id"]
        bank.attrs["metadata_sha256"] = data_cfg["metadata_sha256"]
        bank.attrs["window_size"] = window_size
        bank.attrs["candidate_windows_per_record"] = candidate_windows
        for position, record_id in enumerate(sorted(record_by_id), start=1):
            key = str(record_id)
            if key not in source:
                raise KeyError(f"raw cache is missing metadata Id {record_id}")
            source_dataset = source[key]
            record_starts = evenly_spaced_starts(
                int(source_dataset.shape[0]), window_size, candidate_windows
            )
            windows = np.stack(
                [
                    standardize_window(
                        np.asarray(source_dataset[start : start + window_size]), epsilon
                    )
                    for start in record_starts
                ]
            )
            record = record_by_id[record_id]
            dataset = bank.create_dataset(
                key,
                data=windows,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            dataset.attrs["starts"] = record_starts
            dataset.attrs["system_id"] = record.system_id
            dataset.attrs["canonical_label"] = record.canonical_label
            dataset.attrs["domain_id"] = record.domain_id
            dataset.attrs["sample_rate"] = record.sample_rate
            starts[record_id] = [int(value) for value in record_starts]
            if position % 50 == 0 or position == len(record_by_id):
                print(f"window_bank {position}/{len(record_by_id)}", flush=True)
    window_bank_sha = sha256_file(bank_path)

    seeds = [int(value) for value in manifest_cfg["seeds"]]
    episodes = int(manifest_cfg["episodes_per_valid_cell"])
    k_values = [int(value) for value in manifest_cfg["k_shot"]]
    imbalance_values = [int(value) for value in manifest_cfg["imbalance_valid_k"]]
    state_values = list(manifest_cfg["support_states"])
    cores: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for target in system_ids:
        for seed in seeds:
            for episode in range(episodes):
                core = build_episode_core(
                    target_system=target,
                    seed=seed,
                    episode=episode,
                    split=splits[target],
                    starts=starts,
                    split_seed=split_seed,
                    candidate_windows=candidate_windows,
                    max_k=int(manifest_cfg["max_k_shot"]),
                    query_per_class=int(manifest_cfg["query_per_class"]),
                )
                cores.append(core)
                for state_index, state in enumerate(state_values):
                    valid_k = imbalance_values if state == "imbalance" else k_values
                    for k_shot in valid_k:
                        cells.append(
                            build_cell(
                                core=core,
                                state=state,
                                k_shot=k_shot,
                                split=splits[target],
                                starts=starts,
                                split_seed=split_seed,
                                state_index=state_index,
                                candidate_windows=candidate_windows,
                                label_swap_probability=float(
                                    manifest_cfg["label_swap_probability"]
                                ),
                                outlier_probability=float(
                                    manifest_cfg["outlier_replacement_probability"]
                                ),
                                imbalance_ratio=int(manifest_cfg["imbalance_ratio"]),
                            )
                        )
        print(f"manifest target={target} complete", flush=True)

    expected_cores = len(system_ids) * len(seeds) * episodes
    expected_cells = expected_cell_count(
        len(system_ids), len(seeds), episodes, k_values, imbalance_values
    )
    if len(cores) != expected_cores or len(cells) != expected_cells:
        raise RuntimeError("manifest cardinality differs from the frozen design")

    fold_overlap = 0
    for target in system_ids:
        adaptation_ids = {
            record_id
            for class_id in range(4)
            for record_id in splits[target][class_id]["adaptation"]
        }
        query_ids = {
            record_id
            for class_id in range(4)
            for record_id in splits[target][class_id]["query"]
        }
        fold_overlap += len(adaptation_ids & query_ids)
    if fold_overlap:
        raise RuntimeError("fold-wide adaptation/query raw-record overlap")

    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "protocol_id": config["protocol_id"],
        "metadata_sha256": data_cfg["metadata_sha256"],
        "config_sha256": sha256_file(config_path),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "window_bank_sha256": window_bank_sha,
        "target_system_ids": system_ids,
        "canonical_maps": {
            str(system_id): {
                str(raw): mapped for raw, mapped in canonical_maps[system_id].items()
            }
            for system_id in system_ids
        },
        "base_class_ids": [int(value) for value in data_cfg["base_class_ids"]],
        "novel_class_ids": [int(value) for value in data_cfg["novel_class_ids"]],
        "record_split_scope": manifest_cfg["record_split_scope"],
        "record_split_seed_independent": bool(
            manifest_cfg["record_split_seed_independent"]
        ),
        "record_splits": split_payload,
        "design": manifest_cfg,
        "windowing": window_cfg,
        "episode_cores": cores,
        "cells": cells,
    }
    manifest_path = output_paths["manifest_path"]
    write_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)

    rates: dict[str, float] = {}
    for state in state_values:
        selected = [cell for cell in cells if cell["support_state"] == state]
        rates[state] = float(
            np.mean([cell["realized_corruption_rate"] for cell in selected])
        )
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "protocol_id": config["protocol_id"],
        "status": "PASS",
        "target_systems": len(system_ids),
        "seeds": len(seeds),
        "episode_cores": len(cores),
        "cells": len(cells),
        "expected_episode_cores": expected_cores,
        "expected_cells": expected_cells,
        "fold_wide_raw_record_overlap_count": fold_overlap,
        "record_split_seed_independent": True,
        "nested_support_sets_by_prefix": True,
        "query_identity_shared_across_k_and_state": True,
        "mean_realized_corruption_rate": rates,
        "metadata_sha256": data_cfg["metadata_sha256"],
        "manifest_sha256": manifest_sha,
        "window_bank_sha256": window_bank_sha,
    }
    write_json(output_paths["integrity_report_path"], report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
