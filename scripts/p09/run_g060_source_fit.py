#!/usr/bin/env python3
"""Fit the 75 unique source-only inner representations for P09-G060."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.p09.g060_representation import (
    HSEDLinearGlobalHead,
    WindowBank,
    all_record_ids,
    model_state_sha256,
    sha256_file,
    stable_seed,
    trainable_parameter_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--excluded", nargs=2, type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.all == (args.excluded is not None):
        parser.error("choose exactly one of --all or --excluded A B")
    if args.excluded is not None and args.seed is None:
        parser.error("--seed is required with --excluded")
    if args.all and args.seed is not None:
        parser.error("--seed is not accepted with --all")
    return args


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def device_from_config(config: Mapping[str, Any], override: str | None) -> torch.device:
    execution = config["execution"]
    requested = override or str(execution["device"])
    if requested == "cuda":
        if execution.get("multi_gpu") is not False:
            raise RuntimeError("multi_gpu must be false")
        indices = execution.get("physical_gpu_indices")
        if not isinstance(indices, list) or len(indices) != 1 or int(indices[0]) == 2:
            raise RuntimeError("source fit requires one permitted physical GPU")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible != str(execution["cuda_visible_devices"]):
            raise RuntimeError("CUDA_VISIBLE_DEVICES differs from the frozen config")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one visible CUDA device is required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != str(
            execution["cublas_workspace_config"]
        ):
            raise RuntimeError("CUBLAS_WORKSPACE_CONFIG differs from the frozen config")
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    return torch.device("cpu")


def epoch_keys(
    bank: WindowBank,
    training_systems: Sequence[int],
    *,
    seed: int,
    epoch: int,
    per_system_class: int,
) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    for system_id in training_systems:
        for class_id in (0, 1):
            candidates = bank.keys_for_system_class(system_id, class_id)
            if len(candidates) < per_system_class:
                raise RuntimeError(
                    f"system {system_id} class {class_id} has only {len(candidates)} windows"
                )
            rng = np.random.default_rng(
                stable_seed(seed, epoch, system_id, class_id, 3109)
            )
            indices = rng.choice(
                len(candidates), size=per_system_class, replace=False
            )
            selected.extend(candidates[int(index)] for index in indices)
    return selected


def grouped_batches(
    bank: WindowBank,
    keys: Sequence[tuple[int, int]],
    *,
    seed: int,
    epoch: int,
    batch_size: int,
) -> list[list[tuple[int, int]]]:
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for key in keys:
        groups[bank.records[key[0]].channels].append(key)
    batches: list[list[tuple[int, int]]] = []
    for channels, values in sorted(groups.items()):
        rng = np.random.default_rng(stable_seed(seed, epoch, channels, 3203))
        rng.shuffle(values)
        batches.extend(
            [values[start : start + batch_size] for start in range(0, len(values), batch_size)]
        )
    rng = np.random.default_rng(stable_seed(seed, epoch, 3251))
    rng.shuffle(batches)
    return batches


def fit_trajectory(
    *,
    bank: WindowBank,
    config: Mapping[str, Any],
    excluded: Sequence[int],
    seed: int,
    device: torch.device,
    destination: Path,
    config_sha256: str,
) -> dict[str, Any]:
    systems = [int(value) for value in config["data"]["target_system_ids"]]
    excluded_set = set(int(value) for value in excluded)
    if len(excluded_set) != 2 or not excluded_set <= set(systems):
        raise ValueError("an inner trajectory excludes two distinct registered systems")
    training_systems = sorted(set(systems) - excluded_set)
    fit_cfg = config["source_fit"]
    checkpoint_epochs = [int(value) for value in fit_cfg["checkpoint_epochs"]]
    if max(checkpoint_epochs) != int(fit_cfg["max_epochs"]):
        raise ValueError("max_epochs must equal the last checkpoint epoch")

    torch.manual_seed(stable_seed(seed, *sorted(excluded_set), 3301))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(stable_seed(seed, *sorted(excluded_set), 3301))
        torch.cuda.reset_peak_memory_stats()
    model = HSEDLinearGlobalHead(config["representation"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(fit_cfg["learning_rate"]),
        weight_decay=float(fit_cfg["weight_decay"]),
    )
    training_record_ids = all_record_ids(bank, training_systems, (0, 1))
    novel_record_ids = all_record_ids(bank, training_systems, (2, 3))
    if set(training_record_ids) & set(novel_record_ids):
        raise RuntimeError("base and novel gradient record sets overlap")
    if any(bank.records[value].system_id in excluded_set for value in training_record_ids):
        raise RuntimeError("excluded system entered the gradient record set")

    history: list[dict[str, float | int]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    for epoch in range(1, int(fit_cfg["max_epochs"]) + 1):
        model.train()
        keys = epoch_keys(
            bank,
            training_systems,
            seed=seed,
            epoch=epoch,
            per_system_class=int(fit_cfg["windows_per_system_class_per_epoch"]),
        )
        batches = grouped_batches(
            bank,
            keys,
            seed=seed,
            epoch=epoch,
            batch_size=int(fit_cfg["batch_size"]),
        )
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        for batch_index, batch_keys in enumerate(batches):
            torch.manual_seed(stable_seed(seed, epoch, batch_index, 3407))
            windows_np, sample_rates_np, labels_np = bank.batch(batch_keys)
            if not set(np.unique(labels_np).tolist()) <= {0, 1}:
                raise RuntimeError("canonical novel label reached gradient fit")
            windows = torch.from_numpy(windows_np).to(device)
            sample_rates = torch.from_numpy(sample_rates_np).to(device)
            labels = torch.from_numpy(labels_np).to(device)
            logits, _ = model(windows, sample_rates)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_keys)
            total_correct += int((logits.argmax(dim=1) == labels).sum().detach())
            total_count += len(batch_keys)
        row = {
            "epoch": epoch,
            "loss": total_loss / total_count,
            "accuracy": total_correct / total_count,
            "samples": total_count,
        }
        history.append(row)
        if epoch in checkpoint_epochs:
            payload = {
                "schema_version": 1,
                "experiment_id": config["experiment_id"],
                "protocol_id": config["protocol_id"],
                "status": "completed",
                "stage": "source_inner_representation",
                "excluded_systems": sorted(excluded_set),
                "training_systems": training_systems,
                "seed": seed,
                "epoch": epoch,
                "training_record_ids": training_record_ids,
                "gradient_canonical_classes": [0, 1],
                "canonical_novel_gradient_records": 0,
                "config_sha256": config_sha256,
                "manifest_sha256": config["data"]["manifest_sha256"],
                "window_bank_sha256": config["data"]["window_bank_sha256"],
                "model_state_sha256": model_state_sha256(model),
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
            }
            checkpoint_path = destination / f"epoch_{epoch:03d}.pt"
            torch.save(payload, checkpoint_path)
            checkpoint_rows.append(
                {
                    "epoch": epoch,
                    "path": checkpoint_path.name,
                    "sha256": sha256_file(checkpoint_path),
                    "model_state_sha256": payload["model_state_sha256"],
                }
            )
        print(
            f"excluded={sorted(excluded_set)} seed={seed} epoch={epoch} "
            f"loss={row['loss']:.6f} accuracy={row['accuracy']:.4f}",
            flush=True,
        )

    elapsed = time.perf_counter() - start_time
    peak_memory = (
        int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
    )
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "protocol_id": config["protocol_id"],
        "status": "completed",
        "excluded_systems": sorted(excluded_set),
        "training_systems": training_systems,
        "seed": seed,
        "epochs": int(fit_cfg["max_epochs"]),
        "training_record_ids": training_record_ids,
        "training_record_count": len(training_record_ids),
        "gradient_canonical_classes": [0, 1],
        "canonical_novel_gradient_records": 0,
        "excluded_system_gradient_records": 0,
        "trainable_parameters": trainable_parameter_count(model),
        "elapsed_seconds": elapsed,
        "peak_accelerator_memory_bytes": peak_memory,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "history": history,
        "checkpoints": checkpoint_rows,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    write_json(destination / "fit_report.json", report)
    return report


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_sha = sha256_file(config_path)
    data_cfg = config["data"]
    fit_cfg = config["source_fit"]
    if sha256_file(Path(data_cfg["manifest_path"])) != data_cfg["manifest_sha256"]:
        raise RuntimeError("episode-manifest SHA-256 mismatch")
    device = device_from_config(config, args.device)
    torch.use_deterministic_algorithms(True)
    output_root = Path(fit_cfg["output_dir"]).resolve()
    marker_path = output_root / "source_fit_contract.json"
    if output_root.exists():
        if not args.resume:
            raise FileExistsError(
                f"source-fit output exists; use --resume after audit: {output_root}"
            )
        if not marker_path.exists():
            raise RuntimeError("source-fit root lacks its contract marker")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("config_sha256") != config_sha:
            raise RuntimeError("resume config differs from the source-fit contract")
    else:
        output_root.mkdir(parents=True)
        write_json(
            marker_path,
            {
                "schema_version": 1,
                "experiment_id": config["experiment_id"],
                "protocol_id": config["protocol_id"],
                "config_sha256": config_sha,
                "manifest_sha256": data_cfg["manifest_sha256"],
                "window_bank_sha256": data_cfg["window_bank_sha256"],
                "device": str(device),
            },
        )

    systems = [int(value) for value in data_cfg["target_system_ids"]]
    seeds = [int(value) for value in fit_cfg["seeds"]]
    if args.all:
        jobs = [
            (pair, seed)
            for pair in itertools.combinations(systems, 2)
            for seed in seeds
        ]
    else:
        jobs = [(tuple(sorted(args.excluded)), int(args.seed))]
        if args.seed not in seeds:
            raise ValueError("requested seed is not frozen in the config")

    with WindowBank(
        Path(data_cfg["window_bank_path"]),
        expected_sha256=data_cfg["window_bank_sha256"],
    ) as bank:
        for excluded, seed in jobs:
            pair_name = "exclude_" + "_".join(f"{value:02d}" for value in excluded)
            destination = output_root / pair_name / f"seed_{seed}"
            if destination.exists():
                report_path = destination / "fit_report.json"
                if args.resume and report_path.exists():
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    if report.get("status") == "completed":
                        print(f"skip completed {pair_name} seed={seed}", flush=True)
                        continue
                raise FileExistsError(f"trajectory output already exists: {destination}")
            partial = destination.with_name(destination.name + ".partial")
            if partial.exists():
                raise FileExistsError(f"failed partial trajectory is retained: {partial}")
            partial.mkdir(parents=True)
            fit_trajectory(
                bank=bank,
                config=config,
                excluded=excluded,
                seed=seed,
                device=device,
                destination=partial,
                config_sha256=config_sha,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial.rename(destination)

    completed = list(output_root.glob("exclude_*/seed_*/fit_report.json"))
    write_json(
        output_root / "source_fit_progress.json",
        {
            "schema_version": 1,
            "status": "completed" if len(completed) == 75 else "partial",
            "completed_trajectories": len(completed),
            "expected_trajectories": 75,
            "config_sha256": config_sha,
        },
    )
    print(f"source-fit trajectories completed={len(completed)}/75", flush=True)


if __name__ == "__main__":
    main()
