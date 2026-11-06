"""Replay Experiment A's selected checkpoint without fitting or selecting a model.

Use the original configuration/overrides, selected checkpoint, per-seed metrics and
split file. The explicit simulation profile verifies the same code on simulated
Dummy_Data records; the default MFPT profile retains Experiment A unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
import torch


NUM_CLASSES = 3
LOGIT_ATOL = 1e-6
LOGIT_RTOL = 1e-5
METRIC_ATOL = 1e-6


def export_predictions(task, loader, checkpoint: Path, output: Path) -> dict:
    """Keep native test batches and export their actual file/window coordinates."""
    dataset = loader.dataset
    expected = Counter(
        (int(item["file_id"]), int(item["window_id"]))
        for item in dataset.get_file_windows_list()
    )
    observed: Counter = Counter()
    targets, decisions = [], []
    linear_reference = task.args_model.name == "GlobalAverageLinear"
    max_abs_logit_error = 0.0 if linear_reference else None
    rows = []
    task.eval()

    # Same_system_Sampler keeps evaluation order and the short final batch.
    # In particular, do not rebatch TimesNet's batch-dependent period selection.
    with torch.inference_mode(), (output / "predictions.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "batch_index", "batch_position", "file_id", "window_id",
            "raw_start", "raw_end", "target", "prediction",
            "logit_0", "logit_1", "logit_2", "checkpoint",
        ])
        writer.writeheader()
        for batch_index, batch in enumerate(loader):
            logits = task(batch).detach().cpu()
            target = batch["y"].detach().cpu().reshape(-1)
            if logits.shape != (len(target), NUM_CLASSES):
                raise ValueError(f"Experiment A expects [B,3] logits, got {logits.shape}")
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Selected-state logits contain NaN or Inf")
            if not torch.all((target >= 0) & (target < NUM_CLASSES)):
                raise ValueError("Experiment A requires the unchanged label IDs 0,1,2")
            decision = logits.argmax(dim=1)

            if linear_reference:
                x = batch["x"].detach().cpu().numpy().astype(np.float32, copy=False)
                weight = task.network.classifier.weight.detach().cpu().numpy()
                bias = task.network.classifier.bias.detach().cpu().numpy()
                reference = x.mean(axis=1, dtype=np.float32) @ weight.T + bias
                difference = float(np.max(np.abs(logits.numpy() - reference)))
                max_abs_logit_error = max(max_abs_logit_error, difference)
                np.testing.assert_allclose(
                    logits.numpy(), reference, atol=LOGIT_ATOL, rtol=LOGIT_RTOL
                )
                if not np.array_equal(decision.numpy(), reference.argmax(axis=1)):
                    raise AssertionError("Native and explicit mean-linear class decisions differ")

            file_ids = batch["file_id"].tolist()
            window_ids = batch["window_id"].tolist()
            if len(file_ids) != len(target) or len(window_ids) != len(target):
                raise ValueError("Every prediction must retain its own file and window ID")
            for position in range(len(target)):
                file_id = int(file_ids[position])
                window_id = int(window_ids[position])
                start, end = dataset.dataset_dict[file_id].window_intervals[window_id]
                row = {
                    "batch_index": batch_index,
                    "batch_position": position,
                    "file_id": file_id,
                    "window_id": window_id,
                    "raw_start": int(start),
                    "raw_end": int(end),
                    "target": int(target[position]),
                    "prediction": int(decision[position]),
                    "logit_0": float(logits[position, 0]),
                    "logit_1": float(logits[position, 1]),
                    "logit_2": float(logits[position, 2]),
                    "checkpoint": str(checkpoint),
                }
                writer.writerow(row)
                rows.append(row)
                observed[(file_id, window_id)] += 1
            targets.extend(target.tolist())
            decisions.extend(decision.tolist())

    if observed != expected:
        raise AssertionError(
            f"Prediction population differs: missing={expected - observed}, "
            f"extra={observed - expected}"
        )
    if not targets:
        raise ValueError("The declared test population is empty")
    y = np.asarray(targets, dtype=np.int64)
    pred = np.asarray(decisions, dtype=np.int64)
    confusion = np.bincount(NUM_CLASSES * y + pred, minlength=NUM_CLASSES**2).reshape(
        NUM_CLASSES, NUM_CLASSES
    )
    tp = confusion.diagonal()
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    # Fixed full class list; zero-denominator F1 is defined as zero.
    f1 = np.divide(2 * tp, denominator, out=np.zeros(NUM_CLASSES), where=denominator != 0)
    with (output / "per_record.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_id", "windows", "window_accuracy"])
        writer.writeheader()
        for file_id in sorted(dataset.dataset_dict):
            selected = [row for row in rows if row["file_id"] == file_id]
            writer.writerow({
                "file_id": int(file_id),
                "windows": len(selected),
                "window_accuracy": sum(r["target"] == r["prediction"] for r in selected) / len(selected),
            })
    return {
        "windows": len(targets),
        "records": len(dataset.dataset_dict),
        "accuracy": float(tp.sum() / confusion.sum()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": confusion.tolist(),
        "max_abs_logit_error": max_abs_logit_error,
        "linear_reference_checked": linear_reference,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", choices=("mfpt", "simulation"), default="mfpt")
    parser.add_argument("--local-config")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Exact selected checkpoint returned by the original run; trust only your own checkpoint.")
    parser.add_argument("--reported-metrics", required=True, type=Path,
                        help="Original test_result_<iteration>.csv, not the across-seed mean.")
    parser.add_argument("--split-manifest", required=True, type=Path,
                        help="Original run's split file; it is never overwritten.")
    parser.add_argument("--seed", required=True, type=int, choices=(17, 18, 19))
    parser.add_argument("--output", required=True, type=Path,
                        help="New directory for prediction export and independent recomputation.")
    args = parser.parse_args(argv)

    from pytorch_lightning import seed_everything
    from phmfactory.config import analyze_config
    from src.configs.config_utils import dict_to_namespace
    from src.data_factory import build_data
    from src.model_factory import build_model
    from src.task_factory import build_task
    from src.runtime.classification import _close_data_factory, _namespaces

    analysis = analyze_config(args.config, local_config=args.local_config,
                              override_values=args.override)
    config = analysis.runtime_config()
    if (config["task"]["type"], config["task"]["name"], config["task"]["loss"]) != (
        "DG", "classification", "CE"
    ):
        raise ValueError("This exporter is limited to Experiment A's DG/classification CE path")
    if config["model"]["name"] not in {"GlobalAverageLinear", "TimesNet"}:
        raise ValueError("Use an Experiment A mean-linear or fixed TimesNet configuration")
    if config["trainer"]["device"] != "cpu" or config["trainer"]["devices"] != 1:
        raise ValueError("Experiment A replay requires the declared single CPU execution")
    required_data = {
        "window_size": 2048, "num_window": 16, "batch_size": 32,
        "window_sampling_strategy": "evenly_spaced", "window_sampling_seed": 17,
        "normalization": "none", "dtype": "float32",
        "train_noise_snr": None, "evaluation_noise_snr": None,
    }
    if args.profile == "simulation":
        required_data.update(window_size=128, num_window=8, batch_size=16)
    for key, expected_value in required_data.items():
        if config["data"][key] != expected_value:
            raise ValueError(f"Experiment A fixes data.{key}={expected_value!r}")
    checkpoint = args.checkpoint.resolve(strict=True)
    original_split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    with args.reported_metrics.open(encoding="utf-8", newline="") as handle:
        reported = list(csv.DictReader(handle))
    if len(reported) != 1:
        raise ValueError("Provide the original one-row, per-seed test metrics file")
    args.output.mkdir(parents=True, exist_ok=False)

    # DG does not support read_only partition manifests. Reconstruct with the
    # unchanged selector and compare membership, writing only into the new directory.
    config["data"]["split"]["manifest_path"] = str(args.output / "replay_split.json")
    config["data"]["cache_dir"] = str(args.output / "cache")
    namespaces = _namespaces(dict_to_namespace(config))
    environment, data, model, task_args, trainer = namespaces
    seed_everything(args.seed)
    factory = build_data(data, task_args)
    try:
        replay_split = json.loads((args.output / "replay_split.json").read_text(encoding="utf-8"))
        for key in ("strategy", "task_type", "seed", "group_key", "stratify_key",
                    "test_policy", "fractions", "normalization", "splits"):
            if original_split[key] != replay_split[key]:
                raise AssertionError(f"Replay differs from the original split field {key!r}")
        metadata = factory.get_metadata()
        network = build_model(model, metadata=metadata)
        task = build_task(args_task=task_args, network=network, args_data=data,
                          args_model=model, args_trainer=trainer,
                          args_environment=environment, metadata=metadata)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        task.load_state_dict(saved["state_dict"], strict=True)
        task.cpu()
        loader = factory.get_dataloader("test")
        expected_name = "RM_007_MFPT" if args.profile == "mfpt" else "Dummy_Data"
        if loader.dataset.expected_dataset_names() != (expected_name,):
            raise ValueError(f"Replay profile {args.profile!r} requires {expected_name}")
        if args.profile == "simulation":
            if any(metadata[file_id]["Data_Origin"] != "simulation"
                   for file_id in loader.dataset.dataset_dict):
                raise ValueError("Simulation replay requires explicitly simulated metadata")
        result = export_predictions(task, loader, checkpoint, args.output)
        expected_records = 6 if args.profile == "mfpt" else 9
        expected_windows = expected_records * required_data["num_window"]
        if result["records"] != expected_records or result["windows"] != expected_windows:
            raise AssertionError(f"Replay requires {expected_records} records and {expected_windows} windows")
        for metric, key in (("accuracy", f"test_acc_{expected_name}"), ("macro_f1", f"test_f1_{expected_name}")):
            value = float(reported[0][key])
            if not np.isfinite(value) or abs(result[metric] - value) > METRIC_ATOL:
                raise AssertionError(f"{key}: reported={value}, independently recomputed={result[metric]}")
        result.update({"profile": args.profile, "checkpoint": str(checkpoint), "seed": args.seed,
                       "source_metrics": str(args.reported_metrics.resolve()),
                       "source_split": str(args.split_manifest.resolve()),
                       "config": config,
                       "torch_version": torch.__version__,
                       "logit_atol": LOGIT_ATOL, "logit_rtol": LOGIT_RTOL,
                       "metric_atol": METRIC_ATOL})
        (args.output / "recomputed_metrics.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(f"predictions={args.output / 'predictions.csv'}")
        print(f"recomputed_metrics={args.output / 'recomputed_metrics.json'}")
    finally:
        _close_data_factory(factory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
