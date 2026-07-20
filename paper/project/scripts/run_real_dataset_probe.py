#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


PAPER_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PAPER_ROOT.parents[2]
DATA_ROOT = Path("/home/user/data/PHMbenchdata/PHM-Vibench")
METADATA_PATH = DATA_ROOT / "metadata.xlsx"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PAPER_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PAPER_ROOT / "code"))

from src.data_factory import build_data  # noqa: E402
from moe_model import NNSPNMoE  # noqa: E402


DATASETS = {
    "CWRU": {"dataset_id": 1},
    "XJTU": {"dataset_id": 2},
    "THU_006": {"dataset_id": 6},
}


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def dataset_num_classes(dataset_id: int) -> int:
    df = pd.read_excel(METADATA_PATH)
    subset = df[(df["Dataset_id"] == dataset_id) & (df["Label"] != -1)]
    if subset.empty:
        raise ValueError(f"no metadata rows for dataset_id={dataset_id}")
    return int(subset["Label"].max()) + 1


def build_args(dataset_id: int, batch_size: int) -> tuple[SimpleNamespace, SimpleNamespace]:
    args_data = SimpleNamespace(
        data_dir=str(DATA_ROOT),
        metadata_file="metadata.xlsx",
        batch_size=batch_size,
        num_workers=0,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        normalization="standardization",
        window_size=1024,
        stride=256,
        num_window=8,
        dtype="float32",
        pin_memory=False,
    )
    args_task = SimpleNamespace(
        name="classification",
        type="DG",
        target_system_id=[dataset_id],
        target_domain_num=1,
        source_domain_id=None,
        target_domain_id=None,
        loss="CE",
        metrics=["acc"],
        optimizer="adam",
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        epochs=1,
        lr=0.001,
        weight_decay=0.0001,
    )
    return args_data, args_task


def reduce_signal(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.mean(dim=-1)
    return x


def regularization_total(metadata: Dict[str, Any], device: torch.device) -> torch.Tensor:
    losses = metadata.get("regularization_losses", {})
    if not losses:
        return torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)
    for value in losses.values():
        if torch.is_tensor(value):
            total = total + value
        else:
            total = total + torch.tensor(float(value), device=device)
    return total


def run_single_dataset(
    dataset_name: str,
    dataset_id: int,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    max_train_batches: int,
    max_test_batches: int,
    required_test_acc: float,
) -> Dict[str, Any]:
    args_data, args_task = build_args(dataset_id, batch_size)
    data_factory = build_data(args_data, args_task)
    train_loader = data_factory.get_dataloader("train")
    test_loader = data_factory.get_dataloader("test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = dataset_num_classes(dataset_id)
    model = NNSPNMoE(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_losses: List[float] = []
    train_batches_used = 0
    for _epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            if max_train_batches > 0 and step >= max_train_batches:
                break
            x = reduce_signal(batch["x"].float().to(device))
            y = batch["y"].long().to(device)
            optimizer.zero_grad()
            logits, metadata = model(x)
            loss = criterion(logits, y) + 0.1 * regularization_total(metadata, device)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            train_batches_used += 1

    model.eval()
    correct = 0
    total = 0
    route_entropies: List[float] = []
    top_weights: List[float] = []
    test_batches_used = 0
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            if max_test_batches > 0 and step >= max_test_batches:
                break
            x = reduce_signal(batch["x"].float().to(device))
            y = batch["y"].long().to(device)
            logits, metadata = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == y).sum().item())
            total += int(y.numel())
            routing = metadata["routing_weights"]
            entropy = -(routing * torch.log(torch.clamp(routing, min=1e-12))).sum(dim=1)
            route_entropies.extend(float(item) for item in entropy.detach().cpu())
            top_weights.extend(float(item) for item in routing.max(dim=1).values.detach().cpu())
            test_batches_used += 1

    summary = {
        "dataset": dataset_name,
        "dataset_id": dataset_id,
        "status": "success",
        "num_classes": num_classes,
        "device": str(device),
        "signal_reduce": "mean_channel",
        "epochs": epochs,
        "batch_size": batch_size,
        "train_batches_used": train_batches_used,
        "test_batches_used": test_batches_used,
        "mean_train_loss": float(np.mean(train_losses)) if train_losses else None,
        "test_acc": float(correct / total) if total else 0.0,
        "route_entropy": float(np.mean(route_entropies)) if route_entropies else None,
        "top_expert_weight": float(np.mean(top_weights)) if top_weights else None,
        "required_test_acc": required_test_acc,
        "threshold_pass": bool(total) and float(correct / total) >= required_test_acc,
        "in_domain_98_pass": bool(total) and required_test_acc >= 0.98 and float(correct / total) >= required_test_acc,
    }
    dump_json(output_dir / f"{dataset_name.lower()}_probe_summary.json", summary)
    data_factory.data.close()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal real-data MOE probe on CWRU/XJTU.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=["CWRU", "XJTU"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-train-batches", type=int, default=12)
    parser.add_argument("--max-test-batches", type=int, default=12)
    parser.add_argument("--required-test-acc", type=float, default=0.98)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    successful: List[str] = []
    failed: List[str] = []
    rows: List[Dict[str, Any]] = []

    for dataset_name in args.datasets:
        dataset_key = dataset_name.upper()
        if dataset_key not in DATASETS:
            raise ValueError(f"unsupported dataset: {dataset_name}")
        try:
            row = run_single_dataset(
                dataset_name=dataset_key,
                dataset_id=DATASETS[dataset_key]["dataset_id"],
                output_dir=output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_train_batches=args.max_train_batches,
                max_test_batches=args.max_test_batches,
                required_test_acc=args.required_test_acc,
            )
            successful.append(dataset_key)
            rows.append(row)
        except Exception as exc:
            failed.append(dataset_key)
            rows.append(
                {
                    "dataset": dataset_key,
                    "dataset_id": DATASETS[dataset_key]["dataset_id"],
                    "status": "failed",
                    "error": str(exc),
                }
            )

    accuracies = [row["test_acc"] for row in rows if row.get("status") == "success" and row.get("test_acc") is not None]
    threshold_pass_datasets = [row["dataset"] for row in rows if row.get("status") == "success" and row.get("threshold_pass")]
    threshold_failed_datasets = [
        row["dataset"]
        for row in rows
        if row.get("status") != "success" or not row.get("threshold_pass")
    ]
    summary = {
        "bound": len(successful) == len(args.datasets),
        "probe_type": "real_data_minimal_probe",
        "datasets": args.datasets,
        "successful_datasets": successful,
        "failed_datasets": failed,
        "success_count": len(successful),
        "failed_count": len(failed),
        "accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "mean_test_acc": float(np.mean(accuracies)) if accuracies else 0.0,
        "required_test_acc": args.required_test_acc,
        "threshold_pass_datasets": threshold_pass_datasets,
        "threshold_failed_datasets": threshold_failed_datasets,
        "threshold_pass": len(threshold_failed_datasets) == 0 and bool(successful),
        "in_domain_98_pass": args.required_test_acc >= 0.98 and len(threshold_failed_datasets) == 0 and bool(successful),
        "rows": rows,
        "training_budget": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_train_batches": args.max_train_batches,
            "max_test_batches": args.max_test_batches,
        },
    }
    dump_json(output_dir / "dataset_bridge_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
