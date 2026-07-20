#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


CURRENT_DIR = Path(__file__).resolve().parent
WORKTREE_PAPER_ROOT = CURRENT_DIR.parent
PROJECT_REGISTRY = WORKTREE_PAPER_ROOT / "autoresearch" / "project_registry.json"

if PROJECT_REGISTRY.exists():
    registry = json.loads(PROJECT_REGISTRY.read_text(encoding="utf-8"))
    PAPER_ROOT = Path(registry.get("paper_root", WORKTREE_PAPER_ROOT)).resolve()
    REPO_ROOT = Path(registry.get("exec_root", PAPER_ROOT.parent.parent.parent)).resolve()
else:
    PAPER_ROOT = WORKTREE_PAPER_ROOT
    REPO_ROOT = PAPER_ROOT.parent.parent.parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PAPER_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PAPER_ROOT / "code"))

from run_real_dataset_probe import (  # noqa: E402
    DATASETS,
    build_args,
    dataset_num_classes,
    dump_json,
    reduce_signal,
    regularization_total,
)
from src.data_factory import build_data  # noqa: E402
from moe_model import NNSPNMoE  # noqa: E402


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def run_single_ablation(
    dataset_name: str,
    dataset_id: int,
    num_experts: int,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    max_train_batches: int,
    max_test_batches: int,
) -> Dict[str, Any]:
    args_data, args_task = build_args(dataset_id, batch_size)
    data_factory = build_data(args_data, args_task)
    train_loader = data_factory.get_dataloader("train")
    test_loader = data_factory.get_dataloader("test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = dataset_num_classes(dataset_id)
    model = NNSPNMoE(num_classes=num_classes, num_experts=num_experts).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_losses: List[float] = []
    train_batches_used = 0
    for _epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            if step >= max_train_batches:
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
            if step >= max_test_batches:
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

    row = {
        "dataset": dataset_name,
        "dataset_id": dataset_id,
        "num_experts": num_experts,
        "status": "success",
        "num_classes": num_classes,
        "device": str(device),
        "parameter_count": count_parameters(model),
        "epochs": epochs,
        "batch_size": batch_size,
        "train_batches_used": train_batches_used,
        "test_batches_used": test_batches_used,
        "mean_train_loss": float(np.mean(train_losses)) if train_losses else None,
        "test_acc": float(correct / total) if total else 0.0,
        "route_entropy": float(np.mean(route_entropies)) if route_entropies else None,
        "top_expert_weight": float(np.mean(top_weights)) if top_weights else None,
        "expert_pool": model.get_model_description()["experts"],
    }
    dump_json(output_dir / f"{dataset_name.lower()}_{num_experts}experts_summary.json", row)
    data_factory.data.close()
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a bounded 3/5/8 expert ablation probe for NNSPN-MoE.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=["CWRU"])
    parser.add_argument("--expert-counts", nargs="+", type=int, default=[3, 5, 8])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-train-batches", type=int, default=4)
    parser.add_argument("--max-test-batches", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    failed_runs: List[Dict[str, Any]] = []
    successful_expert_counts = set()

    for dataset_name in args.datasets:
        dataset_key = dataset_name.upper()
        if dataset_key not in DATASETS:
            raise ValueError(f"unsupported dataset: {dataset_name}")
        dataset_id = DATASETS[dataset_key]["dataset_id"]
        for num_experts in args.expert_counts:
            try:
                row = run_single_ablation(
                    dataset_name=dataset_key,
                    dataset_id=dataset_id,
                    num_experts=num_experts,
                    output_dir=output_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    max_train_batches=args.max_train_batches,
                    max_test_batches=args.max_test_batches,
                )
                rows.append(row)
                successful_expert_counts.add(num_experts)
            except Exception as exc:
                failed_runs.append(
                    {
                        "dataset": dataset_key,
                        "dataset_id": dataset_id,
                        "num_experts": num_experts,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

    curve_rows: List[Dict[str, Any]] = []
    for num_experts in args.expert_counts:
        subset = [row for row in rows if row["num_experts"] == num_experts]
        if not subset:
            continue
        curve_rows.append(
            {
                "num_experts": num_experts,
                "datasets": sorted({row["dataset"] for row in subset}),
                "parameter_count": int(np.mean([row["parameter_count"] for row in subset])),
                "mean_test_acc": float(np.mean([row["test_acc"] for row in subset])),
                "mean_route_entropy": float(np.mean([row["route_entropy"] for row in subset])),
                "mean_top_expert_weight": float(np.mean([row["top_expert_weight"] for row in subset])),
            }
        )

    requested_counts = sorted(set(args.expert_counts))
    summary = {
        "bound": sorted(successful_expert_counts) == requested_counts and not failed_runs,
        "probe_type": "expert_count_ablation_probe",
        "datasets": [dataset.upper() for dataset in args.datasets],
        "expert_counts": requested_counts,
        "successful_expert_counts": sorted(successful_expert_counts),
        "failed_runs": failed_runs,
        "rows": rows,
        "curve_rows": curve_rows,
        "training_budget": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_train_batches": args.max_train_batches,
            "max_test_batches": args.max_test_batches,
        },
    }
    dump_json(output_dir / "ablation_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
