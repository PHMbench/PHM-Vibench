"""Configure the existing frozen/Tent streams through the public research entry."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import yaml
from pytorch_lightning import seed_everything
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from phmfactory.source_stream import run_source_only_stream
from phmfactory.tent_stream import run_tent_stream
from src.config_schema import AdaptationProtocolConfig
from src.configs.config_utils import dict_to_namespace
from src.data_factory import build_data
from src.model_factory import build_model
from src.runtime.classification import _close_data_factory
from src.task_factory.Components.metrics import grouped_classification_metrics
from src.utils.run_summary import write_run_summary

PHASES = ("evaluate",)


def execute(config: dict, phase: str, output: Path, *, checkpoint: Path | None = None):
    """Evaluate one fixed source checkpoint; never fit/select on target labels."""
    if phase != "evaluate":
        raise ValueError("TTA/stream implements phase=evaluate")
    task = config["task"]
    protocol = AdaptationProtocolConfig.model_validate(task["protocol"], strict=True)
    method = task["method"]
    if method not in {"source_only", "bn_only", "tent"}:
        raise ValueError("task.method must be source_only, bn_only or tent")
    if (method == "source_only") != (protocol.regime == "source_only"):
        raise ValueError("source_only method and regime must be selected together")
    if config["trainer"]["device"] != "cpu" or config["trainer"]["devices"] != 1:
        raise ValueError("This configured stream uses one explicit CPU device")
    if config["environment"]["iterations"] != 1:
        raise ValueError("One stream configuration executes one seed; repeat explicitly")
    if config["model"].get("weights_path"):
        raise ValueError("Pass one task.checkpoint or --checkpoint, not a second model weight source")
    if set(task["metrics"]) != {"acc", "f1"}:
        raise ValueError("This closed-set stream reports exactly acc and f1")
    units = task["evaluation_units"]
    if not units or len(set(units)) != len(units) or not set(units) <= {"window", "record", "asset"}:
        raise ValueError("evaluation_units must select window, record and/or asset uniquely")
    selected = Path(checkpoint if checkpoint is not None else task["checkpoint"]).resolve(strict=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed = config["environment"]["seed"]
    seed_everything(seed)
    args = dict_to_namespace(config)
    # Declared source vocabulary only; target evaluator labels cannot set the head.
    model = build_model(args.model, metadata=None).cpu().eval()
    factory = build_data(args.data, args.task)
    logits_all, targets, rows = [], [], []
    try:
        metadata = factory.get_metadata()
        classes = int(args.model.num_classes)
        accuracy = MulticlassAccuracy(num_classes=classes, average="micro")
        f1 = MulticlassF1Score(num_classes=classes, average="macro")
        def evaluate(logits, evaluation):
            target = evaluation["y"].long()
            accuracy.update(logits, target)
            f1.update(logits, target)
            logits_all.append(logits.cpu())
            targets.append(target.cpu())
            for position, (file_id, window_id) in enumerate(zip(evaluation["file_id"], evaluation["window_id"])):
                file_id, window_id = int(file_id), int(window_id)
                meta = metadata[file_id]
                start, end = factory.test_dataset.dataset_dict[file_id].window_intervals[window_id]
                row = dict(batch=len(logits_all)-1, position=position, file_id=file_id,
                           window_id=window_id, raw_start=start, raw_end=end,
                           domain_id=meta["Domain_id"], target=int(target[position]),
                           prediction=int(logits[position].argmax()))
                if "asset" in units:
                    row["asset_id"] = meta["asset_id"]
                row.update({f"logit_{k}": float(logits[position, k]) for k in range(classes)})
                rows.append(row)
        loader = factory.get_dataloader("test")
        if method == "source_only":
            count = run_source_only_stream(model, loader, protocol,
                                          checkpoint_path=selected, evaluate=evaluate)
            state = {"model": model.state_dict(), "optimizer": None}
            budget = dict(samples=count, updates=0, target_labels_used=0, resets=0)
        else:
            adapter = run_tent_stream(model, loader, protocol, checkpoint_path=selected,
                                      learning_rate=task["lr"], evaluate=evaluate,
                                      update=method == "tent")
            state = adapter.state_dict()
            budget = dict(samples=adapter.stream_num_samples, updates=adapter.stream_num_updates,
                          target_labels_used=0, resets=adapter.num_resets)
        frame = pd.DataFrame(rows)
        values = dict(test_acc=float(accuracy.compute()), test_f1=float(f1.compute()))
        logits, truth = torch.cat(logits_all), torch.cat(targets)
        for unit, column in (("record", "file_id"), ("asset", "asset_id")):
            if unit in units:
                means = grouped_classification_metrics(logits, truth, frame[column].tolist(), classes)
                values.update({f"test_{unit}_macro_{name}": value for name, value in means.items()})
        state["python_rng"] = random.getstate()
        state["numpy_rng"] = np.random.get_state()
        state["torch_rng"] = torch.get_rng_state()
        state["stream_budget"] = budget
        torch.save(state, output / "final_state.pt")
        frame.to_csv(output / "predictions.csv", index=False)
        pd.DataFrame([values]).to_csv(output / "all_results.csv", index=False)
        summary = write_run_summary(output / "run_summary.json", results=[values], seeds=[seed], config=config)
        (output / "budget.json").write_text(json.dumps(budget, indent=2) + "\n")
        executed_config = deepcopy(config)
        executed_config["task"]["checkpoint"] = str(selected)
        (output / "config.yaml").write_text(yaml.safe_dump(executed_config, sort_keys=False))
        return dict(status="succeeded", result_dir=str(output), run_summary=str(output / "run_summary.json"),
                    test_metrics=str(output / "all_results.csv"), source_checkpoint=str(selected),
                    primary_metrics=summary["metrics"], **budget)
    finally:
        _close_data_factory(factory)
