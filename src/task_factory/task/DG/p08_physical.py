"""Bounded P08 source-only optimization and record-level estimators.

Data Factory alone reads signals and fits condition transforms. Model Factory alone
constructs the model. These research functions do not register another pipeline.
Synthetic fixture outputs are software checks, never industrial evidence.
"""
from __future__ import annotations

import copy
import csv
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from src.data_factory.p08_data import ConditionEncoder, load_records, split_records, windows
from src.model_factory import build_model


ARMS = {
    "B0": ("index", "none"), "B1": ("physical", "none"),
    "F01": ("index", "film"), "P0": ("physical", "film"),
    "TOKEN": ("physical", "token_concat"), "LATE": ("physical", "late_concat"),
}
Record = dict[str, Any]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def requested_device(config: dict) -> torch.device:
    trainer = config["trainer"]
    if trainer.get("devices", 1) != 1:
        raise ValueError("P08 requires one device; DDP is not part of this protocol")
    if config["data"]["evidence_kind"] == "industrial" and Path(sys.prefix).name != "LQ_signal":
        raise RuntimeError("Industrial P08 runs require the existing LQ_signal environment")
    requested = trainer["device"]
    if requested == "cpu":
        return torch.device("cpu")
    if requested not in {"cuda", "cuda:0"}:
        raise ValueError(f"Unsupported explicit P08 device: {requested}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0 explicitly; only physical GPU 0 is authorized")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Requested CUDA device is unavailable; there is no CPU fallback")
    # PyTorch requires this before its first cuBLAS operation in deterministic mode.
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic CUDA")
    return torch.device("cuda:0")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _split(config: dict, records: list[Record], target: str, seed: int):
    train, val, test = split_records(records, target, config["data"]["validation_fraction"], config["task"]["split_seed"])
    if any(r["system_id"] == target for r in train + val):
        raise ValueError("Target record leaked into source train or validation")
    if not test or any(r["system_id"] != target for r in test):
        raise ValueError("Test population must contain exactly the selected held-out system")
    groups = [{(r["system_id"], r["physical_unit_id"]) for r in part} for part in (train, val, test)]
    if any(a & b for a, b in itertools.combinations(groups, 2)):
        raise ValueError("Physical-unit overlap between train, validation, or test")
    expected = set(range(config["model"]["num_classes"]))
    for name, part in (("train", train), ("validation", val), ("test", test)):
        for system in sorted({r["system_id"] for r in part}):
            if {r["label"] for r in part if r["system_id"] == system} != expected:
                raise ValueError(f"{name}/{system} lacks the declared common ontology")
    return train, val, test


def _split_ids(train, val, test) -> dict:
    return {name: [r["record_id"] for r in part]
            for name, part in (("train", train), ("validation", val), ("test", test))}


def _model_config(config: dict, encoder: ConditionEncoder, arm: str) -> dict:
    args = copy.deepcopy(config["model"])
    if (args["type"], args["name"]) != ("ISFM", "M_P08_PhysicalConditioning"):
        raise ValueError("P08 runner requires the declared M_P08_PhysicalConditioning model")
    components = {"embedding": "E_01_HSE", "backbone": "shared_transformer_block",
                  "task_head": "shared_common_ontology_linear"}
    if any(args.get(key) != value for key, value in components.items()):
        raise ValueError("P08 model component declarations differ from the implemented shared HSE path")
    if args.get("weights_path"):
        raise ValueError("P08 source-only fits do not accept externally selected weights")
    if args.get("condition_dim") not in (None, encoder.output_dim):
        raise ValueError("model.condition_dim differs from the source-fitted encoder")
    args["condition_dim"] = encoder.output_dim
    args["coordinates"], args["fusion"] = ARMS[arm]
    return args


def _patches(x: torch.Tensor, model, generator: torch.Generator | None = None):
    maximum = x.shape[1] - model.embedding.patch_size_L
    if maximum < 0:
        raise ValueError("HSE patch length exceeds the declared signal window")
    count = model.embedding.num_patches
    if generator is None:
        starts = torch.linspace(0, maximum, count).round().long().expand(len(x), -1)
    else:
        starts = torch.randint(maximum + 1, (len(x), count), generator=generator)
    starts = starts.to(x.device)
    return {"start_indices_L": starts, "start_indices_C": torch.zeros_like(starts)}


def _forward(model, x, record, condition, *, generator=None, detached=False, features=False):
    kwargs = dict(fs=torch.full((len(x),), float(record["sampling_rate"]), device=x.device),
                  condition=condition.expand(len(x), -1).to(x.device),
                  detach_condition=detached, **_patches(x, model, generator))
    if features:
        kwargs["return_features"] = True
    return model(x, **kwargs)


def predict(model, records: list[Record], encoder: ConditionEncoder, config: dict,
            device: torch.device, *, intervention="correct", donors=None,
            first_window_only: bool = False) -> list[dict]:
    """Average window probabilities within each recording before any score."""
    model.eval()
    rows = []
    batch_size = config["trainer"]["batch_size"]
    with torch.no_grad():
        for record in records:
            condition_record = donors[record["record_id"]] if donors is not None else record
            condition = encoder.transform([condition_record], intervention=(
                "correct" if intervention == "detached" else intervention))
            probabilities, representations = [], []
            signal_windows = windows(record, config["data"])
            if first_window_only:
                signal_windows = signal_windows[:1]
            for start in range(0, len(signal_windows), batch_size):
                x = signal_windows[start:start + batch_size].to(device)
                logits, representation = _forward(model, x, record, condition,
                    detached=intervention == "detached", features=True)
                if not torch.isfinite(logits).all() or not torch.isfinite(representation).all():
                    raise FloatingPointError(f"Nonfinite prediction: {record['record_id']}")
                probabilities.append(logits.softmax(-1).cpu())
                representations.append(representation.cpu())
            probability = torch.cat(probabilities).mean(0).tolist()
            rows.append({key: record[key] for key in (
                "record_id", "system_id", "physical_unit_id", "label")})
            rows[-1].update(probabilities=probability,
                            representation=torch.cat(representations).mean(0).tolist())
    return rows


def scores(rows: list[dict], num_classes: int) -> dict:
    if not rows:
        raise ValueError("Cannot evaluate an empty record population")
    labels = list(range(num_classes))
    y = np.asarray([r["label"] for r in rows])
    probabilities = np.asarray([r["probabilities"] for r in rows])
    prediction = probabilities.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=labels, zero_division=0)
    squared = .5 * ((probabilities - np.eye(num_classes)[y]) ** 2).sum(1)
    by_system: dict[str, dict[str, list[float]]] = {}
    for record, value in zip(rows, squared):
        by_system.setdefault(record["system_id"], {}).setdefault(
            record["physical_unit_id"], []).append(float(value))
    brier = float(np.mean([np.mean([np.mean(values) for values in units.values()])
                           for units in by_system.values()]))
    return {"record_macro_f1": float(f1.mean()), "record_balanced_accuracy": float(recall.mean()),
            "source_equal_system_equal_unit_record_brier": brier,
            "record_count": len(rows), "physical_unit_count": sum(map(len, by_system.values())),
            "confusion_matrix": confusion_matrix(y, prediction, labels=labels).tolist(),
            "per_class": [{"label": label, "precision": float(precision[label]),
                           "recall": float(recall[label]), "f1": float(f1[label]),
                           "support": int(support[label])} for label in labels]}


def _cluster_ci(rows_by_arm: dict[str, list[dict]], coefficients: dict[str, float],
                num_classes: int, seed: int, repeats: int) -> list[float]:
    """Paired unit resampling conditional on already fitted models; no seed pooling."""
    first = next(iter(rows_by_arm.values()))
    ids = [r["record_id"] for r in first]
    for rows in rows_by_arm.values():
        if [r["record_id"] for r in rows] != ids:
            raise ValueError("Paired comparison requires the same ordered target records")
    groups = sorted({r["physical_unit_id"] for r in first})
    if len(groups) < 2:
        raise ValueError("Unit bootstrap requires at least two independent physical units")
    indices = {g: np.array([i for i, r in enumerate(first) if r["physical_unit_id"] == g]) for g in groups}
    y = np.asarray([r["label"] for r in first])
    predicted = {arm: np.asarray([r["probabilities"] for r in rows]).argmax(1)
                 for arm, rows in rows_by_arm.items()}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        selected = np.concatenate([indices[g] for g in rng.choice(groups, len(groups), replace=True)])
        values.append(sum(weight * f1_score(y[selected], predicted[arm][selected],
            labels=list(range(num_classes)), average="macro", zero_division=0)
            for arm, weight in coefficients.items()))
    return np.quantile(values, [.025, .975]).tolist()


def _save_predictions(path: Path, rows: list[dict], num_classes: int) -> None:
    write_json(path.with_suffix(".json"), rows)
    fields = ["record_id", "system_id", "physical_unit_id", "label"]
    with path.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + [f"p{i}" for i in range(num_classes)])
        writer.writeheader()
        for row in rows:
            writer.writerow({**{key: row[key] for key in fields},
                             **{f"p{i}": value for i, value in enumerate(row["probabilities"])}})


def _hierarchy(records):
    result: dict[str, dict[str, list[Record]]] = {}
    for r in records:
        result.setdefault(r["system_id"], {}).setdefault(r["physical_unit_id"], []).append(r)
    return result


def fit(config: dict, train: list[Record], val: list[Record], *, target: str,
        seed: int, arm: str, lr: float, weight_decay: float, output: Path,
        device: torch.device, epochs: int, updates: int, overfit: bool = False,
        encoder_train: list[Record] | None = None) -> dict:
    """Only source train and source validation are accepted; no target object is used."""
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if any(r["system_id"] == target for r in train + val + (encoder_train or [])):
        raise ValueError("fit() received a target record")
    _seed(seed)
    encoder = ConditionEncoder(config["data"]["continuous_fields"],
                               config["data"]["categorical_fields"]).fit(
                                   train if encoder_train is None else encoder_train, target)
    args = _model_config(config, encoder, arm)
    model = build_model(SimpleNamespace(**args), metadata=None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    hierarchy = _hierarchy(train)
    best, best_epoch = math.inf, None
    effective = copy.deepcopy(config)
    effective["model"] = args
    effective["trainer"].update(num_epochs=epochs, updates_per_epoch=updates)
    write_json(output / "config.json", effective)
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for update in range(updates):
            if overfit:
                record = train[(epoch * updates + update) % len(train)]
            else:
                units = hierarchy[rng.choice(sorted(hierarchy))]
                candidates = units[rng.choice(sorted(units))]
                record = candidates[rng.randrange(len(candidates))]
            signal_windows = windows(record, config["data"])
            indices = (torch.zeros(config["trainer"]["batch_size"], dtype=torch.long) if overfit else
                       torch.randint(len(signal_windows), (config["trainer"]["batch_size"],), generator=generator))
            x = signal_windows[indices].to(device)
            condition = encoder.transform([record])
            logits = _forward(model, x, record, condition, generator=generator)
            y = torch.full((len(x),), record["label"], dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite source-training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError("Missing or nonfinite source-training gradients")
            active = [model.embedding, model.norm, model.backbone, model.head]
            if model.fusion != "none":
                active.append(getattr(model, model.fusion))
            if any(p.grad is None for module in active for p in module.parameters() if p.requires_grad):
                raise RuntimeError("An active model branch is disconnected from the objective")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite optimizer parameters")
            losses.append(float(loss.detach()))
        rows = predict(model, val, encoder, config, device, first_window_only=overfit)
        metric = scores(rows, args["num_classes"])
        brier = metric["source_equal_system_equal_unit_record_brier"]
        history.append({"epoch": epoch + 1, "updates": updates, "train_ce": float(np.mean(losses)),
                        "source_validation_brier": brier})
        with (output / "logs.jsonl").open("a") as stream:
            stream.write(json.dumps(history[-1], allow_nan=False) + "\n")
        if brier < best:
            best, best_epoch = brier, epoch + 1
            torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "model_config": args, "encoder": encoder.state_dict(), "target": target,
                        "seed": seed, "arm": arm, "epoch": best_epoch, "config": effective,
                        "train_record_ids": [r["record_id"] for r in train],
                        "validation_record_ids": [r["record_id"] for r in val],
                        "hyperparameters": {"learning_rate": lr, "weight_decay": weight_decay},
                        "source_validation_brier": best}, output / "checkpoint.pt")
            _save_predictions(output / "source_validation_predictions", rows, args["num_classes"])
    result = {"status": "fit_completed", "target": target, "seed": seed, "arm": arm,
              "hyperparameters": {"learning_rate": lr, "weight_decay": weight_decay},
              "source_validation_brier": best, "selected_epoch": best_epoch,
              "epochs_executed": epochs, "updates_executed": epochs * updates,
              "checkpoint": "checkpoint.pt", "parameter_counts": model.parameter_counts(),
              "fit_and_source_validation_seconds": time.perf_counter() - started}
    write_json(output / "fit_result.json", result)
    return result


def _load_checkpoint(path: Path, target: str, seed: int, arm: str, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (checkpoint["target"], checkpoint["seed"], checkpoint["arm"]) != (target, seed, arm):
        raise ValueError("Checkpoint target, seed, or arm differs from the requested evaluation")
    model = build_model(SimpleNamespace(**checkpoint["model_config"]), metadata=None).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    encoder = ConditionEncoder.from_state_dict(checkpoint["encoder"])
    return checkpoint, model, encoder


def _completed_evaluation(root: Path, target: str, seed: int, arm: str) -> dict:
    run_path = root / f"target-{target}" / f"seed-{seed}" / f"arm-{arm}" / "result.json"
    if not run_path.is_file() or not (root / "summary.json").is_file():
        raise ValueError("Ablation requires a completed comparison/benchmark run, not a residual checkpoint")
    result = json.loads(run_path.read_text())
    summary = json.loads((root / "summary.json").read_text())
    identity = (target, seed, arm)
    if (result.get("target"), result.get("seed"), result.get("arm")) != identity or result.get("status") != "completed":
        raise ValueError("Ablation comparison completion record has a different identity or failed status")
    included = any((row.get("target"), row.get("seed"), row.get("arm")) == identity
                   and row.get("status") == "completed" for row in summary.get("results", []))
    if summary.get("status") != "completed" or summary.get("command") not in {"compare", "benchmark"} or not included:
        raise ValueError("Ablation checkpoint is not included in a completed comparison/benchmark summary")
    return result


def _selected(selection_root: Path, config: dict, target: str, seed: int, arm: str) -> dict:
    donor_arm = "B1" if config["task"]["comparison"] == "matched" else arm
    if config["task"]["comparison"] == "matched" and arm not in {"B0", "B1", "F01", "P0"}:
        raise ValueError("Matched mode admits exactly the coordinate x conditioning arms")
    tuning_seed = config["task"]["tuning_seed"]
    path = selection_root / f"target-{target}" / f"seed-{tuning_seed}" / f"arm-{donor_arm}" / "selection.json"
    selected = json.loads(path.read_text())
    if (selected["target"], selected["seed"], selected["arm"], selected["status"]) != (
            target, tuning_seed, donor_arm, "completed"):
        raise ValueError(f"Selection identity/status mismatch: {path}")
    for key in ("data", "model", "trainer"):
        if selected["base_config"][key] != config[key]:
            raise ValueError(f"Selection {key} differs from the requested experiment: {path}")
    expected_grid = list(itertools.product(config["task"]["search_learning_rates"],
                                           config["task"]["search_weight_decays"]))
    trials = selected["trials"]
    if len(trials) != len(expected_grid) or any(t["status"] != "completed" for t in trials):
        raise ValueError("The selected baseline search has missing or failed trials")
    if [(t["hyperparameters"]["learning_rate"], t["hyperparameters"]["weight_decay"]) for t in trials] != expected_grid:
        raise ValueError("Selected search space differs from the declared common finite search")
    winner = min(trials, key=lambda trial: trial["source_validation_brier"])
    if winner["hyperparameters"] != selected["hyperparameters"] or winner["directory"] != selected["selected_trial"]:
        raise ValueError("Chosen hyperparameters do not minimize source-validation Brier")
    if selected["base_config"]["task"]["split_seed"] != config["task"]["split_seed"]:
        raise ValueError("Source split seed differs from the tuning protocol")
    if json.loads((selection_root / "records.json").read_text()) != load_records(config["data"]):
        raise ValueError("Inventory metadata changed after source-only tuning")
    return selected


def _provenance(config: dict) -> dict:
    root = Path(__file__).resolve().parents[4]
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short"], cwd=root, capture_output=True, text=True, check=True).stdout
    return {"code_sha": sha, "working_tree_status": dirty, "command": sys.argv,
            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "evidence_kind": config["data"]["evidence_kind"], "industrial_claim_accepted": False}


def _donors(config, all_records, train, test):
    path = config["data"].get("wrong_condition_donors")
    if not path:
        return None
    sources = {r["record_id"]: r for r in train}
    with Path(path).open(newline="") as stream:
        manifest = list(csv.DictReader(stream))
    by_target = {}
    target_ids = {r["record_id"] for r in test}
    for row in manifest:
        if row["record_id"] not in target_ids:
            continue
        if row["record_id"] in by_target or row["compatible"] != "true" or not row["compatibility_basis"].strip():
            raise ValueError("Wrong-condition donors require unique, explicitly compatible assignments")
        if row["donor_record_id"] not in sources:
            raise ValueError("Wrong-condition donor must belong to this fold's source training population")
        by_target[row["record_id"]] = sources[row["donor_record_id"]]
    if set(by_target) != target_ids:
        raise ValueError("Wrong-condition donor manifest does not cover every target record")
    return by_target


def _contrasts(results: list[dict], config: dict, output: Path) -> None:
    comparisons = []
    for target, seed in sorted({(r["target"], r["seed"]) for r in results}):
        records = {r["arm"]: r for r in results if (r["target"], r["seed"]) == (target, seed)}
        if config["task"]["comparison"] == "matched":
            definitions = {"delta_H": {"B1": 1, "B0": -1},
                           "delta_P_at_index": {"F01": 1, "B0": -1},
                           "delta_P_at_physical": {"P0": 1, "B1": -1},
                           "interaction": {"P0": 1, "B1": -1, "F01": -1, "B0": 1}}
        else:
            definitions = {f"P0_minus_{arm}": {"P0": 1, arm: -1} for arm in ("B1", "LATE", "TOKEN")}
        for name, coefficients in definitions.items():
            if not set(coefficients) <= records.keys():
                continue
            rows = {a: json.loads((output / records[a]["directory"] / "predictions.json").read_text()) for a in coefficients}
            value = sum(w * records[a]["metrics"]["record_macro_f1"] for a, w in coefficients.items())
            comparisons.append({"target": target, "seed": seed, "contrast": name, "delta_macro_f1": value,
                "ci95_conditional_on_fitted_models": _cluster_ci(rows, coefficients, config["model"]["num_classes"],
                    seed, config["task"].get("bootstrap_replicates", 1000))})
    write_json(output / "contrasts.json", comparisons)


def execute(config: dict, command: str, output: Path, *, target: str | None = None,
            seed: int | None = None, arms: list[str] | None = None,
            selection: Path | None = None) -> dict:
    """CLI owner calls this once with the public resolver's effective configuration."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        return _execute(config, command, output, target=target, seed=seed, arms=arms, selection=selection)
    except Exception as exc:
        failure_path = output / "failed.json"
        if failure_path.exists():
            failure_path = output / f"failed-{time.time_ns()}.json"
        write_json(failure_path, {"status": "failed", "command": command,
            "target": target, "seed": seed, "arms": arms,
            "exception": type(exc).__name__, "reason": str(exc), "traceback": traceback.format_exc()})
        raise


def _execute(config: dict, command: str, output: Path, *, target, seed, arms, selection) -> dict:
    if command not in {"data-check", "smoke", "overfit", "tune", "compare", "benchmark", "ablate"}:
        raise ValueError(f"Unknown P08 action: {command}")
    prior_results = []
    if (output / "config.json").exists():
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Output root already contains a different experiment configuration")
        if (output / "summary.json").exists():
            previous = json.loads((output / "summary.json").read_text())
            if previous["command"] != command:
                raise ValueError("Use a separate output root for each experiment phase")
            prior_results = previous["results"]
    else:
        write_json(output / "config.json", config)
    with (output / "provenance.jsonl").open("a") as stream:
        stream.write(json.dumps(_provenance(config), allow_nan=False) + "\n")
    device = requested_device(config)
    records = load_records(config["data"])
    if (output / "records.json").exists():
        if json.loads((output / "records.json").read_text()) != records:
            raise ValueError("Output root inventory metadata has changed")
    else:
        write_json(output / "records.json", records)
    systems = config["task"]["systems"]
    seeds = config["task"]["seeds"]
    if len(set(systems)) != len(systems) or set(systems) != {r["system_id"] for r in records}:
        raise ValueError("task.systems must enumerate the qualified inventory exactly once")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Configured random seeds must be nonempty and unique")
    for value in seeds:
        _seed(value)
    _seed(config["task"]["split_seed"])
    _seed(config["task"]["tuning_seed"])
    if target is not None and target not in systems:
        raise ValueError("Requested target is outside the fixed system set")
    if seed is not None and seed not in seeds:
        raise ValueError("Requested seed is outside the fixed seed set")
    if config["task"]["comparison"] not in {"matched", "tuned"}:
        raise ValueError("task.comparison must be matched or tuned")
    if config["task"].get("loss") != "CE":
        raise ValueError("P08 implements source cross-entropy only")
    if config["task"].get("selection_metric", "source_system_unit_balanced_record_brier") != "source_system_unit_balanced_record_brier":
        raise ValueError("P08 checkpoint selection requires source system/unit-balanced record Brier")
    if config["task"].get("target_metric", "record_macro_f1") != "record_macro_f1":
        raise ValueError("P08 target primary metric is record_macro_f1")
    arms = list(config["task"]["arms"] if arms is None else arms)
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= ARMS.keys():
        raise ValueError("Arms must be a nonempty unique subset of B0,B1,F01,P0,TOKEN,LATE")
    epochs = _positive_int(config["trainer"]["num_epochs"], "trainer.num_epochs")
    updates = _positive_int(config["trainer"]["updates_per_epoch"], "trainer.updates_per_epoch")
    _positive_int(config["trainer"]["batch_size"], "trainer.batch_size")
    _positive_int(config["task"].get("bootstrap_replicates", 1000), "task.bootstrap_replicates")
    grid = list(itertools.product(config["task"]["search_learning_rates"], config["task"]["search_weight_decays"]))
    if not grid or len(set(grid)) != len(grid) or any(not math.isfinite(lr) or lr <= 0 or
            not math.isfinite(wd) or wd < 0 for lr, wd in grid):
        raise ValueError("Hyperparameter search must be finite, nonempty and unique")
    if config["data"]["evidence_kind"] == "industrial" and grid != list(itertools.product([.0003, .001, .003], [0., .0001])):
        raise ValueError("Industrial P08 runs use the frozen common six-trial search")
    if command in {"compare", "benchmark", "ablate"} and selection is None:
        raise ValueError(f"{command} requires --selection; no untuned fallback")
    if command == "benchmark" and (target is not None or seed is not None):
        raise ValueError("benchmark evaluates the full declared matrix; use compare for a subset")
    if command == "benchmark":
        required = {"B0", "B1", "F01", "P0"} if config["task"]["comparison"] == "matched" else set(config["task"]["arms"])
        if set(arms) != required or not {"B1", "P0"} <= required:
            raise ValueError("benchmark requires the complete declared comparison arms; use compare for subsets")
    targets = [target] if target is not None else systems
    selected_seeds = [seed] if seed is not None else seeds
    if command == "tune":
        if seed is not None and seed != config["task"]["tuning_seed"]:
            raise ValueError("tune requires the frozen task.tuning_seed")
        selected_seeds = [config["task"]["tuning_seed"]]
    if command == "data-check":
        selected_seeds = [config["task"]["split_seed"]]
    if command == "ablate" and (target is None) != (seed is None):
        raise ValueError("ablate requires both --target and --seed, or neither for the full matrix")
    if command in {"smoke", "overfit", "compare"} and (target is None or seed is None):
        raise ValueError(f"{command} requires explicit --target and --seed")
    choices = {}
    if command in {"compare", "benchmark"}:
        # Fail the whole requested comparison before any target signal is opened.
        for held_out, repetition, arm in itertools.product(targets, selected_seeds, arms):
            chosen = _selected(Path(selection), config, held_out, repetition, arm)
            if chosen["split"] != _split_ids(*_split(config, records, held_out, repetition)):
                raise ValueError("Source tuning and evaluation splits differ")
            choices[(held_out, repetition, arm)] = chosen
    results = list(prior_results)
    for held_out, repetition in itertools.product(targets, selected_seeds):
        train, val, test = _split(config, records, held_out, repetition)
        fold = output / f"target-{held_out}" / f"seed-{repetition}"
        fold.mkdir(parents=True, exist_ok=True)
        split_ids = _split_ids(train, val, test)
        if (fold / "split.json").exists():
            if json.loads((fold / "split.json").read_text()) != split_ids:
                raise ValueError("Existing output fold has a different record split")
        else:
            write_json(fold / "split.json", split_ids)
        if command == "data-check":
            if (fold / "data_check.json").exists():
                raise FileExistsError(f"Refusing to overwrite {fold / 'data_check.json'}")
            # This explicit read checks shape/finiteness only; nothing is fitted on target.
            counts = {name: [{"record_id": r["record_id"], "windows": len(windows(r, config["data"]))} for r in part]
                      for name, part in (("train", train), ("validation", val), ("test_sanity_only", test))}
            encoder = ConditionEncoder(config["data"]["continuous_fields"], config["data"]["categorical_fields"]).fit(train, held_out)
            write_json(fold / "data_check.json", {"records": counts, "source_encoder": encoder.state_dict(),
                                                  "target_reads_are_sanity_only": True})
            results.append({"target": held_out, "seed": repetition, "status": "completed"})
            continue
        for arm in arms:
            arm_path = fold / f"arm-{arm}"
            if command == "tune":
                arm_path.mkdir()
                trials = []
                for index, (lr, wd) in enumerate(grid):
                    trial = fit(config, train, val, target=held_out, seed=repetition, arm=arm,
                                lr=lr, weight_decay=wd, output=arm_path / f"trial-{index:02d}",
                                device=device, epochs=epochs, updates=updates)
                    trial["directory"] = f"trial-{index:02d}"
                    trial["status"] = "completed"
                    write_json(arm_path / trial["directory"] / "result.json", trial)
                    trials.append(trial)
                winner = min(trials, key=lambda t: t["source_validation_brier"])
                selected = {"status": "completed", "target": held_out, "seed": repetition, "arm": arm,
                    "base_config": config, "split": split_ids, "trials": trials,
                    "hyperparameters": winner["hyperparameters"], "selected_trial": winner["directory"],
                    "selection_rule": "minimum equal-system/equal-unit record Brier; earliest epoch/trial breaks ties",
                    "source_validation_brier": winner["source_validation_brier"]}
                write_json(arm_path / "selection.json", selected)
                results.append({key: selected[key] for key in ("status", "target", "seed", "arm", "hyperparameters", "source_validation_brier")})
                continue
            if command == "ablate":
                if arm != "P0":
                    raise ValueError("ablate requires --arms P0 and an already fitted P0 checkpoint")
                arm_path.mkdir()
                completed = _completed_evaluation(Path(selection), held_out, repetition, "P0")
                checkpoint_path = Path(selection) / f"target-{held_out}" / f"seed-{repetition}" / "arm-P0" / "checkpoint.pt"
                if json.loads((Path(selection) / "records.json").read_text()) != records:
                    raise ValueError("Inventory metadata changed after checkpoint fitting")
                checkpoint, model, encoder = _load_checkpoint(checkpoint_path, held_out, repetition, "P0", device)
                if checkpoint["epoch"] != completed["selected_epoch"] or checkpoint["source_validation_brier"] != completed["source_validation_brier"]:
                    raise ValueError("Ablation checkpoint no longer matches its completed evaluation")
                if checkpoint["train_record_ids"] != split_ids["train"] or checkpoint["validation_record_ids"] != split_ids["validation"]:
                    raise ValueError("Ablation data split differs from checkpoint source fit")
                for key in ("data", "trainer"):
                    if checkpoint["config"][key] != config[key]:
                        raise ValueError(f"Ablation {key} differs from the fitted checkpoint")
                donors = _donors(config, records, train, test)
                correct_rows = None
                for intervention in ("correct", "missing", "default", "detached", "wrong"):
                    if intervention == "wrong" and donors is None:
                        write_json(arm_path / "wrong.json", {"status": "not_identifiable", "reason": "No predefined physically compatible source donor manifest"})
                        continue
                    if intervention == "wrong":
                        correct_conditions = encoder.transform(test)
                        changed_conditions = encoder.transform([donors[r["record_id"]] for r in test])
                        identifiable = (correct_conditions != changed_conditions).any(dim=1)
                        write_json(arm_path / "wrong_identifiability.json", {
                            "record_count": len(test), "changed_count": int(identifiable.sum()),
                            "unchanged_record_ids": [r["record_id"] for r, changed in zip(test, identifiable) if not changed]})
                        if not identifiable.all():
                            write_json(arm_path / "wrong.json", {"status": "not_identifiable", "reason": "Donor encoded condition is unchanged for some target records; no partial-population metric emitted"})
                            continue
                    rows = predict(model, test, encoder, config, device,
                                   intervention="correct" if intervention == "wrong" else intervention,
                                   donors=donors if intervention == "wrong" else None)
                    _save_predictions(arm_path / f"predictions_{intervention}", rows, config["model"]["num_classes"])
                    if correct_rows is None:
                        correct_rows = rows
                    metric = scores(rows, config["model"]["num_classes"])
                    metric["mean_probability_l1_from_correct"] = float(np.mean([np.abs(np.array(a["probabilities"])-b["probabilities"]).sum() for a, b in zip(rows, correct_rows)]))
                    metric["mean_representation_l2_from_correct"] = float(np.mean([np.linalg.norm(np.array(a["representation"])-b["representation"]) for a, b in zip(rows, correct_rows)]))
                    write_json(arm_path / f"metrics_{intervention}.json", metric)
                    results.append({"target": held_out, "seed": repetition, "arm": arm, "intervention": intervention, "metrics": metric})
                continue
            chosen = {"learning_rate": .001, "weight_decay": 0.}
            if command in {"compare", "benchmark"}:
                selected = choices[(held_out, repetition, arm)]
                chosen = selected["hyperparameters"]
            fit_train, fit_val = train, val
            run_epochs, run_updates = epochs, updates
            if command == "smoke":
                run_epochs, run_updates = 1, 2
            if command == "overfit":
                subset = {}
                for r in train:
                    subset.setdefault((r["system_id"], r["label"]), r)
                fit_train = list(subset.values())
                fit_val = fit_train
                run_epochs, run_updates = 1, min(100, _positive_int(config["trainer"].get("overfit_steps", 100), "trainer.overfit_steps"))
            result = fit(config, fit_train, fit_val, target=held_out, seed=repetition, arm=arm,
                         lr=chosen["learning_rate"], weight_decay=chosen["weight_decay"], output=arm_path,
                         device=device, epochs=run_epochs, updates=run_updates, overfit=command == "overfit",
                         encoder_train=train if command == "overfit" else None)
            checkpoint, model, encoder = _load_checkpoint(arm_path / "checkpoint.pt", held_out, repetition, arm, device)
            if command in {"compare", "benchmark"}:
                rows = predict(model, test, encoder, config, device)
                _save_predictions(arm_path / "predictions", rows, config["model"]["num_classes"])
                metric = scores(rows, config["model"]["num_classes"])
                metric["parameter_counts"] = model.parameter_counts()
                metric["macro_f1_ci95_fixed_model_unit_bootstrap"] = _cluster_ci({arm: rows}, {arm: 1}, config["model"]["num_classes"], repetition, config["task"].get("bootstrap_replicates", 1000))
                write_json(arm_path / "metrics.json", metric)
                result.update(metrics=metric, directory=str(arm_path.relative_to(output)))
            elif command == "overfit":
                rows = predict(model, fit_train, encoder, config, device, first_window_only=True)
                accuracy = float(np.mean([np.argmax(r["probabilities"]) == r["label"] for r in rows]))
                result["training_record_accuracy"] = accuracy
                if accuracy < .95:
                    result["status"] = "failed"
                    write_json(arm_path / "overfit.json", result)
                    raise RuntimeError(f"Small-data overfit failed: training accuracy {accuracy:.3f} < 0.95; stop formal runs")
                result["status"] = "completed"
                write_json(arm_path / "overfit.json", result)
            result["status"] = "completed"
            write_json(arm_path / "result.json", result)
            results.append(result)
    summary = {"status": "completed", "command": command, "evidence_kind": config["data"]["evidence_kind"],
               "industrial_claim_accepted": False, "results": results}
    if command in {"compare", "benchmark"}:
        _contrasts(results, config, output)
        with (output / "summary.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["target", "seed", "arm", "record_macro_f1", "record_balanced_accuracy"])
            for r in results:
                writer.writerow([r["target"], r["seed"], r["arm"], r["metrics"]["record_macro_f1"], r["metrics"]["record_balanced_accuracy"]])
    write_json(output / "summary.json", summary)
    return summary
