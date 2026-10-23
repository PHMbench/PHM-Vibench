"""Existing bounded operator-path objectives and estimators.

Migrated from the P07 P4 reference experiment without changing its optimization,
source-validation selection, intervention, or coefficient-one path semantics.
The network and extraction algorithms remain owned by Model Factory.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_factory.operator_path_export import validate_data
from src.model_factory.X_model.P07OperatorPath import ConvControl, FAMILY, OPERATORS, OperatorNet

ARMS = ("sparse", "routing_concentration", "uniform", "dense", "learned", "cnn", "resnet1d", "proposed", "unbounded")
TRAINING_KEYS = {"lr", "weight_decay", "batch_size", "epochs", "residual_weight", "concentration_weight", "head_lr_multiplier"}


def source_partitions(data: dict) -> dict:
    """Physical source identities bind selected checkpoints across explicit exports."""
    return {split: [dict(unit_id=str(unit), label=int(data["y"][mask][0]),
                        domains=sorted(str(v) for v in set(data["domain"][mask])))
                   for unit in np.unique(data["unit_id"][data["split"] == split])
                   for mask in [(data["split"] == split) & (data["unit_id"] == unit)]]
            for split in ("train", "val")}


def objective(model: nn.Module, logits: torch.Tensor, labels: torch.Tensor,
              trace: object, arm: str, residual_weight: float,
              concentration_weight: float) -> torch.Tensor:
    """Training objective; checkpoint selection always uses unregularized CE."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    loss = nn.functional.cross_entropy(logits, labels)
    if arm == "proposed":
        loss = loss + residual_weight * model.residual_loss(trace)
    elif arm == "routing_concentration":
        loss = loss + concentration_weight * model.routing_concentration_loss(trace)
    return loss


def fail(output: Path, error: Exception) -> None:
    write_json(output / "failure.json", dict(status="failed", error_type=type(error).__name__, error=str(error)))




def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build(arm: str, channels: int, classes: int) -> nn.Module:
    if arm == "resnet1d":
        from src.model_factory.CNN.ResNet1D import Model
        class ResNetInterface(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.network = Model(SimpleNamespace(input_dim=channels, num_classes=classes,
                    block_type="basic", layers=[2, 2, 2, 2], initial_channels=64))

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
                return self.network(x), None

            @property
            def head(self) -> nn.Module:
                return self.network.classifier
        return ResNetInterface()
    if arm == "cnn":
        return ConvControl(channels, classes)
    choices = {"proposed": ("sparse", False, True), "sparse": ("sparse", False, True),
               "routing_concentration": ("sparse", False, True),
               "dense": ("dense", False, True), "uniform": ("uniform", False, True),
               "learned": ("sparse", True, True), "unbounded": ("sparse", False, False)}
    if arm not in choices:
        raise ValueError(f"Unknown arm: {arm}")
    mode, learned, bounded = choices[arm]
    return OperatorNet(channels, classes, mode=mode, learned=learned, bounded=bounded)


def make_optimizer(model: nn.Module, lr: float, weight_decay: float,
                   head_lr_multiplier: float) -> torch.optim.AdamW:
    """Same explicit head/body learning-rate rule for every supported classifier."""
    if lr <= 0 or head_lr_multiplier <= 0:
        raise ValueError("Learning rate and head multiplier must be positive")
    head = list(model.head.parameters())
    identities = {id(parameter) for parameter in head}
    body = [parameter for parameter in model.parameters() if id(parameter) not in identities]
    return torch.optim.AdamW([
        {"params": body, "lr": lr, "name": "body"},
        {"params": head, "lr": lr * head_lr_multiplier, "name": "head"},
    ], weight_decay=weight_decay)


def normalize(data: dict) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Fit channel statistics on training windows only; never on target domains."""
    training = data["split"] == "train"
    mean = data["x"][training].mean((0, 1), keepdims=True)
    scale = data["x"][training].std((0, 1), keepdims=True)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Training channel scale is zero/nonfinite; no default normalization is inserted")
    return torch.from_numpy(((data["x"] - mean) / scale).astype(np.float32)), mean, scale


def train(data: dict, arm: str, seed: int, epochs: int, device: torch.device,
          residual_weight: float, output: Path, *, lr: float = 0.002,
          weight_decay: float = 0.0001, batch_size: int = 32,
          concentration_weight: float = 0.1, head_lr_multiplier: float = 1.0) -> tuple[nn.Module, torch.Tensor, dict]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    masks = {s: data["split"] == s for s in ("train", "val")}
    x, mean, scale = normalize(data)
    y = torch.from_numpy(data["y"].astype(np.int64))
    classes = len(np.unique(data["y"][masks["train"]]))
    model = build(arm, x.shape[2], classes).to(device)
    optimizer = make_optimizer(model, lr, weight_decay, head_lr_multiplier)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(x[masks["train"]], y[masks["train"]]), batch_size=batch_size,
                        shuffle=True, generator=generator)
    best_loss, best_state, best_epoch = float("inf"), None, None
    history = []
    for epoch in range(epochs):
        model.train()
        training_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, trace = model(xb)
            loss = objective(model, logits, yb, trace, arm, residual_weight, concentration_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss in arm={arm},seed={seed},epoch={epoch}; run not rescued")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError(f"Nonfinite gradient in arm={arm},seed={seed},epoch={epoch}")
            optimizer.step()
            training_loss += float(loss.detach()) * len(xb)
        model.eval()
        with torch.no_grad():
            validation_sum = 0.0
            for xb, yb in DataLoader(TensorDataset(x[masks["val"]], y[masks["val"]]), batch_size=batch_size):
                validation_logits, _ = model(xb.to(device))
                validation_sum += float(nn.functional.cross_entropy(validation_logits, yb.to(device), reduction="sum"))
            validation_loss = validation_sum / int(masks["val"].sum())
        if not np.isfinite(validation_loss):
            raise RuntimeError("Nonfinite validation objective")
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(dict(epoch=epoch, train_loss=training_loss / int(masks["train"].sum()), val_loss=validation_loss))
        write_csv(output / "training.csv", history)
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    torch.save(dict(state_dict=best_state, arm=arm, seed=seed, mean=mean.tolist(), scale=scale.tolist(),
                    normalization_dtype=str(data["x"].dtype), source_partitions=source_partitions(data),
                    channels=x.shape[2], classes=classes, selected_epoch=best_epoch), output / "model.pt")
    write_csv(output / "training.csv", history)
    validation_indices = np.flatnonzero(masks["val"])
    with torch.no_grad():
        validation_predictions = torch.cat([model(batch.to(device))[0].argmax(-1).cpu()
            for batch in x[masks["val"]].split(batch_size)]).numpy()
    write_csv(output / "validation_predictions.csv", [dict(sample=int(index),
        unit_id=str(data["unit_id"][index]), domain=str(data["domain"][index]),
        target=int(data["y"][index]), prediction=int(validation_predictions[k]))
        for k, index in enumerate(validation_indices)])
    return model, x, dict(best_val_loss=best_loss, selected_epoch=best_epoch,
                                    optimizer_groups=[dict(name=group["name"], lr=group["lr"],
                                        parameters=sum(parameter.numel() for parameter in group["params"])) for group in optimizer.param_groups],
                                    parameter_count=sum(p.numel() for p in model.parameters()))


def fixed_score(logits: torch.Tensor, winner: int, rival: int) -> float:
    return float(logits[0, winner] - logits[0, rival])


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, data: dict, arm: str, seed: int,
             output: Path, intervention_limit: int, search_budget: int,
             strategies: tuple[str, ...] = ("cost",), batch_size: int = 32,
             partition: str = "test") -> dict:
    idx = np.flatnonzero(data["split"] == partition)
    device = next(model.parameters()).device
    logits = torch.cat([model(batch.to(device))[0].cpu() for batch in x[idx].split(batch_size)])
    prediction = logits.argmax(-1).cpu().numpy()
    rows = [dict(seed=seed, arm=arm, unit_id=str(data["unit_id"][i]), domain=str(data["domain"][i]),
                 sample=int(i), partition=partition, target=int(data["y"][i]), prediction=int(prediction[k]))
            for k, i in enumerate(idx)]
    write_csv(output / "predictions.csv", rows)
    # Report utility over independent record means as well as conventional window metrics.
    group_accuracy = [np.mean([r["prediction"] == r["target"] for r in rows if r["unit_id"] == unit])
                      for unit in np.unique(data["unit_id"][idx])]
    metrics = dict(seed=seed, arm=arm, status="completed", evaluation_partition=partition,
                   evaluated_windows=len(idx), evaluated_units=len(group_accuracy), record_mean_accuracy=float(np.mean(group_accuracy)),
                   window_balanced_accuracy=float(balanced_accuracy_score(data["y"][idx], prediction)),
                   window_macro_f1=float(f1_score(data["y"][idx], prediction, average="macro")))
    if not isinstance(model, OperatorNet):
        metrics["interventions"] = "not_applicable_black_box"
        return metrics
    # Deterministic class-balanced selection, independent of correctness or confidence.
    chosen = []
    labels = sorted(set(data["y"][idx]))
    by_class = {label: idx[data["y"][idx] == label].tolist() for label in labels}
    for offset in range(max(map(len, by_class.values()))):
        for label in labels:
            if offset < len(by_class[label]):
                chosen.append(by_class[label][offset])
    chosen = chosen[:intervention_limit]
    effect_rows, route_rows = [], []
    rng = np.random.default_rng(seed)
    for i in chosen:
        sample = x[i:i + 1].to(device)
        reference, trace = model(sample)
        winner, rival = reference[0].topk(2).indices.tolist()
        base_score = fixed_score(reference, winner, rival)
        frozen = [a.detach() for a in trace.weights]
        for stage in range(model.stages):
            weights = frozen[stage][0].cpu().numpy()
            top, low = int(weights.argmax()), int(weights.argmin())
            active_other = [j for j in range(6) if j != top and weights[j] > 1e-8]
            random_edge = int(rng.choice(active_other)) if active_other else -1
            for edge, name in enumerate(OPERATORS):
                modified, _ = model(sample, frozen=frozen, intervention=(stage, edge, "ZERO"))
                live, _ = model(sample, intervention=(stage, edge, "ZERO"))
                removed, _ = model(sample, intervention=(stage, edge, "ZERO"), policy_removal=True)
                necessity = base_score - fixed_score(modified, winner, rival)
                live_necessity = base_score - fixed_score(live, winner, rival)
                policy = base_score - fixed_score(removed, winner, rival)
                same = next((k for k in range(6) if k != edge and FAMILY[OPERATORS[k]] == FAMILY[name]), None)
                same_effect, unrelated_effect = None, None
                same_name, unrelated_name = "", ""
                # Learned branches do not inherit named-operator family semantics.
                if same is not None and not model.learned:
                    target_norm = float(trace.branches[stage][:, same].norm())
                    unrelated = min((k for k in range(6) if FAMILY[OPERATORS[k]] != FAMILY[name]),
                                    key=lambda k: abs(float(trace.branches[stage][:, k].norm()) - target_norm))
                    same_name, unrelated_name = OPERATORS[same], OPERATORS[unrelated]
                    same_logits, _ = model(sample, frozen=frozen, intervention=(stage, edge, same_name))
                    unrelated_logits, _ = model(sample, frozen=frozen, intervention=(stage, edge, unrelated_name))
                    same_effect = base_score - fixed_score(same_logits, winner, rival)
                    unrelated_effect = base_score - fixed_score(unrelated_logits, winner, rival)
                effect_rows.append(dict(seed=seed, arm=arm, unit_id=str(data["unit_id"][i]), sample=int(i),
                    stage=stage, edge=edge, operator=name if not model.learned else f"learned_{edge}",
                    allocation=float(weights[edge]), branch_norm=float(trace.branches[stage][:, edge].norm()),
                    necessity=necessity, policy_removal=policy, live_necessity=live_necessity, routing_compensation=live_necessity-necessity,
                    top=edge == top, low=edge == low, random=edge == random_edge,
                    random_eligible=bool(active_other), same_replacement=same_name, unrelated_replacement=unrelated_name,
                    same_effect=same_effect, unrelated_effect=unrelated_effect,
                    substitution_eligible=same is not None and not model.learned and weights[edge] > 1e-8))
        for strategy in strategies:
            extracted = model.extract(sample, budget=search_budget, strategy=strategy)
            route_rows.append(dict(seed=seed, arm=arm, unit_id=str(data["unit_id"][i]), sample=int(i),
                                   partition=partition, target=int(data["y"][i]), domain=str(data["domain"][i]),
                                   **{k: json.dumps(v) if k == "path" else v for k, v in extracted.items()}))
    write_csv(output / "interventions.csv", effect_rows)
    write_csv(output / "extractions.csv", route_rows)
    metrics["intervention_windows"] = len(chosen)
    for strategy in strategies:
        selected = [row for row in route_rows if row["strategy"] == strategy]
        units = sorted({row["unit_id"] for row in selected})
        unit_coverage = [np.mean([row["accepted"] for row in selected if row["unit_id"] == unit]) for unit in units]
        unit_argmax = [np.mean([row["checked_argmax_accepted"] for row in selected if row["unit_id"] == unit]) for unit in units]
        prefix = "" if strategy == "cost" else f"{strategy}_"
        metrics.update({prefix + "explanation_coverage": float(np.mean([r["accepted"] for r in selected])),
                        prefix + "unit_mean_explanation_coverage": float(np.mean(unit_coverage)),
                        prefix + "unit_mean_checked_argmax_coverage": float(np.mean(unit_argmax)),
                        prefix + "checked_argmax_coverage": float(np.mean([r["checked_argmax_accepted"] for r in selected])),
                        prefix + "argmax_agreement": float(np.mean([r["argmax_agrees"] for r in selected])),
                        prefix + "analytic_sufficient_rate": float(np.mean([r["analytic_sufficient"] for r in selected])),
                        prefix + "mean_total_queries": float(np.mean([r["total_queries"] for r in selected])),
                        prefix + "mean_extraction_seconds": float(np.mean([r["extraction_seconds"] for r in selected]))})
    for strategy in strategies:
        selected = [row for row in route_rows if row["strategy"] == strategy]
        accepted = [row for row in selected if row["accepted"]]
        prefix = "" if strategy == "cost" else f"{strategy}_"
        applicable = model.bounded and not model.learned
        metrics[prefix + "analytic_sufficient_rate"] = (float(np.mean([r["analytic_sufficient"] for r in selected]))
                                                        if applicable else None)
        metrics[prefix + "analytic_sufficient_given_accepted"] = (float(np.mean([r["analytic_sufficient"] for r in accepted]))
                                                                 if applicable and accepted else None)
        metrics[prefix + "accepted_paths"] = len(accepted)
        metrics[prefix + "cohort_windows"] = len(selected)
        metrics[prefix + "cohort_units"] = len({r["unit_id"] for r in selected})
    return metrics


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def validate_training(training: dict) -> None:
    if set(training) != TRAINING_KEYS:
        raise ValueError(f"Training config needs exactly {sorted(TRAINING_KEYS)}")
    for key in ("epochs", "batch_size"):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in TRAINING_KEYS - {"epochs", "batch_size"}:
        if not isinstance(training[key], (int, float)) or not np.isfinite(training[key]) or training[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if training["lr"] == 0 or training["head_lr_multiplier"] == 0:
        raise ValueError("lr and head_lr_multiplier must be positive")


def subset(data: dict, indices: np.ndarray) -> dict:
    return {key: value[indices].copy() if key != "sampling_rate" else value.copy() for key, value in data.items()}


def train_configured(data: dict, arm: str, seed: int, training: dict,
                     device: torch.device, output: Path) -> tuple[nn.Module, torch.Tensor, dict]:
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", dict(arm=arm, seed=seed, training=training,
        selection="minimum source-validation cross entropy", normalization="train-only channel mean/std"))
    started = time.perf_counter()
    try:
        model, x, details = train(data, arm, seed, training["epochs"], device,
            training["residual_weight"], output, lr=training["lr"], weight_decay=training["weight_decay"],
            batch_size=training["batch_size"], concentration_weight=training["concentration_weight"],
            head_lr_multiplier=training["head_lr_multiplier"])
        details["training_seconds"] = time.perf_counter() - started
        write_json(output / "selection_metrics.json", details)
        return model, x, details
    except Exception as error:
        fail(output, error)
        raise


def tune(data: dict, arms: list[str], seeds: list[int], config: dict,
         device: torch.device, output: Path) -> dict:
    """HPO receives source partitions only. Never evaluate a candidate on target test."""
    source = subset(data, np.flatnonzero(data["split"] != "test"))
    selections, rows = {}, []
    for arm in arms:
        candidates = []
        for index, patch in enumerate(config["search"]):
            training = config["training"] | patch
            losses = []
            for seed in seeds:
                target = output / f"{arm}-candidate{index}-seed{seed}"
                _, _, details = train_configured(source, arm, seed, training, device, target)
                loss = details["best_val_loss"]
                if not np.isfinite(loss):
                    raise RuntimeError("Candidate validation loss is nonfinite; search stopped")
                losses.append(loss)
                rows.append(dict(arm=arm, candidate=index, seed=seed, val_loss=loss, **training))
                write_csv(output / "search.csv", rows)
            candidates.append(dict(candidate=index, training=training, mean_val_loss=float(np.mean(losses))))
        selections[arm] = min(candidates, key=lambda candidate: (candidate["mean_val_loss"], candidate["candidate"]))
    selection = dict(arms=selections, tuning_seeds=seeds, candidate_count=len(config["search"]),
                     selection_rule="minimum mean source-validation CE across tuning seeds; index breaks ties")
    write_json(output / "selection.json", selection)
    return selection


def overfit(data: dict, arm: str, seed: int, config: dict, device: torch.device, output: Path) -> dict:
    """Diagnostic fit of a tiny source-training subset; no target or model selection."""
    torch.manual_seed(seed)
    x, mean, scale = normalize(subset(data, np.flatnonzero(data["split"] == "train")))
    y_all = data["y"][data["split"] == "train"]
    indices = np.concatenate([np.flatnonzero(y_all == label)[:2] for label in np.unique(y_all)])
    xb, yb = x[indices].to(device), torch.tensor(y_all[indices], device=device)
    model = build(arm, xb.shape[2], len(np.unique(y_all))).to(device)
    optimizer = make_optimizer(model, config["overfit"]["lr"], 0, config["overfit"]["head_lr_multiplier"])
    rows = []
    for step in range(config["overfit"]["steps"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(xb)
        loss = nn.functional.cross_entropy(logits, yb)
        if not torch.isfinite(loss):
            raise RuntimeError("Small-data overfit produced nonfinite loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Small-data overfit produced nonfinite gradients")
        optimizer.step()
        rows.append(dict(step=step, train_ce=float(loss.detach()), gradients_finite=True))
    model.eval()
    with torch.no_grad():
        logits, _ = model(xb)
        accuracy = float((logits.argmax(-1) == yb).float().mean())
        ce = float(nn.functional.cross_entropy(logits, yb))
    write_csv(output / "training.csv", rows)
    write_csv(output / "predictions.csv", [dict(training_sample=int(index), target=int(yb[k]),
        prediction=int(logits[k].argmax())) for k, index in enumerate(indices)])
    torch.save(dict(state_dict=model.state_dict(), mean=mean.tolist(), scale=scale.tolist(),
                    arm=arm, seed=seed, diagnostic_only=True), output / "model.pt")
    passed = accuracy >= config["overfit"]["min_accuracy"] and ce <= config["overfit"]["max_ce"]
    metrics = dict(arm=arm, seed=seed, status="completed" if passed else "failed",
                   training_accuracy=accuracy, training_ce=ce, initial_training_ce=rows[0]["train_ce"],
                   steps=config["overfit"]["steps"], gradients_finite=True, diagnostic_only=True)
    metrics["optimizer_groups"] = [dict(name=group["name"], lr=group["lr"],
        parameters=sum(parameter.numel() for parameter in group["params"])) for group in optimizer.param_groups]
    write_json(output / "metrics.json", metrics)
    if not passed:
        raise RuntimeError(f"Small-data overfit gate failed for {arm}: accuracy={accuracy:.4f}, CE={ce:.4f}")
    return metrics
