"""Installed execution of the existing bounded operator-path experiment.

The public config loader owns YAML composition. This module consumes its resolved
mapping and an explicitly exported NPZ; it does not read paper code or Git state.
Training, tuning and overfit phases do not evaluate the target partition.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.model_factory.X_model.P07OperatorPath import OPERATORS, OperatorNet

from .operators import (
    ARMS, TRAINING_KEYS, build, evaluate, normalize, overfit, subset, train_configured,
    source_partitions, tune, validate_data, validate_training, write_csv, write_json,
)

PHASES = ("export", "export-source", "data-check", "source-check", "tune", "train", "overfit", "evaluate", "replay")


def requested_device(config: Mapping[str, Any], device: str | None = None) -> torch.device:
    trainer = config["trainer"]
    if trainer.get("devices", 1) != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Operator-path experiments require one device; DDP is forbidden")
    requested = device if device is not None else trainer["device"]
    if requested == "cpu":
        return torch.device("cpu")
    if requested not in {"cuda", "cuda:0"}:
        raise ValueError("Request cpu or one explicitly visible CUDA device")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit() or int(visible) == 2:
        raise ValueError("Set one physical CUDA_VISIBLE_DEVICES index; physical GPU 2 is forbidden")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Requested CUDA device is unavailable; there is no CPU fallback")
    return torch.device("cuda:0")


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    task = config["task"]
    if task.get("name") != "operator_path" or task.get("execution") != "research":
        raise ValueError("Declare task.name=operator_path and task.execution=research")
    settings = dict(task["operator_path"])
    validate_training(settings["training"])
    arms = settings["arms"]
    if not arms or len(arms) != len(set(arms)) or set(arms) - set(ARMS):
        raise ValueError("Declare distinct supported operator-path arms")
    if settings.get("comparison") not in {"competitive", "mechanism"}:
        raise ValueError("comparison must distinguish competitive from mechanism controls")
    for key in ("seeds", "tuning_seeds"):
        seeds = settings[key]
        if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError(f"{key} must contain distinct nonnegative integers")
    if not settings["search"]:
        raise ValueError("Declare a nonempty common search budget")
    for patch in settings["search"]:
        if not isinstance(patch, dict) or set(patch) - TRAINING_KEYS:
            raise ValueError("Search candidates may change declared training parameters only")
        candidate = settings["training"] | patch
        validate_training(candidate)
        if any(candidate[k] != settings["training"][k] for k in ("epochs", "batch_size")):
            raise ValueError("Search cannot change the shared epoch or batch budget")
    if type(settings["intervention_limit"]) is not int or settings["intervention_limit"] < 1:
        raise ValueError("intervention_limit must be positive")
    if type(settings["search_budget"]) is not int or settings["search_budget"] < 20:
        raise ValueError("Both extraction strategies require at least 20 total forward calls")
    if settings.get("evaluation_partition") not in {"val", "test"}:
        raise ValueError("Explicit evaluation_partition must be val or test")
    return settings


def load_data(path: Path, domain_holdout: bool, *, source_only: bool = False) -> dict[str, np.ndarray]:
    """Consume the existing explicit P4 export without layout or label inference."""
    with np.load(path, allow_pickle=False) as archive:
        if source_only and set(archive["split"]) != {"train", "val"}:
            raise ValueError("Source-only phases require a train/val-only NPZ; target signals must remain unopened")
        data = {name: archive[name] for name in archive.files}
    validate_data(data, domain_holdout=domain_holdout,
                  required_splits=("train", "val") if source_only else ("train", "val", "test"))
    return data


def load_checkpoint(path: Path, data: dict[str, np.ndarray], device: torch.device,
                    settings: dict | None = None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"state_dict", "arm", "seed", "mean", "scale", "channels", "classes", "selected_epoch",
                "normalization_dtype", "source_partitions"}
    if not required <= checkpoint.keys():
        raise ValueError(f"Checkpoint lacks selected-model fields: {sorted(required - checkpoint.keys())}")
    if settings is not None and (checkpoint["arm"] not in settings["arms"] or checkpoint["seed"] not in settings["seeds"]):
        raise ValueError("Explicit checkpoint arm/seed is not declared in the configuration")
    if checkpoint["channels"] != data["x"].shape[2] or checkpoint["classes"] != len(np.unique(data["y"])):
        raise ValueError("Checkpoint channel/class contract differs from the explicit input")
    if checkpoint["source_partitions"] != source_partitions(data):
        raise ValueError("Checkpoint source unit/label/domain partitions differ from the explicit input")
    dtype = np.dtype(checkpoint["normalization_dtype"])
    if not np.issubdtype(dtype, np.floating) or dtype != data["x"].dtype:
        raise ValueError("Checkpoint normalization dtype differs from the explicit input")
    mean = np.asarray(checkpoint["mean"], dtype=dtype)
    scale = np.asarray(checkpoint["scale"], dtype=dtype)
    expected = (1, 1, checkpoint["channels"])
    if mean.shape != expected or scale.shape != expected or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Checkpoint must contain finite train-fitted channel statistics")
    model = build(checkpoint["arm"], checkpoint["channels"], checkpoint["classes"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    x = torch.from_numpy(((data["x"] - mean) / scale).astype(np.float32))
    if not torch.isfinite(x).all():
        raise ValueError("Checkpoint normalization produced nonfinite inputs")
    return checkpoint, model, x


def selected_training(settings: dict, selection: dict, arm: str) -> dict:
    if selection.get("candidate_count") != len(settings["search"]) or selection.get("tuning_seeds") != settings["tuning_seeds"]:
        raise ValueError("Selection search budget or tuning seeds differ from the current declaration")
    if selection.get("training") != settings["training"] or selection.get("search") != settings["search"]:
        raise ValueError("Selection training/search configuration differs from the current declaration")
    owner = "proposed" if settings["comparison"] == "mechanism" and arm in {"sparse", "routing_concentration", "unbounded"} else arm
    if owner not in selection["arms"]:
        raise ValueError(f"Source-only tuning has no configuration for {owner}")
    chosen = selection["arms"][owner]
    index = chosen["candidate"]
    if type(index) is not int or not 0 <= index < len(settings["search"]):
        raise ValueError("Selection candidate index is outside the declared common search")
    expected = settings["training"] | settings["search"][index]
    if chosen["training"] != expected:
        raise ValueError("Selection training values do not match its declared candidate")
    return expected.copy()


def _truth(value: str, name: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"Saved {name} must be True or False")
    return value == "True"


@torch.no_grad()
def replay_paths(checkpoint_path: Path, data: dict[str, np.ndarray], extraction_path: Path,
                 output: Path, device: torch.device, partition: str, cohort_limit: int,
                 settings: dict | None = None) -> dict:
    """Reload checkpoint and execute saved paths; never search for replacements.

    Numerical comparison tolerance checks serialization/reloading only. The
    scientific acceptance inequality retains its original exact 0.49 threshold.
    Additional verification forwards do not enter extraction-budget accounting.
    """
    checkpoint, model, x = load_checkpoint(checkpoint_path, data, device, settings)
    if not isinstance(model, OperatorNet):
        raise ValueError("A black-box control has no operator paths to replay")
    with extraction_path.open(newline="", encoding="utf-8") as stream:
        saved = list(csv.DictReader(stream))
    if not saved:
        raise ValueError("Extraction file has no prespecified cohort rows")
    indices = np.flatnonzero(data["split"] == partition)
    by_class = [indices[data["y"][indices] == label].tolist() for label in sorted(set(data["y"][indices]))]
    cohort = [group[offset] for offset in range(max(map(len, by_class)))
              for group in by_class if offset < len(group)][:cohort_limit]
    expected = {(strategy, int(index)) for strategy in ("cost", "perturbation") for index in cohort}
    labels = tuple(f"learned_{j}" for j in range(6)) if model.learned else OPERATORS
    seen, rows = set(), []
    for row in saved:
        sample = int(row["sample"])
        identity = (row["strategy"], sample)
        if identity in seen or row["strategy"] not in {"cost", "perturbation"}:
            raise ValueError("Extraction rows must have unique strategy/sample identities")
        seen.add(identity)
        if not 0 <= sample < len(x) or str(data["unit_id"][sample]) != row["unit_id"] or str(data["split"][sample]) != partition:
            raise ValueError("Saved extraction sample/unit/partition differs from the explicit input")
        if row["partition"] != partition or int(row["target"]) != int(data["y"][sample]) or row["domain"] != str(data["domain"][sample]):
            raise ValueError("Saved extraction label/domain/partition differs from the explicit input")
        if row["arm"] != checkpoint["arm"] or int(row["seed"]) != checkpoint["seed"]:
            raise ValueError("Saved extraction arm/seed differs from the explicit checkpoint")
        accepted = _truth(row["accepted"], "accepted")
        path_labels = json.loads(row["path"])
        result = dict(sample=sample, unit_id=row["unit_id"], strategy=row["strategy"],
                      accepted=accepted, target=int(data["y"][sample]), prediction=None,
                      agrees=None, exact_gap=None, analytic_sufficient=False,
                      replay_forward_calls=0)
        if not accepted:
            if path_labels is not None:
                raise ValueError("Rejected extraction cannot contain a returned path")
            rows.append(result)
            continue
        if not isinstance(path_labels, list) or len(path_labels) != model.stages or any(p not in labels for p in path_labels):
            raise ValueError("Saved path must name one declared operator per stage")
        path = tuple(labels.index(name) for name in path_labels)
        sample_x = x[sample:sample + 1].to(device)
        reference, trace = model(sample_x)
        discrete = model.discrete(sample_x, path)
        top = reference.topk(2, dim=-1).values[0]
        margin = float(top[0] - top[1])
        gap = float((reference - discrete).abs().max())
        prediction = int(discrete.argmax(-1))
        if int(reference.argmax(-1)) != int(row["prediction"]):
            raise ValueError("Reloaded reference prediction differs from saved extraction")
        for name, observed in (("margin", margin), ("exact_gap", gap)):
            if not np.isclose(observed, float(row[name]), rtol=1e-6, atol=1e-6):
                raise ValueError(f"Independent replay changed saved {name}")
        if not (margin > 0 and gap <= 0.49 * margin) or prediction != int(reference.argmax(-1)):
            raise ValueError("Returned path fails the original output-discrepancy acceptance criterion")
        applicable = model.bounded and not model.learned
        analytic = float(model.bound(trace, path)) if applicable else None
        sufficient = analytic is not None and 2 * analytic < margin
        if applicable and not np.isclose(analytic, float(row["analytic_bound"]), rtol=1e-6, atol=1e-6):
            raise ValueError("Independent replay changed the analytical bound")
        if sufficient != _truth(row["analytic_sufficient"], "analytic_sufficient"):
            raise ValueError("Independent replay changed the analytical sufficient condition")
        result.update(prediction=prediction, agrees=True, exact_gap=gap,
                      analytic_sufficient=sufficient, replay_forward_calls=2)
        rows.append(result)
    if seen != expected:
        raise ValueError("Saved extractions do not cover both strategies on the complete prespecified cohort")
    summaries = {}
    for strategy in sorted({r["strategy"] for r in rows}):
        cohort = [r for r in rows if r["strategy"] == strategy]
        accepted = [r for r in cohort if r["accepted"]]
        units = sorted({r["unit_id"] for r in cohort})
        applicable = model.bounded and not model.learned
        summaries[strategy] = dict(
            cohort_windows=len(cohort), cohort_units=len(units), accepted_paths=len(accepted),
            explanation_coverage=float(np.mean([r["accepted"] for r in cohort])),
            unit_mean_explanation_coverage=float(np.mean([
                np.mean([r["accepted"] for r in cohort if r["unit_id"] == unit]) for unit in units])),
            prediction_fidelity_given_accepted=float(np.mean([r["agrees"] for r in accepted])) if accepted else None,
            returned_path_accuracy=float(np.mean([r["prediction"] == r["target"] for r in accepted])) if accepted else None,
            wrong_given_accepted=float(np.mean([r["prediction"] != r["target"] for r in accepted])) if accepted else None,
            accepted_and_wrong_fraction=sum(r["prediction"] != r["target"] for r in accepted) / len(cohort),
            analytic_sufficient_rate=float(np.mean([r["analytic_sufficient"] for r in cohort])) if applicable else None,
            analytic_sufficient_given_accepted=float(np.mean([r["analytic_sufficient"] for r in accepted])) if applicable and accepted else None,
            replay_forward_calls=sum(r["replay_forward_calls"] for r in cohort),
        )
    write_csv(output / "replay.csv", rows)
    write_json(output / "replay_metrics.json", summaries)
    return summaries


def execute(config: Mapping[str, Any], phase: str, output: Path, *, data: Path,
            checkpoint: Path | None = None, selection: Path | None = None,
            device: str | None = None) -> dict[str, Any]:
    """Run one explicit phase; a software result is not scientific qualification."""
    if phase not in PHASES:
        raise ValueError(f"Unknown operator-path phase {phase!r}; choose {PHASES}")
    settings = _settings(config)
    if phase in {"export", "export-source"}:
        if checkpoint is not None or selection is not None:
            raise ValueError("Data export cannot consume a checkpoint or selection")
        from src.data_factory.operator_path_export import export_spec
        manifest = export_spec(Path(data), Path(output), source_only=phase == "export-source")
        result = dict(phase=phase, result_dir=str(Path(output).resolve()), manifest=str(manifest.resolve()),
                      source_only=phase == "export-source", status="completed", evidence_state="derived_data_not_qualified")
        result["run_summary"] = str(Path(output).resolve() / "run_summary.json")
        write_json(Path(result["run_summary"]), result)
        return result
    requested = requested_device(config, device)
    data_path = Path(data).expanduser().resolve(strict=True)
    domain_holdout = config["data"]["domain_holdout"]
    if type(domain_holdout) is not bool:
        raise ValueError("data.domain_holdout must be an explicit boolean")
    dataset = load_data(data_path, domain_holdout, source_only=phase in {"source-check", "tune", "train", "overfit"})
    if phase in {"evaluate", "replay"} and checkpoint is None:
        raise ValueError("Evaluation/replay requires an explicit selected checkpoint")
    if phase in {"train", "replay"} and selection is None:
        raise ValueError("Train requires source-only selection; replay requires saved extractions CSV")
    if phase not in {"evaluate", "replay"} and checkpoint is not None:
        raise ValueError("This phase cannot consume a checkpoint")
    if phase in {"tune", "overfit", "data-check", "source-check", "evaluate"} and selection is not None:
        raise ValueError("This phase does not consume --selection")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite operator-path results: {output}")
    selected = None
    if phase == "train":
        selected = json.loads(Path(selection).read_text())
        if selected.get("data_path") != str(data_path):
            raise ValueError("Source-only selection belongs to a different explicit data export")
        for arm in settings["arms"]:
            selected_training(settings, selected, arm)
    output.mkdir(parents=True)
    write_json(output / "config.json", dict(config=config, phase=phase, data_path=str(data_path),
        checkpoint=str(checkpoint) if checkpoint is not None else None,
        selection=str(selection) if selection is not None else None))
    result: dict[str, Any] = dict(phase=phase, result_dir=str(output), evidence_state="software_execution_only")
    arms = [arm for arm in ARMS if arm in settings["arms"]]
    source = subset(dataset, np.flatnonzero(dataset["split"] != "test"))
    if phase in {"data-check", "source-check"}:
        normalize(source)
        result["splits"] = {split: dict(windows=int((dataset["split"] == split).sum()),
            units=len(np.unique(dataset["unit_id"][dataset["split"] == split])),
            domains=sorted(str(v) for v in set(dataset["domain"][dataset["split"] == split])))
            for split in (("train", "val") if phase == "source-check" else ("train", "val", "test"))}
    elif phase == "tune":
        chosen = tune(source, arms, settings["tuning_seeds"], settings, requested, output)
        chosen.update(data_path=str(data_path), training=settings["training"], search=settings["search"])
        write_json(output / "selection.json", chosen)
        result["selection"] = str(output / "selection.json")
    elif phase == "train":
        result["runs"] = []
        for seed in settings["seeds"]:
            for arm in arms:
                target = output / f"{arm}-seed{seed}"
                training = selected_training(settings, selected, arm)
                _, _, details = train_configured(source, arm, seed, training, requested, target)
                result["runs"].append(dict(arm=arm, seed=seed, best_checkpoint=str(target / "model.pt"), **details))
    elif phase == "overfit":
        result["runs"] = []
        for seed in settings["seeds"]:
            for arm in arms:
                target = output / f"{arm}-seed{seed}"
                target.mkdir()
                result["runs"].append(overfit(source, arm, seed, settings, requested, target))
    elif phase == "evaluate":
        restored, model, x = load_checkpoint(Path(checkpoint), dataset, requested, settings)
        result["test_metrics"] = evaluate(model, x, dataset, restored["arm"], restored["seed"], output,
            settings["intervention_limit"], settings["search_budget"], strategies=("cost", "perturbation"),
            batch_size=settings["training"]["batch_size"], partition=settings["evaluation_partition"])
        result["best_checkpoint"] = str(Path(checkpoint).resolve())
        if isinstance(model, OperatorNet):
            result["independent_replay"] = replay_paths(Path(checkpoint), dataset, output / "extractions.csv",
                output, requested, settings["evaluation_partition"], settings["intervention_limit"], settings)
    else:
        result["independent_replay"] = replay_paths(Path(checkpoint), dataset, Path(selection), output,
            requested, settings["evaluation_partition"], settings["intervention_limit"], settings)
    result["run_summary"] = str(output / "run_summary.json")
    result["status"] = "completed"
    write_json(output / "run_summary.json", result)
    return result
