"""Native source-only study stages for the existing fixed signal-view estimator."""
from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stdout, redirect_stderr
import csv
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax

from .features import ProgramSpec, view_indices
from .metrics import aggregate_metrics, equal_unit_weights
from .readouts import OffsetReadout, ReadoutState, baseline_quality, fit_readouts

PHASES = ("sanity", "baseline_smoke", "method_smoke", "overfit", "tune", "fit", "source_compare", "evaluate", "analyze", "plot")


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON mapping: {path}")
    return value


def _arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _study(config: Mapping[str, Any]) -> dict[str, Any]:
    study = deepcopy(config["study"])
    if study["protocol"] not in {"op1", "condition_dg"}:
        raise ValueError("A bound OP1 or condition_dg protocol is required; HAR is not yet bound")
    if study["training_seeds"] != [11, 23, 37] or config["trainer"]["num_epochs"] != 100:
        raise ValueError("This study retains 100 epochs and registered seeds 11/23/37")
    if study.get("evaluation_role") != "exploratory":
        raise ValueError("Historical target exposure is not qualified for confirmatory labeling")
    if study["protocol"] == "condition_dg":
        source, target = set(study["source_domains"]), set(study["target_domains"])
        if len(source) < 2 or len(target) != 1 or source & target:
            raise ValueError("DG requires at least two source domains and one disjoint target")
    ProgramSpec.from_dict(study["program"])
    if len(study["class_names"]) != config["model"]["num_classes"]:
        raise ValueError("Fixed class names and model class count disagree")
    return study


def _prepare(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    """Write separate source inputs using declared entity roles; raw data stay immutable."""
    from src.data_factory.data_utils import read_metadata_table
    from src.data_factory.splitting import _sha256_file

    study = _study(config)
    source_root = Path(config["data"]["data_dir"]).resolve()
    if output.resolve().is_relative_to(source_root):
        raise ValueError("Derived preparation/output must be outside the read-only raw data root")
    metadata_path = source_root / config["data"]["metadata_file"]
    frame = read_metadata_table(metadata_path)
    frame = frame.loc[frame.Dataset_id.eq(study["dataset_id"])].copy()
    if frame.empty or frame.Id.duplicated().any():
        raise ValueError("Declared dataset must have a nonempty unique record population")
    group_key = study["unit_column"]
    if group_key not in frame and study["dataset_id"] == 20 and set(frame.Name) == {"RM_027_PU"}:
        # This reader's first path component is the published physical bearing ID.
        frame[group_key] = frame.File.str.split("/").str[0]
    if group_key not in frame or frame[group_key].isna().any():
        raise ValueError("Physical entity identity is unqualified; record/window IDs are not substitutes")
    if set(frame.Sample_rate) != {study["program"]["sample_rate_hz"]}:
        raise ValueError("Metadata sampling rate differs from the declared feature profile")
    plan = _load(study["role_map"])
    assignments = dict(plan["assignments"])
    for unit in plan.get("grow_units", []):
        assignments[unit] = "grow"
    for unit in plan.get("select_units", []):
        assignments[unit] = "select"
    frame["Role"] = frame[group_key].astype(str).map(assignments)
    if frame.Role.isna().any() or set(frame.Role) != {"fit", "tune", "grow", "select", "test"}:
        raise ValueError("The explicit physical role map must cover fit/tune/grow/select/test")
    observed_counts = frame.groupby("Role")[group_key].nunique().to_dict()
    if observed_counts != study["role_counts"]:
        raise ValueError(f"Physical role counts differ from the frozen protocol: {observed_counts}")
    if frame.groupby(group_key).Label.nunique().gt(1).any():
        raise ValueError("Physical entity has inconsistent class labels")
    if study["protocol"] == "condition_dg":
        if set(frame.Domain_id) != set(study["source_domains"]) | set(study["target_domains"]):
            raise ValueError("Observed operating-condition domains differ from the frozen protocol")
        source_mask = frame.Role.ne("test") & frame.Domain_id.isin(study["source_domains"])
        target_mask = frame.Role.eq("test") & frame.Domain_id.isin(study["target_domains"])
    else:
        source_mask, target_mask = frame.Role.ne("test"), frame.Role.eq("test")
    used = frame.loc[source_mask | target_mask].copy()
    frame.loc[~(source_mask | target_mask), ["Id", group_key, "Role", "Domain_id"]].to_csv(
        output / "excluded_cross_combinations.csv", index=False)
    expected_classes = set(range(len(study["class_names"])))
    for role in ("fit", "tune", "grow", "select", "test"):
        rows = used.loc[used.Role.eq(role)]
        if set(rows.Label) != expected_classes:
            raise ValueError(f"Frozen {role} population lacks declared class support")
    if set(used.loc[used.Role.ne("test"), group_key]) & set(used.loc[used.Role.eq("test"), group_key]):
        raise ValueError("Source and target physical entities overlap")
    source = used.loc[used.Role.ne("test")].copy()
    source_path, combined_path = output / "source_metadata.csv", output / "evaluation_metadata.csv"
    source.to_csv(source_path, index=False)
    used.to_csv(combined_path, index=False)

    def manifest(rows: Any, path: Path, metadata_file: Path) -> dict[str, Any]:
        value = {"schema_version": 1, "group_key": group_key, "stratify_key": "Label",
                 "metadata_file_sha256": _sha256_file(metadata_file),
                 "partitions": {role: {"ids": part.Id.astype(int).tolist(),
                                       "groups": sorted(part[group_key].astype(str).unique())}
                                for role, part in rows.groupby("Role", sort=False)}}
        _json(path, value)
        return value

    source_manifest = manifest(source, output / "source_split.json", source_path)
    full_manifest = manifest(used, output / "evaluation_split.json", combined_path)
    native = {key: deepcopy(config[key]) for key in ("environment", "data", "model", "trainer")}
    native["pipeline"] = "Pipeline_01_Fault_Diagnosis"
    native["task"] = deepcopy(study["native_task"])
    native["trainer"]["test_after_fit"] = False
    native["environment"].update(iterations=1, seed=11, wandb=False, swanlab=False)
    native["data"].update(metadata_file=str(source_path), cache_dir=str(output / "source_cache"),
        split={"strategy": "grouped_metadata", "group_key": group_key, "stratify_key": "Label",
               "manifest_mode": "read_only", "manifest_path": str(output / "source_split.json"),
               "test_policy": "partition", "partition_map": {"train": "fit", "val": "tune", "test": "grow"}})
    evaluation = deepcopy(native)
    evaluation["data"].update(metadata_file=str(combined_path), cache_dir=str(output / "evaluation_cache"))
    evaluation["data"]["split"].update(manifest_path=str(output / "evaluation_split.json"),
                                       partition_map={"train": "fit", "val": "tune", "test": "test"})
    role_records = {role: {"record_ids": part["ids"],
        "unit_by_record": {str(row.Id): str(getattr(row, group_key))
                           for row in used.loc[used.Role.eq(role)].itertuples()}}
        for role, part in full_manifest["partitions"].items()}
    prepared = {"study": study, "source_config": native, "evaluation_config": evaluation,
                "roles": role_records, "source_roles": list(source_manifest["partitions"]),
                "source_metadata": str(source_path), "evaluation_metadata": str(combined_path),
                "independent_unit": group_key, "source_root": str(source_root),
                "preparation": str(output.resolve()), "raw_data_modified": False}
    _json(output / "prepared.json", prepared)
    return prepared


def _temperature(logits: np.ndarray, labels: np.ndarray, units: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if (logits.ndim != 2 or len(logits) == 0 or not np.isfinite(logits).all()
            or labels.shape != (len(logits),) or labels.dtype.kind not in "iu"
            or labels.min() < 0 or labels.max() >= logits.shape[1]):
        raise ValueError("Temperature fitting requires aligned finite source-tune logits and labels")
    weights = equal_unit_weights(units)
    def nll(temperature: float) -> float:
        z = logits / temperature
        return float(weights @ (logsumexp(z, axis=1) - z[np.arange(len(labels)), labels]))
    result = minimize_scalar(nll, bounds=(.05, 20.), method="bounded", options={"xatol": 1e-8, "maxiter": 500})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Source-tune temperature fitting failed")
    return min([.05, float(result.x), 20.], key=nll)


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("Cannot report an empty result table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _native_configuration(prepared: Mapping[str, Any], model: Mapping[str, Any], seed: int,
                          lr: float, wd: float) -> dict[str, Any]:
    config = deepcopy(prepared["source_config"])
    config["model"] = deepcopy(model)
    config["environment"]["seed"] = seed
    config["task"].update(lr=lr, weight_decay=wd, epochs=100)
    return config


def _tune(prepared: Mapping[str, Any], config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    from .native import train_native
    study = prepared["study"]
    models = study["baseline_models"]
    trial_grid = ([(.001, .0001)] if study["protocol"] == "op1" else
                  [(.001, 0.), (.001, .0001), (.0003, 0.), (.0003, .0001)])
    selections, trials, failures = {}, [], []
    for name, model in models.items():
        candidates = []
        for index, (lr, wd) in enumerate(trial_grid):
            native = _native_configuration(prepared, model, 11, lr, wd)
            trial_path = output / name / f"trial_{index}"
            try:
                record = train_native(native, trial_path)
                tune = _arrays(record["tune_reference"])
                metrics = aggregate_metrics(softmax(tune["logits"], axis=1), tune["y"], tune["unit"])
                row = dict(model=name, trial=index, seed=11, lr=lr, weight_decay=wd,
                           epochs=100, status="completed", macro_f1=metrics["macro_f1"],
                           accuracy=metrics["accuracy"], brier=metrics["brier"],
                           checkpoint=record["checkpoint"])
                candidates.append(((-row["macro_f1"], -row["accuracy"], row["brier"], index), record, native))
                trials.append(row)
            except Exception as error:
                trial_path.mkdir(parents=True, exist_ok=True)
                _json(trial_path / "failure.json", {"status": "failed", "error": str(error),
                    "traceback": traceback.format_exc(), "target_scored": False})
                failures.append({"model": name, "trial": index, "error": str(error)})
        # A partial grid cannot silently become a lower search budget.
        if len(candidates) != len(trial_grid):
            _json(output / "tuning_failures.json", {"failures": failures, "completed_trials": trials})
            raise RuntimeError(f"Incomplete frozen search for {name}; repair failed trials before selecting")
        _, record, native = min(candidates, key=lambda value: value[0])
        records = {"11": record}
        for seed in (23, 37):
            repeated = deepcopy(native)
            repeated["environment"]["seed"] = seed
            records[str(seed)] = train_native(repeated, output / name / f"seed_{seed}")
        selections[name] = {"native_config": native, "runs": records,
                            "selection_rule": "uncalibrated equal-unit Macro-F1, accuracy, Brier, trial order"}
    _write_csv(output / "tuning_trials.csv", trials)
    selection = {"prepared": str(Path(prepared["preparation"]) / "prepared.json"),
                 "models": selections, "target_scored": False, "protocol": study["protocol"],
                 "trial_grid": trial_grid, "tuning_seed": 11, "training_seeds": [11, 23, 37]}
    _json(output / "selection.json", selection)
    return {"status": "completed", "selection": str(output / "selection.json"), "target_scored": False}


def _fit(prepared: Mapping[str, Any], selection: Mapping[str, Any], output: Path) -> dict[str, Any]:
    from .native import export_native
    study = prepared["study"]
    expected_models = set(study["baseline_models"])
    expected_seeds = {str(seed) for seed in study["training_seeds"]}
    if not expected_models or set(selection["models"]) != expected_models:
        raise ValueError("Selection must contain exactly every registered model before source freezing")
    for chosen in selection["models"].values():
        if set(chosen["runs"]) != expected_seeds:
            raise ValueError("Selection must contain exactly every registered seed before source freezing")
        for seed, record in chosen["runs"].items():
            if record["native_config"]["environment"]["seed"] != int(seed):
                raise ValueError("Native training record and registered seed disagree")
    freezes, qualifications = [], []
    for model_name, chosen in selection["models"].items():
        for seed in study["training_seeds"]:
            record = chosen["runs"][str(seed)]
            tune = _arrays(record["tune_reference"])
            temperature = _temperature(tune["logits"], tune["y"], tune["unit"])
            probability = softmax(tune["logits"] / temperature, axis=1)
            quality = baseline_quality(probability, tune["y"], tune["unit"], .8)
            quality.update(model=model_name, seed=seed, **aggregate_metrics(probability, tune["y"], tune["unit"]))
            qualifications.append(quality)
            destination = output / model_name / f"seed_{seed}"
            destination.mkdir(parents=True, exist_ok=False)
            _json(destination / "qualification.json", quality)
            if study["protocol"] == "op1" and not quality["qualified"]:
                _json(destination / "completion.json", {"status": "baseline_below_floor", "target_scored": False})
                continue
            state: dict[str, Any] = {"native_reference": True}
            if model_name == "TSPN":
                exported = export_native(record["native_config"], Path(record["checkpoint"]),
                    {role: prepared["roles"][role] for role in ("fit", "tune", "grow", "select")},
                    {"program": study["program"], "class_names": study["class_names"]},
                    destination / "source_export", record["native_config"]["model"]["device"],
                    reference=Path(record["tune_reference"]))
                values = _arrays(exported["operator_inputs"])
                masks = {role: values["role"] == role for role in ("grow", "select")}
                arguments = {f"{role}_{key}": values[field][masks[role]]
                    for role in masks for key, field in (("features", "program_features"), ("native", "native_features"),
                                                        ("labels", "y"), ("units", "unit"))}
                arguments.update({f"{role}_logits": values["logits"][masks[role]] / temperature for role in masks})
                fitted = fit_readouts(**arguments, feature_names=values["feature_names"].tolist())
                state = fitted.to_dict()
            freeze = {"schema_version": 1, "model": model_name, "seed": seed,
                      "checkpoint": record["checkpoint"], "temperature": temperature,
                      "states": state, "class_names": study["class_names"], "protocol": study["protocol"],
                      "prepared": selection["prepared"], "native_config": record["native_config"],
                      "target_scored": False}
            path = destination / "frozen.json"
            _json(path, freeze)
            freezes.append(str(path))
    # Below-floor OP1 seeds remain visible; they cannot be pooled as a passing subset.
    complete = len(freezes) == len(expected_models) * len(expected_seeds)
    index = {"status": "completed" if complete else "incomplete_qualification", "freezes": freezes,
             "qualifications": qualifications, "expected_seeds": study["training_seeds"],
             "prepared": selection["prepared"], "target_scored": False}
    _json(output / "frozen_index.json", index)
    return dict(index, selection=str(output / "frozen_index.json"))


def _complete_frozen_family(prepared: Mapping[str, Any], frozen_index: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the entire declared cohort before any target access or replay."""
    if frozen_index["status"] != "completed":
        raise ValueError("All registered model/seed states must be frozen before target evaluation")
    study = prepared["study"]
    expected = {(model, seed) for model in study["baseline_models"] for seed in study["training_seeds"]}
    family = [_load(path) for path in frozen_index["freezes"]]
    observed = [(state["model"], state["seed"]) for state in family]
    if not expected or len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Frozen family must contain exactly every registered model/seed with no duplicates")
    return family


def _evaluate(prepared: Mapping[str, Any], frozen_index: Mapping[str, Any], output: Path) -> dict[str, Any]:
    from .native import export_target_native
    family = _complete_frozen_family(prepared, frozen_index)
    evaluated = []
    study = prepared["study"]
    for frozen_path, frozen in zip(frozen_index["freezes"], family):
        native = deepcopy(prepared["evaluation_config"])
        native["model"] = deepcopy(frozen["native_config"]["model"])
        native["environment"]["seed"] = frozen["seed"]
        destination = output / frozen["model"] / f"seed_{frozen['seed']}"
        exported = export_target_native(native, Path(frozen["checkpoint"]), prepared["roles"]["test"],
            {"program": study["program"], "class_names": study["class_names"]}, destination / "target_export",
            native["model"]["device"], Path(frozen_path))
        values = _arrays(exported["operator_inputs"])
        if not np.all(values["role"] == "test"):
            raise ValueError("Target export includes undeclared source items")
        logits = values["logits"] / frozen["temperature"]
        if frozen["model"] == "TSPN":
            state = ReadoutState.from_dict(frozen["states"])
            probabilities = state.predict(values["program_features"], values["native_features"], logits)
            readout = OffsetReadout.from_dict(state.states["signal_program"]["readout"])
            terms = readout.contributions(values["program_features"])
            corrected = readout.logits(values["program_features"], logits)
            views = view_indices(values["program_features"].shape[1] // 42)
            view_terms = np.stack([terms[:, columns].sum(axis=1) for columns in views.values()], axis=1)
            reconstructed = logits + readout.intercept + view_terms.sum(axis=1)
            error = float(np.max(np.abs(corrected - reconstructed)))
            if error > 1e-10:
                raise RuntimeError("Saved view terms fail raw-logit reconstruction")
            np.savez_compressed(destination / "explanations.npz", offset=logits,
                corrected_logits=corrected, intercept=readout.intercept, feature_terms=terms,
                view_terms=view_terms, view_names=np.asarray(list(views)),
                class_logodds_correction=corrected - corrected[:, :1] - logits + logits[:, :1],
                mixing_strength=state.states["signal_program"]["projection"]["strength"],
                accepted=state.states["signal_program"]["guard"]["accepted"])
            _json(destination / "explanation_check.json", {"raw_logit_max_abs": error,
                "scope": "raw correction only; projected log-odds are generally nonlinear"})
        else:
            probabilities = {frozen["model"]: softmax(logits, axis=1)}
        metric_rows = []
        for method, probability in probabilities.items():
            metrics = aggregate_metrics(probability, values["y"], values["unit"])
            metric_rows.append({"method": method, "seed": frozen["seed"],
                               **{key: metrics[key] for key in ("brier", "accuracy", "macro_f1", "independent_units")}})
        _write_csv(destination / "metrics.csv", metric_rows)
        path = destination / "predictions.npz"
        np.savez_compressed(path, predictions=np.stack(list(probabilities.values()), axis=1),
                            methods=np.asarray(list(probabilities)), y=values["y"], unit=values["unit"],
                            item=values["item"], class_names=values["class_names"], seed=frozen["seed"])
        evaluated.append({"model": frozen["model"], "seed": frozen["seed"], "predictions": str(path)})
    _json(output / "evaluation.json", {"status": "completed", "runs": evaluated, "protocol": study["protocol"],
                                      "prepared": str(Path(prepared["preparation"]) / "prepared.json")})
    return {"status": "completed", "selection": str(output / "evaluation.json")}


def _source_compare(prepared: Mapping[str, Any], frozen_index: Mapping[str, Any], output: Path) -> dict[str, Any]:
    family = _complete_frozen_family(prepared, frozen_index)
    rows = []
    for path, frozen in zip(frozen_index["freezes"], family):
        if frozen["model"] != "TSPN":
            continue
        values = _arrays(Path(path).parent / "source_export" / "select.npz")
        state = ReadoutState.from_dict(frozen["states"])
        predictions = state.predict(values["program_features"], values["native_features"],
                                    values["logits"] / frozen["temperature"])
        for method, probabilities in predictions.items():
            metrics = aggregate_metrics(probabilities, values["y"], values["unit"])
            rows.append({"method": method, "seed": frozen["seed"],
                **{key: metrics[key] for key in ("brier", "accuracy", "macro_f1", "independent_units")}})
    _write_csv(output / "source_select_metrics.csv", rows)
    return {"status": "completed", "metrics": str(output / "source_select_metrics.csv"),
            "target_scored": False, "scope": "select diagnostics; these labels already determined the guard"}


def _execute_phase(config: Mapping[str, Any], phase: str, output: Path, *, data: Path | None = None,
            selection: Path | None = None, seed: int | None = None, device: str | None = None) -> dict[str, Any]:
    """Run one explicit stage through ``phmfactory research`` without target fallback."""
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase!r}; expected {PHASES}")
    if seed is not None:
        raise ValueError("Seeds come from the frozen protocol; arbitrary --seed changes are not accepted")
    configuration = deepcopy(dict(config))
    _study(configuration)
    if device is not None:
        if device not in {"cpu", "cuda:0"}:
            raise ValueError("Explicit cpu or physical-GPU-0 binding is required")
        if phase not in {"baseline_smoke", "method_smoke", "overfit"}:
            raise ValueError("Set training device in the same explicit local configuration before sanity; --device is for smoke only")
    output = Path(output).resolve()
    _json(output / "config.json", configuration)
    _json(output / "command.json", {"argv": sys.argv, "phase": phase})
    try:
        if phase == "sanity":
            prepared = _prepare(configuration, output)
            result = {"status": "completed", "data": str(output / "prepared.json"), "target_scored": False}
        else:
            if data is None:
                raise ValueError("Provide --data <sanity-output>/prepared.json")
            prepared = _load(data)
            if prepared["study"] != configuration["study"]:
                raise ValueError("Current study configuration differs from frozen preparation")
            if phase in {"baseline_smoke", "method_smoke", "overfit"}:
                from .native import train_native
                key = "ResNet1D" if phase in {"baseline_smoke", "overfit"} and "ResNet1D" in configuration["study"]["baseline_models"] else "TSPN"
                model = configuration["study"]["baseline_models"][key]
                native = _native_configuration(prepared, model, 11, .001, .0001)
                native["task"]["epochs"] = native["trainer"]["num_epochs"] = 100 if phase == "overfit" else 1
                native["data"]["num_window"] = 2
                if device is not None:
                    native["model"]["device"] = device
                    native["trainer"].update(device="cpu" if device == "cpu" else "cuda", devices=1)
                result = dict(train_native(native, output / "native_smoke"), status="smoke_only", target_scored=False)
                if phase == "overfit":
                    if result["fit_accuracy"] < .99:
                        raise RuntimeError("Tiny-source overfit check failed at the restored checkpoint; inspect retained native logs")
                    result.update(status="overfit_diagnostic_only", threshold=.99)
            elif phase == "tune":
                result = _tune(prepared, configuration, output)
            else:
                if selection is None:
                    raise ValueError("Provide --selection from the preceding source/evaluation stage")
                previous = _load(selection)
                if phase in {"fit", "source_compare", "evaluate", "analyze"}:
                    if Path(previous["prepared"]).resolve() != Path(data).resolve():
                        raise ValueError("Preceding artifact belongs to another prepared population/fold")
                if phase == "fit":
                    result = _fit(prepared, previous, output)
                elif phase == "source_compare":
                    result = _source_compare(prepared, previous, output)
                elif phase == "evaluate":
                    result = _evaluate(prepared, previous, output)
                elif phase == "analyze":
                    from .analysis import analyze
                    result = analyze(prepared, previous, output)
                else:
                    from .analysis import plot
                    result = plot(Path(selection).parent / "summary_metrics.csv", output)
        _json(output / "completion.json", result)
        return result
    except Exception as error:
        _json(output / "failure.json", {"status": "failed", "error": str(error), "phase": phase,
                                        "traceback": traceback.format_exc()})
        raise


def execute(config: Mapping[str, Any], phase: str, output: Path, *, data: Path | None = None,
            selection: Path | None = None, seed: int | None = None, device: str | None = None) -> dict[str, Any]:
    """Keep direct logs and failures for the public scientific-stage invocation."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "stdout.log").open("w", encoding="utf-8") as stdout, (output / "stderr.log").open("w", encoding="utf-8") as stderr:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                return _execute_phase(config, phase, output, data=data, selection=selection, seed=seed, device=device)
            except Exception as error:
                _json(output / "failure.json", {"status": "failed", "error": str(error), "phase": phase,
                                               "traceback": traceback.format_exc()})
                raise
