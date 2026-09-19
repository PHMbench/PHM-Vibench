"""Summarize D1 costs from explicit JSON/YAML artifacts, without model/data access.

The frozen source matrix determines missing slots. Label pools are deduplicated
across seeds and reused predictors; timing intervals retain their original scope.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

CORE_ARMS = ("MLP16", "O", "UO", "RO", "RC")
SEEDS = (42, 123, 456)


def _read(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def _status(scope: dict, execution: dict) -> str:
    if execution.get("exit_status") not in (None, 0) or scope.get("status") == "failed":
        return "failed"
    if execution.get("exit_status") == 0 and scope.get("status") in {"candidate_fitted", "classifier_fitted"}:
        return "completed"
    if scope.get("status") in {"candidate_fitted", "classifier_fitted"}:
        return "exit_unverified"
    return "running" if scope or execution else "missing"


def _access(scope: dict, pool: str) -> dict:
    names = ("fit_access", "fit") if pool == "source_fit" else ("selection_access", "selection")
    return next((scope[name] for name in names if name in scope), {})


def _run_specs(root: Path) -> list[tuple[str, int, str, Path]]:
    diagnostics = sorted((root/"configs"/"diagnostics").glob("*.yaml"))
    if not diagnostics:
        raise FileNotFoundError("The predeclared diagnostic configurations are required.")
    return [("p0", 42, "reference", root/"reference")] + [
        (arm, seed, "core", root/"core"/arm/f"seed_{seed}") for arm in CORE_ARMS for seed in SEEDS
    ] + [(path.stem, 42, "diagnostic", root/"diagnostics"/path.stem/"seed_42") for path in diagnostics] + [
        ("ResNet1D", seed, "baseline", root/"baselines"/"ResNet1D"/f"seed_{seed}") for seed in SEEDS
    ]


def _counts(records: list[dict], windows_per_acquisition: int) -> dict:
    acquisitions = {str(row["acquisition_id"]) for row in records}
    if len(acquisitions) != len(records):
        raise ValueError("Frozen metadata contains duplicate acquisition identities.")
    return dict(groups=len({str(row["unit_id"]) for row in records}), acquisitions=len(acquisitions),
                windows=len(acquisitions)*windows_per_acquisition, labelled_acquisitions=len(acquisitions))


def label_pools(root: Path, frozen: Path, runs: list[dict], f1: dict, f2: dict, runtime: dict) -> list[dict]:
    data_path = frozen/"data_config.yaml" if (frozen/"data_config.yaml").is_file() else root/"configs"/"data.yaml"
    data = yaml.safe_load(data_path.read_text())
    plan = f1.get("plan") or yaml.safe_load((root/"configs"/"d1_plan.yaml").read_text())
    dataset = next(row for row in data["datasets"] if row["name"] == plan["dataset"])
    sources = set(map(str, dataset["source_domains"]))
    domains = sorted(sources | set(map(str, dataset["domain_sequence"])))
    records = _read(frozen/"records.json")
    windows = int(data["data"]["windows_per_unit"])
    observed_runs = [run for run in runs if run["scope"]]
    independent = f2.get("mode", plan["mode"]) == "independent"
    consumed = {"source_fit": any(_access(run["scope"], "source_fit") for run in observed_runs),
                "source_selection": any(_access(run["scope"], "source_selection") for run in observed_runs) or bool(f1),
                "assessment": bool(f2) and independent, "test": bool(runtime)}
    splits = {"source_fit": "update", "source_selection": "validation", "assessment": "assessment", "test": "test"}
    rows, accessed = [], []
    for pool, split in splits.items():
        relevant_domains = domains if pool == "test" else sorted(sources)
        selected = None if records is None else [r for r in records if r["split"] == split and str(r["domain"]) in relevant_domains]
        access_status = "accessed" if consumed[pool] else "reserved_not_accessed"
        if pool == "assessment" and not independent:
            access_status = "not_applicable_empirical"
        if selected is not None and consumed[pool]:
            accessed.extend(selected)
        for domain in ["all", *relevant_domains]:
            counts, basis = {}, "missing_frozen_metadata"
            if selected is not None:
                part = selected if domain == "all" else [r for r in selected if str(r["domain"]) == domain]
                counts = _counts(part, windows)
                basis = "frozen acquisition identities; windows from frozen windows_per_unit"
            elif domain == "all" and pool in {"source_fit", "source_selection"}:
                observed = [_access(run["scope"], pool) for run in observed_runs]
                observed = [value for value in observed if value]
                if observed:
                    fields = ("groups", "acquisitions", "windows", "labelled_acquisitions")
                    if any(tuple(value.get(key) for key in fields) != tuple(observed[0].get(key) for key in fields) for value in observed):
                        raise ValueError(f"Observed source runs disagree on the shared {pool} label pool.")
                    counts, basis = observed[0], "observed result_scope pool counts; reused once across runs"
            rows.append(dict(pool=pool, split=split, domain=domain, access_status=access_status, **counts, counts_basis=basis,
                             reference_reuses_pool=pool in {"source_fit", "source_selection"}))
    if records is not None:
        # Each acquisition belongs to one partition; shared groups across conditions
        # count once globally and once in each relevant condition, never once/run.
        for domain in ["all", *domains]:
            part = accessed if domain == "all" else [r for r in accessed if str(r["domain"]) == domain]
            rows.append(dict(pool="distinct_accessed_union", split="multiple", domain=domain, access_status="accessed",
                             **_counts(part, windows), counts_basis="union of accessed frozen acquisition identities",
                             reference_reuses_pool=True))
    else:
        rows.append(dict(pool="distinct_accessed_union", domain="all", access_status="unverified",
                         counts_basis="requires frozen acquisition identities; pool sizes are not summed"))
    return rows


def summarize(run_root: Path, frozen_root: Path, latency_dirs: list[Path], output: Path) -> dict:
    root, frozen = run_root.resolve(), frozen_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    specs = _run_specs(root)
    missing, runs, costs = [], [], []
    p0_scope = _read(root/"reference"/"result_scope.json") or {}
    p0_parameters = p0_scope.get("trainable_parameters")
    parameters = {}
    for arm, seed, role, directory in specs:
        scope_path, execution_path = directory/"result_scope.json", directory/"execution_status.json"
        scope, execution = _read(scope_path) or {}, _read(execution_path) or {}
        missing.extend(str(path) for path in (scope_path, execution_path) if not path.is_file())
        status = _status(scope, execution)
        runs.append(dict(arm=arm, seed=seed, role=role, status=status, scope=scope))
        trainable = scope.get("trainable_parameters")
        direct = trainable if role in {"reference", "baseline"} else scope.get("total_parameters")
        reference = None if role == "reference" or not scope else scope.get("reference_parameters", p0_parameters)
        stored = (direct+reference if direct is not None and reference is not None else None) if role == "baseline" else direct
        parameters[(arm, seed)] = dict(direct=direct, reference=reference, stored=stored)
        base = dict(arm=arm, seed=seed, role=role, status=status, trainable_parameters=trainable,
                    direct_parameters=direct, reference_parameters=reference, stored_parameters=stored,
                    optimizer_steps=scope.get("optimizer_steps"), peak_allocated_gpu_bytes=scope.get("peak_allocated_gpu_bytes"),
                    exit_status=execution.get("exit_status"), benchmark_commit=execution.get("benchmark_commit"), run_directory=str(directory))
        stages = [("fit", "training_seconds", "source_fit", "training loop including forward/backward/update"),
                  ("source_selection", "source_selection_seconds" if "source_selection_seconds" in scope else "selection_seconds",
                   "source_selection", "per-epoch source-validation evaluation and checkpoint score"),
                  ("export", "export_seconds", None, "selected checkpoint/validation artifact writes"),
                  ("source_materialization", "source_materialization_seconds", None, "source waveform read and deterministic windows")]
        for stage, field, pool, boundary in stages:
            costs.append(dict(base, stage=stage, seconds=scope.get(field), timer_field=field, time_boundary=boundary,
                              measurement_status="measured" if field in scope else "missing", label_pool=pool,
                              artifact=str(scope_path), **(_access(scope, pool) if pool else {})))
        wall_field = "monotonic_wall_seconds" if "monotonic_wall_seconds" in execution else "wall_seconds"
        costs.append(dict(base, stage="process", seconds=execution.get(wall_field), timer_field=wall_field,
                          time_boundary="whole invocation monotonic timer; includes startup and I/O; overlaps internal timers",
                          measurement_status="measured" if wall_field in execution else "missing", artifact=str(execution_path)))
        if "filesystem_observed_interval_seconds" in execution:
            costs.append(dict(base, stage="filesystem_observed_interval", seconds=execution["filesystem_observed_interval_seconds"],
                              timer_field="filesystem_observed_interval_seconds", time_boundary=execution["interval_boundary"],
                              measurement_status="filesystem_interval_not_process_timer", artifact=str(execution_path)))

    firnet = _read(root/"firnet_status.json")
    if firnet is None:
        missing.append(str(root/"firnet_status.json"))
    for seed in SEEDS:
        costs.append(dict(arm="FIRNet", seed=seed, role="baseline", stage="fit", status=(firnet or {}).get("status", "missing"),
                          measurement_status="not_run" if firnet else "missing", reason=(firnet or {}).get("reason"), artifact=str(root/"firnet_status.json")))
    f1, f2, runtime = _read(frozen/"F1.json") or {}, _read(frozen/"F2.json") or {}, _read(frozen/"test"/"runtime.json") or {}
    protected = [
        ("source_freeze_and_selection", f1, frozen/"F1.json", "source_selection_seconds", "source_selection",
         "all-candidate source inference, temperature/coefficient selection, bundle and prediction export"),
        ("assessment_and_adoption", f2, frozen/"F2.json", "assessment_seconds", "assessment" if f2.get("mode") != "empirical" else "source_selection",
         "assessment predictions and decision plus deployment bundle and adopted-source prediction export"),
        ("test_export_invocation", runtime, frozen/"test"/"runtime.json", "inference_seconds", "test",
         runtime.get("note", "current test-export invocation; not necessarily all resumed invocation costs")),
    ]
    for stage, contents, path, field, pool, boundary in protected:
        if not path.is_file():
            missing.append(str(path))
        costs.append(dict(arm="shared", role="frozen_protocol", stage=stage, status="completed" if contents else "missing",
                          seconds=contents.get(field), timer_field=field, time_boundary=boundary, label_pool=pool,
                          measurement_status="measured" if field in contents else "missing", artifact=str(path)))
    labels = label_pools(root, frozen, runs, f1, f2, runtime)
    if not (frozen/"records.json").is_file():
        missing.append(str(frozen/"records.json"))
    label_counts = {row["pool"]: row for row in labels if row["domain"] == "all"}
    for row in costs:
        if row.get("role") == "frozen_protocol" and row["status"] == "completed":
            counts = label_counts[row["label_pool"]]
            row.update({key: counts.get(key) for key in ("groups", "acquisitions", "windows", "labelled_acquisitions")})
    specs_by_bundle = {str(Path(spec["bundle"]).resolve()): spec for spec in (f2 or f1).get("predictors", [])}
    latency = []
    for directory in latency_dirs:
        path = directory.resolve()/"latency.json"
        measured = _read(path)
        if measured is None:
            missing.append(str(path))
            latency.append(dict(measurement_status="missing", artifact=str(path)))
            continue
        spec = specs_by_bundle.get(str(Path(measured["bundle"]).resolve()), {})
        deployment = measured["deployment"]
        candidate_parameters = parameters.get((spec.get("arm"), spec.get("seed")), {}).get("direct")
        for row in measured["paths"]:
            executed = candidate_parameters
            if row["path"] == "p0" or deployment["kind"] == "temperature" or (row["path"] == "deployed" and deployment["alpha"] == 0):
                executed = p0_parameters
            elif row["path"] == "deployed" and spec.get("arm") == "ResNet1D" and 0 < deployment["alpha"] < 1:
                executed = parameters.get(("ResNet1D", spec.get("seed")), {}).get("stored")
            latency.append(dict(arm=spec.get("arm"), seed=spec.get("seed"), role=spec.get("role"),
                                **{key: value for key, value in row.items() if key != "repeats_ms"},
                                measurement_status="measured", executed_parameters=executed,
                                total_loaded_parameters=measured.get("total_parameters"), bundle=measured["bundle"],
                                device=measured["device"], gpu=measured.get("gpu"), dtype=measured["dtype"],
                                input_shape=json.dumps(measured["input_shape"]), warmup=measured["warmup"], repeats=measured["repeats"],
                                group_id=measured["group_id"], acquisition_id=measured["acquisition_id"], domain=measured["domain"],
                                split=measured["split"], window_id=measured["window_id"], io_boundary=measured["io_boundary"],
                                deployment_kind=deployment["kind"], alpha=deployment["alpha"], artifact=str(path)))
    notes = dict(run_root=str(root), frozen_root=str(frozen), expected_source_runs=len(specs),
                 completed_source_runs=sum(run["status"] == "completed" for run in runs),
                 missing_artifacts=sorted(set(missing)), firnet_status=firnet,
                 interpretation=[
                     "Blank fields mean unavailable/not separately measured; they are not zero costs.",
                     "Process/filesystem/protected-stage intervals overlap component timers and must not be summed together.",
                     "Reference pretraining is one separate run; reused labels are not charged once per arm or seed.",
                     "Label rows are unique acquisition pools, with physical groups deduplicated across conditions; reserved rows are not evidence of access.",
                     "Window counts use the frozen windows_per_unit declaration; one acquisition label is reused across its windows.",
                     "ResNet direct parameters exclude the stored reference; measured peak memory may include both resident models.",
                     "Latency repetitions are timing repeats of one source-validation input, not independent specimens.",
                     "No latency was rerun; no checkpoint, raw waveform or prediction NPZ was read.",
                 ])
    output.mkdir(parents=True)
    _csv(output/"cost.csv", costs, ["arm", "seed", "role", "stage", "status", "measurement_status", "seconds", "timer_field",
         "time_boundary", "trainable_parameters", "direct_parameters", "reference_parameters", "stored_parameters", "optimizer_steps",
         "peak_allocated_gpu_bytes", "label_pool", "groups", "acquisitions", "windows", "labelled_acquisitions", "exit_status",
         "benchmark_commit", "reason", "run_directory", "artifact"])
    _csv(output/"label_budget.csv", labels, ["pool", "split", "domain", "access_status", "groups", "acquisitions", "windows",
         "labelled_acquisitions", "reference_reuses_pool", "counts_basis"])
    _csv(output/"latency.csv", latency, ["arm", "seed", "role", "path", "measurement_status", "median_ms", "q1_ms", "q3_ms", "iqr_ms",
         "peak_allocated_bytes", "executed_parameters", "total_loaded_parameters", "device", "gpu", "dtype", "input_shape", "warmup", "repeats",
         "group_id", "acquisition_id", "domain", "split", "window_id", "io_boundary", "deployment_kind", "alpha", "bundle", "artifact"])
    (output/"notes.json").write_text(json.dumps(notes, indent=2, allow_nan=False)+"\n")
    return notes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--frozen-root", type=Path)
    p.add_argument("--latency-dir", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    summarize(args.run_root, args.frozen_root or args.run_root/"frozen", args.latency_dir, args.output)


if __name__ == "__main__":
    main()
