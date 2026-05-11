"""Build auditable PHM generative benchmark-effect reports.

The runner has two intentionally separate modes:

- ``--dry-run`` writes a command plan for real PHM train/sample/eval/paperpack runs.
- ``--from-runs`` aggregates existing run directories into effect tables.

This keeps long training out of CI while still making the benchmark contract
explicit and testable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


QUALITY_PREFIXES = ("temporal_", "spectral_", "distribution_", "diversity_")
UTILITY_PREFIXES = ("tstr_", "trts_", "utility_")
STATUS_SUFFIXES = ("_status", "_reason", "_status_code")


@dataclass(frozen=True)
class BenchmarkMethod:
    method: str
    label: str
    train_config: str
    condition_sampling_policy: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkMatrix:
    path: Path
    benchmark_id: str
    dataset: str
    output_dir: Path
    baseline_method: str
    seeds: list[int]
    python: str
    data_check: dict[str, Any]
    overrides: dict[str, Any]
    methods: list[BenchmarkMethod]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"matrix must be a YAML mapping: {path}")
    return data


def load_matrix(path: Path) -> BenchmarkMatrix:
    data = _read_yaml(path)
    benchmark = data.get("benchmark") or {}
    if not isinstance(benchmark, dict):
        raise ValueError("matrix.benchmark must be a mapping")

    methods = []
    for item in data.get("methods") or []:
        if not isinstance(item, dict):
            raise ValueError("each matrix method must be a mapping")
        method = str(item.get("method", "")).strip()
        train_config = str(item.get("train_config", "")).strip()
        if not method or not train_config:
            raise ValueError("each method requires method and train_config")
        methods.append(
            BenchmarkMethod(
                method=method,
                label=str(item.get("label", method)),
                train_config=train_config,
                condition_sampling_policy=str(item.get("condition_sampling_policy", "")),
                overrides=dict(item.get("overrides") or {}),
            )
        )

    if not methods:
        raise ValueError("matrix must define at least one method")

    seeds = [int(seed) for seed in benchmark.get("seeds", [])]
    if not seeds:
        raise ValueError("benchmark.seeds must not be empty")

    return BenchmarkMatrix(
        path=path,
        benchmark_id=str(benchmark.get("id", "phm_generative_benchmark_effect")),
        dataset=str(benchmark.get("dataset", "unknown_dataset")),
        output_dir=Path(str(benchmark.get("output_dir", "results/paper/phm_generative/benchmark_effect"))),
        baseline_method=str(benchmark.get("baseline_method", methods[0].method)),
        seeds=seeds,
        python=str(benchmark.get("python", sys.executable)),
        data_check=dict(benchmark.get("data_check") or {}),
        overrides=dict(benchmark.get("overrides") or {}),
        methods=methods,
    )


def _format_override_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _append_overrides(cmd: list[str], overrides: dict[str, Any]) -> list[str]:
    for key, value in overrides.items():
        cmd.extend(["--override", f"{key}={_format_override_value(value)}"])
    return cmd


def _stage_output_dir(matrix: BenchmarkMatrix, method: str, seed: int, stage: str) -> str:
    return str(matrix.output_dir / "runs" / method / f"seed_{seed}" / stage)


def _stage_project(matrix: BenchmarkMatrix, method: str, seed: int, stage: str) -> str:
    return f"{matrix.benchmark_id}_{method}_seed{seed}_{stage}"


def _stage_overrides(
    matrix: BenchmarkMatrix,
    method: BenchmarkMethod,
    seed: int,
    stage: str,
    extra_overrides: dict[str, Any],
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    overrides.update(matrix.overrides)
    overrides.update(method.overrides)
    overrides.update(
        {
            "environment.seed": seed,
            "environment.project": _stage_project(matrix, method.method, seed, stage),
            "environment.output_dir": _stage_output_dir(matrix, method.method, seed, stage),
            "task.generative.mode": stage,
            "task.generative.synthetic_dataset_id": f"{matrix.dataset}_{method.method}_seed{seed}",
        }
    )
    if method.condition_sampling_policy:
        overrides["task.generative.condition_sampling_policy"] = method.condition_sampling_policy
    overrides.update(extra_overrides)
    return overrides


def build_run_plan(matrix: BenchmarkMatrix, cli_overrides: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manual_overrides = _parse_cli_overrides(cli_overrides)
    for method in matrix.methods:
        for seed in matrix.seeds:
            stage_specs = [
                ("train", {}),
                (
                    "sample",
                    {
                        "task.generative.checkpoint_path": str(
                            matrix.output_dir
                            / "runs"
                            / method.method
                            / f"seed_{seed}"
                            / "train"
                            / "<experiment_name>"
                            / "iter_0"
                            / "checkpoints"
                            / "best.ckpt"
                        )
                    },
                ),
                (
                    "eval",
                    {
                        "task.generative.generated_path": str(
                            matrix.output_dir
                            / "runs"
                            / method.method
                            / f"seed_{seed}"
                            / "sample"
                            / "<experiment_name>"
                            / "iter_0"
                            / "synthetic"
                            / "samples.pt"
                        ),
                        "task.generative.eval_split": "train",
                    },
                ),
            ]
            for stage, stage_extra in stage_specs:
                overrides = _stage_overrides(matrix, method, seed, stage, stage_extra)
                overrides.update(manual_overrides)
                cmd = [matrix.python, "main.py", "--config", method.train_config]
                _append_overrides(cmd, overrides)
                rows.append(
                    {
                        "benchmark_id": matrix.benchmark_id,
                        "dataset": matrix.dataset,
                        "method": method.method,
                        "method_label": method.label,
                        "seed": seed,
                        "stage": stage,
                        "config": method.train_config,
                        "command": shlex.join(cmd),
                    }
                )
            paperpack_run_dir = (
                matrix.output_dir / "runs" / method.method / f"seed_{seed}" / "eval" / "<experiment_name>" / "iter_0"
            )
            paperpack_cmd = [
                matrix.python,
                "-m",
                "scripts.paperpack_generative",
                "--run_dir",
                str(paperpack_run_dir),
            ]
            rows.append(
                {
                    "benchmark_id": matrix.benchmark_id,
                    "dataset": matrix.dataset,
                    "method": method.method,
                    "method_label": method.label,
                    "seed": seed,
                    "stage": "paperpack",
                    "config": "",
                    "command": shlex.join(paperpack_cmd),
                }
            )
    return rows


def _parse_cli_overrides(items: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid override {item!r}; expected key=value")
        key, value = item.split("=", 1)
        try:
            parsed[key] = yaml.safe_load(value)
        except yaml.YAMLError:
            parsed[key] = value
    return parsed


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_run_plan(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(
        path,
        rows,
        ["benchmark_id", "dataset", "method", "method_label", "seed", "stage", "config", "command"],
    )


def validate_matrix_inputs(matrix: BenchmarkMatrix, *, allow_missing_data: bool) -> list[str]:
    errors: list[str] = []
    method_ids = {method.method for method in matrix.methods}
    if matrix.baseline_method not in method_ids:
        errors.append(f"baseline_method is not defined in methods: {matrix.baseline_method}")
    for method in matrix.methods:
        if not Path(method.train_config).exists():
            errors.append(f"missing train_config for {method.method}: {method.train_config}")
    data_dir = matrix.data_check.get("data_dir") or matrix.overrides.get("data.data_dir")
    metadata_file = matrix.data_check.get("metadata_file") or matrix.overrides.get("data.metadata_file")
    if data_dir and metadata_file:
        metadata_path = Path(str(data_dir)) / str(metadata_file)
        if not metadata_path.exists() and not allow_missing_data:
            errors.append(f"missing PHM metadata: {metadata_path}")
    elif not allow_missing_data:
        errors.append("matrix must define data_check.data_dir and data_check.metadata_file")
    return errors


def _to_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _finite_float(value: Any) -> float | None:
    numeric = _to_float(value)
    if numeric is None or not math.isfinite(numeric):
        return None
    return numeric


def _is_metric_key(key: str) -> bool:
    if key in {"source_path", "method", "seed", "dataset"}:
        return False
    if key.endswith(STATUS_SUFFIXES):
        return False
    return key.startswith(QUALITY_PREFIXES) or key.startswith(UTILITY_PREFIXES)


def _metric_category(metric: str) -> str:
    if metric.startswith(UTILITY_PREFIXES):
        return "utility"
    return "quality"


def _metric_direction(metric: str) -> str:
    if metric.startswith(UTILITY_PREFIXES) or metric.startswith("diversity_"):
        return "higher_better"
    return "lower_better"


def _status_for(row: dict[str, str], metric: str, numeric: float | None) -> str:
    status = row.get(f"{metric}_status", "").strip()
    if status:
        return status
    status_code = _finite_float(row.get(f"{metric}_status_code"))
    if status_code is not None:
        return "ok" if status_code >= 1.0 else "not_computable"
    return "ok" if numeric is not None else "not_computable"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _infer_method(path: Path, methods: list[BenchmarkMethod], manifest: dict[str, Any] | None) -> str:
    candidates = " ".join([str(path), json.dumps(manifest or {}, ensure_ascii=False)])
    for method in methods:
        if method.method in candidates:
            return method.method
    synthetic_id = str((manifest or {}).get("synthetic_dataset_id", ""))
    if synthetic_id:
        return synthetic_id
    return "unknown_method"


def _infer_seed(path: Path, manifest: dict[str, Any] | None) -> int | None:
    seed = (manifest or {}).get("sampling", {}).get("seed")
    if seed is not None:
        return int(seed)
    match = re.search(r"seed[_-]?(\d+)", str(path))
    return int(match.group(1)) if match else None


def _run_status(manifest: dict[str, Any] | None) -> tuple[str, str]:
    if manifest is None:
        return "exploratory", "missing synthetic_data_manifest.json"
    validity = manifest.get("validity", {})
    status = str(validity.get("status") or "")
    benchmark_valid = validity.get("benchmark_valid")
    if status == "benchmark-valid" or benchmark_valid is True:
        return "benchmark-valid", ""
    missing = validity.get("missing_evidence", [])
    reason = ";".join(str(item) for item in missing) if isinstance(missing, list) else str(missing)
    return status or "exploratory", reason


def collect_metric_records(run_dirs: list[Path], matrix: BenchmarkMatrix) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for root in run_dirs:
        for metric_path in sorted(root.rglob("generative_eval_metrics.csv")):
            manifest_paths = sorted(metric_path.parent.rglob("synthetic_data_manifest.json"))
            if not manifest_paths:
                manifest_paths = sorted(metric_path.parent.parent.rglob("synthetic_data_manifest.json"))
            manifest_path = manifest_paths[-1] if manifest_paths else None
            manifest = _read_json(manifest_path) if manifest_path is not None else None
            method = _infer_method(metric_path, matrix.methods, manifest)
            seed = _infer_seed(metric_path, manifest)
            run_status, status_reason = _run_status(manifest)
            for row in _metric_rows(metric_path):
                for metric, value in row.items():
                    if not _is_metric_key(metric):
                        continue
                    numeric = _finite_float(value)
                    status = _status_for(row, metric, numeric)
                    missing = numeric is None or status == "not_computable"
                    records.append(
                        {
                            "dataset": matrix.dataset,
                            "method": method,
                            "seed": seed if seed is not None else "",
                            "metric": metric,
                            "category": _metric_category(metric),
                            "direction": _metric_direction(metric),
                            "value": numeric,
                            "status": status,
                            "reason": row.get(f"{metric}_reason", ""),
                            "missing": missing,
                            "benchmark_status": run_status,
                            "benchmark_status_reason": status_reason,
                            "manifest_path": str(manifest_path) if manifest_path else "",
                            "metric_source_path": str(metric_path),
                        }
                    )
    return records


def aggregate_effects(records: list[dict[str, Any]], baseline_method: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault((str(record["dataset"]), str(record["method"]), str(record["metric"])), []).append(record)

    rows: list[dict[str, Any]] = []
    for (dataset, method, metric), items in sorted(groups.items()):
        values = [float(item["value"]) for item in items if item.get("value") is not None and not item.get("missing")]
        missing_items = [item for item in items if item.get("missing")]
        reasons = sorted({str(item.get("reason") or item.get("benchmark_status_reason") or "") for item in missing_items})
        statuses = {str(item.get("benchmark_status", "exploratory")) for item in items}
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "metric": metric,
                "category": items[0]["category"],
                "direction": items[0]["direction"],
                "mean": statistics.mean(values) if values else "",
                "std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else ""),
                "n": len(values),
                "missing_count": len(missing_items),
                "missing_reasons": " | ".join(reason for reason in reasons if reason),
                "baseline_method": baseline_method,
                "baseline_mean": "",
                "delta_vs_baseline": "",
                "relative_delta_vs_baseline": "",
                "rank": "",
                "benchmark_status": "benchmark-valid" if statuses == {"benchmark-valid"} else "exploratory",
                "manifest_paths": ";".join(sorted({str(item.get("manifest_path", "")) for item in items if item.get("manifest_path")})),
                "metric_source_paths": ";".join(
                    sorted({str(item.get("metric_source_path", "")) for item in items if item.get("metric_source_path")})
                ),
            }
        )

    baseline_by_metric = {
        (row["dataset"], row["metric"]): row
        for row in rows
        if row["method"] == baseline_method and row["mean"] != ""
    }
    for row in rows:
        baseline = baseline_by_metric.get((row["dataset"], row["metric"]))
        if baseline is None:
            continue
        baseline_mean = float(baseline["mean"])
        row["baseline_mean"] = baseline_mean
        if row["mean"] != "":
            mean = float(row["mean"])
            delta = mean - baseline_mean
            row["delta_vs_baseline"] = delta
            row["relative_delta_vs_baseline"] = delta / abs(baseline_mean) if baseline_mean else ""

    for dataset_metric in sorted({(row["dataset"], row["metric"]) for row in rows}):
        metric_rows = [row for row in rows if (row["dataset"], row["metric"]) == dataset_metric and row["mean"] != ""]
        if not metric_rows:
            continue
        direction = str(metric_rows[0]["direction"])
        reverse = direction == "higher_better"
        metric_rows.sort(key=lambda row: float(row["mean"]), reverse=reverse)
        for rank, row in enumerate(metric_rows, start=1):
            row["rank"] = rank

    return rows


def write_missing_metrics(path: Path, records: list[dict[str, Any]]) -> None:
    missing = [record for record in records if record.get("missing")]
    lines = ["# Missing Benchmark-Effect Metrics", ""]
    if not missing:
        lines.append("No missing quality or utility metrics were reported.")
    else:
        for record in missing:
            reason = record.get("reason") or record.get("benchmark_status_reason") or "no reason recorded"
            lines.append(
                f"- `{record['method']}` seed `{record['seed']}` `{record['metric']}`: {reason} "
                f"({record['metric_source_path']})"
            )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, matrix: BenchmarkMatrix, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# PHM Generative Benchmark Effect Report",
        "",
        f"Benchmark: `{matrix.benchmark_id}`",
        f"Dataset: `{matrix.dataset}`",
        f"Baseline method: `{matrix.baseline_method}`",
        "",
        "## Top Utility Metrics",
        "",
    ]
    utility = [row for row in summary_rows if row["category"] == "utility" and row["mean"] != ""]
    if utility:
        utility.sort(key=lambda row: (row["metric"], int(row["rank"] or 999)))
        for row in utility:
            lines.append(
                f"- `{row['metric']}` `{row['method']}`: mean={row['mean']} "
                f"delta={row['delta_vs_baseline']} rank={row['rank']}"
            )
    else:
        lines.append("No computable utility metrics were found.")
    lines.extend(["", "## Evidence", ""])
    lines.append("Every summary row must retain metric and manifest source paths in `benchmark_effect_summary.csv`.")
    lines.append("Rows are exploratory unless all contributing manifests are benchmark-valid.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, matrix: BenchmarkMatrix, run_dirs: list[Path], summary_rows: list[dict[str, Any]]) -> None:
    statuses = sorted({str(row.get("benchmark_status", "exploratory")) for row in summary_rows})
    payload = {
        "benchmark_id": matrix.benchmark_id,
        "dataset": matrix.dataset,
        "baseline_method": matrix.baseline_method,
        "matrix_path": str(matrix.path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dirs": [str(path) for path in run_dirs],
        "summary_rows": len(summary_rows),
        "benchmark_statuses": statuses,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_effect_report(matrix: BenchmarkMatrix, run_dirs: list[Path], output_dir: Path) -> Path:
    records = collect_metric_records(run_dirs, matrix)
    summary_rows = aggregate_effects(records, matrix.baseline_method)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "benchmark_effect_summary.csv",
        summary_rows,
        [
            "dataset",
            "method",
            "metric",
            "category",
            "direction",
            "mean",
            "std",
            "n",
            "missing_count",
            "missing_reasons",
            "baseline_method",
            "baseline_mean",
            "delta_vs_baseline",
            "relative_delta_vs_baseline",
            "rank",
            "benchmark_status",
            "manifest_paths",
            "metric_source_paths",
        ],
    )
    write_missing_metrics(output_dir / "missing_metrics.md", records)
    write_report(output_dir / "benchmark_effect_report.md", matrix, summary_rows)
    write_manifest(output_dir / "benchmark_effect_manifest.json", matrix, run_dirs, summary_rows)
    return output_dir


def execute_plan(rows: list[dict[str, Any]], out_csv: Path) -> int:
    placeholder_rows = [row for row in rows if "<experiment_name>" in str(row.get("command", ""))]
    if placeholder_rows:
        stages = ", ".join(sorted({str(row.get("stage", "")) for row in placeholder_rows}))
        raise ValueError(
            "execution plan contains placeholder artifact paths for stage(s): "
            f"{stages}. Use --stages train for executable smoke runs, then aggregate "
            "completed run directories with --from-runs."
        )
    executed: list[dict[str, Any]] = []
    for row in rows:
        cmd = shlex.split(str(row["command"]))
        start = time.perf_counter()
        result = subprocess.run(cmd, text=True, capture_output=True)
        executed.append(
            {
                **row,
                "returncode": result.returncode,
                "wall_clock_sec": f"{time.perf_counter() - start:.6f}",
                "stdout_tail": result.stdout[-1000:],
                "stderr_tail": result.stderr[-1000:],
            }
        )
        if result.returncode != 0:
            break
    _write_csv(
        out_csv,
        executed,
        [
            "benchmark_id",
            "dataset",
            "method",
            "method_label",
            "seed",
            "stage",
            "config",
            "command",
            "returncode",
            "wall_clock_sec",
            "stdout_tail",
            "stderr_tail",
        ],
    )
    return 1 if any(int(row["returncode"]) != 0 for row in executed) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PHM generative benchmark-effect runner.")
    parser.add_argument("--matrix", required=True, help="Benchmark-effect matrix YAML.")
    parser.add_argument("--output-dir", default=None, help="Override benchmark output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Write run_plan.csv without executing commands.")
    parser.add_argument("--from-runs", nargs="*", default=None, help="Aggregate existing run directories.")
    parser.add_argument("--execute", action="store_true", help="Execute the generated command plan.")
    parser.add_argument(
        "--stages",
        default="train,sample,eval,paperpack",
        help="Comma-separated stages to include in the command plan.",
    )
    parser.add_argument("--allow-missing-data", action="store_true", help="Allow missing local PHM data in planning mode.")
    parser.add_argument("--override", action="append", default=[], help="Extra key=value override appended to commands.")
    args = parser.parse_args(argv)

    matrix = load_matrix(Path(args.matrix))
    output_dir = Path(args.output_dir) if args.output_dir else matrix.output_dir
    matrix = BenchmarkMatrix(
        path=matrix.path,
        benchmark_id=matrix.benchmark_id,
        dataset=matrix.dataset,
        output_dir=output_dir,
        baseline_method=matrix.baseline_method,
        seeds=matrix.seeds,
        python=matrix.python,
        data_check=matrix.data_check,
        overrides=matrix.overrides,
        methods=matrix.methods,
    )

    errors = validate_matrix_inputs(matrix, allow_missing_data=bool(args.allow_missing_data))
    if errors:
        for error in errors:
            print(f"[FAIL] {error}", file=sys.stderr)
        return 2

    if args.dry_run or args.execute:
        plan_rows = build_run_plan(matrix, args.override)
        allowed_stages = {stage.strip() for stage in str(args.stages).split(",") if stage.strip()}
        if allowed_stages:
            plan_rows = [row for row in plan_rows if row["stage"] in allowed_stages]
        write_run_plan(output_dir / "run_plan.csv", plan_rows)
        print(f"[OK] run plan written: {output_dir / 'run_plan.csv'}")
        if args.execute:
            try:
                return execute_plan(plan_rows, output_dir / "execution_summary.csv")
            except ValueError as exc:
                print(f"[FAIL] {exc}", file=sys.stderr)
                return 2

    if args.from_runs is not None:
        run_dirs = [Path(path) for path in args.from_runs]
        missing = [path for path in run_dirs if not path.exists()]
        if missing:
            for path in missing:
                print(f"[FAIL] run_dir does not exist: {path}", file=sys.stderr)
            return 2
        out = build_effect_report(matrix, run_dirs, output_dir)
        print(f"[OK] benchmark-effect report written: {out}")
        return 0

    if not args.dry_run and not args.execute:
        parser.error("choose --dry-run, --execute, or --from-runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
