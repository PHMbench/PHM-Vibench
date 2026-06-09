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
PLAN_STAGE_ORDER = ("train", "sample", "eval", "paperpack")
VALID_PLAN_STAGES = set(PLAN_STAGE_ORDER)


@dataclass(frozen=True)
class BenchmarkMethod:
    method: str
    label: str
    train_config: str
    condition_sampling_policy: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkDataset:
    dataset: str
    dataset_id: str
    name: str
    data_check: dict[str, Any]
    overrides: dict[str, Any]
    protocol: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkResource:
    gpu_ids: list[str]
    max_parallel_runs: int
    require_cuda: bool


@dataclass(frozen=True)
class BenchmarkMatrix:
    path: Path
    benchmark_id: str
    dataset: str
    datasets: list[BenchmarkDataset]
    output_dir: Path
    baseline_method: str
    min_datasets: int
    seeds: list[int]
    python: str
    resource: BenchmarkResource
    data_check: dict[str, Any]
    overrides: dict[str, Any]
    methods: list[BenchmarkMethod]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"matrix must be a YAML mapping: {path}")
    return data


def _load_datasets(
    data: dict[str, Any], benchmark: dict[str, Any]
) -> list[BenchmarkDataset]:
    raw_datasets = data.get("datasets") or benchmark.get("datasets")
    matrix_data_check = dict(benchmark.get("data_check") or {})
    matrix_overrides = dict(benchmark.get("overrides") or {})
    if raw_datasets is None:
        dataset = str(benchmark.get("dataset", "unknown_dataset"))
        return [
            BenchmarkDataset(
                dataset=dataset,
                dataset_id=str(benchmark.get("dataset_id", "")),
                name=str(benchmark.get("dataset_name", dataset)),
                data_check=matrix_data_check,
                overrides={},
                protocol=dict(benchmark.get("protocol") or {}),
            )
        ]
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError("benchmark.datasets must be a non-empty list when provided")

    datasets: list[BenchmarkDataset] = []
    for item in raw_datasets:
        if not isinstance(item, dict):
            raise ValueError("each benchmark dataset must be a mapping")
        dataset_id = str(item.get("dataset_id", ""))
        name = str(item.get("name") or item.get("dataset") or f"dataset_{dataset_id}")
        dataset = str(item.get("dataset") or name)
        data_check = dict(matrix_data_check)
        data_check.update(dict(item.get("data_check") or {}))
        overrides = dict(item.get("overrides") or {})
        if "data.data_dir" not in overrides and "data.data_dir" in matrix_overrides:
            overrides["data.data_dir"] = matrix_overrides["data.data_dir"]
        if (
            "data.metadata_file" not in overrides
            and "data.metadata_file" in matrix_overrides
        ):
            overrides["data.metadata_file"] = matrix_overrides["data.metadata_file"]
        datasets.append(
            BenchmarkDataset(
                dataset=dataset,
                dataset_id=dataset_id,
                name=name,
                data_check=data_check,
                overrides=overrides,
                protocol=dict(item.get("protocol") or {}),
            )
        )
    return datasets


def _load_resource(
    data: dict[str, Any], benchmark: dict[str, Any]
) -> BenchmarkResource:
    resource = data.get("resource") or benchmark.get("resource") or {}
    if not isinstance(resource, dict):
        raise ValueError("benchmark.resource must be a mapping when provided")
    gpu_ids = [str(item) for item in resource.get("gpu_ids", [])]
    max_parallel_runs = int(
        resource.get("max_parallel_runs") or (len(gpu_ids) if gpu_ids else 1)
    )
    if max_parallel_runs < 1:
        raise ValueError("resource.max_parallel_runs must be >= 1")
    return BenchmarkResource(
        gpu_ids=gpu_ids,
        max_parallel_runs=max_parallel_runs,
        require_cuda=bool(resource.get("require_cuda", False)),
    )


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
                condition_sampling_policy=str(
                    item.get("condition_sampling_policy", "")
                ),
                overrides=dict(item.get("overrides") or {}),
            )
        )

    if not methods:
        raise ValueError("matrix must define at least one method")

    seeds = [int(seed) for seed in benchmark.get("seeds", [])]
    if not seeds:
        raise ValueError("benchmark.seeds must not be empty")
    datasets = _load_datasets(data, benchmark)
    resource = _load_resource(data, benchmark)

    return BenchmarkMatrix(
        path=path,
        benchmark_id=str(benchmark.get("id", "phm_generative_benchmark_effect")),
        dataset=datasets[0].dataset,
        datasets=datasets,
        output_dir=Path(
            str(
                benchmark.get(
                    "output_dir", "results/paper/phm_generative/benchmark_effect"
                )
            )
        ),
        baseline_method=str(benchmark.get("baseline_method", methods[0].method)),
        min_datasets=int(benchmark.get("min_datasets", 1)),
        seeds=seeds,
        python=str(benchmark.get("python", sys.executable)),
        resource=resource,
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


def _stage_output_dir(
    matrix: BenchmarkMatrix,
    dataset: BenchmarkDataset,
    method: str,
    seed: int,
    stage: str,
) -> str:
    return str(
        matrix.output_dir / "runs" / dataset.dataset / method / f"seed_{seed}" / stage
    )


def _stage_project(
    matrix: BenchmarkMatrix,
    dataset: BenchmarkDataset,
    method: str,
    seed: int,
    stage: str,
) -> str:
    return f"{matrix.benchmark_id}_{dataset.dataset}_{method}_seed{seed}_{stage}"


def _stage_overrides(
    matrix: BenchmarkMatrix,
    dataset: BenchmarkDataset,
    method: BenchmarkMethod,
    seed: int,
    stage: str,
    extra_overrides: dict[str, Any],
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    overrides.update(matrix.overrides)
    overrides.update(dataset.overrides)
    overrides.update(method.overrides)
    overrides.update(
        {
            "environment.seed": seed,
            "environment.project": _stage_project(
                matrix, dataset, method.method, seed, stage
            ),
            "environment.output_dir": _stage_output_dir(
                matrix, dataset, method.method, seed, stage
            ),
            "task.generative.mode": stage,
            "task.generative.synthetic_dataset_id": f"{dataset.dataset}_{method.method}_seed{seed}",
        }
    )
    if method.condition_sampling_policy:
        overrides[
            "task.generative.condition_sampling_policy"
        ] = method.condition_sampling_policy
    overrides.update(extra_overrides)
    return overrides


def _command_with_resource(
    matrix: BenchmarkMatrix, cmd: list[str], row_index: int
) -> tuple[list[str], str]:
    if not matrix.resource.gpu_ids:
        return cmd, ""
    usable = matrix.resource.gpu_ids[: matrix.resource.max_parallel_runs]
    gpu_id = usable[row_index % len(usable)]
    return ["env", f"CUDA_VISIBLE_DEVICES={gpu_id}", *cmd], gpu_id


def build_run_plan(
    matrix: BenchmarkMatrix, cli_overrides: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manual_overrides = _parse_cli_overrides(cli_overrides)
    row_index = 0
    for dataset in matrix.datasets:
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
                                / dataset.dataset
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
                                / dataset.dataset
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
                    overrides = _stage_overrides(
                        matrix, dataset, method, seed, stage, stage_extra
                    )
                    overrides.update(manual_overrides)
                    cmd = [matrix.python, "main.py", "--config", method.train_config]
                    _append_overrides(cmd, overrides)
                    cmd, gpu_id = _command_with_resource(matrix, cmd, row_index)
                    rows.append(
                        {
                            "benchmark_id": matrix.benchmark_id,
                            "dataset": dataset.dataset,
                            "dataset_id": dataset.dataset_id,
                            "dataset_name": dataset.name,
                            "method": method.method,
                            "method_label": method.label,
                            "seed": seed,
                            "stage": stage,
                            "gpu_id": gpu_id,
                            "config": method.train_config,
                            "command": shlex.join(cmd),
                        }
                    )
                    row_index += 1
                paperpack_run_dir = (
                    matrix.output_dir
                    / "runs"
                    / dataset.dataset
                    / method.method
                    / f"seed_{seed}"
                    / "eval"
                    / "<experiment_name>"
                    / "iter_0"
                )
                paperpack_cmd = [
                    matrix.python,
                    "-m",
                    "scripts.paperpack_generative",
                    "--run_dir",
                    str(paperpack_run_dir),
                ]
                paperpack_cmd, gpu_id = _command_with_resource(
                    matrix, paperpack_cmd, row_index
                )
                rows.append(
                    {
                        "benchmark_id": matrix.benchmark_id,
                        "dataset": dataset.dataset,
                        "dataset_id": dataset.dataset_id,
                        "dataset_name": dataset.name,
                        "method": method.method,
                        "method_label": method.label,
                        "seed": seed,
                        "stage": "paperpack",
                        "gpu_id": gpu_id,
                        "config": "",
                        "command": shlex.join(paperpack_cmd),
                    }
                )
                row_index += 1
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
        [
            "benchmark_id",
            "dataset",
            "dataset_id",
            "dataset_name",
            "method",
            "method_label",
            "seed",
            "stage",
            "gpu_id",
            "config",
            "command",
        ],
    )


def write_blocked_run_status_ledger(
    path: Path, rows: list[dict[str, Any]], *, status: str, reason: str
) -> None:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("benchmark_id", ""),
            row.get("dataset", ""),
            row.get("dataset_name", ""),
            row.get("method", ""),
            row.get("method_label", ""),
            row.get("seed", ""),
        )
        item = groups.setdefault(
            key,
            {
                "benchmark_id": row.get("benchmark_id", ""),
                "dataset": row.get("dataset", ""),
                "dataset_name": row.get("dataset_name", ""),
                "method": row.get("method", ""),
                "method_label": row.get("method_label", ""),
                "seed": row.get("seed", ""),
                "planned_stages": set(),
                "status": status,
                "reason": reason,
            },
        )
        item["planned_stages"].add(row.get("stage", ""))
    ledger_rows: list[dict[str, Any]] = []
    for item in groups.values():
        stages = [
            stage for stage in PLAN_STAGE_ORDER if stage in item["planned_stages"]
        ]
        ledger_rows.append({**item, "planned_stages": ";".join(stages)})
    ledger_rows.sort(
        key=lambda item: (
            str(item["dataset"]),
            str(item["method"]),
            str(item["seed"]),
        )
    )
    _write_csv(
        path,
        ledger_rows,
        [
            "benchmark_id",
            "dataset",
            "dataset_name",
            "method",
            "method_label",
            "seed",
            "planned_stages",
            "status",
            "reason",
        ],
    )


def validate_matrix_inputs(
    matrix: BenchmarkMatrix, *, allow_missing_data: bool
) -> list[str]:
    errors: list[str] = []
    if len(matrix.datasets) < matrix.min_datasets:
        errors.append(
            "matrix defines "
            f"{len(matrix.datasets)} dataset(s), below required minimum {matrix.min_datasets}"
        )
    if matrix.resource.gpu_ids and matrix.resource.max_parallel_runs > len(
        matrix.resource.gpu_ids
    ):
        errors.append(
            "resource.max_parallel_runs cannot exceed number of resource.gpu_ids"
        )
    method_ids = {method.method for method in matrix.methods}
    if matrix.baseline_method not in method_ids:
        errors.append(
            f"baseline_method is not defined in methods: {matrix.baseline_method}"
        )
    for method in matrix.methods:
        if not Path(method.train_config).exists():
            errors.append(
                f"missing train_config for {method.method}: {method.train_config}"
            )
    for dataset in matrix.datasets:
        data_dir = (
            dataset.data_check.get("data_dir")
            or dataset.overrides.get("data.data_dir")
            or matrix.data_check.get("data_dir")
            or matrix.overrides.get("data.data_dir")
        )
        metadata_file = (
            dataset.data_check.get("metadata_file")
            or dataset.overrides.get("data.metadata_file")
            or matrix.data_check.get("metadata_file")
            or matrix.overrides.get("data.metadata_file")
        )
        if data_dir and metadata_file:
            metadata_path = Path(str(data_dir)) / str(metadata_file)
            if not metadata_path.exists() and not allow_missing_data:
                errors.append(
                    f"missing PHM metadata for {dataset.dataset}: {metadata_path}"
                )
        elif not allow_missing_data:
            errors.append(
                f"matrix dataset {dataset.dataset} must define data_check.data_dir "
                "and data_check.metadata_file"
            )
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


def _infer_method(
    path: Path, methods: list[BenchmarkMethod], manifest: dict[str, Any] | None
) -> str:
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


def _infer_dataset(
    path: Path, matrix: BenchmarkMatrix, manifest: dict[str, Any] | None
) -> str:
    payload = manifest or {}
    for key in ("dataset", "dataset_name", "source_dataset", "dataset_id"):
        value = payload.get(key)
        if value:
            value_text = str(value)
            for dataset in matrix.datasets:
                if value_text in {dataset.dataset, dataset.dataset_id, dataset.name}:
                    return dataset.dataset
            return value_text
    candidates = " ".join([str(path), json.dumps(payload, ensure_ascii=False)])
    for dataset in matrix.datasets:
        if dataset.dataset and dataset.dataset in candidates:
            return dataset.dataset
        if dataset.dataset_id and re.search(
            rf"dataset[_-]?{re.escape(dataset.dataset_id)}\b", candidates
        ):
            return dataset.dataset
    return matrix.dataset


def _run_status(manifest: dict[str, Any] | None) -> tuple[str, str]:
    if manifest is None:
        return "exploratory", "missing synthetic_data_manifest.json"
    validity = manifest.get("validity", {})
    status = str(validity.get("status") or "")
    benchmark_valid = validity.get("benchmark_valid")
    if status == "benchmark-valid" or benchmark_valid is True:
        return "benchmark-valid", ""
    missing = validity.get("missing_evidence", [])
    reason = (
        ";".join(str(item) for item in missing)
        if isinstance(missing, list)
        else str(missing)
    )
    return status or "exploratory", reason


def collect_metric_records(
    run_dirs: list[Path], matrix: BenchmarkMatrix
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for root in run_dirs:
        for metric_path in sorted(root.rglob("generative_eval_metrics.csv")):
            manifest_paths = sorted(
                metric_path.parent.rglob("synthetic_data_manifest.json")
            )
            if not manifest_paths:
                manifest_paths = sorted(
                    metric_path.parent.parent.rglob("synthetic_data_manifest.json")
                )
            manifest_path = manifest_paths[-1] if manifest_paths else None
            manifest = _read_json(manifest_path) if manifest_path is not None else None
            method = _infer_method(metric_path, matrix.methods, manifest)
            seed = _infer_seed(metric_path, manifest)
            dataset = _infer_dataset(metric_path, matrix, manifest)
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
                            "dataset": dataset,
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
                            "manifest_path": str(manifest_path)
                            if manifest_path
                            else "",
                            "metric_source_path": str(metric_path),
                        }
                    )
    return records


def aggregate_effects(
    records: list[dict[str, Any]], baseline_method: str
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(
            (str(record["dataset"]), str(record["method"]), str(record["metric"])), []
        ).append(record)

    rows: list[dict[str, Any]] = []
    for (dataset, method, metric), items in sorted(groups.items()):
        values = [
            float(item["value"])
            for item in items
            if item.get("value") is not None and not item.get("missing")
        ]
        missing_items = [item for item in items if item.get("missing")]
        reasons = sorted(
            {
                str(item.get("reason") or item.get("benchmark_status_reason") or "")
                for item in missing_items
            }
        )
        statuses = {str(item.get("benchmark_status", "exploratory")) for item in items}
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "metric": metric,
                "category": items[0]["category"],
                "direction": items[0]["direction"],
                "mean": statistics.mean(values) if values else "",
                "std": statistics.stdev(values)
                if len(values) > 1
                else (0.0 if values else ""),
                "n": len(values),
                "missing_count": len(missing_items),
                "missing_reasons": " | ".join(reason for reason in reasons if reason),
                "baseline_method": baseline_method,
                "baseline_mean": "",
                "delta_vs_baseline": "",
                "relative_delta_vs_baseline": "",
                "rank": "",
                "benchmark_status": "benchmark-valid"
                if statuses == {"benchmark-valid"}
                else "exploratory",
                "manifest_paths": ";".join(
                    sorted(
                        {
                            str(item.get("manifest_path", ""))
                            for item in items
                            if item.get("manifest_path")
                        }
                    )
                ),
                "metric_source_paths": ";".join(
                    sorted(
                        {
                            str(item.get("metric_source_path", ""))
                            for item in items
                            if item.get("metric_source_path")
                        }
                    )
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
            row["relative_delta_vs_baseline"] = (
                delta / abs(baseline_mean) if baseline_mean else ""
            )

    for dataset_metric in sorted({(row["dataset"], row["metric"]) for row in rows}):
        metric_rows = [
            row
            for row in rows
            if (row["dataset"], row["metric"]) == dataset_metric and row["mean"] != ""
        ]
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
            reason = (
                record.get("reason")
                or record.get("benchmark_status_reason")
                or "no reason recorded"
            )
            lines.append(
                f"- `{record['method']}` seed `{record['seed']}` `{record['metric']}`: {reason} "
                f"({record['metric_source_path']})"
            )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(
    path: Path, matrix: BenchmarkMatrix, summary_rows: list[dict[str, Any]]
) -> None:
    lines = [
        "# PHM Generative Benchmark Effect Report",
        "",
        f"Benchmark: `{matrix.benchmark_id}`",
        "Datasets: " + ", ".join(f"`{dataset.dataset}`" for dataset in matrix.datasets),
        f"Baseline method: `{matrix.baseline_method}`",
        "",
        "## Top Utility Metrics",
        "",
    ]
    utility = [
        row
        for row in summary_rows
        if row["category"] == "utility" and row["mean"] != ""
    ]
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
    lines.append(
        "Every summary row must retain metric and manifest source paths in "
        "`benchmark_effect_summary.csv`."
    )
    lines.append(
        "Rows are exploratory unless all contributing manifests are benchmark-valid."
    )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(
    path: Path,
    matrix: BenchmarkMatrix,
    run_dirs: list[Path],
    summary_rows: list[dict[str, Any]],
) -> None:
    statuses = sorted(
        {str(row.get("benchmark_status", "exploratory")) for row in summary_rows}
    )
    configured_datasets = [dataset.dataset for dataset in matrix.datasets]
    configured_dataset_set = set(configured_datasets)
    observed_datasets = sorted(
        {
            str(row.get("dataset", "")).strip()
            for row in summary_rows
            if row.get("dataset")
        }
    )
    observed_dataset_set = set(observed_datasets)
    missing_datasets = [
        dataset
        for dataset in configured_datasets
        if dataset not in observed_dataset_set
    ]
    unexpected_datasets = [
        dataset
        for dataset in observed_datasets
        if dataset not in configured_dataset_set
    ]
    observed_configured_datasets = [
        dataset for dataset in configured_datasets if dataset in observed_dataset_set
    ]
    input_gaps = [
        f"missing configured dataset evidence: {dataset}"
        for dataset in missing_datasets
    ]
    input_gaps.extend(
        f"unexpected observed dataset evidence: {dataset}"
        for dataset in unexpected_datasets
    )
    if len(observed_configured_datasets) < matrix.min_datasets:
        input_gaps.append(
            "observed configured "
            f"{len(observed_configured_datasets)} dataset(s), below required minimum "
            f"{matrix.min_datasets}"
        )
    payload = {
        "benchmark_id": matrix.benchmark_id,
        "datasets": [
            {
                "dataset": dataset.dataset,
                "dataset_id": dataset.dataset_id,
                "name": dataset.name,
                "protocol": dataset.protocol,
            }
            for dataset in matrix.datasets
        ],
        "configured_dataset_count": len(configured_datasets),
        "observed_datasets": observed_datasets,
        "observed_dataset_count": len(observed_datasets),
        "observed_configured_datasets": observed_configured_datasets,
        "observed_configured_dataset_count": len(observed_configured_datasets),
        "missing_datasets": missing_datasets,
        "unexpected_datasets": unexpected_datasets,
        "min_datasets": matrix.min_datasets,
        "min_datasets_met": len(observed_configured_datasets) >= matrix.min_datasets,
        "baseline_method": matrix.baseline_method,
        "matrix_path": str(matrix.path),
        "resource": {
            "gpu_ids": matrix.resource.gpu_ids,
            "max_parallel_runs": matrix.resource.max_parallel_runs,
            "require_cuda": matrix.resource.require_cuda,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dirs": [str(path) for path in run_dirs],
        "summary_rows": len(summary_rows),
        "benchmark_statuses": statuses,
        "input_gaps": input_gaps,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_effect_report(
    matrix: BenchmarkMatrix, run_dirs: list[Path], output_dir: Path
) -> Path:
    records = collect_metric_records(run_dirs, matrix)
    if not records:
        raise ValueError(
            "no generative_eval_metrics.csv records found under run_dirs: "
            + ", ".join(str(path) for path in run_dirs)
        )
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
    write_manifest(
        output_dir / "benchmark_effect_manifest.json", matrix, run_dirs, summary_rows
    )
    return output_dir


def collect_gpu_preflight_results(matrix: BenchmarkMatrix) -> list[dict[str, Any]]:
    """Verify declared CUDA resources without silently falling back to CPU."""

    results: list[dict[str, Any]] = []
    if not matrix.resource.gpu_ids:
        if matrix.resource.require_cuda:
            results.append(
                {
                    "gpu_id": "",
                    "status": "failed",
                    "returncode": "",
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "error": "resource.require_cuda is true but resource.gpu_ids is empty",
                }
            )
        return results
    probe = (
        "import torch; "
        "assert torch.cuda.is_available(), 'torch cuda unavailable'; "
        "assert torch.cuda.device_count() == 1, "
        "f'expected one visible GPU, got {torch.cuda.device_count()}'; "
        "print(torch.cuda.get_device_name(0))"
    )
    for gpu_id in matrix.resource.gpu_ids:
        result = subprocess.run(
            ["env", f"CUDA_VISIBLE_DEVICES={gpu_id}", matrix.python, "-c", probe],
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()[-1:]
            error = (
                f"GPU {gpu_id} failed CUDA preflight: "
                f"{detail[0] if detail else 'unknown error'}"
            )
            results.append(
                {
                    "gpu_id": gpu_id,
                    "status": "failed",
                    "returncode": result.returncode,
                    "stdout_tail": result.stdout[-1000:],
                    "stderr_tail": result.stderr[-1000:],
                    "error": error,
                }
            )
        else:
            results.append(
                {
                    "gpu_id": gpu_id,
                    "status": "passed",
                    "returncode": result.returncode,
                    "stdout_tail": result.stdout[-1000:],
                    "stderr_tail": result.stderr[-1000:],
                    "error": "",
                }
            )
    return results


def _gpu_preflight_errors(results: list[dict[str, Any]]) -> list[str]:
    return [
        str(result["error"])
        for result in results
        if result.get("status") == "failed" and result.get("error")
    ]


def preflight_gpu_resource(matrix: BenchmarkMatrix) -> list[str]:
    return _gpu_preflight_errors(collect_gpu_preflight_results(matrix))


def write_gpu_preflight_report(
    path: Path, matrix: BenchmarkMatrix, results: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_id": matrix.benchmark_id,
        "matrix_path": str(matrix.path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": matrix.python,
        "require_cuda": matrix.resource.require_cuda,
        "gpu_ids": matrix.resource.gpu_ids,
        "max_parallel_runs": matrix.resource.max_parallel_runs,
        "passed": not _gpu_preflight_errors(results),
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _latest_path(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return sorted(paths, key=lambda path: path.stat().st_mtime)[-1]


def _resolve_placeholder_path(raw_path: str) -> str:
    path = Path(raw_path)
    parts = path.parts
    if "<experiment_name>" not in parts:
        return raw_path
    index = parts.index("<experiment_name>")
    base = Path(*parts[:index])
    suffix = parts[index + 1 :]
    if not base.exists():
        raise ValueError(f"placeholder base path does not exist: {base}")

    candidate: Path | None = None
    if "checkpoints" in suffix:
        best = _latest_path(list(base.rglob("best.ckpt")))
        candidate = best or _latest_path(list(base.rglob("*.ckpt")))
    elif suffix and suffix[-1] == "samples.pt":
        candidate = _latest_path(list(base.rglob("samples.pt")))
    elif suffix and suffix[-1] == "iter_0":
        candidate = _latest_path(
            [item for item in base.rglob("iter_0") if item.is_dir()]
        )

    if candidate is None:
        raise ValueError(f"could not resolve placeholder artifact path: {raw_path}")
    return str(candidate)


def _resolve_placeholder_arg(arg: str) -> str:
    if "<experiment_name>" not in arg:
        return arg
    if "=" in arg:
        key, value = arg.split("=", 1)
        return f"{key}={_resolve_placeholder_path(value)}"
    return _resolve_placeholder_path(arg)


def _override_value(cmd: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for index, arg in enumerate(cmd):
        if arg == "--override" and index + 1 < len(cmd):
            value = cmd[index + 1]
            if value.startswith(prefix):
                return value.split("=", 1)[1]
        elif arg.startswith("--override="):
            value = arg.split("=", 1)[1]
            if value.startswith(prefix):
                return value.split("=", 1)[1]
    return None


def _paperpack_run_dir(cmd: list[str]) -> Path | None:
    for index, arg in enumerate(cmd):
        if arg == "--run_dir" and index + 1 < len(cmd):
            return Path(cmd[index + 1])
        if arg.startswith("--run_dir="):
            return Path(arg.split("=", 1)[1])
    return None


def _completed_stage_artifact(stage: str, cmd: list[str]) -> Path | None:
    if stage in {"train", "sample", "eval"}:
        output_dir = _override_value(cmd, "environment.output_dir")
        if output_dir is None:
            return None
        root = Path(output_dir)
        if stage == "train":
            return _latest_path(list(root.rglob("train_result_0.csv")))
        if stage == "sample":
            return _latest_path(list(root.rglob("samples.pt")))
        return _latest_path(list(root.rglob("generative_eval_metrics.csv")))
    if stage == "paperpack":
        run_dir = _paperpack_run_dir(cmd)
        if run_dir is None:
            return None
        artifact = run_dir / "paperpack" / "figure_sources" / "manifest_index.json"
        return artifact if artifact.exists() else None
    return None


def execute_plan(
    rows: list[dict[str, Any]],
    out_csv: Path,
    *,
    skip_existing: bool = False,
    max_runs: int | None = None,
) -> int:
    executed: list[dict[str, Any]] = []
    run_count = 0
    for row in rows:
        try:
            cmd = [
                _resolve_placeholder_arg(arg)
                for arg in shlex.split(str(row["command"]))
            ]
        except ValueError as exc:
            raise ValueError(
                f"failed to resolve artifact path for stage {row.get('stage', '')}: {exc}"
            ) from exc
        stage = str(row.get("stage", ""))
        if skip_existing:
            artifact = _completed_stage_artifact(stage, cmd)
            if artifact is not None:
                executed.append(
                    {
                        **row,
                        "command": shlex.join(cmd),
                        "returncode": 0,
                        "wall_clock_sec": "0.000000",
                        "stdout_tail": f"[SKIP] existing {stage} artifact: {artifact}",
                        "stderr_tail": "",
                    }
                )
                continue
        if max_runs is not None and run_count >= max_runs:
            break
        start = time.perf_counter()
        result = subprocess.run(cmd, text=True, capture_output=True)
        run_count += 1
        executed.append(
            {
                **row,
                "command": shlex.join(cmd),
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
            "dataset_id",
            "dataset_name",
            "method",
            "method_label",
            "seed",
            "stage",
            "gpu_id",
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
    parser = argparse.ArgumentParser(
        description="PHM generative benchmark-effect runner."
    )
    parser.add_argument("--matrix", required=True, help="Benchmark-effect matrix YAML.")
    parser.add_argument(
        "--output-dir", default=None, help="Override benchmark output directory."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write run_plan.csv without executing commands.",
    )
    parser.add_argument(
        "--from-runs",
        nargs="*",
        default=None,
        help="Aggregate existing run directories.",
    )
    parser.add_argument(
        "--execute", action="store_true", help="Execute the generated command plan."
    )
    parser.add_argument(
        "--stages",
        default="train,sample,eval,paperpack",
        help="Comma-separated stages to include in the command plan.",
    )
    parser.add_argument(
        "--allow-missing-data",
        action="store_true",
        help="Allow missing local PHM data in planning mode.",
    )
    parser.add_argument(
        "--preflight-gpu",
        action="store_true",
        help="Verify resource.gpu_ids with CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra key=value override appended to commands.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip stage commands when the expected stage artifact already exists.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Execute at most this many non-skipped commands from the plan.",
    )
    args = parser.parse_args(argv)

    if args.max_runs is not None and args.max_runs < 1:
        print("[FAIL] --max-runs must be >= 1", file=sys.stderr)
        return 2

    selected_modes = [
        name
        for name, selected in (
            ("--dry-run", args.dry_run),
            ("--execute", args.execute),
            ("--from-runs", args.from_runs is not None),
        )
        if selected
    ]
    if len(selected_modes) > 1:
        print(
            "[FAIL] choose exactly one of --dry-run, --execute, or --from-runs; "
            f"got {', '.join(selected_modes)}",
            file=sys.stderr,
        )
        return 2
    if args.from_runs is not None and not args.from_runs:
        print(
            "[FAIL] --from-runs requires at least one run directory",
            file=sys.stderr,
        )
        return 2

    matrix = load_matrix(Path(args.matrix))
    output_dir = Path(args.output_dir) if args.output_dir else matrix.output_dir
    matrix = BenchmarkMatrix(
        path=matrix.path,
        benchmark_id=matrix.benchmark_id,
        dataset=matrix.dataset,
        datasets=matrix.datasets,
        output_dir=output_dir,
        baseline_method=matrix.baseline_method,
        min_datasets=matrix.min_datasets,
        seeds=matrix.seeds,
        python=matrix.python,
        resource=matrix.resource,
        data_check=matrix.data_check,
        overrides=matrix.overrides,
        methods=matrix.methods,
    )

    errors = validate_matrix_inputs(
        matrix, allow_missing_data=bool(args.allow_missing_data)
    )
    if errors:
        for error in errors:
            print(f"[FAIL] {error}", file=sys.stderr)
        return 2

    allowed_stages: set[str] = set()
    if args.dry_run or args.execute:
        allowed_stages = {
            stage.strip() for stage in str(args.stages).split(",") if stage.strip()
        }
        invalid_stages = sorted(allowed_stages - VALID_PLAN_STAGES)
        if invalid_stages:
            print(
                "[FAIL] invalid stage(s): "
                + ", ".join(invalid_stages)
                + "; expected one or more of "
                + ", ".join(sorted(VALID_PLAN_STAGES)),
                file=sys.stderr,
            )
            return 2

    if args.execute and matrix.resource.require_cuda and not args.preflight_gpu:
        print(
            "[FAIL] --execute requires --preflight-gpu when resource.require_cuda is true",
            file=sys.stderr,
        )
        return 2

    if args.execute and matrix.resource.require_cuda and len(allowed_stages) != 1:
        print(
            "[FAIL] --execute with resource.require_cuda requires exactly one "
            "--stages value",
            file=sys.stderr,
        )
        return 2

    if args.preflight_gpu:
        gpu_preflight_results = collect_gpu_preflight_results(matrix)
        write_gpu_preflight_report(
            output_dir / "gpu_preflight_report.json", matrix, gpu_preflight_results
        )
        gpu_errors = _gpu_preflight_errors(gpu_preflight_results)
        if gpu_errors:
            if args.dry_run or args.execute:
                blocked_rows = build_run_plan(matrix, args.override)
                if allowed_stages:
                    blocked_rows = [
                        row for row in blocked_rows if row["stage"] in allowed_stages
                    ]
                if blocked_rows:
                    write_blocked_run_status_ledger(
                        output_dir / "blocked_run_status_ledger.csv",
                        blocked_rows,
                        status="BLOCKED_GPU_PREFLIGHT",
                        reason="; ".join(gpu_errors),
                    )
            for error in gpu_errors:
                print(f"[FAIL] {error}", file=sys.stderr)
            return 2
        print("[OK] CUDA resource preflight passed")

    if args.dry_run or args.execute:
        plan_rows = build_run_plan(matrix, args.override)
        if allowed_stages:
            plan_rows = [row for row in plan_rows if row["stage"] in allowed_stages]
        if not plan_rows:
            print("[FAIL] stage filter produced an empty run plan", file=sys.stderr)
            return 2
        write_run_plan(output_dir / "run_plan.csv", plan_rows)
        print(f"[OK] run plan written: {output_dir / 'run_plan.csv'}")
        if args.execute:
            try:
                return execute_plan(
                    plan_rows,
                    output_dir / "execution_summary.csv",
                    skip_existing=bool(args.skip_existing),
                    max_runs=args.max_runs,
                )
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
        try:
            out = build_effect_report(matrix, run_dirs, output_dir)
        except ValueError as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 2
        print(f"[OK] benchmark-effect report written: {out}")
        return 0

    if not args.dry_run and not args.execute:
        parser.error("choose --dry-run, --execute, or --from-runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
