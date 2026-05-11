"""Create a minimal generative paperpack from a PHM-GenBench run directory."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


QUALITY_PREFIXES = (
    "temporal_",
    "spectral_",
    "distribution_",
    "diversity_",
)
UTILITY_PREFIXES = ("tstr_", "trts_", "utility_")
EFFICIENCY_KEYS = {
    "parameter_count",
    "sampling_nfe",
    "sampling_wall_clock_sec",
    "metric_compute_time_sec",
    "samples_per_second",
    "peak_memory_bytes",
}
LEAKAGE_PREFIXES = ("leakage_",)
STATUS_SUFFIXES = ("_status", "_reason", "_status_code")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_metric_rows(run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(run_dir.rglob("generative_eval_metrics.csv")):
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["source_path"] = str(path)
                rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _is_value_metric(key: str) -> bool:
    return key != "source_path" and not key.endswith(STATUS_SUFFIXES)


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


def _selected_metric(key: str, prefixes: tuple[str, ...], keys: set[str] | None = None) -> bool:
    if not _is_value_metric(key):
        return False
    if keys is not None:
        return key in keys
    return key.startswith(prefixes)


def _metric_records(
    rows: list[dict[str, str]],
    prefixes: tuple[str, ...],
    keys: set[str] | None = None,
    *,
    category: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        for key, value in row.items():
            if not _selected_metric(key, prefixes, keys):
                continue
            numeric = _to_float(value)
            status = row.get(f"{key}_status", "")
            if not status:
                status = "ok" if numeric is not None and math.isfinite(numeric) else "not_computable"
            records.append(
                {
                    "category": category,
                    "metric": key,
                    "value": value,
                    "numeric_value": numeric,
                    "status": status,
                    "reason": row.get(f"{key}_reason", ""),
                    "source_path": row.get("source_path", ""),
                    "ablation_factor": row.get("ablation_factor", ""),
                    "ablation_level": row.get("ablation_level", ""),
                }
            )
    return records


def _metric_table(rows: list[dict[str, str]], prefixes: tuple[str, ...], keys: set[str] | None = None):
    out = []
    for row in rows:
        for key, value in row.items():
            if _selected_metric(key, prefixes, keys):
                out.append({"metric": key, "value": value, "source_path": row.get("source_path", "")})
    return out


def _aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["metric"]), []).append(record)
    out = []
    for metric, metric_records in sorted(grouped.items()):
        values = [
            float(record["numeric_value"])
            for record in metric_records
            if record.get("numeric_value") is not None
            and math.isfinite(float(record["numeric_value"]))
            and str(record.get("status", "ok")) == "ok"
        ]
        missing = [
            record
            for record in metric_records
            if record.get("numeric_value") is None
            or not math.isfinite(float(record["numeric_value"]))
            or str(record.get("status", "ok")) == "not_computable"
        ]
        reasons = sorted({str(record.get("reason", "")).strip() for record in missing if record.get("reason")})
        source_paths = sorted({str(record.get("source_path", "")) for record in metric_records if record.get("source_path")})
        out.append(
            {
                "metric": metric,
                "mean": statistics.mean(values) if values else "",
                "std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else ""),
                "n": len(values),
                "missing_count": len(missing),
                "missing_reasons": " | ".join(reasons),
                "source_paths": ";".join(source_paths),
            }
        )
    return out


def _missing_metric_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing = []
    for record in records:
        numeric = record.get("numeric_value")
        not_finite = numeric is None or not math.isfinite(float(numeric))
        if not_finite or str(record.get("status", "ok")) == "not_computable":
            missing.append(
                {
                    "category": record.get("category", ""),
                    "metric": record.get("metric", ""),
                    "status": record.get("status", ""),
                    "reason": record.get("reason", ""),
                    "value": record.get("value", ""),
                    "source_path": record.get("source_path", ""),
                }
            )
    return missing


def _write_missing_metrics(path: Path, missing_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not missing_rows:
        path.write_text("# Missing Metrics\n\nNo missing metrics were reported.\n", encoding="utf-8")
        return
    grouped: dict[str, list[str]] = {}
    for row in missing_rows:
        reason = str(row.get("reason", "")).strip() or "no reason recorded"
        grouped.setdefault(str(row.get("metric", "")), []).append(reason)
    lines = ["# Missing Metrics", ""]
    for metric, reasons in sorted(grouped.items()):
        unique_reasons = sorted(set(reasons))
        lines.append(f"- `{metric}`: {len(reasons)} missing; reasons: {' | '.join(unique_reasons)}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _manifest_completeness_rows(manifests: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in manifests:
        manifest = _read_json(path)
        validity = manifest.get("validity", {})
        evidence = validity.get("evidence", {})
        normalization = manifest.get("normalization", {})
        rows.append(
            {
                "manifest_path": str(path),
                "synthetic_dataset_id": manifest.get("synthetic_dataset_id", ""),
                "validity_status": validity.get("status", ""),
                "benchmark_valid": validity.get("benchmark_valid", ""),
                "missing_evidence": ";".join(validity.get("missing_evidence", [])),
                "normalization_params_recorded": normalization.get("params_recorded", ""),
                "protocol_hash": evidence.get("protocol_hash", ""),
                "config_hash": evidence.get("config_hash", ""),
                "dependency_lock_hash": evidence.get("dependency_lock_hash", ""),
                "leakage_checks": evidence.get("leakage_checks", ""),
                "condition_sampling_policy": evidence.get("condition_sampling_policy", ""),
                "condition_counts": evidence.get("condition_counts", ""),
                "metric_status_reason_recorded": evidence.get("metric_status_reason_recorded", ""),
            }
        )
    return rows


def _run_index_rows(manifests: list[Path], metric_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in manifests:
        manifest = _read_json(path)
        rows.append(
            {
                "source_type": "manifest",
                "source_path": str(path),
                "synthetic_dataset_id": manifest.get("synthetic_dataset_id", ""),
                "seed": manifest.get("sampling", {}).get("seed", ""),
                "config_path": manifest.get("config", {}).get("config_path", ""),
                "validity_status": manifest.get("validity", {}).get("status", ""),
                "benchmark_valid": manifest.get("validity", {}).get("benchmark_valid", ""),
                "metric_rows": "",
                "utility_protocol_id": "",
                "utility_source_split": "",
                "utility_reference_split": "",
            }
        )
    metric_paths = sorted({row.get("source_path", "") for row in metric_rows if row.get("source_path")})
    for path in metric_paths:
        path_rows = [row for row in metric_rows if row.get("source_path") == path]
        utility_protocols = sorted({row.get("utility_protocol_id", "") for row in path_rows if row.get("utility_protocol_id")})
        utility_sources = sorted({row.get("utility_source_split", "") for row in path_rows if row.get("utility_source_split")})
        utility_references = sorted({row.get("utility_reference_split", "") for row in path_rows if row.get("utility_reference_split")})
        rows.append(
            {
                "source_type": "metrics",
                "source_path": path,
                "synthetic_dataset_id": "",
                "seed": "",
                "config_path": "",
                "validity_status": "",
                "benchmark_valid": "",
                "metric_rows": len(path_rows),
                "utility_protocol_id": ";".join(utility_protocols),
                "utility_source_split": ";".join(utility_sources),
                "utility_reference_split": ";".join(utility_references),
            }
        )
    return rows


def _ablation_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        factor = str(record.get("ablation_factor", "")).strip()
        level = str(record.get("ablation_level", "")).strip()
        if not factor and not level:
            continue
        groups.setdefault((factor, level, str(record["metric"])), []).append(record)
    out = []
    for (factor, level, metric), group_records in sorted(groups.items()):
        aggregate = _aggregate_records(group_records)[0]
        out.append(
            {
                "factor": factor,
                "level": level,
                "metric": metric,
                "mean": aggregate["mean"],
                "std": aggregate["std"],
                "n": aggregate["n"],
                "source_paths": aggregate["source_paths"],
            }
        )
    return out


def _write_reproducibility(path: Path, manifest: dict[str, Any] | None, run_dir: Path) -> None:
    if manifest is None:
        text = (
            "# Reproducibility Statement\n\n"
            f"Run directory: `{run_dir}`\n\n"
            "No synthetic data manifest was found in this run directory.\n"
        )
    else:
        config = manifest.get("config", {})
        protocol = manifest.get("protocol", {})
        environment = manifest.get("environment", {})
        validity = manifest.get("validity", {})
        sampling = manifest.get("sampling", {})
        text = "\n".join(
            [
                "# Reproducibility Statement",
                "",
                f"Run directory: `{run_dir}`",
                f"Synthetic dataset: `{manifest.get('synthetic_dataset_id', '')}`",
                f"Validity: `{validity.get('status', '')}`",
                f"Benchmark valid: `{validity.get('benchmark_valid', '')}`",
                "",
                "## Config",
                f"- Path: `{config.get('config_path', '')}`",
                f"- SHA256: `{config.get('config_hash', '')}`",
                "",
                "## Protocol",
                f"- Path: `{protocol.get('protocol_path', '')}`",
                f"- SHA256: `{protocol.get('protocol_hash', '')}`",
                "",
                "## Environment",
                f"- Python: `{environment.get('python', '')}`",
                f"- Torch: `{environment.get('torch', '')}`",
                f"- Dependency lock hash: `{environment.get('dependency_lock_hash', '')}`",
                "",
                "## Sampling",
                f"- Sampler: `{sampling.get('sampler_id', '')}`",
                f"- Steps/NFE: `{sampling.get('num_steps', '')}`",
                f"- Seed: `{sampling.get('seed', '')}`",
                f"- Shape: `{sampling.get('shape', '')}`",
                "",
            ]
        )
    path.write_text(text, encoding="utf-8")


def build_paperpack(run_dir: Path) -> Path:
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    paperpack = run_dir / "paperpack"
    tables = paperpack / "tables"
    figure_sources = paperpack / "figure_sources"
    appendix = paperpack / "appendix"
    tables.mkdir(parents=True, exist_ok=True)
    figure_sources.mkdir(parents=True, exist_ok=True)
    appendix.mkdir(parents=True, exist_ok=True)

    manifests = sorted(run_dir.rglob("synthetic_data_manifest.json"))
    manifest = _read_json(manifests[-1]) if manifests else None
    metric_rows = _read_metric_rows(run_dir)
    quality_records = _metric_records(metric_rows, QUALITY_PREFIXES, category="quality")
    utility_records = _metric_records(metric_rows, UTILITY_PREFIXES, category="utility")
    efficiency_records = _metric_records(metric_rows, (), EFFICIENCY_KEYS, category="efficiency")
    leakage_records = _metric_records(metric_rows, LEAKAGE_PREFIXES, category="leakage")
    all_records = quality_records + utility_records + efficiency_records + leakage_records

    _write_reproducibility(paperpack / "reproducibility_statement.md", manifest, run_dir)
    index = {
        "run_dir": str(run_dir),
        "synthetic_manifest_paths": [str(path) for path in manifests],
        "metric_rows": len(metric_rows),
    }
    (figure_sources / "manifest_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metric_fields = ["metric", "value", "source_path"]
    _write_csv(tables / "table_quality.csv", _metric_table(metric_rows, QUALITY_PREFIXES), metric_fields)
    _write_csv(tables / "table_utility.csv", _metric_table(metric_rows, UTILITY_PREFIXES), metric_fields)
    _write_csv(tables / "table_efficiency.csv", _metric_table(metric_rows, (), EFFICIENCY_KEYS), metric_fields)
    _write_csv(tables / "table_leakage.csv", _metric_table(metric_rows, LEAKAGE_PREFIXES), metric_fields)
    aggregate_fields = [
        "metric",
        "mean",
        "std",
        "n",
        "missing_count",
        "missing_reasons",
        "source_paths",
    ]
    _write_csv(
        tables / "table_quality_mean_std.csv",
        _aggregate_records(quality_records),
        aggregate_fields,
    )
    _write_csv(
        tables / "table_utility_mean_std.csv",
        _aggregate_records(utility_records),
        aggregate_fields,
    )
    _write_csv(
        tables / "table_efficiency_mean_std.csv",
        _aggregate_records(efficiency_records),
        aggregate_fields,
    )
    _write_csv(
        tables / "table_ablation.csv",
        _ablation_rows(all_records),
        ["factor", "level", "metric", "mean", "std", "n", "source_paths"],
    )
    _write_csv(
        appendix / "run_index.csv",
        _run_index_rows(manifests, metric_rows),
        [
            "source_type",
            "source_path",
            "synthetic_dataset_id",
            "seed",
            "config_path",
            "validity_status",
            "benchmark_valid",
            "metric_rows",
            "utility_protocol_id",
            "utility_source_split",
            "utility_reference_split",
        ],
    )
    _write_csv(
        appendix / "manifest_completeness.csv",
        _manifest_completeness_rows(manifests),
        [
            "manifest_path",
            "synthetic_dataset_id",
            "validity_status",
            "benchmark_valid",
            "missing_evidence",
            "normalization_params_recorded",
            "protocol_hash",
            "config_hash",
            "dependency_lock_hash",
            "leakage_checks",
            "condition_sampling_policy",
            "condition_counts",
            "metric_status_reason_recorded",
        ],
    )
    missing_rows = _missing_metric_rows(all_records)
    _write_csv(
        appendix / "missing_metrics.csv",
        missing_rows,
        ["category", "metric", "status", "reason", "value", "source_path"],
    )
    _write_missing_metrics(appendix / "missing_metrics.md", missing_rows)
    figure_fields = ["category", "metric", "value", "status", "reason", "source_path"]
    _write_csv(
        figure_sources / "spectra_overlay.csv",
        [record for record in quality_records if str(record["metric"]).startswith("spectral_")],
        figure_fields,
    )
    _write_csv(
        figure_sources / "temporal_overlay.csv",
        [record for record in quality_records if str(record["metric"]).startswith("temporal_")],
        figure_fields,
    )
    barplot_rows = []
    for category, records in [
        ("quality", quality_records),
        ("utility", utility_records),
        ("efficiency", efficiency_records),
        ("leakage", leakage_records),
    ]:
        for row in _aggregate_records(records):
            barplot_rows.append({"category": category, **row})
    _write_csv(
        figure_sources / "metric_barplot.csv",
        barplot_rows,
        ["category", *aggregate_fields],
    )
    return paperpack


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a minimal generative paperpack.")
    parser.add_argument("--run_dir", required=True, help="Run directory containing manifests/metrics.")
    args = parser.parse_args()
    out = build_paperpack(Path(args.run_dir))
    print(f"[OK] paperpack written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
