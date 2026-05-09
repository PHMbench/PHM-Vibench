"""Create a minimal generative paperpack from a PHM-GenBench run directory."""

from __future__ import annotations

import argparse
import csv
import json
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


def _metric_table(rows: list[dict[str, str]], prefixes: tuple[str, ...], keys: set[str] | None = None):
    out = []
    for row in rows:
        for key, value in row.items():
            if key == "source_path":
                continue
            if keys is not None:
                selected = key in keys
            else:
                selected = key.startswith(prefixes)
            if selected:
                out.append({"metric": key, "value": value, "source_path": row.get("source_path", "")})
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
    tables.mkdir(parents=True, exist_ok=True)
    figure_sources.mkdir(parents=True, exist_ok=True)

    manifests = sorted(run_dir.rglob("synthetic_data_manifest.json"))
    manifest = _read_json(manifests[-1]) if manifests else None
    metric_rows = _read_metric_rows(run_dir)

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
    _write_csv(tables / "table_ablation.csv", [], ["factor", "level", "metric", "value", "source_path"])
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
