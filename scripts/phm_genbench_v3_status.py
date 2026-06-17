"""Summarize PHM-GenBench v0.3 real-run stage completion.

This is a filesystem audit helper for long-running paper matrix jobs. It does
not infer paper readiness; it only records which configured rows have completed
stage-ledger evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from scripts.generative_benchmark_effect import load_matrix


STAGES = ("train", "sample", "eval", "paperpack")


def _ledger_stages(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    stages = data.get("stages", {})
    return stages if isinstance(stages, dict) else {}


def _repair_ledger_metadata(
    path: Path,
    *,
    benchmark_id: str,
    dataset: Any,
    method: Any,
    seed: int,
    run_root: Path,
    status: str,
    completed: list[str],
) -> None:
    if not path.exists():
        return
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(ledger, dict):
        return
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ledger.setdefault("schema_version", "0.3.0")
    ledger.setdefault("stages", {})
    ledger.setdefault("created_at", now)
    ledger.update(
        {
            "benchmark_id": benchmark_id,
            "dataset": str(dataset.dataset),
            "dataset_id": str(dataset.dataset_id),
            "dataset_name": str(dataset.name),
            "method": str(method.method),
            "method_label": str(method.label),
            "seed": str(seed),
            "current_stage": completed[-1] if completed else "",
            "config_path": str(method.train_config),
            "output_dir": str(run_root),
            "status": status,
            "updated_at": now,
        }
    )
    path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")


def _partial_files(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    patterns = (
        "metrics.csv",
        "normalization_params.json",
        "*.ckpt",
        "samples.pt",
        "generative_eval_metrics.csv",
    )
    files: list[Path] = []
    for pattern in patterns:
        files.extend(run_root.rglob(pattern))
    return files


def _latest_metrics_line(run_root: Path) -> tuple[str, str, str, str]:
    metrics_files = list(run_root.rglob("metrics.csv"))
    if not metrics_files:
        return "", "", "", ""
    latest = max(metrics_files, key=lambda path: path.stat().st_mtime)
    lines = latest.read_text(encoding="utf-8").splitlines()
    if not lines:
        return str(latest), "", "", ""
    line = lines[-1]
    parts = [part.strip() for part in line.split(",")]
    epoch = parts[0] if parts else ""
    step = parts[1] if len(parts) > 1 else ""
    return str(latest), line, epoch, step


def build_rows(
    matrix_path: Path,
    output_dir: Path,
    *,
    active_window_sec: float,
    repair_ledger_metadata: bool = False,
) -> list[dict[str, str]]:
    matrix = load_matrix(matrix_path)
    rows: list[dict[str, str]] = []
    now = time.time()
    for dataset in matrix.datasets:
        for method in matrix.methods:
            for seed in matrix.seeds:
                run_root = (
                    output_dir
                    / "runs"
                    / dataset.dataset
                    / method.method
                    / f"seed_{seed}"
                )
                ledger_path = run_root / "stage_ledger.json"
                stages = _ledger_stages(ledger_path)
                completed = [stage for stage in STAGES if stage in stages]
                latest_partial_path = ""
                latest_metric_path, latest_metric_line, latest_epoch, latest_step = (
                    _latest_metrics_line(run_root)
                )
                if completed == list(STAGES):
                    status = "COMPLETE_CHAIN"
                    reason = "all stages complete"
                elif completed:
                    status = "PARTIAL_STAGE_LEDGER"
                    reason = "completed stages: " + ";".join(completed)
                else:
                    partial_files = _partial_files(run_root)
                    if partial_files:
                        latest = max(partial_files, key=lambda path: path.stat().st_mtime)
                        latest_partial_path = str(latest)
                        age_sec = max(0.0, now - latest.stat().st_mtime)
                        if age_sec <= active_window_sec:
                            status = "IN_PROGRESS_NO_LEDGER"
                            reason = (
                                "partial files are being updated; latest="
                                f"{latest}; age_sec={age_sec:.1f}"
                            )
                        else:
                            status = "PARTIAL_INTERRUPTED_NO_LEDGER"
                            reason = (
                                "partial files exist without completed stage ledger; "
                                f"latest={latest}; age_sec={age_sec:.1f}"
                            )
                    else:
                        status = "PENDING"
                        reason = "not started"
                if repair_ledger_metadata:
                    _repair_ledger_metadata(
                        ledger_path,
                        benchmark_id=matrix.benchmark_id,
                        dataset=dataset,
                        method=method,
                        seed=seed,
                        run_root=run_root,
                        status=status,
                        completed=completed,
                    )
                rows.append(
                    {
                        "dataset": dataset.dataset,
                        "method": method.method,
                        "seed": str(seed),
                        "status": status,
                        "completed_stages": ";".join(completed),
                        "stage_ledger_path": str(ledger_path)
                        if ledger_path.exists()
                        else "",
                        "latest_partial_path": latest_partial_path,
                        "latest_metric_path": latest_metric_path,
                        "latest_metric_line": latest_metric_line,
                        "latest_epoch": latest_epoch,
                        "latest_step": latest_step,
                        "reason": reason,
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "dataset",
        "method",
        "seed",
        "status",
        "completed_stages",
        "stage_ledger_path",
        "latest_partial_path",
        "latest_metric_path",
        "latest_metric_line",
        "latest_epoch",
        "latest_step",
        "reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a PHM-GenBench v0.3 real-run status ledger."
    )
    parser.add_argument(
        "--matrix",
        default="configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml",
        help="Six-dataset benchmark matrix YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10",
        help="Real-run output directory containing runs/.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="CSV path to write.",
    )
    parser.add_argument(
        "--active-window-sec",
        type=float,
        default=1800.0,
        help=(
            "Treat partial rows with files updated within this many seconds as "
            "IN_PROGRESS_NO_LEDGER."
        ),
    )
    parser.add_argument(
        "--repair-ledger-metadata",
        action="store_true",
        help=(
            "Augment existing stage_ledger.json files with v0.3 top-level "
            "benchmark/dataset/method/seed metadata without adding stages."
        ),
    )
    args = parser.parse_args(argv)

    rows = build_rows(
        Path(args.matrix),
        Path(args.output_dir),
        active_window_sec=float(args.active_window_sec),
        repair_ledger_metadata=bool(args.repair_ledger_metadata),
    )
    write_csv(Path(args.out), rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(
        "[OK] status ledger written: "
        f"{args.out} "
        + " ".join(f"{key}={counts[key]}" for key in sorted(counts))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
