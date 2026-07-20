#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class DatasetSplitSummary:
    dataset_id: int
    dataset_name: str
    domains: List[int]
    train_domains: List[int]
    test_domains: List[int]
    train_count: int
    test_count: int


def utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def run_command(command: List[str], cwd: Path, env: Dict[str, str]) -> None:
    print("[dataset-bridge] running:")
    print("  ", " ".join(command))
    proc = subprocess.run(command, cwd=str(cwd), env=env, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def read_csv_row(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        row = next(reader, {})
    result: Dict[str, object] = {}
    for key, value in row.items():
        if value is None or value == "":
            result[key] = value
            continue
        try:
            result[key] = float(value)
            if result[key].is_integer():
                result[key] = int(result[key])
        except Exception:
            result[key] = value
    return result


def find_latest_run_dir(root: Path) -> Optional[Path]:
    candidates = [path for path in root.rglob("iter_0") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_first(path_root: Path, suffix: str) -> Optional[Path]:
    matches = sorted(path_root.rglob(suffix))
    return matches[0] if matches else None


def build_split_summary(metadata: pd.DataFrame, dataset_id: int, dataset_name: str, target_domain_num: int) -> DatasetSplitSummary:
    from src.data_factory.ID.Get_id import Get_CDDG_ids

    data = SimpleNamespace(df=metadata)
    task = SimpleNamespace(target_system_id=[dataset_id], target_domain_num=target_domain_num)
    train_ids, test_ids = Get_CDDG_ids(data, task)
    domains = sorted(int(d) for d in metadata.loc[metadata["Dataset_id"] == dataset_id, "Domain_id"].dropna().unique().tolist())
    train_domains = domains[:-1] if len(domains) > 1 else domains
    test_domains = domains[-1:] if len(domains) > 1 else []
    return DatasetSplitSummary(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        domains=domains,
        train_domains=train_domains,
        test_domains=test_domains,
        train_count=len(train_ids),
        test_count=len(test_ids),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal honest CWRU/XJTU dataset bridge evidence pack for MOE")
    parser.add_argument("--run-id", default=utc_stamp(), help="Reuse an external run id so project_runner can collect it")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command and exit")
    parser.add_argument(
        "--config",
        default="configs/demo/02_cross_system/multi_system_cddg.yaml",
        help="Parent-repo config to use",
    )
    args = parser.parse_args()

    paper_root = Path(__file__).resolve().parent.parent
    repo_root = paper_root.parents[2]
    sys.path.insert(0, str(repo_root))
    metadata_path = Path("/home/user/data/PHMbenchdata/PHM-Vibench/metadata.xlsx")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    metadata = pd.read_excel(metadata_path)
    dataset_specs = [
        (1, "CWRU"),
        (2, "XJTU"),
    ]
    split_summaries = [build_split_summary(metadata, dataset_id, name, target_domain_num=1) for dataset_id, name in dataset_specs]
    for split in split_summaries:
        if split.train_count <= 0 or split.test_count <= 0:
            raise RuntimeError(f"Dataset {split.dataset_name} has invalid split: {split}")

    bridge_root = paper_root / "results" / "autoresearch" / args.run_id / "dataset_bridge"
    run_output_root = bridge_root / "parent_cddg"
    bridge_root.mkdir(parents=True, exist_ok=True)
    run_output_root.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "main.py",
        "--config",
        args.config,
        "--override",
        "trainer.num_epochs=1",
        "--override",
        "trainer.device=cpu",
        "--override",
        "model.device=cpu",
        "--override",
        "data.num_workers=0",
        "--override",
        "data.batch_size=64",
        "--override",
        "task.batch_size=32",
        "--override",
        "environment.output_dir=" + str(run_output_root),
        "--override",
        "task.target_system_id=[1,2]",
        "--override",
        "task.target_domain_num=1",
    ]

    if args.dry_run:
        print(json.dumps(
            {
                "run_id": args.run_id,
                "paper_root": str(paper_root),
                "repo_root": str(repo_root),
                "command": command,
                "splits": [asdict(split) for split in split_summaries],
                "run_output_root": str(run_output_root),
            },
            indent=2,
            ensure_ascii=False,
        ))
        return 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    run_command(command, cwd=repo_root, env=env)

    run_dir = find_latest_run_dir(run_output_root)
    if run_dir is None:
        raise RuntimeError(f"Could not locate iter_0 under {run_output_root}")

    manifest_path = find_first(run_dir, "artifacts/manifest.json")
    test_result_path = find_first(run_dir, "test_result_0.csv")
    all_results_path = find_first(run_dir, "all_results.csv")
    config_snapshot_path = find_first(run_dir, "config_snapshot.yaml")

    metrics: Dict[str, object] = {}
    if all_results_path and all_results_path.exists():
        metrics = read_csv_row(all_results_path)
    elif test_result_path and test_result_path.exists():
        metrics = read_csv_row(test_result_path)

    summary = {
        "project_id": "MOE_explainable",
        "run_id": args.run_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "command": command,
        "datasets": [asdict(split) for split in split_summaries],
        "run_output_root": str(run_output_root),
        "run_dir": str(run_dir),
        "artifacts": {
            "manifest": str(manifest_path) if manifest_path else None,
            "test_result": str(test_result_path) if test_result_path else None,
            "all_results": str(all_results_path) if all_results_path else None,
            "config_snapshot": str(config_snapshot_path) if config_snapshot_path else None,
        },
        "bound": bool(manifest_path and (test_result_path or all_results_path)),
        "metrics": metrics,
        "notes": "Honest CWRU+XJTU evidence via parent CDDG entrypoint with target_system_id=[1,2] and target_domain_num=1.",
    }

    summary_path = bridge_root / "dataset_bridge_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (bridge_root / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    run_meta = {
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "datasets": summary["datasets"],
        "command": command,
        "collector_summary": str(summary_path),
    }
    (bridge_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
