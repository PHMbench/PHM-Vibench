#!/usr/bin/env python3
"""
Collect Paper2 schema v1 runs into a single CSV table.

Usage:
  python Paper/Explainable_FD_Toolkit/scripts/collect_results_master.py \
    --roots Paper/1D-2D_fusion_explainable/outputs Paper/MOE_explainable/outputs \
    --out Paper/Explainable_FD_Toolkit/results_table_master.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SCHEMA_VERSION = "paper2_schema_v1"


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to parse run_meta.yaml") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _iter_run_dirs(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("run_meta.yaml"):
            yield p.parent


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _safe_get(d: Dict[str, Any], path: str) -> Optional[Any]:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def collect_rows(run_dirs: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        run_meta_path = run_dir / "run_meta.yaml"
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        run_meta = _read_yaml(run_meta_path)
        if run_meta.get("schema_version") != SCHEMA_VERSION:
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("schema_version") != SCHEMA_VERSION:
            continue

        row: Dict[str, Any] = {}
        row.update(
            {
                "paper_id": _safe_get(run_meta, "paper.paper_id"),
                "dataset_id": _safe_get(run_meta, "run.dataset_id"),
                "model_id": _safe_get(run_meta, "run.model_id"),
                "seed": _safe_get(run_meta, "run.seed"),
                "run_id": _safe_get(run_meta, "run.run_id"),
                "run_dir": str(run_dir),
                "git_commit": _safe_get(run_meta, "git.commit"),
                "git_dirty": _safe_get(run_meta, "git.dirty"),
            }
        )

        test_acc = _safe_get(metrics, "split_metrics.test.accuracy")
        row["test_accuracy"] = test_acc
        row["test_f1_macro"] = _safe_get(metrics, "split_metrics.test.f1_macro")

        row["faithfulness_del_k_auc"] = _safe_get(metrics, "explainability.faithfulness.del_k_auc")
        row["stability_spearman_mean"] = _safe_get(metrics, "explainability.stability.spearman_mean")
        row["eff_time_ms_per_sample"] = _safe_get(metrics, "explainability.efficiency.time_ms_per_sample")
        row["sparsity_rules_activated_mean"] = _safe_get(metrics, "explainability.sparsity.rules_activated_mean")

        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    roots = [Path(p) for p in args.roots]
    run_dirs = list(_iter_run_dirs(roots))
    rows = collect_rows(run_dirs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

