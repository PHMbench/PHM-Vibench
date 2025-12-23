from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class CollectResult:
    rows: List[Dict[str, str]]
    metric_keys: List[str]


def _safe_json_load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _flatten_dict(prefix: str, data: Dict[str, Any]) -> Dict[str, str]:
    flat: Dict[str, str] = {}
    for key, value in data.items():
        col = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_dict(col, value))
        else:
            flat[col] = _stringify(value)
    return flat


def _read_optional_json(run_dir: Path, rel_path: str) -> Optional[Dict[str, Any]]:
    if not rel_path:
        return None
    candidate = (run_dir / rel_path).resolve() if not Path(rel_path).is_absolute() else Path(rel_path)
    if not candidate.exists():
        return None
    return _safe_json_load(candidate)


def _discover_manifests(root: Path) -> List[Path]:
    return sorted(root.glob("**/artifacts/manifest.json"))


def _collect_one(manifest_path: Path) -> Tuple[Dict[str, str], List[str]]:
    manifest = _safe_json_load(manifest_path) or {}
    run_dir = manifest_path.parents[1]

    base_row: Dict[str, str] = {
        "manifest_path": str(manifest_path),
        "run_dir": str(run_dir),
        "paper_id": _stringify(manifest.get("paper_id")),
        "preset_version": _stringify(manifest.get("preset_version")),
        "run_id": _stringify(manifest.get("run_id")),
        "stage": _stringify(manifest.get("stage")),
        "timestamp": _stringify(manifest.get("timestamp")),
        "config_snapshot": _stringify(manifest.get("config_snapshot")),
        "metrics_path": _stringify(manifest.get("metrics_path") or manifest.get("metrics")),
        "figures_dir": _stringify(manifest.get("figures_dir")),
        "explain_dir": _stringify(manifest.get("explain_dir")),
        "explain_summary_path": _stringify(manifest.get("explain_summary") or manifest.get("explain_summary_path")),
        "distilled_dir": _stringify(manifest.get("distilled_dir")),
    }

    # data metadata snapshot
    meta_snapshot_path = _stringify(manifest.get("data_metadata_snapshot"))
    meta_snapshot = _read_optional_json(run_dir, meta_snapshot_path) if meta_snapshot_path else None
    if meta_snapshot:
        base_row["meta_source"] = _stringify(meta_snapshot.get("meta_source"))
        base_row["degraded"] = _stringify(meta_snapshot.get("degraded"))
        base_row["missing_keys"] = _stringify(meta_snapshot.get("missing_keys"))
    else:
        base_row["meta_source"] = ""
        base_row["degraded"] = ""
        base_row["missing_keys"] = ""

    # eligibility
    eligibility_path = _stringify(manifest.get("eligibility"))
    eligibility = _read_optional_json(run_dir, eligibility_path) if eligibility_path else None
    if eligibility:
        base_row["explain_ok"] = _stringify(eligibility.get("ok"))
        base_row["explainer_id"] = _stringify(eligibility.get("explainer_id"))
        base_row["explain_reasons"] = _stringify(eligibility.get("reasons"))
    else:
        base_row["explain_ok"] = ""
        base_row["explainer_id"] = ""
        base_row["explain_reasons"] = ""

    # metrics (best effort): flatten a dict if provided inline
    metric_keys: List[str] = []
    metrics_inline = manifest.get("metrics_inline")
    if isinstance(metrics_inline, dict):
        flat_metrics = _flatten_dict("metric", metrics_inline)
        base_row.update(flat_metrics)
        metric_keys = sorted(flat_metrics.keys())

    return base_row, metric_keys


def collect_manifests(root: Path) -> CollectResult:
    manifests = _discover_manifests(root)
    rows: List[Dict[str, str]] = []
    all_metric_keys: set[str] = set()
    for mp in manifests:
        row, metric_keys = _collect_one(mp)
        rows.append(row)
        all_metric_keys.update(metric_keys)
    return CollectResult(rows=rows, metric_keys=sorted(all_metric_keys))


def write_csv(path: Path, rows: List[Dict[str, str]], preferred_cols: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # stable superset of columns
    all_cols = set(preferred_cols)
    for row in rows:
        all_cols.update(row.keys())
    cols = preferred_cols + [c for c in sorted(all_cols) if c not in preferred_cols]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in cols})


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Collect UXFD run manifests into CSV.")
    parser.add_argument("--input", type=str, default="save", help="Root directory to search manifests under.")
    parser.add_argument("--out_dir", type=str, default="reports", help="Output directory for CSV files.")
    parser.add_argument("--runs_csv", type=str, default="uxfd_runs.csv", help="Filename for run-level CSV.")
    args = parser.parse_args(argv)

    root = Path(args.input)
    out_dir = Path(args.out_dir)
    result = collect_manifests(root)

    preferred = [
        "paper_id",
        "preset_version",
        "run_id",
        "stage",
        "timestamp",
        "manifest_path",
        "run_dir",
        "config_snapshot",
        "metrics_path",
        "figures_dir",
        "explain_dir",
        "explain_summary_path",
        "distilled_dir",
        "meta_source",
        "degraded",
        "missing_keys",
        "explain_ok",
        "explainer_id",
        "explain_reasons",
    ] + result.metric_keys

    write_csv(out_dir / args.runs_csv, result.rows, preferred_cols=preferred)
    print(f"[collect] manifests={len(result.rows)} -> {(out_dir / args.runs_csv).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
