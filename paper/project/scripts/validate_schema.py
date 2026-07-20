#!/usr/bin/env python3
"""
Validate Paper2 schema v1 for run outputs.

Usage:
  python Paper/Explainable_FD_Toolkit/scripts/validate_schema.py --run_dir <RUN_DIR>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


SCHEMA_VERSION = "paper2_schema_v1"


def _require(d: Dict[str, Any], key: str, path: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing key: {path}.{key}")
    return d[key]


def _validate_run_meta(run_meta: Dict[str, Any]) -> None:
    if run_meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("run_meta.yaml: schema_version mismatch")

    paper = _require(run_meta, "paper", "run_meta")
    _require(paper, "paper_id", "run_meta.paper")
    _require(paper, "paper_dir", "run_meta.paper")

    run = _require(run_meta, "run", "run_meta")
    _require(run, "run_id", "run_meta.run")
    _require(run, "seed", "run_meta.run")
    _require(run, "dataset_id", "run_meta.run")
    _require(run, "model_id", "run_meta.run")

    repro = _require(run_meta, "repro", "run_meta")
    _require(repro, "command", "run_meta.repro")
    _require(repro, "config_path", "run_meta.repro")

    git = _require(run_meta, "git", "run_meta")
    _require(git, "commit", "run_meta.git")
    _require(git, "dirty", "run_meta.git")

    env = _require(run_meta, "env", "run_meta")
    _require(env, "python", "run_meta.env")
    _require(env, "torch", "run_meta.env")
    _require(env, "device", "run_meta.env")

    outputs = _require(run_meta, "outputs", "run_meta")
    _require(outputs, "run_dir", "run_meta.outputs")
    _require(outputs, "metrics_path", "run_meta.outputs")


def _validate_metrics(metrics: Dict[str, Any]) -> None:
    if metrics.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("metrics.json: schema_version mismatch")

    _require(metrics, "paper_id", "metrics")
    _require(metrics, "dataset_id", "metrics")
    _require(metrics, "model_id", "metrics")
    _require(metrics, "seed", "metrics")
    _require(metrics, "task", "metrics")

    split_metrics = _require(metrics, "split_metrics", "metrics")
    test = split_metrics.get("test")
    if not isinstance(test, dict):
        raise ValueError("metrics.json: split_metrics.test must be a dict")
    if "accuracy" not in test:
        raise KeyError("metrics.json: split_metrics.test.accuracy required (minimum)")

    _require(metrics, "explainability", "metrics")
    _require(metrics, "artifacts", "metrics")


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to validate run_meta.yaml") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_run_dir(run_dir: Path) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    run_meta_path = run_dir / "run_meta.yaml"
    metrics_path = run_dir / "metrics.json"

    if not run_meta_path.exists():
        errors.append(f"Missing file: {run_meta_path}")
        return False, errors
    if not metrics_path.exists():
        errors.append(f"Missing file: {metrics_path}")
        return False, errors

    try:
        run_meta = _read_yaml(run_meta_path)
        _validate_run_meta(run_meta)
    except Exception as exc:
        errors.append(str(exc))

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        _validate_metrics(metrics)
    except Exception as exc:
        errors.append(str(exc))

    return len(errors) == 0, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=str)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ok, errors = validate_run_dir(run_dir)
    if ok:
        print(f"[OK] schema valid: {run_dir}")
        raise SystemExit(0)
    print(f"[FAIL] schema invalid: {run_dir}")
    for e in errors:
        print(f"- {e}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()

