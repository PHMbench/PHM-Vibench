#!/usr/bin/env python3
"""Smoke runner for Toolkit evidence-chain ablation surfaces."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONDITIONS: Mapping[str, Mapping[str, Any]] = {
    "schema_off": {
        "label": "schema validation disabled",
        "schema_validation_enabled": False,
        "metric_families": ["faithfulness", "stability", "latency"],
        "manifest_enabled": True,
        "snapshot_enabled": True,
        "comparator_mode": "toolkit",
    },
    "metrics_subset_off": {
        "label": "faithfulness and stability metric family disabled",
        "schema_validation_enabled": True,
        "metric_families": ["latency"],
        "manifest_enabled": True,
        "snapshot_enabled": True,
        "comparator_mode": "toolkit",
    },
    "manifest_off": {
        "label": "standardized manifest disabled",
        "schema_validation_enabled": True,
        "metric_families": ["faithfulness", "stability", "latency"],
        "manifest_enabled": False,
        "snapshot_enabled": True,
        "comparator_mode": "toolkit",
    },
    "snapshot_off": {
        "label": "fixed seed and config snapshot disabled",
        "schema_validation_enabled": True,
        "metric_families": ["faithfulness", "stability", "latency"],
        "manifest_enabled": True,
        "snapshot_enabled": False,
        "comparator_mode": "toolkit",
    },
    "posthoc_only": {
        "label": "post-hoc comparator only",
        "schema_validation_enabled": True,
        "metric_families": ["faithfulness", "stability", "latency"],
        "manifest_enabled": True,
        "snapshot_enabled": True,
        "comparator_mode": "shap_lime_captum_proxy",
    },
}


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _proxy_metrics(condition: str) -> Dict[str, float]:
    config = CONDITIONS[condition]
    faithfulness = 0.74 if "faithfulness" in config["metric_families"] else 0.0
    stability = 0.69 if "stability" in config["metric_families"] else 0.0
    schema_completeness = 1.0 if config["schema_validation_enabled"] else 0.0
    manifest_completeness = 1.0 if config["manifest_enabled"] else 0.0
    snapshot_completeness = 1.0 if config["snapshot_enabled"] else 0.0
    comparator_penalty = 0.12 if config["comparator_mode"] == "shap_lime_captum_proxy" else 0.0
    toolkit_score = (
        0.30 * faithfulness
        + 0.25 * stability
        + 0.20 * schema_completeness
        + 0.15 * manifest_completeness
        + 0.10 * snapshot_completeness
        - comparator_penalty
    )
    return {
        "faithfulness_proxy": faithfulness,
        "stability_proxy": stability,
        "schema_completeness_proxy": schema_completeness,
        "manifest_completeness_proxy": manifest_completeness,
        "snapshot_completeness_proxy": snapshot_completeness,
        "toolkit_score_proxy": round(max(toolkit_score, 0.0), 6),
    }


def _write_condition(condition: str, output_root: Path, seed: int) -> None:
    started_at = datetime.now().isoformat()
    started_perf = time.perf_counter()
    config = CONDITIONS[condition]

    run_root = output_root / condition / f"seed_{seed}"
    inputs_root = run_root / "inputs"
    outputs_root = run_root / "outputs"
    logs_root = run_root / "logs"
    artifacts_root = run_root / "artifacts"
    for directory in (inputs_root, outputs_root, logs_root, artifacts_root):
        directory.mkdir(parents=True, exist_ok=True)

    input_config_path = inputs_root / "ablation_config.json"
    output_summary_path = outputs_root / "ablation_summary.json"
    input_config_path.write_text(
        json.dumps({"condition": condition, **config}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_summary_path.write_text(
        json.dumps(
            {
                "condition": condition,
                "label": config["label"],
                "decision": "smoke-only",
                "accepted_evidence": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics = {
        "paper_id": "Explainable_FD_Toolkit",
        "protocol_id": "toolkit_ablation_smoke",
        "condition_id": condition,
        "accepted_evidence": False,
        "acceptance_blocker": "smoke runner only; no same-protocol GPU reviewer evidence",
        "seed": seed,
        "sample_count": 3,
        "metric_definitions": {
            "faithfulness_proxy": "deterministic smoke placeholder for artifact shape only",
            "stability_proxy": "deterministic smoke placeholder for artifact shape only",
            "toolkit_score_proxy": "weighted proxy, not a manuscript metric",
        },
        **_proxy_metrics(condition),
    }
    ended_at = datetime.now().isoformat()
    run_meta = {
        "paper_id": "Explainable_FD_Toolkit",
        "protocol_id": "toolkit_ablation_smoke",
        "condition_id": condition,
        "accepted_evidence": False,
        "seed": seed,
        "command": "python " + " ".join(sys.argv),
        "working_directory": str(Path.cwd()),
        "submodule_commit": _git_commit(),
        "input_artifact_paths": [str(input_config_path)],
        "output_artifact_paths": [str(output_summary_path)],
        "log_path": str(logs_root),
        "metrics_path": str(run_root / "metrics.json"),
        "started_at": started_at,
        "ended_at": ended_at,
        "runtime_seconds": time.perf_counter() - started_perf,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "GPU model": "unavailable in smoke environment",
        "GPU count": 0,
        "batch size": 1,
        "precision": "deterministic-smoke",
        "dataset split": "synthetic_toolkit_smoke",
        "OOM or failure reason": "",
    }

    (run_root / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_root / "run_meta.yaml").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"{condition}: score={metrics['toolkit_score_proxy']:.3f}, "
        f"accepted_evidence={metrics['accepted_evidence']}"
    )


def _conditions(selection: str) -> Sequence[str]:
    if selection == "all":
        return tuple(CONDITIONS)
    return (selection,)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate non-accepted Toolkit ablation smoke artifacts."
    )
    parser.add_argument("--condition", choices=["all", *CONDITIONS], default="all")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/toolkit_ablation_smoke"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for condition in _conditions(args.condition):
        _write_condition(condition, args.output, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
