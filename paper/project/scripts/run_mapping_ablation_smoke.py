#!/usr/bin/env python3
"""Generate non-accepted smoke artifacts for cross-method mapping ablation."""

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
    "no_mapping": {
        "label": "remove cross-method mapping module",
        "mapping_enabled": False,
        "source_backed_checks": False,
        "negative_mapping_checks": False,
        "mapped_submodules": [],
    },
    "scripted_mapping": {
        "label": "scripted cross-method mapping hook",
        "mapping_enabled": True,
        "source_backed_checks": False,
        "negative_mapping_checks": False,
        "mapped_submodules": ["1d2d", "moe", "fuzzy", "toolkit", "llm", "operator_attention"],
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
    mapped_count = len(config["mapped_submodules"])
    mapping_coverage = mapped_count / 6.0
    source_support = 1.0 if config["source_backed_checks"] else 0.0
    negative_support = 1.0 if config["negative_mapping_checks"] else 0.0
    consistency = 0.72 if config["mapping_enabled"] else 0.18
    claim_readiness = (
        0.35 * mapping_coverage
        + 0.25 * source_support
        + 0.20 * negative_support
        + 0.20 * consistency
    )
    return {
        "mapping_coverage_proxy": round(mapping_coverage, 6),
        "source_backing_proxy": round(source_support, 6),
        "negative_mapping_proxy": round(negative_support, 6),
        "framework_consistency_proxy": round(consistency, 6),
        "claim_readiness_proxy": round(claim_readiness, 6),
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
        "paper_id": "Neuralsymbolic_theory",
        "protocol_id": "mapping_ablation_smoke",
        "condition_id": condition,
        "accepted_evidence": False,
        "acceptance_blocker": "smoke runner only; no source-backed sibling-submodule evidence",
        "seed": seed,
        "sample_count": len(config["mapped_submodules"]),
        "metric_definitions": {
            "mapping_coverage_proxy": "deterministic smoke placeholder for artifact shape only",
            "source_backing_proxy": "0 until sibling-submodule source checks are implemented",
            "claim_readiness_proxy": "weighted proxy, not a manuscript metric",
        },
        **_proxy_metrics(condition),
    }
    ended_at = datetime.now().isoformat()
    run_meta = {
        "paper_id": "Neuralsymbolic_theory",
        "protocol_id": "mapping_ablation_smoke",
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
        "dataset split": "synthetic_mapping_smoke",
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
        f"{condition}: readiness={metrics['claim_readiness_proxy']:.3f}, "
        f"accepted_evidence={metrics['accepted_evidence']}"
    )


def _conditions(selection: str) -> Sequence[str]:
    if selection == "all":
        return tuple(CONDITIONS)
    return (selection,)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate non-accepted neural-symbolic mapping ablation smoke artifacts."
    )
    parser.add_argument("--condition", choices=["all", *CONDITIONS], default="all")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mapping_ablation_smoke"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for condition in _conditions(args.condition):
        _write_condition(condition, args.output, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
