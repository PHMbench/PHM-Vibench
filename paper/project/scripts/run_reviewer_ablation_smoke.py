#!/usr/bin/env python3
"""Generate non-accepted smoke artifacts for Fuzzy-XFD reviewer ablations."""

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
    "hard_threshold": {
        "label": "hard-threshold inference replacement",
        "inference_mode": "hard_threshold",
        "safety_fallback_enabled": True,
        "rule_output_enabled": True,
    },
    "no_safety_fallback": {
        "label": "remove safety fallback path",
        "inference_mode": "fuzzy",
        "safety_fallback_enabled": False,
        "rule_output_enabled": True,
    },
    "no_rule_output": {
        "label": "remove rule-level explanation output",
        "inference_mode": "fuzzy",
        "safety_fallback_enabled": True,
        "rule_output_enabled": False,
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
    fuzzy_mode = config["inference_mode"] == "fuzzy"
    safety_enabled = bool(config["safety_fallback_enabled"])
    rule_output_enabled = bool(config["rule_output_enabled"])

    membership_coverage = 0.84 if fuzzy_mode else 0.18
    hard_threshold_rate = 0.12 if fuzzy_mode else 1.0
    safety_coverage = 0.91 if safety_enabled else 0.0
    unsupported_case_rate = 0.07 if safety_enabled else 0.31
    rule_trace_coverage = 0.88 if rule_output_enabled else 0.0
    explanation_completeness = (
        0.35 * membership_coverage + 0.35 * rule_trace_coverage + 0.30 * safety_coverage
    )
    reviewer_readiness = (
        0.30 * (1.0 - hard_threshold_rate)
        + 0.30 * safety_coverage
        + 0.25 * rule_trace_coverage
        + 0.15 * (1.0 - unsupported_case_rate)
    )
    return {
        "membership_coverage_proxy": round(membership_coverage, 6),
        "hard_threshold_rate_proxy": round(hard_threshold_rate, 6),
        "safety_coverage_proxy": round(safety_coverage, 6),
        "unsupported_case_rate_proxy": round(unsupported_case_rate, 6),
        "rule_trace_coverage_proxy": round(rule_trace_coverage, 6),
        "explanation_completeness_proxy": round(explanation_completeness, 6),
        "reviewer_readiness_proxy": round(reviewer_readiness, 6),
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

    input_config_path = inputs_root / "reviewer_ablation_config.json"
    output_summary_path = outputs_root / "reviewer_ablation_summary.json"
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
        "paper_id": "Paper_fuzzy_XFD",
        "protocol_id": "fuzzy_reviewer_ablation_smoke",
        "condition_id": condition,
        "accepted_evidence": False,
        "acceptance_blocker": "smoke runner only; no same-protocol reviewer-ablation artifacts",
        "seed": seed,
        "sample_count": 5,
        "metric_definitions": {
            "membership_coverage_proxy": "deterministic smoke placeholder for artifact shape only",
            "safety_coverage_proxy": "deterministic smoke placeholder for artifact shape only",
            "rule_trace_coverage_proxy": "deterministic smoke placeholder for artifact shape only",
            "reviewer_readiness_proxy": "weighted proxy, not a manuscript metric",
        },
        **_proxy_metrics(condition),
    }
    ended_at = datetime.now().isoformat()
    run_meta = {
        "paper_id": "Paper_fuzzy_XFD",
        "protocol_id": "fuzzy_reviewer_ablation_smoke",
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
        "dataset split": "synthetic_fuzzy_reviewer_smoke",
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
        f"{condition}: readiness={metrics['reviewer_readiness_proxy']:.3f}, "
        f"accepted_evidence={metrics['accepted_evidence']}"
    )


def _conditions(selection: str) -> Sequence[str]:
    if selection == "all":
        return tuple(CONDITIONS)
    return (selection,)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate non-accepted Fuzzy-XFD reviewer-ablation smoke artifacts."
    )
    parser.add_argument("--condition", choices=["all", *CONDITIONS], default="all")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/reviewer_ablation_smoke"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for condition in _conditions(args.condition):
        _write_condition(condition, args.output, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
