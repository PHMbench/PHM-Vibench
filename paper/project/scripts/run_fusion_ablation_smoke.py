#!/usr/bin/env python3
"""Generate non-accepted smoke artifacts for 1D-2D fusion ablation surfaces."""

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
    "fft_only_proxy": {
        "label": "FFT-only signal layer proxy",
        "one_d_path": True,
        "two_d_path": True,
        "frequency_only": True,
        "statistical_features": True,
        "alignment_enabled": True,
        "fusion_mode": "frequency_only_proxy",
        "known_real_path_issue": None,
    },
    "legacy_ablation_surface": {
        "label": "legacy 1D-only/2D-only/no-statistical surface proxy",
        "one_d_path": "variant",
        "two_d_path": "variant",
        "frequency_only": False,
        "statistical_features": "variant",
        "alignment_enabled": True,
        "fusion_mode": "legacy_surface_proxy",
        "legacy_configs": [
            "configs/ablation/config_1D_only.yaml",
            "configs/ablation/config_2D_only.yaml",
            "configs/ablation/config_no_statistical.yaml",
        ],
        "known_real_path_issue": "stale_thu_paths_gpu2_missing_unified_baseline",
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
    one_d = 0.7 if config["one_d_path"] else 0.0
    two_d = 0.7 if config["two_d_path"] else 0.0
    if config["one_d_path"] == "variant":
        one_d = 0.5
    if config["two_d_path"] == "variant":
        two_d = 0.5
    statistical = 0.7 if config["statistical_features"] else 0.0
    if config["statistical_features"] == "variant":
        statistical = 0.35
    spectral = 0.8 if config["frequency_only"] else 0.45
    alignment = 0.7 if config["alignment_enabled"] else 0.2
    rewrite_readiness = 0.2 if config["known_real_path_issue"] else 1.0
    fusion_score = (
        0.20 * one_d
        + 0.20 * two_d
        + 0.20 * spectral
        + 0.15 * statistical
        + 0.15 * alignment
        + 0.10 * rewrite_readiness
    )
    return {
        "one_d_path_proxy": round(one_d, 6),
        "two_d_path_proxy": round(two_d, 6),
        "spectral_path_proxy": round(spectral, 6),
        "statistical_feature_proxy": round(statistical, 6),
        "alignment_proxy": round(alignment, 6),
        "current_root_rewrite_proxy": round(rewrite_readiness, 6),
        "fusion_ablation_score_proxy": round(fusion_score, 6),
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
        "paper_id": "1D-2D_fusion_explainable",
        "protocol_id": "fusion_ablation_smoke",
        "condition_id": condition,
        "accepted_evidence": False,
        "acceptance_blocker": "smoke runner only; no same-protocol Fusion1D2D GPU evidence",
        "seed": seed,
        "sample_count": 3,
        "metric_definitions": {
            "spectral_path_proxy": "deterministic smoke placeholder for artifact shape only",
            "current_root_rewrite_proxy": "0-1 proxy for whether the old runner is root/GPU compatible",
            "fusion_ablation_score_proxy": "weighted proxy, not a manuscript metric",
        },
        **_proxy_metrics(condition),
    }
    ended_at = datetime.now().isoformat()
    run_meta = {
        "paper_id": "1D-2D_fusion_explainable",
        "protocol_id": "fusion_ablation_smoke",
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
        "dataset split": "synthetic_fusion_smoke",
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
        f"{condition}: score={metrics['fusion_ablation_score_proxy']:.3f}, "
        f"accepted_evidence={metrics['accepted_evidence']}"
    )


def _conditions(selection: str) -> Sequence[str]:
    if selection == "all":
        return tuple(CONDITIONS)
    return (selection,)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate non-accepted 1D-2D fusion ablation smoke artifacts."
    )
    parser.add_argument("--condition", choices=["all", *CONDITIONS], default="all")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fusion_ablation_smoke"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for condition in _conditions(args.condition):
        _write_condition(condition, args.output, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
