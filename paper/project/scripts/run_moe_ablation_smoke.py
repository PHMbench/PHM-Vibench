#!/usr/bin/env python3
"""Generate non-accepted smoke artifacts for MoE ablation surfaces."""

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
    "no_load_balance": {
        "label": "remove load-balance regularization",
        "load_balance_enabled": False,
        "sparsity_enabled": True,
        "routing_temperature": 1.0,
        "expert_families": ["low_pass", "harmonic", "envelope"],
        "router_mode": "learned",
    },
    "no_sparsity": {
        "label": "remove sparsity regularization",
        "load_balance_enabled": True,
        "sparsity_enabled": False,
        "routing_temperature": 1.0,
        "expert_families": ["low_pass", "harmonic", "envelope"],
        "router_mode": "learned",
    },
    "temperature_sweep": {
        "label": "router temperature sweep",
        "load_balance_enabled": True,
        "sparsity_enabled": True,
        "routing_temperature": [0.5, 1.0, 2.0],
        "expert_families": ["low_pass", "harmonic", "envelope"],
        "router_mode": "learned",
    },
    "remove_expert_family": {
        "label": "remove harmonic expert family",
        "load_balance_enabled": True,
        "sparsity_enabled": True,
        "routing_temperature": 1.0,
        "expert_families": ["low_pass", "envelope"],
        "removed_expert_family": "harmonic",
        "router_mode": "learned",
    },
    "uniform_router": {
        "label": "uniform/equal-weight router",
        "load_balance_enabled": True,
        "sparsity_enabled": True,
        "routing_temperature": 1.0,
        "expert_families": ["low_pass", "harmonic", "envelope"],
        "router_mode": "uniform",
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


def _temperature_values(value: Any) -> Sequence[float]:
    if isinstance(value, list):
        return tuple(float(item) for item in value)
    return (float(value),)


def _proxy_metrics(condition: str) -> Dict[str, Any]:
    config = CONDITIONS[condition]
    temperatures = _temperature_values(config["routing_temperature"])
    family_count = len(config["expert_families"])
    learned_router = config["router_mode"] == "learned"

    load_balance = 0.82 if config["load_balance_enabled"] else 0.46
    sparsity = 0.77 if config["sparsity_enabled"] else 0.41
    specialization = 0.70 + 0.05 * max(family_count - 2, 0)
    if not learned_router:
        specialization = 0.33
    route_entropy = 0.55 + 0.10 * load_balance - 0.04 * (max(temperatures) - 1.0)
    stability = 0.62 + 0.12 * load_balance + 0.05 * sparsity
    if not learned_router:
        stability -= 0.10
    if condition == "temperature_sweep":
        stability -= 0.06

    score = (
        0.25 * load_balance
        + 0.20 * sparsity
        + 0.20 * specialization
        + 0.20 * route_entropy
        + 0.15 * stability
    )
    sweep_rows = [
        {
            "routing_temperature": temperature,
            "route_entropy_proxy": round(route_entropy - 0.03 * abs(temperature - 1.0), 6),
            "top_expert_weight_proxy": round(1.0 - min(route_entropy, 0.95) + 0.05 * temperature, 6),
        }
        for temperature in temperatures
    ]
    return {
        "load_balance_proxy": round(load_balance, 6),
        "sparsity_proxy": round(sparsity, 6),
        "expert_specialization_proxy": round(specialization, 6),
        "route_entropy_proxy": round(max(route_entropy, 0.0), 6),
        "route_stability_proxy": round(max(stability, 0.0), 6),
        "moe_score_proxy": round(max(score, 0.0), 6),
        "temperature_sweep_rows": sweep_rows,
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
        "paper_id": "MOE_explainable",
        "protocol_id": "moe_ablation_smoke",
        "condition_id": condition,
        "accepted_evidence": False,
        "acceptance_blocker": "smoke runner only; no same-protocol GPU reviewer evidence",
        "seed": seed,
        "sample_count": 3,
        "metric_definitions": {
            "route_entropy_proxy": "deterministic smoke placeholder for artifact shape only",
            "load_balance_proxy": "deterministic smoke placeholder for artifact shape only",
            "moe_score_proxy": "weighted proxy, not a manuscript metric",
        },
        **_proxy_metrics(condition),
    }
    ended_at = datetime.now().isoformat()
    run_meta = {
        "paper_id": "MOE_explainable",
        "protocol_id": "moe_ablation_smoke",
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
        "dataset split": "synthetic_moe_smoke",
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
        f"{condition}: score={metrics['moe_score_proxy']:.3f}, "
        f"accepted_evidence={metrics['accepted_evidence']}"
    )


def _conditions(selection: str) -> Sequence[str]:
    if selection == "all":
        return tuple(CONDITIONS)
    return (selection,)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate non-accepted MoE ablation smoke artifacts."
    )
    parser.add_argument("--condition", choices=["all", *CONDITIONS], default="all")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/moe_ablation_smoke"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for condition in _conditions(args.condition):
        _write_condition(condition, args.output, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
