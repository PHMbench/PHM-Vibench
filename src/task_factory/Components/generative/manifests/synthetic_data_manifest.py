from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

FORBIDDEN_SOURCE_SPLITS = {"val", "valid", "validation", "test", "target_test"}


def _normalization_has_params(normalization: dict[str, Any]) -> bool:
    per_channel = normalization.get("per_channel")
    params_artifact = normalization.get("params_artifact")
    params_hash = normalization.get("params_hash")
    if per_channel:
        return True
    return bool(params_artifact and params_hash)


def _has_evidence_value(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return bool(text) and text not in {"unspecified", "missing", "none", "null"}


def _leakage_checks_passed(leakage_checks: dict[str, Any]) -> bool:
    nearest = str(leakage_checks.get("nearest_neighbor_check", "")).lower()
    return bool(leakage_checks.get("split_guard_passed")) and nearest in {"passed", "pass", "ok"}


def build_synthetic_data_manifest(
    *,
    synthetic_dataset_id: str,
    model_type: str,
    model_name: str,
    loss_id: str,
    checkpoint_path: str,
    generator_run_id: str,
    source_split: str,
    domain_map_path: str,
    domain_map_hash: str,
    normalization: dict[str, Any],
    sampler_id: str,
    num_steps: int,
    seed: int,
    num_samples: int,
    shape: list[int] | tuple[int, ...],
    config_path: str = "configs/demo/10_generative/dummy_generative_cfm.yaml",
    protocol_path: str = "docs/schemas/generative_protocol.schema.json",
    protocol_hash: str = "unspecified",
    config_hash: str = "unspecified",
    dependency_lock_hash: str = "unspecified",
    status: str = "exploratory",
    leakage_checks: dict[str, Any] | None = None,
    condition_sampling_policy: str = "match_train_distribution",
    condition_counts: dict[str, int] | None = None,
    metric_status_reason_recorded: bool = False,
    sampler_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if source_split is None:
        raise ValueError("source_split is required")
    split_name = str(source_split).strip().lower()
    if split_name in FORBIDDEN_SOURCE_SPLITS:
        raise ValueError(
            "synthetic_data_manifest.source_split cannot use validation/test data; "
            f"got source_split={source_split!r}"
        )
    if not domain_map_hash:
        raise ValueError("domain_map_hash is required")
    if status not in {"benchmark-valid", "exploratory", "docs-only"}:
        raise ValueError(f"invalid validity status: {status}")

    normalization = dict(normalization)
    for field in ["method", "scope"]:
        if not normalization.get(field):
            raise ValueError(f"normalization.{field} is required")
    normalization["params_recorded"] = _normalization_has_params(normalization)

    checks = dict(
        leakage_checks
        or {
            "split_guard_passed": False,
            "nearest_neighbor_check": "not_run",
        }
    )
    counts = condition_counts or {}
    evidence = {
        "protocol_hash": _has_evidence_value(protocol_hash),
        "config_hash": _has_evidence_value(config_hash),
        "dependency_lock_hash": _has_evidence_value(dependency_lock_hash),
        "normalization_params": normalization["params_recorded"],
        "leakage_checks": _leakage_checks_passed(checks),
        "condition_sampling_policy": _has_evidence_value(condition_sampling_policy),
        "condition_counts": bool(counts),
        "metric_status_reason_recorded": bool(metric_status_reason_recorded),
    }
    benchmark_ready = (
        evidence["protocol_hash"]
        and evidence["config_hash"]
        and evidence["dependency_lock_hash"]
        and evidence["normalization_params"]
        and evidence["leakage_checks"]
        and evidence["condition_sampling_policy"]
        and evidence["condition_counts"]
        and evidence["metric_status_reason_recorded"]
    )
    missing_evidence = [key for key, ok in evidence.items() if not ok]
    reason = "requires complete benchmark evidence"
    if status == "benchmark-valid" and not benchmark_ready:
        status = "exploratory"
        reason = (
            "benchmark-valid requested but downgraded because evidence is incomplete: "
            + ", ".join(missing_evidence)
        )
    elif status == "benchmark-valid":
        reason = "all benchmark evidence passed"

    return {
        "schema_version": "0.1.0",
        "synthetic_dataset_id": synthetic_dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "protocol_id": "phmgen_cfm_v0_protocol",
            "protocol_path": protocol_path,
            "protocol_hash": protocol_hash,
        },
        "config": {
            "config_path": config_path,
            "config_hash": config_hash,
            "config_contract": "5-block environment/data/model/task/trainer with task.generative.*",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "dependency_lock_hash": dependency_lock_hash,
        },
        "generator": {
            "model_type": model_type,
            "model_name": model_name,
            "loss_id": loss_id,
            "checkpoint_path": checkpoint_path,
            "generator_run_id": generator_run_id,
        },
        "source_data": {
            "source_split": source_split,
            "forbidden_splits": ["val", "test", "target_test"],
            "domain_map_path": domain_map_path,
            "domain_map_hash": domain_map_hash,
        },
        "normalization": normalization,
        "conditions": {
            "condition_keys": ["fault_label", "domain_id"],
            "condition_sampling_policy": condition_sampling_policy,
            "condition_counts": counts,
        },
        "sampling": {
            "sampler_id": sampler_id,
            "num_steps": int(num_steps),
            "seed": int(seed),
            "num_samples": int(num_samples),
            "shape": list(shape),
            "sampler_metadata": dict(sampler_metadata or {}),
        },
        "validity": {
            "status": status,
            "allowed_status": ["benchmark-valid", "exploratory", "docs-only"],
            "benchmark_valid": status == "benchmark-valid",
            "reason": reason,
            "evidence": evidence,
            "missing_evidence": missing_evidence,
            "leakage_checks": checks,
        },
    }


def write_synthetic_data_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
