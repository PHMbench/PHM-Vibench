from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.task_factory.Components.generative.metrics import REQUIRED_METRICS
from src.utils.generative_evidence import runtime_environment


FORBIDDEN_GENERATION_SPLITS = frozenset(
    {"val", "valid", "validation", "test", "target_test"}
)


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return bool(text) and text not in {"none", "null", "missing", "unavailable"}


def build_synthetic_manifest(
    *,
    synthetic_dataset_id: str,
    method_id: str,
    model_type: str,
    model_name: str,
    loss_id: str,
    sampler_id: str,
    source_split: str,
    seed: int,
    num_steps: int,
    num_samples: int,
    shape: list[int] | tuple[int, ...],
    condition_sampling_policy: str,
    condition_counts: dict[str, int],
    checkpoint_evidence: dict[str, Any],
    normalization_evidence: dict[str, Any],
    config_evidence: dict[str, str],
    protocol_evidence: dict[str, str],
    code_evidence: dict[str, str],
    dependency_evidence: dict[str, str],
    data_evidence: dict[str, str],
    generated_evidence: dict[str, str],
    leakage_metrics: dict[str, Any],
    population_metrics: dict[str, Any] | None = None,
    sampler_metadata: dict[str, Any] | None = None,
    scientific_status: str = "exploratory",
) -> dict[str, Any]:
    split = str(source_split).strip().lower()
    if split in FORBIDDEN_GENERATION_SPLITS:
        raise ValueError(
            f"generation source_split cannot be {source_split!r}; use train data only"
        )
    if split != "train":
        raise ValueError(
            f"generation source_split must be train, got {source_split!r}"
        )
    if scientific_status not in {"exploratory", "docs-only"}:
        raise ValueError(
            f"unsupported scientific status for v0.2.1 smoke: {scientific_status}"
        )
    if int(num_steps) <= 0 or int(num_samples) <= 0:
        raise ValueError("num_steps and num_samples must be positive")
    if not condition_counts:
        raise ValueError("condition_counts are required")

    checkpoint_ok = bool(checkpoint_evidence.get("strict")) and _nonempty(
        checkpoint_evidence.get("sha256")
    )
    normalization_ok = (
        normalization_evidence.get("source_split") == "train"
        and normalization_evidence.get("scope") == "per_channel"
        and _nonempty(normalization_evidence.get("sha256"))
    )
    config_ok = _nonempty(config_evidence.get("path")) and _nonempty(
        config_evidence.get("sha256")
    )
    protocol_ok = _nonempty(protocol_evidence.get("path")) and _nonempty(
        protocol_evidence.get("sha256")
    )
    code_ok = _nonempty(code_evidence.get("commit"))
    dependency_ok = _nonempty(dependency_evidence.get("sha256"))
    data_ok = all(
        _nonempty(data_evidence.get(key))
        for key in ("metadata_path", "metadata_sha256", "domain_map_path", "domain_map_sha256")
    )
    generated_ok = _nonempty(generated_evidence.get("path")) and _nonempty(
        generated_evidence.get("sha256")
    )
    leakage_ok = all(
        leakage_metrics.get(name, {}).get("status") == "ok"
        for name in ("nearest_neighbor_leakage_l2", "duplicate_rate")
    )
    population_required = method_id == "population_aware_cfm"
    population_metrics = dict(population_metrics or {})
    population_ok = (
        population_metrics.get("population_dependency_mmd", {}).get("status")
        == "ok"
    )

    evidence = {
        "strict_checkpoint": checkpoint_ok,
        "train_normalization": normalization_ok,
        "config_hash": config_ok,
        "protocol": protocol_ok,
        "code_commit": code_ok,
        "dependency_hash": dependency_ok,
        "data_hashes": data_ok,
        "generated": generated_ok,
        "condition_counts": bool(condition_counts),
        "leakage_metrics": leakage_ok,
    }
    if population_required:
        evidence["population_metrics"] = population_ok
    missing = [name for name, passed in evidence.items() if not passed]

    manifest = {
        "schema_version": "0.2.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "synthetic_dataset_id": synthetic_dataset_id,
        "method": {
            "method_id": method_id,
            "model_type": model_type,
            "model_name": model_name,
            "loss_id": loss_id,
            "sampler_id": sampler_id,
        },
        "source_data": {
            "source_split": "train",
            **data_evidence,
        },
        "normalization": normalization_evidence,
        "checkpoint": checkpoint_evidence,
        "config": config_evidence,
        "protocol": protocol_evidence,
        "code": code_evidence,
        "dependency_lock": dependency_evidence,
        "environment": runtime_environment(),
        "generated_artifact": generated_evidence,
        "conditions": {
            "direct_keys": ["fault_label", "domain_id"],
            "sampling_policy": condition_sampling_policy,
            "counts": condition_counts,
        },
        "sampling": {
            "seed": int(seed),
            "num_steps": int(num_steps),
            "num_samples": int(num_samples),
            "shape": list(shape),
            "sampler_metadata": dict(sampler_metadata or {}),
        },
        "leakage": leakage_metrics,
        "population": population_metrics,
        "validity": {
            "scientific_status": scientific_status,
            "runtime_smoke_eligible": not missing,
            "missing_evidence": missing,
            "evidence": evidence,
            "benchmark_valid": False,
            "paper_ready": False,
        },
    }
    if population_required:
        manifest["population"] = population_metrics
    return manifest


def build_evaluation_manifest(
    *,
    generated_path: str,
    generated_sha256: str,
    synthetic_manifest_path: str,
    synthetic_manifest_sha256: str,
    metrics_path: str,
    metrics_sha256: str,
    reference_split: str,
    metrics: dict[str, Any],
    training_wall_clock_seconds: float | None,
    sampling_wall_clock_seconds: float | None,
) -> dict[str, Any]:
    split = str(reference_split).strip().lower()
    if split in {"test", "target_test"}:
        raise ValueError(
            "test-reference evaluation is not eligible in the maintained smoke contract"
        )

    method_required_metrics = list(
        metrics.get("summary", {}).get("required_for_method", [])
    )
    metric_names = list(REQUIRED_METRICS)
    metric_names.extend(
        name for name in method_required_metrics if name not in metric_names
    )
    statuses = {
        name: str(metrics.get(name, {}).get("status", "missing"))
        for name in metric_names
    }
    missing_status = [
        name for name, status in statuses.items() if status == "missing"
    ]
    failed = [name for name, status in statuses.items() if status == "failed"]
    not_computable = [
        name for name, status in statuses.items() if status == "not_computable"
    ]
    reasons_recorded = all(
        status == "ok" or bool(metrics.get(name, {}).get("reason"))
        for name, status in statuses.items()
    )
    artifacts_present = all(
        _nonempty(value)
        for value in (
            generated_path,
            generated_sha256,
            synthetic_manifest_path,
            synthetic_manifest_sha256,
            metrics_path,
            metrics_sha256,
        )
    )
    runtime_smoke_eligible = (
        artifacts_present
        and not missing_status
        and not failed
        and reasons_recorded
    )
    paper_smoke_metric_eligible = runtime_smoke_eligible and not not_computable

    manifest = {
        "schema_version": "0.2.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generated_artifact": {
            "path": generated_path,
            "sha256": generated_sha256,
        },
        "synthetic_manifest": {
            "path": synthetic_manifest_path,
            "sha256": synthetic_manifest_sha256,
        },
        "metrics_artifact": {
            "path": metrics_path,
            "sha256": metrics_sha256,
        },
        "reference_split": split,
        "metric_statuses": statuses,
        "metric_summary": {
            "ok": sum(status == "ok" for status in statuses.values()),
            "not_computable": len(not_computable),
            "failed": len(failed),
            "missing": len(missing_status),
            "not_computable_metrics": not_computable,
            "failed_metrics": failed,
            "missing_metrics": missing_status,
            "reasons_recorded": reasons_recorded,
        },
        "runtime": {
            "training_wall_clock_seconds": training_wall_clock_seconds,
            "sampling_wall_clock_seconds": sampling_wall_clock_seconds,
        },
        "promotion": {
            "runtime_smoke_eligible": runtime_smoke_eligible,
            "paper_smoke_metric_eligible": paper_smoke_metric_eligible,
            "sanity_ok": False,
            "paper_smoke_ready": False,
            "benchmark_valid": False,
            "reason": (
                "CPU/GPU E-chain and post-merge evidence are evaluated outside "
                "this single-run manifest"
            ),
        },
    }
    if method_required_metrics:
        manifest["metric_summary"][
            "required_for_method"
        ] = method_required_metrics
    return manifest
