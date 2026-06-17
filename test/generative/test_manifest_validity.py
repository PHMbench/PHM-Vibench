from __future__ import annotations

import pytest

from src.task_factory.Components.generative.manifests.synthetic_data_manifest import (
    build_synthetic_data_manifest,
)


def _complete_manifest(**overrides):
    kwargs = {
        "synthetic_dataset_id": "synthetic-smoke",
        "model_type": "generative_model",
        "model_name": "phm_cfm_mlp1d",
        "loss_id": "conditional_flow_matching",
        "checkpoint_path": "checkpoint.ckpt",
        "generator_run_id": "run-001",
        "source_split": "train",
        "domain_map_path": "configs/domain_maps/dummy_domain_map.csv",
        "domain_map_hash": "domain-hash",
        "normalization": {
            "method": "standardization",
            "scope": "per_channel",
            "params_artifact": "normalization_params.json",
            "params_hash": "normalization-hash",
        },
        "sampler_id": "euler_ode",
        "num_steps": 8,
        "seed": 0,
        "num_samples": 4,
        "shape": [4, 2, 128],
        "config_hash": "config-hash",
        "protocol_hash": "protocol-hash",
        "dependency_lock_hash": "dependency-hash",
        "status": "benchmark-valid",
        "leakage_checks": {
            "split_guard_passed": True,
            "nearest_neighbor_check": "passed",
        },
        "condition_sampling_policy": "grid",
        "condition_counts": {
            "fault=0,domain=0": 2,
            "fault=1,domain=0": 2,
        },
        "metric_status_reason_recorded": True,
    }
    kwargs.update(overrides)
    return build_synthetic_data_manifest(**kwargs)


def test_manifest_keeps_benchmark_valid_when_all_evidence_is_present() -> None:
    manifest = _complete_manifest()

    assert manifest["validity"]["status"] == "benchmark-valid"
    assert manifest["validity"]["benchmark_valid"] is True
    assert manifest["validity"]["missing_evidence"] == []
    assert all(manifest["validity"]["evidence"].values())


@pytest.mark.parametrize(
    ("field", "override", "missing_key"),
    [
        ("config_hash", {"config_hash": "unspecified"}, "config_hash"),
        ("protocol_hash", {"protocol_hash": "unspecified"}, "protocol_hash"),
        ("dependency_lock_hash", {"dependency_lock_hash": "missing"}, "dependency_lock_hash"),
        (
            "normalization",
            {"normalization": {"method": "standardization", "scope": "per_channel"}},
            "normalization_params",
        ),
        (
            "leakage_checks",
            {"leakage_checks": {"split_guard_passed": True, "nearest_neighbor_check": "not_run"}},
            "leakage_checks",
        ),
        ("condition_counts", {"condition_counts": {}}, "condition_counts"),
        (
            "metric_status_reason_recorded",
            {"metric_status_reason_recorded": False},
            "metric_status_reason_recorded",
        ),
        (
            "condition_sampling_split_verified",
            {"condition_sampling_split_verified": False},
            "condition_sampling_split_verified",
        ),
    ],
)
def test_manifest_downgrades_benchmark_valid_when_evidence_is_missing(
    field: str,
    override: dict,
    missing_key: str,
) -> None:
    manifest = _complete_manifest(**override)

    assert field
    assert manifest["validity"]["status"] == "exploratory"
    assert manifest["validity"]["benchmark_valid"] is False
    assert missing_key in manifest["validity"]["missing_evidence"]
    assert missing_key in manifest["validity"]["reason"]


@pytest.mark.parametrize("source_split", ["val", "valid", "validation", "test", "target_test"])
def test_manifest_rejects_forbidden_source_splits(source_split: str) -> None:
    with pytest.raises(ValueError, match="source_split cannot use validation/test data"):
        _complete_manifest(source_split=source_split)
