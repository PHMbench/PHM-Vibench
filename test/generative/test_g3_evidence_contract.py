from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.task_factory.Components.generative.manifests import build_synthetic_manifest
from src.task_factory.Components.generative.normalization import (
    build_normalization_evidence,
    load_normalization_evidence,
    write_normalization_evidence,
)
from src.utils.generative_evidence import (
    strict_load_lightning_checkpoint,
    update_stage_ledger,
)


def _synthetic_manifest_kwargs() -> dict:
    return {
        "synthetic_dataset_id": "dummy",
        "method_id": "cfm",
        "model_type": "generative_model",
        "model_name": "phm_cfm_mlp1d",
        "loss_id": "conditional_flow_matching",
        "sampler_id": "euler_ode",
        "source_split": "train",
        "seed": 0,
        "num_steps": 2,
        "num_samples": 2,
        "shape": [2, 2, 16],
        "condition_sampling_policy": "first_metadata_repeated",
        "condition_counts": {"fault=0,domain=0": 2},
        "checkpoint_evidence": {"path": "model.ckpt", "sha256": "a", "strict": True},
        "normalization_evidence": {
            "path": "normalization_params.json",
            "sha256": "b",
            "source_split": "train",
            "scope": "per_channel",
        },
        "config_evidence": {"path": "resolved_config.json", "sha256": "c"},
        "protocol_evidence": {"path": "protocol.json", "sha256": "d"},
        "code_evidence": {"commit": "deadbeef"},
        "dependency_evidence": {"path": "requirements.txt", "sha256": "e"},
        "data_evidence": {
            "metadata_path": "metadata.csv",
            "metadata_sha256": "f",
            "domain_map_path": "domain_map.json",
            "domain_map_sha256": "g",
        },
        "generated_evidence": {"path": "samples.pt", "sha256": "h"},
        "leakage_metrics": {
            "nearest_neighbor_leakage_l2": {"value": 1.0, "status": "ok", "reason": ""},
            "duplicate_rate": {"value": 0.0, "status": "ok", "reason": ""},
        },
    }


def test_synthetic_manifest_requires_protocol_and_sample_hashes() -> None:
    manifest = build_synthetic_manifest(**_synthetic_manifest_kwargs())

    assert manifest["protocol"]["sha256"] == "d"
    assert manifest["generated_artifact"]["sha256"] == "h"
    assert manifest["validity"]["runtime_smoke_eligible"] is True


def test_synthetic_manifest_rejects_non_constitution_status() -> None:
    kwargs = _synthetic_manifest_kwargs()
    kwargs["scientific_status"] = "benchmark-candidate"

    with pytest.raises(ValueError, match="unsupported scientific status"):
        build_synthetic_manifest(**kwargs)


def test_docs_only_manifest_remains_non_promotional() -> None:
    kwargs = _synthetic_manifest_kwargs()
    kwargs["scientific_status"] = "docs-only"

    manifest = build_synthetic_manifest(**kwargs)

    assert manifest["validity"]["scientific_status"] == "docs-only"
    assert manifest["validity"]["benchmark_valid"] is False
    assert manifest["validity"]["paper_ready"] is False


def test_population_aware_manifest_requires_population_metric() -> None:
    kwargs = _synthetic_manifest_kwargs()
    kwargs["method_id"] = "population_aware_cfm"
    kwargs["population_metrics"] = {
        "population_dependency_mmd": {
            "value": 0.1,
            "status": "ok",
            "reason": "",
        }
    }

    manifest = build_synthetic_manifest(**kwargs)

    assert manifest["validity"]["evidence"]["population_metrics"] is True
    assert manifest["validity"]["runtime_smoke_eligible"] is True


def test_population_aware_manifest_downgrades_missing_population_metric() -> None:
    kwargs = _synthetic_manifest_kwargs()
    kwargs["method_id"] = "population_aware_cfm"

    manifest = build_synthetic_manifest(**kwargs)

    assert manifest["validity"]["runtime_smoke_eligible"] is False
    assert "population_metrics" in manifest["validity"]["missing_evidence"]


@pytest.mark.parametrize("missing_key", ["protocol_evidence", "generated_evidence"])
def test_synthetic_manifest_downgrades_missing_provenance(missing_key: str) -> None:
    kwargs = _synthetic_manifest_kwargs()
    kwargs[missing_key] = {}

    manifest = build_synthetic_manifest(**kwargs)

    assert manifest["validity"]["runtime_smoke_eligible"] is False
    assert missing_key.removesuffix("_evidence") in manifest["validity"]["missing_evidence"]


def test_normalization_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    evidence = build_normalization_evidence(
        torch.randn(3, 2, 8),
        method="standardization",
    )
    path, digest, _ = write_normalization_evidence(
        str(tmp_path / "normalization_params.json"),
        evidence,
    )
    assert load_normalization_evidence(path, expected_hash=digest)["sha256"] == digest

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["num_windows"] = 99
    Path(path).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_normalization_evidence(path, expected_hash=digest)


def test_checkpoint_restore_is_strict(tmp_path: Path) -> None:
    module = nn.Linear(2, 1)
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": module.state_dict()}, checkpoint)
    evidence = strict_load_lightning_checkpoint(module, checkpoint)
    assert evidence["strict"] is True

    incompatible = nn.Linear(3, 1)
    with pytest.raises(RuntimeError):
        strict_load_lightning_checkpoint(incompatible, checkpoint)


def test_stage_ledger_preserves_failure_and_completed_siblings(tmp_path: Path) -> None:
    path = tmp_path / "stage_ledger.json"
    update_stage_ledger(path, stage="train", values={"status": "completed"})
    update_stage_ledger(
        path,
        stage="sample",
        values={"status": "failed", "error": "boom"},
    )

    ledger = json.loads(path.read_text(encoding="utf-8"))
    assert ledger["stages"]["train"]["status"] == "completed"
    assert ledger["stages"]["sample"] == {"status": "failed", "error": "boom"}


def test_stage_ledger_running_state_clears_stale_failure(tmp_path: Path) -> None:
    path = tmp_path / "stage_ledger.json"
    update_stage_ledger(
        path,
        stage="train",
        values={"status": "failed", "error": "old failure"},
    )
    update_stage_ledger(
        path,
        stage="train",
        values={"status": "running", "run_dir": "new-run"},
    )

    ledger = json.loads(path.read_text(encoding="utf-8"))
    assert ledger["stages"]["train"] == {
        "status": "running",
        "run_dir": "new-run",
    }
