from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

import src.utils.p05_tuning_candidate as tuning_candidate
import src.utils.p05_tuning_selection as tuning_selection
from src.utils.p05_tuning_candidate import (
    export_p05_tuning_validation_candidate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = (
    REPO_ROOT
    / "configs"
    / "experiments"
    / "p05"
    / "protocol"
    / "neural_tuning_matrix_p05_v1.yaml"
)
MATERIALIZER_PATH = REPO_ROOT / "scripts" / "materialize_p05_neural_job.py"
SPEC = importlib.util.spec_from_file_location(
    "materialize_p05_neural_job_for_candidate_test",
    MATERIALIZER_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MATERIALIZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATERIALIZER)

GPU_UUID = "GPU-P05-CANDIDATE-TEST"
PROVENANCE = {
    "source_metadata_sha256": "11" * 32,
    "derived_metadata_sha256": "22" * 32,
    "signal_cache_manifest_sha256": "33" * 32,
    "split_manifest_sha256": "44" * 32,
    "normalization_sha256": "55" * 32,
    "train_weight_plan_sha256": "66" * 32,
    "validation_weight_plan_sha256": "77" * 32,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _write_semantic(path: Path, semantic: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    semantic_hash = _hash_bytes(_canonical(semantic))
    manifest = {**semantic, "content": {"semantic_sha256": semantic_hash}}
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return semantic_hash


def _rewrite_semantic(path: Path, mutate) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    semantic = {key: value for key, value in manifest.items() if key != "content"}
    mutate(semantic)
    _write_semantic(path, semantic)


def _fixture(tmp_path: Path) -> dict[str, Any]:
    materialized = MATERIALIZER.materialize_p05_neural_job(
        stage="tuning",
        job_id="P05-TUNE-M-CWRU-LR1E3",
        gpu_uuid=GPU_UUID,
        output_package=tmp_path / "materialized",
    )
    materialized_manifest_path = Path(materialized["manifest_path"])
    materialized_config_path = Path(materialized["config_path"])

    # Runtime writes a distinct, canonicalized snapshot.  It remains
    # scientifically equivalent to the materialized source config, but its
    # bytes (and therefore its hash) need not be identical.
    config_path = tmp_path / "run" / "config_snapshot.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(
        materialized_config_path.read_bytes() + b"\n# runtime snapshot\n"
    )

    checkpoint_path = tmp_path / "run" / "best.ckpt"
    checkpoint_path.write_bytes(b"p05-best-validation-loss-checkpoint")
    checkpoint_sha256 = _hash_file(checkpoint_path)

    code_path = tmp_path / "run" / "code_snapshot.json"
    code_sha256 = _write_semantic(
        code_path,
        {
            "schema_name": "p05.code_snapshot",
            "schema_version": 1,
            "paper_id": "P05",
            "identity": "candidate-test-fixture",
        },
    )
    run_contract_path = tmp_path / "run" / "run_contract.json"
    _write_semantic(
        run_contract_path,
        {
            "schema_name": "p05.run_artifact_bundle",
            "schema_version": 1,
            "paper_id": "P05",
            "dataset_id": 1,
            "normalization_plan": {
                "sha256": PROVENANCE["normalization_sha256"]
            },
            "weight_plans": {
                "train": {"sha256": PROVENANCE["train_weight_plan_sha256"]},
                "validation": {
                    "sha256": PROVENANCE["validation_weight_plan_sha256"]
                },
            },
            "runtime_identity": {
                "schema_version": 1,
                "paper_id": "P05",
                "evidence_mode": True,
                "cuda_visible_devices": str(materialized["physical_gpu_index"]),
                "physical_gpu_index": materialized["physical_gpu_index"],
                "gpu_uuid": GPU_UUID,
                "expected_gpu_uuid": GPU_UUID,
                "identity_source": "nvidia-smi:index,uuid",
                "accelerator": "gpu",
                "devices": 1,
                "gpus": 1,
                "strategy": "auto",
                "precision": 32,
                "deterministic": True,
            },
            "provenance": {
                "checkpoint_sha256": checkpoint_sha256,
                "code_sha256": code_sha256,
                "config_sha256": _hash_file(config_path),
                "model_sha256": "88" * 32,
            },
        },
    )
    return {
        "materialized_job_manifest_path": materialized_manifest_path,
        "source_matrix_path": MATRIX_PATH,
        "val_loss": 0.25,
        "val_f1_macro": 0.875,
        "checkpoint_epoch": 7,
        "epochs_completed": 12,
        "data_roles_constructed": ["train", "validation"],
        "test_access_count": 0,
        "config_snapshot_path": config_path,
        "code_snapshot_manifest_path": code_path,
        "run_contract_manifest_path": run_contract_path,
        "checkpoint_path": checkpoint_path,
        "provenance": dict(PROVENANCE),
    }


def test_exports_exact_selector_candidate_with_complete_hash_bindings(tmp_path) -> None:
    inputs = _fixture(tmp_path)
    result = export_p05_tuning_validation_candidate(
        tmp_path / "candidate",
        **inputs,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.status == "created"
    assert manifest["schema_name"] == "p05.tuning_validation_candidate"
    assert manifest["schema_version"] == 1
    assert manifest["protocol_bundle_sha256"] == (
        tuning_selection.PROTOCOL_BUNDLE_SHA256
    )
    assert manifest["source_matrix_sha256"] == _hash_file(MATRIX_PATH)
    assert manifest["job"] == {
        "job_id": "P05-TUNE-M-CWRU-LR1E3",
        "phase": "tuning",
        "arm_id": "P05-M",
        "dataset": "CWRU",
        "dataset_id": 1,
        "seed": 20260801,
        "learning_rate": 0.001,
    }
    assert manifest["execution"]["status"] == "completed"
    assert manifest["execution"]["evidence_eligible"] is False
    assert manifest["execution"]["claim_decision"] == "not_performed"
    assert manifest["execution"]["data_roles_constructed"] == [
        "train",
        "validation",
    ]
    assert manifest["execution"]["test_access_count"] == 0
    assert manifest["validation"]["val_loss"] == 0.25
    assert manifest["validation"]["val_f1_macro"] == 0.875
    assert manifest["provenance"] == PROVENANCE
    assert manifest["artifacts"]["config_snapshot"]["sha256"] == _hash_file(
        Path(inputs["config_snapshot_path"])
    )
    semantic = {key: value for key, value in manifest.items() if key != "content"}
    assert manifest["content"]["semantic_sha256"] == _hash_bytes(
        _canonical(semantic)
    )

    # This is the actual exact-schema consumer, not a parallel test schema.
    consumed = tuning_selection._load_candidate(result.manifest_path)
    assert consumed.job_id == "P05-TUNE-M-CWRU-LR1E3"
    assert consumed.val_loss == tuning_selection.Decimal("0.25")
    assert consumed.val_f1_macro == tuning_selection.Decimal("0.875")


def test_create_only_reuses_exact_content_and_preserves_conflict(tmp_path) -> None:
    inputs = _fixture(tmp_path)
    package = tmp_path / "candidate"
    created = export_p05_tuning_validation_candidate(package, **inputs)
    before = created.manifest_path.read_bytes()
    reused = export_p05_tuning_validation_candidate(package, **inputs)
    assert reused.status == "reused"
    assert reused.manifest_path.read_bytes() == before

    changed = {**inputs, "val_loss": 0.20}
    with pytest.raises(FileExistsError, match="content conflicts"):
        export_p05_tuning_validation_candidate(package, **changed)
    assert created.manifest_path.read_bytes() == before


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"data_roles_constructed": ["train", "validation", "test"]}, "test construction"),
        ({"test_access_count": 1}, "test_access_count"),
        ({"val_loss": float("nan")}, "finite"),
        ({"val_f1_macro": 1.1}, "at most"),
        ({"checkpoint_epoch": 12}, "completed zero-based epoch"),
        ({"epochs_completed": 61}, "epochs_completed"),
    ],
)
def test_rejects_test_access_nonfinite_metrics_and_invalid_epochs(
    tmp_path,
    changes,
    message,
) -> None:
    inputs = _fixture(tmp_path)
    inputs.update(changes)
    with pytest.raises((TypeError, ValueError), match=message):
        export_p05_tuning_validation_candidate(tmp_path / "candidate", **inputs)
    assert not (tmp_path / "candidate").exists()


def test_rejects_materialized_manifest_matrix_and_config_drift(tmp_path) -> None:
    inputs = _fixture(tmp_path)
    materialized_path = Path(inputs["materialized_job_manifest_path"])
    payload = json.loads(materialized_path.read_text(encoding="utf-8"))
    payload["stage"] = "decisive"
    materialized_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic hash mismatch"):
        export_p05_tuning_validation_candidate(
            tmp_path / "bad-materialized",
            **inputs,
        )

    inputs = _fixture(tmp_path / "matrix")
    drifted_matrix = tmp_path / "matrix" / "drifted_matrix.yaml"
    drifted_matrix.write_bytes(MATRIX_PATH.read_bytes() + b"\n")
    inputs["source_matrix_path"] = drifted_matrix
    with pytest.raises(ValueError, match="source matrix SHA-256 mismatch"):
        export_p05_tuning_validation_candidate(tmp_path / "bad-matrix", **inputs)

    inputs = _fixture(tmp_path / "materialized-config")
    materialized_manifest = json.loads(
        Path(inputs["materialized_job_manifest_path"]).read_text(encoding="utf-8")
    )
    materialized_config = (
        Path(inputs["materialized_job_manifest_path"]).parent
        / materialized_manifest["config_file"]
    )
    materialized_config.write_bytes(materialized_config.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="materialized P05 config file SHA-256 mismatch"):
        export_p05_tuning_validation_candidate(
            tmp_path / "bad-materialized-config",
            **inputs,
        )

    inputs = _fixture(tmp_path / "runtime-config")
    Path(inputs["config_snapshot_path"]).write_bytes(
        Path(inputs["config_snapshot_path"]).read_bytes() + b"\n"
    )
    with pytest.raises(ValueError, match="config_sha256 conflicts"):
        export_p05_tuning_validation_candidate(
            tmp_path / "bad-runtime-config",
            **inputs,
        )


def test_rejects_artifact_hash_and_run_contract_cross_binding_drift(tmp_path) -> None:
    inputs = _fixture(tmp_path)
    Path(inputs["checkpoint_path"]).write_bytes(b"tampered-checkpoint")
    with pytest.raises(ValueError, match="checkpoint_sha256 conflicts"):
        export_p05_tuning_validation_candidate(
            tmp_path / "bad-checkpoint",
            **inputs,
        )

    inputs = _fixture(tmp_path / "code")
    code_path = Path(inputs["code_snapshot_manifest_path"])
    code_payload = json.loads(code_path.read_text(encoding="utf-8"))
    code_payload["identity"] = "tampered"
    code_path.write_text(json.dumps(code_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic hash mismatch"):
        export_p05_tuning_validation_candidate(tmp_path / "bad-code", **inputs)

    inputs = _fixture(tmp_path / "runtime")
    run_contract_path = Path(inputs["run_contract_manifest_path"])
    _rewrite_semantic(
        run_contract_path,
        lambda value: value["runtime_identity"].update(
            physical_gpu_index=1,
            cuda_visible_devices="1",
        ),
    )
    with pytest.raises(ValueError, match="runtime_identity"):
        export_p05_tuning_validation_candidate(tmp_path / "bad-runtime", **inputs)

    inputs = _fixture(tmp_path / "provenance")
    inputs["provenance"]["normalization_sha256"] = "99" * 32
    with pytest.raises(ValueError, match="normalization hash conflicts"):
        export_p05_tuning_validation_candidate(
            tmp_path / "bad-provenance",
            **inputs,
        )


def test_atomic_failure_and_symlink_target_leave_no_partial_package(
    tmp_path,
    monkeypatch,
) -> None:
    inputs = _fixture(tmp_path)

    def fail_install(source, target):
        del source, target
        raise RuntimeError("synthetic install failure")

    monkeypatch.setattr(tuning_candidate, "_rename_directory_noreplace", fail_install)
    package = tmp_path / "candidate"
    with pytest.raises(RuntimeError, match="synthetic install failure"):
        export_p05_tuning_validation_candidate(package, **inputs)
    assert not package.exists()
    assert not list(tmp_path.glob(".candidate.*.tmp"))

    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(FileExistsError, match="symlink"):
        export_p05_tuning_validation_candidate(linked, **inputs)
    assert not list(real.iterdir())
