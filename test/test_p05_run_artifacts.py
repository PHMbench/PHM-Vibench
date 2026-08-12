from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import src.utils.p05_run_artifacts as run_artifacts
from src.data_factory.p05_weighting import ExpectedRole, build_weight_plan
from src.data_factory.protocol_transforms import (
    WindowObservation,
    fit_train_channel_standardization,
)
from src.utils.p05_run_artifacts import export_p05_run_artifact_bundle


CONFIG_HASH = "a" * 64
MODEL_HASH = "b" * 64
CHECKPOINT_HASH = "c" * 64
CODE_HASH = "d" * 64


def _normalization_plan():
    observations = [
        WindowObservation("g1-0", "g1", np.asarray([[0.0, 2.0], [2.0, 4.0]])),
        WindowObservation("g2-0", "g2", np.asarray([[6.0, 8.0], [8.0, 10.0]])),
    ]
    return fit_train_channel_standardization(
        lambda: iter(observations),
        dataset_id=1,
        channel_names=("drive_end", "fan_end"),
        expected_window_size=2,
        expected_windows_per_group={"g1": 1, "g2": 1},
    )


def _weight_plans():
    train = pd.DataFrame(
        [
            {
                "Id": index + 1,
                "Dataset_id": 1,
                "Label": label,
                "Protocol_Group": f"train-{label}",
                "Protocol_Split": "train",
            }
            for index, label in enumerate((0, 1, 2, 3))
        ]
    )
    validation = pd.DataFrame(
        [
            {
                "Id": 11,
                "Dataset_id": 1,
                "Label": 0,
                "Protocol_Group": "val-a",
                "Protocol_Split": "validation",
            },
            {
                "Id": 12,
                "Dataset_id": 1,
                "Label": 1,
                "Protocol_Group": "val-a",
                "Protocol_Split": "validation",
            },
            {
                "Id": 13,
                "Dataset_id": 1,
                "Label": 2,
                "Protocol_Group": "val-b",
                "Protocol_Split": "validation",
            },
        ]
    )
    return {
        "train": build_weight_plan(
            train,
            dataset_id=1,
            role="train",
            expected=ExpectedRole(4, 4, {0: 1, 1: 1, 2: 1, 3: 1}, 16),
        ),
        "val": build_weight_plan(
            validation,
            dataset_id=1,
            role="validation",
            expected=ExpectedRole(3, 2, {0: 1, 1: 1, 2: 1}, 16),
        ),
    }


def _runtime_identity():
    return {
        "schema_version": 1,
        "paper_id": "P05",
        "evidence_mode": True,
        "cuda_visible_devices": "0",
        "physical_gpu_index": 0,
        "gpu_uuid": "GPU-EXPECTED",
        "expected_gpu_uuid": "GPU-EXPECTED",
        "identity_source": "nvidia-smi:index,uuid",
        "accelerator": "gpu",
        "devices": 1,
        "gpus": 1,
        "strategy": "auto",
        "precision": 32,
        "deterministic": True,
    }


def _export(package, **overrides):
    values = {
        "normalization_plan": _normalization_plan(),
        "weight_plans": _weight_plans(),
        "runtime_identity": _runtime_identity(),
        "config_sha256": CONFIG_HASH,
        "model_sha256": MODEL_HASH,
        "checkpoint_sha256": CHECKPOINT_HASH,
        "code_sha256": CODE_HASH,
    }
    values.update(overrides)
    return export_p05_run_artifact_bundle(package, **values)


def test_run_artifact_bundle_creates_and_semantically_reuses(tmp_path) -> None:
    package = tmp_path / "run-artifacts"

    created = _export(package)
    before = created.manifest_path.read_bytes()
    reused = _export(package)

    assert created.status == "created"
    assert reused.status == "reused"
    assert reused.manifest_path.read_bytes() == before
    manifest = json.loads(before)
    assert manifest["schema_name"] == "p05.run_artifact_bundle"
    assert manifest["dataset_id"] == 1
    assert manifest["normalization_plan"]["fit_role"] == "train"
    assert set(manifest["weight_plans"]) == {"train", "validation"}
    assert manifest["weight_plans"]["train"]["role"] == "train"
    assert manifest["weight_plans"]["validation"]["role"] == "validation"
    assert manifest["runtime_identity"]["gpu_uuid"] == "GPU-EXPECTED"
    assert manifest["provenance"] == {
        "checkpoint_sha256": CHECKPOINT_HASH,
        "code_sha256": CODE_HASH,
        "config_sha256": CONFIG_HASH,
        "model_sha256": MODEL_HASH,
    }
    semantic = {key: value for key, value in manifest.items() if key != "content"}
    assert manifest["content"]["semantic_sha256"] == run_artifacts._sha256_bytes(
        run_artifacts._canonical_json_bytes(semantic)
    )
    assert created.semantic_sha256 == manifest["content"]["semantic_sha256"]
    assert created.manifest_sha256 == reused.manifest_sha256
    assert not list(tmp_path.glob(".run-artifacts.*.tmp"))


def test_run_artifact_bundle_conflict_preserves_existing_bytes(tmp_path) -> None:
    package = tmp_path / "conflict"
    created = _export(package)
    before = created.manifest_path.read_bytes()

    with pytest.raises(FileExistsError, match="conflicts"):
        _export(package, code_sha256="e" * 64)

    assert created.manifest_path.read_bytes() == before


@pytest.mark.parametrize("plan_name", ["normalization", "train_weight", "val_weight"])
def test_run_artifact_bundle_rejects_tampered_source_hashes_before_write(
    tmp_path,
    plan_name,
) -> None:
    normalization = _normalization_plan()
    weights = _weight_plans()
    if plan_name == "normalization":
        normalization = replace(normalization, sha256="0" * 64)
    elif plan_name == "train_weight":
        weights["train"] = replace(weights["train"], sha256="0" * 64)
    else:
        weights["val"] = replace(weights["val"], sha256="0" * 64)
    package = tmp_path / plan_name

    with pytest.raises(ValueError, match="source SHA-256 does not match"):
        _export(
            package,
            normalization_plan=normalization,
            weight_plans=weights,
        )

    assert not package.exists()


def test_run_artifact_bundle_rejects_nonfinite_weights_and_missing_role(tmp_path) -> None:
    weights = _weight_plans()
    weights["train"] = replace(
        weights["train"],
        record_weights={1: float("nan")},
    )
    with pytest.raises(ValueError, match="finite and positive"):
        _export(tmp_path / "nan", weight_plans=weights)

    with pytest.raises(ValueError, match="exactly train and val"):
        _export(tmp_path / "missing", weight_plans={"train": _weight_plans()["train"]})

    assert not (tmp_path / "nan").exists()
    assert not (tmp_path / "missing").exists()


def test_run_artifact_bundle_rejects_runtime_contract_drift_and_bad_hash(tmp_path) -> None:
    runtime = _runtime_identity()
    runtime["physical_gpu_index"] = 1
    with pytest.raises(ValueError, match="must match cuda_visible_devices"):
        _export(tmp_path / "runtime", runtime_identity=runtime)

    with pytest.raises(ValueError, match="code_sha256"):
        _export(tmp_path / "hash", code_sha256="not-a-sha256")

    assert not (tmp_path / "runtime").exists()
    assert not (tmp_path / "hash").exists()


def test_run_artifact_bundle_refuses_symlink_target(tmp_path) -> None:
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    target = tmp_path / "linked"
    target.symlink_to(real_directory, target_is_directory=True)

    with pytest.raises(FileExistsError, match="symlink"):
        _export(target)

    assert not list(real_directory.iterdir())


def test_run_artifact_bundle_write_failure_leaves_no_partial_package(
    tmp_path,
    monkeypatch,
) -> None:
    package = tmp_path / "failed"

    def fail_write(path, content):
        del path, content
        raise RuntimeError("synthetic manifest failure")

    monkeypatch.setattr(run_artifacts, "_write_manifest_file", fail_write)
    with pytest.raises(RuntimeError, match="synthetic manifest failure"):
        _export(package)

    assert not package.exists()
    assert not list(tmp_path.glob(".failed.*.tmp"))


def test_run_artifact_bundle_rejects_tampered_existing_manifest(tmp_path) -> None:
    package = tmp_path / "tampered"
    created = _export(package)
    manifest = json.loads(created.manifest_path.read_text(encoding="utf-8"))
    manifest["runtime_identity"]["gpu_uuid"] = "GPU-TAMPERED"
    created.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FileExistsError, match="semantic hash does not match"):
        _export(package)
