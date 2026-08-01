from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

import src.utils.p05_materialized_job_binding as binding_module
from src.configs.p05_contract import P05ExperimentContract
from src.utils.p05_materialized_job_binding import (
    verify_p05_materialized_job_binding,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = REPO_ROOT / "configs" / "experiments" / "p05" / "protocol"
PILOT_SCRIPT = REPO_ROOT / "scripts" / "materialize_p05_pilot_job.py"
NEURAL_SCRIPT = REPO_ROOT / "scripts" / "materialize_p05_neural_job.py"
TUNING_MATRIX = PROTOCOL_DIR / "neural_tuning_matrix_p05_v1.yaml"


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PILOT_MATERIALIZER = _load_script("p05_binding_pilot_materializer", PILOT_SCRIPT)
NEURAL_MATERIALIZER = _load_script("p05_binding_neural_materializer", NEURAL_SCRIPT)

ARMS = ("P05-M", "P05-B0", "P05-B1", "P05-B3")
DATASETS = (("CWRU", 1), ("XJTU", 2))
GPU_UUID = "GPU-P05-BINDING-TEST"


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


def _write_semantic(path: Path, semantic: dict[str, Any]) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    semantic_hash = _hash_bytes(_canonical(semantic))
    manifest = {**semantic, "content": {"semantic_sha256": semantic_hash}}
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return semantic_hash, _hash_file(path)


def _rewrite_semantic(path: Path, mutate) -> tuple[str, str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    semantic = {key: value for key, value in manifest.items() if key != "content"}
    mutate(semantic)
    return _write_semantic(path, semantic)


def _selection_manifest(tmp_path: Path) -> tuple[Path, str]:
    selections: list[dict[str, Any]] = []
    selection_index: dict[str, dict[str, Any]] = {}
    for dataset, dataset_id in DATASETS:
        for arm in ARMS:
            key = f"{dataset}/{arm}"
            job_id = f"P05-TUNE-{arm[4:]}-{dataset}-LR1E3"
            row = {
                "selection_id": f"P05-TUNING-SELECTION-{dataset}-{arm[4:]}",
                "arm_id": arm,
                "dataset": dataset,
                "dataset_id": dataset_id,
                "selected_learning_rate": 0.001,
                "selected_job_id": job_id,
                "selected_checkpoint_epoch": 7,
                "selected_val_f1_macro": 0.8,
                "selected_val_loss": 0.25,
                "selection_reason": "validation_macro_f1_max",
                "selected_config_sha256": _hash_bytes(f"{key}:config".encode()),
                "selected_code_sha256": _hash_bytes(f"{key}:code".encode()),
                "selected_run_contract_sha256": _hash_bytes(
                    f"{key}:run-contract".encode()
                ),
                "selected_checkpoint_sha256": _hash_bytes(
                    f"{key}:checkpoint".encode()
                ),
                "source_candidate_semantic_sha256": _hash_bytes(
                    f"{key}:candidate".encode()
                ),
            }
            row_index = len(selections)
            selections.append(row)
            selection_index[key] = {
                "row_index": row_index,
                "selection_id": row["selection_id"],
                "selected_learning_rate": row["selected_learning_rate"],
                "selected_job_id": row["selected_job_id"],
                "selected_checkpoint_sha256": row[
                    "selected_checkpoint_sha256"
                ],
                "selected_run_contract_sha256": row[
                    "selected_run_contract_sha256"
                ],
            }
    path = tmp_path / "selection" / "manifest.json"
    _, direct_hash = _write_semantic(
        path,
        {
            "schema_name": "p05.tuning_selection",
            "schema_version": 1,
            "paper_id": "P05",
            "phase": "tuning_selection",
            "status": "computed_unadjudicated",
            "claim_decision": "not_performed",
            "evidence_eligible": False,
            "test_access": "forbidden_and_not_performed",
            "protocol_bundle_sha256": binding_module.PROTOCOL_BUNDLE_SHA256,
            "source_matrix_sha256": _hash_file(TUNING_MATRIX),
            "protocol": {"fixture": "frozen-selector-output"},
            "candidates": [],
            "selections": selections,
            "selection_index": selection_index,
        },
    )
    return path, direct_hash


def _runtime(index: int, uuid: str = GPU_UUID) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "paper_id": "P05",
        "evidence_mode": True,
        "cuda_visible_devices": str(index),
        "physical_gpu_index": index,
        "gpu_uuid": uuid,
        "expected_gpu_uuid": GPU_UUID,
        "identity_source": "nvidia-smi:index,uuid",
        "accelerator": "gpu",
        "devices": 1,
        "gpus": 1,
        "strategy": "auto",
        "precision": 32,
        "deterministic": True,
    }


def _contract(
    *,
    phase: str,
    arm: str,
    dataset: str,
    seed: int,
) -> P05ExperimentContract:
    return P05ExperimentContract(
        arm_id=arm,
        dataset=dataset,
        dataset_id=1 if dataset == "CWRU" else 2,
        phase=phase,
        seed=seed,
        trace_export=arm == "P05-M" and phase != "tuning",
    )


def _materialized(tmp_path: Path, phase: str) -> dict[str, Any]:
    if phase == "pilot":
        materialized = PILOT_MATERIALIZER.materialize_p05_pilot_job(
            job_id="P05-PILOT-B0-CWRU",
            gpu_uuid=GPU_UUID,
            output_package=tmp_path / "job",
        )
        contract = _contract(
            phase="pilot",
            arm="P05-B0",
            dataset="CWRU",
            seed=20260801,
        )
        selection_path = None
    elif phase == "tuning":
        materialized = NEURAL_MATERIALIZER.materialize_p05_neural_job(
            stage="tuning",
            job_id="P05-TUNE-B1-XJTU-LR3E4",
            gpu_uuid=GPU_UUID,
            output_package=tmp_path / "job",
        )
        contract = _contract(
            phase="tuning",
            arm="P05-B1",
            dataset="XJTU",
            seed=20260801,
        )
        selection_path = None
    else:
        selection_path, selection_hash = _selection_manifest(tmp_path)
        materialized = NEURAL_MATERIALIZER.materialize_p05_neural_job(
            stage="decisive",
            job_id="P05-DEC-B3-XJTU-S42",
            gpu_uuid=GPU_UUID,
            output_package=tmp_path / "job",
            tuning_selection_manifest=selection_path,
            tuning_selection_sha256=selection_hash,
        )
        contract = _contract(
            phase="decisive",
            arm="P05-B3",
            dataset="XJTU",
            seed=42,
        )
    return {
        "config_path": Path(materialized["config_path"]),
        "manifest_path": Path(materialized["manifest_path"]),
        "physical_gpu_index": materialized["physical_gpu_index"],
        "contract": contract,
        "selection_path": selection_path,
    }


def _verify_inputs(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_path": fixture["config_path"],
        "experiment_contract": fixture["contract"],
        "runtime_identity": _runtime(fixture["physical_gpu_index"]),
        "cli_overrides": [],
        "local_config": None,
    }


@pytest.mark.parametrize("phase", ["pilot", "tuning", "decisive"])
def test_accepts_exact_materialized_packages_for_all_three_phases(
    tmp_path,
    phase,
) -> None:
    fixture = _materialized(tmp_path, phase)
    result = verify_p05_materialized_job_binding(**_verify_inputs(fixture))

    assert result.phase == phase
    assert result.config_path == fixture["config_path"].resolve()
    assert result.config_sha256 == _hash_file(fixture["config_path"])
    assert result.materialized_manifest_path == fixture["manifest_path"].resolve()
    assert result.materialized_manifest_sha256 == _hash_file(
        fixture["manifest_path"]
    )
    assert len(result.materialized_manifest_semantic_sha256) == 64
    assert result.physical_gpu_index == fixture["physical_gpu_index"]
    assert result.gpu_uuid == GPU_UUID
    assert result.evidence_eligible is False
    if phase == "pilot":
        assert result.launch_plan_sha256 == _hash_file(
            binding_module.PILOT_LAUNCH_PLAN_PATH
        )
        assert result.tuning_selection_sha256 is None
    elif phase == "tuning":
        assert result.launch_plan_sha256 is None
        assert result.tuning_selection_sha256 is None
    else:
        assert result.tuning_selection_path == fixture["selection_path"].resolve()
        assert result.tuning_selection_sha256 == _hash_file(
            fixture["selection_path"]
        )
        assert result.tuning_selection_semantic_sha256 is not None
        assert result.selected_tuning_job_id == "P05-TUNE-B3-XJTU-LR1E3"
        assert result.selected_checkpoint_sha256 is not None
        assert result.selected_run_contract_sha256 is not None
    with pytest.raises(FrozenInstanceError):
        result.job_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("cli_overrides", "local_config", "message"),
    [
        (["trainer.num_epochs=1"], None, "CLI override"),
        ([], "configs/local/unsafe.yaml", "local_config"),
    ],
)
def test_rejects_every_nonempty_cli_or_local_override(
    tmp_path,
    cli_overrides,
    local_config,
    message,
) -> None:
    fixture = _materialized(tmp_path, "tuning")
    inputs = _verify_inputs(fixture)
    inputs.update(cli_overrides=cli_overrides, local_config=local_config)
    with pytest.raises(ValueError, match=message):
        verify_p05_materialized_job_binding(**inputs)


def test_rejects_config_hash_drift_and_config_or_manifest_symlinks(tmp_path) -> None:
    fixture = _materialized(tmp_path / "hash", "pilot")
    fixture["config_path"].write_bytes(fixture["config_path"].read_bytes() + b"\n")
    with pytest.raises(ValueError, match="config raw SHA-256 mismatch"):
        verify_p05_materialized_job_binding(**_verify_inputs(fixture))

    fixture = _materialized(tmp_path / "config-link", "pilot")
    outside = tmp_path / "outside-config.yaml"
    fixture["config_path"].replace(outside)
    fixture["config_path"].symlink_to(outside)
    with pytest.raises(ValueError, match="must not be a symlink"):
        verify_p05_materialized_job_binding(**_verify_inputs(fixture))

    fixture = _materialized(tmp_path / "manifest-link", "pilot")
    outside_manifest = tmp_path / "outside-manifest.json"
    fixture["manifest_path"].replace(outside_manifest)
    fixture["manifest_path"].symlink_to(outside_manifest)
    with pytest.raises(ValueError, match="must not be a symlink"):
        verify_p05_materialized_job_binding(**_verify_inputs(fixture))


def test_rejects_duplicate_nonfinite_semantic_and_status_manifest_drift(tmp_path) -> None:
    fixture = _materialized(tmp_path / "duplicate", "pilot")
    payload = fixture["manifest_path"].read_text(encoding="utf-8")
    payload = payload.replace(
        '  "paper_id": "P05",',
        '  "paper_id": "P05",\n  "paper_id": "P05",',
        1,
    )
    fixture["manifest_path"].write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid P05 materialized job manifest"):
        verify_p05_materialized_job_binding(**_verify_inputs(fixture))

    fixture = _materialized(tmp_path / "nan", "pilot")
    payload = fixture["manifest_path"].read_text(encoding="utf-8")
    payload = payload.replace('"physical_gpu_index": 0', '"physical_gpu_index": NaN')
    fixture["manifest_path"].write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid P05 materialized job manifest"):
        verify_p05_materialized_job_binding(**_verify_inputs(fixture))

    fixture = _materialized(tmp_path / "semantic", "pilot")
    payload = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
    payload["claim_support"] = "allowed"
    fixture["manifest_path"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic hash mismatch"):
        verify_p05_materialized_job_binding(**_verify_inputs(fixture))

    fixture = _materialized(tmp_path / "status", "pilot")
    _rewrite_semantic(
        fixture["manifest_path"],
        lambda value: value.update(claim_support="allowed"),
    )
    with pytest.raises(ValueError, match="claim_support"):
        verify_p05_materialized_job_binding(**_verify_inputs(fixture))


def test_rejects_matrix_launch_plan_job_and_gpu_binding_drift(
    tmp_path,
    monkeypatch,
) -> None:
    fixture = _materialized(tmp_path / "matrix", "tuning")
    matrix_copy = tmp_path / "matrix" / "matrix-copy.yaml"
    matrix_copy.write_bytes(binding_module.TUNING_MATRIX_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(binding_module, "TUNING_MATRIX_PATH", matrix_copy)
    with pytest.raises(ValueError, match="matrix SHA-256 mismatch"):
        verify_p05_materialized_job_binding(**_verify_inputs(fixture))

    monkeypatch.setattr(binding_module, "TUNING_MATRIX_PATH", TUNING_MATRIX)
    fixture = _materialized(tmp_path / "launch", "pilot")
    launch_copy = tmp_path / "launch" / "launch-copy.yaml"
    launch_copy.write_bytes(
        binding_module.PILOT_LAUNCH_PLAN_PATH.read_bytes() + b"\n"
    )
    monkeypatch.setattr(binding_module, "PILOT_LAUNCH_PLAN_PATH", launch_copy)
    with pytest.raises(ValueError, match="launch_plan_sha256"):
        verify_p05_materialized_job_binding(**_verify_inputs(fixture))

    monkeypatch.setattr(
        binding_module,
        "PILOT_LAUNCH_PLAN_PATH",
        PROTOCOL_DIR / "pilot_launch_plan_p05_v1.yaml",
    )
    fixture = _materialized(tmp_path / "job", "tuning")
    _rewrite_semantic(
        fixture["manifest_path"],
        lambda value: value.update(job_id="P05-TUNE-M-CWRU-LR1E3"),
    )
    with pytest.raises(ValueError, match="matrix row|job_id|config"):
        verify_p05_materialized_job_binding(**_verify_inputs(fixture))

    fixture = _materialized(tmp_path / "gpu2", "pilot")
    _rewrite_semantic(
        fixture["manifest_path"],
        lambda value: value.update(physical_gpu_index=2),
    )
    with pytest.raises(ValueError, match="GPU0 or GPU1"):
        verify_p05_materialized_job_binding(**_verify_inputs(fixture))

    fixture = _materialized(tmp_path / "uuid", "pilot")
    inputs = _verify_inputs(fixture)
    inputs["runtime_identity"] = _runtime(
        fixture["physical_gpu_index"],
        uuid="GPU-OBSERVED-DIFFERENT",
    )
    with pytest.raises(ValueError, match="gpu_uuid"):
        verify_p05_materialized_job_binding(**inputs)


def _rebind_selection_direct_hash(fixture: dict[str, Any]) -> None:
    selection_hash = _hash_file(fixture["selection_path"])
    _rewrite_semantic(
        fixture["manifest_path"],
        lambda value: value["tuning_selection"].update(sha256=selection_hash),
    )


def test_decisive_reopens_selection_and_rejects_direct_or_semantic_drift(
    tmp_path,
) -> None:
    fixture = _materialized(tmp_path / "direct", "decisive")
    fixture["selection_path"].write_bytes(
        fixture["selection_path"].read_bytes() + b" "
    )
    with pytest.raises(ValueError, match="direct SHA-256 mismatch"):
        verify_p05_materialized_job_binding(**_verify_inputs(fixture))

    fixture = _materialized(tmp_path / "semantic", "decisive")
    selection = json.loads(fixture["selection_path"].read_text(encoding="utf-8"))
    selection["status"] = "adjudicated"
    fixture["selection_path"].write_text(json.dumps(selection), encoding="utf-8")
    _rebind_selection_direct_hash(fixture)
    with pytest.raises(ValueError, match="semantic hash mismatch"):
        verify_p05_materialized_job_binding(**_verify_inputs(fixture))


def test_decisive_rejects_selection_state_index_source_and_summary_drift(
    tmp_path,
) -> None:
    fixture = _materialized(tmp_path / "state", "decisive")
    _rewrite_semantic(
        fixture["selection_path"],
        lambda value: value.update(status="adjudicated"),
    )
    _rebind_selection_direct_hash(fixture)
    with pytest.raises(ValueError, match="status"):
        verify_p05_materialized_job_binding(**_verify_inputs(fixture))

    fixture = _materialized(tmp_path / "index", "decisive")
    _rewrite_semantic(
        fixture["selection_path"],
        lambda value: value["selection_index"]["XJTU/P05-B3"].update(
            row_index=0
        ),
    )
    _rebind_selection_direct_hash(fixture)
    with pytest.raises(ValueError, match="index conflicts"):
        verify_p05_materialized_job_binding(**_verify_inputs(fixture))

    fixture = _materialized(tmp_path / "source", "decisive")
    _rewrite_semantic(
        fixture["selection_path"],
        lambda value: value.update(source_matrix_sha256="00" * 32),
    )
    selection_hash = _hash_file(fixture["selection_path"])

    def update_materialized_source(value):
        value["tuning_selection"].update(
            sha256=selection_hash,
            source_matrix_sha256="00" * 32,
        )

    _rewrite_semantic(fixture["manifest_path"], update_materialized_source)
    with pytest.raises(ValueError, match="canonical tuning matrix"):
        verify_p05_materialized_job_binding(**_verify_inputs(fixture))

    fixture = _materialized(tmp_path / "summary", "decisive")
    _rewrite_semantic(
        fixture["manifest_path"],
        lambda value: value["tuning_selection"].update(
            selected_checkpoint_sha256="00" * 32
        ),
    )
    with pytest.raises(ValueError, match="selected_checkpoint_sha256"):
        verify_p05_materialized_job_binding(**_verify_inputs(fixture))
