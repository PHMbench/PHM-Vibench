from __future__ import annotations

import hashlib
import importlib.util
import json
import shlex
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.config_schema import ExperimentConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
TUNING_MATRIX_PATH = (
    REPO_ROOT
    / "configs"
    / "experiments"
    / "p05"
    / "protocol"
    / "neural_tuning_matrix_p05_v1.yaml"
)
DECISIVE_MATRIX_PATH = TUNING_MATRIX_PATH.with_name(
    "neural_decisive_matrix_p05_v1.yaml"
)
PILOT_MATRIX_PATH = TUNING_MATRIX_PATH.with_name("pilot_matrix_p05_v1.yaml")
MATERIALIZER_PATH = REPO_ROOT / "scripts" / "materialize_p05_neural_job.py"
SPEC = importlib.util.spec_from_file_location(
    "materialize_p05_neural_job",
    MATERIALIZER_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

ARMS = ("P05-M", "P05-B0", "P05-B1", "P05-B3")
DATASETS = ("CWRU", "XJTU")
LEARNING_RATES = (0.001, 0.0003)
DECISIVE_SEEDS = (42, 123, 456, 789, 1024)
COMMAND_PREFIX = ("conda", "run", "-n", "LQ_signal", "python")
GPU_UUID = "GPU-P05-TEST-0001"


def _load_matrix(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_selection_manifest(
    tmp_path: Path,
    *,
    selected_by_key: dict[str, float] | None = None,
    source_matrix_sha256: str | None = None,
) -> tuple[Path, str, dict[str, float]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    selected = selected_by_key or {
        f"{dataset}/{arm}": LEARNING_RATES[
            (dataset == "XJTU") ^ (arm in {"P05-B1", "P05-B3"})
        ]
        for dataset in DATASETS
        for arm in ARMS
    }
    assert set(selected) == {
        f"{dataset}/{arm}" for dataset in DATASETS for arm in ARMS
    }
    selections: list[dict[str, Any]] = []
    selection_index: dict[str, dict[str, Any]] = {}
    for dataset_index, dataset in enumerate(DATASETS, start=1):
        for arm in ARMS:
            key = f"{dataset}/{arm}"
            lr = selected[key]
            label = "LR1E3" if lr == 0.001 else "LR3E4"
            job_id = f"P05-TUNE-{arm.removeprefix('P05-')}-{dataset}-{label}"
            selection_id = (
                f"P05-TUNING-SELECTION-{dataset}-{arm.removeprefix('P05-')}"
            )
            row = {
                "selection_id": selection_id,
                "arm_id": arm,
                "dataset": dataset,
                "dataset_id": dataset_index,
                "selected_learning_rate": lr,
                "selected_job_id": job_id,
                "selected_checkpoint_epoch": 17,
                "selected_val_f1_macro": 0.75,
                "selected_val_loss": 0.5,
                "selection_reason": "unique_primary_validation_macro_f1_maximum",
                "selected_config_sha256": _digest(f"{key}:config"),
                "selected_code_sha256": _digest(f"{key}:code"),
                "selected_run_contract_sha256": _digest(f"{key}:contract"),
                "selected_checkpoint_sha256": _digest(f"{key}:checkpoint"),
                "source_candidate_semantic_sha256": _digest(f"{key}:candidate"),
            }
            row_index = len(selections)
            selections.append(row)
            selection_index[key] = {
                "row_index": row_index,
                "selection_id": selection_id,
                "selected_learning_rate": lr,
                "selected_job_id": job_id,
                "selected_checkpoint_sha256": row["selected_checkpoint_sha256"],
                "selected_run_contract_sha256": row[
                    "selected_run_contract_sha256"
                ],
            }
    manifest = {
        "schema_name": "p05.tuning_selection",
        "schema_version": 1,
        "paper_id": "P05",
        "phase": "tuning_selection",
        "status": "computed_unadjudicated",
        "claim_decision": "not_performed",
        "evidence_eligible": False,
        "test_access": "forbidden_and_not_performed",
        "protocol_bundle_sha256": MODULE.PROTOCOL_BUNDLE_SHA256,
        "source_matrix_sha256": (
            source_matrix_sha256
            or hashlib.sha256(TUNING_MATRIX_PATH.read_bytes()).hexdigest()
        ),
        "selections": selections,
        "selection_index": selection_index,
        "content": {"semantic_sha256": _digest("selection-semantic")},
    }
    path = tmp_path / "selection.json"
    payload = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest(), selected


def _config_and_manifest(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = Path(result["config_path"])
    manifest_path = Path(result["manifest_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    assert isinstance(manifest, dict)
    assert set(Path(result["package_dir"]).iterdir()) == {
        config_path,
        manifest_path,
    }
    assert result["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert result["manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert ExperimentConfig.model_validate(config, strict=True)
    return config, manifest


def test_neural_matrices_are_exact_factorials_with_safe_two_gpu_waves() -> None:
    tuning = _load_matrix(TUNING_MATRIX_PATH)
    decisive = _load_matrix(DECISIVE_MATRIX_PATH)
    MODULE._validate_matrix(tuning, stage="tuning")
    MODULE._validate_matrix(decisive, stage="decisive")

    assert hashlib.sha256(PILOT_MATRIX_PATH.read_bytes()).hexdigest() == (
        MODULE.PILOT_SHA256
    )
    assert len(tuning["jobs"]) == 16
    assert {
        (job["arm"], job["dataset"], job["learning_rate"])
        for job in tuning["jobs"]
    } == set(product(ARMS, DATASETS, LEARNING_RATES))
    assert {job["seed"] for job in tuning["jobs"]} == {20260801}
    assert tuning["stage_contract"]["environment_stage"] == "fit_validate_only"
    assert tuning["stage_contract"]["maximum_epochs"] == 60
    assert tuning["stage_contract"]["patience"] == 10
    assert tuning["stage_contract"]["trace_export"] == "false_for_every_arm"
    assert {
        tuning["stage_contract"][field]
        for field in (
            "test_dataset_construction",
            "test_cache_access",
            "test_metric_access",
        )
    } == {"forbidden"}

    assert len(decisive["jobs"]) == 40
    assert {
        (job["arm"], job["dataset"], job["seed"])
        for job in decisive["jobs"]
    } == set(product(ARMS, DATASETS, DECISIVE_SEEDS))
    assert all(
        "learning_rate" not in job and "lr" not in job
        for job in decisive["jobs"]
    )
    assert decisive["stage_contract"]["environment_stage"] == "fit_validate_test"
    assert decisive["stage_contract"]["maximum_epochs"] == 100
    assert decisive["stage_contract"]["patience"] == 15
    assert decisive["stage_contract"]["learning_rate"]["source"] == (
        "bound_hash_verified_tuning_selection_manifest_only"
    )
    assert decisive["stage_contract"]["trace_export"] == {
        "P05-M": True,
        "P05-B0": False,
        "P05-B1": False,
        "P05-B3": False,
    }

    for matrix, expected_wave_count in ((tuning, 8), (decisive, 20)):
        runtime = matrix["runtime"]
        assert runtime["allowed_physical_gpu_indices"] == [0, 1]
        assert runtime["forbidden_physical_gpu_indices"] == [2]
        assert runtime["maximum_concurrent_processes_per_gpu"] == 1
        assert runtime["one_gpu_per_process"] is True
        assert len(matrix["execution_waves"]) == expected_wave_count
        launched = []
        for wave in matrix["execution_waves"]:
            concurrent = wave["concurrent_jobs"]
            assert len(concurrent) == 2
            assert {item["physical_gpu_index"] for item in concurrent} == {0, 1}
            launched.extend(item["job_id"] for item in concurrent)
        assert len(launched) == len(set(launched)) == len(matrix["jobs"])
        assert set(launched) == {job["id"] for job in matrix["jobs"]}
        for job in matrix["jobs"]:
            command = tuple(shlex.split(job["materialize_command"]))
            assert command[: len(COMMAND_PREFIX)] == COMMAND_PREFIX

    assert tuning["outputs"] == decisive["outputs"] == {
        "materialization_status": "created-not-executed",
        "execution_status": "not_started",
        "evidence_status": "unadjudicated",
        "claim_support_before_ledger_and_audit": "forbidden",
    }


def test_all_sixteen_tuning_jobs_materialize_without_test_or_trace(
    tmp_path: Path,
) -> None:
    matrix = _load_matrix(TUNING_MATRIX_PATH)
    assignments = {
        item["job_id"]: item["physical_gpu_index"]
        for wave in matrix["execution_waves"]
        for item in wave["concurrent_jobs"]
    }
    for job in matrix["jobs"]:
        result = MODULE.materialize_p05_neural_job(
            stage="tuning",
            job_id=job["id"],
            gpu_uuid=GPU_UUID,
            output_package=tmp_path / job["id"],
        )
        config, manifest = _config_and_manifest(result)
        assert result["status"] == "created-not-executed"
        assert result["evidence_status"] == "unadjudicated"
        assert result["physical_gpu_index"] == assignments[job["id"]]
        assert config["environment"]["seed"] == 20260801
        assert config["environment"]["stage"] == "fit_validate_only"
        assert config["task"]["p05_run_phase"] == "tuning"
        assert config["task"]["p05_arm_id"] == job["arm"]
        assert config["task"]["lr"] == job["learning_rate"]
        assert config["task"]["p05_trace_export"] is False
        assert config["trainer"]["num_epochs"] == 60
        assert config["trainer"]["early_stopping"] is True
        assert config["trainer"]["patience"] == 10
        assert config["trainer"]["expected_gpu_uuid"] == GPU_UUID
        assert manifest["materialization_status"] == "created-not-executed"
        assert manifest["execution_status"] == "not_started"
        assert manifest["evidence_status"] == "unadjudicated"
        assert manifest["tuning_selection"] is None
        assert manifest["pilot_common_contract"]["pilot_matrix_sha256"] == (
            MODULE.PILOT_SHA256
        )
        uxfd = config["model"]["uxfd"]
        enabled = (
            uxfd["fuzzy"]["enable"],
            uxfd["neural_residual"]["enable"],
            uxfd["anfis"]["enable"],
        )
        assert enabled == {
            "P05-M": (True, False, False),
            "P05-B0": (False, False, False),
            "P05-B1": (False, True, False),
            "P05-B3": (False, False, True),
        }[job["arm"]]
        if job["arm"] == "P05-B1":
            assert uxfd["neural_residual"]["hidden_dim"] == (
                26 if job["dataset"] == "CWRU" else 29
            )


def test_all_forty_decisive_jobs_derive_lr_only_from_verified_selection(
    tmp_path: Path,
) -> None:
    selection_path, selection_hash, selected = _write_selection_manifest(tmp_path)
    matrix = _load_matrix(DECISIVE_MATRIX_PATH)
    for job in matrix["jobs"]:
        result = MODULE.materialize_p05_neural_job(
            stage="decisive",
            job_id=job["id"],
            gpu_uuid=GPU_UUID,
            output_package=tmp_path / job["id"],
            tuning_selection_manifest=selection_path,
            tuning_selection_sha256=selection_hash,
        )
        config, manifest = _config_and_manifest(result)
        expected_lr = selected[job["tuning_selection_key"]]
        assert result["learning_rate"] == expected_lr
        assert config["environment"]["seed"] == job["seed"]
        assert config["environment"]["stage"] == "fit_validate_test"
        assert config["task"]["p05_run_phase"] == "decisive"
        assert config["task"]["lr"] == expected_lr
        assert config["task"]["p05_trace_export"] is (job["arm"] == "P05-M")
        assert config["trainer"]["num_epochs"] == 100
        assert config["trainer"]["early_stopping"] is True
        assert config["trainer"]["patience"] == 15
        assert manifest["learning_rate_source"] == (
            "bound_hash_verified_tuning_selection_manifest"
        )
        assert manifest["tuning_selection"]["sha256"] == selection_hash
        assert manifest["tuning_selection"]["key"] == job["tuning_selection_key"]
        assert manifest["materialization_status"] == "created-not-executed"
        assert manifest["evidence_status"] == "unadjudicated"


def test_selection_and_pilot_hash_bindings_fail_closed_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = "P05-DEC-M-CWRU-S42"
    selection_path, selection_hash, _ = _write_selection_manifest(tmp_path)

    selection_path.write_bytes(selection_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="selection manifest SHA-256 mismatch"):
        MODULE.materialize_p05_neural_job(
            stage="decisive",
            job_id=job_id,
            gpu_uuid=GPU_UUID,
            output_package=tmp_path / "tampered-selection-package",
            tuning_selection_manifest=selection_path,
            tuning_selection_sha256=selection_hash,
        )
    assert not (tmp_path / "tampered-selection-package").exists()

    bad_source_path, bad_source_hash, _ = _write_selection_manifest(
        tmp_path / "bad-source",
        source_matrix_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="source matrix SHA-256 mismatch"):
        MODULE.materialize_p05_neural_job(
            stage="decisive",
            job_id=job_id,
            gpu_uuid=GPU_UUID,
            output_package=tmp_path / "bad-source-package",
            tuning_selection_manifest=bad_source_path,
            tuning_selection_sha256=bad_source_hash,
        )

    real_sha256_file = MODULE._sha256_file

    def report_pilot_drift(path: Path) -> str:
        if Path(path).name == PILOT_MATRIX_PATH.name:
            return "0" * 64
        return real_sha256_file(path)

    monkeypatch.setattr(MODULE, "_sha256_file", report_pilot_drift)
    with pytest.raises(ValueError, match="pilot common-contract SHA-256 mismatch"):
        MODULE.materialize_p05_neural_job(
            stage="tuning",
            job_id="P05-TUNE-M-CWRU-LR1E3",
            gpu_uuid=GPU_UUID,
            output_package=tmp_path / "pilot-drift-package",
        )
    assert not (tmp_path / "pilot-drift-package").exists()


def test_materialization_is_atomic_create_only_and_stage_bindings_are_strict(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    first = MODULE.materialize_p05_neural_job(
        stage="tuning",
        job_id="P05-TUNE-B3-XJTU-LR3E4",
        gpu_uuid=GPU_UUID,
        output_package=package,
    )
    before = {
        path.name: path.read_bytes() for path in Path(first["package_dir"]).iterdir()
    }
    with pytest.raises(FileExistsError, match="already exists"):
        MODULE.materialize_p05_neural_job(
            stage="tuning",
            job_id="not-a-job",
            gpu_uuid="not-a-uuid",
            output_package=package,
        )
    assert {
        path.name: path.read_bytes() for path in Path(first["package_dir"]).iterdir()
    } == before

    with pytest.raises(
        ValueError,
        match="tuning materialization forbids selection-manifest bindings",
    ):
        MODULE.materialize_p05_neural_job(
            stage="tuning",
            job_id="P05-TUNE-M-XJTU-LR1E3",
            gpu_uuid=GPU_UUID,
            output_package=tmp_path / "tuning-with-selection",
            tuning_selection_manifest=tmp_path / "unused",
            tuning_selection_sha256="0" * 64,
        )
    with pytest.raises(
        ValueError,
        match="decisive materialization requires selection path and SHA-256",
    ):
        MODULE.materialize_p05_neural_job(
            stage="decisive",
            job_id="P05-DEC-M-XJTU-S42",
            gpu_uuid=GPU_UUID,
            output_package=tmp_path / "decisive-without-selection",
        )
    with pytest.raises(ValueError, match="gpu_uuid"):
        MODULE.materialize_p05_neural_job(
            stage="tuning",
            job_id="P05-TUNE-M-XJTU-LR1E3",
            gpu_uuid="GPU2",
            output_package=tmp_path / "bad-uuid",
        )


def test_matrix_validator_rejects_gpu2_trace_test_and_decisive_lr_drift() -> None:
    tuning = _load_matrix(TUNING_MATRIX_PATH)
    tuning_gpu2 = deepcopy(tuning)
    tuning_gpu2["execution_waves"][0]["concurrent_jobs"][0][
        "physical_gpu_index"
    ] = 2
    with pytest.raises(ValueError, match="GPU0 and GPU1"):
        MODULE._validate_matrix(tuning_gpu2, stage="tuning")

    tuning_trace = deepcopy(tuning)
    tuning_trace["arms"]["P05-M"]["config"]["task"]["p05_trace_export"] = True
    with pytest.raises(ValueError, match="arm configuration drift"):
        MODULE._validate_matrix(tuning_trace, stage="tuning")

    tuning_test = deepcopy(tuning)
    tuning_test["stage_contract"]["environment_stage"] = "fit_validate_test"
    with pytest.raises(ValueError, match="stage contract drift"):
        MODULE._validate_matrix(tuning_test, stage="tuning")

    decisive = _load_matrix(DECISIVE_MATRIX_PATH)
    decisive_lr = deepcopy(decisive)
    decisive_lr["jobs"][0]["learning_rate"] = 0.001
    with pytest.raises(ValueError, match="forbids learning-rate fields"):
        MODULE._validate_matrix(decisive_lr, stage="decisive")
