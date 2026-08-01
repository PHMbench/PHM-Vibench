from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.utils.p05_attempt_record import begin_p05_attempt, finish_p05_attempt
from src.utils.p05_post_pilot_budget import (
    P05PilotTimingBinding,
    create_p05_post_pilot_budget_decision,
    verify_p05_post_pilot_budget_decision,
)


JOBS = {
    "P05-PILOT-B0-CWRU": ("P05-B0", "CWRU", 1, 0),
    "P05-PILOT-M-CWRU": ("P05-M", "CWRU", 1, 0),
    "P05-PILOT-B0-XJTU": ("P05-B0", "XJTU", 2, 1),
    "P05-PILOT-M-XJTU": ("P05-M", "XJTU", 2, 1),
}
MATRIX_HASH = "2920bf24053579c0fbe192f780c07f0c2d59401be84a5b5c8c4bae6a3b3fb138"
LAUNCH_HASH = "a7a3b74b3decae5a6c0df34444db52e9b1ccf5078215c95572b9c52ecb757cfe"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_hash(value: object) -> str:
    content = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _direct_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_hashed_manifest(path: Path, semantic: dict) -> tuple[str, str]:
    semantic_hash = _canonical_hash(semantic)
    payload = {**semantic, "content": {"semantic_sha256": semantic_hash}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return semantic_hash, _direct_hash(path)


def _rewrite_hashed_manifest(path: Path, payload: dict) -> str:
    semantic = {key: value for key, value in payload.items() if key != "content"}
    return _write_hashed_manifest(path, semantic)[0]


def _runtime(index: int) -> dict:
    uuid = f"GPU-TEST-{index}"
    return {
        "accelerator": "gpu",
        "cuda_visible_devices": str(index),
        "deterministic": True,
        "devices": 1,
        "evidence_mode": True,
        "expected_gpu_uuid": uuid,
        "gpu_uuid": uuid,
        "gpus": 1,
        "identity_source": "nvidia-smi:index,uuid",
        "paper_id": "P05",
        "physical_gpu_index": index,
        "precision": 32,
        "schema_version": 1,
        "strategy": "auto",
    }


def _timing_manifest(path: Path, *, startup: float, epoch: float) -> str:
    semantic = {
        "artifact_class": "engineering_pilot_timing",
        "claim_support": "forbidden",
        "cuda_device": "cuda:0",
        "evidence_eligible": False,
        "measurement_status": "complete",
        "measurements": {
            "epoch_seconds_1_through_5": [epoch] * 5,
            "median_epoch_seconds_2_through_5": epoch,
            "peak_allocated_memory": 100,
            "peak_reserved_memory": 200,
            "startup_seconds": startup,
        },
        "paper_id": "P05",
        "schema_name": "p05.non_evidence_pilot_timing",
        "schema_version": 1,
        "timing_contract": {
            "epoch_end_includes_scheduled_validation": True,
            "expected_complete_epochs": 5,
            "median_epoch_numbers": [2, 3, 4, 5],
        },
    }
    return _write_hashed_manifest(path, semantic)[0]


def _materialized_job(root: Path, job_id: str) -> tuple[Path, Path, str]:
    arm, _dataset, dataset_id, physical = JOBS[job_id]
    package = root / "materialized"
    package.mkdir(parents=True)
    config_path = package / "config.yaml"
    config = {
        "environment": {
            "iterations": 1,
            "seed": 20260801,
            "stage": "fit_validate_only",
        },
        "data": {
            "allow_download": False,
            "batch_size": 64,
            "window_size": 4096,
        },
        "task": {
            "p05_arm_id": arm,
            "p05_evidence_mode": True,
            "p05_run_phase": "pilot",
            "target_system_id": [dataset_id],
        },
        "trainer": {
            "device": "cuda",
            "early_stopping": False,
            "expected_gpu_uuid": f"GPU-TEST-{physical}",
            "gpus": 1,
            "num_epochs": 5,
            "p05_pilot_mode": True,
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    semantic = {
        "claim_support": "forbidden",
        "config_file": "config.yaml",
        "config_sha256": _direct_hash(config_path),
        "evidence_eligible": False,
        "expected_gpu_uuid": f"GPU-TEST-{physical}",
        "job_id": job_id,
        "launch_plan_sha256": LAUNCH_HASH,
        "matrix_id": "P05-PILOT-v1",
        "matrix_sha256": MATRIX_HASH,
        "paper_id": "P05",
        "physical_gpu_index": physical,
        "schema_name": "p05.materialized_pilot_config",
        "schema_version": 1,
        "scientific_overrides": "forbidden",
    }
    manifest_path = package / "manifest.json"
    semantic_hash = _write_hashed_manifest(manifest_path, semantic)[0]
    return manifest_path, config_path, semantic_hash


def _run_contract(
    path: Path,
    *,
    dataset_id: int,
    runtime: dict,
    config_hash: str,
    checkpoint_hash: str,
    model_hash: str,
    code_hash: str,
) -> str:
    dataset = "CWRU" if dataset_id == 1 else "XJTU"
    semantic = {
        "dataset_id": dataset_id,
        "normalization_plan": {
            "dataset_id": dataset_id,
            "fit_role": "train",
            "sha256": _digest(f"normalization:{dataset}"),
        },
        "paper_id": "P05",
        "provenance": {
            "checkpoint_sha256": checkpoint_hash,
            "code_sha256": code_hash,
            "config_sha256": config_hash,
            "model_sha256": model_hash,
        },
        "runtime_identity": runtime,
        "schema_name": "p05.run_artifact_bundle",
        "schema_version": 1,
        "weight_plans": {
            "train": {"sha256": _digest(f"train-weight:{dataset}")},
            "validation": {"sha256": _digest(f"val-weight:{dataset}")},
        },
    }
    return _write_hashed_manifest(path, semantic)[0]


def _binding(
    root: Path,
    job_id: str,
    *,
    startup: float,
    epoch: float,
    central_unit: float,
    d03_unit: float,
) -> tuple[P05PilotTimingBinding, dict, Path | None]:
    arm, dataset, dataset_id, physical = JOBS[job_id]
    root.mkdir(parents=True)
    materialized_manifest, config_path, materialized_hash = _materialized_job(
        root, job_id
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_snapshot_path = root / "config_snapshot.yaml"
    config_snapshot_path.write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    config_hash = _direct_hash(config_snapshot_path)
    assert config_hash != _direct_hash(config_path)
    timing_path = root / "timing" / "manifest.json"
    timing_hash = _timing_manifest(timing_path, startup=startup, epoch=epoch)
    checkpoint_hash = _digest(f"checkpoint:{job_id}")
    model_hash = _digest(f"model:{job_id}")
    code_hash = _digest("common-code")
    runtime = _runtime(physical)
    run_contract_path = root / "run-contract" / "manifest.json"
    run_contract_hash = _run_contract(
        run_contract_path,
        dataset_id=dataset_id,
        runtime=runtime,
        config_hash=config_hash,
        checkpoint_hash=checkpoint_hash,
        model_hash=model_hash,
        code_hash=code_hash,
    )
    attempt_provenance = {
        "source_metadata_sha256": _digest("source"),
        "derived_metadata_sha256": _digest("derived"),
        "signal_cache_manifest_sha256": _digest("cache"),
        "split_manifest_sha256": _digest(f"split:{dataset}"),
        "config_snapshot_sha256": config_hash,
        "code_snapshot_sha256": code_hash,
        "normalization_sha256": _digest(f"normalization:{dataset}"),
        "train_weight_plan_sha256": _digest(f"train-weight:{dataset}"),
        "validation_weight_plan_sha256": _digest(f"val-weight:{dataset}"),
    }
    attempt_dir = root / "attempt"
    begin_p05_attempt(
        attempt_dir,
        attempt_id=job_id,
        arm_id=arm,
        phase="pilot",
        dataset_id=dataset_id,
        seed=20260801,
        command_argv=[
            "conda",
            "run",
            "-n",
            "LQ_signal",
            "python",
            "main.py",
            "--config",
            str(config_path),
        ],
        working_directory=root,
        package_versions={"python": "3.10"},
        device_identity=runtime,
        provenance=attempt_provenance,
        started_at_utc="2026-08-01T00:00:00+00:00",
    )
    run = {
        "arm": arm,
        "checkpoint_sha256": checkpoint_hash,
        "config_sha256": config_hash,
        "dataset": dataset,
        "device_uuid": runtime["gpu_uuid"],
        "model_sha256": model_hash,
        "physical_gpu_index": physical,
    }
    evaluator_package = None
    outputs = {
        "all_results": _digest(f"all-results:{job_id}"),
        "checkpoint": checkpoint_hash,
        "code_snapshot": code_hash,
        "config_snapshot": config_hash,
        "materialized_job": materialized_hash,
        "pilot_timing": timing_hash,
        "result": _digest(f"result:{job_id}"),
        "run_contract": run_contract_hash,
    }
    if arm == "P05-M":
        evaluator_package, evaluator_hash = _evaluator_summary(
            root,
            dataset=dataset,
            m_run=run,
            central_unit=central_unit,
            d03_unit=d03_unit,
        )
        outputs.update(
            {
                "pilot_d03": _digest(f"d03:{dataset}"),
                "pilot_evaluator_benchmark": evaluator_hash,
                "trace_val": _digest(f"trace-val:{dataset}"),
            }
        )
    finish_p05_attempt(
        attempt_dir,
        status="completed",
        output_artifact_sha256=outputs,
        finished_at_utc="2026-08-01T00:01:00+00:00",
    )
    descriptor = P05PilotTimingBinding(
        timing_manifest_path=timing_path,
        attempt_package_dir=attempt_dir,
        materialized_job_manifest_path=materialized_manifest,
        run_contract_manifest_path=run_contract_path,
    )
    return descriptor, run, evaluator_package


def _component(unit: float, calls: int) -> dict:
    total = unit * 256 * calls
    return {
        "forward_calls_per_window": calls,
        "seconds_per_forward_call_per_window": unit,
        "seconds_per_window": unit * calls,
        "total_seconds": total,
    }


def _evaluator_summary(
    root: Path,
    *,
    dataset: str,
    m_run: dict,
    central_unit: float,
    d03_unit: float,
) -> tuple[Path, str]:
    central_components = {
        "consequent_shuffles": _component(central_unit, 32),
        "original_trace": _component(central_unit, 1),
        "rule_deletions": _component(central_unit, 10),
    }
    d03_components = {
        "noise_draws": _component(d03_unit, 32),
        "original_trace": _component(d03_unit, 1),
    }
    semantic = {
        "benchmarks": {
            "central_e1_e2": {
                "actual_forward_calls": 43,
                "components": central_components,
                "seconds_per_window": central_unit * 43,
                "source_semantic_sha256": _digest(f"central:{dataset}"),
                "total_seconds": central_unit * 43 * 256,
            },
            "d03": {
                "actual_forward_calls": 33,
                "chunk_count": 1,
                "components": d03_components,
                "seconds_per_window": d03_unit * 33,
                "source_semantic_sha256": _digest(f"d03:{dataset}"),
                "total_seconds": d03_unit * 33 * 256,
            },
        },
        "conclusion_control": {
            "budget_decision": "not_performed",
            "claim_decisions": "forbidden",
            "paper_evidence": False,
            "performance_conclusion": "forbidden",
            "scientific_status": "unadjudicated",
        },
        "purpose": {
            "budget_cap_gpu_hours": 168,
            "makes_budget_decision": False,
            "role": "observed_input_to_gpu_hour_budget_forecast_only",
        },
        "schema_name": "p05.pilot_evaluator_benchmark",
        "schema_version": 1,
        "scope": {
            "dataset": dataset,
            "model_seed": 20260801,
            "partition_sample_count": 304 if dataset == "CWRU" else 5268,
            "sample_count": 256,
            "sample_id_semantic_sha256": _digest(f"sample-ids:{dataset}"),
            "selection": "first_256_after_stable_sample_id_sort",
            "split": "validation",
        },
        "shared_provenance": {
            "checkpoint_sha256": m_run["checkpoint_sha256"],
            "config_sha256": m_run["config_sha256"],
            "device_uuid": m_run["device_uuid"],
            "model_sha256": m_run["model_sha256"],
            "physical_gpu_index": m_run["physical_gpu_index"],
        },
        "status": "engineering_non_evidence",
    }
    package = root / f"evaluator-{dataset}"
    semantic_hash, _ = _write_hashed_manifest(package / "manifest.json", semantic)
    return package, semantic_hash


def _inputs(
    root: Path,
    *,
    startup: float = 1.0,
    epoch: float = 1.0,
    central_unit: float = 0.001,
    d03_unit: float = 0.001,
) -> tuple[list[P05PilotTimingBinding], list[Path]]:
    bindings = []
    evaluators = []
    for job_id in JOBS:
        binding, _run, evaluator = _binding(
            root / job_id,
            job_id,
            startup=startup,
            epoch=epoch,
            central_unit=central_unit,
            d03_unit=d03_unit,
        )
        bindings.append(binding)
        if evaluator is not None:
            evaluators.append(evaluator)
    return bindings, evaluators


def _create(root: Path, **timing):
    bindings, evaluators = _inputs(root / "inputs", **timing)
    return create_p05_post_pilot_budget_decision(
        root / "budget",
        pilot_bindings=bindings,
        evaluator_benchmark_package_dirs=evaluators,
    )


def test_budget_locks_full_program_and_records_exact_formulas(tmp_path) -> None:
    result = _create(tmp_path)
    manifest = verify_p05_post_pilot_budget_decision(result.package_dir)

    assert result.status == "locked_first_acceptable_program"
    assert result.selected_program == "full_90_with_d03"
    assert manifest["decision"] == {
        "ablations_retained": True,
        "d03_retained": True,
        "first_acceptable_program_locked": True,
        "selected_program": "full_90_with_d03",
        "status": "locked_first_acceptable_program",
    }
    assert manifest["frozen_contract"]["job_counts"] == {
        "central_program": 60,
        "decisive_central": 40,
        "full_program": 90,
        "pilot": 4,
        "retraining_ablations": 30,
        "tuning": 16,
    }
    assert manifest["frozen_contract"]["statistics_sha256"] == (
        "c0b9a0baedddbc8e6ee76c465b19b6c447b85f66123dfb65cdc9bb7460525ecb"
    )
    assert manifest["frozen_contract"]["protocol_bundle_sha256"] == (
        "8d01361c39a778d437ce235ad1e8d3877313f128d6593fbb74812a4b237a1654"
    )
    cwru_m_input = manifest["inputs"]["pilot_jobs"]["P05-PILOT-M-CWRU"]
    assert cwru_m_input["materialized_job"]["config_direct_sha256"] != (
        cwru_m_input["run_contract"]["provenance"]["config_sha256"]
    )
    cwru_central = manifest["evaluation_forecast"]["per_dataset"]["CWRU"][
        "programs"
    ]["central_mandatory"]["components"]
    assert cwru_central["central_original"][
        "structural_multiplier_per_window_per_seed"
    ] == 4
    assert cwru_central["central_original"]["seed_count"] == 5
    assert cwru_central["central_original"]["test_window_count"] == 368
    assert cwru_central["central_original"][
        "final_forward_window_multiplier"
    ] == 4 * 5 * 368
    xjtu_ablation = manifest["evaluation_forecast"]["per_dataset"]["XJTU"][
        "programs"
    ]["retraining_ablations"]["components"]
    assert xjtu_ablation["central_deletion"][
        "structural_multiplier_per_window_per_seed"
    ] == 35
    assert xjtu_ablation["central_shuffle"][
        "final_forward_window_multiplier"
    ] == 96 * 5 * 26588
    assert manifest["training_forecast"]["per_dataset"]["CWRU"]["stages"][
        "pilot"
    ]["jobs_per_dataset"] == 2
    assert manifest["conclusion_control"]["performance_conclusion"] == "forbidden"

    with pytest.raises(FileExistsError, match="create-only"):
        create_p05_post_pilot_budget_decision(
            result.package_dir,
            pilot_bindings=[],
            evaluator_benchmark_package_dirs=[],
        )


@pytest.mark.parametrize(
    ("central_unit", "d03_unit", "expected_program", "expected_status"),
    [
        (0.001, 1.0, "full_90_without_d03", "locked_first_acceptable_program"),
        (0.02, 0.001, "central_60_mandatory", "locked_first_acceptable_program"),
        (
            0.08,
            0.001,
            None,
            "stop_requires_human_protocol_or_resource_amendment",
        ),
    ],
)
def test_budget_decision_order_and_stop_branch(
    tmp_path,
    central_unit,
    d03_unit,
    expected_program,
    expected_status,
) -> None:
    result = _create(
        tmp_path,
        central_unit=central_unit,
        d03_unit=d03_unit,
    )
    manifest = verify_p05_post_pilot_budget_decision(result.package_dir)
    assert result.selected_program == expected_program
    assert result.status == expected_status
    assert manifest["decision"]["selected_program"] == expected_program
    assert manifest["decision"]["status"] == expected_status
    if expected_program is None:
        assert manifest["programs"][2]["within_168_gpu_hour_cap"] is False


def test_budget_rejects_missing_grid_and_cross_binding_tamper(tmp_path) -> None:
    bindings, evaluators = _inputs(tmp_path / "inputs")
    with pytest.raises(ValueError, match="exactly four"):
        create_p05_post_pilot_budget_decision(
            tmp_path / "missing",
            pilot_bindings=bindings[:3],
            evaluator_benchmark_package_dirs=evaluators,
        )

    terminal_path = bindings[0].attempt_package_dir / "terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["outputs"]["pilot_timing"] = "0" * 64
    semantic = {key: value for key, value in terminal.items() if key != "content"}
    terminal["content"]["semantic_sha256"] = _canonical_hash(semantic)
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    with pytest.raises(ValueError, match="outputs do not bind"):
        create_p05_post_pilot_budget_decision(
            tmp_path / "tampered",
            pilot_bindings=bindings,
            evaluator_benchmark_package_dirs=evaluators,
        )


def test_budget_rejects_arm_output_and_summary_cross_link_tamper(tmp_path) -> None:
    bindings, evaluators = _inputs(tmp_path / "missing-output")
    terminal_path = bindings[0].attempt_package_dir / "terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    del terminal["outputs"]["result"]
    _rewrite_hashed_manifest(terminal_path, terminal)
    with pytest.raises(ValueError, match="output key set"):
        create_p05_post_pilot_budget_decision(
            tmp_path / "missing-output-budget",
            pilot_bindings=bindings,
            evaluator_benchmark_package_dirs=evaluators,
        )

    bindings, evaluators = _inputs(tmp_path / "wrong-materialized")
    terminal_path = bindings[0].attempt_package_dir / "terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["outputs"]["materialized_job"] = "0" * 64
    _rewrite_hashed_manifest(terminal_path, terminal)
    with pytest.raises(ValueError, match="outputs do not bind"):
        create_p05_post_pilot_budget_decision(
            tmp_path / "wrong-materialized-budget",
            pilot_bindings=bindings,
            evaluator_benchmark_package_dirs=evaluators,
        )

    bindings, evaluators = _inputs(tmp_path / "wrong-summary")
    m_binding = next(
        binding
        for binding in bindings
        if "P05-PILOT-M-CWRU" in str(binding.attempt_package_dir)
    )
    terminal_path = m_binding.attempt_package_dir / "terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["outputs"]["pilot_evaluator_benchmark"] = "0" * 64
    _rewrite_hashed_manifest(terminal_path, terminal)
    with pytest.raises(ValueError, match="does not bind its evaluator summary"):
        create_p05_post_pilot_budget_decision(
            tmp_path / "wrong-summary-budget",
            pilot_bindings=bindings,
            evaluator_benchmark_package_dirs=evaluators,
        )

    bindings, evaluators = _inputs(tmp_path / "wrong-d03")
    m_binding = next(
        binding
        for binding in bindings
        if "P05-PILOT-M-XJTU" in str(binding.attempt_package_dir)
    )
    terminal_path = m_binding.attempt_package_dir / "terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["outputs"]["pilot_d03"] = "0" * 64
    _rewrite_hashed_manifest(terminal_path, terminal)
    with pytest.raises(ValueError, match="does not bind the summary D03 source"):
        create_p05_post_pilot_budget_decision(
            tmp_path / "wrong-d03-budget",
            pilot_bindings=bindings,
            evaluator_benchmark_package_dirs=evaluators,
        )

    bindings, evaluators = _inputs(tmp_path / "wrong-gpu")
    evaluator = next(path for path in evaluators if "CWRU" in path.name)
    summary_path = evaluator / "manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["shared_provenance"]["device_uuid"] = "GPU-WRONG"
    summary_semantic = _rewrite_hashed_manifest(summary_path, summary)
    m_binding = next(
        binding
        for binding in bindings
        if "P05-PILOT-M-CWRU" in str(binding.attempt_package_dir)
    )
    terminal_path = m_binding.attempt_package_dir / "terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["outputs"]["pilot_evaluator_benchmark"] = summary_semantic
    _rewrite_hashed_manifest(terminal_path, terminal)
    with pytest.raises(ValueError, match="blocked GPU identity"):
        create_p05_post_pilot_budget_decision(
            tmp_path / "wrong-gpu-budget",
            pilot_bindings=bindings,
            evaluator_benchmark_package_dirs=evaluators,
        )


def test_budget_rejects_run_preprocessing_hash_tamper(tmp_path) -> None:
    bindings, evaluators = _inputs(tmp_path / "inputs")
    binding = bindings[0]
    run_path = binding.run_contract_manifest_path
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["normalization_plan"]["sha256"] = "0" * 64
    run_semantic = _rewrite_hashed_manifest(run_path, run)
    terminal_path = binding.attempt_package_dir / "terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["outputs"]["run_contract"] = run_semantic
    _rewrite_hashed_manifest(terminal_path, terminal)

    with pytest.raises(ValueError, match="normalization/weight-plan hashes differ"):
        create_p05_post_pilot_budget_decision(
            tmp_path / "budget",
            pilot_bindings=bindings,
            evaluator_benchmark_package_dirs=evaluators,
        )


def test_budget_verifier_rejects_rehashed_promotion_and_formula_drift(tmp_path) -> None:
    promoted = _create(tmp_path / "promoted")
    payload = json.loads(promoted.manifest_path.read_text(encoding="utf-8"))
    payload["conclusion_control"]["claim_decision"] = "pass"
    semantic = {key: value for key, value in payload.items() if key != "content"}
    payload["content"]["semantic_sha256"] = _canonical_hash(semantic)
    promoted.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="conclusion control"):
        verify_p05_post_pilot_budget_decision(promoted.package_dir)

    drifted = _create(tmp_path / "drifted")
    payload = json.loads(drifted.manifest_path.read_text(encoding="utf-8"))
    component = payload["evaluation_forecast"]["per_dataset"]["CWRU"][
        "programs"
    ]["central_mandatory"]["components"]["central_original"]
    component["final_forward_window_multiplier"] += 1
    semantic = {key: value for key, value in payload.items() if key != "content"}
    payload["content"]["semantic_sha256"] = _canonical_hash(semantic)
    drifted.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation forecast"):
        verify_p05_post_pilot_budget_decision(drifted.package_dir)


@pytest.mark.parametrize(
    "invalid_content",
    [
        '{"schema_name":"a","schema_name":"b"}',
        '{"schema_name":"a","value":NaN}',
        '{"schema_name":"a","value":Infinity}',
        '{"schema_name":"a","value":-Infinity}',
    ],
)
def test_budget_verifier_rejects_duplicate_keys_and_nan(
    tmp_path, invalid_content
) -> None:
    package = tmp_path / _digest(invalid_content)[:8]
    package.mkdir()
    (package / "manifest.json").write_text(invalid_content, encoding="utf-8")
    with pytest.raises(ValueError, match="strict finite JSON"):
        verify_p05_post_pilot_budget_decision(package)
