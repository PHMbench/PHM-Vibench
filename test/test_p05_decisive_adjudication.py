from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.utils.p05_decisive_adjudication import (
    B4_FCM_INITIALISATION_SEED,
    CENTRAL_GPU_BUDGET_JOB_COUNT,
    DECISIVE_EXECUTION_ARTIFACT_COUNT,
    EXPECTED_JOB_IDS,
    build_p05_decisive_collection_manifest,
    expected_p05_decisive_job_ids,
)


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


def _token(value: str) -> str:
    return _hash_bytes(value.encode("utf-8"))


def _write_standard(
    path: Path,
    semantic: dict[str, Any],
    *,
    content_extra: dict[str, str] | None = None,
) -> tuple[str, str]:
    semantic_hash = _hash_bytes(_canonical(semantic))
    manifest = {
        **semantic,
        "content": {
            **(content_extra or {}),
            "semantic_sha256": semantic_hash,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return semantic_hash, _hash_file(path)


class _Artifacts:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.selection_path = root / "selection.json"
        self.selection_semantic, self.selection_direct = _write_standard(
            self.selection_path,
            {
                "schema_name": "p05.tuning_selection",
                "schema_version": 1,
                "paper_id": "P05",
                "phase": "tuning_selection",
                "status": "computed_unadjudicated",
                "claim_decision": "not_performed",
                "evidence_eligible": False,
                "test_access": "forbidden_and_not_performed",
            },
        )
        self.b0: dict[tuple[str, int], tuple[str, str]] = {}

    @staticmethod
    def _identity(job_id: str) -> tuple[str, str, int, int | None, str]:
        if job_id.startswith("P05-DEC-"):
            _p05, _dec, short_arm, dataset, seed_token = job_id.split("-")
            return f"P05-{short_arm}", dataset, 1 if dataset == "CWRU" else 2, int(seed_token[1:]), "neural"
        if job_id.startswith("P05-CPU-B2-"):
            _p05, _cpu, _b2, dataset, seed_token = job_id.split("-")
            return "P05-B2", dataset, 1 if dataset == "CWRU" else 2, int(seed_token[1:]), "b2"
        _p05, _cpu, _b4, dataset = job_id.split("-")
        return "P05-B4", dataset, 1 if dataset == "CWRU" else 2, None, "b4"

    def make(
        self,
        job_id: str,
        *,
        terminal_status: str = "completed",
        nan_evidence: bool = False,
        result_test_leak: bool = False,
        run_normalization_mismatch: bool = False,
    ) -> dict[str, Any]:
        arm, dataset, dataset_id, model_seed, kind = self._identity(job_id)
        attempt_seed = B4_FCM_INITIALISATION_SEED if kind == "b4" else int(model_seed)
        job_root = self.root / job_id
        materialized_dir = job_root / "materialized"
        materialized_dir.mkdir(parents=True)

        code_hash = _token("shared-code")
        config_snapshot_hash = _token(f"resolved-snapshot:{job_id}")
        provenance = {
            "source_metadata_sha256": _token("source-metadata"),
            "derived_metadata_sha256": _token("derived-metadata"),
            "signal_cache_manifest_sha256": _token("signal-cache"),
            "split_manifest_sha256": _token(f"split:{dataset}"),
            "config_snapshot_sha256": config_snapshot_hash,
            "code_snapshot_sha256": code_hash,
            "normalization_sha256": _token(f"normalization:{dataset}"),
            "train_weight_plan_sha256": _token(f"train-weights:{dataset}"),
            "validation_weight_plan_sha256": _token(f"validation-weights:{dataset}"),
        }

        if kind == "neural":
            raw_config = materialized_dir / "config.yaml"
            raw_config.write_text(f"job: {job_id}\nraw_order: true\n", encoding="utf-8")
            raw_config_hash = _hash_file(raw_config)
            assert raw_config_hash != config_snapshot_hash
            materialized_semantic = {
                "schema_name": "p05.materialized_neural_job",
                "schema_version": 1,
                "paper_id": "P05",
                "protocol_id": "P05-G040-v3.2",
                "matrix_id": "P05-NEURAL-DECISIVE-v1",
                "matrix_sha256": _hash_file(
                    Path(__file__).resolve().parents[1]
                    / "configs/experiments/p05/protocol/neural_decisive_matrix_p05_v1.yaml"
                ),
                "stage": "decisive",
                "job_id": job_id,
                "arm": arm,
                "dataset": dataset,
                "seed": model_seed,
                "learning_rate": 0.001,
                "learning_rate_source": "bound_hash_verified_tuning_selection_manifest",
                "physical_gpu_index": 0,
                "expected_gpu_uuid": "GPU-test-uuid",
                "tuning_selection": {
                    "path": str(self.selection_path),
                    "sha256": self.selection_direct,
                    "source_matrix_sha256": _token("tuning-matrix"),
                    "key": f"{dataset}/{arm}",
                    "row_index": 0,
                    "selected_learning_rate": 0.001,
                    "selected_job_id": f"P05-TUNE-{arm[4:]}-{dataset}-LR1E3",
                    "selected_checkpoint_sha256": _token(f"tune-ckpt:{dataset}:{arm}"),
                    "selected_run_contract_sha256": _token(f"tune-run:{dataset}:{arm}"),
                },
                "config_file": "config.yaml",
                "config_sha256": raw_config_hash,
                "materialization_status": "created-not-executed",
                "execution_status": "not_started",
                "evidence_status": "unadjudicated",
                "claim_support": "forbidden_before_ledger_and_audit",
            }
            materialized_path = materialized_dir / "manifest.json"
            materialized_hash, _ = _write_standard(materialized_path, materialized_semantic)
            runtime = {
                "physical_gpu_index": 0,
                "gpu_uuid": "GPU-test-uuid",
                "expected_gpu_uuid": "GPU-test-uuid",
                "accelerator": "cuda",
            }
            checkpoint_hash = _token(f"checkpoint:{job_id}")
            model_hash = _token(f"model:{job_id}")
            run_semantic = {
                "schema_name": "p05.run_artifact_bundle",
                "schema_version": 1,
                "paper_id": "P05",
                "dataset_id": dataset_id,
                "normalization_plan": {
                    "sha256": (
                        _token("mismatched-normalization")
                        if run_normalization_mismatch
                        else provenance["normalization_sha256"]
                    )
                },
                "weight_plans": {
                    "train": {"sha256": provenance["train_weight_plan_sha256"]},
                    "validation": {
                        "sha256": provenance["validation_weight_plan_sha256"]
                    },
                },
                "runtime_identity": runtime,
                "provenance": {
                    "checkpoint_sha256": checkpoint_hash,
                    "code_sha256": code_hash,
                    "config_sha256": config_snapshot_hash,
                    "model_sha256": model_hash,
                },
            }
            run_path = job_root / "run" / "manifest.json"
            run_hash, _ = _write_standard(run_path, run_semantic)
            evidence_dir = job_root / "evidence"
            evidence_dir.mkdir()
            if arm == "P05-M":
                arrays_path = evidence_dir / "evaluation_arrays.npz"
                np.savez(arrays_path, value=np.array([np.nan if nan_evidence else 1.0]))
                c3_path = evidence_dir / "c3.json"
                c3_path.write_text('{"status":"computed_unadjudicated"}\n', encoding="utf-8")
                trace_provenance = {
                    "checkpoint_sha256": checkpoint_hash,
                    "config_sha256": config_snapshot_hash,
                    "model_sha256": model_hash,
                }
                evidence_semantic = {
                    "schema_name": "p05.c2_c3_evaluation_bundle",
                    "schema_version": 2,
                    "conclusion_control": {
                        "claim_decisions": "not_performed",
                        "decisive": False,
                        "status": "computed_unadjudicated",
                        "predictive_cost_gate": "not_evaluated",
                        "operational_wording_gate": "not_evaluated",
                    },
                    "frozen_parameters": {"dataset": dataset, "model_seed": model_seed},
                    "inputs": {
                        "validation_trace": {"provenance": trace_provenance},
                        "evaluation_trace": {"provenance": trace_provenance},
                    },
                    "outputs": {
                        "arrays_file": arrays_path.name,
                        "c3_file": c3_path.name,
                    },
                }
                evidence_path = evidence_dir / "manifest.json"
                evidence_hash, _ = _write_standard(
                    evidence_path,
                    evidence_semantic,
                    content_extra={
                        "arrays_sha256": _hash_file(arrays_path),
                        "c3_sha256": _hash_file(c3_path),
                    },
                )
                evidence_output = "evaluation"
                evidence_column = "p05_evaluation_semantic_sha256"
            else:
                evidence_path, evidence_hash = self._prediction(
                    evidence_dir,
                    checkpoint_hash=checkpoint_hash,
                    config_hash=config_snapshot_hash,
                    code_hash=code_hash,
                    run_hash=run_hash,
                    nan_evidence=nan_evidence,
                )
                evidence_output = "predictions"
                evidence_column = "p05_prediction_semantic_sha256"
            result_path = job_root / "result.csv"
            result_row = {
                "materialized_job_id": job_id,
                "materialized_job_semantic_sha256": materialized_hash,
                "run_contract_semantic_sha256": run_hash,
                evidence_column: evidence_hash,
                "metric": "0.5",
            }
            if result_test_leak:
                result_row["selection_split"] = "test"
            self._csv(result_path, result_row)
            result_hash = _hash_file(result_path)
            outputs = {
                "all_results": _token(f"all:{job_id}"),
                "checkpoint": checkpoint_hash,
                "code_snapshot": code_hash,
                "config_snapshot": config_snapshot_hash,
                "materialized_job": materialized_hash,
                "result": result_hash,
                "run_contract": run_hash,
                evidence_output: evidence_hash,
            }
            if arm == "P05-M":
                outputs.update(
                    {
                        "diagnostics_test": _token(f"diag-test:{job_id}"),
                        "diagnostics_val": _token(f"diag-val:{job_id}"),
                        "trace_test": _token(f"trace-test:{job_id}"),
                        "trace_val": _token(f"trace-val:{job_id}"),
                    }
                )
            command = ["python", "main.py", "--config", str(raw_config)]
            if arm == "P05-B0":
                self.b0[(dataset, int(model_seed))] = (checkpoint_hash, run_hash)
        else:
            job_file = materialized_dir / "job.yaml"
            job_file.write_text(f"job_id: {job_id}\n", encoding="utf-8")
            assert _hash_file(job_file) != config_snapshot_hash
            materialized_semantic = {
                "schema_name": "p05.materialized_cpu_arm_job_manifest",
                "schema_version": 1,
                "paper_id": "P05",
                "protocol_id": "P05-G040-v3.2",
                "matrix_id": "P05-CPU-ARMS-v1",
                "job_id": job_id,
                "arm": arm,
                "dataset": dataset,
                "matrix_sha256": _hash_file(
                    Path(__file__).resolve().parents[1]
                    / "configs/experiments/p05/protocol/cpu_arm_matrix_p05_v1.yaml"
                ),
                "job_file": "job.yaml",
                "job_sha256": _hash_file(job_file),
                "materialization_status": "created_not_executed",
                "execution_status": "not_started",
                "evidence_status": "unadjudicated",
                "claim_support": "forbidden_before_ledger_and_audit",
            }
            materialized_path = materialized_dir / "manifest.json"
            materialized_hash, _ = _write_standard(materialized_path, materialized_semantic)
            run_path = None
            if kind == "b2":
                parent_checkpoint, parent_run = self.b0.get(
                    (dataset, int(model_seed)),
                    (_token(f"parent-checkpoint:{job_id}"), _token(f"parent-run:{job_id}")),
                )
                result_dir = job_root / "b2"
                result_dir.mkdir()
                checkpoint_path = result_dir / "checkpoint.npz"
                np.savez(checkpoint_path, weight=np.array([1.0], dtype=np.float32))
                checkpoint_hash = _hash_file(checkpoint_path)
                result_semantic = {
                    "schema_name": "p05.b2_posthoc_fuzzy_surrogate",
                    "schema_version": 1,
                    "paper_id": "P05",
                    "baseline_id": "P05-B2",
                    "evidence_status": "unadjudicated",
                    "model": {"num_classes": 4 if dataset == "CWRU" else 2},
                    "provenance": {
                        "model_seed": model_seed,
                        "b0_checkpoint_sha256": parent_checkpoint,
                        "b0_run_artifact_semantic_sha256": parent_run,
                    },
                    "checkpoint": {"file": checkpoint_path.name},
                }
                result_path = result_dir / "manifest.json"
                result_hash, _ = _write_standard(
                    result_path,
                    result_semantic,
                    content_extra={"checkpoint_sha256": checkpoint_hash},
                )
                evidence_path, evidence_hash = self._prediction(
                    job_root / "predictions",
                    checkpoint_hash=checkpoint_hash,
                    config_hash=config_snapshot_hash,
                    code_hash=code_hash,
                    run_hash=_token(f"cpu-run:{job_id}"),
                    nan_evidence=nan_evidence,
                )
                outputs = {
                    "checkpoint": checkpoint_hash,
                    "code_snapshot": code_hash,
                    "config_snapshot": config_snapshot_hash,
                    "materialized_job": materialized_hash,
                    "predictions": evidence_hash,
                    "result": result_hash,
                }
            else:
                result_dir = job_root / "b4"
                result_dir.mkdir()
                model_path = result_dir / "model.npz"
                predictions_path = result_dir / "predictions.npz"
                np.savez(model_path, center=np.array([1.0]))
                np.savez(
                    predictions_path,
                    score=np.array([np.nan if nan_evidence else 1.0]),
                )
                b4_semantic = {
                    "schema_name": "p05.b4_classical_fuzzy",
                    "schema_version": 1,
                    "paper_id": "P05",
                    "baseline_id": "P05-B4",
                    "evidence_status": "unadjudicated",
                    "dataset_id": dataset_id,
                    "fit_id": f"P05-B4-dataset-{dataset_id}",
                    "fit_contract": {
                        "fits_per_dataset": 1,
                        "model_seed_repetition": "forbidden_as_redundant_deterministic_fit",
                        "clustering": {
                            "initialization_seed": B4_FCM_INITIALISATION_SEED
                        },
                    },
                    "provenance": {
                        "predictions": {
                            "train": {"sample_count": 1},
                            "validation": {"sample_count": 1},
                            "test": {"sample_count": 1},
                        }
                    },
                }
                result_hash = _hash_bytes(_canonical(b4_semantic))
                b4_manifest = {
                    **b4_semantic,
                    "semantic_sha256": result_hash,
                    "files": {
                        model_path.name: _hash_file(model_path),
                        predictions_path.name: _hash_file(predictions_path),
                    },
                }
                result_path = result_dir / "manifest.json"
                result_path.write_text(
                    json.dumps(b4_manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                evidence_path = result_path
                evidence_hash = result_hash
                checkpoint_hash = _hash_file(model_path)
                outputs = {
                    "code_snapshot": code_hash,
                    "config_snapshot": config_snapshot_hash,
                    "materialized_job": materialized_hash,
                    "predictions": result_hash,
                    "result": result_hash,
                }
            command = ["python", "cpu_runner.py", "--job", str(job_file)]

        attempt_dir = job_root / "attempt"
        (attempt_dir / "invalidations").mkdir(parents=True)
        phase = "decisive" if kind == "neural" else "cpu_baseline"
        runtime = runtime if kind == "neural" else {"accelerator": "cpu"}
        start_semantic = {
            "schema_name": "p05.experiment_attempt",
            "schema_version": 1,
            "paper_id": "P05",
            "attempt": {
                "attempt_id": f"attempt.{job_id}",
                "arm_id": arm,
                "phase": phase,
                "dataset_id": dataset_id,
                "seed": attempt_seed,
                "status": "running",
            },
            "execution": {
                "command_argv": command,
                "working_directory": str(self.root),
                "device_identity": runtime,
            },
            "provenance": provenance,
            "unavailable_reasons": {},
            "retry": {"retry_of_start_semantic_sha256": None, "reason": None},
        }
        start_hash, _ = _write_standard(attempt_dir / "start.json", start_semantic)
        terminal_semantic = {
            "schema_name": "p05.experiment_attempt",
            "schema_version": 1,
            "paper_id": "P05",
            "attempt_id": f"attempt.{job_id}",
            "start_semantic_sha256": start_hash,
            "terminal": {
                "status": terminal_status,
                "claim_decision": "not_performed",
            },
            "outputs": outputs,
            "missing_outputs": {},
            "failure": None,
        }
        _write_standard(attempt_dir / "terminal.json", terminal_semantic)
        return {
            "job_id": job_id,
            "attempt_package_dir": attempt_dir,
            "materialized_manifest_path": materialized_path,
            "run_manifest_path": run_path,
            "result_path": result_path,
            "evidence_manifest_path": evidence_path,
        }

    @staticmethod
    def _csv(path: Path, row: dict[str, str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _prediction(
        directory: Path,
        *,
        checkpoint_hash: str,
        config_hash: str,
        code_hash: str,
        run_hash: str,
        nan_evidence: bool,
    ) -> tuple[Path, str]:
        directory.mkdir(parents=True, exist_ok=True)
        arrays_path = directory / "prediction_arrays.npz"
        np.savez(arrays_path, logits=np.array([np.nan if nan_evidence else 1.0]))
        semantic = {
            "schema_name": "p05.window_predictions",
            "schema_version": 1,
            "paper_id": "P05",
            "arrays_file": arrays_path.name,
            "conclusion_control": {
                "claim_decisions": "not_performed",
                "decisive": False,
                "status": "unadjudicated",
            },
            "evidence_status": "unadjudicated",
            "splits": {"test": {"sample_count": 1}},
            "provenance": {
                "checkpoint_sha256": checkpoint_hash,
                "config_sha256": config_hash,
                "code_sha256": code_hash,
                "run_contract_sha256": run_hash,
            },
        }
        manifest_path = directory / "manifest.json"
        semantic_hash, _ = _write_standard(
            manifest_path,
            semantic,
            content_extra={"arrays_sha256": _hash_file(arrays_path)},
        )
        return manifest_path, semantic_hash


def test_registry_distinguishes_budget_60_from_decisive_52() -> None:
    assert expected_p05_decisive_job_ids() == EXPECTED_JOB_IDS
    assert CENTRAL_GPU_BUDGET_JOB_COUNT == 60
    assert DECISIVE_EXECUTION_ARTIFACT_COUNT == 52
    assert len(EXPECTED_JOB_IDS) == len(set(EXPECTED_JOB_IDS)) == 52
    assert sum(job.startswith("P05-DEC-") for job in EXPECTED_JOB_IDS) == 40
    assert sum(job.startswith("P05-CPU-B2-") for job in EXPECTED_JOB_IDS) == 10
    b4 = [job for job in EXPECTED_JOB_IDS if job.startswith("P05-CPU-B4-")]
    assert b4 == ["P05-CPU-B4-CWRU", "P05-CPU-B4-XJTU"]
    assert all("PILOT" not in job and "TUNE" not in job for job in EXPECTED_JOB_IDS)


def test_empty_collection_is_explicitly_incomplete_and_unadjudicated() -> None:
    result = build_p05_decisive_collection_manifest([])
    assert result.status == "collection_incomplete"
    assert result.collected_job_count == 0
    assert len(result.missing_job_ids) == 52
    assert result.manifest["conclusion_control"] == {
        "claim_decisions": "not_performed",
        "statistical_adjudication": "not_performed",
        "p_values": "not_computed",
        "evidence_status": "collection_incomplete",
        "positive_claim_support": "forbidden_before_separate_registered_adjudication",
    }
    semantic = {key: value for key, value in result.manifest.items() if key != "content"}
    assert _hash_bytes(_canonical(semantic)) == result.semantic_sha256


def test_exact_52_artifacts_become_computed_unadjudicated(tmp_path: Path) -> None:
    artifacts = _Artifacts(tmp_path)
    descriptors = [artifacts.make(job_id) for job_id in EXPECTED_JOB_IDS]
    result = build_p05_decisive_collection_manifest(descriptors)
    assert result.status == "computed_unadjudicated"
    assert result.collected_job_count == 52
    assert result.missing_job_ids == ()
    records = result.manifest["collection"]["records"]
    assert sum(record["kind"] == "neural" for record in records) == 40
    assert sum(record["kind"] == "b2" for record in records) == 10
    b4 = [record for record in records if record["kind"] == "b4"]
    assert len(b4) == 2
    assert all(record["model_seed"] is None for record in b4)
    assert all(record["attempt_seed"] == B4_FCM_INITIALISATION_SEED for record in b4)
    assert result.manifest["conclusion_control"]["p_values"] == "not_computed"


def test_valid_subset_stays_collection_incomplete(tmp_path: Path) -> None:
    descriptor = _Artifacts(tmp_path).make("P05-DEC-B1-CWRU-S42")
    result = build_p05_decisive_collection_manifest([descriptor])
    assert result.status == "collection_incomplete"
    assert result.collected_job_count == 1
    assert len(result.missing_job_ids) == 51


def test_duplicate_job_is_rejected(tmp_path: Path) -> None:
    descriptor = _Artifacts(tmp_path).make("P05-DEC-B1-CWRU-S42")
    with pytest.raises(ValueError, match="duplicate decisive job"):
        build_p05_decisive_collection_manifest([descriptor, descriptor])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"terminal_status": "failed"}, "only a completed attempt"),
        ({"nan_evidence": True}, "contains NaN or Inf"),
        ({"result_test_leak": True}, "test-fitted selection"),
        ({"run_normalization_mismatch": True}, "normalization hash differs"),
    ],
)
def test_failed_nonfinite_leaked_or_provenance_mismatch_is_rejected(
    tmp_path: Path,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    descriptor = _Artifacts(tmp_path).make("P05-DEC-B1-CWRU-S42", **kwargs)
    with pytest.raises(ValueError, match=message):
        build_p05_decisive_collection_manifest([descriptor])


def test_invalidation_and_hash_tamper_are_rejected(tmp_path: Path) -> None:
    descriptor = _Artifacts(tmp_path).make("P05-DEC-B1-CWRU-S42")
    invalidation = Path(descriptor["attempt_package_dir"]) / "invalidations" / "invalid.json"
    invalidation.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalidated attempt"):
        build_p05_decisive_collection_manifest([descriptor])

    invalidation.unlink()
    arrays = Path(descriptor["evidence_manifest_path"]).parent / "prediction_arrays.npz"
    arrays.write_bytes(arrays.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_p05_decisive_collection_manifest([descriptor])


def test_descriptor_is_exact_and_unknown_jobs_are_rejected(tmp_path: Path) -> None:
    descriptor = _Artifacts(tmp_path).make("P05-DEC-B1-CWRU-S42")
    incomplete = dict(descriptor)
    incomplete.pop("result_path")
    with pytest.raises(ValueError, match="descriptor field mismatch"):
        build_p05_decisive_collection_manifest([incomplete])
    unknown = dict(descriptor, job_id="P05-PILOT-M-CWRU")
    with pytest.raises(ValueError, match="unregistered decisive job_id"):
        build_p05_decisive_collection_manifest([unknown])
