#!/usr/bin/env python3
"""Create one hash-bound, create-only P05 neural tuning or decisive job package."""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import sys
import tempfile
from collections.abc import Mapping
from itertools import product
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config_schema import ExperimentConfig  # noqa: E402
from src.configs.p05_contract import validate_p05_experiment_contract  # noqa: E402


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
CONFIG_NAME = "config.yaml"
MANIFEST_NAME = "manifest.json"
COMMAND_PREFIX = ("conda", "run", "-n", "LQ_signal", "python")
ARMS = ("P05-M", "P05-B0", "P05-B1", "P05-B3")
DATASETS = ("CWRU", "XJTU")
LEARNING_RATES = (0.001, 0.0003)
DECISIVE_SEEDS = (42, 123, 456, 789, 1024)
PILOT_SHA256 = "2920bf24053579c0fbe192f780c07f0c2d59401be84a5b5c8c4bae6a3b3fb138"
PROTOCOL_BUNDLE_SHA256 = (
    "8d01361c39a778d437ce235ad1e8d3877313f128d6593fbb74812a4b237a1654"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _real_file(path: str | Path, *, name: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    if candidate.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"{name} does not exist: {candidate}") from exc
    if not resolved.is_file():
        raise ValueError(f"{name} must be a real file: {resolved}")
    return resolved


def _load_yaml(path: Path, *, name: str) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one YAML mapping")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains forbidden constant {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _required_gpu_uuid(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("GPU-")
        or len(value) > 128
        or any(character.isspace() or not character.isprintable() for character in value)
        or "REQUIRED" in value
    ):
        raise ValueError("gpu_uuid must be an observed printable NVIDIA GPU-* UUID")
    return value


def _is_exact_number(value: Any, expected: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) == expected
    )


def _arm_configs(*, trace_for_m: bool) -> dict[str, Any]:
    return {
        "P05-M": {
            "role": "proposed_fuzzy_head",
            "config": {
                "model": {
                    "uxfd": {
                        "fuzzy": {"enable": True},
                        "neural_residual": {"enable": False},
                        "anfis": {"enable": False},
                    }
                },
                "task": {
                    "p05_arm_id": "P05-M",
                    "p05_trace_export": trace_for_m,
                },
            },
        },
        "P05-B0": {
            "role": "same_backbone_no_fuzzy_head",
            "config": {
                "model": {
                    "uxfd": {
                        "fuzzy": {"enable": False},
                        "neural_residual": {"enable": False},
                        "anfis": {"enable": False},
                    }
                },
                "task": {
                    "p05_arm_id": "P05-B0",
                    "p05_trace_export": False,
                },
            },
        },
        "P05-B1": {
            "role": "parameter_matched_neural_residual",
            "config": {
                "model": {
                    "uxfd": {
                        "fuzzy": {"enable": False},
                        "neural_residual": {"enable": True},
                        "anfis": {"enable": False},
                    }
                },
                "task": {
                    "p05_arm_id": "P05-B1",
                    "p05_trace_export": False,
                },
            },
            "dataset_config": {
                "CWRU": {
                    "model": {"uxfd": {"neural_residual": {"hidden_dim": 26}}}
                },
                "XJTU": {
                    "model": {"uxfd": {"neural_residual": {"hidden_dim": 29}}}
                },
            },
        },
        "P05-B3": {
            "role": "anfis_complete_logits_head",
            "config": {
                "model": {
                    "uxfd": {
                        "fuzzy": {"enable": False},
                        "neural_residual": {"enable": False},
                        "anfis": {
                            "enable": True,
                            "num_features": 8,
                            "num_membership_functions": 3,
                            "num_rules": 10,
                            "antecedent_temperature": 1.0,
                            "min_width": 0.0001,
                            "firing_epsilon": 1.0e-12,
                        },
                    }
                },
                "task": {
                    "p05_arm_id": "P05-B3",
                    "p05_trace_export": False,
                },
            },
        },
    }


def _expected_stage_contract(stage: str) -> dict[str, Any]:
    if stage == "tuning":
        return {
            "phase": "tuning",
            "factors": {
                "arms": list(ARMS),
                "datasets": list(DATASETS),
                "learning_rates": list(LEARNING_RATES),
            },
            "job_count": 16,
            "seed": 20260801,
            "environment_stage": "fit_validate_only",
            "maximum_epochs": 60,
            "early_stopping": True,
            "patience": 10,
            "optimizer": "adam",
            "weight_decay": 0.0001,
            "scheduler": None,
            "checkpoint_selection": "minimum_validation_loss",
            "learning_rate_selection": {
                "primary": "maximum_checkpoint_validation_macro_f1",
                "tie_tolerance": 0.0001,
                "tie_breakers": [
                    "lower_validation_loss",
                    "lower_learning_rate",
                ],
            },
            "test_dataset_construction": "forbidden",
            "test_cache_access": "forbidden",
            "test_metric_access": "forbidden",
            "trace_export": "false_for_every_arm",
        }
    return {
        "phase": "decisive",
        "factors": {
            "arms": list(ARMS),
            "datasets": list(DATASETS),
            "seeds": list(DECISIVE_SEEDS),
        },
        "job_count": 40,
        "environment_stage": "fit_validate_test",
        "maximum_epochs": 100,
        "early_stopping": True,
        "patience": 15,
        "optimizer": "adam",
        "weight_decay": 0.0001,
        "scheduler": None,
        "checkpoint_selection": "minimum_validation_loss",
        "test_evaluation": "exactly_once_per_frozen_checkpoint",
        "learning_rate": {
            "source": "bound_hash_verified_tuning_selection_manifest_only",
            "allowed_values": list(LEARNING_RATES),
            "matrix_job_field": "forbidden",
        },
        "tuning_selection_manifest": {
            "schema_name": "p05.tuning_selection",
            "schema_version": 1,
            "path_binding": "required",
            "sha256_binding": "required",
            "selection_index_key_format": "DATASET/ARM",
            "required_selection_count": 8,
        },
        "trace_export": {
            "P05-M": True,
            "P05-B0": False,
            "P05-B1": False,
            "P05-B3": False,
        },
    }


def _validate_matrix(matrix: Mapping[str, Any], *, stage: str) -> None:
    if stage not in {"tuning", "decisive"}:
        raise ValueError("stage must be tuning or decisive")
    expected_top = {
        "schema_version",
        "kind",
        "paper_id",
        "protocol_id",
        "matrix_id",
        "status",
        "pilot_common_contract",
        "stage_contract",
        "runtime",
        "outputs",
        "arms",
        "datasets",
        "jobs",
        "execution_waves",
    }
    if set(matrix) != expected_top:
        raise ValueError(f"P05 {stage} matrix top-level schema drift")
    identity = {
        "tuning": (
            "p05_frozen_neural_tuning_execution_matrix",
            "P05-NEURAL-TUNING-v1",
            "frozen_declarative_awaiting_gpu_uuid_binding",
        ),
        "decisive": (
            "p05_frozen_neural_decisive_execution_matrix",
            "P05-NEURAL-DECISIVE-v1",
            "frozen_declarative_awaiting_gpu_uuid_and_tuning_selection_binding",
        ),
    }[stage]
    if matrix["schema_version"] != 1:
        raise ValueError(f"P05 {stage} matrix schema_version must be 1")
    if (
        matrix["kind"] != identity[0]
        or matrix["matrix_id"] != identity[1]
        or matrix["status"] != identity[2]
    ):
        raise ValueError(f"P05 {stage} matrix identity or status drift")
    if matrix["paper_id"] != "P05" or matrix["protocol_id"] != "P05-G040-v3.2":
        raise ValueError(f"P05 {stage} protocol identity drift")

    pilot = matrix["pilot_common_contract"]
    expected_pilot = {
        "path": "configs/experiments/p05/protocol/pilot_matrix_p05_v1.yaml",
        "sha256": PILOT_SHA256,
        "selector": "common_config",
        "reused_sections": [
            "common_config",
            "datasets.CWRU.config",
            "datasets.XJTU.config",
        ],
        "hash_mismatch": "hard_error",
    }
    if pilot != expected_pilot:
        raise ValueError(f"P05 {stage} pilot common-contract reference drift")
    if matrix["stage_contract"] != _expected_stage_contract(stage):
        raise ValueError(f"P05 {stage} frozen stage contract drift")

    expected_runtime = {
        "conda_environment": "LQ_signal",
        "command_prefix": list(COMMAND_PREFIX),
        "materializer": "scripts/materialize_p05_neural_job.py",
        "allowed_physical_gpu_indices": [0, 1],
        "forbidden_physical_gpu_indices": [2],
        "maximum_concurrent_processes_per_gpu": 1,
        "one_gpu_per_process": True,
        "distributed_execution": "forbidden",
        "gpu_uuid_binding": "required_at_materialization",
        "network_use": "forbidden",
        "automatic_download": "forbidden",
        "output_collision": "atomic_create_only_hard_error",
    }
    if matrix["runtime"] != expected_runtime:
        raise ValueError(f"P05 {stage} runtime contract drift")
    expected_outputs = {
        "materialization_status": "created-not-executed",
        "execution_status": "not_started",
        "evidence_status": "unadjudicated",
        "claim_support_before_ledger_and_audit": "forbidden",
    }
    if matrix["outputs"] != expected_outputs:
        raise ValueError(f"P05 {stage} output-state contract drift")
    if matrix["arms"] != _arm_configs(trace_for_m=stage == "decisive"):
        raise ValueError(f"P05 {stage} arm configuration drift")
    expected_datasets = {
        "CWRU": {
            "pilot_dataset_key": "CWRU",
            "dataset_id": 1,
            "num_classes": 4,
        },
        "XJTU": {
            "pilot_dataset_key": "XJTU",
            "dataset_id": 2,
            "num_classes": 2,
        },
    }
    if matrix["datasets"] != expected_datasets:
        raise ValueError(f"P05 {stage} dataset binding drift")

    jobs = matrix["jobs"]
    expected_count = 16 if stage == "tuning" else 40
    if not isinstance(jobs, list) or len(jobs) != expected_count:
        raise ValueError(f"P05 {stage} matrix must contain exactly {expected_count} jobs")
    if any(not isinstance(job, Mapping) for job in jobs):
        raise ValueError(f"P05 {stage} jobs must be mappings")
    ids = [job.get("id") for job in jobs]
    if len(set(ids)) != expected_count or any(
        not isinstance(job_id, str) or not job_id for job_id in ids
    ):
        raise ValueError(f"P05 {stage} job IDs must be unique non-empty strings")

    expected_cells: set[tuple[Any, ...]]
    observed_cells: set[tuple[Any, ...]] = set()
    for job in jobs:
        arm = job.get("arm")
        dataset = job.get("dataset")
        if arm not in ARMS or dataset not in DATASETS:
            raise ValueError(f"P05 {stage} job has an unregistered arm or dataset")
        short_arm = arm.removeprefix("P05-")
        if stage == "tuning":
            if set(job) != {
                "id",
                "arm",
                "dataset",
                "learning_rate",
                "seed",
                "output_dir",
                "materialize_command",
            }:
                raise ValueError("P05 tuning job schema drift")
            lr = job["learning_rate"]
            if not any(_is_exact_number(lr, candidate) for candidate in LEARNING_RATES):
                raise ValueError("P05 tuning learning rate must be 0.001 or 0.0003")
            if type(job["seed"]) is not int or job["seed"] != 20260801:
                raise ValueError("P05 tuning seed must be exactly 20260801")
            label = "LR1E3" if _is_exact_number(lr, 0.001) else "LR3E4"
            expected_id = f"P05-TUNE-{short_arm}-{dataset}-{label}"
            observed_cells.add((arm, dataset, float(lr)))
            expected_command = COMMAND_PREFIX + (
                "scripts/materialize_p05_neural_job.py",
                "--stage",
                "tuning",
                "--job-id",
                expected_id,
                "--gpu-uuid",
                "GPU_UUID_AT_LAUNCH",
                "--output-package",
                "CREATE_ONLY_PACKAGE_PATH",
            )
        else:
            if set(job) != {
                "id",
                "arm",
                "dataset",
                "seed",
                "tuning_selection_key",
                "output_dir",
                "materialize_command",
            }:
                raise ValueError(
                    "P05 decisive job schema forbids learning-rate fields"
                )
            seed = job["seed"]
            if type(seed) is not int or seed not in DECISIVE_SEEDS:
                raise ValueError("P05 decisive seed is not registered")
            if job["tuning_selection_key"] != f"{dataset}/{arm}":
                raise ValueError("P05 decisive tuning-selection key drift")
            expected_id = f"P05-DEC-{short_arm}-{dataset}-S{seed}"
            observed_cells.add((arm, dataset, seed))
            expected_command = COMMAND_PREFIX + (
                "scripts/materialize_p05_neural_job.py",
                "--stage",
                "decisive",
                "--job-id",
                expected_id,
                "--gpu-uuid",
                "GPU_UUID_AT_LAUNCH",
                "--tuning-selection-manifest",
                "TUNING_SELECTION_MANIFEST_PATH",
                "--tuning-selection-sha256",
                "TUNING_SELECTION_MANIFEST_SHA256",
                "--output-package",
                "CREATE_ONLY_PACKAGE_PATH",
            )
        if job["id"] != expected_id:
            raise ValueError(f"P05 {stage} job ID does not bind all frozen factors")
        expected_output = f"results/experiments/p05/{stage}/{expected_id}"
        if job["output_dir"] != expected_output:
            raise ValueError(f"P05 {stage} output directory drift")
        try:
            command = tuple(shlex.split(job["materialize_command"]))
        except ValueError as exc:
            raise ValueError(f"invalid P05 {stage} materialization command") from exc
        if command != expected_command:
            raise ValueError(
                f"P05 {stage} command must exactly start with and bind "
                "conda run -n LQ_signal python"
            )

    if stage == "tuning":
        expected_cells = set(product(ARMS, DATASETS, LEARNING_RATES))
    else:
        expected_cells = set(product(ARMS, DATASETS, DECISIVE_SEEDS))
    if observed_cells != expected_cells:
        raise ValueError(f"P05 {stage} jobs do not form the exact frozen factorial")

    waves = matrix["execution_waves"]
    if not isinstance(waves, list) or len(waves) != expected_count // 2:
        raise ValueError(f"P05 {stage} wave count drift")
    launched: list[str] = []
    for index, wave in enumerate(waves, start=1):
        if not isinstance(wave, Mapping) or set(wave) != {"wave", "concurrent_jobs"}:
            raise ValueError(f"P05 {stage} wave schema drift")
        if type(wave["wave"]) is not int or wave["wave"] != index:
            raise ValueError(f"P05 {stage} wave numbering drift")
        concurrent = wave["concurrent_jobs"]
        if not isinstance(concurrent, list) or len(concurrent) != 2:
            raise ValueError(f"P05 {stage} waves require exactly two processes")
        expected_pair = jobs[(index - 1) * 2 : index * 2]
        for assignment, expected_job, gpu_index in zip(
            concurrent, expected_pair, (0, 1), strict=True
        ):
            if not isinstance(assignment, Mapping) or set(assignment) != {
                "job_id",
                "physical_gpu_index",
            }:
                raise ValueError(f"P05 {stage} wave assignment schema drift")
            if assignment["job_id"] != expected_job["id"]:
                raise ValueError(f"P05 {stage} wave order does not bind the job list")
            if (
                type(assignment["physical_gpu_index"]) is not int
                or assignment["physical_gpu_index"] != gpu_index
            ):
                raise ValueError(
                    f"P05 {stage} waves permit one process each on GPU0 and GPU1"
                )
            launched.append(assignment["job_id"])
    if len(launched) != len(set(launched)) or set(launched) != set(ids):
        raise ValueError(f"P05 {stage} waves must cover every job exactly once")


def _load_pilot_contract(
    matrix: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path, str]:
    reference = matrix["pilot_common_contract"]
    pilot_path = _real_file(reference["path"], name="P05 frozen pilot matrix")
    observed_hash = _sha256_file(pilot_path)
    if observed_hash != reference["sha256"]:
        raise ValueError("P05 frozen pilot common-contract SHA-256 mismatch")
    pilot = _load_yaml(pilot_path, name="P05 frozen pilot matrix")
    if (
        pilot.get("matrix_id") != "P05-PILOT-v1"
        or pilot.get("protocol_id") != "P05-G040-v3.2"
        or pilot.get("status") != "frozen_declarative"
    ):
        raise ValueError("P05 frozen pilot matrix identity or status drift")
    common = pilot.get(reference["selector"])
    if not isinstance(common, dict):
        raise ValueError("P05 frozen pilot common_config is missing")
    pilot_datasets = pilot.get("datasets")
    if not isinstance(pilot_datasets, Mapping):
        raise ValueError("P05 frozen pilot datasets are missing")
    configs: dict[str, dict[str, Any]] = {}
    for dataset_name, expected_id, expected_classes in (
        ("CWRU", 1, 4),
        ("XJTU", 2, 2),
    ):
        dataset = pilot_datasets.get(dataset_name)
        if (
            not isinstance(dataset, Mapping)
            or dataset.get("dataset_id") != expected_id
            or dataset.get("num_classes") != expected_classes
            or not isinstance(dataset.get("config"), dict)
        ):
            raise ValueError(f"P05 pilot {dataset_name} dataset contract drift")
        configs[dataset_name] = copy.deepcopy(dataset["config"])
    return copy.deepcopy(common), configs, pilot_path, observed_hash


def _selected_job(
    matrix: Mapping[str, Any], *, job_id: str
) -> Mapping[str, Any]:
    selected = [job for job in matrix["jobs"] if job["id"] == job_id]
    if len(selected) != 1:
        raise ValueError(f"job_id must identify exactly one P05 neural job: {job_id!r}")
    return selected[0]


def _physical_gpu_index(matrix: Mapping[str, Any], *, job_id: str) -> int:
    matches = [
        assignment["physical_gpu_index"]
        for wave in matrix["execution_waves"]
        for assignment in wave["concurrent_jobs"]
        if assignment["job_id"] == job_id
    ]
    if len(matches) != 1 or type(matches[0]) is not int or matches[0] not in {0, 1}:
        raise ValueError("P05 neural job must have one GPU0/GPU1 wave assignment")
    return matches[0]


def _verified_selection(
    *,
    manifest_path: str | Path,
    expected_sha256: str,
    job: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(expected_sha256, str)
        or _SHA256_PATTERN.fullmatch(expected_sha256) is None
    ):
        raise ValueError("tuning_selection_sha256 must be one hexadecimal SHA-256")
    selection_path = _real_file(
        manifest_path,
        name="P05 tuning-selection manifest",
    )
    observed_hash = _sha256_file(selection_path)
    if observed_hash != expected_sha256.lower():
        raise ValueError("P05 tuning-selection manifest SHA-256 mismatch")
    manifest = _load_json(selection_path, name="P05 tuning-selection manifest")
    if (
        manifest.get("schema_name") != "p05.tuning_selection"
        or manifest.get("schema_version") != 1
    ):
        raise ValueError("P05 tuning-selection manifest schema drift")
    if (
        manifest.get("paper_id") != "P05"
        or manifest.get("phase") != "tuning_selection"
        or manifest.get("status") != "computed_unadjudicated"
        or manifest.get("claim_decision") != "not_performed"
        or manifest.get("evidence_eligible") is not False
        or manifest.get("test_access") != "forbidden_and_not_performed"
    ):
        raise ValueError("P05 tuning-selection scientific state drift")
    if manifest.get("protocol_bundle_sha256") != PROTOCOL_BUNDLE_SHA256:
        raise ValueError("P05 tuning-selection protocol-bundle SHA-256 mismatch")

    tuning_matrix_path = _real_file(
        TUNING_MATRIX_PATH,
        name="P05 neural tuning matrix",
    )
    tuning_matrix = _load_yaml(tuning_matrix_path, name="P05 neural tuning matrix")
    _validate_matrix(tuning_matrix, stage="tuning")
    tuning_matrix_hash = _sha256_file(tuning_matrix_path)
    if manifest.get("source_matrix_sha256") != tuning_matrix_hash:
        raise ValueError("P05 tuning-selection source matrix SHA-256 mismatch")

    selections = manifest.get("selections")
    selection_index = manifest.get("selection_index")
    expected_keys = {f"{dataset}/{arm}" for dataset in DATASETS for arm in ARMS}
    if not isinstance(selections, list) or len(selections) != 8:
        raise ValueError("P05 tuning-selection manifest requires exactly eight rows")
    if not isinstance(selection_index, Mapping) or set(selection_index) != expected_keys:
        raise ValueError("P05 tuning-selection index must cover dataset x arm exactly")

    rows: dict[str, Mapping[str, Any]] = {}
    tuning_jobs = {entry["id"]: entry for entry in tuning_matrix["jobs"]}
    required_hashes = (
        "selected_checkpoint_sha256",
        "selected_config_sha256",
        "selected_code_sha256",
        "selected_run_contract_sha256",
        "source_candidate_semantic_sha256",
    )
    for row_number, row in enumerate(selections):
        if not isinstance(row, Mapping):
            raise ValueError("P05 tuning-selection rows must be mappings")
        required_row_fields = {
            "selection_id",
            "arm_id",
            "dataset",
            "dataset_id",
            "selected_learning_rate",
            "selected_job_id",
            "selected_checkpoint_epoch",
            "selected_val_f1_macro",
            "selected_val_loss",
            "selection_reason",
            *required_hashes,
        }
        if not required_row_fields.issubset(row):
            raise ValueError("P05 tuning-selection row schema drift")
        arm = row.get("arm_id")
        dataset = row.get("dataset")
        key = f"{dataset}/{arm}"
        if arm not in ARMS or dataset not in DATASETS or key in rows:
            raise ValueError("P05 tuning-selection rows must uniquely cover dataset x arm")
        expected_dataset_id = 1 if dataset == "CWRU" else 2
        if row.get("dataset_id") != expected_dataset_id:
            raise ValueError("P05 tuning-selection dataset ID drift")
        lr = row.get("selected_learning_rate")
        if not any(_is_exact_number(lr, value) for value in LEARNING_RATES):
            raise ValueError("P05 tuning-selection learning rate is not registered")
        selected_job_id = row.get("selected_job_id")
        tuning_job = tuning_jobs.get(selected_job_id)
        if (
            not isinstance(tuning_job, Mapping)
            or tuning_job.get("arm") != arm
            or tuning_job.get("dataset") != dataset
            or float(tuning_job.get("learning_rate")) != float(lr)
        ):
            raise ValueError("P05 tuning-selection row does not bind a matching tuning job")
        for field in required_hashes:
            value = row.get(field)
            if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError(f"P05 tuning-selection {field} must be a SHA-256")
        if (
            type(row.get("selected_checkpoint_epoch")) is not int
            or row["selected_checkpoint_epoch"] < 0
        ):
            raise ValueError("P05 tuning-selection checkpoint epoch must be non-negative")
        if not isinstance(row.get("selection_id"), str) or not row["selection_id"]:
            raise ValueError("P05 tuning-selection selection_id must be non-empty")
        if not isinstance(row.get("selection_reason"), str) or not row["selection_reason"]:
            raise ValueError("P05 tuning-selection reason must be non-empty")
        for field in ("selected_val_f1_macro", "selected_val_loss"):
            value = row.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"P05 tuning-selection {field} must be finite")
        index_entry = selection_index[key]
        if not isinstance(index_entry, Mapping):
            raise ValueError("P05 tuning-selection index entries must be mappings")
        if set(index_entry) != {
            "row_index",
            "selection_id",
            "selected_learning_rate",
            "selected_job_id",
            "selected_checkpoint_sha256",
            "selected_run_contract_sha256",
        }:
            raise ValueError("P05 tuning-selection index entry schema drift")
        cross_checks = {
            "row_index": row_number,
            "selection_id": row["selection_id"],
            "selected_learning_rate": lr,
            "selected_job_id": selected_job_id,
            "selected_checkpoint_sha256": row["selected_checkpoint_sha256"],
            "selected_run_contract_sha256": row["selected_run_contract_sha256"],
        }
        for field, expected in cross_checks.items():
            if index_entry.get(field) != expected:
                raise ValueError(
                    f"P05 tuning-selection index disagrees with row for {field}"
                )
        rows[key] = row
    if set(rows) != expected_keys:
        raise ValueError("P05 tuning-selection rows do not cover dataset x arm exactly")

    key = job["tuning_selection_key"]
    row = rows[key]
    return {
        "path": str(selection_path),
        "sha256": observed_hash,
        "source_matrix_sha256": tuning_matrix_hash,
        "key": key,
        "row_index": selection_index[key]["row_index"],
        "selected_learning_rate": float(row["selected_learning_rate"]),
        "selected_job_id": row["selected_job_id"],
        "selected_checkpoint_sha256": row["selected_checkpoint_sha256"],
        "selected_run_contract_sha256": row["selected_run_contract_sha256"],
    }


def _resolve_config(
    *,
    stage: str,
    matrix: Mapping[str, Any],
    job: Mapping[str, Any],
    common: Mapping[str, Any],
    dataset_configs: Mapping[str, Mapping[str, Any]],
    gpu_uuid: str,
    selection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    arm = matrix["arms"][job["arm"]]
    config = _deep_merge(common, dataset_configs[job["dataset"]])
    config = _deep_merge(config, arm["config"])
    dataset_specific = arm.get("dataset_config", {}).get(job["dataset"], {})
    config = _deep_merge(config, dataset_specific)
    learning_rate = (
        job["learning_rate"]
        if stage == "tuning"
        else selection["selected_learning_rate"] if selection is not None else None
    )
    epochs, patience, environment_stage = (
        (60, 10, "fit_validate_only")
        if stage == "tuning"
        else (100, 15, "fit_validate_test")
    )
    config = _deep_merge(
        config,
        {
            "environment": {
                "project": job["id"].lower().replace("-", "_"),
                "seed": job["seed"],
                "output_dir": job["output_dir"],
                "stage": environment_stage,
                "notes": (
                    f"P05 frozen neural {stage} job; materialized but not executed "
                    "or adjudicated."
                ),
            },
            "task": {
                "lr": learning_rate,
                "p05_run_phase": stage,
            },
            "trainer": {
                "p05_pilot_mode": False,
                "expected_gpu_uuid": _required_gpu_uuid(gpu_uuid),
                "num_epochs": epochs,
                "early_stopping": True,
                "patience": patience,
            },
        },
    )
    ExperimentConfig.model_validate(config, strict=True)
    contract = validate_p05_experiment_contract(
        config["environment"],
        config["data"],
        config["model"],
        config["task"],
        config["trainer"],
        object(),
    )
    if (
        contract is None
        or contract.arm_id != job["arm"]
        or contract.dataset != job["dataset"]
        or contract.phase != stage
        or contract.seed != job["seed"]
    ):
        raise ValueError("resolved P05 neural config failed frozen contract binding")
    return config


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic P05 neural materialization requires Linux renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), str(target))
    raise OSError(error_number, os.strerror(error_number), str(target))


def materialize_p05_neural_job(
    *,
    stage: str,
    job_id: str,
    gpu_uuid: str,
    output_package: str | Path,
    matrix_path: str | Path | None = None,
    tuning_selection_manifest: str | Path | None = None,
    tuning_selection_sha256: str | None = None,
) -> dict[str, Any]:
    """Materialize one P05 job without running it or adjudicating evidence."""

    if stage not in {"tuning", "decisive"}:
        raise ValueError("stage must be tuning or decisive")
    target = Path(os.path.abspath(os.fspath(output_package)))
    if target.is_symlink() or target.exists():
        raise FileExistsError(f"P05 neural materialization target already exists: {target}")

    selected_matrix = (
        Path(matrix_path)
        if matrix_path is not None
        else TUNING_MATRIX_PATH if stage == "tuning" else DECISIVE_MATRIX_PATH
    )
    matrix_file = _real_file(selected_matrix, name=f"P05 {stage} matrix")
    matrix = _load_yaml(matrix_file, name=f"P05 {stage} matrix")
    _validate_matrix(matrix, stage=stage)
    common, dataset_configs, pilot_path, pilot_hash = _load_pilot_contract(matrix)
    job = _selected_job(matrix, job_id=job_id)
    physical_index = _physical_gpu_index(matrix, job_id=job_id)
    verified_uuid = _required_gpu_uuid(gpu_uuid)

    selection: dict[str, Any] | None = None
    if stage == "tuning":
        if tuning_selection_manifest is not None or tuning_selection_sha256 is not None:
            raise ValueError("P05 tuning materialization forbids selection-manifest bindings")
    else:
        if tuning_selection_manifest is None or tuning_selection_sha256 is None:
            raise ValueError(
                "P05 decisive materialization requires selection path and SHA-256"
            )
        selection = _verified_selection(
            manifest_path=tuning_selection_manifest,
            expected_sha256=tuning_selection_sha256,
            job=job,
        )

    config = _resolve_config(
        stage=stage,
        matrix=matrix,
        job=job,
        common=common,
        dataset_configs=dataset_configs,
        gpu_uuid=verified_uuid,
        selection=selection,
    )
    config_bytes = yaml.safe_dump(
        config,
        sort_keys=False,
        allow_unicode=False,
    ).encode("utf-8")
    config_hash = _sha256_bytes(config_bytes)
    semantic_manifest: dict[str, Any] = {
        "schema_name": "p05.materialized_neural_job",
        "schema_version": 1,
        "paper_id": "P05",
        "protocol_id": matrix["protocol_id"],
        "matrix_id": matrix["matrix_id"],
        "matrix_sha256": _sha256_file(matrix_file),
        "stage": stage,
        "job_id": job_id,
        "arm": job["arm"],
        "dataset": job["dataset"],
        "seed": job["seed"],
        "learning_rate": float(config["task"]["lr"]),
        "learning_rate_source": (
            "frozen_tuning_matrix_job"
            if stage == "tuning"
            else "bound_hash_verified_tuning_selection_manifest"
        ),
        "physical_gpu_index": physical_index,
        "expected_gpu_uuid": verified_uuid,
        "pilot_common_contract": {
            "path": str(pilot_path),
            "pilot_matrix_sha256": pilot_hash,
            "common_config_sha256": _sha256_bytes(_canonical_json_bytes(common)),
        },
        "tuning_selection": selection,
        "config_file": CONFIG_NAME,
        "config_sha256": config_hash,
        "materialization_status": "created-not-executed",
        "execution_status": "not_started",
        "evidence_status": "unadjudicated",
        "claim_support": "forbidden_before_ledger_and_audit",
        "scientific_overrides": "forbidden",
    }
    semantic_hash = _sha256_bytes(_canonical_json_bytes(semantic_manifest))
    manifest = {
        **semantic_manifest,
        "content": {"semantic_sha256": semantic_hash},
    }
    manifest_bytes = _pretty_json_bytes(manifest)

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"P05 neural materialization parent must be real: {parent}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(parent))
    )
    try:
        for name, payload in (
            (CONFIG_NAME, config_bytes),
            (MANIFEST_NAME, manifest_bytes),
        ):
            with (temporary / name).open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        _fsync_directory(temporary)
        _rename_directory_noreplace(temporary, target)
        _fsync_directory(parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    return {
        "status": "created-not-executed",
        "package_dir": str(target),
        "config_path": str(target / CONFIG_NAME),
        "config_sha256": config_hash,
        "manifest_path": str(target / MANIFEST_NAME),
        "manifest_sha256": _sha256_file(target / MANIFEST_NAME),
        "semantic_sha256": semantic_hash,
        "stage": stage,
        "job_id": job_id,
        "physical_gpu_index": physical_index,
        "learning_rate": float(config["task"]["lr"]),
        "evidence_status": "unadjudicated",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("tuning", "decisive"), required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output-package", required=True)
    parser.add_argument("--matrix-path")
    parser.add_argument("--tuning-selection-manifest")
    parser.add_argument("--tuning-selection-sha256")
    args = parser.parse_args()
    result = materialize_p05_neural_job(
        stage=args.stage,
        job_id=args.job_id,
        gpu_uuid=args.gpu_uuid,
        output_package=args.output_package,
        matrix_path=args.matrix_path,
        tuning_selection_manifest=args.tuning_selection_manifest,
        tuning_selection_sha256=args.tuning_selection_sha256,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
