from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml
from pydantic import ValidationError

from src.config_schema import ExperimentConfig
from src.configs.config_utils import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = (
    REPO_ROOT
    / "configs"
    / "experiments"
    / "p05"
    / "protocol"
    / "pilot_matrix_p05_v1.yaml"
)
LAUNCH_PLAN_PATH = MATRIX_PATH.with_name("pilot_launch_plan_p05_v1.yaml")
GPU_UUID_PLACEHOLDER = "__REQUIRED_GPU_UUID_AT_LAUNCH__"

EXPECTED_CONFIG_KEYS = {
    "pipeline",
    "environment",
    "data",
    "model",
    "task",
    "trainer",
}
EXPECTED_SECTION_KEYS = {
    "environment": {
        "PROJECT_HOME",
        "project",
        "seed",
        "output_dir",
        "iterations",
        "stage",
        "notes",
        "wandb",
        "swanlab",
    },
    "data": {
        "data_dir",
        "metadata_path",
        "allow_download",
        "cache_mode",
        "cache_path",
        "cache_manifest_path",
        "p05_evidence_mode",
        "batch_size",
        "window_size",
        "stride",
        "train_ratio",
        "num_window",
        "window_sampling_strategy",
        "normalization",
        "dtype",
        "num_workers",
        "drop_last_train",
        "noise_snr",
        "split_strategy",
        "split",
    },
    "model": {
        "type",
        "name",
        "in_dim",
        "out_dim",
        "in_channels",
        "out_channels",
        "scale",
        "skip_connection",
        "internal_instance_normalization",
        "device",
        "num_classes",
        "signal_processing_configs",
        "feature_extractor_configs",
        "uxfd",
    },
    "task": {
        "type",
        "name",
        "target_system_id",
        "loss",
        "metrics",
        "optimizer",
        "lr",
        "weight_decay",
        "scheduler",
        "p05_evidence_mode",
        "p05_run_phase",
        "p05_arm_id",
        "p05_trace_export",
        "sample_weight_key",
    },
    "trainer": {
        "name",
        "p05_evidence_mode",
        "p05_pilot_mode",
        "expected_gpu_uuid",
        "num_epochs",
        "device",
        "accelerator",
        "devices",
        "gpus",
        "num_nodes",
        "num_processes",
        "strategy",
        "precision",
        "deterministic",
        "monitor",
        "monitor_mode",
        "save_top_k",
        "early_stopping",
        "pruning",
        "log_every_n_steps",
    },
}


def _load_matrix() -> dict[str, Any]:
    value = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_job(matrix: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    arm = matrix["arms"][job["arm"]]
    dataset = matrix["datasets"][job["dataset"]]
    config = _deep_merge(matrix["common_config"], dataset["config"])
    config = _deep_merge(config, arm["config"])
    return _deep_merge(config, job["config"])


def _leaf_differences(
    left: Any,
    right: Any,
    path: tuple[str, ...] = (),
) -> set[tuple[str, ...]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: set[tuple[str, ...]] = set()
        for key in set(left) | set(right):
            if key not in left or key not in right:
                differences.add(path + (str(key),))
            else:
                differences.update(
                    _leaf_differences(left[key], right[key], path + (str(key),))
                )
        return differences
    return set() if left == right else {path}


def _assert_exact_resolved_shape(config: Mapping[str, Any]) -> None:
    assert set(config) == EXPECTED_CONFIG_KEYS
    for section, keys in EXPECTED_SECTION_KEYS.items():
        assert set(config[section]) == keys
    assert set(config["data"]["split"]) == {
        "strategy",
        "split_key",
        "group_key",
        "seed",
        "test_policy",
        "manifest_path",
    }
    assert set(config["model"]["signal_processing_configs"]) == {"layer1"}
    assert set(config["model"]["uxfd"]) == {
        "enable_sp2d",
        "operator_attention",
        "logic",
        "fuzzy",
    }
    assert set(config["model"]["uxfd"]["operator_attention"]) == {"enable"}
    assert set(config["model"]["uxfd"]["logic"]) == {"enable"}
    assert set(config["model"]["uxfd"]["fuzzy"]) == {
        "enable",
        "num_fuzzy_features",
        "num_membership_functions",
        "num_rules",
        "logit_scale",
        "antecedent_temperature",
        "min_width",
        "firing_epsilon",
    }


def test_pilot_matrix_is_exact_full_factorial_and_non_evidence() -> None:
    matrix = _load_matrix()

    assert set(matrix) == {
        "schema_version",
        "kind",
        "paper_id",
        "protocol_id",
        "matrix_id",
        "status",
        "evidence_eligible",
        "design",
        "launch_gate",
        "common_config",
        "arms",
        "datasets",
        "jobs",
    }
    assert matrix["schema_version"] == 1
    assert matrix["kind"] == "p05_frozen_pilot_matrix"
    assert matrix["protocol_id"] == "P05-G040-v3.2"
    assert matrix["status"] == "frozen_declarative"
    assert matrix["evidence_eligible"] is False
    assert matrix["design"]["type"] == "full_factorial_2x2_engineering_pilot"
    assert matrix["design"]["claim_support"] == "forbidden"
    assert matrix["design"]["replicate_level"] == (
        "one_unreplicated_engineering_pilot_per_cell"
    )

    expected_cells = set(product(("P05-B0", "P05-M"), ("CWRU", "XJTU")))
    jobs = matrix["jobs"]
    assert len(jobs) == 4
    assert {(job["arm"], job["dataset"]) for job in jobs} == expected_cells
    assert len({job["id"] for job in jobs}) == 4
    assert all(set(job) == {"id", "arm", "dataset", "config"} for job in jobs)


def test_each_pilot_job_strictly_loads_and_binds_frozen_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    matrix = _load_matrix()
    fixed = matrix["design"]["fixed_contract"]
    output_dirs = set()

    for job in matrix["jobs"]:
        config = _resolve_job(matrix, job)
        _assert_exact_resolved_shape(config)
        typed = ExperimentConfig.model_validate(config, strict=True)
        lightweight = load_config(config)
        dataset = matrix["datasets"][job["dataset"]]
        fuzzy_enabled = matrix["arms"][job["arm"]]["config"]["model"]["uxfd"][
            "fuzzy"
        ]["enable"]

        assert typed.pipeline == "Pipeline_05_Explainable_Fault_Diagnosis"
        assert typed.environment.seed == fixed["seed"] == 20260801
        assert typed.environment.iterations == 1
        assert typed.environment.stage == fixed["stage"] == "fit_validate_only"
        assert typed.data.batch_size == fixed["batch_size"] == 64
        assert typed.data.split is not None
        assert typed.data.split.strategy == "preassigned_metadata"
        assert typed.data.split.seed == 20260801
        assert config["data"]["num_window"] == dataset["windows_per_record"]
        assert config["model"]["num_classes"] == dataset["num_classes"]
        assert config["task"]["target_system_id"] == [dataset["dataset_id"]]
        assert config["model"]["uxfd"]["fuzzy"]["enable"] is fuzzy_enabled
        assert config["task"]["lr"] == fixed["learning_rate"] == 1.0e-3
        assert config["task"]["optimizer"] == fixed["optimizer"] == "adam"
        assert config["task"]["loss"] == "CE_weighted"
        assert config["task"]["scheduler"] is None
        assert config["trainer"]["num_epochs"] == fixed["epochs"] == 5
        assert config["trainer"]["early_stopping"] is False
        assert config["trainer"]["device"] == config["model"]["device"] == "cuda"
        assert config["trainer"]["gpus"] == config["trainer"]["devices"] == 1
        assert config["trainer"]["p05_evidence_mode"] is True
        assert config["trainer"]["p05_pilot_mode"] is True
        assert config["task"]["p05_evidence_mode"] is True
        assert config["task"]["p05_run_phase"] == "pilot"
        assert config["task"]["p05_arm_id"] == job["arm"]
        assert config["task"]["p05_trace_export"] is (job["arm"] == "P05-M")
        assert config["data"]["p05_evidence_mode"] is True
        assert config["trainer"]["expected_gpu_uuid"] == GPU_UUID_PLACEHOLDER
        assert lightweight.data.metadata_path == str(
            (REPO_ROOT / config["data"]["metadata_path"]).resolve()
        )
        assert lightweight.data.metadata_file == lightweight.data.metadata_path
        output_dirs.add(config["environment"]["output_dir"])

    assert len(output_dirs) == 4


def test_strict_schema_rejects_stringly_typed_frozen_integer() -> None:
    matrix = _load_matrix()
    config = _resolve_job(matrix, matrix["jobs"][0])
    config["data"]["batch_size"] = "64"

    with pytest.raises(ValidationError, match="batch_size"):
        ExperimentConfig.model_validate(config, strict=True)


def test_only_registered_factor_paths_differ_between_paired_jobs() -> None:
    matrix = _load_matrix()
    resolved = {
        (job["arm"], job["dataset"]): _resolve_job(matrix, job)
        for job in matrix["jobs"]
    }
    arm_difference_paths = {
        ("environment", "project"),
        ("environment", "output_dir"),
        ("model", "uxfd", "fuzzy", "enable"),
        ("task", "p05_arm_id"),
        ("task", "p05_trace_export"),
    }
    dataset_difference_paths = {
        ("environment", "project"),
        ("environment", "output_dir"),
        ("data", "num_window"),
        ("data", "split", "manifest_path"),
        ("model", "num_classes"),
        ("task", "target_system_id"),
    }

    for dataset in ("CWRU", "XJTU"):
        assert _leaf_differences(
            resolved[("P05-B0", dataset)],
            resolved[("P05-M", dataset)],
        ) == arm_difference_paths
    for arm in ("P05-B0", "P05-M"):
        assert _leaf_differences(
            resolved[(arm, "CWRU")],
            resolved[(arm, "XJTU")],
        ) == dataset_difference_paths


def test_launch_gate_remains_fail_closed_until_runtime_bindings_exist() -> None:
    matrix = _load_matrix()
    gate = matrix["launch_gate"]

    assert gate["status"] == "blocked"
    assert gate["launch_plan_path"].endswith("pilot_launch_plan_p05_v1.yaml")
    assert gate["expected_gpu_uuid_placeholder"] == GPU_UUID_PLACEHOLDER
    assert gate["required_external_environment"]["conda_environment"] == "LQ_signal"
    assert gate["satisfied_implementation_gates"] == [
        "registered_CUDA_epoch_timing_and_memory_collector_targeted_tests_pass"
    ]
    assert set(gate["forbidden_during_pilot"]) == {
        "test_dataset_construction",
        "test_cache_access",
        "test_metric_access",
        "scientific_key_overrides",
    }


def test_launch_plan_freezes_same_gpu_dataset_blocks_and_forbids_gpu2() -> None:
    plan = yaml.safe_load(LAUNCH_PLAN_PATH.read_text(encoding="utf-8"))

    assert plan["status"] == "frozen_awaiting_physical_gpu_uuid_binding"
    assert plan["evidence_eligible"] is False
    assert plan["claim_support"] == "forbidden"
    runtime = plan["runtime"]
    assert runtime["allowed_physical_gpu_indices"] == [0, 1]
    assert runtime["forbidden_physical_gpu_indices"] == [2]
    assert runtime["one_process_per_job"] is True
    assert runtime["one_gpu_per_process"] is True
    assert runtime["distributed_execution"] == "forbidden"

    blocks = plan["blocking"]["blocks"]
    assert blocks == [
        {
            "block_id": "P05-PILOT-BLOCK-CWRU-GPU0",
            "dataset": "CWRU",
            "physical_gpu_index": 0,
            "ordered_jobs": ["P05-PILOT-B0-CWRU", "P05-PILOT-M-CWRU"],
        },
        {
            "block_id": "P05-PILOT-BLOCK-XJTU-GPU1",
            "dataset": "XJTU",
            "physical_gpu_index": 1,
            "ordered_jobs": ["P05-PILOT-M-XJTU", "P05-PILOT-B0-XJTU"],
        },
    ]
    waves = plan["execution_waves"]
    assert [entry["wave"] for entry in waves] == [1, 2]
    assert all(len(entry["concurrent_jobs"]) == 2 for entry in waves)
    launched = [
        item["job_id"]
        for wave in waves
        for item in wave["concurrent_jobs"]
    ]
    assert len(launched) == len(set(launched)) == 4
