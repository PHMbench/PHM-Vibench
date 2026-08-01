"""Locked post-pilot GPU-hour forecast and retention decision for P05."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import yaml

from src.explain_factory.p05_pilot_evaluator_benchmark import (
    MANIFEST_NAME as EVALUATOR_MANIFEST_NAME,
    verify_p05_pilot_evaluator_benchmark,
)


SCHEMA_NAME = "p05.post_pilot_gpu_hour_budget_decision"
SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
SAFETY_FACTOR = 1.5
GPU_HOUR_CAP = 168.0
FINAL_SEED_COUNT = 5

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID = re.compile(r"^GPU-[!-~]+$")

_FROZEN_HASHES = {
    "ablation_protocol_sha256": (
        "ef0e45bf7a104b80644bd04eb22db5d87129fa59768b4c6be68b9244c2fdbfd1"
    ),
    "config_bridge_sha256": (
        "9c6f747bfe9b973e02a8c0887fc8a25b06bd12f53a13e4e72c9643d495ddf0e1"
    ),
    "experiment_plan_sha256": (
        "84a5fca1649eeb594e1b45cb292cae7035aa21317601709fb44f6b2361d2ae78"
    ),
    "pilot_launch_plan_sha256": (
        "a7a3b74b3decae5a6c0df34444db52e9b1ccf5078215c95572b9c52ecb757cfe"
    ),
    "pilot_matrix_sha256": (
        "2920bf24053579c0fbe192f780c07f0c2d59401be84a5b5c8c4bae6a3b3fb138"
    ),
    "protocol_bundle_sha256": (
        "8d01361c39a778d437ce235ad1e8d3877313f128d6593fbb74812a4b237a1654"
    ),
    "statistics_sha256": (
        "c0b9a0baedddbc8e6ee76c465b19b6c447b85f66123dfb65cdc9bb7460525ecb"
    ),
}

_JOBS = {
    "P05-PILOT-B0-CWRU": {
        "arm": "P05-B0",
        "dataset": "CWRU",
        "dataset_id": 1,
        "physical_gpu_index": 0,
    },
    "P05-PILOT-M-CWRU": {
        "arm": "P05-M",
        "dataset": "CWRU",
        "dataset_id": 1,
        "physical_gpu_index": 0,
    },
    "P05-PILOT-B0-XJTU": {
        "arm": "P05-B0",
        "dataset": "XJTU",
        "dataset_id": 2,
        "physical_gpu_index": 1,
    },
    "P05-PILOT-M-XJTU": {
        "arm": "P05-M",
        "dataset": "XJTU",
        "dataset_id": 2,
        "physical_gpu_index": 1,
    },
}

_TEST_WINDOWS = {"CWRU": 23 * 16, "XJTU": (6317 + 330) * 4}
_VALIDATION_WINDOWS = {"CWRU": 19 * 16, "XJTU": (1071 + 246) * 4}
_TRAINING_STAGES = {
    "pilot": {"jobs_per_dataset": 2, "maximum_epochs": 5},
    "tuning": {"jobs_per_dataset": 8, "maximum_epochs": 60},
    "decisive_central": {"jobs_per_dataset": 20, "maximum_epochs": 100},
    "retraining_ablations": {"jobs_per_dataset": 15, "maximum_epochs": 100},
}
_EVALUATION_COMPONENTS = {
    "central_mandatory": {
        "central_original": 4,
        "central_deletion": 10,
        "central_shuffle": 32,
    },
    "d03": {"d03_original": 1, "d03_noise": 32},
    "retraining_ablations": {
        "central_original": 3,
        "central_deletion": 10 + 5 + 20,
        "central_shuffle": 3 * 32,
    },
}
_BASE_PILOT_OUTPUTS = {
    "all_results",
    "checkpoint",
    "code_snapshot",
    "config_snapshot",
    "materialized_job",
    "pilot_timing",
    "result",
    "run_contract",
}
_M_PILOT_OUTPUTS = _BASE_PILOT_OUTPUTS | {
    "pilot_d03",
    "pilot_evaluator_benchmark",
    "trace_val",
}


@dataclass(frozen=True)
class P05PilotTimingBinding:
    """Four-way provenance binding for one pilot timing artifact."""

    timing_manifest_path: Path
    attempt_package_dir: Path
    materialized_job_manifest_path: Path
    run_contract_manifest_path: Path


@dataclass(frozen=True)
class P05PostPilotBudgetResult:
    package_dir: Path
    manifest_path: Path
    semantic_sha256: str
    manifest_sha256: str
    status: str
    selected_program: str | None


@dataclass(frozen=True)
class _HashedManifest:
    path: Path
    payload: Mapping[str, Any]
    semantic_sha256: str
    direct_sha256: str


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


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _positive_integer(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _absolute_path(value: Any, *, name: str) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(value)))
    except TypeError as exc:
        raise TypeError(f"{name} must be path-like") from exc


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a real file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{name} is not strict finite JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _hashed_manifest(path: Any, *, name: str) -> _HashedManifest:
    source = _absolute_path(path, name=name)
    payload = _strict_json(source, name=name)
    content = _mapping(payload.get("content"), name=f"{name}.content")
    if set(content) != {"semantic_sha256"}:
        raise ValueError(f"{name}.content must contain only semantic_sha256")
    semantic_sha256 = _required_sha256(
        content["semantic_sha256"], name=f"{name}.content.semantic_sha256"
    )
    semantic = {key: value for key, value in payload.items() if key != "content"}
    if semantic_sha256 != _sha256_bytes(_canonical_json_bytes(semantic)):
        raise ValueError(f"{name} semantic SHA-256 does not match")
    return _HashedManifest(
        path=source,
        payload=MappingProxyType(payload),
        semantic_sha256=semantic_sha256,
        direct_sha256=_sha256_file(source),
    )


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_yaml_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML mapping key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_yaml_mapping,
)


def _strict_yaml(path: Path, *, name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a real file: {path}")
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueSafeLoader)
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"{name} is not strict YAML") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must contain a YAML mapping")
    return value


def _validate_timing_manifest(value: _HashedManifest) -> Mapping[str, Any]:
    manifest = value.payload
    expected = {
        "artifact_class",
        "claim_support",
        "content",
        "cuda_device",
        "evidence_eligible",
        "measurement_status",
        "measurements",
        "paper_id",
        "schema_name",
        "schema_version",
        "timing_contract",
    }
    if set(manifest) != expected or (
        manifest.get("schema_name") != "p05.non_evidence_pilot_timing"
        or manifest.get("schema_version") != 1
        or manifest.get("paper_id") != "P05"
        or manifest.get("artifact_class") != "engineering_pilot_timing"
        or manifest.get("measurement_status") != "complete"
        or manifest.get("evidence_eligible") is not False
        or manifest.get("claim_support") != "forbidden"
        or manifest.get("cuda_device") != "cuda:0"
    ):
        raise ValueError("pilot timing schema or non-evidence CUDA status is invalid")
    contract = _mapping(manifest["timing_contract"], name="timing_contract")
    if (
        contract.get("expected_complete_epochs") != 5
        or contract.get("median_epoch_numbers") != [2, 3, 4, 5]
        or contract.get("epoch_end_includes_scheduled_validation") is not True
    ):
        raise ValueError("pilot timing does not bind five complete epochs")
    measurements = _mapping(manifest["measurements"], name="measurements")
    if set(measurements) != {
        "epoch_seconds_1_through_5",
        "median_epoch_seconds_2_through_5",
        "peak_allocated_memory",
        "peak_reserved_memory",
        "startup_seconds",
    }:
        raise ValueError("pilot timing measurements are incomplete")
    startup = _finite_nonnegative(
        measurements["startup_seconds"], name="startup_seconds"
    )
    epoch_seconds = measurements["epoch_seconds_1_through_5"]
    if not isinstance(epoch_seconds, list) or len(epoch_seconds) != 5:
        raise ValueError("pilot timing requires five epoch durations")
    epochs = [
        _finite_nonnegative(item, name=f"epoch_seconds[{index}]")
        for index, item in enumerate(epoch_seconds)
    ]
    if any(item <= 0.0 for item in epochs):
        raise ValueError("pilot epoch durations must be positive")
    median = _finite_nonnegative(
        measurements["median_epoch_seconds_2_through_5"],
        name="median_epoch_seconds_2_through_5",
    )
    if median != float(statistics.median(epochs[1:])):
        raise ValueError("pilot median does not equal epochs 2 through 5")
    allocated = measurements["peak_allocated_memory"]
    reserved = measurements["peak_reserved_memory"]
    if type(allocated) is not int or allocated < 0:
        raise ValueError("pilot peak allocated memory is invalid")
    if type(reserved) is not int or reserved < allocated:
        raise ValueError("pilot peak reserved memory is invalid")
    return {"median_epoch_seconds": median, "startup_seconds": startup}


def _validate_materialized_job(
    value: _HashedManifest,
) -> tuple[str, Mapping[str, Any], Path]:
    manifest = value.payload
    expected_keys = {
        "claim_support",
        "config_file",
        "config_sha256",
        "content",
        "evidence_eligible",
        "expected_gpu_uuid",
        "job_id",
        "launch_plan_sha256",
        "matrix_id",
        "matrix_sha256",
        "paper_id",
        "physical_gpu_index",
        "schema_name",
        "schema_version",
        "scientific_overrides",
    }
    if set(manifest) != expected_keys or (
        manifest.get("schema_name") != "p05.materialized_pilot_config"
        or manifest.get("schema_version") != 1
        or manifest.get("paper_id") != "P05"
        or manifest.get("matrix_id") != "P05-PILOT-v1"
        or manifest.get("evidence_eligible") is not False
        or manifest.get("claim_support") != "forbidden"
        or manifest.get("scientific_overrides") != "forbidden"
        or manifest.get("matrix_sha256") != _FROZEN_HASHES["pilot_matrix_sha256"]
        or manifest.get("launch_plan_sha256")
        != _FROZEN_HASHES["pilot_launch_plan_sha256"]
    ):
        raise ValueError("materialized pilot manifest differs from the frozen contract")
    job_id = manifest.get("job_id")
    if job_id not in _JOBS:
        raise ValueError("materialized pilot job_id is not registered")
    expected = _JOBS[job_id]
    if manifest.get("physical_gpu_index") != expected["physical_gpu_index"]:
        raise ValueError("materialized pilot physical GPU index is invalid")
    uuid = manifest.get("expected_gpu_uuid")
    if not isinstance(uuid, str) or _GPU_UUID.fullmatch(uuid) is None:
        raise ValueError("materialized pilot GPU UUID is invalid")
    if manifest.get("config_file") != "config.yaml":
        raise ValueError("materialized pilot config filename is invalid")
    config_path = value.path.parent / "config.yaml"
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("materialized pilot config file is missing")
    if _sha256_file(config_path) != _required_sha256(
        manifest.get("config_sha256"), name="materialized config_sha256"
    ):
        raise ValueError("materialized pilot config direct SHA-256 differs")
    if set(value.path.parent.iterdir()) != {value.path, config_path}:
        raise ValueError("materialized pilot package has unexpected content")
    config = _strict_yaml(config_path, name="materialized pilot config")
    environment = _mapping(config.get("environment"), name="config.environment")
    data = _mapping(config.get("data"), name="config.data")
    task = _mapping(config.get("task"), name="config.task")
    trainer = _mapping(config.get("trainer"), name="config.trainer")
    if (
        environment.get("seed") != 20260801
        or environment.get("iterations") != 1
        or environment.get("stage") != "fit_validate_only"
        or task.get("p05_arm_id") != expected["arm"]
        or task.get("p05_run_phase") != "pilot"
        or task.get("p05_evidence_mode") is not True
        or task.get("target_system_id") != [expected["dataset_id"]]
        or data.get("batch_size") != 64
        or data.get("window_size") != 4096
        or data.get("allow_download") is not False
        or trainer.get("p05_pilot_mode") is not True
        or trainer.get("num_epochs") != 5
        or trainer.get("early_stopping") is not False
        or trainer.get("device") != "cuda"
        or trainer.get("gpus") != 1
        or trainer.get("expected_gpu_uuid") != uuid
    ):
        raise ValueError("materialized pilot config differs from the frozen cell")
    return job_id, expected, config_path


def _validate_attempt(
    package_value: Any,
) -> tuple[_HashedManifest, _HashedManifest]:
    package = _absolute_path(package_value, name="attempt_package_dir")
    if package.is_symlink() or not package.is_dir():
        raise ValueError("attempt package must be a real directory")
    invalidations = package / "invalidations"
    if invalidations.is_symlink() or not invalidations.is_dir():
        raise ValueError("attempt invalidations registry is missing")
    if any(invalidations.iterdir()):
        raise ValueError("invalidated pilot attempts cannot enter the budget forecast")
    expected_entries = {package / "start.json", package / "terminal.json", invalidations}
    if set(package.iterdir()) != expected_entries:
        raise ValueError("attempt package has unexpected or incomplete content")
    start = _hashed_manifest(package / "start.json", name="attempt start")
    terminal = _hashed_manifest(package / "terminal.json", name="attempt terminal")
    if (
        start.payload.get("schema_name") != "p05.experiment_attempt"
        or start.payload.get("schema_version") != 1
        or start.payload.get("paper_id") != "P05"
        or terminal.payload.get("schema_name") != "p05.experiment_attempt"
        or terminal.payload.get("schema_version") != 1
        or terminal.payload.get("paper_id") != "P05"
    ):
        raise ValueError("attempt schema identity is invalid")
    if terminal.payload.get("start_semantic_sha256") != start.semantic_sha256:
        raise ValueError("attempt terminal does not bind its start semantic hash")
    terminal_state = _mapping(terminal.payload.get("terminal"), name="terminal")
    if (
        terminal_state.get("status") != "completed"
        or terminal_state.get("claim_decision") != "not_performed"
        or terminal.payload.get("failure") is not None
        or terminal.payload.get("missing_outputs") != {}
    ):
        raise ValueError("pilot attempt is not a completed claim-neutral attempt")
    if start.payload.get("unavailable_reasons") != {} or any(
        value is None
        for value in _mapping(
            start.payload.get("provenance"), name="attempt provenance"
        ).values()
    ):
        raise ValueError("completed pilot attempt has incomplete provenance")
    return start, terminal


def _validate_run_contract(value: _HashedManifest) -> Mapping[str, str]:
    manifest = value.payload
    expected = {
        "content",
        "dataset_id",
        "normalization_plan",
        "paper_id",
        "provenance",
        "runtime_identity",
        "schema_name",
        "schema_version",
        "weight_plans",
    }
    if set(manifest) != expected or (
        manifest.get("schema_name") != "p05.run_artifact_bundle"
        or manifest.get("schema_version") != 1
        or manifest.get("paper_id") != "P05"
        or manifest.get("dataset_id") not in {1, 2}
    ):
        raise ValueError("run-contract manifest schema is invalid")
    normalization = _mapping(
        manifest.get("normalization_plan"), name="run normalization_plan"
    )
    normalization_sha256 = _required_sha256(
        normalization.get("sha256"), name="run normalization_plan.sha256"
    )
    weight_plans = _mapping(manifest.get("weight_plans"), name="run weight_plans")
    if set(weight_plans) != {"train", "validation"}:
        raise ValueError("run-contract weight plans must contain train and validation")
    weight_sha256 = {
        role: _required_sha256(
            _mapping(
                weight_plans[role], name=f"run weight_plans.{role}"
            ).get("sha256"),
            name=f"run weight_plans.{role}.sha256",
        )
        for role in ("train", "validation")
    }
    provenance = _mapping(manifest.get("provenance"), name="run provenance")
    if set(provenance) != {
        "checkpoint_sha256",
        "code_sha256",
        "config_sha256",
        "model_sha256",
    }:
        raise ValueError("run-contract provenance is incomplete")
    for name, digest in provenance.items():
        _required_sha256(digest, name=f"run provenance {name}")
    runtime = _mapping(manifest.get("runtime_identity"), name="runtime identity")
    required_runtime = {
        "accelerator": "gpu",
        "deterministic": True,
        "devices": 1,
        "evidence_mode": True,
        "gpus": 1,
        "identity_source": "nvidia-smi:index,uuid",
        "paper_id": "P05",
        "precision": 32,
        "schema_version": 1,
        "strategy": "auto",
    }
    if any(runtime.get(name) != expected for name, expected in required_runtime.items()):
        raise ValueError("run-contract runtime identity is not single-GPU P05")
    physical = runtime.get("physical_gpu_index")
    visible = runtime.get("cuda_visible_devices")
    uuid = runtime.get("gpu_uuid")
    if (
        type(physical) is not int
        or physical not in {0, 1}
        or visible != str(physical)
        or not isinstance(uuid, str)
        or _GPU_UUID.fullmatch(uuid) is None
        or runtime.get("expected_gpu_uuid") != uuid
    ):
        raise ValueError("run-contract GPU identity is invalid")
    return {
        "normalization_sha256": normalization_sha256,
        "train_weight_plan_sha256": weight_sha256["train"],
        "validation_weight_plan_sha256": weight_sha256["validation"],
    }


def _command_config_path(start: Mapping[str, Any]) -> Path:
    execution = _mapping(start.get("execution"), name="attempt execution")
    command = execution.get("command_argv")
    if not isinstance(command, list) or command[:6] != [
        "conda",
        "run",
        "-n",
        "LQ_signal",
        "python",
        "main.py",
    ]:
        raise ValueError("attempt command is not the registered LQ_signal entrypoint")
    if command.count("--config") != 1 or any(
        item in command for item in ("--override", "--local_config")
    ):
        raise ValueError("pilot attempt command contains an unregistered config override")
    index = command.index("--config")
    if index + 1 >= len(command) or not isinstance(command[index + 1], str):
        raise ValueError("pilot attempt command has no config path")
    working = _absolute_path(execution.get("working_directory"), name="working_directory")
    requested = Path(command[index + 1])
    if not requested.is_absolute():
        requested = working / requested
    return Path(os.path.abspath(os.fspath(requested)))


def _input_hash_record(value: _HashedManifest) -> dict[str, str]:
    return {
        "direct_sha256": value.direct_sha256,
        "path": str(value.path),
        "semantic_sha256": value.semantic_sha256,
    }


def _validate_binding(binding: P05PilotTimingBinding) -> dict[str, Any]:
    if not isinstance(binding, P05PilotTimingBinding):
        raise TypeError("pilot bindings must be P05PilotTimingBinding instances")
    timing = _hashed_manifest(binding.timing_manifest_path, name="pilot timing")
    timing_values = _validate_timing_manifest(timing)
    materialized = _hashed_manifest(
        binding.materialized_job_manifest_path,
        name="materialized pilot job",
    )
    job_id, job, config_path = _validate_materialized_job(materialized)
    start, terminal = _validate_attempt(binding.attempt_package_dir)
    run_contract = _hashed_manifest(
        binding.run_contract_manifest_path,
        name="pilot run contract",
    )
    run_preprocessing = _validate_run_contract(run_contract)

    attempt = _mapping(start.payload.get("attempt"), name="attempt identity")
    if (
        attempt.get("arm_id") != job["arm"]
        or attempt.get("phase") != "pilot"
        or attempt.get("dataset_id") != job["dataset_id"]
        or attempt.get("seed") != 20260801
        or attempt.get("status") != "running"
    ):
        raise ValueError("attempt identity differs from its materialized pilot job")
    if _command_config_path(start.payload) != config_path:
        raise ValueError("attempt command does not use the materialized pilot config")
    run = run_contract.payload
    if run.get("dataset_id") != job["dataset_id"]:
        raise ValueError("run-contract dataset differs from the pilot job")
    runtime = _mapping(run.get("runtime_identity"), name="run runtime")
    device = _mapping(
        _mapping(start.payload.get("execution"), name="attempt execution").get(
            "device_identity"
        ),
        name="attempt device identity",
    )
    if dict(device) != dict(runtime):
        raise ValueError("attempt and run-contract device identities differ")
    if (
        runtime.get("physical_gpu_index") != job["physical_gpu_index"]
        or runtime.get("gpu_uuid") != materialized.payload.get("expected_gpu_uuid")
    ):
        raise ValueError("materialized, attempt, and run-contract GPU bindings differ")

    outputs = _mapping(terminal.payload.get("outputs"), name="attempt outputs")
    expected_outputs = _M_PILOT_OUTPUTS if job["arm"] == "P05-M" else _BASE_PILOT_OUTPUTS
    if set(outputs) != expected_outputs:
        raise ValueError(
            f"{job['arm']} pilot terminal output key set differs from its contract"
        )
    for name, digest in outputs.items():
        _required_sha256(digest, name=f"attempt output {name}")
    run_provenance = _mapping(run.get("provenance"), name="run provenance")
    attempt_provenance = _mapping(
        start.payload.get("provenance"), name="attempt provenance"
    )
    required_output_links = {
        "pilot_timing": timing.semantic_sha256,
        "run_contract": run_contract.semantic_sha256,
        "materialized_job": materialized.semantic_sha256,
        "checkpoint": run_provenance["checkpoint_sha256"],
        "config_snapshot": run_provenance["config_sha256"],
        "code_snapshot": run_provenance["code_sha256"],
    }
    if any(outputs.get(name) != digest for name, digest in required_output_links.items()):
        raise ValueError("attempt outputs do not bind timing/checkpoint/run-contract")
    if (
        attempt_provenance.get("config_snapshot_sha256")
        != run_provenance["config_sha256"]
        or attempt_provenance.get("code_snapshot_sha256")
        != run_provenance["code_sha256"]
    ):
        raise ValueError("attempt and run-contract provenance differ")
    if any(
        attempt_provenance.get(name) != digest
        for name, digest in run_preprocessing.items()
    ):
        raise ValueError(
            "attempt and run-contract normalization/weight-plan hashes differ"
        )
    return {
        "arm": job["arm"],
        "attempt": {
            "start": _input_hash_record(start),
            "terminal": _input_hash_record(terminal),
            "terminal_outputs": dict(outputs),
        },
        "dataset": job["dataset"],
        "dataset_id": job["dataset_id"],
        "device_uuid": runtime["gpu_uuid"],
        "job_id": job_id,
        "materialized_job": {
            **_input_hash_record(materialized),
            "config_direct_sha256": _sha256_file(config_path),
            "config_path": str(config_path),
        },
        "physical_gpu_index": job["physical_gpu_index"],
        "provenance": dict(attempt_provenance),
        "run_contract": {
            **_input_hash_record(run_contract),
            "provenance": dict(run_provenance),
        },
        "timing": {
            **_input_hash_record(timing),
            **timing_values,
        },
    }


def _validate_binding_grid(records: Sequence[Mapping[str, Any]]) -> None:
    by_job = {record["job_id"]: record for record in records}
    if len(by_job) != 4 or set(by_job) != set(_JOBS):
        raise ValueError("pilot timing bindings must cover the exact four-job grid")
    common_fields = (
        "source_metadata_sha256",
        "derived_metadata_sha256",
        "signal_cache_manifest_sha256",
        "code_snapshot_sha256",
    )
    first = records[0]["provenance"]
    for field in common_fields:
        if any(record["provenance"][field] != first[field] for record in records):
            raise ValueError(f"pilot bindings have inconsistent common provenance: {field}")
    dataset_fields = (
        "split_manifest_sha256",
        "normalization_sha256",
        "train_weight_plan_sha256",
        "validation_weight_plan_sha256",
    )
    for dataset in ("CWRU", "XJTU"):
        paired = [record for record in records if record["dataset"] == dataset]
        if len(paired) != 2 or {record["arm"] for record in paired} != {
            "P05-M",
            "P05-B0",
        }:
            raise ValueError(f"pilot binding grid is incomplete for {dataset}")
        for field in dataset_fields:
            if paired[0]["provenance"][field] != paired[1]["provenance"][field]:
                raise ValueError(
                    f"{dataset} M/B0 provenance differs for {field}"
                )
        if paired[0]["device_uuid"] != paired[1]["device_uuid"]:
            raise ValueError(f"{dataset} M/B0 pilots must share the blocked GPU UUID")
    by_index = {
        index: {record["device_uuid"] for record in records if record["physical_gpu_index"] == index}
        for index in (0, 1)
    }
    if any(len(values) != 1 for values in by_index.values()) or (
        next(iter(by_index[0])) == next(iter(by_index[1]))
    ):
        raise ValueError("physical GPU indices must bind two distinct stable UUIDs")


def _validate_evaluator_benchmarks(
    package_dirs: Sequence[str | Path],
    pilot_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(package_dirs, (str, bytes)) or len(package_dirs) != 2:
        raise ValueError("exactly two evaluator benchmark packages are required")
    records: dict[str, Any] = {}
    basis: dict[str, Any] = {}
    m_runs = {
        record["dataset"]: record
        for record in pilot_records
        if record["arm"] == "P05-M"
    }
    for package_value in package_dirs:
        package = _absolute_path(package_value, name="evaluator benchmark package")
        manifest_path = package / EVALUATOR_MANIFEST_NAME
        strict = _hashed_manifest(manifest_path, name="evaluator benchmark")
        verified = verify_p05_pilot_evaluator_benchmark(package)
        if _canonical_json_bytes(dict(verified)) != _canonical_json_bytes(
            dict(strict.payload)
        ):
            raise ValueError("evaluator verifier result differs from strict JSON input")
        scope = _mapping(strict.payload.get("scope"), name="evaluator scope")
        dataset = scope.get("dataset")
        if dataset not in {"CWRU", "XJTU"} or dataset in records:
            raise ValueError("evaluator benchmarks must cover CWRU and XJTU exactly")
        if scope.get("partition_sample_count") != _VALIDATION_WINDOWS[dataset]:
            raise ValueError("evaluator benchmark validation-window count is invalid")
        shared = _mapping(
            strict.payload.get("shared_provenance"),
            name="evaluator shared provenance",
        )
        run_provenance = m_runs[dataset]["run_contract"]["provenance"]
        for name in ("config_sha256", "checkpoint_sha256", "model_sha256"):
            if shared.get(name) != run_provenance[name]:
                raise ValueError(
                    f"{dataset} evaluator does not bind the P05-M {name}"
                )
        if (
            shared.get("physical_gpu_index")
            != m_runs[dataset]["physical_gpu_index"]
            or shared.get("device_uuid") != m_runs[dataset]["device_uuid"]
        ):
            raise ValueError(
                f"{dataset} evaluator does not bind the P05-M blocked GPU identity"
            )
        benchmarks = _mapping(
            strict.payload.get("benchmarks"), name="evaluator benchmarks"
        )
        central = _mapping(benchmarks.get("central_e1_e2"), name="central")
        d03 = _mapping(benchmarks.get("d03"), name="D03")
        terminal_outputs = _mapping(
            m_runs[dataset]["attempt"]["terminal_outputs"],
            name=f"{dataset} P05-M terminal outputs",
        )
        if terminal_outputs.get("pilot_evaluator_benchmark") != strict.semantic_sha256:
            raise ValueError(
                f"{dataset} P05-M terminal does not bind its evaluator summary"
            )
        if terminal_outputs.get("pilot_d03") != d03.get("source_semantic_sha256"):
            raise ValueError(
                f"{dataset} P05-M terminal does not bind the summary D03 source"
            )
        central_components = _mapping(central.get("components"), name="central components")
        d03_components = _mapping(d03.get("components"), name="D03 components")
        unit_costs = {
            "central_deletion": central_components["rule_deletions"][
                "seconds_per_forward_call_per_window"
            ],
            "central_original": central_components["original_trace"][
                "seconds_per_forward_call_per_window"
            ],
            "central_shuffle": central_components["consequent_shuffles"][
                "seconds_per_forward_call_per_window"
            ],
            "d03_noise": d03_components["noise_draws"][
                "seconds_per_forward_call_per_window"
            ],
            "d03_original": d03_components["original_trace"][
                "seconds_per_forward_call_per_window"
            ],
        }
        unit_costs = {
            name: _finite_nonnegative(value, name=f"{dataset} {name}")
            for name, value in unit_costs.items()
        }
        records[dataset] = _input_hash_record(strict)
        basis[dataset] = {
            "source_benchmark_semantic_sha256": strict.semantic_sha256,
            "unit_seconds_per_forward_call_per_window": unit_costs,
        }
    if set(records) != {"CWRU", "XJTU"}:
        raise ValueError("evaluator benchmark grid is incomplete")
    return dict(sorted(records.items())), dict(sorted(basis.items()))


def _training_basis(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for dataset in ("CWRU", "XJTU"):
        candidates = {
            record["arm"]: {
                "job_id": record["job_id"],
                "median_epoch_seconds": record["timing"]["median_epoch_seconds"],
                "startup_seconds": record["timing"]["startup_seconds"],
            }
            for record in records
            if record["dataset"] == dataset
        }
        startup = max(item["startup_seconds"] for item in candidates.values())
        epoch = max(item["median_epoch_seconds"] for item in candidates.values())
        result[dataset] = {
            "candidate_measurements": dict(sorted(candidates.items())),
            "conservative_rule": (
                "independent_maximum_of_same_dataset_P05_M_or_P05_B0_"
                "startup_and_median_epoch_2_through_5"
            ),
            "selected_median_epoch_seconds": epoch,
            "selected_startup_seconds": startup,
        }
    return result


def _build_training_forecast(basis: Mapping[str, Any]) -> dict[str, Any]:
    datasets = {}
    totals = {stage: 0.0 for stage in _TRAINING_STAGES}
    for dataset in ("CWRU", "XJTU"):
        startup = _finite_nonnegative(
            basis[dataset]["selected_startup_seconds"],
            name=f"{dataset} selected startup",
        )
        epoch = _finite_nonnegative(
            basis[dataset]["selected_median_epoch_seconds"],
            name=f"{dataset} selected epoch",
        )
        stages = {}
        for stage, contract in _TRAINING_STAGES.items():
            per_job = startup + epoch * contract["maximum_epochs"]
            seconds = per_job * contract["jobs_per_dataset"]
            stages[stage] = {
                **contract,
                "formula": (
                    "jobs_per_dataset * (selected_startup_seconds + "
                    "selected_median_epoch_seconds * maximum_epochs)"
                ),
                "forecast_seconds": seconds,
                "per_job_seconds": per_job,
            }
            totals[stage] += seconds
        datasets[dataset] = {"stages": stages}
    full = sum(totals.values())
    central = full - totals["retraining_ablations"]
    return {
        "per_dataset": datasets,
        "stage_totals_seconds": totals,
        "program_totals_seconds": {
            "central_60_jobs": central,
            "full_90_jobs": full,
        },
    }


def _evaluation_component(
    *,
    unit_seconds: float,
    structural_multiplier: int,
    test_windows: int,
) -> dict[str, Any]:
    multiplier = structural_multiplier * FINAL_SEED_COUNT * test_windows
    return {
        "final_forward_window_multiplier": multiplier,
        "formula": (
            "unit_seconds_per_forward_call_per_window * "
            "structural_multiplier_per_window_per_seed * seed_count * "
            "test_window_count"
        ),
        "forecast_seconds": unit_seconds * multiplier,
        "seed_count": FINAL_SEED_COUNT,
        "structural_multiplier_per_window_per_seed": structural_multiplier,
        "test_window_count": test_windows,
        "unit_seconds_per_forward_call_per_window": unit_seconds,
    }


def _build_evaluation_forecast(basis: Mapping[str, Any]) -> dict[str, Any]:
    datasets = {}
    totals = {program: 0.0 for program in _EVALUATION_COMPONENTS}
    for dataset in ("CWRU", "XJTU"):
        units = basis[dataset]["unit_seconds_per_forward_call_per_window"]
        test_windows = _TEST_WINDOWS[dataset]
        programs = {}
        for program, multipliers in _EVALUATION_COMPONENTS.items():
            components = {
                name: _evaluation_component(
                    unit_seconds=_finite_nonnegative(
                        units[name], name=f"{dataset} {name} unit timing"
                    ),
                    structural_multiplier=structural,
                    test_windows=test_windows,
                )
                for name, structural in multipliers.items()
            }
            seconds = sum(item["forecast_seconds"] for item in components.values())
            programs[program] = {
                "components": components,
                "forecast_seconds": seconds,
            }
            totals[program] += seconds
        datasets[dataset] = {"programs": programs, "test_window_count": test_windows}
    return {"per_dataset": datasets, "program_totals_seconds": totals}


def _build_programs(
    training: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> list[dict[str, Any]]:
    train = training["program_totals_seconds"]
    evaluate = evaluation["program_totals_seconds"]
    contracts = (
        ("full_90_with_d03", 90, True, True),
        ("full_90_without_d03", 90, False, True),
        ("central_60_mandatory", 60, False, False),
    )
    programs = []
    for order, (program_id, jobs, d03, ablations) in enumerate(contracts, start=1):
        training_seconds = train["full_90_jobs" if ablations else "central_60_jobs"]
        evaluation_seconds = evaluate["central_mandatory"]
        if d03:
            evaluation_seconds += evaluate["d03"]
        if ablations:
            evaluation_seconds += evaluate["retraining_ablations"]
        pre_margin = training_seconds + evaluation_seconds
        forecast_seconds = pre_margin * SAFETY_FACTOR
        forecast_hours = forecast_seconds / 3600.0
        programs.append(
            {
                "ablations_retained": ablations,
                "d03_retained": d03,
                "evaluation_seconds": evaluation_seconds,
                "forecast_gpu_hours": forecast_hours,
                "forecast_seconds_after_safety_factor": forecast_seconds,
                "gpu_job_count": jobs,
                "order": order,
                "pre_margin_seconds": pre_margin,
                "program_id": program_id,
                "safety_factor": SAFETY_FACTOR,
                "training_seconds": training_seconds,
                "within_168_gpu_hour_cap": forecast_hours <= GPU_HOUR_CAP,
            }
        )
    return programs


def _decision(programs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = next(
        (program for program in programs if program["within_168_gpu_hour_cap"]),
        None,
    )
    if selected is None:
        return {
            "ablations_retained": False,
            "d03_retained": False,
            "first_acceptable_program_locked": False,
            "selected_program": None,
            "status": "stop_requires_human_protocol_or_resource_amendment",
        }
    return {
        "ablations_retained": selected["ablations_retained"],
        "d03_retained": selected["d03_retained"],
        "first_acceptable_program_locked": True,
        "selected_program": selected["program_id"],
        "status": "locked_first_acceptable_program",
    }


def _stored_manifest_path(
    value: Any,
    *,
    name: str,
    extra_keys: set[str] | None = None,
) -> Path:
    record = _mapping(value, name=name)
    expected = {"direct_sha256", "path", "semantic_sha256"} | (extra_keys or set())
    if set(record) != expected:
        raise ValueError(f"{name} hash record is incomplete or unexpected")
    _required_sha256(record["direct_sha256"], name=f"{name}.direct_sha256")
    _required_sha256(record["semantic_sha256"], name=f"{name}.semantic_sha256")
    if not isinstance(record["path"], str):
        raise ValueError(f"{name}.path must be an absolute path string")
    source = _absolute_path(record["path"], name=f"{name}.path")
    if str(source) != record["path"]:
        raise ValueError(f"{name}.path must already be absolute and normalized")
    return source


def _revalidated_input_bases(
    inputs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    pilot_jobs = _mapping(inputs.get("pilot_jobs"), name="inputs.pilot_jobs")
    evaluators = _mapping(
        inputs.get("evaluator_benchmarks"),
        name="inputs.evaluator_benchmarks",
    )
    rebuilt_records = []
    for job_id in sorted(_JOBS):
        stored = _mapping(pilot_jobs[job_id], name=f"inputs.pilot_jobs.{job_id}")
        attempt = _mapping(stored.get("attempt"), name=f"{job_id}.attempt")
        if set(attempt) != {"start", "terminal", "terminal_outputs"}:
            raise ValueError(f"{job_id} attempt input record is incomplete")
        start_path = _stored_manifest_path(
            attempt["start"], name=f"{job_id}.attempt.start"
        )
        terminal_path = _stored_manifest_path(
            attempt["terminal"], name=f"{job_id}.attempt.terminal"
        )
        if (
            start_path.name != "start.json"
            or terminal_path.name != "terminal.json"
            or start_path.parent != terminal_path.parent
        ):
            raise ValueError(f"{job_id} attempt paths do not identify one package")
        timing_path = _stored_manifest_path(
            stored.get("timing"),
            name=f"{job_id}.timing",
            extra_keys={"median_epoch_seconds", "startup_seconds"},
        )
        materialized_path = _stored_manifest_path(
            stored.get("materialized_job"),
            name=f"{job_id}.materialized_job",
            extra_keys={"config_direct_sha256", "config_path"},
        )
        run_contract_path = _stored_manifest_path(
            stored.get("run_contract"),
            name=f"{job_id}.run_contract",
            extra_keys={"provenance"},
        )
        rebuilt = _validate_binding(
            P05PilotTimingBinding(
                timing_manifest_path=timing_path,
                attempt_package_dir=start_path.parent,
                materialized_job_manifest_path=materialized_path,
                run_contract_manifest_path=run_contract_path,
            )
        )
        if _canonical_json_bytes(stored) != _canonical_json_bytes(rebuilt):
            raise ValueError(f"{job_id} stored input record differs from its sources")
        rebuilt_records.append(rebuilt)
    _validate_binding_grid(rebuilt_records)

    evaluator_packages = []
    for dataset in ("CWRU", "XJTU"):
        manifest_path = _stored_manifest_path(
            evaluators[dataset], name=f"inputs.evaluator_benchmarks.{dataset}"
        )
        if manifest_path.name != EVALUATOR_MANIFEST_NAME:
            raise ValueError(f"{dataset} evaluator input path is not its manifest")
        evaluator_packages.append(manifest_path.parent)
    rebuilt_evaluators, evaluator_basis = _validate_evaluator_benchmarks(
        evaluator_packages,
        rebuilt_records,
    )
    if _canonical_json_bytes(evaluators) != _canonical_json_bytes(rebuilt_evaluators):
        raise ValueError("stored evaluator inputs differ from their source manifests")
    return _training_basis(rebuilt_records), evaluator_basis


def _validate_formula_blocks(manifest: Mapping[str, Any]) -> None:
    if manifest.get("frozen_contract") != {
        **_FROZEN_HASHES,
        "aggregate_gpu_hour_cap": GPU_HOUR_CAP,
        "decision_order": [
            "full_90_with_d03",
            "full_90_without_d03",
            "central_60_mandatory",
        ],
        "job_counts": {
            "central_program": 60,
            "decisive_central": 40,
            "full_program": 90,
            "pilot": 4,
            "retraining_ablations": 30,
            "tuning": 16,
        },
        "safety_factor": SAFETY_FACTOR,
    }:
        raise ValueError("budget artifact frozen contract is invalid")
    basis = _mapping(manifest.get("timing_basis"), name="timing_basis")
    evaluator_basis = _mapping(
        manifest.get("evaluator_basis"), name="evaluator_basis"
    )
    expected_basis, expected_evaluator_basis = _revalidated_input_bases(
        _mapping(manifest.get("inputs"), name="inputs")
    )
    if _canonical_json_bytes(basis) != _canonical_json_bytes(expected_basis):
        raise ValueError(
            "timing basis does not preserve the independent same-dataset maxima"
        )
    if _canonical_json_bytes(evaluator_basis) != _canonical_json_bytes(
        expected_evaluator_basis
    ):
        raise ValueError("evaluator basis differs from its bound summary components")
    expected_training = _build_training_forecast(basis)
    expected_evaluation = _build_evaluation_forecast(evaluator_basis)
    if _canonical_json_bytes(manifest.get("training_forecast")) != _canonical_json_bytes(
        expected_training
    ):
        raise ValueError("training forecast formula or boundary differs")
    if _canonical_json_bytes(
        manifest.get("evaluation_forecast")
    ) != _canonical_json_bytes(expected_evaluation):
        raise ValueError("evaluation forecast formula or multiplier differs")
    expected_programs = _build_programs(expected_training, expected_evaluation)
    if _canonical_json_bytes(manifest.get("programs")) != _canonical_json_bytes(
        expected_programs
    ):
        raise ValueError("program forecast or safety-factor formula differs")
    if manifest.get("decision") != _decision(expected_programs):
        raise ValueError("budget retention decision is not the first acceptable program")


def _validate_semantic_manifest(manifest: Mapping[str, Any]) -> None:
    expected_keys = {
        "conclusion_control",
        "decision",
        "evaluator_basis",
        "evaluation_forecast",
        "frozen_contract",
        "inputs",
        "programs",
        "schema_name",
        "schema_version",
        "status",
        "timing_basis",
        "training_forecast",
    }
    if set(manifest) != expected_keys or (
        manifest.get("schema_name") != SCHEMA_NAME
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "engineering_budget_decision_only"
    ):
        raise ValueError("post-pilot budget artifact schema or status is invalid")
    if manifest.get("conclusion_control") != {
        "claim_decision": "forbidden",
        "engineering_budget_decision": "locked",
        "paper_evidence": False,
        "performance_conclusion": "forbidden",
        "timing_is_model_performance_evidence": False,
    }:
        raise ValueError("post-pilot budget artifact conclusion control is invalid")
    inputs = _mapping(manifest.get("inputs"), name="inputs")
    if set(inputs) != {"evaluator_benchmarks", "pilot_jobs"}:
        raise ValueError("post-pilot budget inputs are incomplete")
    pilot_jobs = _mapping(inputs["pilot_jobs"], name="inputs.pilot_jobs")
    evaluators = _mapping(
        inputs["evaluator_benchmarks"], name="inputs.evaluator_benchmarks"
    )
    if set(pilot_jobs) != set(_JOBS) or set(evaluators) != {"CWRU", "XJTU"}:
        raise ValueError("post-pilot budget input grid is incomplete")
    _validate_formula_blocks(manifest)


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
        raise RuntimeError("atomic create-only P05 budget export requires renameat2")
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


def create_p05_post_pilot_budget_decision(
    package_dir: str | Path,
    *,
    pilot_bindings: Sequence[P05PilotTimingBinding],
    evaluator_benchmark_package_dirs: Sequence[str | Path],
) -> P05PostPilotBudgetResult:
    """Validate six pilot inputs and lock the first cap-compliant program."""

    target = _absolute_path(package_dir, name="package_dir")
    if target.is_symlink() or target.exists():
        raise FileExistsError(f"post-pilot budget artifact is create-only: {target}")
    if isinstance(pilot_bindings, (str, bytes)) or len(pilot_bindings) != 4:
        raise ValueError("exactly four pilot timing bindings are required")
    records = [_validate_binding(binding) for binding in pilot_bindings]
    _validate_binding_grid(records)
    evaluator_inputs, evaluator_basis = _validate_evaluator_benchmarks(
        evaluator_benchmark_package_dirs,
        records,
    )
    timing_basis = _training_basis(records)
    training = _build_training_forecast(timing_basis)
    evaluation = _build_evaluation_forecast(evaluator_basis)
    programs = _build_programs(training, evaluation)
    decision = _decision(programs)
    semantic = {
        "conclusion_control": {
            "claim_decision": "forbidden",
            "engineering_budget_decision": "locked",
            "paper_evidence": False,
            "performance_conclusion": "forbidden",
            "timing_is_model_performance_evidence": False,
        },
        "decision": decision,
        "evaluator_basis": evaluator_basis,
        "evaluation_forecast": evaluation,
        "frozen_contract": {
            **_FROZEN_HASHES,
            "aggregate_gpu_hour_cap": GPU_HOUR_CAP,
            "decision_order": [
                "full_90_with_d03",
                "full_90_without_d03",
                "central_60_mandatory",
            ],
            "job_counts": {
                "central_program": 60,
                "decisive_central": 40,
                "full_program": 90,
                "pilot": 4,
                "retraining_ablations": 30,
                "tuning": 16,
            },
            "safety_factor": SAFETY_FACTOR,
        },
        "inputs": {
            "evaluator_benchmarks": evaluator_inputs,
            "pilot_jobs": {
                record["job_id"]: record for record in sorted(
                    records, key=lambda item: item["job_id"]
                )
            },
        },
        "programs": programs,
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "engineering_budget_decision_only",
        "timing_basis": timing_basis,
        "training_forecast": training,
    }
    _validate_semantic_manifest(semantic)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("post-pilot budget parent must be a real directory")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(parent))
    )
    try:
        semantic_sha256 = _sha256_bytes(_canonical_json_bytes(semantic))
        manifest = {**semantic, "content": {"semantic_sha256": semantic_sha256}}
        manifest_path = temporary / MANIFEST_NAME
        with manifest_path.open("xb") as handle:
            handle.write(_pretty_json_bytes(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary)
        _rename_directory_noreplace(temporary, target)
        _fsync_directory(parent)
        installed = target / MANIFEST_NAME
        return P05PostPilotBudgetResult(
            package_dir=target,
            manifest_path=installed,
            semantic_sha256=semantic_sha256,
            manifest_sha256=_sha256_file(installed),
            status=decision["status"],
            selected_program=decision["selected_program"],
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_p05_post_pilot_budget_decision(
    package_dir: str | Path,
) -> Mapping[str, Any]:
    """Strictly verify one locked engineering budget decision manifest."""

    package = _absolute_path(package_dir, name="package_dir")
    if package.is_symlink() or not package.is_dir():
        raise FileNotFoundError("post-pilot budget artifact must be a real directory")
    entries = {entry.name: entry for entry in package.iterdir()}
    if set(entries) != {MANIFEST_NAME}:
        raise ValueError("post-pilot budget package has unexpected content")
    loaded = _hashed_manifest(entries[MANIFEST_NAME], name="post-pilot budget manifest")
    semantic = {
        key: value for key, value in loaded.payload.items() if key != "content"
    }
    _validate_semantic_manifest(semantic)
    return MappingProxyType(dict(loaded.payload))


__all__ = [
    "P05PilotTimingBinding",
    "P05PostPilotBudgetResult",
    "create_p05_post_pilot_budget_decision",
    "verify_p05_post_pilot_budget_decision",
]
