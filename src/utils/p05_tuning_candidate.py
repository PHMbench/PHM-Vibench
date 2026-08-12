"""Create a hash-bound validation candidate for one P05 tuning job.

The producer accepts only a completed, validation-only re-evaluation of the
single best checkpoint selected by validation loss.  It verifies the frozen
materialized job, its source matrix, and every referenced artifact before it
atomically emits the exact schema consumed by :mod:`p05_tuning_selection`.
The result remains unadjudicated and is never paper evidence by itself.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml


SCHEMA_NAME = "p05.tuning_validation_candidate"
SCHEMA_VERSION = 1
MATERIALIZED_SCHEMA_NAME = "p05.materialized_neural_job"
CODE_SCHEMA_NAME = "p05.code_snapshot"
RUN_CONTRACT_SCHEMA_NAME = "p05.run_artifact_bundle"
MANIFEST_NAME = "manifest.json"
PROTOCOL_BUNDLE_SHA256 = (
    "8d01361c39a778d437ce235ad1e8d3877313f128d6593fbb74812a4b237a1654"
)

ARMS = ("P05-M", "P05-B0", "P05-B1", "P05-B3")
DATASET_IDS = {"CWRU": 1, "XJTU": 2}
LEARNING_RATES = (Decimal("0.001"), Decimal("0.0003"))
_RATE_TOKENS = {
    Decimal("0.001"): "LR1E3",
    Decimal("0.0003"): "LR3E4",
}
TUNING_SEED = 20260801
MAX_EPOCHS = 60
PATIENCE = 10

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_GPU_UUID_PATTERN = re.compile(r"^GPU-[!-~]{1,124}$")
_PROVENANCE_KEYS = frozenset(
    {
        "source_metadata_sha256",
        "derived_metadata_sha256",
        "signal_cache_manifest_sha256",
        "split_manifest_sha256",
        "normalization_sha256",
        "train_weight_plan_sha256",
        "validation_weight_plan_sha256",
    }
)
_MATERIALIZED_KEYS = frozenset(
    {
        "schema_name",
        "schema_version",
        "paper_id",
        "protocol_id",
        "matrix_id",
        "matrix_sha256",
        "stage",
        "job_id",
        "arm",
        "dataset",
        "seed",
        "learning_rate",
        "learning_rate_source",
        "physical_gpu_index",
        "expected_gpu_uuid",
        "pilot_common_contract",
        "tuning_selection",
        "config_file",
        "config_sha256",
        "materialization_status",
        "execution_status",
        "evidence_status",
        "claim_support",
        "scientific_overrides",
        "content",
    }
)
_RUN_CONTRACT_KEYS = frozenset(
    {
        "schema_name",
        "schema_version",
        "paper_id",
        "dataset_id",
        "normalization_plan",
        "weight_plans",
        "runtime_identity",
        "provenance",
        "content",
    }
)


@dataclass(frozen=True)
class P05TuningValidationCandidateResult:
    """Paths, hashes, and create/reuse state for one candidate package."""

    package_dir: Path
    manifest_path: Path
    semantic_sha256: str
    manifest_sha256: str
    status: str


@dataclass(frozen=True)
class _VerifiedJob:
    job_id: str
    arm_id: str
    dataset: str
    dataset_id: int
    learning_rate: Decimal
    source_matrix_sha256: str
    physical_gpu_index: int
    expected_gpu_uuid: str


class _UniqueSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueSafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ValueError("YAML mapping key must be hashable") from exc
        if duplicate:
            raise ValueError(f"duplicate YAML mapping key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


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


def _required_hash(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _exact(value: Any, expected: Any, *, name: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{name} must be exactly {expected!r}")


def _exact_keys(value: Any, expected: frozenset[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = dict(value)
    if set(result) != expected:
        missing = sorted(expected - set(result))
        unexpected = sorted(set(result) - expected, key=str)
        raise ValueError(
            f"{name} fields do not match the frozen contract: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _real_file(path_input: str | Path, *, name: str) -> Path:
    path = Path(os.path.abspath(os.fspath(path_input)))
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{name} is unavailable: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"{name} must be a regular file: {path}")
    return resolved


def _load_json(path: Path, *, name: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {name}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object: {path}")
    return value, payload


def _load_yaml(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueSafeLoader)
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"invalid {name}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one YAML mapping: {path}")
    return value


def _verify_semantic_manifest(
    path_input: str | Path,
    *,
    name: str,
    schema_name: str,
) -> tuple[Path, dict[str, Any], str]:
    path = _real_file(path_input, name=name)
    manifest, _ = _load_json(path, name=name)
    if manifest.get("schema_name") != schema_name:
        raise ValueError(f"{name}.schema_name must be {schema_name!r}")
    content = manifest.get("content")
    if not isinstance(content, Mapping) or set(content) != {"semantic_sha256"}:
        raise ValueError(f"{name}.content must contain only semantic_sha256")
    recorded = _required_hash(
        content["semantic_sha256"],
        name=f"{name}.content.semantic_sha256",
    )
    semantic = {key: value for key, value in manifest.items() if key != "content"}
    actual = _sha256_bytes(_canonical_json_bytes(semantic))
    if actual != recorded:
        raise ValueError(f"{name} semantic hash mismatch: {path}")
    return path, manifest, actual


def _decimal_number(
    value: Any,
    *,
    name: str,
    minimum: Decimal,
    maximum: Decimal | None = None,
) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a JSON number")
    try:
        converted = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not converted.is_finite() or converted < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    if maximum is not None and converted > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return converted


def _float(value: Decimal) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("candidate contains a non-finite numeric value")
    return converted


def _nested_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _validate_matrix(
    matrix_path_input: str | Path,
) -> tuple[Path, dict[str, Any], str]:
    path = _real_file(matrix_path_input, name="P05 tuning source matrix")
    matrix_hash = _sha256_file(path)
    matrix = _load_yaml(path, name="P05 tuning source matrix")
    identity = {
        "schema_version": 1,
        "kind": "p05_frozen_neural_tuning_execution_matrix",
        "paper_id": "P05",
        "protocol_id": "P05-G040-v3.2",
        "matrix_id": "P05-NEURAL-TUNING-v1",
        "status": "frozen_declarative_awaiting_gpu_uuid_binding",
    }
    for key, expected in identity.items():
        _exact(matrix.get(key), expected, name=f"source_matrix.{key}")

    stage = _nested_mapping(matrix.get("stage_contract"), name="source_matrix.stage_contract")
    stage_exact = {
        "phase": "tuning",
        "job_count": 16,
        "seed": TUNING_SEED,
        "environment_stage": "fit_validate_only",
        "maximum_epochs": MAX_EPOCHS,
        "early_stopping": True,
        "patience": PATIENCE,
        "checkpoint_selection": "minimum_validation_loss",
        "test_dataset_construction": "forbidden",
        "test_cache_access": "forbidden",
        "test_metric_access": "forbidden",
        "trace_export": "false_for_every_arm",
    }
    for key, expected in stage_exact.items():
        _exact(stage.get(key), expected, name=f"source_matrix.stage_contract.{key}")
    factors = _nested_mapping(
        stage.get("factors"),
        name="source_matrix.stage_contract.factors",
    )
    _exact(factors.get("arms"), list(ARMS), name="source_matrix factors.arms")
    _exact(
        factors.get("datasets"),
        list(DATASET_IDS),
        name="source_matrix factors.datasets",
    )
    rates = factors.get("learning_rates")
    if not isinstance(rates, list) or len(rates) != len(LEARNING_RATES):
        raise ValueError("source_matrix factors.learning_rates schema drift")
    observed_rates = {
        _decimal_number(
            value,
            name="source_matrix factors.learning_rates entry",
            minimum=Decimal("0"),
        )
        for value in rates
    }
    if observed_rates != set(LEARNING_RATES):
        raise ValueError("source_matrix factors.learning_rates contract drift")

    runtime = _nested_mapping(matrix.get("runtime"), name="source_matrix.runtime")
    runtime_exact = {
        "allowed_physical_gpu_indices": [0, 1],
        "forbidden_physical_gpu_indices": [2],
        "maximum_concurrent_processes_per_gpu": 1,
        "one_gpu_per_process": True,
        "distributed_execution": "forbidden",
        "network_use": "forbidden",
        "automatic_download": "forbidden",
    }
    for key, expected in runtime_exact.items():
        _exact(runtime.get(key), expected, name=f"source_matrix.runtime.{key}")

    outputs = _nested_mapping(matrix.get("outputs"), name="source_matrix.outputs")
    output_exact = {
        "materialization_status": "created-not-executed",
        "execution_status": "not_started",
        "evidence_status": "unadjudicated",
        "claim_support_before_ledger_and_audit": "forbidden",
    }
    for key, expected in output_exact.items():
        _exact(outputs.get(key), expected, name=f"source_matrix.outputs.{key}")

    datasets = _nested_mapping(matrix.get("datasets"), name="source_matrix.datasets")
    if set(datasets) != set(DATASET_IDS):
        raise ValueError("source_matrix.datasets must contain exactly CWRU and XJTU")
    for dataset, dataset_id in DATASET_IDS.items():
        record = _nested_mapping(
            datasets[dataset],
            name=f"source_matrix.datasets.{dataset}",
        )
        _exact(
            record.get("dataset_id"),
            dataset_id,
            name=f"source_matrix.datasets.{dataset}.dataset_id",
        )

    jobs = matrix.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 16:
        raise ValueError("source_matrix.jobs must contain exactly 16 entries")
    if any(not isinstance(job, Mapping) for job in jobs):
        raise ValueError("source_matrix.jobs entries must be mappings")
    job_ids = [job.get("id") for job in jobs]
    if any(not isinstance(job_id, str) or not job_id for job_id in job_ids):
        raise ValueError("source_matrix job IDs must be non-empty strings")
    if len(set(job_ids)) != len(job_ids):
        raise ValueError("source_matrix job IDs must be unique")
    expected_cells = {
        (arm, dataset, rate)
        for arm in ARMS
        for dataset in DATASET_IDS
        for rate in LEARNING_RATES
    }
    observed_cells: set[tuple[str, str, Decimal]] = set()
    for job in jobs:
        row = _exact_keys(
            job,
            frozenset(
                {
                    "id",
                    "arm",
                    "dataset",
                    "learning_rate",
                    "seed",
                    "output_dir",
                    "materialize_command",
                }
            ),
            name="source_matrix job",
        )
        arm = row["arm"]
        dataset = row["dataset"]
        if arm not in ARMS or dataset not in DATASET_IDS:
            raise ValueError("source_matrix job has an unregistered arm or dataset")
        _exact(row["seed"], TUNING_SEED, name="source_matrix job.seed")
        rate = _decimal_number(
            row["learning_rate"],
            name="source_matrix job.learning_rate",
            minimum=Decimal("0"),
        )
        if rate not in LEARNING_RATES:
            raise ValueError("source_matrix job learning rate is unregistered")
        expected_id = f"P05-TUNE-{arm[4:]}-{dataset}-{_RATE_TOKENS[rate]}"
        _exact(row["id"], expected_id, name="source_matrix job.id")
        _exact(
            row["output_dir"],
            f"results/experiments/p05/tuning/{expected_id}",
            name="source_matrix job.output_dir",
        )
        observed_cells.add((arm, dataset, rate))
    if observed_cells != expected_cells:
        raise ValueError("source_matrix jobs do not form the exact tuning factorial")

    waves = matrix.get("execution_waves")
    if not isinstance(waves, list) or len(waves) != 8:
        raise ValueError("source_matrix must contain exactly eight execution waves")
    assignments: list[tuple[str, int]] = []
    for wave_number, wave in enumerate(waves, start=1):
        wave_map = _nested_mapping(wave, name="source_matrix execution wave")
        _exact(
            wave_map.get("wave"),
            wave_number,
            name="source_matrix execution wave number",
        )
        concurrent = wave_map.get("concurrent_jobs")
        if not isinstance(concurrent, list) or len(concurrent) != 2:
            raise ValueError("each source_matrix wave must contain exactly two jobs")
        indices: set[int] = set()
        for assignment in concurrent:
            item = _exact_keys(
                assignment,
                frozenset({"job_id", "physical_gpu_index"}),
                name="source_matrix wave assignment",
            )
            job_id = item["job_id"]
            gpu_index = item["physical_gpu_index"]
            if job_id not in set(job_ids):
                raise ValueError("source_matrix wave references an unknown job")
            if type(gpu_index) is not int or gpu_index not in {0, 1}:
                raise ValueError("source_matrix wave may use only GPU0 or GPU1")
            indices.add(gpu_index)
            assignments.append((job_id, gpu_index))
        if indices != {0, 1}:
            raise ValueError("each source_matrix wave must use GPU0 and GPU1 once")
    assigned_job_ids = [job_id for job_id, _ in assignments]
    if len(set(assigned_job_ids)) != 16 or set(assigned_job_ids) != set(job_ids):
        raise ValueError("source_matrix must assign every tuning job exactly once")
    return path, matrix, matrix_hash


def _validate_materialized_job(
    materialized_job_manifest_path: str | Path,
    *,
    source_matrix_path: str | Path,
) -> _VerifiedJob:
    manifest_path, manifest, _ = _verify_semantic_manifest(
        materialized_job_manifest_path,
        name="P05 materialized tuning job manifest",
        schema_name=MATERIALIZED_SCHEMA_NAME,
    )
    _exact_keys(manifest, _MATERIALIZED_KEYS, name="materialized_job")
    fixed = {
        "schema_version": 1,
        "paper_id": "P05",
        "protocol_id": "P05-G040-v3.2",
        "matrix_id": "P05-NEURAL-TUNING-v1",
        "stage": "tuning",
        "seed": TUNING_SEED,
        "learning_rate_source": "frozen_tuning_matrix_job",
        "tuning_selection": None,
        "config_file": "config.yaml",
        "materialization_status": "created-not-executed",
        "execution_status": "not_started",
        "evidence_status": "unadjudicated",
        "claim_support": "forbidden_before_ledger_and_audit",
        "scientific_overrides": "forbidden",
    }
    for key, expected in fixed.items():
        _exact(manifest[key], expected, name=f"materialized_job.{key}")

    arm = manifest["arm"]
    if arm not in ARMS:
        raise ValueError(f"materialized_job.arm must be one of {list(ARMS)}")
    dataset = manifest["dataset"]
    if dataset not in DATASET_IDS:
        raise ValueError(
            f"materialized_job.dataset must be one of {list(DATASET_IDS)}"
        )
    learning_rate = _decimal_number(
        manifest["learning_rate"],
        name="materialized_job.learning_rate",
        minimum=Decimal("0"),
    )
    if learning_rate not in LEARNING_RATES:
        raise ValueError("materialized_job.learning_rate must be 0.001 or 0.0003")
    expected_job_id = (
        f"P05-TUNE-{arm[4:]}-{dataset}-{_RATE_TOKENS[learning_rate]}"
    )
    _exact(manifest["job_id"], expected_job_id, name="materialized_job.job_id")

    physical_gpu_index = manifest["physical_gpu_index"]
    if type(physical_gpu_index) is not int or physical_gpu_index not in {0, 1}:
        raise ValueError("materialized_job.physical_gpu_index must be 0 or 1")
    expected_gpu_uuid = manifest["expected_gpu_uuid"]
    if (
        not isinstance(expected_gpu_uuid, str)
        or _GPU_UUID_PATTERN.fullmatch(expected_gpu_uuid) is None
        or "REQUIRED" in expected_gpu_uuid
    ):
        raise ValueError("materialized_job.expected_gpu_uuid must be an observed GPU-* UUID")

    _, matrix, matrix_hash = _validate_matrix(source_matrix_path)
    registered_matrix_hash = _required_hash(
        manifest["matrix_sha256"],
        name="materialized_job.matrix_sha256",
    )
    if registered_matrix_hash != matrix_hash:
        raise ValueError("materialized job source matrix SHA-256 mismatch")
    materialized_pilot = _exact_keys(
        manifest["pilot_common_contract"],
        frozenset({"path", "pilot_matrix_sha256", "common_config_sha256"}),
        name="materialized_job.pilot_common_contract",
    )
    matrix_pilot = _nested_mapping(
        matrix.get("pilot_common_contract"),
        name="source_matrix.pilot_common_contract",
    )
    pilot_hash = _required_hash(
        materialized_pilot["pilot_matrix_sha256"],
        name="materialized_job.pilot_common_contract.pilot_matrix_sha256",
    )
    if pilot_hash != _required_hash(
        matrix_pilot.get("sha256"),
        name="source_matrix.pilot_common_contract.sha256",
    ):
        raise ValueError("materialized job pilot-matrix hash conflicts with source matrix")
    _required_hash(
        materialized_pilot["common_config_sha256"],
        name="materialized_job.pilot_common_contract.common_config_sha256",
    )
    if not isinstance(materialized_pilot["path"], str) or not materialized_pilot["path"]:
        raise ValueError("materialized_job.pilot_common_contract.path must be non-empty")
    matching_jobs = [
        dict(job) for job in matrix["jobs"] if job.get("id") == expected_job_id
    ]
    if len(matching_jobs) != 1:
        raise ValueError("materialized job must bind exactly one source-matrix job")
    matrix_job = matching_jobs[0]
    expected_job_fields = {
        "arm": arm,
        "dataset": dataset,
        "seed": TUNING_SEED,
    }
    for key, expected in expected_job_fields.items():
        _exact(matrix_job.get(key), expected, name=f"source_matrix.job.{key}")
    matrix_rate = _decimal_number(
        matrix_job.get("learning_rate"),
        name="source_matrix.job.learning_rate",
        minimum=Decimal("0"),
    )
    if matrix_rate != learning_rate:
        raise ValueError("materialized learning rate conflicts with source-matrix job")

    assignments = [
        assignment
        for wave in matrix.get("execution_waves", [])
        if isinstance(wave, Mapping)
        for assignment in wave.get("concurrent_jobs", [])
        if isinstance(assignment, Mapping)
        and assignment.get("job_id") == expected_job_id
    ]
    if len(assignments) != 1:
        raise ValueError("source matrix must assign the tuning job exactly once")
    _exact(
        assignments[0].get("physical_gpu_index"),
        physical_gpu_index,
        name="source_matrix job physical_gpu_index",
    )

    materialized_config_path = _real_file(
        manifest_path.parent / manifest["config_file"],
        name="materialized P05 config file",
    )
    materialized_config_hash = _required_hash(
        manifest["config_sha256"],
        name="materialized_job.config_sha256",
    )
    if _sha256_file(materialized_config_path) != materialized_config_hash:
        raise ValueError("materialized P05 config file SHA-256 mismatch")
    _validate_config_contract(
        materialized_config_path,
        arm=arm,
        learning_rate=learning_rate,
        expected_gpu_uuid=expected_gpu_uuid,
        name="materialized P05 config",
    )

    return _VerifiedJob(
        job_id=expected_job_id,
        arm_id=arm,
        dataset=dataset,
        dataset_id=DATASET_IDS[dataset],
        learning_rate=learning_rate,
        source_matrix_sha256=matrix_hash,
        physical_gpu_index=physical_gpu_index,
        expected_gpu_uuid=expected_gpu_uuid,
    )


def _validate_config_contract(
    config_path: Path,
    *,
    arm: str,
    learning_rate: Decimal,
    expected_gpu_uuid: str,
    name: str,
) -> None:
    config = _load_yaml(config_path, name=name)
    environment = _nested_mapping(config.get("environment"), name="config.environment")
    task = _nested_mapping(config.get("task"), name="config.task")
    trainer = _nested_mapping(config.get("trainer"), name="config.trainer")
    config_exact = (
        (environment, "seed", TUNING_SEED, "config.environment.seed"),
        (environment, "stage", "fit_validate_only", "config.environment.stage"),
        (task, "p05_run_phase", "tuning", "config.task.p05_run_phase"),
        (task, "p05_arm_id", arm, "config.task.p05_arm_id"),
        (task, "p05_trace_export", False, "config.task.p05_trace_export"),
        (trainer, "p05_pilot_mode", False, "config.trainer.p05_pilot_mode"),
        (trainer, "expected_gpu_uuid", expected_gpu_uuid, "config.trainer.expected_gpu_uuid"),
        (trainer, "num_epochs", MAX_EPOCHS, "config.trainer.num_epochs"),
        (trainer, "early_stopping", True, "config.trainer.early_stopping"),
        (trainer, "patience", PATIENCE, "config.trainer.patience"),
        (trainer, "device", "cuda", "config.trainer.device"),
        (trainer, "accelerator", "gpu", "config.trainer.accelerator"),
        (trainer, "devices", 1, "config.trainer.devices"),
        (trainer, "gpus", 1, "config.trainer.gpus"),
        (trainer, "num_nodes", 1, "config.trainer.num_nodes"),
        (trainer, "num_processes", 1, "config.trainer.num_processes"),
        (trainer, "strategy", "auto", "config.trainer.strategy"),
        (trainer, "precision", 32, "config.trainer.precision"),
        (trainer, "deterministic", True, "config.trainer.deterministic"),
        (trainer, "monitor", "val_loss", "config.trainer.monitor"),
        (trainer, "monitor_mode", "min", "config.trainer.monitor_mode"),
        (trainer, "save_top_k", 1, "config.trainer.save_top_k"),
    )
    for container, key, expected, field_name in config_exact:
        _exact(container.get(key), expected, name=field_name)
    config_rate = _decimal_number(
        task.get("lr"),
        name="config.task.lr",
        minimum=Decimal("0"),
    )
    if config_rate != learning_rate:
        raise ValueError(f"{name} learning rate conflicts with its job")


def _validate_runtime_config(
    config_snapshot_path: str | Path,
    *,
    job: _VerifiedJob,
) -> tuple[Path, str]:
    path = _real_file(config_snapshot_path, name="P05 runtime config snapshot")
    _validate_config_contract(
        path,
        arm=job.arm_id,
        learning_rate=job.learning_rate,
        expected_gpu_uuid=job.expected_gpu_uuid,
        name="P05 runtime config snapshot",
    )
    return path, _sha256_file(path)


def _validate_provenance(value: Mapping[str, str]) -> dict[str, str]:
    provenance = _exact_keys(value, _PROVENANCE_KEYS, name="provenance")
    return {
        key: _required_hash(provenance[key], name=f"provenance.{key}")
        for key in sorted(_PROVENANCE_KEYS)
    }


def _validate_run_contract(
    manifest: Mapping[str, Any],
    *,
    job: _VerifiedJob,
    config_sha256: str,
    code_sha256: str,
    checkpoint_sha256: str,
    provenance: Mapping[str, str],
) -> None:
    _exact_keys(manifest, _RUN_CONTRACT_KEYS, name="run_contract")
    fixed = {
        "schema_version": 1,
        "paper_id": "P05",
        "dataset_id": job.dataset_id,
    }
    for key, expected in fixed.items():
        _exact(manifest[key], expected, name=f"run_contract.{key}")

    artifact_hashes = _nested_mapping(
        manifest.get("provenance"),
        name="run_contract.provenance",
    )
    if set(artifact_hashes) != {
        "checkpoint_sha256",
        "code_sha256",
        "config_sha256",
        "model_sha256",
    }:
        raise ValueError("run_contract.provenance schema drift")
    expected_artifact_hashes = {
        "checkpoint_sha256": checkpoint_sha256,
        "code_sha256": code_sha256,
        "config_sha256": config_sha256,
    }
    for key, expected in expected_artifact_hashes.items():
        observed = _required_hash(
            artifact_hashes.get(key),
            name=f"run_contract.provenance.{key}",
        )
        if observed != expected:
            raise ValueError(f"run contract {key} conflicts with candidate artifact")
    _required_hash(
        artifact_hashes.get("model_sha256"),
        name="run_contract.provenance.model_sha256",
    )

    normalization = _nested_mapping(
        manifest.get("normalization_plan"),
        name="run_contract.normalization_plan",
    )
    normalization_hash = _required_hash(
        normalization.get("sha256"),
        name="run_contract.normalization_plan.sha256",
    )
    if normalization_hash != provenance["normalization_sha256"]:
        raise ValueError("run contract normalization hash conflicts with provenance")

    weight_plans = _nested_mapping(
        manifest.get("weight_plans"),
        name="run_contract.weight_plans",
    )
    if set(weight_plans) != {"train", "validation"}:
        raise ValueError("run_contract.weight_plans must contain train and validation")
    for role, provenance_key in (
        ("train", "train_weight_plan_sha256"),
        ("validation", "validation_weight_plan_sha256"),
    ):
        plan = _nested_mapping(
            weight_plans[role],
            name=f"run_contract.weight_plans.{role}",
        )
        observed = _required_hash(
            plan.get("sha256"),
            name=f"run_contract.weight_plans.{role}.sha256",
        )
        if observed != provenance[provenance_key]:
            raise ValueError(f"run contract {role} weight hash conflicts with provenance")

    runtime = _nested_mapping(
        manifest.get("runtime_identity"),
        name="run_contract.runtime_identity",
    )
    runtime_exact = {
        "schema_version": 1,
        "paper_id": "P05",
        "evidence_mode": True,
        "cuda_visible_devices": str(job.physical_gpu_index),
        "physical_gpu_index": job.physical_gpu_index,
        "gpu_uuid": job.expected_gpu_uuid,
        "expected_gpu_uuid": job.expected_gpu_uuid,
        "identity_source": "nvidia-smi:index,uuid",
        "accelerator": "gpu",
        "devices": 1,
        "gpus": 1,
        "strategy": "auto",
        "precision": 32,
        "deterministic": True,
    }
    if set(runtime) != set(runtime_exact):
        raise ValueError("run_contract.runtime_identity schema drift")
    for key, expected in runtime_exact.items():
        _exact(runtime[key], expected, name=f"run_contract.runtime_identity.{key}")


def _semantic_candidate(
    *,
    job: _VerifiedJob,
    val_loss: Decimal,
    val_f1_macro: Decimal,
    checkpoint_epoch: int,
    epochs_completed: int,
    config_path: Path,
    config_sha256: str,
    code_path: Path,
    code_sha256: str,
    run_contract_path: Path,
    run_contract_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "paper_id": "P05",
        "protocol_bundle_sha256": PROTOCOL_BUNDLE_SHA256,
        "source_matrix_sha256": job.source_matrix_sha256,
        "job": {
            "job_id": job.job_id,
            "phase": "tuning",
            "arm_id": job.arm_id,
            "dataset": job.dataset,
            "dataset_id": job.dataset_id,
            "seed": TUNING_SEED,
            "learning_rate": _float(job.learning_rate),
        },
        "execution": {
            "status": "completed",
            "stage": "fit_validate_only",
            "evidence_eligible": False,
            "claim_decision": "not_performed",
            "data_roles_constructed": ["train", "validation"],
            "test_access_count": 0,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "epochs_completed": epochs_completed,
            "checkpoint_monitor": "val_loss",
            "checkpoint_mode": "min",
            "save_top_k": 1,
            "selected_checkpoint_count": 1,
        },
        "validation": {
            "partition": "validation",
            "checkpoint_epoch": checkpoint_epoch,
            "val_loss": _float(val_loss),
            "val_f1_macro": _float(val_f1_macro),
            "loss_definition": "group_equal_weighted_cross_entropy",
            "macro_f1_construction": "one_epoch_level_weighted_confusion_matrix",
            "weighting": "equal_group_then_equal_window",
            "zero_division": 0,
        },
        "artifacts": {
            "config_snapshot": {
                "path": str(config_path),
                "sha256": config_sha256,
            },
            "code_snapshot": {
                "path": str(code_path),
                "semantic_sha256": code_sha256,
            },
            "run_contract": {
                "path": str(run_contract_path),
                "semantic_sha256": run_contract_sha256,
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha256,
            },
        },
        "provenance": dict(provenance),
    }


def _result(
    target: Path,
    manifest: Mapping[str, Any],
    *,
    status: str,
) -> P05TuningValidationCandidateResult:
    manifest_path = target / MANIFEST_NAME
    return P05TuningValidationCandidateResult(
        package_dir=target,
        manifest_path=manifest_path,
        semantic_sha256=str(manifest["content"]["semantic_sha256"]),
        manifest_sha256=_sha256_file(manifest_path),
        status=status,
    )


def _verify_candidate_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, _ = _load_json(path, name="existing P05 tuning candidate")
    content = manifest.get("content")
    if not isinstance(content, Mapping) or set(content) != {"semantic_sha256"}:
        raise ValueError("existing P05 tuning candidate content hash is invalid")
    recorded = _required_hash(
        content["semantic_sha256"],
        name="existing candidate.content.semantic_sha256",
    )
    semantic = {key: value for key, value in manifest.items() if key != "content"}
    if _sha256_bytes(_canonical_json_bytes(semantic)) != recorded:
        raise ValueError("existing P05 tuning candidate semantic hash mismatch")
    return manifest, semantic


def _reuse_existing(
    target: Path,
    semantic_manifest: Mapping[str, Any],
) -> P05TuningValidationCandidateResult:
    if target.is_symlink() or not target.is_dir():
        raise FileExistsError(f"invalid existing P05 tuning candidate target: {target}")
    entries = {entry.name: entry for entry in target.iterdir()}
    if set(entries) != {MANIFEST_NAME}:
        raise FileExistsError(f"incomplete existing P05 tuning candidate: {target}")
    manifest_path = entries[MANIFEST_NAME]
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileExistsError(f"invalid existing P05 tuning candidate: {target}")
    try:
        manifest, existing_semantic = _verify_candidate_manifest(manifest_path)
    except (TypeError, ValueError) as exc:
        raise FileExistsError(
            f"existing P05 tuning candidate manifest is invalid: {target}"
        ) from exc
    if _canonical_json_bytes(existing_semantic) != _canonical_json_bytes(
        semantic_manifest
    ):
        raise FileExistsError(f"existing P05 tuning candidate content conflicts: {target}")
    return _result(target, manifest, status="reused")


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
        raise RuntimeError("atomic create-only export requires Linux renameat2")
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


def _write_new(
    target: Path,
    semantic_manifest: Mapping[str, Any],
) -> P05TuningValidationCandidateResult:
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"P05 tuning candidate parent must be a real directory: {parent}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(parent))
    )
    try:
        semantic_hash = _sha256_bytes(_canonical_json_bytes(semantic_manifest))
        manifest = {
            **semantic_manifest,
            "content": {"semantic_sha256": semantic_hash},
        }
        manifest_path = temporary / MANIFEST_NAME
        with manifest_path.open("xb") as handle:
            handle.write(_pretty_json_bytes(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary)
        try:
            _rename_directory_noreplace(temporary, target)
        except FileExistsError:
            return _reuse_existing(target, semantic_manifest)
        _fsync_directory(parent)
        return _result(target, manifest, status="created")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def export_p05_tuning_validation_candidate(
    package_dir: str | Path,
    *,
    materialized_job_manifest_path: str | Path,
    source_matrix_path: str | Path,
    val_loss: float,
    val_f1_macro: float,
    checkpoint_epoch: int,
    epochs_completed: int,
    data_roles_constructed: Sequence[str],
    test_access_count: int,
    config_snapshot_path: str | Path,
    code_snapshot_manifest_path: str | Path,
    run_contract_manifest_path: str | Path,
    checkpoint_path: str | Path,
    provenance: Mapping[str, str],
) -> P05TuningValidationCandidateResult:
    """Validate and atomically create one validation-only tuning candidate.

    ``val_loss`` and ``val_f1_macro`` must be the exact epoch-level weighted
    metrics obtained by re-validating the selected minimum-validation-loss
    checkpoint.  The caller must provide the observed data roles and test
    access count; anything other than train+validation and zero fails closed.
    """

    if isinstance(data_roles_constructed, (str, bytes)):
        raise TypeError("data_roles_constructed must be a sequence of role names")
    roles = list(data_roles_constructed)
    if roles != ["train", "validation"]:
        raise ValueError(
            "data_roles_constructed must be exactly ['train', 'validation']; "
            "test construction is forbidden during tuning"
        )
    _exact(test_access_count, 0, name="test_access_count")
    if type(epochs_completed) is not int or not 1 <= epochs_completed <= MAX_EPOCHS:
        raise ValueError(f"epochs_completed must be an integer in [1, {MAX_EPOCHS}]")
    if (
        type(checkpoint_epoch) is not int
        or not 0 <= checkpoint_epoch < epochs_completed
    ):
        raise ValueError("checkpoint_epoch must identify one completed zero-based epoch")
    loss = _decimal_number(
        val_loss,
        name="val_loss",
        minimum=Decimal("0"),
    )
    f1 = _decimal_number(
        val_f1_macro,
        name="val_f1_macro",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    job = _validate_materialized_job(
        materialized_job_manifest_path,
        source_matrix_path=source_matrix_path,
    )
    runtime_config_path, runtime_config_sha256 = _validate_runtime_config(
        config_snapshot_path,
        job=job,
    )
    normalized_provenance = _validate_provenance(provenance)

    code_path, code_manifest, code_sha256 = _verify_semantic_manifest(
        code_snapshot_manifest_path,
        name="P05 code snapshot",
        schema_name=CODE_SCHEMA_NAME,
    )
    _exact(code_manifest.get("schema_version"), 1, name="code_snapshot.schema_version")
    _exact(code_manifest.get("paper_id"), "P05", name="code_snapshot.paper_id")

    checkpoint = _real_file(checkpoint_path, name="P05 tuning checkpoint")
    checkpoint_sha256 = _sha256_file(checkpoint)
    run_contract_path, run_contract, run_contract_sha256 = _verify_semantic_manifest(
        run_contract_manifest_path,
        name="P05 run contract",
        schema_name=RUN_CONTRACT_SCHEMA_NAME,
    )
    _validate_run_contract(
        run_contract,
        job=job,
        config_sha256=runtime_config_sha256,
        code_sha256=code_sha256,
        checkpoint_sha256=checkpoint_sha256,
        provenance=normalized_provenance,
    )

    semantic_manifest = _semantic_candidate(
        job=job,
        val_loss=loss,
        val_f1_macro=f1,
        checkpoint_epoch=checkpoint_epoch,
        epochs_completed=epochs_completed,
        config_path=runtime_config_path,
        config_sha256=runtime_config_sha256,
        code_path=code_path,
        code_sha256=code_sha256,
        run_contract_path=run_contract_path,
        run_contract_sha256=run_contract_sha256,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        provenance=normalized_provenance,
    )
    target = Path(os.path.abspath(os.fspath(package_dir)))
    if target.is_symlink():
        raise FileExistsError(f"refusing P05 tuning candidate through symlink: {target}")
    if target.exists():
        return _reuse_existing(target, semantic_manifest)
    return _write_new(target, semantic_manifest)


__all__ = [
    "P05TuningValidationCandidateResult",
    "export_p05_tuning_validation_candidate",
]
