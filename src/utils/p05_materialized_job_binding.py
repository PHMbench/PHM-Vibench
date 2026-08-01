"""Fail-closed pre-training binding for materialized P05 GPU jobs.

This verifier is intentionally read-only.  It proves that the exact config
selected for a P05 run is still the create-only materializer output registered
by the canonical pilot, tuning, or decisive execution matrix.  Decisive jobs
also re-open and fully verify the tuning-selection manifest instead of trusting
the abbreviated selection record embedded in the job manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

from src.configs.p05_contract import (
    P05ExperimentContract,
    validate_p05_experiment_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_DIR = REPO_ROOT / "configs" / "experiments" / "p05" / "protocol"
PILOT_MATRIX_PATH = PROTOCOL_DIR / "pilot_matrix_p05_v1.yaml"
PILOT_LAUNCH_PLAN_PATH = PROTOCOL_DIR / "pilot_launch_plan_p05_v1.yaml"
TUNING_MATRIX_PATH = PROTOCOL_DIR / "neural_tuning_matrix_p05_v1.yaml"
DECISIVE_MATRIX_PATH = PROTOCOL_DIR / "neural_decisive_matrix_p05_v1.yaml"

PROTOCOL_ID = "P05-G040-v3.2"
PROTOCOL_BUNDLE_SHA256 = (
    "8d01361c39a778d437ce235ad1e8d3877313f128d6593fbb74812a4b237a1654"
)
CONFIG_NAME = "config.yaml"
MANIFEST_NAME = "manifest.json"
ARMS = ("P05-M", "P05-B0", "P05-B1", "P05-B3")
DATASET_IDS = {"CWRU": 1, "XJTU": 2}
LEARNING_RATES = (Decimal("0.001"), Decimal("0.0003"))
DECISIVE_SEEDS = (42, 123, 456, 789, 1024)
TUNING_SEED = 20260801

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_GPU_UUID_PATTERN = re.compile(r"^GPU-[!-~]{1,124}$")
_RATE_TOKENS = {
    Decimal("0.001"): "LR1E3",
    Decimal("0.0003"): "LR3E4",
}
_PILOT_MANIFEST_KEYS = frozenset(
    {
        "schema_name",
        "schema_version",
        "paper_id",
        "matrix_id",
        "job_id",
        "physical_gpu_index",
        "expected_gpu_uuid",
        "config_file",
        "config_sha256",
        "matrix_sha256",
        "launch_plan_sha256",
        "evidence_eligible",
        "claim_support",
        "scientific_overrides",
        "content",
    }
)
_NEURAL_MANIFEST_KEYS = frozenset(
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
_SELECTION_KEYS = frozenset(
    {
        "schema_name",
        "schema_version",
        "paper_id",
        "phase",
        "status",
        "claim_decision",
        "evidence_eligible",
        "test_access",
        "protocol_bundle_sha256",
        "source_matrix_sha256",
        "protocol",
        "candidates",
        "selections",
        "selection_index",
        "content",
    }
)
_SELECTION_ROW_KEYS = frozenset(
    {
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
        "selected_config_sha256",
        "selected_code_sha256",
        "selected_run_contract_sha256",
        "selected_checkpoint_sha256",
        "source_candidate_semantic_sha256",
    }
)
_SELECTION_INDEX_KEYS = frozenset(
    {
        "row_index",
        "selection_id",
        "selected_learning_rate",
        "selected_job_id",
        "selected_checkpoint_sha256",
        "selected_run_contract_sha256",
    }
)
_RUNTIME_KEYS = frozenset(
    {
        "schema_version",
        "paper_id",
        "evidence_mode",
        "cuda_visible_devices",
        "physical_gpu_index",
        "gpu_uuid",
        "expected_gpu_uuid",
        "identity_source",
        "accelerator",
        "devices",
        "gpus",
        "strategy",
        "precision",
        "deterministic",
    }
)


@dataclass(frozen=True)
class P05MaterializedJobBinding:
    """Immutable identity record returned after a successful preflight."""

    config_path: Path
    config_sha256: str
    materialized_manifest_path: Path
    materialized_manifest_sha256: str
    materialized_manifest_semantic_sha256: str
    matrix_path: Path
    matrix_sha256: str
    launch_plan_path: Path | None
    launch_plan_sha256: str | None
    job_id: str
    phase: str
    arm_id: str
    dataset: str
    dataset_id: int
    seed: int
    learning_rate: float
    physical_gpu_index: int
    gpu_uuid: str
    evidence_eligible: bool
    tuning_selection_path: Path | None
    tuning_selection_sha256: str | None
    tuning_selection_semantic_sha256: str | None
    selected_tuning_job_id: str | None
    selected_checkpoint_sha256: str | None
    selected_run_contract_sha256: str | None


@dataclass(frozen=True)
class _MatrixBinding:
    path: Path
    sha256: str
    job: dict[str, Any]
    physical_gpu_index: int
    launch_plan_path: Path | None = None
    launch_plan_sha256: str | None = None


@dataclass(frozen=True)
class _SelectionBinding:
    path: Path
    sha256: str
    semantic_sha256: str
    selected_job_id: str
    selected_checkpoint_sha256: str
    selected_run_contract_sha256: str
    selected_learning_rate: Decimal


class _UniqueSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


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


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


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


def _verify_semantic_json(
    path: Path,
    *,
    name: str,
) -> tuple[dict[str, Any], str, str]:
    manifest, payload = _load_json(path, name=name)
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
    return manifest, actual, _sha256_bytes(payload)


def _decimal(
    value: Any,
    *,
    name: str,
    allowed: Sequence[Decimal] | None = None,
) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a JSON number")
    try:
        converted = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not converted.is_finite():
        raise ValueError(f"{name} must be finite")
    if allowed is not None and converted not in allowed:
        raise ValueError(f"{name} is not a registered value")
    return converted


def _reject_mutation_inputs(
    *,
    cli_overrides: Sequence[str] | None,
    local_config: Any,
) -> None:
    if cli_overrides is not None:
        if isinstance(cli_overrides, (str, bytes)):
            if cli_overrides:
                raise ValueError("P05 evidence jobs forbid every CLI override")
        else:
            try:
                overrides = list(cli_overrides)
            except TypeError as exc:
                raise TypeError("cli_overrides must be a sequence or None") from exc
            if overrides:
                raise ValueError("P05 evidence jobs forbid every CLI override")
    if local_config is None:
        return
    if isinstance(local_config, (str, bytes)) and not local_config:
        return
    try:
        if not isinstance(local_config, (str, bytes)) and len(local_config) == 0:
            return
    except TypeError:
        pass
    raise ValueError("P05 evidence jobs forbid every non-empty local_config")


def _materializer_package(config_path_input: str | Path) -> tuple[Path, Path]:
    raw_config = Path(os.path.abspath(os.fspath(config_path_input)))
    if raw_config.name != CONFIG_NAME:
        raise ValueError("P05 evidence config_path must name materializer config.yaml")
    package = raw_config.parent
    if package.is_symlink() or not package.is_dir():
        raise ValueError("P05 materializer package must be a real directory")
    config_path = _real_file(raw_config, name="P05 materialized config")
    if config_path.parent != package.resolve(strict=True):
        raise ValueError("P05 config_path must remain inside its materializer package")
    entries = {entry.name: entry for entry in package.iterdir()}
    if set(entries) != {CONFIG_NAME, MANIFEST_NAME}:
        raise ValueError(
            "P05 materializer package must contain only config.yaml and manifest.json"
        )
    manifest_path = _real_file(entries[MANIFEST_NAME], name="P05 materialized manifest")
    return config_path, manifest_path


def _validate_runtime_identity(
    value: Mapping[str, Any],
    *,
    physical_gpu_index: int,
    expected_gpu_uuid: str,
) -> dict[str, Any]:
    runtime = _exact_keys(value, _RUNTIME_KEYS, name="runtime_identity")
    exact = {
        "schema_version": 1,
        "paper_id": "P05",
        "evidence_mode": True,
        "cuda_visible_devices": str(physical_gpu_index),
        "physical_gpu_index": physical_gpu_index,
        "gpu_uuid": expected_gpu_uuid,
        "expected_gpu_uuid": expected_gpu_uuid,
        "identity_source": "nvidia-smi:index,uuid",
        "accelerator": "gpu",
        "devices": 1,
        "gpus": 1,
        "strategy": "auto",
        "precision": 32,
        "deterministic": True,
    }
    for key, expected in exact.items():
        _exact(runtime[key], expected, name=f"runtime_identity.{key}")
    return runtime


def _validate_contract(value: Any) -> P05ExperimentContract:
    if not isinstance(value, P05ExperimentContract):
        raise TypeError("experiment_contract must be a P05ExperimentContract")
    if value.arm_id not in ARMS:
        raise ValueError("experiment_contract arm is not registered")
    if value.dataset not in DATASET_IDS:
        raise ValueError("experiment_contract dataset is not registered")
    _exact(
        value.dataset_id,
        DATASET_IDS[value.dataset],
        name="experiment_contract.dataset_id",
    )
    if value.phase not in {"pilot", "tuning", "decisive"}:
        raise ValueError("experiment_contract phase is not registered")
    if type(value.seed) is not int:
        raise TypeError("experiment_contract.seed must be an integer")
    if type(value.trace_export) is not bool:
        raise TypeError("experiment_contract.trace_export must be boolean")
    return value


def _config_binding(
    config_path: Path,
    *,
    config_sha256: str,
    contract: P05ExperimentContract,
    expected_gpu_uuid: str,
) -> Decimal:
    if _sha256_file(config_path) != config_sha256:
        raise ValueError("P05 materialized config raw SHA-256 mismatch")
    config = _load_yaml(config_path, name="P05 materialized config")
    environment = _mapping(config.get("environment"), name="config.environment")
    data = _mapping(config.get("data"), name="config.data")
    model = _mapping(config.get("model"), name="config.model")
    task = _mapping(config.get("task"), name="config.task")
    trainer = _mapping(config.get("trainer"), name="config.trainer")
    derived_contract = validate_p05_experiment_contract(
        environment,
        data,
        model,
        task,
        trainer,
        object(),
    )
    if derived_contract != contract:
        raise ValueError(
            "P05 materialized config does not reproduce experiment_contract"
        )
    exact = (
        (environment.get("seed"), contract.seed, "config.environment.seed"),
        (task.get("p05_arm_id"), contract.arm_id, "config.task.p05_arm_id"),
        (task.get("p05_run_phase"), contract.phase, "config.task.p05_run_phase"),
        (
            task.get("target_system_id"),
            [contract.dataset_id],
            "config.task.target_system_id",
        ),
        (
            task.get("p05_trace_export"),
            contract.trace_export,
            "config.task.p05_trace_export",
        ),
        (task.get("p05_evidence_mode"), True, "config.task.p05_evidence_mode"),
        (
            trainer.get("expected_gpu_uuid"),
            expected_gpu_uuid,
            "config.trainer.expected_gpu_uuid",
        ),
        (trainer.get("p05_evidence_mode"), True, "config.trainer.p05_evidence_mode"),
        (trainer.get("device"), "cuda", "config.trainer.device"),
        (trainer.get("accelerator"), "gpu", "config.trainer.accelerator"),
        (trainer.get("devices"), 1, "config.trainer.devices"),
        (trainer.get("gpus"), 1, "config.trainer.gpus"),
        (trainer.get("num_nodes"), 1, "config.trainer.num_nodes"),
        (trainer.get("num_processes"), 1, "config.trainer.num_processes"),
        (trainer.get("strategy"), "auto", "config.trainer.strategy"),
        (trainer.get("precision"), 32, "config.trainer.precision"),
        (trainer.get("deterministic"), True, "config.trainer.deterministic"),
    )
    for observed, expected, name in exact:
        _exact(observed, expected, name=name)
    learning_rate = _decimal(
        task.get("lr"),
        name="config.task.lr",
        allowed=LEARNING_RATES,
    )
    return learning_rate


def _matrix_file(path: Path, *, name: str) -> tuple[Path, dict[str, Any], str]:
    real = _real_file(path, name=name)
    return real, _load_yaml(real, name=name), _sha256_file(real)


def _wave_assignments(
    waves: Any,
    *,
    expected_job_ids: set[str],
    expected_wave_count: int,
    name: str,
) -> dict[str, int]:
    if not isinstance(waves, list) or len(waves) != expected_wave_count:
        raise ValueError(f"{name} must contain exactly {expected_wave_count} waves")
    assignments: dict[str, int] = {}
    for expected_wave, wave in enumerate(waves, start=1):
        row = _mapping(wave, name=f"{name} wave")
        _exact(row.get("wave"), expected_wave, name=f"{name} wave number")
        concurrent = row.get("concurrent_jobs")
        if not isinstance(concurrent, list) or len(concurrent) != 2:
            raise ValueError(f"each {name} wave must contain exactly two jobs")
        indices: set[int] = set()
        for raw in concurrent:
            item = _exact_keys(
                raw,
                frozenset({"job_id", "physical_gpu_index"}),
                name=f"{name} assignment",
            )
            job_id = item["job_id"]
            index = item["physical_gpu_index"]
            if job_id not in expected_job_ids or job_id in assignments:
                raise ValueError(f"{name} must assign every registered job once")
            if type(index) is not int or index not in {0, 1}:
                raise ValueError(f"{name} may use only physical GPU0 or GPU1")
            assignments[job_id] = index
            indices.add(index)
        if indices != {0, 1}:
            raise ValueError(f"each {name} wave must use GPU0 and GPU1 exactly once")
    if set(assignments) != expected_job_ids:
        raise ValueError(f"{name} does not cover the registered job matrix")
    return assignments


def _pilot_matrix_binding(job_id: str) -> _MatrixBinding:
    matrix_path, matrix, matrix_hash = _matrix_file(
        PILOT_MATRIX_PATH,
        name="canonical P05 pilot matrix",
    )
    identity = {
        "schema_version": 1,
        "kind": "p05_frozen_pilot_matrix",
        "paper_id": "P05",
        "protocol_id": PROTOCOL_ID,
        "matrix_id": "P05-PILOT-v1",
        "status": "frozen_declarative",
        "evidence_eligible": False,
    }
    _exact_keys(
        matrix,
        frozenset(
            {
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
        ),
        name="pilot matrix",
    )
    for key, expected in identity.items():
        _exact(matrix.get(key), expected, name=f"pilot_matrix.{key}")
    jobs = matrix.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 4:
        raise ValueError("pilot matrix must contain exactly four jobs")
    expected_ids = {
        f"P05-PILOT-{arm[4:]}-{dataset}"
        for arm in ("P05-B0", "P05-M")
        for dataset in DATASET_IDS
    }
    job_rows = [
        _exact_keys(
            job,
            frozenset({"id", "arm", "dataset", "config"}),
            name="pilot matrix job",
        )
        for job in jobs
    ]
    if {job.get("id") for job in job_rows} != expected_ids:
        raise ValueError("pilot matrix jobs do not form the exact 2x2 design")
    for row in job_rows:
        arm = row["arm"]
        dataset = row["dataset"]
        if arm not in {"P05-B0", "P05-M"} or dataset not in DATASET_IDS:
            raise ValueError("pilot matrix contains an unregistered arm or dataset")
        _exact(
            row["id"],
            f"P05-PILOT-{arm[4:]}-{dataset}",
            name="pilot matrix job ID",
        )
    matches = [job for job in job_rows if job.get("id") == job_id]
    if len(matches) != 1:
        raise ValueError("pilot materialized job does not bind one matrix row")

    gate = _mapping(matrix.get("launch_gate"), name="pilot_matrix.launch_gate")
    _exact(
        gate.get("launch_plan_path"),
        "configs/experiments/p05/protocol/pilot_launch_plan_p05_v1.yaml",
        name="pilot_matrix.launch_gate.launch_plan_path",
    )
    launch_path, launch, launch_hash = _matrix_file(
        PILOT_LAUNCH_PLAN_PATH,
        name="canonical P05 pilot launch plan",
    )
    launch_identity = {
        "schema_version": 1,
        "kind": "p05_frozen_pilot_launch_plan",
        "paper_id": "P05",
        "protocol_id": PROTOCOL_ID,
        "matrix_id": "P05-PILOT-v1",
        "status": "frozen_awaiting_physical_gpu_uuid_binding",
        "evidence_eligible": False,
        "claim_support": "forbidden",
    }
    _exact_keys(
        launch,
        frozenset(
            {
                "schema_version",
                "kind",
                "paper_id",
                "protocol_id",
                "matrix_id",
                "status",
                "evidence_eligible",
                "claim_support",
                "runtime",
                "blocking",
                "execution_waves",
                "launch_command_contract",
                "unresolved_launch_inputs",
            }
        ),
        name="pilot launch plan",
    )
    for key, expected in launch_identity.items():
        _exact(launch.get(key), expected, name=f"pilot_launch_plan.{key}")
    assignments = _wave_assignments(
        launch.get("execution_waves"),
        expected_job_ids=expected_ids,
        expected_wave_count=2,
        name="pilot launch plan",
    )
    return _MatrixBinding(
        path=matrix_path,
        sha256=matrix_hash,
        job=matches[0],
        physical_gpu_index=assignments[job_id],
        launch_plan_path=launch_path,
        launch_plan_sha256=launch_hash,
    )


def _neural_matrix_binding(phase: str, job_id: str) -> _MatrixBinding:
    path = TUNING_MATRIX_PATH if phase == "tuning" else DECISIVE_MATRIX_PATH
    matrix_path, matrix, matrix_hash = _matrix_file(
        path,
        name=f"canonical P05 {phase} matrix",
    )
    identity = {
        "tuning": (
            "p05_frozen_neural_tuning_execution_matrix",
            "P05-NEURAL-TUNING-v1",
            "frozen_declarative_awaiting_gpu_uuid_binding",
            16,
            8,
        ),
        "decisive": (
            "p05_frozen_neural_decisive_execution_matrix",
            "P05-NEURAL-DECISIVE-v1",
            "frozen_declarative_awaiting_gpu_uuid_and_tuning_selection_binding",
            40,
            20,
        ),
    }[phase]
    fixed = {
        "schema_version": 1,
        "kind": identity[0],
        "paper_id": "P05",
        "protocol_id": PROTOCOL_ID,
        "matrix_id": identity[1],
        "status": identity[2],
    }
    _exact_keys(
        matrix,
        frozenset(
            {
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
        ),
        name=f"{phase} matrix",
    )
    for key, expected in fixed.items():
        _exact(matrix.get(key), expected, name=f"{phase}_matrix.{key}")
    jobs = matrix.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != identity[3]:
        raise ValueError(f"{phase} matrix has the wrong job count")
    rows = [_mapping(job, name=f"{phase} matrix job") for job in jobs]
    ids = [job.get("id") for job in rows]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError(f"{phase} matrix job IDs must be non-empty strings")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{phase} matrix job IDs must be unique")

    observed_cells: set[tuple[Any, ...]] = set()
    for row in rows:
        expected_job_keys = (
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
            )
            if phase == "tuning"
            else frozenset(
                {
                    "id",
                    "arm",
                    "dataset",
                    "seed",
                    "tuning_selection_key",
                    "output_dir",
                    "materialize_command",
                }
            )
        )
        _exact_keys(row, expected_job_keys, name=f"{phase} matrix job")
        arm = row.get("arm")
        dataset = row.get("dataset")
        if arm not in ARMS or dataset not in DATASET_IDS:
            raise ValueError(f"{phase} matrix contains an unregistered cell")
        short_arm = arm[4:]
        if phase == "tuning":
            rate = _decimal(
                row.get("learning_rate"),
                name="tuning matrix learning_rate",
                allowed=LEARNING_RATES,
            )
            _exact(row.get("seed"), TUNING_SEED, name="tuning matrix seed")
            expected_id = f"P05-TUNE-{short_arm}-{dataset}-{_RATE_TOKENS[rate]}"
            observed_cells.add((arm, dataset, rate))
        else:
            seed = row.get("seed")
            if type(seed) is not int or seed not in DECISIVE_SEEDS:
                raise ValueError("decisive matrix seed is not registered")
            if "learning_rate" in row:
                raise ValueError("decisive matrix must not contain learning_rate")
            _exact(
                row.get("tuning_selection_key"),
                f"{dataset}/{arm}",
                name="decisive matrix tuning_selection_key",
            )
            expected_id = f"P05-DEC-{short_arm}-{dataset}-S{seed}"
            observed_cells.add((arm, dataset, seed))
        _exact(row.get("id"), expected_id, name=f"{phase} matrix job ID")
    expected_cells = (
        {
            (arm, dataset, rate)
            for arm in ARMS
            for dataset in DATASET_IDS
            for rate in LEARNING_RATES
        }
        if phase == "tuning"
        else {
            (arm, dataset, seed)
            for arm in ARMS
            for dataset in DATASET_IDS
            for seed in DECISIVE_SEEDS
        }
    )
    if observed_cells != expected_cells:
        raise ValueError(f"{phase} matrix does not contain the exact factorial")
    pilot_reference = _exact_keys(
        matrix["pilot_common_contract"],
        frozenset({"path", "sha256", "selector", "reused_sections", "hash_mismatch"}),
        name=f"{phase} matrix pilot_common_contract",
    )
    _exact(
        pilot_reference["path"],
        "configs/experiments/p05/protocol/pilot_matrix_p05_v1.yaml",
        name=f"{phase} matrix pilot path",
    )
    _exact(
        pilot_reference["selector"],
        "common_config",
        name=f"{phase} matrix pilot selector",
    )
    _exact(
        pilot_reference["hash_mismatch"],
        "hard_error",
        name=f"{phase} matrix pilot hash policy",
    )
    canonical_pilot = _real_file(
        PILOT_MATRIX_PATH,
        name="canonical P05 pilot matrix",
    )
    _exact(
        _required_hash(
            pilot_reference["sha256"],
            name=f"{phase} matrix pilot SHA-256",
        ),
        _sha256_file(canonical_pilot),
        name=f"{phase} matrix pilot actual SHA-256",
    )
    assignments = _wave_assignments(
        matrix.get("execution_waves"),
        expected_job_ids=set(ids),
        expected_wave_count=identity[4],
        name=f"{phase} execution matrix",
    )
    matches = [row for row in rows if row["id"] == job_id]
    if len(matches) != 1:
        raise ValueError(f"materialized job does not bind one {phase} matrix row")
    return _MatrixBinding(
        path=matrix_path,
        sha256=matrix_hash,
        job=matches[0],
        physical_gpu_index=assignments[job_id],
    )


def _selection_path(materialized_manifest_path: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError("tuning_selection.path must be non-empty path text")
    path = Path(raw)
    if not path.is_absolute():
        path = materialized_manifest_path.parent / path
    return _real_file(path, name="P05 tuning-selection manifest")


def _verify_materialized_pilot_common(value: Any) -> None:
    summary = _exact_keys(
        value,
        frozenset({"path", "pilot_matrix_sha256", "common_config_sha256"}),
        name="materialized_job.pilot_common_contract",
    )
    canonical_path, pilot, pilot_hash = _matrix_file(
        PILOT_MATRIX_PATH,
        name="canonical P05 pilot matrix",
    )
    raw_path = summary["path"]
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        raise ValueError("materialized pilot common-contract path is invalid")
    referenced = Path(raw_path)
    if not referenced.is_absolute():
        referenced = REPO_ROOT / referenced
    if _real_file(referenced, name="materialized pilot common-contract path") != canonical_path:
        raise ValueError("materialized job does not reference the canonical pilot matrix")
    _exact(
        _required_hash(
            summary["pilot_matrix_sha256"],
            name="materialized pilot_matrix_sha256",
        ),
        pilot_hash,
        name="materialized pilot matrix actual SHA-256",
    )
    common_config = _mapping(
        pilot.get("common_config"),
        name="canonical pilot common_config",
    )
    _exact(
        _required_hash(
            summary["common_config_sha256"],
            name="materialized common_config_sha256",
        ),
        _sha256_bytes(_canonical_json_bytes(common_config)),
        name="materialized pilot common_config actual SHA-256",
    )


def _verify_selection(
    summary_value: Any,
    *,
    materialized_manifest_path: Path,
    dataset: str,
    arm_id: str,
    decisive_matrix_job: Mapping[str, Any],
    decisive_learning_rate: Decimal,
) -> _SelectionBinding:
    summary = _exact_keys(
        summary_value,
        frozenset(
            {
                "path",
                "sha256",
                "source_matrix_sha256",
                "key",
                "row_index",
                "selected_learning_rate",
                "selected_job_id",
                "selected_checkpoint_sha256",
                "selected_run_contract_sha256",
            }
        ),
        name="materialized_job.tuning_selection",
    )
    path = _selection_path(materialized_manifest_path, summary["path"])
    direct_hash = _required_hash(summary["sha256"], name="tuning_selection.sha256")
    if _sha256_file(path) != direct_hash:
        raise ValueError("P05 tuning-selection direct SHA-256 mismatch")
    manifest, semantic_hash, observed_direct = _verify_semantic_json(
        path,
        name="P05 tuning-selection manifest",
    )
    if observed_direct != direct_hash:
        raise ValueError("P05 tuning-selection direct hash changed during verification")
    _exact_keys(manifest, _SELECTION_KEYS, name="tuning-selection manifest")
    scientific_state = {
        "schema_name": "p05.tuning_selection",
        "schema_version": 1,
        "paper_id": "P05",
        "phase": "tuning_selection",
        "status": "computed_unadjudicated",
        "claim_decision": "not_performed",
        "evidence_eligible": False,
        "test_access": "forbidden_and_not_performed",
        "protocol_bundle_sha256": PROTOCOL_BUNDLE_SHA256,
    }
    for key, expected in scientific_state.items():
        _exact(manifest[key], expected, name=f"tuning_selection.{key}")

    _, _, tuning_matrix_hash = _matrix_file(
        TUNING_MATRIX_PATH,
        name="canonical P05 tuning matrix",
    )
    selection_source_hash = _required_hash(
        manifest["source_matrix_sha256"],
        name="tuning_selection.source_matrix_sha256",
    )
    if selection_source_hash != tuning_matrix_hash:
        raise ValueError("tuning selection is not bound to the canonical tuning matrix")
    if _required_hash(
        summary["source_matrix_sha256"],
        name="materialized tuning_selection.source_matrix_sha256",
    ) != tuning_matrix_hash:
        raise ValueError("materialized selection summary has the wrong tuning matrix")

    selections = manifest["selections"]
    index = manifest["selection_index"]
    expected_keys = {f"{dataset_name}/{arm}" for dataset_name in DATASET_IDS for arm in ARMS}
    if not isinstance(selections, list) or len(selections) != 8:
        raise ValueError("tuning selection must contain exactly eight selection rows")
    if not isinstance(index, Mapping) or set(index) != expected_keys:
        raise ValueError("tuning selection index must cover all eight dataset/arm cells")
    rows: dict[str, dict[str, Any]] = {}
    for row_number, raw_row in enumerate(selections):
        row = _exact_keys(raw_row, _SELECTION_ROW_KEYS, name="tuning selection row")
        row_arm = row["arm_id"]
        row_dataset = row["dataset"]
        key = f"{row_dataset}/{row_arm}"
        if row_arm not in ARMS or row_dataset not in DATASET_IDS or key in rows:
            raise ValueError("tuning selection rows must uniquely cover registered cells")
        _exact(
            row["dataset_id"],
            DATASET_IDS[row_dataset],
            name="tuning selection row dataset_id",
        )
        rate = _decimal(
            row["selected_learning_rate"],
            name="tuning selection selected_learning_rate",
            allowed=LEARNING_RATES,
        )
        expected_job = (
            f"P05-TUNE-{row_arm[4:]}-{row_dataset}-{_RATE_TOKENS[rate]}"
        )
        _exact(
            row["selected_job_id"],
            expected_job,
            name="tuning selection selected_job_id",
        )
        for hash_key in (
            "selected_config_sha256",
            "selected_code_sha256",
            "selected_run_contract_sha256",
            "selected_checkpoint_sha256",
            "source_candidate_semantic_sha256",
        ):
            _required_hash(row[hash_key], name=f"tuning selection row {hash_key}")
        checkpoint_epoch = row["selected_checkpoint_epoch"]
        if type(checkpoint_epoch) is not int or checkpoint_epoch < 0:
            raise ValueError("tuning selection checkpoint epoch must be non-negative")
        for text_key in ("selection_id", "selection_reason"):
            text_value = row[text_key]
            if not isinstance(text_value, str) or not text_value:
                raise ValueError(f"tuning selection {text_key} must be non-empty")
        for metric in ("selected_val_f1_macro", "selected_val_loss"):
            value = row[metric]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"tuning selection {metric} must be finite")
        if not 0.0 <= float(row["selected_val_f1_macro"]) <= 1.0:
            raise ValueError("tuning selection selected_val_f1_macro must be in [0, 1]")
        if float(row["selected_val_loss"]) < 0.0:
            raise ValueError("tuning selection selected_val_loss must be non-negative")
        entry = _exact_keys(
            index[key],
            _SELECTION_INDEX_KEYS,
            name="tuning selection index entry",
        )
        expected_index = {
            "row_index": row_number,
            "selection_id": row["selection_id"],
            "selected_learning_rate": row["selected_learning_rate"],
            "selected_job_id": row["selected_job_id"],
            "selected_checkpoint_sha256": row["selected_checkpoint_sha256"],
            "selected_run_contract_sha256": row["selected_run_contract_sha256"],
        }
        if entry != expected_index:
            raise ValueError("tuning selection index conflicts with its selection row")
        rows[key] = row
    if set(rows) != expected_keys:
        raise ValueError("tuning selection rows do not cover all registered cells")

    selected_key = f"{dataset}/{arm_id}"
    _exact(
        decisive_matrix_job.get("tuning_selection_key"),
        selected_key,
        name="decisive matrix tuning_selection_key",
    )
    selected_row = rows[selected_key]
    selected_index = _mapping(index[selected_key], name="selected tuning index")
    summary_exact = {
        "key": selected_key,
        "row_index": selected_index["row_index"],
        "selected_learning_rate": selected_row["selected_learning_rate"],
        "selected_job_id": selected_row["selected_job_id"],
        "selected_checkpoint_sha256": selected_row["selected_checkpoint_sha256"],
        "selected_run_contract_sha256": selected_row["selected_run_contract_sha256"],
    }
    for key, expected in summary_exact.items():
        _exact(summary[key], expected, name=f"materialized tuning_selection.{key}")
    selected_rate = _decimal(
        selected_row["selected_learning_rate"],
        name="selected tuning learning rate",
        allowed=LEARNING_RATES,
    )
    if selected_rate != decisive_learning_rate:
        raise ValueError("decisive learning rate conflicts with verified tuning selection")
    return _SelectionBinding(
        path=path,
        sha256=direct_hash,
        semantic_sha256=semantic_hash,
        selected_job_id=selected_row["selected_job_id"],
        selected_checkpoint_sha256=selected_row["selected_checkpoint_sha256"],
        selected_run_contract_sha256=selected_row["selected_run_contract_sha256"],
        selected_learning_rate=selected_rate,
    )


def verify_p05_materialized_job_binding(
    *,
    config_path: str | Path,
    experiment_contract: P05ExperimentContract,
    runtime_identity: Mapping[str, Any],
    cli_overrides: Sequence[str] | None,
    local_config: Any,
) -> P05MaterializedJobBinding:
    """Verify one exact materializer package immediately before training.

    The function has no write path.  Any CLI/local override, symlink, hash
    mismatch, unsafe GPU binding, matrix drift, or incomplete decisive tuning
    selection raises before a result record is returned.
    """

    _reject_mutation_inputs(
        cli_overrides=cli_overrides,
        local_config=local_config,
    )
    contract = _validate_contract(experiment_contract)
    materialized_config, manifest_path = _materializer_package(config_path)
    manifest, semantic_hash, manifest_hash = _verify_semantic_json(
        manifest_path,
        name="P05 materialized job manifest",
    )

    expected_schema = (
        "p05.materialized_pilot_config"
        if contract.phase == "pilot"
        else "p05.materialized_neural_job"
    )
    _exact(manifest.get("schema_name"), expected_schema, name="materialized schema_name")
    expected_keys = (
        _PILOT_MANIFEST_KEYS if contract.phase == "pilot" else _NEURAL_MANIFEST_KEYS
    )
    _exact_keys(manifest, expected_keys, name="materialized job manifest")
    _exact(manifest["schema_version"], 1, name="materialized schema_version")
    _exact(manifest["paper_id"], "P05", name="materialized paper_id")
    _exact(manifest["config_file"], CONFIG_NAME, name="materialized config_file")
    _exact(
        manifest["scientific_overrides"],
        "forbidden",
        name="materialized scientific_overrides",
    )
    config_hash = _required_hash(
        manifest["config_sha256"],
        name="materialized config_sha256",
    )
    physical_index = manifest["physical_gpu_index"]
    if type(physical_index) is not int or physical_index not in {0, 1}:
        raise ValueError("materialized job may use only physical GPU0 or GPU1")
    expected_uuid = manifest["expected_gpu_uuid"]
    if (
        not isinstance(expected_uuid, str)
        or _GPU_UUID_PATTERN.fullmatch(expected_uuid) is None
        or "REQUIRED" in expected_uuid
    ):
        raise ValueError("materialized expected_gpu_uuid is not an observed GPU-* UUID")
    _validate_runtime_identity(
        runtime_identity,
        physical_gpu_index=physical_index,
        expected_gpu_uuid=expected_uuid,
    )
    learning_rate = _config_binding(
        materialized_config,
        config_sha256=config_hash,
        contract=contract,
        expected_gpu_uuid=expected_uuid,
    )

    selection: _SelectionBinding | None = None
    if contract.phase == "pilot":
        fixed = {
            "matrix_id": "P05-PILOT-v1",
            "evidence_eligible": False,
            "claim_support": "forbidden",
        }
        for key, expected in fixed.items():
            _exact(manifest[key], expected, name=f"materialized pilot {key}")
        matrix = _pilot_matrix_binding(manifest["job_id"])
        _exact(learning_rate, Decimal("0.001"), name="pilot learning_rate")
        _exact(contract.seed, TUNING_SEED, name="pilot seed")
        if contract.arm_id not in {"P05-M", "P05-B0"}:
            raise ValueError("pilot contract arm must be P05-M or P05-B0")
        expected_job_id = f"P05-PILOT-{contract.arm_id[4:]}-{contract.dataset}"
        _exact(manifest["job_id"], expected_job_id, name="materialized pilot job_id")
        _exact(matrix.job.get("arm"), contract.arm_id, name="pilot matrix job arm")
        _exact(
            matrix.job.get("dataset"),
            contract.dataset,
            name="pilot matrix job dataset",
        )
        _exact(
            manifest["launch_plan_sha256"],
            matrix.launch_plan_sha256,
            name="materialized pilot launch_plan_sha256",
        )
        evidence_eligible = False
    else:
        phase = contract.phase
        matrix = _neural_matrix_binding(phase, manifest["job_id"])
        _verify_materialized_pilot_common(manifest["pilot_common_contract"])
        fixed = {
            "protocol_id": PROTOCOL_ID,
            "matrix_id": (
                "P05-NEURAL-TUNING-v1"
                if phase == "tuning"
                else "P05-NEURAL-DECISIVE-v1"
            ),
            "stage": phase,
            "arm": contract.arm_id,
            "dataset": contract.dataset,
            "seed": contract.seed,
            "materialization_status": "created-not-executed",
            "execution_status": "not_started",
            "evidence_status": "unadjudicated",
            "claim_support": "forbidden_before_ledger_and_audit",
        }
        for key, expected in fixed.items():
            _exact(manifest[key], expected, name=f"materialized neural {key}")
        manifest_rate = _decimal(
            manifest["learning_rate"],
            name="materialized neural learning_rate",
            allowed=LEARNING_RATES,
        )
        if manifest_rate != learning_rate:
            raise ValueError("materialized manifest and config learning rates conflict")
        if phase == "tuning":
            _exact(contract.seed, TUNING_SEED, name="tuning seed")
            _exact(
                manifest["learning_rate_source"],
                "frozen_tuning_matrix_job",
                name="tuning learning_rate_source",
            )
            _exact(manifest["tuning_selection"], None, name="tuning selection")
            matrix_rate = _decimal(
                matrix.job.get("learning_rate"),
                name="tuning matrix learning_rate",
                allowed=LEARNING_RATES,
            )
            if matrix_rate != learning_rate:
                raise ValueError("tuning config learning rate conflicts with matrix job")
            expected_job_id = (
                f"P05-TUNE-{contract.arm_id[4:]}-{contract.dataset}-"
                f"{_RATE_TOKENS[learning_rate]}"
            )
        else:
            if contract.seed not in DECISIVE_SEEDS:
                raise ValueError("decisive seed is not registered")
            _exact(
                manifest["learning_rate_source"],
                "bound_hash_verified_tuning_selection_manifest",
                name="decisive learning_rate_source",
            )
            expected_job_id = (
                f"P05-DEC-{contract.arm_id[4:]}-{contract.dataset}-S{contract.seed}"
            )
            selection = _verify_selection(
                manifest["tuning_selection"],
                materialized_manifest_path=manifest_path,
                dataset=contract.dataset,
                arm_id=contract.arm_id,
                decisive_matrix_job=matrix.job,
                decisive_learning_rate=learning_rate,
            )
        _exact(manifest["job_id"], expected_job_id, name="materialized neural job_id")
        _exact(matrix.job.get("arm"), contract.arm_id, name="neural matrix job arm")
        _exact(
            matrix.job.get("dataset"),
            contract.dataset,
            name="neural matrix job dataset",
        )
        _exact(matrix.job.get("seed"), contract.seed, name="neural matrix job seed")
        evidence_eligible = False

    registered_matrix_hash = _required_hash(
        manifest["matrix_sha256"],
        name="materialized matrix_sha256",
    )
    if registered_matrix_hash != matrix.sha256:
        raise ValueError("materialized job matrix SHA-256 mismatch")
    _exact(
        physical_index,
        matrix.physical_gpu_index,
        name="materialized job physical_gpu_index",
    )
    return P05MaterializedJobBinding(
        config_path=materialized_config,
        config_sha256=config_hash,
        materialized_manifest_path=manifest_path,
        materialized_manifest_sha256=manifest_hash,
        materialized_manifest_semantic_sha256=semantic_hash,
        matrix_path=matrix.path,
        matrix_sha256=matrix.sha256,
        launch_plan_path=matrix.launch_plan_path,
        launch_plan_sha256=matrix.launch_plan_sha256,
        job_id=manifest["job_id"],
        phase=contract.phase,
        arm_id=contract.arm_id,
        dataset=contract.dataset,
        dataset_id=contract.dataset_id,
        seed=contract.seed,
        learning_rate=float(learning_rate),
        physical_gpu_index=physical_index,
        gpu_uuid=expected_uuid,
        evidence_eligible=evidence_eligible,
        tuning_selection_path=selection.path if selection is not None else None,
        tuning_selection_sha256=selection.sha256 if selection is not None else None,
        tuning_selection_semantic_sha256=(
            selection.semantic_sha256 if selection is not None else None
        ),
        selected_tuning_job_id=(
            selection.selected_job_id if selection is not None else None
        ),
        selected_checkpoint_sha256=(
            selection.selected_checkpoint_sha256 if selection is not None else None
        ),
        selected_run_contract_sha256=(
            selection.selected_run_contract_sha256 if selection is not None else None
        ),
    )


__all__ = [
    "P05MaterializedJobBinding",
    "verify_p05_materialized_job_binding",
]
