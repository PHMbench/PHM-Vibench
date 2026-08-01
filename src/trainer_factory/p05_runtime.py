"""Fail-closed single-GPU runtime contract for P05 evidence runs."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

import torch


class P05RuntimeContractError(RuntimeError):
    """Raised before trainer construction when the P05 runtime is not exact."""


@dataclass(frozen=True)
class NvidiaGpuIdentity:
    """Physical GPU identity reported by ``nvidia-smi``."""

    physical_index: int
    uuid: str


@dataclass(frozen=True)
class P05RuntimeContract:
    """Trainer kwargs and JSON-serializable provenance for an accepted preflight."""

    trainer_kwargs: Dict[str, Any]
    runtime_identity: Dict[str, Any]


_SINGLE_PROCESS_SIZE_VARS = (
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "SLURM_NTASKS",
    "OMPI_COMM_WORLD_SIZE",
)
_SINGLE_PROCESS_RANK_VARS = (
    "RANK",
    "LOCAL_RANK",
    "SLURM_PROCID",
    "OMPI_COMM_WORLD_RANK",
)


def p05_evidence_mode_enabled(args_trainer: Any) -> bool:
    """Return the exact evidence-mode flag, rejecting truthy lookalikes."""

    value = getattr(args_trainer, "p05_evidence_mode", False)
    if type(value) is not bool:
        raise P05RuntimeContractError(
            "trainer.p05_evidence_mode must be a literal boolean"
        )
    return value


def query_nvidia_smi_gpu(
    physical_index: int,
    *,
    runner: Optional[Callable[..., Any]] = None,
) -> NvidiaGpuIdentity:
    """Query a physical GPU index and UUID through a mockable command runner."""

    command_runner = runner or subprocess.run
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = command_runner(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise P05RuntimeContractError(f"nvidia-smi GPU identity query failed: {exc}") from exc

    rows: Dict[int, str] = {}
    for raw_line in str(getattr(completed, "stdout", "")).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise P05RuntimeContractError(
                f"unexpected nvidia-smi index/UUID row: {raw_line!r}"
            )
        try:
            index = int(fields[0])
        except ValueError as exc:
            raise P05RuntimeContractError(
                f"invalid physical GPU index from nvidia-smi: {fields[0]!r}"
            ) from exc
        uuid = fields[1]
        if not uuid:
            raise P05RuntimeContractError(
                f"nvidia-smi returned an empty UUID for physical GPU {index}"
            )
        if index in rows:
            raise P05RuntimeContractError(
                f"nvidia-smi returned duplicate physical GPU index {index}"
            )
        rows[index] = uuid

    if physical_index not in rows:
        raise P05RuntimeContractError(
            f"nvidia-smi did not report requested physical GPU index {physical_index}"
        )
    return NvidiaGpuIdentity(
        physical_index=physical_index,
        uuid=rows[physical_index],
    )


def _require_single_process(environment: Mapping[str, str]) -> None:
    for name in _SINGLE_PROCESS_SIZE_VARS:
        value = environment.get(name)
        if value is not None and value != "1":
            raise P05RuntimeContractError(
                f"P05 evidence mode forbids distributed execution: {name}={value!r}"
            )
    for name in _SINGLE_PROCESS_RANK_VARS:
        value = environment.get(name)
        if value is not None and value != "0":
            raise P05RuntimeContractError(
                f"P05 evidence mode requires rank zero only: {name}={value!r}"
            )


def _require_optional_int_one(args_trainer: Any, name: str) -> None:
    if not hasattr(args_trainer, name):
        return
    value = getattr(args_trainer, name)
    if type(value) is not int or value != 1:
        raise P05RuntimeContractError(
            f"trainer.{name} must be the integer 1 in P05 evidence mode"
        )


def prepare_p05_runtime(
    args_trainer: Any,
    *,
    environment: Optional[Mapping[str, str]] = None,
    cuda_is_available: Optional[Callable[[], bool]] = None,
    gpu_query: Optional[Callable[[int], NvidiaGpuIdentity]] = None,
) -> Optional[P05RuntimeContract]:
    """Validate and bind the P05 evidence runtime before side effects occur.

    Legacy configurations return ``None`` unchanged. Only a literal boolean
    ``trainer.p05_evidence_mode=true`` activates this contract.
    """

    if not p05_evidence_mode_enabled(args_trainer):
        return None

    runtime_environment = os.environ if environment is None else environment
    visible_devices = runtime_environment.get("CUDA_VISIBLE_DEVICES")
    if visible_devices not in {"0", "1"}:
        raise P05RuntimeContractError(
            "P05 evidence mode requires CUDA_VISIBLE_DEVICES to be exactly '0' or '1'"
        )
    physical_index = int(visible_devices)
    _require_single_process(runtime_environment)

    availability_check = cuda_is_available or torch.cuda.is_available
    if not bool(availability_check()):
        raise P05RuntimeContractError(
            "P05 evidence mode requires CUDA; CPU fallback is forbidden"
        )

    if getattr(args_trainer, "device", None) != "cuda":
        raise P05RuntimeContractError(
            "trainer.device must be exactly 'cuda' in P05 evidence mode"
        )
    configured_accelerator = getattr(args_trainer, "accelerator", None)
    if configured_accelerator not in {None, "gpu"}:
        raise P05RuntimeContractError(
            "trainer.accelerator must be 'gpu' when explicitly configured; auto is forbidden"
        )
    for cardinality_key in ("devices", "gpus", "num_nodes", "num_processes"):
        _require_optional_int_one(args_trainer, cardinality_key)

    configured_precision = getattr(args_trainer, "precision", None)
    if configured_precision is not None and (
        type(configured_precision) is not int or configured_precision != 32
    ):
        raise P05RuntimeContractError(
            "trainer.precision must be the integer 32 in P05 evidence mode"
        )
    configured_deterministic = getattr(args_trainer, "deterministic", None)
    if configured_deterministic is not None and configured_deterministic is not True:
        raise P05RuntimeContractError(
            "trainer.deterministic must be true in P05 evidence mode"
        )
    configured_strategy = getattr(args_trainer, "strategy", None)
    if configured_strategy not in {None, "auto"}:
        raise P05RuntimeContractError(
            "P05 evidence mode forbids DDP and custom distributed strategies"
        )

    expected_uuid = getattr(args_trainer, "expected_gpu_uuid", None)
    if not isinstance(expected_uuid, str) or not expected_uuid:
        raise P05RuntimeContractError(
            "trainer.expected_gpu_uuid is required in P05 evidence mode"
        )
    identity_query = gpu_query or query_nvidia_smi_gpu
    observed_identity = identity_query(physical_index)
    if observed_identity.physical_index != physical_index:
        raise P05RuntimeContractError(
            "nvidia-smi query returned a different physical GPU index"
        )
    if observed_identity.uuid != expected_uuid:
        raise P05RuntimeContractError(
            "trainer.expected_gpu_uuid does not match the nvidia-smi GPU UUID"
        )

    trainer_kwargs: Dict[str, Any] = {
        "accelerator": "gpu",
        "devices": 1,
        "strategy": "auto",
        "precision": 32,
        "deterministic": True,
    }
    runtime_identity: Dict[str, Any] = {
        "schema_version": 1,
        "paper_id": "P05",
        "evidence_mode": True,
        "cuda_visible_devices": visible_devices,
        "physical_gpu_index": physical_index,
        "gpu_uuid": observed_identity.uuid,
        "expected_gpu_uuid": expected_uuid,
        "identity_source": "nvidia-smi:index,uuid",
        "accelerator": "gpu",
        "devices": 1,
        "gpus": 1,
        "strategy": "auto",
        "precision": 32,
        "deterministic": True,
    }

    for name, value in (
        ("accelerator", "gpu"),
        ("devices", 1),
        ("gpus", 1),
        ("strategy", "auto"),
        ("precision", 32),
        ("deterministic", True),
    ):
        setattr(args_trainer, name, value)
    setattr(args_trainer, "p05_runtime_identity", dict(runtime_identity))

    return P05RuntimeContract(
        trainer_kwargs=trainer_kwargs,
        runtime_identity=runtime_identity,
    )
