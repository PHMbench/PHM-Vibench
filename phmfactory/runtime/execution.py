"""Fail-closed execution boundary for public PHMFactory entrypoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from types import ModuleType
from typing import Any, Callable

from phmfactory.runtime.spec import CompiledRunSpec


class ExecutionStatus(str, Enum):
    """Finite states for one public Pipeline invocation."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class PipelineContractError(RuntimeError):
    """Raised when a Pipeline module violates the public execution contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_success_mapping(result: Any, *, pipeline: str) -> Mapping[str, Any]:
    """Return one non-empty success mapping or reject an ambiguous result."""

    if not isinstance(result, Mapping):
        raise PipelineContractError(
            f"Pipeline {pipeline!r} returned {type(result).__name__}; "
            "successful Pipelines must return a non-empty result mapping"
        )
    if not result:
        raise PipelineContractError(
            f"Pipeline {pipeline!r} returned an empty result mapping"
        )

    status = result.get("status")
    if status is not None and status != "succeeded":
        raise PipelineContractError(
            f"Pipeline {pipeline!r} returned status={status!r}; "
            "failures must raise their original exception"
        )
    if "error" in result and status != "succeeded":
        raise PipelineContractError(
            f"Pipeline {pipeline!r} returned an error mapping; "
            "failures must raise their original exception"
        )
    return result


@dataclass
class ExecutionEnvelope:
    """Enforce one Pipeline lifecycle while retaining the original failure."""

    spec: CompiledRunSpec
    pipeline_module: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: str | None = None
    finished_at: str | None = None
    failure_stage: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    def record_failure(self, error: BaseException, *, stage: str) -> None:
        """Record one terminal failure while retaining the original exception."""

        if self.status is ExecutionStatus.FAILED:
            return
        self.status = ExecutionStatus.FAILED
        self.finished_at = _utc_now()
        self.failure_stage = stage
        self.error_type = type(error).__name__
        self.error_message = str(error)

    def execute(self, module: ModuleType | Any, args: Any) -> Mapping[str, Any]:
        """Execute exactly once and require one structured success result."""

        if self.status is not ExecutionStatus.PENDING:
            raise PipelineContractError(
                f"execution envelope cannot run from status {self.status.value!r}"
            )

        entrypoint: Callable[[Any], Any] | None = getattr(module, "pipeline", None)
        if not callable(entrypoint):
            error = PipelineContractError(
                f"Pipeline module {self.pipeline_module!r} has no callable pipeline(args)"
            )
            self.record_failure(error, stage="contract")
            raise error

        self.status = ExecutionStatus.RUNNING
        self.started_at = _utc_now()
        try:
            result = _require_success_mapping(
                entrypoint(args),
                pipeline=self.spec.pipeline,
            )
        except BaseException as error:
            self.record_failure(error, stage="pipeline")
            raise

        self.status = ExecutionStatus.SUCCEEDED
        self.finished_at = _utc_now()
        return result
