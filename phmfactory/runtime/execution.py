"""Fail-closed execution boundary for public PHMFactory entrypoints."""

from __future__ import annotations

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


@dataclass
class ExecutionEnvelope:
    """Record and enforce the lifecycle of one compiled Pipeline invocation."""

    spec: CompiledRunSpec
    pipeline_module: str
    schema_version: int = 1
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: str | None = None
    finished_at: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    def _fail(self, error: BaseException) -> None:
        self.status = ExecutionStatus.FAILED
        self.finished_at = _utc_now()
        self.error_type = type(error).__name__
        self.error_message = str(error)

    def execute(self, module: ModuleType | Any, args: Any) -> Any:
        """Execute exactly once and reject missing or ambiguous success results."""

        if self.status is not ExecutionStatus.PENDING:
            raise PipelineContractError(
                f"execution envelope cannot run from status {self.status.value!r}"
            )

        entrypoint: Callable[[Any], Any] | None = getattr(module, "pipeline", None)
        if not callable(entrypoint):
            error = PipelineContractError(
                f"Pipeline module {self.pipeline_module!r} has no callable pipeline(args)"
            )
            self._fail(error)
            raise error

        self.status = ExecutionStatus.RUNNING
        self.started_at = _utc_now()
        try:
            result = entrypoint(args)
            if result is None:
                raise PipelineContractError(
                    f"Pipeline {self.spec.pipeline!r} returned None; "
                    "successful Pipelines must return an explicit result"
                )
        except BaseException as error:
            self._fail(error)
            raise

        self.status = ExecutionStatus.SUCCEEDED
        self.finished_at = _utc_now()
        return result

    def as_dict(self) -> dict[str, Any]:
        """Return the minimal state needed by the later run-attestation writer."""

        return {
            "schema_version": self.schema_version,
            "run_spec_sha256": self.spec.sha256,
            "pipeline": self.spec.pipeline,
            "pipeline_module": self.pipeline_module,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }
