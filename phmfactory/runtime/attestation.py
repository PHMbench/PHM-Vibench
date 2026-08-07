"""Minimal run-manifest writer for public PHMFactory executions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from phmfactory.runtime.execution import ExecutionEnvelope
from phmfactory.runtime.spec import CompiledRunSpec


class AttestationError(RuntimeError):
    """Base error for the compatibility run-manifest writer."""


class AttestationWriteError(AttestationError):
    """Raised when the run manifest cannot be written."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{uuid4().hex[:8]}"


def _output_root(spec: CompiledRunSpec, *, cwd: Path | None = None) -> Path:
    environment = spec.config.get("environment")
    if not isinstance(environment, Mapping):
        raise AttestationError("compiled config must contain an environment mapping")
    output_dir = environment.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise AttestationError("environment.output_dir is required for the run manifest")
    root = Path(output_dir).expanduser()
    if not root.is_absolute():
        root = (cwd or Path.cwd()) / root
    return root.resolve()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as error:
        raise AttestationWriteError(f"could not write run manifest {path}: {error}") from error


@dataclass
class RunAttestation:
    """Compatibility name for the minimal invocation-level run manifest."""

    spec: CompiledRunSpec
    pipeline_module: str
    run_id: str
    manifest_path: Path
    created_at: str

    @classmethod
    def prepare(
        cls,
        spec: CompiledRunSpec,
        pipeline_module: str,
        envelope: ExecutionEnvelope,
        *,
        cwd: Path | None = None,
    ) -> "RunAttestation":
        """Create the pending run manifest before Pipeline execution."""

        run_id = _new_run_id()
        root = _output_root(spec, cwd=cwd)
        manifest = cls(
            spec=spec,
            pipeline_module=pipeline_module,
            run_id=run_id,
            manifest_path=root / ".phmfactory" / "runs" / run_id / "run_manifest.json",
            created_at=_utc_now(),
        )
        manifest.write(envelope)
        return manifest

    def payload(self, envelope: ExecutionEnvelope) -> dict[str, Any]:
        """Build the minimal user-facing run record."""

        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "status": envelope.status.value,
            "created_at": self.created_at,
            "run_spec": {
                "pipeline": self.spec.pipeline,
                "pipeline_module": self.pipeline_module,
                "requested_config": self.spec.requested_config,
                "resolved_config_path": self.spec.resolved_config_path,
                "overrides": self.spec.overrides,
            },
            "execution": envelope.as_dict(),
            "failure": (
                {
                    "stage": envelope.failure_stage,
                    "type": envelope.error_type,
                    "message": envelope.error_message,
                }
                if envelope.status.value == "failed"
                else None
            ),
        }

    def write(self, envelope: ExecutionEnvelope) -> None:
        """Write the current run state."""

        _write_json(self.manifest_path, self.payload(envelope))
