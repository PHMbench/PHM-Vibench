"""Mandatory minimal run attestation for public PHMFactory executions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping
from uuid import uuid4

from phmfactory.runtime.execution import ExecutionEnvelope
from phmfactory.runtime.spec import CompiledRunSpec


class AttestationError(RuntimeError):
    """Base error for the mandatory run-attestation contract."""


class AttestationWriteError(AttestationError):
    """Raised when a run manifest cannot be written atomically."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _new_run_id(spec: CompiledRunSpec) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{spec.sha256[:12]}-{uuid4().hex[:8]}"


def _output_root(spec: CompiledRunSpec, *, cwd: Path | None = None) -> Path:
    environment = spec.config.get("environment")
    if not isinstance(environment, Mapping):
        raise AttestationError("compiled config must contain an environment mapping")
    output_dir = environment.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise AttestationError(
            "environment.output_dir is required for mandatory run attestation"
        )
    root = Path(output_dir).expanduser()
    if not root.is_absolute():
        root = (cwd or Path.cwd()) / root
    return root.resolve()


def _code_revision() -> dict[str, str | None]:
    github_sha = os.environ.get("GITHUB_SHA", "").strip()
    if github_sha:
        return {"value": github_sha, "source": "GITHUB_SHA"}
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"value": None, "source": "unavailable"}
    revision = completed.stdout.strip()
    return {"value": revision or None, "source": "git"}


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise AttestationWriteError(f"could not write run manifest {path}: {error}") from error


@dataclass(frozen=True)
class RunAttestation:
    """Location and immutable identity of one invocation-level run manifest."""

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
        """Create a pending manifest before importing or executing the Pipeline."""

        run_id = _new_run_id(spec)
        root = _output_root(spec, cwd=cwd)
        attestation = cls(
            spec=spec,
            pipeline_module=pipeline_module,
            run_id=run_id,
            manifest_path=root / ".phmfactory" / "runs" / run_id / "run_manifest.json",
            created_at=_utc_now(),
        )
        attestation.write(envelope)
        return attestation

    def payload(self, envelope: ExecutionEnvelope) -> dict[str, Any]:
        """Build the single invocation-level evidence document."""

        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "status": envelope.status.value,
            "created_at": self.created_at,
            "run_spec": {
                "sha256": self.spec.sha256,
                "pipeline": self.spec.pipeline,
                "pipeline_module": self.pipeline_module,
                "requested_config": self.spec.requested_config,
                "resolved_config_path": self.spec.resolved_config_path,
                "overrides": self.spec.overrides,
            },
            "execution": envelope.as_dict(),
            "code": {"revision": _code_revision()},
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "failure": (
                {
                    "stage": envelope.failure_stage,
                    "type": envelope.error_type,
                    "message": envelope.error_message,
                }
                if envelope.status.value == "failed"
                else None
            ),
            "artifacts": [],
            "evidence": {
                "data": {},
                "protocol": {},
                "seed": {},
                "environment": {},
            },
        }

    def write(self, envelope: ExecutionEnvelope) -> None:
        """Atomically replace the manifest with the current envelope state."""

        _atomic_json_write(self.manifest_path, self.payload(envelope))
