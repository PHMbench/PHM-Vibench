"""Mandatory minimal run attestation for public PHMFactory executions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Mapping
from uuid import uuid4

from phmfactory.runtime.execution import ExecutionEnvelope
from phmfactory.runtime.spec import CompiledRunSpec


SHA256 = re.compile(r"[0-9a-f]{64}")


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


def _json_copy(value: Any, *, label: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise AttestationError(f"{label} must be JSON serializable: {error}") from error


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


@dataclass
class RunAttestation:
    """Identity and accumulated evidence for one public invocation."""

    spec: CompiledRunSpec
    pipeline_module: str
    run_id: str
    manifest_path: Path
    created_at: str
    artifacts: list[dict[str, Any]] = field(default_factory=list, repr=False)
    evidence: dict[str, Any] = field(
        default_factory=lambda: {
            "data": {},
            "protocol": {},
            "seed": {},
            "environment": {},
        },
        repr=False,
    )

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

    def register_artifact(
        self,
        *,
        role: str,
        path: str | Path,
        sha256: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register one existing artifact without silently replacing conflicts."""

        normalized_role = str(role).strip()
        if not normalized_role:
            raise AttestationError("artifact role must be non-empty")
        artifact_path = Path(path).expanduser().resolve()
        if not artifact_path.is_file():
            raise AttestationError(f"artifact does not exist or is not a file: {artifact_path}")
        digest = str(sha256 or "").strip().lower() or None
        if digest is not None and SHA256.fullmatch(digest) is None:
            raise AttestationError(f"artifact sha256 is not 64 lowercase hex: {digest!r}")
        record = {
            "role": normalized_role,
            "path": str(artifact_path),
            "sha256": digest,
            "metadata": _json_copy(dict(metadata or {}), label="artifact metadata"),
        }
        for existing in self.artifacts:
            if existing["role"] == normalized_role and existing["path"] == str(artifact_path):
                if existing != record:
                    raise AttestationError(
                        "conflicting artifact registration for "
                        f"role={normalized_role!r}, path={artifact_path}"
                    )
                return deepcopy(existing)
        self.artifacts.append(record)
        return deepcopy(record)

    def set_evidence(self, section: str, value: Any) -> None:
        """Set one evidence section exactly once unless the value is identical."""

        key = str(section).strip()
        if not key:
            raise AttestationError("evidence section must be non-empty")
        normalized = _json_copy(value, label=f"evidence section {key!r}")
        existing = self.evidence.get(key)
        if existing not in (None, {}, []) and existing != normalized:
            raise AttestationError(f"conflicting evidence section: {key!r}")
        self.evidence[key] = normalized

    def append_evidence(self, section: str, value: Any) -> None:
        """Append one JSON evidence record to an ordered section."""

        key = str(section).strip()
        if not key:
            raise AttestationError("evidence section must be non-empty")
        normalized = _json_copy(value, label=f"evidence section {key!r}")
        current = self.evidence.setdefault(key, [])
        if not isinstance(current, list):
            raise AttestationError(f"evidence section {key!r} is not appendable")
        current.append(normalized)

    def payload(self, envelope: ExecutionEnvelope) -> dict[str, Any]:
        """Build the single invocation-level evidence document."""

        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "status": envelope.status.value,
            "created_at": self.created_at,
            "run_spec": {
                "sha256": self.spec.sha256,
                "effective_config_sha256": self.spec.effective_config_sha256,
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
            "artifacts": deepcopy(self.artifacts),
            "evidence": deepcopy(self.evidence),
        }

    def write(self, envelope: ExecutionEnvelope) -> None:
        """Atomically replace the manifest with the current envelope state."""

        _atomic_json_write(self.manifest_path, self.payload(envelope))
