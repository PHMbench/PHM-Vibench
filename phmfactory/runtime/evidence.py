"""Pipeline-result adapters for the single invocation-level run attestation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from phmfactory.runtime.attestation import AttestationError, RunAttestation
from phmfactory.runtime.spec import CompiledRunSpec


PIPELINE_06 = "Pipeline_06_Generative_Modeling"


def _path_from_spec(value: str, *, cwd: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (cwd or Path.cwd()) / path
    return path.resolve()


def _generative_stage_ledger(spec: CompiledRunSpec) -> Path:
    task = spec.config.get("task")
    task = task if isinstance(task, Mapping) else {}
    generative = task.get("generative")
    generative = generative if isinstance(generative, Mapping) else {}
    configured = generative.get("stage_ledger_path")
    if isinstance(configured, str) and configured.strip():
        return _path_from_spec(configured)

    environment = spec.config.get("environment")
    environment = environment if isinstance(environment, Mapping) else {}
    output_dir = environment.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise AttestationError(
            "Pipeline 06 evidence requires environment.output_dir or "
            "task.generative.stage_ledger_path"
        )
    return _path_from_spec(str(Path(output_dir) / "stage_ledger.json"))


def _register_path_records(
    attestation: RunAttestation,
    *,
    stage: str,
    result: Mapping[str, Any],
) -> None:
    for name, value in result.items():
        if not isinstance(value, Mapping):
            continue
        path = value.get("path")
        if not isinstance(path, str) or not path.strip():
            continue
        candidate = _path_from_spec(path)
        if not candidate.is_file():
            continue
        sha256 = value.get("sha256")
        attestation.register_artifact(
            role=f"generative_{stage}_{name}",
            path=candidate,
            sha256=str(sha256) if sha256 else None,
            metadata={"stage": stage, "source": "pipeline_result"},
        )


def _register_pipeline06(
    attestation: RunAttestation,
    spec: CompiledRunSpec,
    result: Any,
) -> None:
    if not isinstance(result, list) or not result:
        raise AttestationError("Pipeline 06 must return a non-empty stage result list")

    for index, item in enumerate(result):
        if not isinstance(item, Mapping):
            raise AttestationError(
                f"Pipeline 06 result[{index}] must be a mapping, got {type(item).__name__}"
            )
        stage = str(item.get("stage") or "").strip()
        if stage not in {"train", "sample", "eval"}:
            raise AttestationError(
                f"Pipeline 06 result[{index}] has unsupported stage {stage!r}"
            )
        attestation.append_evidence(
            "generative_stages",
            {"iteration": index, **dict(item)},
        )
        _register_path_records(attestation, stage=stage, result=item)

    ledger = _generative_stage_ledger(spec)
    if not ledger.is_file():
        raise AttestationError(f"Pipeline 06 stage ledger is missing: {ledger}")
    attestation.register_artifact(
        role="generative_stage_ledger",
        path=ledger,
        metadata={"pipeline": PIPELINE_06},
    )


def register_pipeline_result_evidence(
    attestation: RunAttestation,
    spec: CompiledRunSpec,
    result: Any,
) -> None:
    """Attach Pipeline-specific evidence without creating another run identity."""

    if spec.pipeline == PIPELINE_06:
        _register_pipeline06(attestation, spec, result)
