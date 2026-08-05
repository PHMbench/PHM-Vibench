from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from phmfactory import cli
from phmfactory.config import ConfigAnalysis, ResolvedConfig, semantic_config_sha256
from phmfactory.runtime import (
    AttestationError,
    AttestationWriteError,
    CompiledRunSpec,
    ExecutionEnvelope,
    ExecutionStatus,
    RunAttestation,
)
from phmfactory.runtime import attestation as attestation_module


def _resolved(tmp_path: Path) -> ResolvedConfig:
    return ResolvedConfig(
        requested="smoke",
        path=tmp_path / "smoke.yaml",
        data={
            "pipeline": "Pipeline_01_Fault_Diagnosis",
            "environment": {"output_dir": str(tmp_path / "outputs")},
        },
        pipeline="Pipeline_01_Fault_Diagnosis",
        overrides={},
    )


def _analysis(tmp_path: Path) -> ConfigAnalysis:
    resolved = _resolved(tmp_path)
    return ConfigAnalysis(
        requested=resolved.requested,
        path=resolved.path,
        effective_config=resolved.data,
        pipeline=resolved.pipeline,
        overrides=resolved.overrides,
        local_config_path=None,
        source_files=(resolved.path,),
        sources={},
        diagnostics=(),
        effective_config_sha256=semantic_config_sha256(resolved.data),
    )


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_prepare_writes_pending_manifest_atomically(tmp_path: Path) -> None:
    spec = CompiledRunSpec.compile(_resolved(tmp_path))
    envelope = ExecutionEnvelope(spec=spec, pipeline_module="src.pipeline")
    attestation = RunAttestation.prepare(spec, "src.pipeline", envelope)

    payload = _load(attestation.manifest_path)
    assert payload["status"] == "pending"
    assert payload["run_spec"]["sha256"] == spec.sha256
    assert (
        payload["run_spec"]["effective_config_sha256"]
        == spec.effective_config_sha256
    )
    assert payload["failure"] is None
    assert not list(attestation.manifest_path.parent.glob("*.tmp"))


def test_success_replaces_pending_manifest(tmp_path: Path) -> None:
    spec = CompiledRunSpec.compile(_resolved(tmp_path))
    envelope = ExecutionEnvelope(spec=spec, pipeline_module="src.pipeline")
    attestation = RunAttestation.prepare(spec, "src.pipeline", envelope)
    envelope.execute(SimpleNamespace(pipeline=lambda args: {"metric": 1.0}), object())
    attestation.write(envelope)

    payload = _load(attestation.manifest_path)
    assert payload["status"] == "succeeded"
    assert payload["execution"]["status"] == "succeeded"
    assert payload["execution"]["finished_at"] is not None


def test_failure_manifest_preserves_stage_and_error(tmp_path: Path) -> None:
    spec = CompiledRunSpec.compile(_resolved(tmp_path))
    envelope = ExecutionEnvelope(spec=spec, pipeline_module="src.pipeline")
    attestation = RunAttestation.prepare(spec, "src.pipeline", envelope)

    def fail(args):
        raise RuntimeError("training failed")

    with pytest.raises(RuntimeError, match="training failed"):
        envelope.execute(SimpleNamespace(pipeline=fail), object())
    attestation.write(envelope)

    payload = _load(attestation.manifest_path)
    assert payload["status"] == "failed"
    assert payload["failure"] == {
        "stage": "pipeline",
        "type": "RuntimeError",
        "message": "training failed",
    }


def test_missing_output_dir_blocks_before_execution(tmp_path: Path) -> None:
    resolved = _resolved(tmp_path)
    resolved.data["environment"] = {}
    spec = CompiledRunSpec.compile(resolved)
    envelope = ExecutionEnvelope(spec=spec, pipeline_module="src.pipeline")

    with pytest.raises(AttestationError, match="environment.output_dir"):
        RunAttestation.prepare(spec, "src.pipeline", envelope)


def test_failed_atomic_replace_leaves_previous_manifest_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = CompiledRunSpec.compile(_resolved(tmp_path))
    envelope = ExecutionEnvelope(spec=spec, pipeline_module="src.pipeline")
    attestation = RunAttestation.prepare(spec, "src.pipeline", envelope)
    before = _load(attestation.manifest_path)
    envelope.execute(SimpleNamespace(pipeline=lambda args: True), object())

    monkeypatch.setattr(
        attestation_module.os,
        "replace",
        lambda *args: (_ for _ in ()).throw(OSError("replace denied")),
    )
    with pytest.raises(AttestationWriteError, match="replace denied"):
        attestation.write(envelope)

    assert _load(attestation.manifest_path) == before


def test_cli_writes_success_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    analysis = _analysis(tmp_path)
    monkeypatch.setattr(cli, "analyze_config", lambda *args, **kwargs: analysis)
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name: SimpleNamespace(pipeline=lambda args: ["ok"]),
    )
    args = argparse.Namespace(
        config="smoke",
        config_path=None,
        local_config=None,
        notes="",
        override=None,
    )

    assert cli.run(args) == ["ok"]
    payload = _load(Path(args.run_manifest_path))
    assert payload["run_id"] == args.run_id
    assert payload["status"] == "succeeded"
    assert (
        payload["run_spec"]["effective_config_sha256"]
        == analysis.effective_config_sha256
    )
    assert args.execution_envelope.status is ExecutionStatus.SUCCEEDED


def test_cli_writes_failed_manifest_and_reraises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = _analysis(tmp_path)
    monkeypatch.setattr(cli, "analyze_config", lambda *args, **kwargs: analysis)

    def fail(args):
        raise ValueError("bad pipeline")

    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name: SimpleNamespace(pipeline=fail),
    )
    args = argparse.Namespace(
        config="smoke",
        config_path=None,
        local_config=None,
        notes="",
        override=None,
    )

    with pytest.raises(ValueError, match="bad pipeline"):
        cli.run(args)

    payload = _load(Path(args.run_manifest_path))
    assert payload["status"] == "failed"
    assert payload["failure"]["stage"] == "pipeline"
    assert payload["failure"]["type"] == "ValueError"
