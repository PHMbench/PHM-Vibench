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


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        config="smoke",
        config_path=None,
        local_config=None,
        notes="",
        override=None,
        allow_experimental=False,
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


def test_missing_output_dir_is_an_attestation_error(tmp_path: Path) -> None:
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


def test_cli_writes_success_manifest_when_diagnostic_writer_is_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = _analysis(tmp_path)
    monkeypatch.setattr(cli, "analyze_config", lambda *args, **kwargs: analysis)
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name: SimpleNamespace(pipeline=lambda args: ["ok"]),
    )
    args = _args()

    assert cli.run(args) == ["ok"]
    payload = _load(Path(args.run_manifest_path))
    assert payload["run_id"] == args.run_id
    assert payload["status"] == "succeeded"
    assert (
        payload["run_spec"]["effective_config_sha256"]
        == analysis.effective_config_sha256
    )
    assert args.execution_envelope.status is ExecutionStatus.SUCCEEDED


def test_cli_writes_failed_manifest_and_reraises_original_pipeline_error(
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
    args = _args()

    with pytest.raises(ValueError, match="bad pipeline"):
        cli.run(args)

    payload = _load(Path(args.run_manifest_path))
    assert payload["status"] == "failed"
    assert payload["failure"]["stage"] == "pipeline"
    assert payload["failure"]["type"] == "ValueError"


def test_cli_continues_when_pending_manifest_cannot_be_prepared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    analysis = _analysis(tmp_path)
    monkeypatch.setattr(cli, "analyze_config", lambda *args, **kwargs: analysis)
    monkeypatch.setattr(
        cli.RunAttestation,
        "prepare",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AttestationWriteError("write denied")
        ),
    )
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name: SimpleNamespace(pipeline=lambda args: ["scientific-result"]),
    )
    args = _args()

    assert cli.run(args) == ["scientific-result"]
    captured = capsys.readouterr()
    assert "pending run manifest could not be recorded" in captured.err
    assert "run_manifest=unavailable" in captured.out
    assert "完成所有实验" in captured.out
    assert args.run_manifest_path is None
    assert args.execution_envelope.status is ExecutionStatus.SUCCEEDED


def test_cli_success_survives_terminal_manifest_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    analysis = _analysis(tmp_path)
    monkeypatch.setattr(cli, "analyze_config", lambda *args, **kwargs: analysis)

    class BrokenAttestation:
        run_id = "diagnostic-only"
        manifest_path = tmp_path / "run_manifest.json"

        @staticmethod
        def write(envelope):
            del envelope
            raise AttestationWriteError("final replace denied")

    monkeypatch.setattr(
        cli.RunAttestation,
        "prepare",
        lambda *args, **kwargs: BrokenAttestation(),
    )
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name: SimpleNamespace(pipeline=lambda args: ["scientific-result"]),
    )
    args = _args()

    assert cli.run(args) == ["scientific-result"]
    captured = capsys.readouterr()
    assert "terminal run manifest could not be recorded" in captured.err
    assert "run_manifest=unavailable" in captured.out
    assert "完成所有实验" in captured.out
    assert args.execution_envelope.status is ExecutionStatus.SUCCEEDED


def test_manifest_write_failure_never_replaces_original_pipeline_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    analysis = _analysis(tmp_path)
    monkeypatch.setattr(cli, "analyze_config", lambda *args, **kwargs: analysis)

    class BrokenAttestation:
        run_id = "diagnostic-only"
        manifest_path = tmp_path / "run_manifest.json"

        @staticmethod
        def write(envelope):
            del envelope
            raise AttestationWriteError("failed replace denied")

    def fail(args):
        raise ValueError("training failed first")

    monkeypatch.setattr(
        cli.RunAttestation,
        "prepare",
        lambda *args, **kwargs: BrokenAttestation(),
    )
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name: SimpleNamespace(pipeline=fail),
    )
    args = _args()

    with pytest.raises(ValueError, match="training failed first"):
        cli.run(args)
    assert "failed run manifest could not be recorded" in capsys.readouterr().err
    assert args.execution_envelope.status is ExecutionStatus.FAILED
