from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from phmfactory import cli
from phmfactory.config import ConfigAnalysis
from phmfactory.runtime import (
    ExecutionEnvelope,
    ExecutionStatus,
    PipelineContractError,
)


PIPELINE = "Pipeline_01_Fault_Diagnosis"


def _envelope(module: str = "src.Pipeline_01_Fault_Diagnosis") -> ExecutionEnvelope:
    return ExecutionEnvelope(pipeline=PIPELINE, pipeline_module=module)


def test_envelope_records_success() -> None:
    envelope = _envelope()
    result = envelope.execute(SimpleNamespace(pipeline=lambda args: ["ok"]), object())

    assert result == ["ok"]
    assert envelope.status is ExecutionStatus.SUCCEEDED
    assert envelope.started_at is not None
    assert envelope.finished_at is not None
    assert envelope.error_type is None


def test_envelope_rejects_none_as_ambiguous_success() -> None:
    envelope = _envelope("src.invalid")

    with pytest.raises(PipelineContractError, match="returned None"):
        envelope.execute(SimpleNamespace(pipeline=lambda args: None), object())

    assert envelope.status is ExecutionStatus.FAILED
    assert envelope.failure_stage == "pipeline"
    assert envelope.error_type == "PipelineContractError"
    assert envelope.finished_at is not None


def test_envelope_rejects_missing_entrypoint() -> None:
    envelope = _envelope("src.invalid")

    with pytest.raises(PipelineContractError, match="no callable pipeline"):
        envelope.execute(SimpleNamespace(), object())

    assert envelope.status is ExecutionStatus.FAILED
    assert envelope.failure_stage == "contract"


def test_envelope_records_and_reraises_pipeline_error() -> None:
    envelope = _envelope("src.invalid")

    def fail(args):
        raise ValueError("invalid training state")

    with pytest.raises(ValueError, match="invalid training state"):
        envelope.execute(SimpleNamespace(pipeline=fail), object())

    assert envelope.status is ExecutionStatus.FAILED
    assert envelope.failure_stage == "pipeline"
    assert envelope.error_type == "ValueError"
    assert envelope.error_message == "invalid training state"


def test_envelope_cannot_execute_twice() -> None:
    envelope = _envelope("src.valid")
    envelope.execute(SimpleNamespace(pipeline=lambda args: True), object())

    with pytest.raises(PipelineContractError, match="cannot run from status"):
        envelope.execute(SimpleNamespace(pipeline=lambda args: True), object())


def test_cli_does_not_print_completion_when_pipeline_returns_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = {
        "pipeline": PIPELINE,
        "environment": {"output_dir": str(tmp_path / "outputs")},
    }
    path = tmp_path / "broken.yaml"
    analysis = ConfigAnalysis(
        requested="broken",
        path=path,
        effective_config=data,
        pipeline=PIPELINE,
        overrides={},
        local_config_path=None,
        source_files=(path,),
        sources={},
    )
    monkeypatch.setattr(cli, "analyze_config", lambda *args, **kwargs: analysis)
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name: SimpleNamespace(pipeline=lambda args: None),
    )

    args = argparse.Namespace(
        config="broken",
        config_path=None,
        local_config=None,
        notes="",
        override=None,
    )
    with pytest.raises(PipelineContractError):
        cli.run(args)

    assert "完成所有实验" not in capsys.readouterr().out
    assert args.execution_envelope.status is ExecutionStatus.FAILED
    assert args.resolved_config_data == data
    assert not (tmp_path / "outputs").exists()
