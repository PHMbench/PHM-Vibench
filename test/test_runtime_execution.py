from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from phmfactory import cli
from phmfactory.config import ConfigAnalysis, ResolvedConfig
from phmfactory.runtime import (
    CompiledRunSpec,
    ExecutionEnvelope,
    ExecutionStatus,
    PipelineContractError,
)


def _spec() -> CompiledRunSpec:
    return CompiledRunSpec.compile(
        ResolvedConfig(
            requested="smoke",
            path=Path("/tmp/smoke.yaml"),
            data={"pipeline": "Pipeline_01_Fault_Diagnosis"},
            pipeline="Pipeline_01_Fault_Diagnosis",
            overrides={},
        )
    )


def test_envelope_records_structured_success() -> None:
    envelope = ExecutionEnvelope(
        spec=_spec(),
        pipeline_module="src.Pipeline_01_Fault_Diagnosis",
    )
    expected = {"status": "succeeded", "result_dir": "/tmp/run"}
    result = envelope.execute(
        SimpleNamespace(pipeline=lambda args: expected),
        object(),
    )

    assert result == expected
    assert envelope.status is ExecutionStatus.SUCCEEDED
    assert envelope.started_at is not None
    assert envelope.finished_at is not None
    assert envelope.error_type is None


@pytest.mark.parametrize(
    "result",
    (None, True, False, "success", [], (), {}),
)
def test_envelope_rejects_ambiguous_success_values(result) -> None:
    envelope = ExecutionEnvelope(spec=_spec(), pipeline_module="src.invalid")

    with pytest.raises(PipelineContractError, match="must return|empty result"):
        envelope.execute(
            SimpleNamespace(pipeline=lambda args: result),
            object(),
        )

    assert envelope.status is ExecutionStatus.FAILED
    assert envelope.failure_stage == "pipeline"
    assert envelope.error_type == "PipelineContractError"
    assert envelope.finished_at is not None


@pytest.mark.parametrize(
    "result",
    (
        {"status": "failed", "error": "training failed"},
        {"error": "training failed"},
    ),
)
def test_envelope_rejects_returned_failure_mappings(result) -> None:
    envelope = ExecutionEnvelope(spec=_spec(), pipeline_module="src.invalid")

    with pytest.raises(PipelineContractError, match="failures must raise"):
        envelope.execute(
            SimpleNamespace(pipeline=lambda args: result),
            object(),
        )

    assert envelope.status is ExecutionStatus.FAILED
    assert envelope.failure_stage == "pipeline"


def test_envelope_rejects_missing_entrypoint() -> None:
    envelope = ExecutionEnvelope(spec=_spec(), pipeline_module="src.invalid")

    with pytest.raises(PipelineContractError, match="no callable pipeline"):
        envelope.execute(SimpleNamespace(), object())

    assert envelope.status is ExecutionStatus.FAILED
    assert envelope.failure_stage == "contract"


def test_envelope_records_and_reraises_pipeline_error() -> None:
    envelope = ExecutionEnvelope(spec=_spec(), pipeline_module="src.invalid")

    def fail(args):
        raise ValueError("invalid training state")

    with pytest.raises(ValueError, match="invalid training state"):
        envelope.execute(SimpleNamespace(pipeline=fail), object())

    assert envelope.status is ExecutionStatus.FAILED
    assert envelope.failure_stage == "pipeline"
    assert envelope.error_type == "ValueError"
    assert envelope.error_message == "invalid training state"


def test_envelope_cannot_execute_twice() -> None:
    envelope = ExecutionEnvelope(spec=_spec(), pipeline_module="src.valid")
    envelope.execute(
        SimpleNamespace(
            pipeline=lambda args: {"status": "succeeded", "result_dir": "/tmp/run"}
        ),
        object(),
    )

    with pytest.raises(PipelineContractError, match="cannot run from status"):
        envelope.execute(
            SimpleNamespace(
                pipeline=lambda args: {
                    "status": "succeeded",
                    "result_dir": "/tmp/run",
                }
            ),
            object(),
        )


def test_cli_does_not_print_completion_when_pipeline_result_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = {
        "pipeline": "Pipeline_01_Fault_Diagnosis",
        "environment": {"output_dir": str(tmp_path / "outputs")},
    }
    path = tmp_path / "broken.yaml"
    analysis = ConfigAnalysis(
        requested="broken",
        path=path,
        effective_config=data,
        pipeline="Pipeline_01_Fault_Diagnosis",
        overrides={},
        local_config_path=None,
        source_files=(path,),
        sources={},
        diagnostics=(),
        effective_config_sha256="internal-only",
    )
    monkeypatch.setattr(cli, "analyze_config", lambda *args, **kwargs: analysis)
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name: SimpleNamespace(pipeline=lambda args: "success"),
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

    output = capsys.readouterr().out
    assert "run=completed" not in output
    assert "完成所有实验" not in output
    assert args.execution_envelope.status is ExecutionStatus.FAILED
    assert not hasattr(args, "run_manifest_path")
    assert not (tmp_path / "outputs").exists()
