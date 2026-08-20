from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from phmfactory import cli
from phmfactory.config import ConfigAnalysis
from phmfactory.pipelines import (
    PipelineMaturityError,
    pipeline_descriptor,
    require_pipeline_access,
)
from phmfactory.runtime import ExecutionStatus


def _analysis(tmp_path: Path, pipeline: str) -> ConfigAnalysis:
    data = {
        "pipeline": pipeline,
        "environment": {"output_dir": str(tmp_path / "runs")},
    }
    path = tmp_path / "maturity.yaml"
    return ConfigAnalysis(
        requested="maturity-test",
        path=path,
        effective_config=data,
        pipeline=pipeline,
        overrides={},
        local_config_path=None,
        source_files=(path,),
        sources={},
    )


def test_supported_pipeline_requires_no_opt_in() -> None:
    descriptor = require_pipeline_access("Pipeline_01_Fault_Diagnosis")
    assert descriptor.maturity == "supported"
    assert descriptor.opt_in_required is False


@pytest.mark.parametrize(
    "pipeline,maturity",
    (
        ("Pipeline_03_Multitask_Pretraining_Finetuning", "experimental"),
        ("Pipeline_04_Unified_Evaluation", "experimental_blocked"),
    ),
)
def test_experimental_pipelines_are_blocked_by_default(
    pipeline: str,
    maturity: str,
) -> None:
    with pytest.raises(PipelineMaturityError, match="--allow-experimental"):
        require_pipeline_access(pipeline)
    descriptor = require_pipeline_access(pipeline, allow_experimental=True)
    assert descriptor.maturity == maturity
    assert descriptor.opt_in_required is True


def test_legacy_alias_resolves_to_same_descriptor() -> None:
    assert pipeline_descriptor("Pipeline_04_unified_metric") == pipeline_descriptor(
        "Pipeline_04_Unified_Evaluation",
        warn=False,
    )


def test_cli_blocks_experimental_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = _analysis(tmp_path, "Pipeline_03_Multitask_Pretraining_Finetuning")
    monkeypatch.setattr(cli, "analyze_config", lambda *args, **kwargs: analysis)
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name: pytest.fail("blocked Pipeline must not be imported"),
    )
    args = argparse.Namespace(
        config="maturity-test",
        config_path=None,
        local_config=None,
        notes="",
        override=None,
        allow_experimental=False,
    )

    with pytest.raises(PipelineMaturityError):
        cli.run(args)

    assert args.execution_envelope.status is ExecutionStatus.FAILED
    assert args.execution_envelope.failure_stage == "maturity"
    assert args.execution_envelope.error_type == "PipelineMaturityError"
    assert args.resolved_config_data == analysis.effective_config
    assert not (tmp_path / "runs").exists()


def test_cli_explicit_opt_in_allows_experimental_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = _analysis(tmp_path, "Pipeline_03_Multitask_Pretraining_Finetuning")
    monkeypatch.setattr(cli, "analyze_config", lambda *args, **kwargs: analysis)
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name: SimpleNamespace(pipeline=lambda args: {"experimental": True}),
    )
    args = argparse.Namespace(
        config="maturity-test",
        config_path=None,
        local_config=None,
        notes="",
        override=None,
        allow_experimental=True,
    )

    assert cli.run(args) == {"experimental": True}
    assert args.pipeline_descriptor.maturity == "experimental"
    assert args.execution_envelope.status is ExecutionStatus.SUCCEEDED
    assert args.resolved_config_data == analysis.effective_config
    assert not (tmp_path / "runs").exists()


def test_non_opt_in_descriptors_remain_discoverable() -> None:
    assert pipeline_descriptor("Pipeline_05_Explainable_Fault_Diagnosis").maturity == (
        "compatibility"
    )
    assert pipeline_descriptor("Pipeline_06_Generative_Modeling").maturity == (
        "experimental_contract"
    )
