from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from phmfactory.config import ResolvedConfig
from phmfactory.runtime import CompiledRunSpec
import src.Pipeline_02_Pretraining_Few_Shot as pipeline02


def _compiled(tmp_path: Path, *, stages=None) -> CompiledRunSpec:
    data = {
        "pipeline": "Pipeline_02_Pretraining_Few_Shot",
        "environment": {"output_dir": str(tmp_path), "iterations": 1, "seed": 1},
        "data": {"data_dir": str(tmp_path), "metadata_file": "metadata.csv"},
        "model": {"name": "model", "type": "test"},
        "task": {"name": "classification", "type": "pretrain"},
        "trainer": {"device": "cpu", "gpus": 1},
    }
    if stages is not None:
        data["stages"] = stages
    return CompiledRunSpec.compile(
        ResolvedConfig(
            requested="pipeline02",
            path=tmp_path / "pipeline02.yaml",
            data=data,
            pipeline="Pipeline_02_Pretraining_Few_Shot",
            overrides={"trainer": {"num_epochs": 1}},
        )
    )


def _args(tmp_path: Path, *, stages=None, **values) -> Namespace:
    payload = {
        "compiled_run_spec": _compiled(tmp_path, stages=stages),
        "config_path": str(tmp_path / "pipeline02.yaml"),
        "override": ["trainer.num_epochs=1"],
        "fs_config_path": None,
    }
    payload.update(values)
    return Namespace(**payload)


def test_single_stage_reuses_shared_classification_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        pipeline02,
        "run_classification_pipeline",
        lambda args: calls.append(args) or [{"loss": 1.0}],
    )
    monkeypatch.setattr(
        pipeline02,
        "TwoStageOrchestrator",
        lambda *args, **kwargs: pytest.fail("orchestrator must not be constructed"),
    )
    args = _args(tmp_path)

    assert pipeline02.pipeline(args) == [{"loss": 1.0}]
    assert calls == [args]


def test_compiled_stages_select_one_unified_orchestrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class Orchestrator:
        def __init__(self, config, cli_overrides):
            observed["config"] = config
            observed["overrides"] = cli_overrides

        def run_complete(self):
            return {"stages": 2}

    monkeypatch.setattr(pipeline02, "TwoStageOrchestrator", Orchestrator)
    monkeypatch.setattr(
        pipeline02,
        "run_classification_pipeline",
        lambda args: pytest.fail("single-stage runtime must not run"),
    )
    args = _args(
        tmp_path,
        stages=[{"name": "pretrain", "overrides": {}}, {"name": "finetune", "overrides": {}}],
    )

    assert pipeline02.pipeline(args) == {"stages": 2}
    assert observed["overrides"] == []
    assert observed["config"]["stages"][0]["name"] == "pretrain"


def test_orchestrator_error_is_not_converted_to_single_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Orchestrator:
        def __init__(self, config, cli_overrides):
            pass

        def run_complete(self):
            raise RuntimeError("stage failed")

    monkeypatch.setattr(pipeline02, "TwoStageOrchestrator", Orchestrator)
    monkeypatch.setattr(
        pipeline02,
        "run_classification_pipeline",
        lambda args: pytest.fail("fallback is forbidden"),
    )

    with pytest.raises(RuntimeError, match="stage failed"):
        pipeline02.pipeline(_args(tmp_path, stages=[{"overrides": {}}]))


def test_legacy_dual_yaml_requires_explicit_mode(tmp_path: Path) -> None:
    args = _args(tmp_path, fs_config_path="fewshot.yaml")
    with pytest.raises(ValueError, match="legacy_dual_yaml"):
        pipeline02.pipeline(args)


def test_explicit_legacy_dual_yaml_uses_only_adapter_and_orchestrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unified = SimpleNamespace(stages=[object(), object()])
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        pipeline02,
        "adapt_p02",
        lambda pretrain, fewshot, local: observed.update(
            pretrain=pretrain, fewshot=fewshot, local=local
        )
        or unified,
    )

    class Orchestrator:
        def __init__(self, config):
            assert config is unified

        def run_complete(self):
            return {"legacy": True}

    monkeypatch.setattr(pipeline02, "TwoStageOrchestrator", Orchestrator)
    args = _args(
        tmp_path,
        fs_config_path="fewshot.yaml",
        pipeline_mode="legacy_dual_yaml",
        local_config=None,
        override=None,
    )

    assert pipeline02.pipeline(args) == {"legacy": True}
    assert observed == {
        "pretrain": str(tmp_path / "pipeline02.yaml"),
        "fewshot": "fewshot.yaml",
        "local": None,
    }


def test_pipeline02_no_longer_exposes_manual_duplicate_stage_runners() -> None:
    assert not hasattr(pipeline02, "run_stage")
    assert not hasattr(pipeline02, "run_pretraining_stage")
    assert not hasattr(pipeline02, "run_fewshot_stage")
    assert not hasattr(pipeline02, "_run_single_stage_from_cfg")
