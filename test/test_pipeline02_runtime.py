from __future__ import annotations

from argparse import Namespace
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

from phmfactory.config import ResolvedConfig
from phmfactory.runtime import CompiledRunSpec
import src.Pipeline_02_Pretraining_Few_Shot as pipeline02
import src.utils.training.two_stage_orchestrator as orchestrator_module
from src.utils.pipeline_config.base_utils import load_pretrained_weights
from src.utils.utils import load_best_model_checkpoint

model_factory_module = importlib.import_module("src.model_factory.model_factory")


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)


class TinyLightningTask(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.network = TinyModel()


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
    expected = {
        "stage_1": {
            "checkpoint_path": "stage1.ckpt",
            "metrics": {"test_loss": 1.0},
        }
    }

    class Orchestrator:
        def __init__(self, config, cli_overrides):
            observed["config"] = config
            observed["overrides"] = cli_overrides

        def run_complete(self):
            return expected

    monkeypatch.setattr(pipeline02, "TwoStageOrchestrator", Orchestrator)
    monkeypatch.setattr(
        pipeline02,
        "run_classification_pipeline",
        lambda args: pytest.fail("single-stage runtime must not run"),
    )
    args = _args(
        tmp_path,
        stages=[
            {"name": "pretrain", "overrides": {}},
            {"name": "finetune", "overrides": {}},
        ],
    )

    assert pipeline02.pipeline(args) == expected
    assert observed["overrides"] == []
    assert observed["config"]["stages"][0]["name"] == "pretrain"


def test_multistage_rejects_stage_without_evaluation_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Orchestrator:
        def __init__(self, config, cli_overrides):
            pass

        def run_complete(self):
            return {"stage_1": {"checkpoint_path": "stage1.ckpt", "metrics": {}}}

    monkeypatch.setattr(pipeline02, "TwoStageOrchestrator", Orchestrator)

    with pytest.raises(RuntimeError, match="did not complete evaluation"):
        pipeline02.pipeline(_args(tmp_path, stages=[{"overrides": {}}]))


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
    expected = {
        "stage_1": {
            "checkpoint_path": "stage1.ckpt",
            "metrics": {"test_loss": 1.0},
        }
    }

    class Orchestrator:
        def __init__(self, config):
            assert config is unified

        def run_complete(self):
            return expected

    monkeypatch.setattr(pipeline02, "TwoStageOrchestrator", Orchestrator)
    args = _args(
        tmp_path,
        fs_config_path="fewshot.yaml",
        pipeline_mode="legacy_dual_yaml",
        local_config=None,
        override=None,
    )

    assert pipeline02.pipeline(args) == expected
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


def test_model_factory_fails_when_configured_checkpoint_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = SimpleNamespace(Model=lambda args, metadata: TinyModel())
    monkeypatch.setattr(
        model_factory_module.importlib,
        "import_module",
        lambda module_path: module,
    )
    args = SimpleNamespace(
        type="Dummy",
        name="Tiny",
        num_classes=2,
        weights_path=str(tmp_path / "missing.ckpt"),
    )

    with pytest.raises(
        FileNotFoundError,
        match="Configured checkpoint does not exist",
    ):
        model_factory_module.model_factory(args, metadata=None)


def test_load_ckpt_restores_real_lightning_network_state(tmp_path: Path) -> None:
    source_task = TinyLightningTask()
    target = TinyModel()
    for parameter in target.parameters():
        torch.nn.init.zeros_(parameter)

    checkpoint = tmp_path / "lightning.ckpt"
    torch.save({"state_dict": source_task.state_dict()}, checkpoint)

    model_factory_module.load_ckpt(target, str(checkpoint))

    for expected, actual in zip(source_task.network.parameters(), target.parameters()):
        assert torch.equal(expected, actual)


def test_load_ckpt_non_strict_rejects_zero_matches(tmp_path: Path) -> None:
    checkpoint = tmp_path / "wrong.ckpt"
    torch.save({"unrelated.weight": torch.ones(1)}, checkpoint)

    with pytest.raises(RuntimeError, match="matched zero model parameters"):
        model_factory_module.load_ckpt(
            TinyModel(),
            str(checkpoint),
            strict=False,
        )


def _stage_orchestrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    test_result=None,
    test_error: Exception | None = None,
):
    orchestrator = orchestrator_module.MultiStageOrchestrator.__new__(
        orchestrator_module.MultiStageOrchestrator
    )
    orchestrator.cfg = SimpleNamespace()
    orchestrator.dry_run = False

    env = SimpleNamespace(seed=1)
    data = SimpleNamespace()
    model = SimpleNamespace()
    task = SimpleNamespace()
    trainer_config = SimpleNamespace()
    monkeypatch.setattr(
        orchestrator,
        "_stage_to_namespaces",
        lambda stage_cfg: (env, data, model, task, trainer_config),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "path_name",
        lambda config: (str(tmp_path), "stage"),
    )
    monkeypatch.setattr(orchestrator_module, "seed_everything", lambda seed: None)

    close_events: list[str] = []
    monkeypatch.setattr(orchestrator_module, "init_lab", lambda *args: None)
    monkeypatch.setattr(
        orchestrator_module,
        "close_lab",
        lambda: close_events.append("lab"),
    )

    class DataResource:
        def close(self):
            close_events.append("data")

    class DataFactory:
        data = DataResource()

        def get_metadata(self):
            return None

        def get_dataloader(self, split):
            return split

    monkeypatch.setattr(
        orchestrator_module,
        "build_data",
        lambda args_data, args_task: DataFactory(),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "build_model",
        lambda args_model, metadata: TinyModel(),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "build_task",
        lambda **kwargs: SimpleNamespace(),
    )

    callback = ModelCheckpoint()
    callback.best_model_path = str(tmp_path / "best.ckpt")

    class Trainer:
        callbacks = [callback]

        def fit(self, *args):
            return None

        def test(self, *args):
            if test_error is not None:
                raise test_error
            return test_result

    monkeypatch.setattr(
        orchestrator_module,
        "build_trainer",
        lambda *args: Trainer(),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "load_best_model_checkpoint",
        lambda lightning_task, trainer: lightning_task,
    )
    return orchestrator, close_events


@pytest.mark.parametrize("method_name", ["run_pretrain", "run_adapt"])
def test_stage_evaluation_error_propagates_and_resources_close(
    method_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, close_events = _stage_orchestrator(
        tmp_path,
        monkeypatch,
        test_error=RuntimeError("evaluation failed"),
    )

    method = getattr(orchestrator, method_name)
    kwargs = (
        {"stage_cfg": object()}
        if method_name == "run_pretrain"
        else {"stage_cfg": object(), "checkpoint_path": None}
    )
    with pytest.raises(RuntimeError, match="evaluation failed"):
        method(**kwargs)

    assert close_events == ["data", "lab"]


@pytest.mark.parametrize("test_result", [[], [{}]])
def test_stage_rejects_empty_test_metrics_and_resources_close(
    test_result,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, close_events = _stage_orchestrator(
        tmp_path,
        monkeypatch,
        test_result=test_result,
    )

    with pytest.raises(RuntimeError, match="non-empty metrics mapping"):
        orchestrator.run_pretrain(stage_cfg=object())

    assert close_events == ["data", "lab"]


def test_pipeline02_adapt_uses_explicit_backbone_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_task = TinyLightningTask()
    checkpoint = tmp_path / "stage1.ckpt"
    torch.save({"state_dict": source_task.state_dict()}, checkpoint)

    target = TinyModel()
    for parameter in target.parameters():
        torch.nn.init.zeros_(parameter)

    orchestrator = orchestrator_module.MultiStageOrchestrator.__new__(
        orchestrator_module.MultiStageOrchestrator
    )
    orchestrator.cfg = SimpleNamespace()
    orchestrator.dry_run = False

    env = SimpleNamespace(seed=1)
    data = SimpleNamespace()
    model = SimpleNamespace()
    task = SimpleNamespace()
    trainer_config = SimpleNamespace()
    monkeypatch.setattr(
        orchestrator,
        "_stage_to_namespaces",
        lambda stage_cfg: (env, data, model, task, trainer_config),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "path_name",
        lambda config: (str(tmp_path), "adapt"),
    )
    monkeypatch.setattr(orchestrator_module, "seed_everything", lambda seed: None)
    monkeypatch.setattr(orchestrator_module, "init_lab", lambda *args: None)
    monkeypatch.setattr(orchestrator_module, "close_lab", lambda: None)

    class DataFactory:
        def get_metadata(self):
            return None

        def get_dataloader(self, split):
            return split

    monkeypatch.setattr(
        orchestrator_module,
        "build_data",
        lambda args_data, args_task: DataFactory(),
    )

    def build_model(args_model, metadata):
        assert not hasattr(args_model, "weights_path")
        return target

    monkeypatch.setattr(orchestrator_module, "build_model", build_model)
    monkeypatch.setattr(
        orchestrator_module,
        "build_task",
        lambda **kwargs: SimpleNamespace(),
    )

    best_checkpoint = tmp_path / "stage2.ckpt"
    callback = ModelCheckpoint()
    callback.best_model_path = str(best_checkpoint)

    class Trainer:
        callbacks = [callback]

        def fit(self, *args):
            return None

        def test(self, *args):
            return [{"loss": 0.0}]

    monkeypatch.setattr(
        orchestrator_module,
        "build_trainer",
        lambda *args: Trainer(),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "load_best_model_checkpoint",
        lambda lightning_task, trainer: lightning_task,
    )

    result = orchestrator.run_adapt(
        stage_cfg=object(),
        checkpoint_path=str(checkpoint),
    )

    for expected, actual in zip(source_task.network.parameters(), target.parameters()):
        assert torch.equal(expected, actual)
    assert result["checkpoint_path"] == str(best_checkpoint)


def test_best_checkpoint_must_exist_before_evaluation(tmp_path: Path) -> None:
    callback = ModelCheckpoint()
    callback.best_model_path = str(tmp_path / "missing-best.ckpt")
    trainer = SimpleNamespace(callbacks=[callback])

    with pytest.raises(
        FileNotFoundError,
        match="Best checkpoint does not exist",
    ):
        load_best_model_checkpoint(TinyModel(), trainer)


def test_best_checkpoint_path_cannot_be_empty() -> None:
    callback = ModelCheckpoint()
    callback.best_model_path = ""
    trainer = SimpleNamespace(callbacks=[callback])

    with pytest.raises(
        RuntimeError,
        match="did not produce a best checkpoint",
    ):
        load_best_model_checkpoint(TinyModel(), trainer)


def test_pretrained_loader_rejects_checkpoint_without_backbone_weights(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "no-backbone.ckpt"
    torch.save(
        {"state_dict": {"task_head.weight": torch.ones(1)}},
        checkpoint,
    )

    with pytest.raises(
        RuntimeError,
        match="no transferable 'network.' backbone",
    ):
        load_pretrained_weights(
            TinyModel(),
            str(checkpoint),
            strict=False,
        )
