from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.Pipeline_02_pretrain_fewshot as p02


def _write_config(path, body: str) -> str:
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_p02_requires_explicit_pipeline_mode(tmp_path) -> None:
    config = _write_config(tmp_path / "missing_mode.yaml", "pipeline: Pipeline_02_pretrain_fewshot\n")

    with pytest.raises(ValueError, match="pipeline_mode"):
        p02.pipeline(SimpleNamespace(config_path=config, override=None, fs_config_path=None))


def test_p02_rejects_unknown_pipeline_mode(tmp_path) -> None:
    config = _write_config(
        tmp_path / "bad_mode.yaml",
        "pipeline: Pipeline_02_pretrain_fewshot\npipeline_mode: guessed\n",
    )

    with pytest.raises(ValueError, match="Unsupported pipeline_mode"):
        p02.pipeline(SimpleNamespace(config_path=config, override=None, fs_config_path=None))


def test_p02_single_mode_dispatches_without_fallback(tmp_path, monkeypatch) -> None:
    config = _write_config(
        tmp_path / "single.yaml",
        "pipeline: Pipeline_02_pretrain_fewshot\npipeline_mode: single\n",
    )
    called = {}

    def fake_single(args, cfg_dict=None):
        called["mode"] = "single"
        called["cfg"] = cfg_dict
        return {"ok": True}

    monkeypatch.setattr(p02, "run_single_stage", fake_single)

    result = p02.pipeline(SimpleNamespace(config_path=config, override=None, fs_config_path=None))

    assert result == {"ok": True}
    assert called["mode"] == "single"
    assert called["cfg"]["pipeline_mode"] == "single"


def test_p02_staged_mode_requires_stages(tmp_path) -> None:
    config = _write_config(
        tmp_path / "staged.yaml",
        "pipeline: Pipeline_02_pretrain_fewshot\npipeline_mode: staged\n",
    )

    with pytest.raises(ValueError, match="stages"):
        p02.pipeline(SimpleNamespace(config_path=config, override=None, fs_config_path=None))


def test_p02_legacy_mode_requires_fs_config_path(tmp_path) -> None:
    config = _write_config(
        tmp_path / "legacy.yaml",
        "pipeline: Pipeline_02_pretrain_fewshot\npipeline_mode: legacy\n",
    )

    with pytest.raises(ValueError, match="fs_config_path"):
        p02.pipeline(SimpleNamespace(config_path=config, override=None, fs_config_path=None))


def test_p02_run_stage_writes_result_manifest_and_applies_checkpoint(tmp_path, monkeypatch) -> None:
    config = {
        "environment": {"seed": 5},
        "data": {},
        "model": {},
        "task": {"name": "classification"},
        "trainer": {},
    }
    task = object()
    best_task = object()
    trainer = SimpleNamespace(callbacks=[], fit_calls=[], test_calls=[])
    captured = {}

    class FakeData:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeDataFactory:
        def __init__(self):
            self.data = FakeData()

        def get_dataloader(self, split):
            return f"{split}-loader"

    data_factory = FakeDataFactory()

    def fake_fit(task_arg, train_loader, val_loader):
        trainer.fit_calls.append((task_arg, train_loader, val_loader))

    def fake_test(task_arg, test_loader):
        trainer.test_calls.append((task_arg, test_loader))
        return [{"acc": 0.9}]

    trainer.fit = fake_fit
    trainer.test = fake_test

    monkeypatch.setattr(p02, "load_config", lambda config_path, local_config=None: config)
    monkeypatch.setattr(
        p02,
        "prepare_run_context",
        lambda configs, args_environment, args_trainer, iteration: SimpleNamespace(
            run_dir=tmp_path / "run",
            logger_name="stage",
            seed=11,
        ),
    )
    monkeypatch.setattr(p02, "seed_everything", lambda seed: captured.setdefault("seed", seed))
    monkeypatch.setattr(p02, "init_lab", lambda *args, **kwargs: captured.setdefault("init_lab", True))
    monkeypatch.setattr(p02, "close_lab", lambda: captured.setdefault("close_lab", True))
    monkeypatch.setattr(p02, "load_best_model_checkpoint", lambda task_arg, trainer_arg: best_task)

    def fake_build_training_stack(**kwargs):
        captured["weights_path"] = getattr(kwargs["args_model"], "weights_path", "")
        return SimpleNamespace(data_factory=data_factory, task=task, trainer=trainer)

    def fake_write_test_result_and_manifest(**kwargs):
        captured["manifest_call"] = kwargs

    monkeypatch.setattr(p02, "build_training_stack", fake_build_training_stack)
    monkeypatch.setattr(p02, "write_test_result_and_manifest", fake_write_test_result_and_manifest)

    returned_task, returned_trainer = p02.run_stage(
        "stage.yaml",
        ckpt_path="pretrain.ckpt",
        iteration=3,
    )

    assert returned_task is best_task
    assert returned_trainer is trainer
    assert trainer.fit_calls == [(task, "train-loader", "val-loader")]
    assert trainer.test_calls == [(best_task, "test-loader")]
    assert captured["weights_path"] == "pretrain.ckpt"
    assert captured["seed"] == 11
    assert captured["manifest_call"]["metrics"] == {"acc": 0.9}
    assert captured["manifest_call"]["iteration"] == 3
    assert captured["manifest_call"]["stage"] == "test"
    assert captured["manifest_call"]["manifest_required"] is True
    assert data_factory.data.closed is True
    assert captured["close_lab"] is True
