from __future__ import annotations

from types import SimpleNamespace

from src.utils.training import run_contract


def test_prepare_run_context_snapshots_config(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    env = SimpleNamespace(seed=10)
    trainer = SimpleNamespace()
    cfg = {"environment": {"seed": 10}}
    saved = {}

    monkeypatch.setattr(run_contract, "path_name", lambda config, iteration=0: (str(run_dir), "exp"))

    def fake_save_config(config, output_path):
        saved["config"] = config
        output_path.write_text("snapshot: true\n", encoding="utf-8")

    monkeypatch.setattr(run_contract, "save_config", fake_save_config)

    ctx = run_contract.prepare_run_context(cfg, env, trainer, iteration=3)

    assert ctx.run_dir == run_dir
    assert ctx.logger_name == "exp"
    assert ctx.seed == 13
    assert trainer.logger_name == "exp"
    assert trainer.run_dir == str(run_dir)
    assert (run_dir / "config_snapshot.yaml").exists()
    assert saved["config"] is cfg


def test_build_training_stack_wires_factories_and_sidecars(tmp_path, monkeypatch) -> None:
    calls = []
    data_factory = SimpleNamespace(
        get_metadata=lambda: {"channels": 2},
    )
    model = object()
    task = SimpleNamespace()
    trainer = SimpleNamespace()

    monkeypatch.setattr(run_contract, "build_data", lambda args_data, args_task: data_factory)
    monkeypatch.setattr(
        run_contract,
        "write_run_artifact_sidecars",
        lambda **kwargs: ({"x_shape": [1, 2]}, "batch", False),
    )
    monkeypatch.setattr(
        run_contract,
        "build_model",
        lambda args_model, metadata: calls.append(("model", metadata)) or model,
    )
    monkeypatch.setattr(
        run_contract,
        "build_task",
        lambda **kwargs: calls.append(("task", kwargs["metadata"])) or task,
    )
    monkeypatch.setattr(
        run_contract,
        "build_trainer",
        lambda args_environment, args_trainer, args_data, run_dir: trainer,
    )

    components = run_contract.build_training_stack(
        args_environment=SimpleNamespace(),
        args_data=SimpleNamespace(),
        args_model=SimpleNamespace(),
        args_task=SimpleNamespace(),
        args_trainer=SimpleNamespace(),
        run_dir=tmp_path,
        attach_data_factory=True,
        sidecar_config={"pipeline": "Pipeline_01_default"},
    )

    assert components.data_factory is data_factory
    assert components.model is model
    assert components.task is task
    assert components.trainer is trainer
    assert components.batch_meta == {"x_shape": [1, 2]}
    assert components.meta_source == "batch"
    assert components.degraded is False
    assert getattr(task, "_data_factory") is data_factory
    assert calls == [("model", {"channels": 2}), ("task", {"channels": 2})]


def test_write_test_result_and_manifest(tmp_path, monkeypatch) -> None:
    manifest_calls = []

    def fake_write_run_manifest(**kwargs):
        manifest_calls.append(kwargs)
        path = tmp_path / "artifacts" / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr(run_contract, "write_run_manifest", fake_write_run_manifest)

    metrics_path = run_contract.write_test_result_and_manifest(
        run_dir=tmp_path,
        metrics={"acc": 1.0},
        iteration=2,
        args_trainer=SimpleNamespace(logger_name="demo"),
        seed=7,
        trainer=SimpleNamespace(),
        stage="test",
        manifest_required=True,
    )

    assert metrics_path == tmp_path / "test_result_2.csv"
    assert "acc" in metrics_path.read_text(encoding="utf-8")
    assert manifest_calls[0]["run_id"] == "demo"
    assert manifest_calls[0]["seed"] == 7
    assert manifest_calls[0]["required"] is True


def test_write_test_result_respects_disabled_manifest(tmp_path, monkeypatch) -> None:
    def fail_if_called(**kwargs):
        raise AssertionError("manifest writer should not be called")

    monkeypatch.setattr(run_contract, "write_run_manifest", fail_if_called)

    metrics_path = run_contract.write_test_result_and_manifest(
        run_dir=tmp_path,
        metrics={"acc": 1.0},
        iteration=0,
        args_trainer=SimpleNamespace(
            logger_name="demo",
            extensions=SimpleNamespace(report=SimpleNamespace(enable=False, manifest=True)),
        ),
        seed=7,
        trainer=SimpleNamespace(),
    )

    assert metrics_path == tmp_path / "test_result_0.csv"
    assert "acc" in metrics_path.read_text(encoding="utf-8")
    assert not (tmp_path / "artifacts" / "manifest.json").exists()
