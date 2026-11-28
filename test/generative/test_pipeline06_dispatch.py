from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

import src.Pipeline_06_Generative_Modeling as pipeline06


def _configs(mode: str, iterations: int = 2, **generative_values) -> SimpleNamespace:
    return SimpleNamespace(
        environment=SimpleNamespace(iterations=iterations),
        data=SimpleNamespace(),
        model=SimpleNamespace(),
        task=SimpleNamespace(
            generative=SimpleNamespace(mode=mode, **generative_values)
        ),
        trainer=SimpleNamespace(),
    )


@pytest.mark.parametrize(
    ("mode", "handler_name", "stage_values"),
    [
        ("train", "_run_train_stage", {}),
        (
            "sample",
            "_run_sample_stage",
            {
                "checkpoint_path": "checkpoint.ckpt",
                "normalization_path": "normalization_params.json",
                "normalization_sha256": "digest",
            },
        ),
        (
            "eval",
            "_run_eval_stage",
            {"generated_path": "samples.pt"},
        ),
    ],
)
def test_pipeline_dispatches_only_the_selected_stage(
    monkeypatch,
    mode: str,
    handler_name: str,
    stage_values: dict[str, str],
) -> None:
    configs = _configs(mode, iterations=2, **stage_values)
    calls: list[tuple[str, int]] = []

    monkeypatch.setattr(pipeline06, "_load_configs", lambda args: configs)

    for candidate in ["_run_train_stage", "_run_sample_stage", "_run_eval_stage"]:
        if candidate == handler_name:
            monkeypatch.setattr(
                pipeline06,
                candidate,
                lambda args, current_configs, iteration, selected=mode: (
                    calls.append((selected, iteration))
                    or {"mode": selected, "iteration": iteration}
                ),
            )
        else:
            monkeypatch.setattr(
                pipeline06,
                candidate,
                lambda *args, unexpected=candidate, **kwargs: pytest.fail(
                    f"unexpected handler called: {unexpected}"
                ),
            )

    results = pipeline06.pipeline(SimpleNamespace(config_path="unused.yaml"))

    assert calls == [(mode, 0), (mode, 1)]
    assert results == [
        {"mode": mode, "iteration": 0},
        {"mode": mode, "iteration": 1},
    ]


def test_stage_failure_is_recorded_and_reraised(monkeypatch) -> None:
    configs = _configs("train", iterations=1)
    records = []
    monkeypatch.setattr(pipeline06, "_load_configs", lambda args: configs)
    monkeypatch.setattr(
        pipeline06,
        "_run_train_stage",
        lambda *args: (_ for _ in ()).throw(RuntimeError("stage exploded")),
    )
    monkeypatch.setattr(
        pipeline06,
        "_record_stage",
        lambda configs, stage, **values: records.append((stage, values)),
    )

    with pytest.raises(RuntimeError, match="stage exploded"):
        pipeline06.pipeline(SimpleNamespace(config_path="unused.yaml"))

    assert records == [
        (
            "train",
            {
                "status": "failed",
                "iteration": 0,
                "error_type": "RuntimeError",
                "error": "stage exploded",
            },
        )
    ]


def test_ledger_failure_does_not_replace_stage_failure(monkeypatch) -> None:
    configs = _configs("train", iterations=1)
    monkeypatch.setattr(pipeline06, "_load_configs", lambda args: configs)
    monkeypatch.setattr(
        pipeline06,
        "_run_train_stage",
        lambda *args: (_ for _ in ()).throw(RuntimeError("stage exploded")),
    )
    monkeypatch.setattr(
        pipeline06,
        "_record_stage",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("ledger unavailable")),
    )

    with pytest.raises(RuntimeError, match="stage exploded") as captured:
        pipeline06.pipeline(SimpleNamespace(config_path="unused.yaml"))

    assert isinstance(captured.value.__cause__, OSError)
    assert str(captured.value.__cause__) == "ledger unavailable"


@pytest.mark.parametrize("extension", [False, True])
def test_factory_extension_registers_before_data_construction(
    monkeypatch, tmp_path, extension: bool,
) -> None:
    events = []
    rows = {
        1: {"Id": 1, "Label": 0, "Domain_id": 0},
        2: {"Id": 2, "Label": 1, "Domain_id": 1},
        3: {"Id": 3, "Label": 9, "Domain_id": 9},
    }

    class RowSelection:
        def __getitem__(self, ids):
            return {key: rows[key] for key in ids}

    metadata = SimpleNamespace(
        df=SimpleNamespace(loc=RowSelection()), key_column="Id",
    )
    data = SimpleNamespace(
        get_metadata=lambda: metadata,
        train_dataset=SimpleNamespace(dataset_dict={1: object(), 2: object()}),
    )
    model, task = object(), object()
    data_module = ModuleType("src.data_factory")
    metadata_module = ModuleType("src.data_factory.data_utils")
    model_module = ModuleType("src.model_factory")
    task_module = ModuleType("src.task_factory")
    data_module.events = events
    data_module.build_data = lambda *args: events.append("data") or data
    # The shell gate verifies selection and dispatch without numerical packages.
    def metadata_accessor(selected, *, key_column):
        assert key_column == "Id"
        return selected

    metadata_module.MetadataAccessor = metadata_accessor

    def build_model(*args, **kwargs):
        assert kwargs["metadata"] == {1: rows[1], 2: rows[2]}
        events.append("model")
        return model

    model_module.build_model = build_model
    task_module.build_task = lambda **kwargs: events.append("task") or task
    for module in (data_module, metadata_module, model_module, task_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    args_task = SimpleNamespace()
    module_name = "generative_test_extension"
    if extension:
        (tmp_path / f"{module_name}.py").write_text(
            "from src.data_factory import events\nevents.append('registered')\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        args_task.factory_module = module_name
    try:
        result = pipeline06._build_stack(None, None, args_task, None, None)
    finally:
        sys.modules.pop(module_name, None)
    assert result == (data, model, task)
    assert events == (["registered"] if extension else []) + ["data", "model", "task"]


def test_factory_extension_import_failure_prevents_data_construction(monkeypatch) -> None:
    calls = []
    for name, function in (
        ("src.data_factory", "build_data"),
        ("src.model_factory", "build_model"),
        ("src.task_factory", "build_task"),
    ):
        module = ModuleType(name)
        setattr(module, function, lambda *args, **kwargs: calls.append("built"))
        monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(ModuleNotFoundError, match="missing_generative_extension"):
        pipeline06._build_stack(
            None, None,
            SimpleNamespace(factory_module="missing_generative_extension"),
            None, None,
        )
    assert calls == []
