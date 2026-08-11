from __future__ import annotations

from types import SimpleNamespace

import pytest

from src import Pipeline_01_Fault_Diagnosis as pipeline_module


class _SpyDataFactory:
    def __init__(self, *, forbid_test: bool) -> None:
        self.data = self
        self.forbid_test = forbid_test
        self.requested_splits: list[str] = []
        self.close_calls = 0

    def get_metadata(self) -> object:
        return object()

    def get_dataloader(self, split: str) -> str:
        self.requested_splits.append(split)
        if self.forbid_test and split == "test":
            raise AssertionError("the intervention-backed test split was requested")
        return f"{split}-loader"

    def close(self) -> None:
        self.close_calls += 1


class _SpyTrainer:
    def __init__(self) -> None:
        self.fit_calls: list[tuple[object, str, str]] = []
        self.test_calls: list[tuple[object, str]] = []

    def fit(self, task: object, train_loader: str, val_loader: str) -> None:
        self.fit_calls.append((task, train_loader, val_loader))

    def test(self, task: object, test_loader: str) -> list[dict[str, float]]:
        self.test_calls.append((task, test_loader))
        return [{"test_loss": 0.25, "test_acc": 0.75}]


def _config(*, test_after_fit: bool | None) -> SimpleNamespace:
    trainer_values: dict[str, object] = {}
    if test_after_fit is not None:
        trainer_values["test_after_fit"] = test_after_fit
    return SimpleNamespace(
        data=SimpleNamespace(),
        model=SimpleNamespace(),
        task=SimpleNamespace(name="classification"),
        trainer=SimpleNamespace(**trainer_values),
        environment=SimpleNamespace(iterations=1, seed=17),
    )


def _run_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    test_after_fit: bool | None,
    forbid_test: bool,
):
    config = _config(test_after_fit=test_after_fit)
    data_factory = _SpyDataFactory(forbid_test=forbid_test)
    trainer = _SpyTrainer()
    original_task = object()
    checkpoint_task = object()
    checkpoint_calls: list[tuple[object, object]] = []
    aggregate_calls: list[tuple[object, ...]] = []
    close_lab_calls: list[None] = []
    iteration_path = tmp_path / "run" / "iter_0"
    iteration_path.mkdir(parents=True)

    monkeypatch.setattr(
        pipeline_module,
        "merge_with_local_override",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(pipeline_module, "seed_everything", lambda _seed: None)
    monkeypatch.setattr(
        pipeline_module,
        "path_name",
        lambda _config, _iteration: (str(iteration_path), "p04-gate-test"),
    )
    monkeypatch.setattr(pipeline_module, "init_lab", lambda *_args: None)
    monkeypatch.setattr(pipeline_module, "build_data", lambda *_args: data_factory)
    monkeypatch.setattr(pipeline_module, "build_model", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        pipeline_module,
        "build_task",
        lambda **_kwargs: original_task,
    )
    monkeypatch.setattr(pipeline_module, "build_trainer", lambda *_args: trainer)

    def _load_checkpoint(task: object, built_trainer: object) -> object:
        checkpoint_calls.append((task, built_trainer))
        return checkpoint_task

    monkeypatch.setattr(pipeline_module, "load_best_model_checkpoint", _load_checkpoint)
    monkeypatch.setattr(
        pipeline_module,
        "close_lab",
        lambda: close_lab_calls.append(None),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_write_aggregate_outputs",
        lambda *args: aggregate_calls.append(args),
    )

    args = SimpleNamespace(
        config_path="unused.yaml",
        local_config=None,
        override=None,
        notes="",
    )
    result = pipeline_module.pipeline(args)
    return {
        "result": result,
        "data_factory": data_factory,
        "trainer": trainer,
        "original_task": original_task,
        "checkpoint_task": checkpoint_task,
        "checkpoint_calls": checkpoint_calls,
        "aggregate_calls": aggregate_calls,
        "close_lab_calls": close_lab_calls,
        "iteration_path": iteration_path,
    }


def test_false_gate_never_requests_intervention_test_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    observed = _run_pipeline(
        monkeypatch,
        tmp_path,
        test_after_fit=False,
        forbid_test=True,
    )

    assert observed["result"] == []
    assert observed["data_factory"].requested_splits == ["train", "val"]
    assert observed["trainer"].test_calls == []
    assert observed["checkpoint_calls"] == [
        (observed["original_task"], observed["trainer"])
    ]
    assert observed["data_factory"].close_calls == 1
    assert len(observed["close_lab_calls"]) == 1
    assert observed["aggregate_calls"] == []
    assert not (observed["iteration_path"] / "test_result_0.csv").exists()


def test_missing_gate_defaults_to_existing_test_and_aggregate_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    observed = _run_pipeline(
        monkeypatch,
        tmp_path,
        test_after_fit=None,
        forbid_test=False,
    )

    assert observed["result"] == [{"test_loss": 0.25, "test_acc": 0.75}]
    assert observed["data_factory"].requested_splits == ["train", "val", "test"]
    assert observed["trainer"].test_calls == [
        (observed["checkpoint_task"], "test-loader")
    ]
    assert observed["data_factory"].close_calls == 1
    assert len(observed["close_lab_calls"]) == 1
    assert len(observed["aggregate_calls"]) == 1
    assert (observed["iteration_path"] / "test_result_0.csv").is_file()
