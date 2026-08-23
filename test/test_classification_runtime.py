from __future__ import annotations

from argparse import Namespace
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from phmfactory.config import ResolvedConfig
from phmfactory.runtime import CompiledRunSpec
from src.data_factory import ExplicitDataFactory, resolve_data_factory_class
from src.data_factory.dataset_task.Dataset_cluster import IdIncludedDataset
from src.data_factory.dataset_task.Default_dataset import Default_dataset
from src.data_factory.dataset_task.adapters import (
    DATASET_ADAPTERS,
    resolve_dataset_adapter,
)
from src.data_factory.samplers.Get_sampler import Get_sampler
from src.runtime import classification


task_factory_module = importlib.import_module("src.task_factory.task_factory")
trainer_factory_module = importlib.import_module(
    "src.trainer_factory.trainer_factory"
)


def _config(tmp_path: Path, *, iterations: int = 1) -> dict:
    return {
        "pipeline": "Pipeline_01_Fault_Diagnosis",
        "environment": {
            "iterations": iterations,
            "seed": 7,
            "output_dir": str(tmp_path),
            "project": "runtime-test",
            "wandb": False,
            "swanlab": False,
        },
        "data": {"data_dir": str(tmp_path), "metadata_file": "dummy.csv"},
        "model": {"name": "dummy", "type": "dummy"},
        "task": {"name": "classification", "type": "DG"},
        "trainer": {
            "device": "cpu",
            "gpus": 1,
            "num_epochs": 1,
            "test_after_fit": True,
        },
    }


def _args(tmp_path: Path, *, iterations: int = 1) -> Namespace:
    resolved = ResolvedConfig(
        requested="smoke",
        path=tmp_path / "smoke.yaml",
        data=_config(tmp_path, iterations=iterations),
        pipeline="Pipeline_01_Fault_Diagnosis",
        overrides={},
    )
    compiled = CompiledRunSpec.compile(resolved)
    return Namespace(
        config="smoke",
        config_path=str(resolved.path),
        requested_config="smoke",
        resolved_pipeline=resolved.pipeline,
        compiled_run_spec=compiled,
        override=["trainer.num_epochs=99"],
        notes="",
    )


def _window_args(**values) -> SimpleNamespace:
    payload = {
        "window_size": 4,
        "stride": 4,
        "num_window": 5,
        "window_sampling_strategy": "sequential",
        "window_sampling_seed": 0,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "normalization": "none",
        "dtype": "float32",
        "batch_size": 4,
    }
    payload.update(values)
    return SimpleNamespace(**payload)


def _window_starts(dataset: Default_dataset) -> list[int]:
    return [int(item["x"][0, 0]) for item in dataset]


def _raise_key_error(key):
    raise KeyError(key)


def _raise_module_not_found(message: str):
    raise ModuleNotFoundError(message)


def test_compiled_config_bypasses_legacy_reparse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    monkeypatch.setattr(
        classification,
        "merge_with_local_override",
        lambda *a, **k: pytest.fail("legacy loader must not run"),
    )
    configs = classification.load_runtime_config(args)
    assert configs.environment.seed == 7
    assert configs.trainer.num_epochs == 1
    assert configs.trainer.test_after_fit is True


def test_missing_required_section_fails_closed(tmp_path: Path) -> None:
    args = _args(tmp_path)
    data = args.compiled_run_spec.runtime_config()
    data.pop("task")
    args.compiled_run_spec = CompiledRunSpec.compile(
        ResolvedConfig(
            requested="broken",
            path=tmp_path / "broken.yaml",
            data=data,
            pipeline="Pipeline_01_Fault_Diagnosis",
            overrides={},
        )
    )
    with pytest.raises(ValueError, match="missing required section.*task"):
        classification.load_runtime_config(args)


def test_zero_iterations_fails_before_factory_construction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="iterations must be positive"):
        classification.run_classification_pipeline(_args(tmp_path, iterations=0))


def test_runtime_closes_data_and_lab_when_training_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class DataFactory:
        data = SimpleNamespace(close=lambda: events.append("data-close"))

        def get_metadata(self):
            return {0: {"Label": 0, "Domain_id": 0}}

        def get_dataloader(self, split: str):
            return split

    class Trainer:
        def fit(self, *args):
            raise RuntimeError("fit failed")

    monkeypatch.setattr(
        classification,
        "path_name",
        lambda configs, iteration: (str(tmp_path / "run"), "run"),
    )
    monkeypatch.setattr(
        classification,
        "seed_everything",
        lambda seed: events.append(f"seed:{seed}"),
    )
    monkeypatch.setattr(
        classification,
        "init_lab",
        lambda *args: events.append("lab-open"),
    )
    monkeypatch.setattr(
        classification,
        "close_lab",
        lambda: events.append("lab-close"),
    )
    monkeypatch.setattr(classification, "build_data", lambda *args: DataFactory())
    monkeypatch.setattr(
        classification,
        "build_model",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(classification, "build_task", lambda **kwargs: object())
    monkeypatch.setattr(classification, "build_trainer", lambda *args: Trainer())

    with pytest.raises(RuntimeError, match="fit failed"):
        classification.run_classification_pipeline(_args(tmp_path))

    assert events[-2:] == ["data-close", "lab-close"]


def test_pipeline_wrappers_only_select_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.Pipeline_01_Fault_Diagnosis as pipeline_01
    import src.Pipeline_05_Explainable_Fault_Diagnosis as pipeline_05

    calls: list[object] = []
    monkeypatch.setattr(
        pipeline_01,
        "run_classification_pipeline",
        lambda args: calls.append(("p01", args)) or [],
    )
    monkeypatch.setattr(
        pipeline_05,
        "run_classification_pipeline",
        lambda args, hooks: calls.append(("p05", args, type(hooks).__name__)) or [],
    )

    marker = object()
    assert pipeline_01.pipeline(marker) == []
    assert pipeline_05.pipeline(marker) == []
    assert calls == [("p01", marker), ("p05", marker, "ExplainabilityHooks")]


def test_same_file_fewshot_windows_are_disjoint_across_splits() -> None:
    raw = np.arange(40, dtype=np.float32).reshape(20, 2)
    data = {1: raw}
    metadata = {1: {"Label": 0}}
    args_data = _window_args()
    args_task = SimpleNamespace(type="FS")

    train = Default_dataset(data, metadata, args_data, args_task, "train")
    val = Default_dataset(data, metadata, args_data, args_task, "val")
    test = Default_dataset(data, metadata, args_data, args_task, "test")

    assert _window_starts(train) == [0, 8, 16]
    assert _window_starts(val) == [24]
    assert _window_starts(test) == [32]
    assert set(_window_starts(train)).isdisjoint(_window_starts(val))
    assert set(_window_starts(train)).isdisjoint(_window_starts(test))
    assert set(_window_starts(val)).isdisjoint(_window_starts(test))


def test_valid_alias_uses_the_validation_slice() -> None:
    raw = np.arange(40, dtype=np.float32).reshape(20, 2)
    data = {1: raw}
    metadata = {1: {"Label": 0}}
    args_data = _window_args()
    args_task = SimpleNamespace(type="DG")

    val = Default_dataset(data, metadata, args_data, args_task, "val")
    valid = Default_dataset(data, metadata, args_data, args_task, "valid")

    assert _window_starts(val) == _window_starts(valid) == [24]


def test_evaluation_sampler_keeps_a_short_final_batch() -> None:
    raw = np.arange(40, dtype=np.float32).reshape(20, 2)
    data = {1: raw}
    metadata = {1: {"Label": 0, "Dataset_id": 7}}
    args_data = _window_args(batch_size=8)
    args_task = SimpleNamespace(type="DG")
    val_dataset = Default_dataset(data, metadata, args_data, args_task, "val")
    clustered = IdIncludedDataset({1: val_dataset}, metadata)

    sampler = Get_sampler(args_task, args_data, clustered, mode="val")

    assert len(clustered) == 1
    assert len(sampler) == 1
    assert list(sampler) == [[0]]


def test_short_signal_fails_with_actionable_message() -> None:
    args_data = _window_args(window_size=16)
    args_task = SimpleNamespace(type="DG")

    with pytest.raises(
        ValueError,
        match="Reduce window_size or provide a longer signal",
    ):
        Default_dataset(
            {1: np.zeros((8, 2), dtype=np.float32)},
            {1: {"Label": 0}},
            args_data,
            args_task,
            "train",
        )


def test_default_data_factory_uses_explicit_adapter_resolution() -> None:
    assert resolve_data_factory_class("default") is ExplicitDataFactory


@pytest.mark.parametrize("task_type,task_name", sorted(DATASET_ADAPTERS))
def test_every_registered_dataset_adapter_imports(
    task_type: str,
    task_name: str,
) -> None:
    dataset_class = resolve_dataset_adapter(task_type, task_name)
    assert isinstance(dataset_class, type)


def test_unknown_dataset_adapter_fails_with_registered_combinations() -> None:
    with pytest.raises(
        ValueError,
        match="No dataset adapter is registered.*Add an explicit adapter",
    ):
        resolve_dataset_adapter("unknown", "task")


def test_unknown_data_factory_name_fails_with_available_factories() -> None:
    with pytest.raises(ValueError, match="Unknown data.factory_name"):
        resolve_data_factory_class("auto_guess")


def test_task_import_failure_preserves_requested_module_and_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        task_factory_module.TASK_REGISTRY,
        "get",
        _raise_key_error,
    )
    monkeypatch.setattr(
        task_factory_module.importlib,
        "import_module",
        lambda path: _raise_module_not_found("missing optional package"),
    )
    args_task = SimpleNamespace(type="DG", name="missing_task")

    with pytest.raises(
        ImportError,
        match=(
            "DG.missing_task.*src.task_factory.task.DG.missing_task.*"
            "missing optional package"
        ),
    ) as captured:
        task_factory_module.task_factory(
            args_task=args_task,
            network=object(),
            args_data=SimpleNamespace(),
            args_model=SimpleNamespace(),
            args_trainer=SimpleNamespace(),
            args_environment=SimpleNamespace(),
            metadata={},
        )

    assert isinstance(captured.value.__cause__, ModuleNotFoundError)


def test_task_module_requires_registration_or_task_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        task_factory_module.TASK_REGISTRY,
        "get",
        _raise_key_error,
    )
    monkeypatch.setattr(
        task_factory_module.importlib,
        "import_module",
        lambda path: SimpleNamespace(),
    )

    with pytest.raises(
        AttributeError,
        match="does not register.*does not expose.*'task'",
    ):
        task_factory_module.task_factory(
            args_task=SimpleNamespace(type="DG", name="empty_module"),
            network=object(),
            args_data=SimpleNamespace(),
            args_model=SimpleNamespace(),
            args_trainer=SimpleNamespace(),
            args_environment=SimpleNamespace(),
            metadata={},
        )


def test_task_construction_failure_preserves_original_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenTask:
        def __init__(self, **kwargs):
            del kwargs
            raise ValueError("invalid task dimensions")

    monkeypatch.setattr(
        task_factory_module.TASK_REGISTRY,
        "get",
        lambda key: BrokenTask,
    )

    with pytest.raises(ValueError, match="invalid task dimensions"):
        task_factory_module.task_factory(
            args_task=SimpleNamespace(type="DG", name="classification"),
            network=object(),
            args_data=SimpleNamespace(),
            args_model=SimpleNamespace(),
            args_trainer=SimpleNamespace(),
            args_environment=SimpleNamespace(),
            metadata={},
        )


def test_trainer_import_failure_preserves_requested_module_and_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trainer_factory_module.TRAINER_REGISTRY,
        "get",
        _raise_key_error,
    )
    monkeypatch.setattr(
        trainer_factory_module.importlib,
        "import_module",
        lambda path: _raise_module_not_found("trainer dependency missing"),
    )

    with pytest.raises(
        ImportError,
        match=(
            "MissingTrainer.*src.trainer_factory.MissingTrainer.*"
            "trainer dependency missing"
        ),
    ) as captured:
        trainer_factory_module.trainer_factory(
            args_environment=SimpleNamespace(),
            args_trainer=SimpleNamespace(name="MissingTrainer"),
            args_data=SimpleNamespace(),
            path="results/run",
        )

    assert isinstance(captured.value.__cause__, ModuleNotFoundError)


def test_trainer_construction_failure_preserves_original_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_trainer(**kwargs):
        del kwargs
        raise OSError("output path is read-only")

    monkeypatch.setattr(
        trainer_factory_module.TRAINER_REGISTRY,
        "get",
        lambda key: broken_trainer,
    )

    with pytest.raises(OSError, match="output path is read-only"):
        trainer_factory_module.trainer_factory(
            args_environment=SimpleNamespace(),
            args_trainer=SimpleNamespace(name="Default_trainer"),
            args_data=SimpleNamespace(),
            path="results/run",
        )
