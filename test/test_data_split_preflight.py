from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from src.data_factory.data_factory import (
    DatasetResolutionError,
    SplitContractError,
    data_factory,
    resolve_dataset_class,
    validate_split_preflight,
)
from src.data_factory.dataset_task.Default_dataset import Default_dataset


data_factory_module = importlib.import_module("src.data_factory.data_factory")


def _split_config(**overrides: object) -> SimpleNamespace:
    values = {
        "strategy": "grouped_metadata",
        "group_key": "physical_unit_id",
        "manifest_path": "unused-split-manifest.json",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_grouped_split_config_is_consumed_or_rejected() -> None:
    class NoIoFactory(data_factory):
        def _init_metadata(self, args_data):  # pragma: no cover - must not run
            raise AssertionError("metadata I/O occurred before split preflight")

    args_data = SimpleNamespace(split=_split_config())
    args_task = SimpleNamespace(type="DG", name="classification")

    with pytest.raises(SplitContractError, match="not implemented"):
        NoIoFactory(args_data, args_task)


def test_missing_group_key_fails_closed() -> None:
    args_data = SimpleNamespace(
        split=_split_config(group_key=None),
    )

    with pytest.raises(SplitContractError, match="group_key is required"):
        validate_split_preflight(args_data)


def test_dataset_import_failure_is_fatal() -> None:
    with pytest.raises(DatasetResolutionError, match="exactly one dataset module"):
        resolve_dataset_class("DG", "does_not_exist")


def test_dataset_import_resolves_repository_case_without_fallback() -> None:
    dataset_class = resolve_dataset_class("DG", "classification")

    assert dataset_class.__module__.endswith("DG.Classification_dataset")


def test_default_dataset_requires_the_explicit_default_task_identity() -> None:
    assert resolve_dataset_class("Default_task", "Default_task") is Default_dataset
    with pytest.raises(DatasetResolutionError, match="dataset task directory"):
        resolve_dataset_class("Default_task", "classification")


def test_dataset_internal_import_error_preserves_its_cause(monkeypatch) -> None:
    dependency_error = ModuleNotFoundError("configured dependency is missing")

    def fail_import(_module_name: str):
        raise dependency_error

    monkeypatch.setattr(data_factory_module.importlib, "import_module", fail_import)
    with pytest.raises(DatasetResolutionError, match="Failed to import") as captured:
        resolve_dataset_class("DG", "classification")
    assert captured.value.__cause__ is dependency_error


def test_val_mode_cannot_bypass_split() -> None:
    data = {"record-1": np.arange(8, dtype=np.float32).reshape(8, 1)}
    metadata = {"record-1": {"Label": 1}}
    args_data = SimpleNamespace(
        window_size=2,
        stride=2,
        train_ratio=0.5,
        num_window=4,
        window_sampling_strategy="sequential",
        dtype="float32",
        normalization="none",
        noise_snr=None,
    )
    args_task = SimpleNamespace()

    train = Default_dataset(data, metadata, args_data, args_task, mode="train")
    validation = Default_dataset(data, metadata, args_data, args_task, mode="val")

    assert validation.mode == "valid"
    assert len(train) == len(validation) == 2
    train_windows = {window.tobytes() for window in train.processed_data}
    validation_windows = {window.tobytes() for window in validation.processed_data}
    assert train_windows.isdisjoint(validation_windows)
