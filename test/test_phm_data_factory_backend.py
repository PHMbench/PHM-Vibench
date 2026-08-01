from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest

from src.config_schema.models import DataConfig
from src.configs.config_utils import build_experiment_name, load_config
from src.data_factory import build_data, build_data_repository
from src.data_factory import standalone
from src.data_factory.data_factory import data_factory


def _missing_module(name: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(f"No module named {name!r}", name=name)


def _backend_config(tmp_path: Path) -> Path:
    pd.DataFrame(
        [
            {
                "Id": 1,
                "Dataset_id": 7,
                "Name": "fixture",
                "Visiable": 1,
                "Label": 2,
                "Domain_id": 3,
                "Sample_lenth": 8,
                "Channel": 1,
            }
        ]
    ).to_csv(tmp_path / "metadata.csv", index=False)
    with h5py.File(tmp_path / "cache.h5", "w") as handle:
        handle.create_dataset("1", data=np.arange(8, dtype=np.float64)[:, None])
    config = tmp_path / "phm-data.yaml"
    config.write_text(
        "backend: local\n"
        f"metadata_path: {tmp_path / 'metadata.csv'}\n"
        f"signal_path: {tmp_path / 'cache.h5'}\n",
        encoding="utf-8",
    )
    return config


def _end_to_end_backend_config(tmp_path: Path) -> Path:
    pd.DataFrame(
        [
            {
                "Id": sample_id,
                "Dataset_id": 7,
                "Name": "fixture",
                "Visiable": 1,
                "Label": label,
                "Domain_id": domain,
                "Sample_lenth": 16,
                "Channel": 1,
            }
            for sample_id, label, domain in ((1, 0, 0), (2, 1, 1))
        ]
    ).to_csv(tmp_path / "metadata-e2e.csv", index=False)
    with h5py.File(tmp_path / "signals-e2e.h5", "w") as handle:
        handle.create_dataset("1", data=np.arange(16, dtype=np.float64)[:, None])
        handle.create_dataset("2", data=np.arange(16, 32, dtype=np.float64)[:, None])
    config = tmp_path / "phm-data-e2e.yaml"
    config.write_text(
        "backend: local\n"
        f"metadata_path: {tmp_path / 'metadata-e2e.csv'}\n"
        f"signal_path: {tmp_path / 'signals-e2e.h5'}\n",
        encoding="utf-8",
    )
    return config


def test_data_config_requires_factory_specific_fields() -> None:
    DataConfig(factory_name="phm_data", phm_data_config="backend.yaml")
    DataConfig(data_dir="data", metadata_file="metadata.csv")
    with pytest.raises(ValueError, match="phm_data_config"):
        DataConfig(factory_name="phm_data")
    with pytest.raises(ValueError, match="data.data_dir"):
        DataConfig()


def test_legacy_config_loader_applies_same_conditional_contract() -> None:
    common: dict[str, Any] = {
        "model": {"name": "model", "type": "MLP"},
        "task": {"name": "classification", "type": "DG"},
    }
    config = load_config(
        {
            **common,
            "data": {
                "factory_name": "phm_data",
                "phm_data_config": "config/provider.yaml",
            },
        }
    )
    assert config.data.phm_data_config == "config/provider.yaml"
    assert build_experiment_name(config).startswith("provider/")

    with pytest.raises(ValueError, match="phm_data_config"):
        load_config({**common, "data": {"factory_name": "phm_data"}})
    with pytest.raises(ValueError, match="data.data_dir"):
        load_config({**common, "data": {"factory_name": "default"}})


def test_missing_backend_is_explicit_and_does_not_change_sys_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = tuple(sys.path)

    def missing(_: str) -> Any:
        raise _missing_module("phm_data_factory")

    monkeypatch.setattr(standalone.importlib, "import_module", missing)
    with pytest.raises(ModuleNotFoundError, match="optional and is not installed"):
        build_data_repository(
            SimpleNamespace(factory_name="phm_data", phm_data_config="backend.yaml")
        )
    assert tuple(sys.path) == before


def test_provider_dependency_import_error_is_not_misreported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_dependency(_: str) -> Any:
        raise _missing_module("provider_dependency")

    monkeypatch.setattr(
        standalone.importlib, "import_module", missing_dependency
    )
    with pytest.raises(ModuleNotFoundError) as raised:
        build_data_repository(
            SimpleNamespace(factory_name="phm_data", phm_data_config="backend.yaml")
        )
    assert raised.value.name == "provider_dependency"


def test_wrong_provider_version_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SimpleNamespace(__version__="0.3.0", connect=lambda _: object())
    monkeypatch.setattr(
        standalone.importlib, "import_module", lambda _: provider
    )
    with pytest.raises(RuntimeError, match="requires phm-data-factory 0.2.0"):
        build_data_repository(
            SimpleNamespace(factory_name="phm_data", phm_data_config="backend.yaml")
        )


def test_registered_factory_uses_exact_provider_training_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = pytest.importorskip("phm_data_factory")
    assert provider.__version__ == "0.2.0"
    manifest = provider.agent_contract_manifest()
    assert manifest["api_schema_version"] == "1.0.0"
    assert manifest["capability_schema_version"] == "1.0.0"
    config = _backend_config(tmp_path)
    monkeypatch.setattr(
        data_factory, "_init_dataset", lambda self: ("train", "val", "test")
    )
    monkeypatch.setattr(
        data_factory, "_init_dataloader", lambda self: ("tl", "vl", "xl")
    )
    args_data = SimpleNamespace(
        factory_name="phm_data", phm_data_config=str(config), num_workers=0
    )

    factory = build_data(args_data, SimpleNamespace(name="Default", type="DG"))
    try:
        assert factory.metadata["1"]["Dataset_id"] == 7
        assert factory.metadata["1"]["Label"] == 2
        np.testing.assert_array_equal(factory.data["1"][:, 0], np.arange(8))
    finally:
        factory.close()
        factory.close()
    assert factory._closed is True


def test_provider_backend_reaches_existing_datasets_and_loaders(tmp_path: Path) -> None:
    provider = pytest.importorskip("phm_data_factory")
    assert provider.__version__ == "0.2.0"
    args_data = SimpleNamespace(
        factory_name="phm_data",
        phm_data_config=str(_end_to_end_backend_config(tmp_path)),
        num_workers=0,
        batch_size=1,
        window_size=4,
        stride=1,
        train_ratio=0.5,
        num_window=2,
        dtype="float32",
        normalization="none",
    )
    args_task = SimpleNamespace(
        name="classification",
        type="DG",
        target_system_id=[7],
        source_domain_id=[0],
        target_domain_id=[1],
    )

    with build_data(args_data, args_task) as factory:
        train_batch = next(iter(factory.train_loader))
        test_batch = next(iter(factory.test_loader))
        assert tuple(train_batch["x"].shape) == (1, 4, 1)
        assert tuple(test_batch["x"].shape) == (1, 4, 1)
        assert train_batch["y"].tolist() == [0]
        assert test_batch["y"].tolist() == [1]


def test_backend_config_is_required_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def unexpected_import(_: str) -> Any:
        nonlocal called
        called = True
        raise AssertionError("provider import must not run")

    monkeypatch.setattr(
        standalone.importlib, "import_module", unexpected_import
    )
    with pytest.raises(ValueError, match="phm_data_config"):
        build_data_repository(SimpleNamespace(factory_name="phm_data"))
    assert called is False
