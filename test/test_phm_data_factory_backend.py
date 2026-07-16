from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from src.config_schema.models import DataConfig
from src.data_factory import build_data, build_data_repository
from src.data_factory.data_factory import data_factory


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


def test_data_config_requires_fields_conditionally():
    DataConfig(factory_name="phm_data", phm_data_config="backend.yaml")
    DataConfig(data_dir="data", metadata_file="metadata.csv")
    with pytest.raises(ValueError, match="phm_data_config"):
        DataConfig(factory_name="phm_data")
    with pytest.raises(ValueError, match="data_dir"):
        DataConfig()


def test_registered_factory_uses_repository_and_preserves_training_metadata(
    tmp_path: Path, monkeypatch
):
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


def test_bridge_rejects_silent_backend_fallback():
    with pytest.raises(ValueError, match="phm_data_config"):
        build_data_repository(SimpleNamespace(factory_name="phm_data"))
