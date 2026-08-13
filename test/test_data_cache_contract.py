from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from src.data_factory import ExplicitDataFactory, build_data
from src.data_factory.contracts import (
    format_loader_summary,
    require_nonempty_dataloaders,
)
from src.data_factory.data_factory import data_factory as BaseDataFactory
from src.data_factory.reader.CSV_Signal import read as read_csv_signal


def _factory(metadata):
    factory = ExplicitDataFactory.__new__(ExplicitDataFactory)
    factory.metadata = metadata
    return factory


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=str(tmp_path),
        metadata_file="metadata.csv",
    )


def test_atomic_cache_implementation_is_owned_by_base_factory() -> None:
    assert "_update_name_cache" not in ExplicitDataFactory.__dict__
    assert "_build_final_cache" not in ExplicitDataFactory.__dict__
    assert ExplicitDataFactory._update_name_cache is BaseDataFactory._update_name_cache
    assert ExplicitDataFactory._build_final_cache is BaseDataFactory._build_final_cache


def test_base_factory_rejects_unknown_dataset_adapter() -> None:
    factory = BaseDataFactory.__new__(BaseDataFactory)
    factory.args_task = SimpleNamespace(type="unknown", name="unknown")

    with pytest.raises(ValueError, match="No dataset adapter is registered"):
        factory._init_dataset()


def _write_h5(path: Path, values: dict[str, float]) -> None:
    with h5py.File(path, "w") as h5_file:
        for key, value in values.items():
            h5_file.create_dataset(
                key,
                data=np.full((2, 1), value, dtype=np.float32),
            )


def _keys(path: Path) -> set[str]:
    with h5py.File(path, "r") as h5_file:
        return set(h5_file.keys())


class _Loader:
    def __init__(self, count: int) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count


class _LoaderFactory:
    def __init__(self, **counts: int) -> None:
        self.loaders = {
            split: _Loader(count) for split, count in counts.items()
        }

    def get_dataloader(self, split: str):
        return self.loaders[split]


def test_failed_reader_does_not_publish_partial_dataset_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = {
        1: {"Name": "Demo", "File": "one.csv"},
        2: {"Name": "Demo", "File": "two.csv"},
    }
    factory = _factory(metadata)
    cache_path = tmp_path / "Demo.h5"
    _write_h5(cache_path, {"stable": 9.0})

    def read_one(file_id, meta, args_data):
        if file_id == 1:
            return file_id, np.ones((4, 1), dtype=np.float32), None
        return file_id, None, "raw file is unreadable"

    monkeypatch.setattr(factory, "_read_single_data", read_one)

    with pytest.raises(
        RuntimeError,
        match="Cannot publish cache.*ID 2: raw file is unreadable",
    ):
        factory._update_name_cache("Demo", [1, 2], _args(tmp_path), 1)

    assert _keys(cache_path) == {"stable"}
    assert not (tmp_path / ".Demo.h5.tmp").exists()


def test_successful_dataset_cache_replaces_only_after_all_ids_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = {
        1: {"Name": "Demo", "File": "one.csv"},
        2: {"Name": "Demo", "File": "two.csv"},
    }
    factory = _factory(metadata)

    monkeypatch.setattr(
        factory,
        "_read_single_data",
        lambda file_id, meta, args_data: (
            file_id,
            np.full((4, 1), file_id, dtype=np.float32),
            None,
        ),
    )

    factory._update_name_cache("Demo", [1, 2], _args(tmp_path), 1)

    cache_path = tmp_path / "Demo.h5"
    assert _keys(cache_path) == {"1", "2"}
    assert not (tmp_path / ".Demo.h5.tmp").exists()


def test_generated_cache_uses_explicit_cache_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_directory = tmp_path / "raw-data"
    cache_directory = tmp_path / "runtime-cache"
    metadata = {1: {"Name": "Demo", "File": "one.csv"}}
    factory = _factory(metadata)
    args = _args(raw_directory)
    args.cache_dir = str(cache_directory)
    monkeypatch.setattr(
        factory,
        "_read_single_data",
        lambda file_id, meta, args_data: (
            file_id,
            np.ones((4, 1), dtype=np.float32),
            None,
        ),
    )

    factory._update_name_cache("Demo", [1], args, 1)

    assert _keys(cache_directory / "Demo.h5") == {"1"}
    assert not (raw_directory / "Demo.h5").exists()


def test_complete_published_cache_is_reused_without_source_cache(
    tmp_path: Path,
) -> None:
    metadata = {
        1: {"Name": "Demo", "File": "one.csv"},
        2: {"Name": "Demo", "File": "two.csv"},
    }
    factory = _factory(metadata)
    final_cache = tmp_path / "cache.h5"
    _write_h5(final_cache, {"1": 1.0, "2": 2.0, "extra": 3.0})

    result = factory._build_final_cache(
        metadata,
        _args(tmp_path),
        use_cache=True,
    )

    assert result == str(final_cache)
    assert _keys(final_cache) == {"1", "2", "extra"}
    assert not (tmp_path / "Demo.h5").exists()


def test_incomplete_final_cache_keeps_previous_published_file(tmp_path: Path) -> None:
    metadata = {
        1: {"Name": "Demo", "File": "one.csv"},
        2: {"Name": "Demo", "File": "two.csv"},
    }
    factory = _factory(metadata)
    _write_h5(tmp_path / "Demo.h5", {"1": 1.0})
    final_cache = tmp_path / "cache.h5"
    _write_h5(final_cache, {"stable": 9.0})

    with pytest.raises(
        RuntimeError,
        match="Cannot publish cache.h5.*ID 2",
    ):
        factory._build_final_cache(metadata, _args(tmp_path), use_cache=True)

    assert _keys(final_cache) == {"stable"}
    assert not (tmp_path / ".cache.h5.tmp").exists()


def test_complete_final_cache_atomically_replaces_previous_file(tmp_path: Path) -> None:
    metadata = {
        1: {"Name": "Demo", "File": "one.csv"},
        2: {"Name": "Demo", "File": "two.csv"},
    }
    factory = _factory(metadata)
    _write_h5(tmp_path / "Demo.h5", {"1": 1.0, "2": 2.0})
    final_cache = tmp_path / "cache.h5"
    _write_h5(final_cache, {"stale": 9.0})

    result = factory._build_final_cache(
        metadata,
        _args(tmp_path),
        use_cache=False,
    )

    assert result == str(final_cache)
    assert _keys(final_cache) == {"1", "2"}
    assert not (tmp_path / ".cache.h5.tmp").exists()


def test_empty_task_selection_fails_before_cache_publication(
    tmp_path: Path,
) -> None:
    factory = _factory({})

    with pytest.raises(ValueError, match="contains no data IDs"):
        factory._build_final_cache({}, _args(tmp_path), use_cache=True)

    assert not (tmp_path / "cache.h5").exists()


def test_nonempty_loader_contract_returns_user_readable_counts() -> None:
    factory = _LoaderFactory(train=3, val=1, test=2)
    args_task = SimpleNamespace(type="DG", name="classification")
    args_data = SimpleNamespace(batch_size=8)

    counts = require_nonempty_dataloaders(factory, args_task, args_data)

    assert counts == {"train": 3, "val": 1, "test": 2}
    assert format_loader_summary(counts) == (
        "train=3 batches, val=1 batch, test=2 batches"
    )


def test_zero_batch_loader_fails_with_actionable_configuration_fields() -> None:
    factory = _LoaderFactory(train=3, val=0, test=2)
    args_task = SimpleNamespace(type="FS", name="classification")
    args_data = SimpleNamespace(batch_size=256)

    with pytest.raises(
        RuntimeError,
        match=(
            "FS/classification: val loader has 0 batches.*"
            "window_size.*num_window.*batch_size=256"
        ),
    ):
        require_nonempty_dataloaders(factory, args_task, args_data)


def test_explicit_factory_checks_loaders_after_base_construction(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_base_init(self, args_data, args_task):
        self.train_loader = _Loader(2)
        self.val_loader = _Loader(1)
        self.test_loader = _Loader(1)

    monkeypatch.setattr(BaseDataFactory, "__init__", fake_base_init)
    args_data = SimpleNamespace(batch_size=4)
    args_task = SimpleNamespace(type="DG", name="classification")

    ExplicitDataFactory(args_data, args_task)

    output = capsys.readouterr().out
    assert "train=2 batches, val=1 batch, test=1 batch" in output


def _write_signal_csv(path: Path, offset: float) -> None:
    rows = ["time,sensor_a,sensor_b"]
    for index in range(16):
        rows.append(f"{index},{offset + index},{offset + 2 * index}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_csv_signal_reader_requires_explicit_numeric_columns(tmp_path: Path) -> None:
    signal_path = tmp_path / "signal.csv"
    _write_signal_csv(signal_path, 0.0)
    args_data = SimpleNamespace(
        csv_signal_columns=["sensor_a", "sensor_b"],
        csv_delimiter=",",
        dtype="float32",
    )

    signal = read_csv_signal(signal_path, args_data)
    assert signal.shape == (16, 2)
    assert signal.dtype == np.float32
    assert np.array_equal(signal[0], np.array([0.0, 0.0], dtype=np.float32))

    args_data.csv_signal_columns = ["sensor_a", "missing"]
    with pytest.raises(ValueError, match="missing configured column"):
        read_csv_signal(signal_path, args_data)


def test_compatible_csv_dataset_uses_existing_data_and_task_contracts(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw" / "CSV_Signal"
    raw_dir.mkdir(parents=True)
    _write_signal_csv(raw_dir / "source.csv", 0.0)
    _write_signal_csv(raw_dir / "target.csv", 100.0)
    (tmp_path / "metadata.csv").write_text(
        "Id,Name,File,Dataset_id,Domain_id,Label,Sample_Rate\n"
        "1,CSV_Signal,source.csv,0,0,0,1000\n"
        "2,CSV_Signal,target.csv,0,1,0,1000\n",
        encoding="utf-8",
    )

    args_data = SimpleNamespace(
        factory_name="default",
        data_dir=str(tmp_path),
        metadata_file="metadata.csv",
        batch_size=1,
        num_workers=0,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        unused_ratio=0.0,
        normalization="none",
        window_size=4,
        window_sampling_strategy="evenly_spaced",
        num_window=4,
        window_sampling_seed=0,
        dtype="float32",
        pin_memory=False,
        csv_signal_columns=["sensor_a", "sensor_b"],
    )
    args_task = SimpleNamespace(
        type="DG",
        name="classification",
        target_system_id=[0],
        source_domain_id=[0],
        target_domain_id=[1],
    )

    factory = build_data(args_data, args_task)
    try:
        train_batch = next(iter(factory.train_loader))
        val_batch = next(iter(factory.val_loader))
        test_batch = next(iter(factory.test_loader))

        assert train_batch["x"].shape == (1, 4, 2)
        assert val_batch["x"].shape == (1, 4, 2)
        assert test_batch["x"].shape == (1, 4, 2)
        assert train_batch["file_id"].tolist() == [1]
        assert test_batch["file_id"].tolist() == [2]
        assert factory.split_summary["file_overlap"]["train_test"] == []
    finally:
        factory.data.close()
