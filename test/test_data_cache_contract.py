from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import src.data_factory.reader.CSV_Signal as csv_signal_reader
from src.data_factory import ExplicitDataFactory, build_data
from src.data_factory.contracts import (
    format_loader_summary,
    require_nonempty_dataloaders,
)
from src.data_factory.data_factory import data_factory as BaseDataFactory
from src.data_factory.reader.CSV_Signal import read as read_csv_signal
from src.data_factory.reader.Dummy_Data import read as read_dummy_signal


data_factory_module = importlib.import_module("src.data_factory.data_factory")


def _factory(metadata):
    factory = ExplicitDataFactory.__new__(ExplicitDataFactory)
    factory.metadata = metadata
    return factory


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=str(tmp_path),
        metadata_file="metadata.csv",
    )


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


def _write_dummy_csv(path: Path, offset: float = 0.0, samples: int = 16) -> None:
    rows = ["index,ch1,ch2"]
    for index in range(samples):
        rows.append(f"{index},{offset + index},{offset + 2 * index}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_dummy_reader_consumes_exact_declared_columns(tmp_path: Path) -> None:
    path = tmp_path / "dummy.csv"
    path.write_text(
        "index,ch1,ch2\n"
        "0,0.0,1.0\n"
        "1,0.5,0.5\n"
        "2,1.0,0.0\n",
        encoding="utf-8",
    )

    signal = read_dummy_signal(path)

    assert signal.shape == (3, 2)
    assert signal.dtype == np.float32
    assert np.array_equal(
        signal,
        np.array(
            [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]],
            dtype=np.float32,
        ),
    )


def test_dummy_reader_rejects_missing_file_without_synthetic_fallback(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="does not synthesize fallback signals"):
        read_dummy_signal(tmp_path / "missing.csv")


def test_dummy_reader_rejects_malformed_or_nonfinite_channels(tmp_path: Path) -> None:
    missing_column = tmp_path / "missing_column.csv"
    missing_column.write_text("index,ch1\n0,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        read_dummy_signal(missing_column)

    non_numeric = tmp_path / "non_numeric.csv"
    non_numeric.write_text(
        "index,ch1,ch2\n0,not-a-number,1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must contain only numeric"):
        read_dummy_signal(non_numeric)

    non_finite = tmp_path / "non_finite.csv"
    non_finite.write_text("index,ch1,ch2\n0,NaN,1.0\n", encoding="utf-8")
    with pytest.raises(FloatingPointError, match="contains NaN or Inf"):
        read_dummy_signal(non_finite)


def _cache_policy_args(
    root: Path,
    *,
    use_cache: bool | None = None,
) -> SimpleNamespace:
    values = {
        "factory_name": "default",
        "data_dir": str(root),
        "metadata_file": "metadata.csv",
        "batch_size": 1,
        "num_workers": 0,
        "train_ratio": 0.5,
        "val_ratio": 0.25,
        "test_ratio": 0.25,
        "unused_ratio": 0.0,
        "normalization": "none",
        "window_size": 4,
        "window_sampling_strategy": "evenly_spaced",
        "num_window": 4,
        "window_sampling_seed": 0,
        "dtype": "float32",
        "pin_memory": False,
        "csv_signal_columns": ["sensor_a", "sensor_b"],
    }
    if use_cache is not None:
        values["use_cache"] = use_cache
    return SimpleNamespace(**values)


def _cache_policy_task() -> SimpleNamespace:
    return SimpleNamespace(
        type="DG",
        name="classification",
        target_system_id=[0],
        source_domain_id=[0],
        target_domain_id=[1],
    )


def _write_cache_policy_fixture(root: Path) -> None:
    raw_dir = root / "raw" / "CSV_Signal"
    raw_dir.mkdir(parents=True)
    _write_signal_csv(raw_dir / "source.csv", 0.0)
    _write_signal_csv(raw_dir / "target.csv", 100.0)
    (root / "metadata.csv").write_text(
        "Id,Name,File,Dataset_id,Domain_id,Label,Sample_Rate\n"
        "1,CSV_Signal,source.csv,0,0,0,1000\n"
        "2,CSV_Signal,target.csv,0,1,0,1000\n",
        encoding="utf-8",
    )


def _cached_signal(args_data: SimpleNamespace, args_task: SimpleNamespace) -> np.ndarray:
    factory = build_data(args_data, args_task)
    try:
        return np.asarray(factory.data[1]).copy()
    finally:
        factory.data.close()


def test_public_factory_reloads_current_raw_signal_by_default(tmp_path: Path) -> None:
    _write_cache_policy_fixture(tmp_path)
    args_task = _cache_policy_task()
    args_data = _cache_policy_args(tmp_path)

    first = _cached_signal(args_data, args_task)
    _write_signal_csv(
        tmp_path / "raw" / "CSV_Signal" / "source.csv",
        500.0,
    )
    second = _cached_signal(args_data, args_task)

    assert not np.array_equal(first, second)
    assert float(second.reshape(second.shape[0], -1)[0, 0]) == 500.0


def test_cache_reuse_requires_explicit_boolean_opt_in(tmp_path: Path) -> None:
    _write_cache_policy_fixture(tmp_path)
    args_task = _cache_policy_task()

    _write_signal_csv(
        tmp_path / "raw" / "CSV_Signal" / "source.csv",
        500.0,
    )
    cached = _cached_signal(_cache_policy_args(tmp_path), args_task)

    _write_signal_csv(
        tmp_path / "raw" / "CSV_Signal" / "source.csv",
        900.0,
    )
    reused = _cached_signal(
        _cache_policy_args(tmp_path, use_cache=True),
        args_task,
    )

    assert np.array_equal(reused, cached)
    assert float(reused.reshape(reused.shape[0], -1)[0, 0]) == 500.0

    invalid = ExplicitDataFactory.__new__(ExplicitDataFactory)
    with pytest.raises(TypeError, match="data.use_cache must be a boolean"):
        invalid._init_data(
            SimpleNamespace(
                data_dir=str(tmp_path),
                metadata_file="metadata.csv",
                use_cache="true",
            )
        )


def test_cache_dir_owns_all_derived_hdf5_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw-root"
    cache_root = tmp_path / "derived-cache"
    raw_root.mkdir()
    args_data = SimpleNamespace(
        data_dir=str(raw_root),
        metadata_file="metadata.csv",
        cache_dir=str(cache_root),
    )
    metadata = {
        1: {"Name": "Demo", "File": "one.csv"},
        2: {"Name": "Demo", "File": "two.csv"},
    }
    factory = _factory(metadata)
    monkeypatch.setattr(
        factory,
        "_read_single_data",
        lambda file_id, meta, args: (
            file_id,
            np.full((4, 1), file_id, dtype=np.float32),
            None,
        ),
    )

    factory._update_name_cache("Demo", [1, 2], args_data, 1)
    final_path = factory._build_final_cache(
        metadata,
        args_data,
        use_cache=False,
    )

    assert Path(final_path) == cache_root / "cache.h5"
    assert (cache_root / "Demo.h5").is_file()
    assert (cache_root / "cache.h5").is_file()
    assert not (raw_root / "Demo.h5").exists()
    assert not (raw_root / "cache.h5").exists()

    with h5py.File(cache_root / "cache.h5", "r") as h5_file:
        assert set(h5_file.keys()) == {"1", "2"}


class _ReaderFailure(ValueError):
    pass


def test_explicit_reader_preserves_original_exception_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw" / "CSV_Signal"
    raw_dir.mkdir(parents=True)
    (raw_dir / "signal.csv").write_text(
        "time,sensor_a\n0,1.0\n",
        encoding="utf-8",
    )
    (tmp_path / "metadata.csv").write_text(
        "Id,Name,File\n1,CSV_Signal,signal.csv\n",
        encoding="utf-8",
    )
    factory = _factory(
        {1: {"Name": "CSV_Signal", "File": "signal.csv"}}
    )

    def fail_reader(path, args_data):
        raise _ReaderFailure("reader contract exploded")

    monkeypatch.setattr(csv_signal_reader, "read", fail_reader)

    with pytest.raises(_ReaderFailure, match="reader contract exploded"):
        factory._update_name_cache(
            "CSV_Signal",
            [1],
            _args(tmp_path),
            1,
        )

    assert not (tmp_path / "CSV_Signal.h5").exists()
    assert not (tmp_path / ".CSV_Signal.h5.tmp").exists()


def test_explicit_reader_raises_for_missing_declared_raw_file(tmp_path: Path) -> None:
    (tmp_path / "metadata.csv").write_text(
        "Id,Name,File\n1,CSV_Signal,missing.csv\n",
        encoding="utf-8",
    )
    factory = _factory(
        {1: {"Name": "CSV_Signal", "File": "missing.csv"}}
    )

    with pytest.raises(
        FileNotFoundError,
        match=r"Raw data file not found for ID 1: .*missing\.csv",
    ):
        factory._update_name_cache(
            "CSV_Signal",
            [1],
            _args(tmp_path),
            1,
        )

    assert not (tmp_path / "CSV_Signal.h5").exists()
    assert not (tmp_path / ".CSV_Signal.h5.tmp").exists()


def _reader_output_fixture(tmp_path: Path) -> tuple[ExplicitDataFactory, Path]:
    raw_dir = tmp_path / "raw" / "CSV_Signal"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "signal.csv"
    raw_path.write_text("placeholder\n", encoding="utf-8")
    return (
        _factory({1: {"Name": "CSV_Signal", "File": "signal.csv"}}),
        raw_path,
    )


@pytest.mark.parametrize(
    ("reader_value", "error_type", "message"),
    [
        (None, TypeError, "must return numpy.ndarray"),
        ([1.0, 2.0], TypeError, "must return numpy.ndarray"),
        (np.array(["bad"]), TypeError, "real numeric samples"),
        (
            np.array([[1.0 + 2.0j]], dtype=np.complex64),
            TypeError,
            "real numeric samples",
        ),
        (
            np.empty((0, 1), dtype=np.float32),
            ValueError,
            "returned empty shape",
        ),
        (
            np.ones((1, 1, 1, 1), dtype=np.float32),
            ValueError,
            "supported reader ranks",
        ),
        (
            np.array([[np.nan]], dtype=np.float32),
            FloatingPointError,
            "returned NaN or Inf",
        ),
        (
            np.array([[np.inf]], dtype=np.float32),
            FloatingPointError,
            "returned NaN or Inf",
        ),
    ],
)
def test_invalid_reader_outputs_fail_before_cache_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader_value,
    error_type: type[BaseException],
    message: str,
) -> None:
    factory, _ = _reader_output_fixture(tmp_path)
    monkeypatch.setattr(
        csv_signal_reader,
        "read",
        lambda path, args_data: reader_value,
    )

    with pytest.raises(error_type, match=message):
        factory._update_name_cache(
            "CSV_Signal",
            [1],
            _args(tmp_path),
            1,
        )

    assert not (tmp_path / "CSV_Signal.h5").exists()
    assert not (tmp_path / ".CSV_Signal.h5.tmp").exists()


@pytest.mark.parametrize(
    ("reader_value", "expected_shape"),
    [
        (np.arange(4, dtype=np.float32), (4,)),
        (np.ones((4, 2), dtype=np.float32), (4, 2, 1)),
        (np.ones((4, 2, 1), dtype=np.float32), (4, 2, 1)),
    ],
)
def test_valid_reader_ranks_preserve_existing_cache_representation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader_value: np.ndarray,
    expected_shape: tuple[int, ...],
) -> None:
    factory, raw_path = _reader_output_fixture(tmp_path)
    monkeypatch.setattr(
        csv_signal_reader,
        "read",
        lambda path, args_data: reader_value,
    )

    file_id, observed, error = factory._read_single_data(
        1,
        {"Name": "CSV_Signal", "File": "signal.csv"},
        _args(tmp_path),
    )

    assert file_id == 1
    assert error is None
    assert observed.shape == expected_shape
    assert observed.dtype == reader_value.dtype
    assert raw_path.is_file()


def test_missing_metadata_has_zero_remote_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_calls: list[tuple[tuple, dict]] = []

    def forbidden_remote(*args, **kwargs):
        remote_calls.append((args, kwargs))
        raise AssertionError("normal local runs must not call download_data")

    monkeypatch.setattr(data_factory_module, "download_data", forbidden_remote)
    factory = ExplicitDataFactory.__new__(ExplicitDataFactory)

    with pytest.raises(
        FileNotFoundError,
        match="Normal runs do not download replacement metadata",
    ):
        factory._init_metadata(
            SimpleNamespace(
                data_dir=str(tmp_path),
                metadata_file="missing.csv",
            )
        )

    assert remote_calls == []
    assert not list(tmp_path.glob("*.h5"))


def test_malformed_utf8_metadata_fails_before_cache_publication(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.csv"
    path.write_bytes(
        b"Id,Name,File,Dataset_id,Domain_id,Label,Sample_Rate\n"
        b"1,Dummy_Data,dummy.csv,0,0,0,1000\xff\n"
    )
    factory = ExplicitDataFactory.__new__(ExplicitDataFactory)

    with pytest.raises(UnicodeDecodeError):
        factory._init_metadata(_args(tmp_path))

    assert not list(tmp_path.glob("*.h5"))


def test_tab_separated_content_named_csv_fails_schema_validation(
    tmp_path: Path,
) -> None:
    (tmp_path / "metadata.csv").write_text(
        "Id\tName\tFile\tDataset_id\tDomain_id\tLabel\tSample_Rate\n"
        "1\tDummy_Data\tdummy.csv\t0\t0\t0\t1000\n",
        encoding="utf-8",
    )
    factory = ExplicitDataFactory.__new__(ExplicitDataFactory)

    with pytest.raises(ValueError, match="missing required column 'Id'"):
        factory._init_metadata(_args(tmp_path))

    assert not list(tmp_path.glob("*.h5"))


def test_explicit_tsv_metadata_is_supported(tmp_path: Path) -> None:
    (tmp_path / "metadata.tsv").write_text(
        "Id\tName\tFile\tDataset_id\tDomain_id\tLabel\tSample_Rate\n"
        "1\tDummy_Data\tdummy.csv\t0\t0\t0\t1000\n",
        encoding="utf-8",
    )
    factory = ExplicitDataFactory.__new__(ExplicitDataFactory)

    metadata = factory._init_metadata(
        SimpleNamespace(
            data_dir=str(tmp_path),
            metadata_file="metadata.tsv",
        )
    )

    assert metadata[1]["Name"] == "Dummy_Data"


def _write_tiny_local_dummy_fixture(root: Path) -> None:
    raw_dir = root / "raw" / "Dummy_Data"
    raw_dir.mkdir(parents=True)
    rows = [
        (1, "source_c0.csv", 0, 0),
        (2, "source_c1.csv", 0, 1),
        (3, "target_c0.csv", 1, 0),
        (4, "target_c1.csv", 1, 1),
    ]
    for file_id, file_name, domain_id, label in rows:
        _write_dummy_csv(
            raw_dir / file_name,
            offset=float(file_id * 100),
            samples=64,
        )
    (root / "metadata.csv").write_text(
        "Id,Name,File,Dataset_id,Domain_id,Label,Sample_Rate\n"
        "1,Dummy_Data,source_c0.csv,0,0,0,1000\n"
        "2,Dummy_Data,source_c1.csv,0,0,1,1000\n"
        "3,Dummy_Data,target_c0.csv,0,1,0,1000\n"
        "4,Dummy_Data,target_c1.csv,0,1,1,1000\n",
        encoding="utf-8",
    )


def test_tiny_local_dummy_builds_loaders_without_provider_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_tiny_local_dummy_fixture(tmp_path)
    remote_calls: list[tuple[tuple, dict]] = []

    def forbidden_remote(*args, **kwargs):
        remote_calls.append((args, kwargs))
        raise AssertionError("normal local runs must not call download_data")

    monkeypatch.setattr(data_factory_module, "download_data", forbidden_remote)
    before_modules = set(sys.modules)
    args_data = SimpleNamespace(
        factory_name="default",
        data_dir=str(tmp_path),
        metadata_file="metadata.csv",
        batch_size=2,
        num_workers=0,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        unused_ratio=0.0,
        normalization="none",
        window_size=8,
        window_sampling_strategy="evenly_spaced",
        num_window=4,
        window_sampling_seed=0,
        dtype="float32",
        pin_memory=False,
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
        assert len(factory.train_loader) > 0
        assert len(factory.val_loader) > 0
        assert len(factory.test_loader) > 0
        batch = next(iter(factory.train_loader))
        assert batch["x"].shape[-2:] == (8, 2)
    finally:
        factory.data.close()

    imported = set(sys.modules) - before_modules
    assert remote_calls == []
    assert "huggingface_hub" not in imported
    assert not any(name == "modelscope" or name.startswith("modelscope.") for name in imported)
