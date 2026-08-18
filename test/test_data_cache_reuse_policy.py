from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from src.data_factory import build_data
from src.data_factory.explicit_data_factory import ExplicitDataFactory


def _write_signal_csv(path: Path, offset: float) -> None:
    rows = ["time,sensor_a,sensor_b"]
    for index in range(16):
        rows.append(f"{index},{offset + index},{offset + 2 * index}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_csv_fixture(root: Path) -> None:
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


def _data_args(root: Path, *, use_cache: bool | None = None) -> SimpleNamespace:
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


def _task_args() -> SimpleNamespace:
    return SimpleNamespace(
        type="DG",
        name="classification",
        target_system_id=[0],
        source_domain_id=[0],
        target_domain_id=[1],
    )


def _cached_signal(args_data: SimpleNamespace, args_task: SimpleNamespace) -> np.ndarray:
    factory = build_data(args_data, args_task)
    try:
        return np.asarray(factory.data[1]).copy()
    finally:
        factory.data.close()


def test_public_factory_reloads_current_raw_signal_by_default(tmp_path: Path) -> None:
    _write_csv_fixture(tmp_path)
    args_task = _task_args()
    args_data = _data_args(tmp_path)

    first = _cached_signal(args_data, args_task)

    _write_signal_csv(
        tmp_path / "raw" / "CSV_Signal" / "source.csv",
        500.0,
    )
    second = _cached_signal(args_data, args_task)

    assert not np.array_equal(first, second)
    assert float(second.reshape(second.shape[0], -1)[0, 0]) == 500.0


def test_cache_reuse_requires_explicit_boolean_opt_in(tmp_path: Path) -> None:
    _write_csv_fixture(tmp_path)
    args_task = _task_args()
    fresh_args = _data_args(tmp_path)

    _write_signal_csv(
        tmp_path / "raw" / "CSV_Signal" / "source.csv",
        500.0,
    )
    cached = _cached_signal(fresh_args, args_task)

    _write_signal_csv(
        tmp_path / "raw" / "CSV_Signal" / "source.csv",
        900.0,
    )
    reused = _cached_signal(
        _data_args(tmp_path, use_cache=True),
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
    factory = ExplicitDataFactory.__new__(ExplicitDataFactory)
    factory.metadata = metadata
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
