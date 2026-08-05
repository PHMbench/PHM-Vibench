from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from src.data_factory import ExplicitDataFactory


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
        use_cache=True,
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
