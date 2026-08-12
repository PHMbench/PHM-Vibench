from __future__ import annotations

import io
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from src.data_factory.H5DataDict import H5DataDict
from src.data_factory.data_factory import data_factory
from src.data_factory.data_utils import MetadataAccessor
from src.data_factory.protocol_cache import build_cache_manifest


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Id": 1,
                "Dataset_id": 1,
                "Name": "RM_001_CWRU",
                "File": "one.mat",
                "Label": 0,
            },
            {
                "Id": 2,
                "Dataset_id": 2,
                "Name": "RM_002_XJTU",
                "File": "35Hz12kN/Bearing1_1/one.csv",
                "Label": 1,
            },
            {
                "Id": 3,
                "Dataset_id": 2,
                "Name": "RM_002_XJTU",
                "File": "40Hz10kN/Bearing3_1/two.csv",
                "Label": 0,
            },
        ]
    )


def _write_fixture(root: Path) -> tuple[Path, Path, dict[int, np.ndarray]]:
    metadata_path = root / "metadata.csv"
    _metadata().to_csv(metadata_path, index=False)
    arrays = {
        1: np.arange(24, dtype=np.float64).reshape(12, 2, 1),
        2: (np.arange(32, dtype=np.float64) + 10).reshape(16, 2, 1),
        3: (np.arange(40, dtype=np.float64) - 5).reshape(20, 2, 1),
    }
    cache_path = root / "cache.h5"
    with h5py.File(cache_path, "w") as handle:
        for sample_id, array in arrays.items():
            handle.create_dataset(str(sample_id), data=array)
        handle.create_dataset("999", data=np.zeros((4, 2, 1), dtype=np.float64))
    return metadata_path, cache_path, arrays


def _manifest(root: Path) -> tuple[Path, Path, Path, dict[int, np.ndarray]]:
    metadata_path, cache_path, arrays = _write_fixture(root)
    manifest_path = root / "cache.manifest.json"
    build_cache_manifest(
        cache_path=cache_path,
        metadata_path=metadata_path,
        output_path=manifest_path,
        chunk_rows=3,
        progress_every=1,
        progress_stream=io.StringIO(),
    )
    return metadata_path, cache_path, manifest_path, arrays


def _accessor(frame: pd.DataFrame) -> MetadataAccessor:
    return MetadataAccessor(frame.copy(), key_column="Id")


def test_manifest_builder_is_streamed_read_only_create_only_and_idempotent(
    tmp_path: Path,
) -> None:
    metadata_path, cache_path, arrays = _write_fixture(tmp_path)
    manifest_path = tmp_path / "cache.manifest.json"
    cache_before = cache_path.read_bytes()
    cache_mtime = cache_path.stat().st_mtime_ns
    progress = io.StringIO()

    first = build_cache_manifest(
        cache_path=cache_path,
        metadata_path=metadata_path,
        output_path=manifest_path,
        chunk_rows=3,
        progress_every=1,
        progress_stream=progress,
    )
    manifest_before = manifest_path.read_bytes()
    manifest_mtime = manifest_path.stat().st_mtime_ns
    second = build_cache_manifest(
        cache_path=cache_path,
        metadata_path=metadata_path,
        output_path=manifest_path,
        chunk_rows=3,
        progress_every=1,
        progress_stream=io.StringIO(),
    )

    assert cache_path.read_bytes() == cache_before
    assert cache_path.stat().st_mtime_ns == cache_mtime
    assert first["status"] == "created"
    assert second["status"] == "reused_identical"
    assert manifest_path.read_bytes() == manifest_before
    assert manifest_path.stat().st_mtime_ns == manifest_mtime
    assert "hashed 3/3" in progress.getvalue()

    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert value["kind"] == "p05_verified_signal_cache"
    assert value["cache"]["open_mode"] == "r"
    assert value["cache"]["swmr"] is True
    assert [entry["Id"] for entry in value["entries"]] == [1, 2, 3]
    assert value["entries"][0]["shape"] == list(arrays[1].shape)
    assert value["entries"][0]["dtype"] == "float64"
    assert value["entries"][0]["channel_order"] == ["DE_time", "FE_time"]

    manifest_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_cache_manifest(
            cache_path=cache_path,
            metadata_path=metadata_path,
            output_path=manifest_path,
            chunk_rows=3,
            progress_stream=io.StringIO(),
        )
    assert manifest_path.read_text(encoding="utf-8") == "tampered\n"


def test_verified_accessor_enforces_allowlist_and_manifest_bindings(tmp_path: Path) -> None:
    metadata_path, cache_path, manifest_path, arrays = _manifest(tmp_path)
    active_frame = pd.read_csv(metadata_path).loc[lambda frame: frame["Id"].isin([1, 2])]
    data = H5DataDict(
        cache_path,
        allowed_ids=[1, 2],
        manifest_path=manifest_path,
        metadata=_accessor(active_frame),
    )
    try:
        assert data.keys() == {"1", "2"}
        assert np.array_equal(data[1], arrays[1])
        assert np.array_equal(data["2"], arrays[2])
        assert 3 not in data
        assert 999 not in data
        with pytest.raises(KeyError, match="outside the active metadata allowlist"):
            _ = data[3]
        with pytest.raises(KeyError, match="outside the active metadata allowlist"):
            _ = data[999]
    finally:
        data.close()

    mismatched = active_frame.copy()
    mismatched.loc[mismatched["Id"] == 2, "File"] = "different.csv"
    with pytest.raises(ValueError, match="metadata mismatch"):
        H5DataDict(
            cache_path,
            allowed_ids=[1, 2],
            manifest_path=manifest_path,
            metadata=_accessor(mismatched),
        )


@pytest.mark.parametrize("drift", ["content", "shape", "dtype", "missing"])
def test_verified_accessor_rejects_cache_drift(tmp_path: Path, drift: str) -> None:
    metadata_path, cache_path, manifest_path, _ = _manifest(tmp_path)
    active = _accessor(pd.read_csv(metadata_path).loc[lambda frame: frame["Id"] == 1])
    with h5py.File(cache_path, "a") as handle:
        if drift == "content":
            handle["1"][0, 0, 0] += 1.0
        else:
            del handle["1"]
            if drift == "shape":
                handle.create_dataset("1", data=np.zeros((13, 2, 1), dtype=np.float64))
            elif drift == "dtype":
                handle.create_dataset("1", data=np.zeros((12, 2, 1), dtype=np.float32))

    data = H5DataDict(
        cache_path,
        allowed_ids=[1],
        manifest_path=manifest_path,
        metadata=active,
    )
    try:
        if drift == "missing":
            with pytest.raises(KeyError, match="missing active metadata IDs"):
                _ = data[1]
        elif drift == "content":
            with pytest.raises(ValueError, match="content SHA-256 mismatch"):
                _ = data[1]
        elif drift == "shape":
            with pytest.raises(ValueError, match="shape mismatch"):
                _ = data[1]
        else:
            with pytest.raises(ValueError, match="dtype mismatch"):
                _ = data[1]
    finally:
        data.close()


def test_data_factory_verified_mode_bypasses_download_and_limits_active_ids(
    tmp_path: Path, monkeypatch
) -> None:
    metadata_path, cache_path, manifest_path, arrays = _manifest(tmp_path)
    data_factory_module = importlib.import_module("src.data_factory.data_factory")
    monkeypatch.setattr(
        data_factory_module,
        "download_data",
        lambda *args, **kwargs: pytest.fail("download_data must not run"),
    )
    args = SimpleNamespace(
        cache_mode="read_only_verified",
        allow_download=False,
        metadata_path=str(metadata_path),
        cache_path=str(cache_path),
        cache_manifest_path=str(manifest_path),
    )
    factory = data_factory.__new__(data_factory)
    metadata = factory._init_metadata(args)
    factory.active_metadata = data_factory._metadata_for_ids(metadata, [1, 2])
    data = factory._init_data(args)
    try:
        assert data.keys() == {"1", "2"}
        assert np.array_equal(data[1], arrays[1])
        with pytest.raises(KeyError, match="outside the active metadata allowlist"):
            _ = data[3]
    finally:
        data.close()

    bad_args = SimpleNamespace(
        cache_mode="read_only_verified",
        allow_download=True,
        metadata_path=str(metadata_path),
    )
    with pytest.raises(ValueError, match="allow_download=false"):
        factory._init_metadata(bad_args)


def test_manifest_builder_rejects_missing_key_shape_and_dtype(tmp_path: Path) -> None:
    metadata_path, cache_path, _ = _write_fixture(tmp_path)
    with h5py.File(cache_path, "a") as handle:
        del handle["1"]
    with pytest.raises(KeyError, match="missing metadata Id 1"):
        build_cache_manifest(
            cache_path=cache_path,
            metadata_path=metadata_path,
            output_path=tmp_path / "missing.json",
            progress_stream=io.StringIO(),
        )

    # Restore Id 1 with a wrong shape; the builder must fail before creating a
    # manifest rather than blessing the drift.
    with h5py.File(cache_path, "a") as handle:
        handle.create_dataset("1", data=np.zeros((12, 1, 1), dtype=np.float64))
    with pytest.raises(ValueError, match="shape .*L,2,1"):
        build_cache_manifest(
            cache_path=cache_path,
            metadata_path=metadata_path,
            output_path=tmp_path / "shape.json",
            progress_stream=io.StringIO(),
        )
    assert not (tmp_path / "missing.json").exists()
    assert not (tmp_path / "shape.json").exists()
