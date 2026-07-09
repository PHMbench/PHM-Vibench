from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import scipy.io
from scipy.signal import resample_poly

from src.data_factory.data_factory import (
    _cached_dataset_matches_contract,
    _normalize_reader_output,
    data_factory,
)
from src.data_factory.reader import RM_006_THU


def test_normalize_reader_output_keeps_cache_contract():
    one_dim = np.array([1.0, 2.0, 3.0])
    two_dim = np.array([[1.0, 2.0], [3.0, 4.0]])

    assert _normalize_reader_output(one_dim).shape == (3, 1)
    assert _normalize_reader_output(two_dim).shape == (2, 2)

    with pytest.raises(ValueError, match="must be 1D or 2D"):
        _normalize_reader_output(np.ones((2, 3, 1)))


def test_h5_cache_contract_rejects_stale_shapes(tmp_path):
    path = tmp_path / "cache.h5"
    with h5py.File(path, "w") as h5f:
        valid = h5f.create_dataset("valid", data=np.ones((8, 2), dtype=np.float32))
        stale_rank = h5f.create_dataset("rank3", data=np.ones((8, 2, 1), dtype=np.float32))
        stale_channel = h5f.create_dataset("channel", data=np.ones((8, 1), dtype=np.float32))

        meta = {"Name": "RM_006_THU", "Channel": 2, "Sample_lenth": 8}
        assert _cached_dataset_matches_contract(valid, meta)
        assert not _cached_dataset_matches_contract(stale_rank, meta)
        assert not _cached_dataset_matches_contract(stale_channel, meta)


def test_build_final_cache_rebuild_overwrites_existing_keys(tmp_path):
    args = SimpleNamespace(data_dir=str(tmp_path))
    name_cache = tmp_path / "Dummy_Data.h5"
    final_cache = tmp_path / "cache.h5"
    with h5py.File(name_cache, "w") as h5f:
        h5f.create_dataset("1", data=np.ones((3, 2), dtype=np.float32))
    with h5py.File(final_cache, "w") as h5f:
        h5f.create_dataset("1", data=np.ones((1, 1), dtype=np.float32))

    factory = data_factory.__new__(data_factory)
    factory.metadata = {"1": {"Name": "Dummy_Data", "Channel": 2, "Sample_lenth": 3}}

    factory._build_final_cache({"1": object()}, args, use_cache=False)

    with h5py.File(final_cache, "r") as h5f:
        assert h5f["1"].shape == (3, 2)


def test_update_name_cache_raises_reader_errors(tmp_path):
    factory = data_factory.__new__(data_factory)
    factory.metadata = {"1": {"Name": "RM_006_THU", "File": "missing.mat"}}

    def fail_read(id_key, meta, args_data):
        return id_key, None, "missing paired file"

    factory._read_single_data = fail_read

    with pytest.raises(RuntimeError, match="missing paired file"):
        factory._update_name_cache(
            "RM_006_THU",
            ["1"],
            SimpleNamespace(data_dir=str(tmp_path), metadata_file="metadata.xlsx"),
            max_workers=1,
        )


def test_rm006_thu_reader_pairs_vibration_and_downsampled_voltage(tmp_path):
    root = tmp_path / "raw" / "RM_006_THU"
    vibration_path = root / "vibration" / "health_bearing" / "1hz_1.mat"
    voltage_path = root / "voltage" / "health_bearing" / "1hz_1.txt"
    vibration_path.parent.mkdir(parents=True)
    voltage_path.parent.mkdir(parents=True)

    vibration = np.arange(64, dtype=np.float32)
    voltage = np.column_stack(
        [np.arange(128, dtype=np.float32), np.linspace(0.0, 1.0, 128, dtype=np.float32)]
    )
    scipy.io.savemat(vibration_path, {"hz_1": vibration.reshape(-1, 1)})
    np.savetxt(voltage_path, voltage)

    data = RM_006_THU.read(vibration_path, SimpleNamespace())

    assert data.shape == (64, 2)
    assert data.dtype == np.float32
    np.testing.assert_allclose(data[:, 0], vibration)
    expected_voltage = resample_poly(voltage[:, 1], up=1, down=2).astype(np.float32)
    np.testing.assert_allclose(data[:, 1], expected_voltage, rtol=1e-5, atol=1e-6)


def test_rm006_thu_reader_accepts_voltage_path(tmp_path):
    root = tmp_path / "raw" / "RM_006_THU"
    vibration_path = root / "vibration" / "inner_fault" / "10hz_1.mat"
    voltage_path = root / "voltage" / "inner_fault" / "10hz_1.txt"
    vibration_path.parent.mkdir(parents=True)
    voltage_path.parent.mkdir(parents=True)

    scipy.io.savemat(vibration_path, {"hz_1": np.ones((32, 1), dtype=np.float32)})
    np.savetxt(
        voltage_path,
        np.column_stack([np.arange(64, dtype=np.float32), np.ones(64, dtype=np.float32)]),
    )

    from_vibration = RM_006_THU.read(vibration_path, SimpleNamespace())
    from_voltage = RM_006_THU.read(voltage_path, SimpleNamespace())

    np.testing.assert_allclose(from_vibration, from_voltage)


def test_rm006_thu_reader_requires_paired_file(tmp_path):
    vibration_path = tmp_path / "raw" / "RM_006_THU" / "vibration" / "fault" / "1hz_1.mat"
    vibration_path.parent.mkdir(parents=True)
    scipy.io.savemat(vibration_path, {"hz_1": np.ones((16, 1), dtype=np.float32)})

    with pytest.raises(FileNotFoundError, match="voltage file not found"):
        RM_006_THU.read(vibration_path, SimpleNamespace())
