"""Source-grounded, provider-free MAT reader contracts; no real dataset files."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.io import savemat

from src.data_factory.reader import RM_025_KAIST, RM_026_HUST23


def test_hust23_reads_only_steady_state_data_not_shaft_fs(tmp_path):
    path = tmp_path / "I402.mat"
    savemat(path, {"data": np.array([[1.5, -2.0, 3.0]]), "fs": np.array([[42.0]])})
    actual = RM_026_HUST23.read(path)
    assert actual.shape == (3, 1) and actual.dtype == np.float64
    np.testing.assert_array_equal(actual[:, 0], [1.5, -2.0, 3.0])


def test_hust23_does_not_implicitly_use_runup_or_rpm(tmp_path):
    path = tmp_path / "runup.mat"
    savemat(path, {"ru": np.arange(5.0), "rpm": np.arange(5.0)})
    with pytest.raises(KeyError, match="no steady-state 'data'"):
        RM_026_HUST23.read(path)


def test_hust23_rejects_nonfinite_and_multichannel_records(tmp_path):
    path = tmp_path / "bad.mat"
    savemat(path, {"data": np.array([[1.0, np.nan]])})
    with pytest.raises(FloatingPointError, match="NaN"):
        RM_026_HUST23.read(path)
    savemat(path, {"data": np.arange(12.0).reshape(3, 4)})
    with pytest.raises(ValueError, match="single signal"):
        RM_026_HUST23.read(path)


def test_kaist_vibration_struct_decodes_four_axes_only(tmp_path, monkeypatch):
    path = tmp_path / "0Nm_Normal.mat"
    path.touch()
    matrix = np.arange(32.0).reshape(8, 4)
    # Matches independent digester indexing: Signal[0][0][1][0][0][0]
    nested = [[[None, [[[matrix]]]]]]
    monkeypatch.setattr(RM_025_KAIST, "loadmat", lambda unused: {"Signal": nested})
    actual = RM_025_KAIST.read(path)
    np.testing.assert_array_equal(actual, matrix)
    assert actual.shape == (8, 4) and actual.dtype == np.float64


def test_kaist_rejects_acoustic_and_tdms_as_default_vibration(tmp_path, monkeypatch):
    path = tmp_path / "0Nm_Normal.mat"
    path.touch()
    nested = [[[None, [[[np.zeros((8, 1))]]]]]]
    monkeypatch.setattr(RM_025_KAIST, "loadmat", lambda unused: {"Signal": nested})
    with pytest.raises(ValueError, match="four vibration channels"):
        RM_025_KAIST.read(path)
    tdms = tmp_path / "other.tdms"
    tdms.touch()
    with pytest.raises(ValueError, match="only constant-load"):
        RM_025_KAIST.read(tdms)
