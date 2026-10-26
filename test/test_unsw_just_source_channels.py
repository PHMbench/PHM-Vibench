"""Source-specific channel order/versions from official UNSW and JUST releases."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from src.data_factory.reader import RM_008_UNSW, RM_024_JUST


def _unsw(path, *, old=False, drop=None):
    fields = {
        "Fs": [[51200.0]],
        "accH": np.arange(5.0),
        "accV": np.arange(5.0) + 10,
        "enc1": np.arange(5.0) + 20,
        "enc2": np.arange(5.0) + 30,
        ("load" if old else "loadCell"): np.arange(5.0) + 40,
        "tacho": np.arange(5.0) + 50,
    }
    if drop is not None:
        fields.pop(drop)
    savemat(path, fields)
    return fields


@pytest.mark.parametrize("old", [True, False])
def test_unsw_v1_v3_both_keep_fixed_six_output_positions(tmp_path, old):
    path = tmp_path / "unsw.mat"
    _unsw(path, old=old)
    array = RM_008_UNSW.read(path)
    assert array.shape == (5, 6) and array.dtype == np.float64
    np.testing.assert_array_equal(array[0], [0, 10, 20, 30, 40, 50])


def test_unsw_rejects_missing_or_double_named_channels(tmp_path):
    path = tmp_path / "unsw.mat"
    _unsw(path, drop="enc1")
    with pytest.raises(KeyError, match="enc1"):
        RM_008_UNSW.read(path)
    data = _unsw(path)
    data["load"] = np.arange(5.0) + 99
    savemat(path, data)
    with pytest.raises(ValueError, match="exactly one"):
        RM_008_UNSW.read(path)


def test_unsw_refuses_channel_length_misalignment(tmp_path):
    path = tmp_path / "unsw.mat"
    data = _unsw(path)
    data["enc2"] = np.arange(4.0)
    savemat(path, data)
    with pytest.raises(ValueError, match="length"):
        RM_008_UNSW.read(path)


def test_just_preserves_all_seven_measurements_and_initial_row(tmp_path):
    path = tmp_path / "just.csv"
    raw = np.arange(21).reshape(3, 7)
    pd.DataFrame(raw).to_csv(path, index=False, header=False)
    array = RM_024_JUST.read(path)
    assert array.shape == (3, 7) and array.dtype == np.float64
    np.testing.assert_array_equal(array, raw)


def test_just_rejects_unverified_index_or_header_layout(tmp_path):
    path = tmp_path / "just.csv"
    pd.DataFrame(np.arange(24).reshape(3, 8)).to_csv(path, index=False, header=False)
    with pytest.raises(ValueError, match="7 measurement columns"):
        RM_024_JUST.read(path)
    pd.DataFrame(np.arange(21).reshape(3, 7),
                 columns=[f"ch{i}" for i in range(7)]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no assumed header"):
        RM_024_JUST.read(path)
