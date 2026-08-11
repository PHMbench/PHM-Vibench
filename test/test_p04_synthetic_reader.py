from types import SimpleNamespace

import numpy as np
import pytest

from src.data_factory.reader.P04_Synthetic import read


def test_reader_loads_exact_float32_window(tmp_path) -> None:
    path = tmp_path / "sample.npy"
    expected = np.arange(1024, dtype=np.float32).reshape(512, 2)
    np.save(path, expected, allow_pickle=False)

    actual = read(str(path), SimpleNamespace())

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("shape", "dtype", "message"),
    [
        ((511, 2), np.float32, "shape"),
        ((512, 2), np.float64, "float32"),
    ],
)
def test_reader_rejects_noncanonical_sample(tmp_path, shape, dtype, message) -> None:
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros(shape, dtype=dtype), allow_pickle=False)

    with pytest.raises(ValueError, match=message):
        read(str(path))


def test_reader_rejects_nonfinite_values(tmp_path) -> None:
    path = tmp_path / "bad.npy"
    sample = np.zeros((512, 2), dtype=np.float32)
    sample[0, 0] = np.nan
    np.save(path, sample, allow_pickle=False)

    with pytest.raises(ValueError, match="NaN or Inf"):
        read(str(path))
