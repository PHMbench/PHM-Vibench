"""The historical pump XLSX reader must return numeric rows, not call ndarray.float()."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_factory.reader.RM_011_pump import read


def test_pump_reader_preserves_last_sheet_numeric_columns_after_header(tmp_path):
    pytest.importorskip("openpyxl")
    file_path = tmp_path / "pump.xlsx"
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        pd.DataFrame([[100, 200]]).to_excel(
            writer, sheet_name="first", header=False, index=False,
        )
        pd.DataFrame([
            ["header", "unit"], [1.5, 2.0], [-3.0, 4.0],
        ]).to_excel(writer, sheet_name="last", header=False, index=False)
    values = read(str(file_path))
    assert isinstance(values, np.ndarray)
    assert values.dtype == np.float64
    assert values.shape == (2, 2)
    np.testing.assert_array_equal(
        values, np.asarray([[1.5, 2.0], [-3.0, 4.0]], dtype=np.float64),
    )


def test_pump_reader_does_not_fabricate_values_from_invalid_numeric_cells(tmp_path):
    pytest.importorskip("openpyxl")
    file_path = tmp_path / "invalid_pump.xlsx"
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        pd.DataFrame([
            ["header", "unit"], [1.5, "not a measurement"],
        ]).to_excel(writer, sheet_name="last", header=False, index=False)
    with pytest.raises(ValueError):
        read(str(file_path))
