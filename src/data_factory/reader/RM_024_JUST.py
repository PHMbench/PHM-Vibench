"""JUST slewing-bearing source-v1: seven numerical *measurement* columns.

Original Mendeley hwg8v5j8t6 v1 defines 4 vertical accelerometer, 2
horizontal accelerometer and 1 acoustic-emission signal. No independent
index/time column is specified. Files with an additional column or a header
need original-format qualification; never drop a column by position.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def read(file_path, *args) -> np.ndarray:
    """Return 7 measured streams in the exact source CSV order as float64."""
    del args
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"JUST CSV file not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"JUST source v1 requires CSV, got {path.suffix!r}")
    frame = pd.read_csv(path, header=None)
    if frame.shape[1] != 7 or frame.shape[0] < 2:
        raise ValueError(
            f"JUST source v1 requires 7 measurement columns and >=2 rows; got {frame.shape}. "
            "A possible time/index column needs an independently confirmed source variant."
        )
    try:
        array = frame.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "JUST source v1 requires 7 numeric measurement columns, with no assumed header or time column"
        ) from exc
    if not np.isfinite(array).all():
        raise FloatingPointError("JUST source v1 contains NaN or Inf")
    return np.ascontiguousarray(array)
