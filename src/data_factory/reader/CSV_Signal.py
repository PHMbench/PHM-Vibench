"""Reader for headered CSV files with explicitly configured signal columns."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read(file_path, args_data):
    """Read selected numeric CSV columns as a ``[length, channels]`` array.

    Required data configuration:

    ```yaml
    data:
      csv_signal_columns: [sensor_a, sensor_b]
    ```

    Columns are never guessed. Index, time, label, or metadata columns remain excluded
    unless the user explicitly lists them as signal channels.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV signal file not found: {path}")

    columns = getattr(args_data, "csv_signal_columns", None)
    if not isinstance(columns, (list, tuple)) or not columns:
        raise ValueError(
            "CSV_Signal requires data.csv_signal_columns as a non-empty list "
            "of header names."
        )
    column_names = [str(column) for column in columns]
    if len(set(column_names)) != len(column_names):
        raise ValueError(
            "data.csv_signal_columns contains duplicate column names: "
            f"{column_names}."
        )

    delimiter = str(getattr(args_data, "csv_delimiter", ","))
    frame = pd.read_csv(path, sep=delimiter)
    missing = [column for column in column_names if column not in frame.columns]
    if missing:
        raise ValueError(
            f"CSV signal file {path} is missing configured column(s) {missing}. "
            f"Available columns: {list(frame.columns)}."
        )

    selected = frame.loc[:, column_names]
    try:
        numeric = selected.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Configured CSV signal columns must be numeric: {column_names}."
        ) from exc

    dtype_name = str(getattr(args_data, "dtype", "float32"))
    if dtype_name not in {"float32", "float64"}:
        raise ValueError(
            "CSV_Signal supports data.dtype 'float32' or 'float64', "
            f"got {dtype_name!r}."
        )
    array = numeric.to_numpy(dtype=np.dtype(dtype_name), copy=True)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(
            f"CSV signal file {path} produced invalid shape {array.shape}; "
            "at least one row and one configured channel are required."
        )
    if not np.isfinite(array).all():
        raise FloatingPointError(
            f"CSV signal file {path} contains NaN or Inf in {column_names}."
        )
    return array
