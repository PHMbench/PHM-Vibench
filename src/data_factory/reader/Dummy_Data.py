"""Strict reader for the repository-shipped Dummy_Data CSV fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_SIGNAL_COLUMNS = ("ch1", "ch2")


def read(file_path: str | Path, args_data: Any = None) -> np.ndarray:
    """Read one Dummy_Data CSV as a ``[length, 2]`` float32 signal.

    The offline demo is valid only when it consumes the CSV files shipped with the
    repository or wheel. Missing or malformed inputs fail at the reader boundary;
    this reader never generates substitute signals, pads channels, or guesses columns.
    """

    del args_data
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dummy_Data signal file not found: {path}. The offline demo requires "
            "the repository-shipped data/raw/Dummy_Data/*.csv files and does not "
            "synthesize fallback signals. Reinstall the package or restore the "
            "missing fixture."
        )

    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dummy_Data signal file is empty: {path}") from exc
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Unable to parse Dummy_Data CSV file {path}: {exc}") from exc

    missing = [column for column in _SIGNAL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Dummy_Data signal file {path} is missing required column(s) {missing}. "
            f"Expected channel order {list(_SIGNAL_COLUMNS)}; available columns are "
            f"{list(frame.columns)}."
        )

    selected = frame.loc[:, list(_SIGNAL_COLUMNS)]
    try:
        numeric = selected.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Dummy_Data columns {list(_SIGNAL_COLUMNS)} must contain only numeric "
            f"signal values: {path}."
        ) from exc

    signal = numeric.to_numpy(dtype=np.float32, copy=True)
    if signal.ndim != 2 or signal.shape[0] == 0 or signal.shape[1] != len(_SIGNAL_COLUMNS):
        raise ValueError(
            f"Dummy_Data signal file {path} produced invalid shape {signal.shape}; "
            f"expected [length, {len(_SIGNAL_COLUMNS)}] with at least one row."
        )
    if not np.isfinite(signal).all():
        raise FloatingPointError(
            f"Dummy_Data signal file {path} contains NaN or Inf in "
            f"{list(_SIGNAL_COLUMNS)}."
        )
    return signal


def get_dataset(args_data: Any, file_path: str | Path) -> np.ndarray:
    """Compatibility helper using the same strict reader contract."""

    return read(file_path, args_data)
