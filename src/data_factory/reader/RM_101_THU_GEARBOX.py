"""Reader for MCC5-THU gearbox CSV files."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import fix_byte_order


CSV_COLUMNS = [
    "speed",
    "torque",
    "motor_vibration_x",
    "motor_vibration_y",
    "motor_vibration_z",
    "gearbox_vibration_x",
    "gearbox_vibration_y",
    "gearbox_vibration_z",
]


def read(file_path, *args):
    """Read one MCC5-THU CSV and return ``(length, 8)`` sensor array."""
    try:
        df = pd.read_csv(file_path, usecols=CSV_COLUMNS)
    except ValueError as exc:
        # pandas raises ValueError when one or more columns are missing.
        raise ValueError(f"Missing required columns in {file_path}: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file {file_path}: {exc}") from exc

    data = df.to_numpy(dtype=np.float32, copy=False)
    data = fix_byte_order(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data
