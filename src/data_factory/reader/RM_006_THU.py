from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io
from scipy.signal import resample_poly

from .utils import fix_byte_order


def _load_vibration(path: Path) -> np.ndarray:
    data = scipy.io.loadmat(path)["hz_1"][:, 0]
    return fix_byte_order(np.asarray(data, dtype=np.float32))


def _load_voltage(path: Path) -> np.ndarray:
    data = np.loadtxt(path)[:, 1]
    data = fix_byte_order(np.asarray(data, dtype=np.float32))
    return resample_poly(data, up=1, down=2).astype(np.float32, copy=False)


def _paired_paths(file_path: str | Path) -> tuple[Path, Path]:
    path = Path(file_path)
    parts = list(path.parts)
    if "vibration" in parts:
        idx = parts.index("vibration")
        vibration_path = path
        voltage_parts = parts.copy()
        voltage_parts[idx] = "voltage"
        voltage_path = Path(*voltage_parts).with_suffix(".txt")
    elif "voltage" in parts:
        idx = parts.index("voltage")
        voltage_path = path
        vibration_parts = parts.copy()
        vibration_parts[idx] = "vibration"
        vibration_path = Path(*vibration_parts).with_suffix(".mat")
    else:
        raise ValueError(f"RM_006_THU path must contain vibration or voltage: {path}")
    return vibration_path, voltage_path


def read(file_path, *args):
    """Read THU006 as paired vibration/voltage channels.

    The raw dataset stores vibration at 20480 Hz and voltage at 40960 Hz.
    Downsample voltage by 2, trim both streams to their common length, and
    return the cache contract shape ``(length, 2)``.
    """
    vibration_path, voltage_path = _paired_paths(file_path)
    if not vibration_path.exists():
        raise FileNotFoundError(f"THU006 vibration file not found: {vibration_path}")
    if not voltage_path.exists():
        raise FileNotFoundError(f"THU006 voltage file not found: {voltage_path}")

    vibration = _load_vibration(vibration_path)
    voltage = _load_voltage(voltage_path)
    length = min(vibration.shape[0], voltage.shape[0])
    if length <= 0:
        raise ValueError(f"THU006 paired signals are empty: {vibration_path}, {voltage_path}")

    return np.stack([vibration[:length], voltage[:length]], axis=1).astype(
        np.float32,
        copy=False,
    )


if __name__ == "__main__":
    from utils import test_reader

    test_reader(
        metadata_path="/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/data/metadata_5_data.csv",
        data_dir="/home/user/data/PHMbenchdata/PHM-Vibench/raw",
        name="RM_006_THU",
        output_dir="/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/src/data_factory/reader/output",
        read=read,
    )
