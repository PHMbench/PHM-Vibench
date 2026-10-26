"""KAIST constant-load *vibration* MAT source reader, not a multimodal merger.

The publisher's four accelerometer channels are housing A x/y, housing B x/y
in units of g. The original MAT ``Signal`` structure is decoded using the
published-source layout also used by the independent Damavand KAIST digester.
Acoustic MAT, current/temperature TDMS and varying-speed CSV are *not* silently
accepted under this vibration-only contract.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.io import loadmat


def read(file_path, *args) -> np.ndarray:
    """Return exactly four source acceleration channels, shape ``[samples, 4]``."""
    del args
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"KAIST vibration MAT file not found: {path}")
    if path.suffix.lower() != ".mat":
        raise ValueError("KAIST vibration reader supports only constant-load .mat files")
    payload = loadmat(path)
    if "Signal" not in payload:
        raise KeyError(f"KAIST vibration MAT file {path} is missing 'Signal' struct")
    try:
        raw = np.asarray(payload["Signal"][0][0][1][0][0][0])
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"KAIST source {path} does not match the documented vibration 'Signal' layout"
        ) from exc
    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] != 4:
        raise ValueError(
            f"KAIST source must expose four vibration channels [samples,4], got {raw.shape}"
        )
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError("KAIST vibration channels must be real numerical samples in g")
    numeric = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise FloatingPointError("KAIST vibration contains NaN or Inf")
    return np.ascontiguousarray(numeric)
