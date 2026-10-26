"""Source-scoped steady-state HUST 2023 bearing MAT reader.

Original publication: doi:10.1186/s13104-023-06400-4; ``data`` is the
vibration record. MAT ``fs`` is *shaft frequency*, not sampling frequency.
Run-up signals ``ru``/``ru_raw`` require a separate explicit timebase policy.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.io import loadmat


def read(file_path, *args) -> np.ndarray:
    """Return source ``data`` as a real finite ``[samples, 1]`` signal."""
    del args
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"HUST23 MAT file not found: {path}")
    if path.suffix.lower() != ".mat":
        raise ValueError(f"HUST23 requires a MAT file, observed {path.suffix!r}")
    payload = loadmat(path)
    if "data" not in payload:
        raise KeyError(
            f"HUST23 file {path} has no steady-state 'data' signal; "
            "run-up 'ru'/'ru_raw' requires a separate explicit timebase contract"
        )
    raw = np.asarray(payload["data"])
    if raw.ndim == 1:
        signal = raw
    elif raw.ndim == 2 and 1 in raw.shape:
        signal = raw.reshape(-1)
    else:
        raise ValueError(f"HUST23 'data' requires a single signal vector, got shape={raw.shape}")
    if signal.size < 2 or not np.issubdtype(signal.dtype, np.number) or np.iscomplexobj(signal):
        raise ValueError("HUST23 'data' requires >=2 real numeric samples")
    numeric = np.asarray(signal, dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise FloatingPointError("HUST23 'data' contains NaN or Inf")
    return np.ascontiguousarray(numeric.reshape(-1, 1))
