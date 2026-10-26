"""UNSW run-to-failure MAT: fixed source-key to output-channel alignment.

Mendeley h4df4mgrfb v1 names ``load``; v2/v3 rename this to
``loadCell``. Both versions acquire accH, accV, enc1, enc2, load, tacho.
Never infer an absent channel or promote a voltage to acceleration / N.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.io import loadmat


_SOURCE_COLUMNS = ("accH", "accV", "enc1", "enc2", "load", "tacho")


def read(file_path, *args) -> np.ndarray:
    """Return exactly six recorded streams, always in the declared source order."""
    del args
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"UNSW MAT file not found: {path}")
    if path.suffix.lower() != ".mat":
        raise ValueError(f"UNSW requires MAT input, got {path.suffix!r}")
    data = loadmat(path)
    if "Fs" not in data:
        raise KeyError(f"UNSW MAT {path} has no source sampling-rate 'Fs'")
    try:
        rate = np.asarray(data["Fs"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("UNSW 'Fs' must be a numeric scalar sampling rate in Hz") from exc
    if rate.size != 1 or not np.isfinite(rate).all() or rate.item() <= 0:
        raise ValueError("UNSW 'Fs' must be one positive finite source sampling rate")
    old, new = "load" in data, "loadCell" in data
    if old == new:
        raise ValueError(
            "UNSW requires exactly one load channel: v1 'load' or v2/v3 'loadCell'"
        )
    names = tuple("loadCell" if name == "load" and new else name for name in _SOURCE_COLUMNS)
    arrays = []
    count = None
    for name in names:
        if name not in data:
            raise KeyError(f"UNSW MAT {path} missing required channel {name!r}")
        raw = np.asarray(data[name])
        if raw.ndim == 1:
            signal = raw
        elif raw.ndim == 2 and 1 in raw.shape:
            signal = raw.reshape(-1)
        else:
            raise ValueError(f"UNSW source {name!r} must be one vector; got {raw.shape}")
        if signal.size < 2 or not np.issubdtype(signal.dtype, np.number) or np.iscomplexobj(signal):
            raise ValueError(f"UNSW source {name!r} must contain >=2 real numeric samples")
        if count is not None and signal.size != count:
            raise ValueError(f"UNSW source {name!r} length {signal.size} differs from {count}")
        numeric = np.asarray(signal, dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise FloatingPointError(f"UNSW source {name!r} contains NaN or Inf")
        count = numeric.size
        arrays.append(numeric)
    return np.column_stack(arrays)
