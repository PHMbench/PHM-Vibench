"""Strict reader for the governed P04 synthetic mechanism dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_SHAPE = (512, 2)


def read(file_path: str, args_data: Any = None) -> np.ndarray:
    """Load one immutable P04 synthetic sample.

    The generator writes one NumPy file per already-windowed observation.  This
    reader deliberately performs no resampling, normalization, padding, or
    fallback so a malformed artifact fails before training.
    """

    path = Path(file_path)
    if path.suffix != ".npy":
        raise ValueError(f"P04 synthetic samples must be .npy files: {path}")
    sample = np.load(path, allow_pickle=False)
    if sample.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"P04 synthetic sample must have shape {EXPECTED_SHAPE}, got {sample.shape}"
        )
    if sample.dtype != np.float32:
        raise ValueError(
            f"P04 synthetic sample must use float32, got {sample.dtype}"
        )
    if not np.isfinite(sample).all():
        raise ValueError("P04 synthetic sample contains NaN or Inf")
    return sample
