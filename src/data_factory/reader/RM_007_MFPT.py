"""Strict reader for the MFPT bearing-test-rig MAT files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


_CRITICAL_FREQUENCIES = ("BPFO", "BPFI", "FTF", "BSF")


def _unwrap_singleton(value: Any) -> Any:
    """Unwrap MATLAB singleton containers without flattening signal vectors."""

    current = value
    while isinstance(current, np.ndarray) and current.size == 1:
        current = current.reshape(-1)[0]
    return current


def _field(container: Any, name: str, path: Path) -> Any:
    """Return one required MATLAB struct field."""

    value = _unwrap_singleton(container)
    if isinstance(value, Mapping) and name in value:
        return value[name]
    if hasattr(value, name):
        return getattr(value, name)
    if isinstance(value, np.void) and value.dtype.names and name in value.dtype.names:
        return value[name]
    if isinstance(value, np.ndarray) and value.dtype.names and name in value.dtype.names:
        return value[name]
    raise KeyError(f"MFPT file {path} is missing required field {name!r}.")


def _finite_scalar(value: Any, name: str, path: Path, *, positive: bool) -> float:
    raw = np.asarray(_unwrap_singleton(value))
    if raw.size != 1:
        raise ValueError(
            f"MFPT field {name!r} in {path} must contain one scalar, "
            f"observed shape={raw.shape}."
        )
    try:
        result = float(raw.reshape(-1)[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"MFPT field {name!r} in {path} must be numeric."
        ) from exc
    if not np.isfinite(result):
        raise FloatingPointError(
            f"MFPT field {name!r} in {path} contains NaN or Inf."
        )
    if positive and result <= 0:
        raise ValueError(
            f"MFPT field {name!r} in {path} must be positive, got {result}."
        )
    return result


def _signal(value: Any, path: Path) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim == 2 and 1 in raw.shape:
        raw = raw.reshape(-1)
    elif raw.ndim != 1:
        raise ValueError(
            f"MFPT bearing.gs in {path} must be one vibration vector, "
            f"observed shape={raw.shape}."
        )
    if raw.size < 2:
        raise ValueError(
            f"MFPT bearing.gs in {path} must contain at least two samples."
        )
    if not np.issubdtype(raw.dtype, np.number):
        raise ValueError(
            f"MFPT bearing.gs in {path} must contain numeric samples, "
            f"observed dtype={raw.dtype}."
        )
    try:
        numeric = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"MFPT bearing.gs in {path} cannot be converted to numeric samples."
        ) from exc
    if not np.isfinite(numeric).all():
        raise FloatingPointError(
            f"MFPT bearing.gs in {path} contains NaN or Inf values."
        )
    return numeric.reshape(-1, 1)


def read_record(file_path: str | Path) -> dict[str, Any]:
    """Read and validate one official MFPT bearing-test-rig record.

    The returned signal has shape ``[length, 1]``. Sampling rate, shaft rate,
    load, and the four bearing critical frequencies are validated from the MAT
    payload so preparation code does not infer physical metadata from filenames.
    """

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"MFPT MAT file not found: {path}")
    try:
        payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    except (OSError, ValueError, TypeError, NotImplementedError) as exc:
        raise ValueError(f"Unable to read MFPT MAT file {path}: {exc}") from exc

    if "bearing" not in payload:
        raise KeyError(f"MFPT file {path} is missing required top-level field 'bearing'.")
    bearing = _unwrap_singleton(payload["bearing"])
    signal = _signal(_field(bearing, "gs", path), path)
    sample_rate_hz = _finite_scalar(
        _field(bearing, "sr", path),
        "bearing.sr",
        path,
        positive=True,
    )
    shaft_rate_hz = _finite_scalar(
        _field(bearing, "rate", path),
        "bearing.rate",
        path,
        positive=True,
    )
    load = _finite_scalar(
        _field(bearing, "load", path),
        "bearing.load",
        path,
        positive=False,
    )

    critical: dict[str, float] = {}
    for name in _CRITICAL_FREQUENCIES:
        source = _field(bearing, name, path) if hasattr(bearing, name) else payload.get(name)
        if source is None:
            raise KeyError(
                f"MFPT file {path} is missing required critical frequency {name!r}."
            )
        critical[name] = _finite_scalar(
            source,
            name,
            path,
            positive=True,
        )

    return {
        "signal": signal,
        "sample_rate_hz": sample_rate_hz,
        "shaft_rate_hz": shaft_rate_hz,
        "load": load,
        **critical,
    }


def read(file_path: str | Path, *args: Any) -> np.ndarray:
    """Return the validated single-channel acceleration signal."""

    del args
    return read_record(file_path)["signal"]
