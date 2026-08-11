"""Stateless preprocessing contract for the P07 DIRG validation protocol.

Each source file is handled independently.  The module takes 24 deterministic,
non-overlapping windows distributed uniformly from the first to the last valid
start coordinate, then applies population (``ddof=0``) standardization within
each window and channel.  No statistic can cross a window or file boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np


SCHEMA_VERSION: Final[int] = 1
ALGORITHM_ID: Final[str] = "P07-DIRG-WINDOW-PREPROCESSING-v1"
WINDOW_ALGORITHM_ID: Final[str] = "p07-evenly-distributed-nonoverlap-v1"
EXPECTED_SIGNAL_LENGTH: Final[int] = 512_000
EXPECTED_CHANNELS: Final[int] = 6
WINDOW_COUNT: Final[int] = 24
WINDOW_LENGTH: Final[int] = 4_096
STANDARDIZATION: Final[str] = "per_window_per_channel_population_ddof_0"


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class WindowCoordinate:
    """One half-open, deterministic source interval."""

    window_index: int
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start

    def to_payload(self) -> dict[str, int]:
        return {
            "window_index": self.window_index,
            "start": self.start,
            "stop": self.stop,
            "length": self.length,
        }


def uniform_window_coordinates(
    signal_length: int = EXPECTED_SIGNAL_LENGTH,
    window_length: int = WINDOW_LENGTH,
    window_count: int = WINDOW_COUNT,
) -> tuple[WindowCoordinate, ...]:
    """Return exact integer-floor starts spanning both valid endpoints."""

    for value, label in (
        (signal_length, "signal_length"),
        (window_length, "window_length"),
        (window_count, "window_count"),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{label} must be an integer.")
        if value <= 0:
            raise ValueError(f"{label} must be positive.")
    if window_count < 2:
        raise ValueError("window_count must be at least two for endpoint-uniform spacing.")
    if window_length > signal_length:
        raise ValueError("window_length cannot exceed signal_length.")

    last_start = signal_length - window_length
    coordinates = tuple(
        WindowCoordinate(
            window_index=index,
            start=(index * last_start) // (window_count - 1),
            stop=(index * last_start) // (window_count - 1) + window_length,
        )
        for index in range(window_count)
    )
    return validate_window_coordinates(
        coordinates,
        signal_length=signal_length,
        window_length=window_length,
        window_count=window_count,
    )


def validate_window_coordinates(
    coordinates: Sequence[WindowCoordinate],
    *,
    signal_length: int = EXPECTED_SIGNAL_LENGTH,
    window_length: int = WINDOW_LENGTH,
    window_count: int = WINDOW_COUNT,
) -> tuple[WindowCoordinate, ...]:
    """Fail closed on order, overlap, bounds, length, or spacing drift."""

    if isinstance(coordinates, (str, bytes)):
        raise TypeError("coordinates must be a sequence of WindowCoordinate objects.")
    normalized = tuple(coordinates)
    if len(normalized) != window_count:
        raise ValueError(f"Expected {window_count} coordinates, got {len(normalized)}.")
    for index, coordinate in enumerate(normalized):
        if not isinstance(coordinate, WindowCoordinate):
            raise TypeError("coordinates must contain WindowCoordinate objects.")
        if coordinate.window_index != index:
            raise ValueError("Window coordinate indices are not contiguous and ordered.")
        if coordinate.length != window_length:
            raise ValueError("Window coordinate length drifted.")
        if coordinate.start < 0 or coordinate.stop > signal_length:
            raise ValueError("Window coordinate is outside the source signal.")
        if index and normalized[index - 1].stop > coordinate.start:
            raise ValueError("Window coordinates overlap.")
    expected_starts = tuple(
        (index * (signal_length - window_length)) // (window_count - 1)
        for index in range(window_count)
    )
    if tuple(item.start for item in normalized) != expected_starts:
        raise ValueError("Window coordinates are not the frozen uniform integer grid.")
    if normalized[0].start != 0 or normalized[-1].stop != signal_length:
        raise ValueError("Window coordinates do not bind both source endpoints.")
    return normalized


def coordinate_set_sha256(
    coordinates: Sequence[WindowCoordinate] | None = None,
) -> str:
    normalized = validate_window_coordinates(
        uniform_window_coordinates() if coordinates is None else coordinates
    )
    return hashlib.sha256(
        _canonical_json_bytes([item.to_payload() for item in normalized])
    ).hexdigest()


def _validate_reader_array(data: np.ndarray) -> np.ndarray:
    if not isinstance(data, np.ndarray):
        raise TypeError("DIRG reader output must be a numpy.ndarray.")
    if data.ndim != 2 or data.shape != (
        EXPECTED_SIGNAL_LENGTH,
        EXPECTED_CHANNELS,
    ):
        raise ValueError(
            "DIRG reader output must have exact shape "
            f"({EXPECTED_SIGNAL_LENGTH}, {EXPECTED_CHANNELS}), got {data.shape}."
        )
    if data.dtype.kind not in {"f", "i", "u"}:
        raise TypeError("DIRG reader output must contain real numeric values.")
    values = np.asarray(data, dtype=np.float64)
    if not bool(np.isfinite(values).all()):
        raise ValueError("DIRG reader output contains non-finite values.")
    return values


def population_standardize_windows(windows: np.ndarray) -> np.ndarray:
    """Standardize each ``(length, channel)`` window without shared state."""

    if not isinstance(windows, np.ndarray):
        raise TypeError("windows must be a numpy.ndarray.")
    if windows.shape != (WINDOW_COUNT, WINDOW_LENGTH, EXPECTED_CHANNELS):
        raise ValueError(
            "windows must have exact shape "
            f"({WINDOW_COUNT}, {WINDOW_LENGTH}, {EXPECTED_CHANNELS})."
        )
    values = np.asarray(windows, dtype=np.float64)
    if not bool(np.isfinite(values).all()):
        raise ValueError("DIRG windows contain non-finite values.")
    means = values.mean(axis=1, keepdims=True, dtype=np.float64)
    scales = values.std(axis=1, keepdims=True, ddof=0, dtype=np.float64)
    if not bool(np.isfinite(means).all()) or not bool(np.isfinite(scales).all()):
        raise ValueError("DIRG population moments are non-finite.")
    if bool(np.any(scales <= 0.0)):
        raise ValueError("DIRG window has a zero-variance channel.")
    standardized = (values - means) / scales
    if not bool(np.isfinite(standardized).all()):
        raise ValueError("DIRG standardized windows contain non-finite values.")
    return standardized


def materialize_dirg_windows(
    data: np.ndarray,
    *,
    coordinates: Sequence[WindowCoordinate] | None = None,
) -> np.ndarray:
    """Extract and independently standardize one file's frozen windows."""

    values = _validate_reader_array(data)
    normalized_coordinates = validate_window_coordinates(
        uniform_window_coordinates() if coordinates is None else coordinates
    )
    windows = np.stack(
        [values[item.start : item.stop, :].copy() for item in normalized_coordinates],
        axis=0,
    )
    standardized = population_standardize_windows(windows)
    if standardized.shape != (WINDOW_COUNT, WINDOW_LENGTH, EXPECTED_CHANNELS):
        raise AssertionError("Validated DIRG preprocessing changed its output shape.")
    return standardized


def validate_materialized_windows(windows: np.ndarray) -> np.ndarray:
    """Check shape, finite values, and population moments of an output batch."""

    if not isinstance(windows, np.ndarray):
        raise TypeError("materialized windows must be a numpy.ndarray.")
    if windows.shape != (WINDOW_COUNT, WINDOW_LENGTH, EXPECTED_CHANNELS):
        raise ValueError("Materialized DIRG window shape drifted.")
    if windows.dtype != np.float64:
        raise TypeError("Materialized DIRG windows must be float64.")
    if not bool(np.isfinite(windows).all()):
        raise ValueError("Materialized DIRG windows contain non-finite values.")
    means = windows.mean(axis=1, dtype=np.float64)
    scales = windows.std(axis=1, ddof=0, dtype=np.float64)
    tolerance = 128.0 * np.finfo(np.float64).eps
    if not bool(np.all(np.abs(means) <= tolerance)):
        raise ValueError("Materialized DIRG windows are not zero-mean per channel.")
    if not bool(np.all(np.abs(scales - 1.0) <= tolerance)):
        raise ValueError("Materialized DIRG windows lack unit population scale.")
    return windows


def preprocessing_source_sha256() -> str:
    """Hash this source independently of any generated manifest."""

    source = Path(__file__).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if len(digest) != 64 or not all(character in "0123456789abcdef" for character in digest):
        raise AssertionError("Preprocessing source SHA-256 construction failed.")
    return digest


def preprocessing_contract_payload() -> dict[str, Any]:
    coordinates = uniform_window_coordinates()
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm_id": ALGORITHM_ID,
        "input_shape": [EXPECTED_SIGNAL_LENGTH, EXPECTED_CHANNELS],
        "window_count": WINDOW_COUNT,
        "window_length": WINDOW_LENGTH,
        "window_algorithm_id": WINDOW_ALGORITHM_ID,
        "coordinate_rule": "integer_floor_uniform_starts_including_both_endpoints",
        "coordinate_set_sha256": coordinate_set_sha256(coordinates),
        "standardization": STANDARDIZATION,
        "population_ddof": 0,
        "cross_file_state": False,
        "output_dtype": "float64",
        "source_sha256": preprocessing_source_sha256(),
    }


__all__ = [
    "ALGORITHM_ID",
    "EXPECTED_CHANNELS",
    "EXPECTED_SIGNAL_LENGTH",
    "SCHEMA_VERSION",
    "STANDARDIZATION",
    "WINDOW_COUNT",
    "WINDOW_ALGORITHM_ID",
    "WINDOW_LENGTH",
    "WindowCoordinate",
    "coordinate_set_sha256",
    "materialize_dirg_windows",
    "population_standardize_windows",
    "preprocessing_contract_payload",
    "preprocessing_source_sha256",
    "uniform_window_coordinates",
    "validate_materialized_windows",
    "validate_window_coordinates",
]
