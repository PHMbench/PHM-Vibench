"""Stateless, manifest-bound CWRU preprocessing for P07.

This dedicated module is hashed into the CWRU data/split manifest.  It performs
only shape validation, immutable half-open window extraction, and per-window
per-channel population standardization.  No statistic or state crosses files or
folds, and the supplied recording is never modified in place.
"""

from __future__ import annotations

from typing import Any, Final

import torch

from .cwru_manifest import ManifestSpecimen, WindowCoordinate


PREPROCESSING_PROTOCOL_ID: Final[str] = (
    "P07-CWRU-MANIFEST-WINDOW-POPULATION-STANDARDIZATION-v1"
)


def standardize_window(window: torch.Tensor) -> torch.Tensor:
    """Return stateless per-channel population-standardized ``(L,C)`` data."""

    if not isinstance(window, torch.Tensor):
        raise TypeError("window must be a torch.Tensor.")
    if window.ndim != 2 or any(int(size) <= 0 for size in window.shape):
        raise ValueError("window must have non-empty (length,channels) shape.")
    if not torch.is_floating_point(window) or torch.is_complex(window):
        raise TypeError("window must be a real floating tensor.")
    if not bool(torch.isfinite(window).all()):
        raise ValueError("window contains non-finite values.")
    mean = window.mean(dim=0, keepdim=True)
    scale = (window - mean).square().mean(dim=0, keepdim=True).sqrt()
    if bool((scale <= torch.finfo(window.dtype).eps).any()):
        raise ValueError("window has a constant channel and cannot be standardized.")
    standardized = (window - mean) / scale
    if not bool(torch.isfinite(standardized).all()):
        raise FloatingPointError("Window standardization produced non-finite values.")
    return standardized


def materialize_manifest_windows(
    value: Any,
    specimen: ManifestSpecimen,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Extract exactly the specimen's ordered coordinates into ``(W,L,C)``.

    The reader output must match the manifest's full recording shape.  This
    prevents silent truncation, channel padding, or reader-specific fallback.
    """

    if not isinstance(specimen, ManifestSpecimen):
        raise TypeError("specimen must be a ManifestSpecimen.")
    if dtype not in {torch.float32, torch.float64}:
        raise TypeError("dtype must be torch.float32 or torch.float64.")
    try:
        recording = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"Reader output for {specimen.specimen_key} is not tensor-like."
        ) from error
    if recording.ndim == 3 and int(recording.shape[0]) == 1:
        recording = recording.squeeze(0)
    expected_shape = (specimen.sample_length, specimen.channels)
    if recording.ndim != 2 or tuple(recording.shape) != expected_shape:
        raise ValueError(
            f"Reader output for {specimen.specimen_key} must have shape "
            f"{expected_shape}, got {tuple(recording.shape)}."
        )
    recording = recording.detach().to(dtype=dtype, device="cpu")
    if not bool(torch.isfinite(recording).all()):
        raise ValueError(f"Reader output for {specimen.specimen_key} is non-finite.")

    coordinates = tuple(specimen.windows)
    _validate_window_coordinates(coordinates, specimen)
    return torch.stack(
        [
            standardize_window(recording[item.start : item.stop].clone())
            for item in coordinates
        ],
        dim=0,
    )


def _validate_window_coordinates(
    coordinates: tuple[WindowCoordinate, ...],
    specimen: ManifestSpecimen,
) -> None:
    if not coordinates:
        raise ValueError(f"Specimen {specimen.specimen_key} has no manifest windows.")
    for expected_index, coordinate in enumerate(coordinates):
        if not isinstance(coordinate, WindowCoordinate):
            raise TypeError("Manifest coordinates must be WindowCoordinate objects.")
        if coordinate.index != expected_index:
            raise ValueError("Manifest window indices must be contiguous and ordered.")
        if not 0 <= coordinate.start < coordinate.stop <= specimen.sample_length:
            raise ValueError("Manifest window coordinate lies outside its recording.")
    if any(
        left.stop > right.start
        for left, right in zip(coordinates, coordinates[1:])
    ):
        raise ValueError("Manifest window coordinates overlap.")


__all__ = [
    "PREPROCESSING_PROTOCOL_ID",
    "materialize_manifest_windows",
    "standardize_window",
]
