"""Fail-closed transforms used by registered P05 evidence data."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class WindowSpan:
    """The immutable identity of one window within a source record."""

    index: int
    start: int
    end: int


@dataclass(frozen=True)
class WindowObservation:
    """One identified training window supplied to normalization fitting."""

    sample_id: str
    group_id: str
    values: np.ndarray


@dataclass(frozen=True)
class ChannelStandardizationPlan:
    """Frozen train-only population statistics and their contract digest."""

    dataset_id: int
    channel_names: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    group_window_counts: Mapping[str, int]
    window_size: int
    window_count: int
    sha256: str


def exact_evenly_spaced_spans(
    data_length: int,
    window_size: int,
    count: int,
) -> tuple[WindowSpan, ...]:
    """Return the preregistered integer-floor, non-overlapping window spans.

    For ``i=0..count-1`` the start is exactly
    ``i * (data_length - window_size) // (count - 1)``.  The function refuses
    to reduce the requested count, pad short records, round floating-point
    steps, or accept overlapping windows.
    """

    values = {
        "data_length": data_length,
        "window_size": window_size,
        "count": count,
    }
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values.values()):
        raise TypeError("window contract values must be integers")
    if data_length <= 0 or window_size <= 0:
        raise ValueError("data_length and window_size must be positive")
    if count < 2:
        raise ValueError("the P05 window contract requires at least two windows")
    if data_length < window_size:
        raise ValueError(
            f"record length {data_length} is shorter than window size {window_size}"
        )

    available = data_length - window_size
    starts = [i * available // (count - 1) for i in range(count)]
    spans = tuple(
        WindowSpan(index=index, start=start, end=start + window_size)
        for index, start in enumerate(starts)
    )

    if starts[0] != 0 or starts[-1] != available:
        raise AssertionError("window endpoint construction violated its exact formula")
    if len(set(starts)) != count or starts != sorted(starts):
        raise ValueError("requested protocol windows are not unique and ordered")
    for left, right in zip(spans, spans[1:]):
        if right.start < left.end:
            raise ValueError(
                "requested protocol windows overlap; the record is too short for "
                f"count={count}, window_size={window_size}"
            )
    return spans


def protocol_sample_id(record_id: object, span: WindowSpan) -> str:
    """Build a stable, human-readable window identity."""

    return f"{record_id}:{span.start}:{span.end}"


def _normalization_contract(
    *,
    dataset_id: int,
    channel_names: Sequence[str],
    mean: Sequence[float],
    std: Sequence[float],
    group_window_counts: Mapping[str, int],
    window_size: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "paper_id": "P05",
        "dataset_id": int(dataset_id),
        "fit_role": "train",
        "method": "equal_group_equal_window_equal_point_population_standardization",
        "accumulator_dtype": "float64",
        "channel_names": list(channel_names),
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "group_window_counts": {
            str(group): int(count)
            for group, count in sorted(group_window_counts.items(), key=lambda item: item[0])
        },
        "window_size": int(window_size),
        "window_count": int(sum(group_window_counts.values())),
    }


def _contract_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def fit_train_channel_standardization(
    observation_factory: Callable[[], Iterable[WindowObservation]],
    *,
    dataset_id: int,
    channel_names: Sequence[str],
    expected_window_size: int,
    expected_windows_per_group: Mapping[str, int],
) -> ChannelStandardizationPlan:
    """Fit the preregistered train-only channel transform in float64.

    The callable is traversed once to verify identities/counts, once for the
    weighted mean, and once for the centered population variance.  Every pass
    must yield the exact same identified windows.
    """

    channels = tuple(str(name) for name in channel_names)
    if not channels or any(not name for name in channels):
        raise ValueError("channel_names must be non-empty strings")
    if expected_window_size <= 0:
        raise ValueError("expected_window_size must be positive")
    expected_counts = {
        str(group): int(count) for group, count in expected_windows_per_group.items()
    }
    if not expected_counts or any(count <= 0 for count in expected_counts.values()):
        raise ValueError("expected_windows_per_group must contain positive counts")

    def checked_pass() -> tuple[list[WindowObservation], tuple[str, ...]]:
        observations = list(observation_factory())
        identities: list[str] = []
        counts: dict[str, int] = {}
        for observation in observations:
            if not isinstance(observation.sample_id, str) or not observation.sample_id:
                raise ValueError("normalization sample_id must be a non-empty string")
            if not isinstance(observation.group_id, str) or not observation.group_id:
                raise ValueError("normalization group_id must be a non-empty string")
            values = np.asarray(observation.values)
            if values.shape != (expected_window_size, len(channels)):
                raise ValueError(
                    "normalization window shape mismatch: expected "
                    f"{(expected_window_size, len(channels))}, got {values.shape}"
                )
            if not np.isfinite(values).all():
                raise ValueError("normalization windows must contain only finite values")
            identities.append(observation.sample_id)
            counts[observation.group_id] = counts.get(observation.group_id, 0) + 1
        if len(identities) != len(set(identities)):
            raise ValueError("normalization sample_id values must be unique")
        if counts != expected_counts:
            raise ValueError(
                f"normalization group/window counts mismatch: expected {expected_counts}, "
                f"got {counts}"
            )
        return observations, tuple(identities)

    _, reference_identities = checked_pass()
    group_count = len(expected_counts)
    mean = np.zeros(len(channels), dtype=np.float64)
    observations, identities = checked_pass()
    if identities != reference_identities:
        raise ValueError("normalization observation identities changed between passes")
    for observation in observations:
        values = np.asarray(observation.values, dtype=np.float64)
        point_weight = 1.0 / (
            group_count
            * expected_counts[observation.group_id]
            * expected_window_size
        )
        mean += values.sum(axis=0, dtype=np.float64) * point_weight

    variance = np.zeros(len(channels), dtype=np.float64)
    observations, identities = checked_pass()
    if identities != reference_identities:
        raise ValueError("normalization observation identities changed between passes")
    for observation in observations:
        values = np.asarray(observation.values, dtype=np.float64)
        point_weight = 1.0 / (
            group_count
            * expected_counts[observation.group_id]
            * expected_window_size
        )
        centered = values - mean
        variance += np.square(centered).sum(axis=0, dtype=np.float64) * point_weight

    if not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise ValueError("normalization statistics are not finite")
    if np.any(variance < 0.0):
        raise ValueError("normalization variance must be non-negative")
    std = np.sqrt(variance, dtype=np.float64)
    if np.any(std < 1e-8):
        raise ValueError("normalization channel standard deviation is below 1e-8")
    contract = _normalization_contract(
        dataset_id=dataset_id,
        channel_names=channels,
        mean=mean,
        std=std,
        group_window_counts=expected_counts,
        window_size=expected_window_size,
    )
    digest = hashlib.sha256(_contract_bytes(contract)).hexdigest()
    return ChannelStandardizationPlan(
        dataset_id=int(dataset_id),
        channel_names=channels,
        mean=tuple(float(value) for value in mean),
        std=tuple(float(value) for value in std),
        group_window_counts=expected_counts,
        window_size=expected_window_size,
        window_count=sum(expected_counts.values()),
        sha256=digest,
    )


def apply_train_channel_standardization(
    window: np.ndarray,
    plan: ChannelStandardizationPlan,
    *,
    output_dtype: np.dtype = np.dtype("float32"),
) -> np.ndarray:
    """Apply one frozen plan without refitting any statistics."""

    values = np.asarray(window)
    expected_shape = (plan.window_size, len(plan.channel_names))
    if values.shape != expected_shape:
        raise ValueError(
            f"standardization window shape mismatch: expected {expected_shape}, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("standardization input must contain only finite values")
    mean = np.asarray(plan.mean, dtype=np.float64)
    std = np.asarray(plan.std, dtype=np.float64)
    if np.any(std < 1e-8) or not np.isfinite(std).all():
        raise ValueError("standardization plan contains an invalid standard deviation")
    transformed = (values.astype(np.float64, copy=False) - mean) / std
    dtype = np.dtype(output_dtype)
    if dtype not in {np.dtype("float32"), np.dtype("float64")}:
        raise ValueError("standardization output dtype must be float32 or float64")
    return transformed.astype(dtype, copy=False)
