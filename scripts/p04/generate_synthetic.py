"""Generate and validate the frozen P04 synthetic mechanism dataset.

The on-disk dataset is intentionally simple for the existing PHM-Vibench data
factory::

    metadata.csv
    raw/P04_Synthetic/sample_000000.npy  # float32, shape [512, 2]
    cell_manifest.jsonl
    sample_manifest.jsonl
    partition_manifest.json
    generator_manifest.json
    artifact_hashes.sha256

Generation is deterministic, refuses to overwrite an existing output
directory, and stages a complete dataset before publishing it with one rename.
The model-facing reader must use only the array and target label; mechanism and
nuisance fields are audit metadata for held-out evaluators.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
import platform
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_ID = "p04.synthetic-generator.v1"
SCHEMA_VERSION = "1.0.0"
PARTITION_SCHEMA_ID = "p04.synthetic-partition-manifest.v1"
PARTITION_SCHEMA_VERSION = 1
GENERATOR_DOMAIN = "P04-SYN-v1"
MASTER_SEED = 240401
DATASET_ID = 904
DOMAIN_ID = 0
SAMPLE_ID_BASE = 904_000_000
DATASET_NAME = "P04_Synthetic"
WINDOW_LENGTH = 512
CHANNELS = 2
DRAWS_PER_CELL = 8

DIAGNOSIS_FREQUENCIES = (0.035, 0.050, 0.065, 0.080)
MECHANISMS = (
    "low_frequency",
    "harmonic",
    "impulsive_envelope",
    "aperiodic_residual",
)
NOISE_PSD_SLOPES = (0, -1, -2)
SNR_LEVELS_DB = (0, 10, 20)
AMPLITUDES = (0.75, 1.25)
FREQUENCY_MULTIPLIERS = (0.95, 1.05)
PARTITION_TARGETS = (
    ("intervention", 5),
    ("identification", 5),
    ("optimization_validation", 6),
)
PARTITION_NAMES = (
    "train",
    "optimization_validation",
    "identification",
    "intervention",
)
PARTITION_CELL_COUNTS = {
    "train": 20,
    "optimization_validation": 6,
    "identification": 5,
    "intervention": 5,
}

AMPLITUDE_ABSOLUTE_TOLERANCE = 0.01
SNR_DB_ABSOLUTE_TOLERANCE = 0.01
LOW_POWER_RATIO_MINIMUM = 0.80
LOW_POWER_CUTOFF = 0.12
HARMONIC_PEAK_BIN_TOLERANCE = 1
HARMONIC_COMBINED_POWER_MINIMUM = 0.70
IMPULSE_RECURRENCE_TOLERANCE = 1
IMPULSE_KURTOSIS_MINIMUM = 5.0
RESIDUAL_MAX_AUTOCORRELATION = 0.30
RESIDUAL_MAX_PEAK_SHARE = 0.20
RESIDUAL_EXCESS_CENTER_TOLERANCE = 0.02
NOISE_SLOPE_ABSOLUTE_TOLERANCE = 0.10

METADATA_COLUMNS = (
    "Id",
    "Dataset_id",
    "Domain_id",
    "Label",
    "Name",
    "File",
    "Split_group",
    "Split_stratum",
    "Partition",
    "Mechanism",
    "Nuisance_cell",
    "Draw",
)


class ProtocolValidationError(RuntimeError):
    """Raised when a frozen generator or artifact invariant is violated."""


@dataclass(frozen=True)
class Cell:
    """One complete nuisance cell in the orthogonal diagnosis/mechanism grid."""

    cell_id: int
    diagnosis_label: int
    mechanism_id: int
    mechanism: str
    noise_psd_slope: int
    snr_db: int
    amplitude: float
    frequency_multiplier: float
    tuple_text: str
    allocation_sha256: str
    partition: str = ""

    @property
    def split_group(self) -> str:
        return f"P04_SYN_CELL_{self.cell_id:04d}"

    @property
    def split_stratum(self) -> str:
        return f"Y{self.diagnosis_label}:M{self.mechanism_id}"

    @property
    def frequency_code(self) -> float:
        return (
            DIAGNOSIS_FREQUENCIES[self.diagnosis_label]
            * self.frequency_multiplier
        )

    def factor_values(self) -> tuple[int, int, float, float]:
        return (
            self.noise_psd_slope,
            self.snr_db,
            self.amplitude,
            self.frequency_multiplier,
        )


@dataclass(frozen=True)
class GeneratedSample:
    """A generated sample plus its deterministic audit record."""

    array: np.ndarray
    metadata: Mapping[str, Any]
    manifest: Mapping[str, Any]
    audit: Mapping[str, float]


@dataclass(frozen=True)
class BuildResult:
    """Complete in-memory representation used by dry-run and materialization."""

    cells: tuple[Cell, ...]
    samples: tuple[GeneratedSample, ...]
    validation_summary: Mapping[str, Any]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(record) for record in records)


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def _seed(domain: str, cell: Cell, draw: int) -> int:
    material = (
        f"{GENERATOR_DOMAIN}|master={MASTER_SEED}|{cell.tuple_text}|"
        f"draw={draw}|component={domain}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _cell_tuple_text(
    diagnosis_label: int,
    mechanism: str,
    noise_psd_slope: int,
    snr_db: int,
    amplitude: float,
    frequency_multiplier: float,
) -> str:
    return (
        f"({diagnosis_label},{mechanism},{noise_psd_slope},{snr_db},"
        f"{amplitude:.2f},{frequency_multiplier:.2f})"
    )


def enumerate_cells() -> tuple[Cell, ...]:
    """Enumerate and deterministically allocate all 576 frozen cells."""

    cells: list[Cell] = []
    cell_id = 0
    for diagnosis_label, mechanism_id, noise_slope, snr_db, amplitude, multiplier in itertools.product(
        range(len(DIAGNOSIS_FREQUENCIES)),
        range(len(MECHANISMS)),
        NOISE_PSD_SLOPES,
        SNR_LEVELS_DB,
        AMPLITUDES,
        FREQUENCY_MULTIPLIERS,
    ):
        mechanism = MECHANISMS[mechanism_id]
        tuple_text = _cell_tuple_text(
            diagnosis_label,
            mechanism,
            noise_slope,
            snr_db,
            amplitude,
            multiplier,
        )
        allocation_digest = _sha256_bytes(
            f"{GENERATOR_DOMAIN}|{tuple_text}".encode("utf-8")
        )
        cells.append(
            Cell(
                cell_id=cell_id,
                diagnosis_label=diagnosis_label,
                mechanism_id=mechanism_id,
                mechanism=mechanism,
                noise_psd_slope=noise_slope,
                snr_db=snr_db,
                amplitude=amplitude,
                frequency_multiplier=multiplier,
                tuple_text=tuple_text,
                allocation_sha256=allocation_digest,
            )
        )
        cell_id += 1

    allocated: dict[int, str] = {}
    axes = (
        set(NOISE_PSD_SLOPES),
        set(SNR_LEVELS_DB),
        set(AMPLITUDES),
        set(FREQUENCY_MULTIPLIERS),
    )
    for diagnosis_label in range(4):
        for mechanism_id in range(4):
            remaining = [
                cell
                for cell in cells
                if cell.diagnosis_label == diagnosis_label
                and cell.mechanism_id == mechanism_id
            ]
            for partition, target_count in PARTITION_TARGETS:
                uncovered = tuple(set(values) for values in axes)
                chosen: list[Cell] = []
                for _ in range(target_count):
                    if not remaining:
                        raise ProtocolValidationError("cell allocator exhausted early")

                    def rank(cell: Cell) -> tuple[int, str]:
                        coverage = sum(
                            value in missing
                            for value, missing in zip(cell.factor_values(), uncovered)
                        )
                        return (-coverage, cell.allocation_sha256)

                    selected = min(remaining, key=rank)
                    remaining.remove(selected)
                    chosen.append(selected)
                    for missing, value in zip(uncovered, selected.factor_values()):
                        missing.discard(value)
                if any(uncovered):
                    raise ProtocolValidationError(
                        f"{partition} lacks factor support for Y={diagnosis_label}, "
                        f"M={mechanism_id}"
                    )
                for cell in chosen:
                    allocated[cell.cell_id] = partition
            for cell in remaining:
                allocated[cell.cell_id] = "train"

    result = tuple(replace(cell, partition=allocated[cell.cell_id]) for cell in cells)
    _validate_cell_design(result)
    return result


def _validate_cell_design(cells: Sequence[Cell]) -> None:
    if len(cells) != 576 or len({cell.cell_id for cell in cells}) != 576:
        raise ProtocolValidationError("the design must contain 576 unique cells")
    if len({cell.allocation_sha256 for cell in cells}) != 576:
        raise ProtocolValidationError("cell allocation digests must be unique")

    expected_support = (
        set(NOISE_PSD_SLOPES),
        set(SNR_LEVELS_DB),
        set(AMPLITUDES),
        set(FREQUENCY_MULTIPLIERS),
    )
    for diagnosis_label in range(4):
        for mechanism_id in range(4):
            pair = [
                cell
                for cell in cells
                if cell.diagnosis_label == diagnosis_label
                and cell.mechanism_id == mechanism_id
            ]
            if len(pair) != 36:
                raise ProtocolValidationError("each Y x M pair must contain 36 cells")
            counts = Counter(cell.partition for cell in pair)
            if dict(counts) != PARTITION_CELL_COUNTS:
                raise ProtocolValidationError(
                    f"wrong partition counts for Y={diagnosis_label}, M={mechanism_id}: "
                    f"{dict(counts)}"
                )
            for partition in PARTITION_NAMES:
                subset = [cell for cell in pair if cell.partition == partition]
                observed = tuple(
                    {values[axis] for values in (cell.factor_values() for cell in subset)}
                    for axis in range(4)
                )
                if observed != expected_support:
                    raise ProtocolValidationError(
                        f"{partition} lacks frozen marginal support for "
                        f"Y={diagnosis_label}, M={mechanism_id}"
                    )


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))


def _rms_normalize(value: np.ndarray) -> np.ndarray:
    centered = np.asarray(value, dtype=np.float64) - float(np.mean(value))
    scale = _rms(centered)
    if not math.isfinite(scale) or scale <= 1.0e-12:
        raise ProtocolValidationError("renderer produced zero or non-finite RMS")
    return centered / scale


def _render_clean_base(cell: Cell, rng: np.random.Generator) -> tuple[np.ndarray, Mapping[str, float]]:
    n = np.arange(WINDOW_LENGTH, dtype=np.float64)
    r = cell.frequency_code
    audit: dict[str, float] = {}

    if cell.mechanism == "low_frequency":
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        modulation_phase = float(rng.uniform(0.0, 2.0 * np.pi))
        raw = (1.0 + 0.30 * np.sin(np.pi * (r / 4.0) * n + modulation_phase)) * np.sin(
            np.pi * r * n + phase
        )
    elif cell.mechanism == "harmonic":
        phases = rng.uniform(0.0, 2.0 * np.pi, size=4)
        raw = sum(
            (1.0 / harmonic) * np.sin(np.pi * harmonic * r * n + phases[harmonic - 1])
            for harmonic in range(1, 5)
        )
    elif cell.mechanism == "impulsive_envelope":
        period = int(math.floor((2.0 / r) + 0.5))
        offset = int(rng.integers(0, period))
        impulses = np.zeros(WINDOW_LENGTH, dtype=np.float64)
        impulses[offset::period] = 1.0
        kernel_index = np.arange(32, dtype=np.float64)
        kernel = np.exp(-kernel_index / 2.0) * np.sin(np.pi * 0.35 * kernel_index)
        padded_kernel = np.zeros(WINDOW_LENGTH, dtype=np.float64)
        padded_kernel[: kernel.shape[0]] = kernel
        raw = np.fft.ifft(np.fft.fft(impulses) * np.fft.fft(padded_kernel)).real
    elif cell.mechanism == "aperiodic_residual":
        frequencies = np.fft.rfftfreq(WINDOW_LENGTH) * 2.0
        magnitudes = 1.5 + np.exp(
            -0.5 * np.square((frequencies - (0.20 + (2.0 * r))) / 0.12)
        )
        phases = rng.uniform(0.0, 2.0 * np.pi, size=frequencies.shape[0])
        coefficients = np.zeros(frequencies.shape[0], dtype=np.complex128)
        coefficients[1:-1] = magnitudes[1:-1] * np.exp(1j * phases[1:-1])
        coefficients[-1] = magnitudes[-1] * (1.0 if phases[-1] < np.pi else -1.0)
        raw = np.fft.irfft(coefficients, n=WINDOW_LENGTH)
    else:  # pragma: no cover - Cell construction prevents this branch.
        raise ProtocolValidationError(f"unknown mechanism: {cell.mechanism}")

    return _rms_normalize(raw), audit


def _render_noise(length: int, slope: int, rng: np.random.Generator) -> np.ndarray:
    frequencies = np.fft.rfftfreq(length) * 2.0
    magnitudes = np.zeros_like(frequencies)
    magnitudes[1:] = np.power(frequencies[1:], slope / 2.0)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=frequencies.shape[0])
    coefficients = np.zeros(frequencies.shape[0], dtype=np.complex128)
    coefficients[1:-1] = magnitudes[1:-1] * np.exp(1j * phases[1:-1])
    coefficients[-1] = magnitudes[-1] * (1.0 if phases[-1] < np.pi else -1.0)
    return _rms_normalize(np.fft.irfft(coefficients, n=length))


def _analytic_envelope(signal: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft(signal)
    multiplier = np.zeros(signal.shape[0], dtype=np.float64)
    multiplier[0] = 1.0
    multiplier[signal.shape[0] // 2] = 1.0
    multiplier[1 : signal.shape[0] // 2] = 2.0
    return np.abs(np.fft.ifft(spectrum * multiplier))


def _energy_normalized_linear_autocorrelation(signal: np.ndarray, lag: int) -> float:
    left = signal[:-lag]
    right = signal[lag:]
    denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    if denominator <= 1.0e-15:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _fit_noise_log_power_slope(noise: np.ndarray) -> float:
    frequencies = (np.fft.rfftfreq(noise.shape[0]) * 2.0)[1:-1]
    power = np.square(np.abs(np.fft.rfft(noise))[1:-1])
    if np.any(power <= 0.0):
        raise ProtocolValidationError("isolated noise has a zero fitted-spectrum bin")
    design = np.column_stack((np.log(frequencies), np.ones_like(frequencies)))
    slope, _ = np.linalg.lstsq(design, np.log(power), rcond=None)[0]
    return float(slope)


def _mechanism_audit(
    cell: Cell,
    clean_base: np.ndarray,
    renderer_audit: Mapping[str, float],
) -> dict[str, float]:
    r = cell.frequency_code
    frequencies = np.fft.rfftfreq(WINDOW_LENGTH) * 2.0
    power = np.square(np.abs(np.fft.rfft(clean_base)))
    non_dc_total = float(np.sum(power[1:]))
    result = dict(renderer_audit)

    if cell.mechanism == "low_frequency":
        result["low_frequency_power_ratio"] = float(
            np.sum(power[(frequencies > 0.0) & (frequencies < LOW_POWER_CUTOFF)])
            / non_dc_total
        )
    elif cell.mechanism == "harmonic":
        included_bins: set[int] = set()
        maximum_location_error = 0
        expected_spacing_bins = WINDOW_LENGTH * r / 2.0
        search_radius = max(2, int(math.floor(expected_spacing_bins / 2.0)) - 1)
        for harmonic in range(1, 5):
            expected_bin = int(math.floor((WINDOW_LENGTH * harmonic * r / 2.0) + 0.5))
            lo = max(1, expected_bin - search_radius)
            hi = min(power.shape[0] - 1, expected_bin + search_radius)
            local_bin = lo + int(np.argmax(power[lo : hi + 1]))
            maximum_location_error = max(maximum_location_error, abs(local_bin - expected_bin))
            included_bins.update(
                range(
                    max(1, expected_bin - HARMONIC_PEAK_BIN_TOLERANCE),
                    min(
                        power.shape[0] - 1,
                        expected_bin + HARMONIC_PEAK_BIN_TOLERANCE,
                    )
                    + 1,
                )
            )
        result["harmonic_maximum_peak_location_error_bins"] = float(maximum_location_error)
        result["harmonic_combined_power_ratio"] = float(
            np.sum(power[sorted(included_bins)]) / non_dc_total
        )
    elif cell.mechanism == "impulsive_envelope":
        envelope = _analytic_envelope(clean_base)
        centered = envelope - float(np.mean(envelope))
        second_moment = float(np.mean(np.square(centered)))
        kurtosis = float(np.mean(np.power(centered, 4)) / (second_moment**2))
        period = int(math.floor((2.0 / r) + 0.5))
        candidate_lags = range(max(1, period - 3), min(256, period + 3) + 1)
        recurrence_lag = max(
            candidate_lags,
            key=lambda lag: _energy_normalized_linear_autocorrelation(centered, lag),
        )
        result["impulse_envelope_kurtosis"] = kurtosis
        result["impulse_recurrence_error_samples"] = float(abs(recurrence_lag - period))
    elif cell.mechanism == "aperiodic_residual":
        observed_magnitude = np.abs(np.fft.rfft(clean_base))
        frozen_magnitude = 1.5 + np.exp(
            -0.5 * np.square((frequencies - (0.20 + (2.0 * r))) / 0.12)
        )
        magnitude_scale = float(
            np.median(observed_magnitude[1:] / frozen_magnitude[1:])
        )
        observed_excess = observed_magnitude - (1.5 * magnitude_scale)
        excess_bin = int(np.argmax(observed_excess[1:]) + 1)
        autocorrelations = [
            abs(_energy_normalized_linear_autocorrelation(clean_base, lag))
            for lag in range(1, 257)
        ]
        result["residual_maximum_nonzero_autocorrelation"] = max(autocorrelations)
        result["residual_maximum_discrete_peak_share"] = float(
            np.max(power[1:]) / non_dc_total
        )
        result["residual_excess_peak_center_error"] = abs(
            float(frequencies[excess_bin]) - (0.20 + (2.0 * r))
        )
    return result


def render_sample(cell: Cell, draw: int) -> GeneratedSample:
    """Render one frozen sample without mutable global random state."""

    if draw < 0 or draw >= DRAWS_PER_CELL:
        raise ValueError(f"draw must be in [0, {DRAWS_PER_CELL - 1}]")
    renderer_seed = _seed("renderer", cell, draw)
    channel_seed = _seed("channel", cell, draw)
    noise_seeds = (_seed("noise-channel-1", cell, draw), _seed("noise-channel-2", cell, draw))

    base, renderer_audit = _render_clean_base(cell, np.random.default_rng(renderer_seed))
    clean = np.empty((WINDOW_LENGTH, CHANNELS), dtype=np.float64)
    clean[:, 0] = base * cell.amplitude
    channel_rng = np.random.default_rng(channel_seed)
    channel_gain = float(channel_rng.uniform(0.8, 1.2))
    channel_delay = int(channel_rng.integers(-3, 4))
    clean[:, 1] = np.roll(clean[:, 0] * channel_gain, channel_delay)

    noise = np.empty_like(clean)
    target_ratio = 10.0 ** (cell.snr_db / 20.0)
    for channel_index, noise_seed in enumerate(noise_seeds):
        unit_noise = _render_noise(
            WINDOW_LENGTH,
            cell.noise_psd_slope,
            np.random.default_rng(noise_seed),
        )
        noise[:, channel_index] = unit_noise * (_rms(clean[:, channel_index]) / target_ratio)

    final = np.asarray(clean + noise, dtype=np.float32)
    sample_index = (cell.cell_id * DRAWS_PER_CELL) + draw
    sample_id = SAMPLE_ID_BASE + sample_index
    filename = f"sample_{sample_index:06d}.npy"
    npy_content = _npy_bytes(final)
    signal_sha256 = _sha256_bytes(final.tobytes(order="C"))
    file_sha256 = _sha256_bytes(npy_content)

    mechanism_audit = _mechanism_audit(cell, base, renderer_audit)
    audit: dict[str, float] = {
        "amplitude_absolute_error": abs(_rms(clean[:, 0]) - cell.amplitude),
        "snr_channel_1_absolute_error_db": abs(
            20.0 * math.log10(_rms(clean[:, 0]) / _rms(noise[:, 0])) - cell.snr_db
        ),
        "snr_channel_2_absolute_error_db": abs(
            20.0 * math.log10(_rms(clean[:, 1]) / _rms(noise[:, 1])) - cell.snr_db
        ),
        "noise_channel_1_slope_absolute_error": abs(
            _fit_noise_log_power_slope(noise[:, 0]) - cell.noise_psd_slope
        ),
        "noise_channel_2_slope_absolute_error": abs(
            _fit_noise_log_power_slope(noise[:, 1]) - cell.noise_psd_slope
        ),
        **mechanism_audit,
    }
    metadata: dict[str, Any] = {
        "Id": sample_id,
        "Dataset_id": DATASET_ID,
        "Domain_id": DOMAIN_ID,
        "Label": cell.diagnosis_label,
        "Name": DATASET_NAME,
        "File": filename,
        "Split_group": cell.split_group,
        "Split_stratum": cell.split_stratum,
        "Partition": cell.partition,
        "Mechanism": cell.mechanism,
        "Nuisance_cell": cell.cell_id,
        "Draw": draw,
    }
    manifest: dict[str, Any] = {
        "sample_index": sample_index,
        "sample_id": sample_id,
        "file": f"raw/{DATASET_NAME}/{filename}",
        "file_sha256": file_sha256,
        "signal_sha256": signal_sha256,
        "dtype": "float32",
        "shape": [WINDOW_LENGTH, CHANNELS],
        "cell_id": cell.cell_id,
        "diagnosis_label": cell.diagnosis_label,
        "mechanism_id": cell.mechanism_id,
        "mechanism": cell.mechanism,
        "partition": cell.partition,
        "draw": draw,
        "derived_seeds": {
            "renderer": renderer_seed,
            "channel": channel_seed,
            "noise_channel_1": noise_seeds[0],
            "noise_channel_2": noise_seeds[1],
        },
        "channel_2_gain": channel_gain,
        "channel_2_circular_delay": channel_delay,
    }
    return GeneratedSample(array=final, metadata=metadata, manifest=manifest, audit=audit)


def _validate_and_summarize_samples(
    cells: Sequence[Cell], samples: Sequence[GeneratedSample]
) -> Mapping[str, Any]:
    expected_samples = 576 * DRAWS_PER_CELL
    if len(samples) != expected_samples:
        raise ProtocolValidationError(f"expected {expected_samples} samples")
    sample_ids = [int(sample.manifest["sample_id"]) for sample in samples]
    signal_hashes = [str(sample.manifest["signal_sha256"]) for sample in samples]
    if len(set(sample_ids)) != expected_samples:
        raise ProtocolValidationError("sample IDs must be unique")
    if len(set(signal_hashes)) != expected_samples:
        raise ProtocolValidationError("all generated signal hashes must be unique")

    extrema: dict[str, float] = {
        "maximum_amplitude_absolute_error": 0.0,
        "maximum_snr_absolute_error_db": 0.0,
        "maximum_noise_slope_absolute_error": 0.0,
        "minimum_low_frequency_power_ratio": math.inf,
        "maximum_harmonic_peak_location_error_bins": 0.0,
        "minimum_harmonic_combined_power_ratio": math.inf,
        "maximum_impulse_recurrence_error_samples": 0.0,
        "minimum_impulse_envelope_kurtosis": math.inf,
        "maximum_residual_nonzero_autocorrelation": 0.0,
        "maximum_residual_discrete_peak_share": 0.0,
        "maximum_residual_excess_peak_center_error": 0.0,
    }
    for sample in samples:
        audit = sample.audit
        extrema["maximum_amplitude_absolute_error"] = max(
            extrema["maximum_amplitude_absolute_error"], audit["amplitude_absolute_error"]
        )
        extrema["maximum_snr_absolute_error_db"] = max(
            extrema["maximum_snr_absolute_error_db"],
            audit["snr_channel_1_absolute_error_db"],
            audit["snr_channel_2_absolute_error_db"],
        )
        extrema["maximum_noise_slope_absolute_error"] = max(
            extrema["maximum_noise_slope_absolute_error"],
            audit["noise_channel_1_slope_absolute_error"],
            audit["noise_channel_2_slope_absolute_error"],
        )
        mechanism = str(sample.manifest["mechanism"])
        if mechanism == "low_frequency":
            extrema["minimum_low_frequency_power_ratio"] = min(
                extrema["minimum_low_frequency_power_ratio"],
                audit["low_frequency_power_ratio"],
            )
        elif mechanism == "harmonic":
            extrema["maximum_harmonic_peak_location_error_bins"] = max(
                extrema["maximum_harmonic_peak_location_error_bins"],
                audit["harmonic_maximum_peak_location_error_bins"],
            )
            extrema["minimum_harmonic_combined_power_ratio"] = min(
                extrema["minimum_harmonic_combined_power_ratio"],
                audit["harmonic_combined_power_ratio"],
            )
        elif mechanism == "impulsive_envelope":
            extrema["maximum_impulse_recurrence_error_samples"] = max(
                extrema["maximum_impulse_recurrence_error_samples"],
                audit["impulse_recurrence_error_samples"],
            )
            extrema["minimum_impulse_envelope_kurtosis"] = min(
                extrema["minimum_impulse_envelope_kurtosis"],
                audit["impulse_envelope_kurtosis"],
            )
        elif mechanism == "aperiodic_residual":
            extrema["maximum_residual_nonzero_autocorrelation"] = max(
                extrema["maximum_residual_nonzero_autocorrelation"],
                audit["residual_maximum_nonzero_autocorrelation"],
            )
            extrema["maximum_residual_discrete_peak_share"] = max(
                extrema["maximum_residual_discrete_peak_share"],
                audit["residual_maximum_discrete_peak_share"],
            )
            extrema["maximum_residual_excess_peak_center_error"] = max(
                extrema["maximum_residual_excess_peak_center_error"],
                audit["residual_excess_peak_center_error"],
            )

    failures = {
        "amplitude": extrema["maximum_amplitude_absolute_error"] > AMPLITUDE_ABSOLUTE_TOLERANCE,
        "snr": extrema["maximum_snr_absolute_error_db"] > SNR_DB_ABSOLUTE_TOLERANCE,
        "noise_slope": extrema["maximum_noise_slope_absolute_error"] > NOISE_SLOPE_ABSOLUTE_TOLERANCE,
        "low_power": extrema["minimum_low_frequency_power_ratio"] < LOW_POWER_RATIO_MINIMUM,
        "harmonic_location": extrema["maximum_harmonic_peak_location_error_bins"] > HARMONIC_PEAK_BIN_TOLERANCE,
        "harmonic_power": extrema["minimum_harmonic_combined_power_ratio"] < HARMONIC_COMBINED_POWER_MINIMUM,
        "impulse_recurrence": extrema["maximum_impulse_recurrence_error_samples"] > IMPULSE_RECURRENCE_TOLERANCE,
        "impulse_kurtosis": extrema["minimum_impulse_envelope_kurtosis"] < IMPULSE_KURTOSIS_MINIMUM,
        "residual_autocorrelation": extrema["maximum_residual_nonzero_autocorrelation"] > RESIDUAL_MAX_AUTOCORRELATION,
        "residual_peak_share": extrema["maximum_residual_discrete_peak_share"] > RESIDUAL_MAX_PEAK_SHARE,
        "residual_peak_center": extrema["maximum_residual_excess_peak_center_error"] > RESIDUAL_EXCESS_CENTER_TOLERANCE,
    }
    failed_names = sorted(name for name, failed in failures.items() if failed)
    if failed_names:
        raise ProtocolValidationError(
            "generator validation thresholds failed: " + ", ".join(failed_names)
        )

    per_partition = Counter(str(sample.manifest["partition"]) for sample in samples)
    expected_partition_counts = {
        partition: count * 16 * DRAWS_PER_CELL
        for partition, count in PARTITION_CELL_COUNTS.items()
    }
    if dict(per_partition) != expected_partition_counts:
        raise ProtocolValidationError(
            f"wrong sample partition counts: {dict(per_partition)}"
        )
    for partition in PARTITION_NAMES:
        joint_counts = Counter(
            (
                int(sample.manifest["diagnosis_label"]),
                int(sample.manifest["mechanism_id"]),
            )
            for sample in samples
            if sample.manifest["partition"] == partition
        )
        if len(joint_counts) != 16 or len(set(joint_counts.values())) != 1:
            raise ProtocolValidationError(
                f"{partition} lacks exact joint diagnosis/mechanism balance"
            )

    dataset_digest = hashlib.sha256()
    for sample in samples:
        dataset_digest.update(str(sample.manifest["sample_id"]).encode("ascii"))
        dataset_digest.update(b"\0")
        dataset_digest.update(str(sample.manifest["signal_sha256"]).encode("ascii"))
        dataset_digest.update(b"\n")
    return {
        "status": "passed",
        "cells": len(cells),
        "samples": len(samples),
        "unique_signal_hashes": len(set(signal_hashes)),
        "partition_sample_counts": {
            partition: per_partition[partition] for partition in PARTITION_NAMES
        },
        "dataset_signal_merkle_sha256": dataset_digest.hexdigest(),
        "observed_extrema": extrema,
        "thresholds": {
            "amplitude_absolute": AMPLITUDE_ABSOLUTE_TOLERANCE,
            "snr_db_absolute": SNR_DB_ABSOLUTE_TOLERANCE,
            "noise_slope_absolute": NOISE_SLOPE_ABSOLUTE_TOLERANCE,
            "low_frequency_power_ratio_minimum": LOW_POWER_RATIO_MINIMUM,
            "harmonic_peak_location_fft_bins": HARMONIC_PEAK_BIN_TOLERANCE,
            "harmonic_combined_power_minimum": HARMONIC_COMBINED_POWER_MINIMUM,
            "impulse_recurrence_error_samples": IMPULSE_RECURRENCE_TOLERANCE,
            "impulse_envelope_kurtosis_minimum": IMPULSE_KURTOSIS_MINIMUM,
            "residual_maximum_nonzero_autocorrelation": RESIDUAL_MAX_AUTOCORRELATION,
            "residual_maximum_discrete_peak_share": RESIDUAL_MAX_PEAK_SHARE,
            "residual_excess_peak_center_error": RESIDUAL_EXCESS_CENTER_TOLERANCE,
        },
    }


def build_protocol_dataset() -> BuildResult:
    """Build and validate all 4,608 samples in memory without filesystem writes."""

    cells = enumerate_cells()
    samples = tuple(
        render_sample(cell, draw)
        for cell in cells
        for draw in range(DRAWS_PER_CELL)
    )
    summary = _validate_and_summarize_samples(cells, samples)
    return BuildResult(cells=cells, samples=samples, validation_summary=summary)


def _metadata_bytes(samples: Sequence[GeneratedSample]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=METADATA_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for sample in samples:
        writer.writerow(sample.metadata)
    return stream.getvalue().encode("utf-8")


def _cell_records(cells: Sequence[Cell]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for cell in cells:
        record = asdict(cell)
        record["split_group"] = cell.split_group
        record["split_stratum"] = cell.split_stratum
        record["frequency_code_nyquist"] = cell.frequency_code
        records.append(record)
    return records


def _partition_record(
    partition: str,
    cells: Sequence[Cell],
    samples: Sequence[GeneratedSample],
) -> dict[str, Any]:
    selected_cells = [cell for cell in cells if cell.partition == partition]
    selected_samples = [
        sample for sample in samples if sample.manifest["partition"] == partition
    ]
    ids = [int(sample.manifest["sample_id"]) for sample in selected_samples]
    groups = [cell.split_group for cell in selected_cells]
    label_support = Counter(int(sample.manifest["diagnosis_label"]) for sample in selected_samples)
    mechanism_support = Counter(int(sample.manifest["mechanism_id"]) for sample in selected_samples)
    stratum_support = Counter(cell.split_stratum for cell in selected_cells)
    payload: dict[str, Any] = {
        "name": partition,
        "ids": ids,
        "groups": groups,
        "cell_ids": [cell.cell_id for cell in selected_cells],
        "sample_count": len(ids),
        "group_count": len(groups),
        "label_support": {str(key): label_support[key] for key in sorted(label_support)},
        "mechanism_support": {
            str(key): mechanism_support[key] for key in sorted(mechanism_support)
        },
        "stratum_group_support": {
            key: stratum_support[key] for key in sorted(stratum_support)
        },
    }
    payload["partition_sha256"] = _sha256_bytes(_canonical_json_bytes(payload))
    return payload


def _partition_manifest(
    cells: Sequence[Cell],
    samples: Sequence[GeneratedSample],
    metadata_sha256: str,
) -> dict[str, Any]:
    partitions = {
        name: _partition_record(name, cells, samples) for name in PARTITION_NAMES
    }
    split_mapping = {
        "train": "train",
        "val": "optimization_validation",
        "test": "intervention",
    }
    splits = {
        split: {
            **partitions[partition],
            "split_name": split,
            "source_partition": partition,
        }
        for split, partition in split_mapping.items()
    }
    splits["identification"] = {
        **partitions["identification"],
        "split_name": "identification",
        "source_partition": "identification",
        "runtime_training_split": False,
    }
    return {
        "schema_id": PARTITION_SCHEMA_ID,
        "schema_version": PARTITION_SCHEMA_VERSION,
        "strategy": "grouped_metadata",
        "task_type": "Default_task",
        "seed": MASTER_SEED,
        "group_key": "Split_group",
        "stratify_key": "Split_stratum",
        "test_policy": "partition",
        "metadata_file": "metadata.csv",
        "metadata_file_sha256": metadata_sha256,
        "dataset_id": DATASET_ID,
        "domain_id": DOMAIN_ID,
        "fractions": {
            name: PARTITION_CELL_COUNTS[name] / 36.0 for name in PARTITION_NAMES
        },
        "partition_cell_counts_per_diagnosis_mechanism_pair": {
            name: PARTITION_CELL_COUNTS[name] for name in PARTITION_NAMES
        },
        "partition_map": split_mapping,
        "offline_partition": "identification",
        "normalization": {
            "data_layer_method": "none",
            "model_input_adapter": "per_channel_within_window_standardization_plus_prestandardization_rms",
            "fit_scope": "current_window_only",
        },
        "partitions": partitions,
        "splits": splits,
        "runtime_random_resplit_forbidden": True,
    }


def _generator_manifest(
    result: BuildResult,
    file_hashes: Mapping[str, str],
) -> dict[str, Any]:
    source_path = Path(__file__).resolve()
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "generator_domain": GENERATOR_DOMAIN,
        "master_seed": MASTER_SEED,
        "dataset_id": DATASET_ID,
        "domain_id": DOMAIN_ID,
        "dataset_name": DATASET_NAME,
        "source": {
            "path": "scripts/p04/generate_synthetic.py",
            "sha256": _sha256_file(source_path),
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "layout": {
            "metadata": "metadata.csv",
            "sample_files": f"raw/{DATASET_NAME}/sample_XXXXXX.npy",
            "sample_dtype": "float32",
            "sample_shape": [WINDOW_LENGTH, CHANNELS],
            "cell_manifest": "cell_manifest.jsonl",
            "sample_manifest": "sample_manifest.jsonl",
            "partition_manifest": "partition_manifest.json",
        },
        "model_visibility": {
            "array_and_target_label_only": True,
            "mechanism_and_nuisance_fields_are_audit_only": True,
        },
        "design": {
            "diagnosis_frequency_codes_nyquist": list(DIAGNOSIS_FREQUENCIES),
            "mechanisms": list(MECHANISMS),
            "noise_psd_slopes": list(NOISE_PSD_SLOPES),
            "snr_db": list(SNR_LEVELS_DB),
            "amplitudes": list(AMPLITUDES),
            "frequency_multipliers": list(FREQUENCY_MULTIPLIERS),
            "cells_per_diagnosis_mechanism_pair": 36,
            "total_cells": 576,
            "draws_per_cell": DRAWS_PER_CELL,
            "total_samples": 4608,
            "partition_allocation_order": [
                "intervention",
                "identification",
                "optimization_validation",
                "train_remaining",
            ],
            "partition_tie_break": f"SHA-256({GENERATOR_DOMAIN}|<tuple>) ascending",
        },
        "renderers": {
            "low_frequency": "(1+0.30*sin(pi*(r/4)*n+phi_a))*sin(pi*r*n+phi)",
            "harmonic": "sum(k^-1*sin(pi*k*r*n+phi_k), k=1..4)",
            "impulsive_envelope": "period=round(2/r); circular_conv(impulses, exp(-j/2)*sin(pi*0.35*j), j=0..31)",
            "aperiodic_residual": "rfft magnitude=1.5+exp(-0.5*((f-(0.20+2*r))/0.12)^2), independent phases",
            "additive_noise": "rfft magnitude=f^(noise_psd_slope/2), independent phases per channel, exact RMS SNR rescaling",
        },
        "per_sample_seed_derivation": (
            "uint64_be(SHA-256(P04-SYN-v1|master=240401|<tuple>|"
            "draw=<draw>|component=<component>)[0:8])"
        ),
        "content_hashes": dict(file_hashes),
        "validation": result.validation_summary,
    }


def _artifact_hash_ledger(root: Path) -> bytes:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_hashes.sha256"
    )
    lines = [f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in paths]
    return "".join(lines).encode("utf-8")


def _materialize(root: Path, result: BuildResult) -> None:
    raw_directory = root / "raw" / DATASET_NAME
    raw_directory.mkdir(parents=True, exist_ok=False)
    for sample in result.samples:
        relative_path = Path(str(sample.manifest["file"]))
        _write_exclusive(root / relative_path, _npy_bytes(sample.array))

    metadata_content = _metadata_bytes(result.samples)
    cell_content = _jsonl_bytes(_cell_records(result.cells))
    sample_content = _jsonl_bytes(sample.manifest for sample in result.samples)
    _write_exclusive(root / "metadata.csv", metadata_content)
    _write_exclusive(root / "cell_manifest.jsonl", cell_content)
    _write_exclusive(root / "sample_manifest.jsonl", sample_content)

    partition_manifest = _partition_manifest(
        result.cells,
        result.samples,
        _sha256_bytes(metadata_content),
    )
    partition_content = _canonical_json_bytes(partition_manifest)
    _write_exclusive(root / "partition_manifest.json", partition_content)
    key_hashes = {
        "metadata_sha256": _sha256_bytes(metadata_content),
        "cell_manifest_sha256": _sha256_bytes(cell_content),
        "sample_manifest_sha256": _sha256_bytes(sample_content),
        "partition_manifest_sha256": _sha256_bytes(partition_content),
    }
    generator_content = _canonical_json_bytes(_generator_manifest(result, key_hashes))
    _write_exclusive(root / "generator_manifest.json", generator_content)
    _write_exclusive(root / "artifact_hashes.sha256", _artifact_hash_ledger(root))


def _parse_hash_ledger(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "  " not in line:
            raise ProtocolValidationError(f"invalid hash ledger line {line_number}")
        digest, relative_path = line.split("  ", 1)
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ProtocolValidationError(f"invalid SHA-256 on ledger line {line_number}")
        if not relative_path or relative_path in records:
            raise ProtocolValidationError(f"duplicate/empty path on ledger line {line_number}")
        records[relative_path] = digest
    return records


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[Any]:
    records: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ProtocolValidationError(
                    f"invalid JSONL at {path}:{line_number}: {error}"
                ) from error
    return records


def validate_dataset(root: Path) -> Mapping[str, Any]:
    """Validate hashes, manifests, arrays, partitions, and regenerated invariants."""

    root = root.resolve()
    if not root.is_dir():
        raise ProtocolValidationError(f"dataset directory does not exist: {root}")
    required = {
        "metadata.csv",
        "cell_manifest.jsonl",
        "sample_manifest.jsonl",
        "partition_manifest.json",
        "generator_manifest.json",
        "artifact_hashes.sha256",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise ProtocolValidationError("missing required artifacts: " + ", ".join(missing))

    ledger = _parse_hash_ledger(root / "artifact_hashes.sha256")
    observed_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_hashes.sha256"
    }
    if set(ledger) != observed_files:
        raise ProtocolValidationError("artifact hash ledger inventory does not match files")
    for relative_path, expected_digest in ledger.items():
        if _sha256_file(root / relative_path) != expected_digest:
            raise ProtocolValidationError(f"artifact hash mismatch: {relative_path}")

    expected = build_protocol_dataset()
    expected_metadata = _metadata_bytes(expected.samples)
    expected_cells = _jsonl_bytes(_cell_records(expected.cells))
    expected_samples = _jsonl_bytes(sample.manifest for sample in expected.samples)
    if (root / "metadata.csv").read_bytes() != expected_metadata:
        raise ProtocolValidationError("metadata.csv differs from deterministic protocol bytes")
    if (root / "cell_manifest.jsonl").read_bytes() != expected_cells:
        raise ProtocolValidationError("cell manifest differs from deterministic protocol bytes")
    if (root / "sample_manifest.jsonl").read_bytes() != expected_samples:
        raise ProtocolValidationError("sample manifest differs from deterministic protocol bytes")

    expected_partition = _partition_manifest(
        expected.cells, expected.samples, _sha256_bytes(expected_metadata)
    )
    if _read_json(root / "partition_manifest.json") != expected_partition:
        raise ProtocolValidationError("partition manifest differs from frozen allocation")
    key_hashes = {
        "metadata_sha256": _sha256_bytes(expected_metadata),
        "cell_manifest_sha256": _sha256_bytes(expected_cells),
        "sample_manifest_sha256": _sha256_bytes(expected_samples),
        "partition_manifest_sha256": _sha256_bytes(
            _canonical_json_bytes(expected_partition)
        ),
    }
    expected_generator = _generator_manifest(expected, key_hashes)
    if _read_json(root / "generator_manifest.json") != expected_generator:
        raise ProtocolValidationError("generator manifest or source hash is not frozen")

    disk_sample_manifest = _read_jsonl(root / "sample_manifest.jsonl")
    if len(disk_sample_manifest) != len(expected.samples):
        raise ProtocolValidationError("wrong number of sample manifest records")
    for generated, record in zip(expected.samples, disk_sample_manifest):
        sample_path = root / str(record["file"])
        array = np.load(sample_path, allow_pickle=False)
        if array.dtype != np.float32 or array.shape != (WINDOW_LENGTH, CHANNELS):
            raise ProtocolValidationError(f"wrong sample array contract: {record['file']}")
        if not np.array_equal(array, generated.array):
            raise ProtocolValidationError(f"sample bytes differ from deterministic render: {record['file']}")

    return {
        "status": "passed",
        "root": str(root),
        "generator_manifest_sha256": ledger["generator_manifest.json"],
        "partition_manifest_sha256": ledger["partition_manifest.json"],
        "metadata_file_sha256": ledger["metadata.csv"],
        "sample_count": len(expected.samples),
        "validation": expected.validation_summary,
    }


def generate_dataset(output: Path) -> Mapping[str, Any]:
    """Generate a new immutable dataset directory and return its frozen hashes."""

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    try:
        result = build_protocol_dataset()
        _materialize(staging, result)
        validation = validate_dataset(staging)
        os.rename(staging, output)
        return {**validation, "root": str(output)}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def dry_run() -> Mapping[str, Any]:
    """Generate and validate all samples in memory without writing artifacts."""

    result = build_protocol_dataset()
    return {
        "status": "passed",
        "mode": "dry-run",
        "writes_performed": False,
        "generator_source_sha256": _sha256_file(Path(__file__).resolve()),
        "validation": result.validation_summary,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("dry-run", help="render and validate all 4,608 samples in memory")
    generate_parser = subparsers.add_parser(
        "generate", help="create a new immutable synthetic dataset directory"
    )
    generate_parser.add_argument("--output", required=True, type=Path)
    validate_parser = subparsers.add_parser(
        "validate", help="validate an existing frozen synthetic dataset"
    )
    validate_parser.add_argument("--input", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "dry-run":
        result = dry_run()
    elif args.command == "generate":
        result = generate_dataset(args.output)
    elif args.command == "validate":
        result = validate_dataset(args.input)
    else:  # pragma: no cover - argparse enforces a known command.
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
