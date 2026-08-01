"""Deterministic analytic signal bank for the frozen P08 E1 protocol.

This module has no filesystem side effects.  It generates one noisy 20 ms
underlying signal at 200 kHz and derives all rate copies from that signal, so a
rate copy can never cross the underlying signal's train/validation/test split.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
import json
import math
from numbers import Integral
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray
import scipy
from scipy.signal import resample_poly


SplitName = Literal["train", "validation", "test"]

PROTOCOL_ID: Final = "P08-LOSO-v1.1"
GENERATOR_VERSION: Final = "p08-e1-analytic-bank-v1"
GENERATOR_SEED: Final = 20260801
CLASS_IDS: Final = (0, 1, 2, 3)
SIGNALS_PER_CLASS: Final = 256
SPLIT_COUNTS: Final = {"train": 154, "validation": 51, "test": 51}
NATIVE_RATE_HZ: Final = 200_000
DURATION_S: Final = 0.02
NATIVE_POINTS: Final = 4_000
EVALUATION_RATES_HZ: Final = (12_000, 20_480, 25_600, 48_000, 50_000, 200_000)
SHARED_SINUSOIDS_HZ: Final = (250, 600, 1_200, 3_000)
IMPULSE_FREQUENCY_HZ: Final = {0: None, 1: 97, 2: 73, 3: 131}
PRIVATE_RESONANCE_HZ: Final = {1: 8_000, 2: 18_000, 3: 40_000}
RESONANCE_DECAY_PER_SECOND: Final = 1_200.0
KAISER_BETA: Final = 8.6
SOURCE_SHARED_BAND_HZ: Final = 6_000.0

_FLOAT64_LE = np.dtype("<f8")
_SPLIT_NAMES: Final[tuple[SplitName, ...]] = ("train", "validation", "test")


def _validate_class_id(class_id: int) -> int:
    if isinstance(class_id, bool) or not isinstance(class_id, Integral):
        raise TypeError("class_id must be an integer")
    value = int(class_id)
    if value not in CLASS_IDS:
        raise ValueError(f"class_id must be one of {CLASS_IDS}, got {value}")
    return value


def _validate_underlying_id(underlying_id: int) -> int:
    if isinstance(underlying_id, bool) or not isinstance(underlying_id, Integral):
        raise TypeError("underlying_id must be an integer")
    value = int(underlying_id)
    if not 0 <= value < SIGNALS_PER_CLASS:
        raise ValueError(
            f"underlying_id must be in [0, {SIGNALS_PER_CLASS}), got {value}"
        )
    return value


def _validate_rate(rate_hz: int) -> int:
    if isinstance(rate_hz, bool) or not isinstance(rate_hz, Integral):
        raise TypeError("target_rate_hz must be an integer")
    value = int(rate_hz)
    if value not in EVALUATION_RATES_HZ:
        raise ValueError(
            f"target_rate_hz must be one of {EVALUATION_RATES_HZ}, got {value}"
        )
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def canonical_json_sha256(value: Any) -> str:
    """Return the SHA-256 of canonical, finite JSON."""

    return sha256(_canonical_json_bytes(value)).hexdigest()


def _framed_update(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    digest.update(payload)


def _readonly_float64(samples: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    # Always copy: hashing or wrapping a caller-owned array must not make that
    # array read-only as an accidental side effect.
    result = np.array(samples, dtype=_FLOAT64_LE, order="C", copy=True)
    result.setflags(write=False)
    return result


def samples_sha256(samples: NDArray[np.floating[Any]]) -> str:
    """Hash a one-dimensional signal with an explicit dtype/shape frame."""

    array = _readonly_float64(samples)
    if array.ndim != 1:
        raise ValueError(f"samples must be one-dimensional, got shape {array.shape}")
    digest = sha256()
    _framed_update(
        digest,
        _canonical_json_bytes({"dtype": "<f8", "shape": [int(array.size)]}),
    )
    _framed_update(digest, array.tobytes(order="C"))
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _split_tuple(class_id: int) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    class_id = _validate_class_id(class_id)
    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([GENERATOR_SEED, class_id, 999]))
    )
    shuffled = tuple(int(value) for value in rng.permutation(SIGNALS_PER_CLASS))
    train_end = SPLIT_COUNTS["train"]
    validation_end = train_end + SPLIT_COUNTS["validation"]
    return (
        shuffled[:train_end],
        shuffled[train_end:validation_end],
        shuffled[validation_end:],
    )


def split_underlying_ids(class_id: int) -> dict[SplitName, tuple[int, ...]]:
    """Return the frozen seeded split for one class before rate copies exist."""

    train, validation, test = _split_tuple(class_id)
    return {"train": train, "validation": validation, "test": test}


def split_for_underlying(class_id: int, underlying_id: int) -> SplitName:
    """Resolve one underlying signal to exactly one frozen split."""

    class_id = _validate_class_id(class_id)
    underlying_id = _validate_underlying_id(underlying_id)
    for split_name, ids in split_underlying_ids(class_id).items():
        if underlying_id in ids:
            return split_name
    raise RuntimeError("the frozen split does not cover every underlying ID")


@dataclass(frozen=True, slots=True)
class NativeSignal:
    """One underlying noisy signal and its audit-visible random draws."""

    class_id: int
    underlying_id: int
    split: SplitName
    sample_rate_hz: int
    samples: NDArray[np.float64]
    clean_samples: NDArray[np.float64]
    shared_amplitudes: tuple[float, ...]
    shared_phases_rad: tuple[float, ...]
    fault_amplitude: float | None
    first_impulse_time_s: float | None
    impulse_times_s: tuple[float, ...]
    impulse_indices: tuple[int, ...]
    snr_db: float

    @property
    def clean_sha256(self) -> str:
        return samples_sha256(self.clean_samples)

    @property
    def signal_sha256(self) -> str:
        return samples_sha256(self.samples)

    def audit_metadata(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "underlying_id": self.underlying_id,
            "split": self.split,
            "sample_rate_hz": self.sample_rate_hz,
            "sample_count": int(self.samples.size),
            "shared_amplitudes": list(self.shared_amplitudes),
            "shared_phases_rad": list(self.shared_phases_rad),
            "fault_amplitude": self.fault_amplitude,
            "first_impulse_time_s": self.first_impulse_time_s,
            "impulse_times_s": list(self.impulse_times_s),
            "impulse_indices": list(self.impulse_indices),
            "snr_db": self.snr_db,
            "clean_sha256": self.clean_sha256,
            "signal_sha256": self.signal_sha256,
        }


@dataclass(frozen=True, slots=True)
class RateCopy:
    """One deterministic rate copy of an underlying E1 signal."""

    class_id: int
    underlying_id: int
    split: SplitName
    source_rate_hz: int
    sample_rate_hz: int
    resample_up: int
    resample_down: int
    samples: NDArray[np.float64]
    native_signal_sha256: str

    @property
    def sample_sha256(self) -> str:
        return samples_sha256(self.samples)

    def audit_metadata(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "underlying_id": self.underlying_id,
            "split": self.split,
            "source_rate_hz": self.source_rate_hz,
            "sample_rate_hz": self.sample_rate_hz,
            "sample_count": int(self.samples.size),
            "resample_up": self.resample_up,
            "resample_down": self.resample_down,
            "native_signal_sha256": self.native_signal_sha256,
            "sample_sha256": self.sample_sha256,
        }

    @property
    def record_sha256(self) -> str:
        return canonical_json_sha256(self.audit_metadata())


def _impulse_schedule(
    rng: np.random.Generator,
    class_id: int,
) -> tuple[float | None, tuple[float, ...], tuple[int, ...]]:
    impulse_frequency_hz = IMPULSE_FREQUENCY_HZ[class_id]
    if impulse_frequency_hz is None:
        return None, (), ()

    nominal_period_s = 1.0 / float(impulse_frequency_hz)
    first_time_s = float(rng.uniform(0.0, nominal_period_s))
    event_times: list[float] = []
    event_time_s = first_time_s
    while event_time_s < DURATION_S:
        event_times.append(event_time_s)
        if class_id == 2:
            interval_s = -1.0
            while interval_s <= 0.0:
                epsilon = float(rng.normal(0.0, 0.05))
                interval_s = nominal_period_s * (1.0 + epsilon)
        else:
            interval_s = nominal_period_s
        event_time_s += interval_s

    # Each continuous event is assigned to the sample interval containing it.
    indices = tuple(int(math.floor(time_s * NATIVE_RATE_HZ)) for time_s in event_times)
    if len(set(indices)) != len(indices):
        raise RuntimeError("two impulse events collapsed onto the same native sample")
    if any(index < 0 or index >= NATIVE_POINTS for index in indices):
        raise RuntimeError("an impulse event lies outside the native signal")
    return first_time_s, tuple(event_times), indices


def generate_native_signal(class_id: int, underlying_id: int) -> NativeSignal:
    """Generate one frozen 200 kHz, 20 ms E1 underlying signal.

    Draw order is fixed: shared amplitudes, shared phases, optional fault
    amplitude and schedule, SNR, then iid standard-normal base draws.  The
    finite base vector is centered and RMS-scaled so its realized power
    satisfies the drawn SNR; the transformed vector is not claimed to remain
    iid Gaussian.
    """

    class_id = _validate_class_id(class_id)
    underlying_id = _validate_underlying_id(underlying_id)
    rng = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence([GENERATOR_SEED, class_id, underlying_id])
        )
    )

    amplitudes_array = rng.uniform(0.2, 1.0, size=len(SHARED_SINUSOIDS_HZ))
    phases_array = rng.uniform(0.0, 2.0 * np.pi, size=len(SHARED_SINUSOIDS_HZ))
    time_s = np.arange(NATIVE_POINTS, dtype=np.float64) / float(NATIVE_RATE_HZ)
    shared = np.zeros(NATIVE_POINTS, dtype=np.float64)
    for frequency_hz, amplitude, phase_rad in zip(
        SHARED_SINUSOIDS_HZ, amplitudes_array, phases_array, strict=True
    ):
        shared += amplitude * np.sin(2.0 * np.pi * frequency_hz * time_s + phase_rad)

    fault_amplitude: float | None = None
    first_impulse_time_s: float | None = None
    impulse_times_s: tuple[float, ...] = ()
    impulse_indices: tuple[int, ...] = ()
    fault = np.zeros(NATIVE_POINTS, dtype=np.float64)
    if class_id != 0:
        fault_amplitude = float(rng.uniform(0.5, 1.5))
        first_impulse_time_s, impulse_times_s, impulse_indices = _impulse_schedule(
            rng, class_id
        )
        kernel_index = np.arange(NATIVE_POINTS, dtype=np.float64)
        resonance_hz = PRIVATE_RESONANCE_HZ[class_id]
        kernel = np.exp(
            -RESONANCE_DECAY_PER_SECOND * kernel_index / float(NATIVE_RATE_HZ)
        ) * np.sin(2.0 * np.pi * resonance_hz * kernel_index / float(NATIVE_RATE_HZ))
        # Sparse shift-add is exactly the declared full impulse-train convolution
        # followed by truncation to the first NATIVE_POINTS samples.
        for impulse_index in impulse_indices:
            available = NATIVE_POINTS - impulse_index
            fault[impulse_index:] += fault_amplitude * kernel[:available]

    unscaled_clean = shared + fault
    clean_rms = float(np.sqrt(np.mean(np.square(unscaled_clean), dtype=np.float64)))
    if not math.isfinite(clean_rms) or clean_rms <= 0.0:
        raise RuntimeError("generated clean signal has invalid RMS")
    clean = unscaled_clean / clean_rms

    snr_db = float(rng.uniform(10.0, 30.0))
    noise = rng.normal(0.0, 1.0, size=NATIVE_POINTS).astype(np.float64, copy=False)
    noise -= float(noise.mean(dtype=np.float64))
    noise_rms = float(np.sqrt(np.mean(np.square(noise), dtype=np.float64)))
    if not math.isfinite(noise_rms) or noise_rms <= 0.0:
        raise RuntimeError("generated Gaussian base vector has invalid RMS")
    target_noise_rms = 10.0 ** (-snr_db / 20.0)
    noise *= target_noise_rms / noise_rms
    noisy = clean + noise

    return NativeSignal(
        class_id=class_id,
        underlying_id=underlying_id,
        split=split_for_underlying(class_id, underlying_id),
        sample_rate_hz=NATIVE_RATE_HZ,
        samples=_readonly_float64(noisy),
        clean_samples=_readonly_float64(clean),
        shared_amplitudes=tuple(float(value) for value in amplitudes_array),
        shared_phases_rad=tuple(float(value) for value in phases_array),
        fault_amplitude=fault_amplitude,
        first_impulse_time_s=first_impulse_time_s,
        impulse_times_s=impulse_times_s,
        impulse_indices=impulse_indices,
        snr_db=snr_db,
    )


def expected_rate_points(target_rate_hz: int) -> int:
    """Return ``floor(0.02 * rate + 0.5)`` without float rounding."""

    target_rate_hz = _validate_rate(target_rate_hz)
    exact_points = Fraction(target_rate_hz, 50)
    return (2 * exact_points.numerator + exact_points.denominator) // (
        2 * exact_points.denominator
    )


def make_rate_copy(native: NativeSignal, target_rate_hz: int) -> RateCopy:
    """Derive one rate copy with the protocol's exact polyphase conversion."""

    if native.sample_rate_hz != NATIVE_RATE_HZ or native.samples.size != NATIVE_POINTS:
        raise ValueError("native signal does not satisfy the frozen E1 shape/rate")
    target_rate_hz = _validate_rate(target_rate_hz)
    ratio = Fraction(target_rate_hz, NATIVE_RATE_HZ)
    converted = resample_poly(
        native.samples,
        up=ratio.numerator,
        down=ratio.denominator,
        window=("kaiser", KAISER_BETA),
        padtype="line",
    )
    expected_points = expected_rate_points(target_rate_hz)
    if converted.ndim != 1 or converted.size != expected_points:
        raise RuntimeError(
            "resample_poly returned an unexpected shape: "
            f"rate={target_rate_hz}, expected={expected_points}, got={converted.shape}"
        )
    if not np.isfinite(converted).all():
        raise RuntimeError("resample_poly produced a non-finite value")
    return RateCopy(
        class_id=native.class_id,
        underlying_id=native.underlying_id,
        split=native.split,
        source_rate_hz=NATIVE_RATE_HZ,
        sample_rate_hz=target_rate_hz,
        resample_up=ratio.numerator,
        resample_down=ratio.denominator,
        samples=_readonly_float64(converted),
        native_signal_sha256=native.signal_sha256,
    )


def generate_rate_copies(class_id: int, underlying_id: int) -> tuple[RateCopy, ...]:
    """Generate all six ordered rate copies for one underlying signal."""

    native = generate_native_signal(class_id, underlying_id)
    return tuple(make_rate_copy(native, rate) for rate in EVALUATION_RATES_HZ)


def iter_rate_copies(
    split: SplitName | None = None,
    rates_hz: Sequence[int] = EVALUATION_RATES_HZ,
) -> Iterator[RateCopy]:
    """Yield the bank in class, underlying-ID, then requested-rate order."""

    if split is not None and split not in _SPLIT_NAMES:
        raise ValueError(f"split must be one of {_SPLIT_NAMES} or None, got {split!r}")
    rates = tuple(_validate_rate(rate) for rate in rates_hz)
    if not rates:
        raise ValueError("rates_hz must not be empty")
    if len(set(rates)) != len(rates):
        raise ValueError("rates_hz must not contain duplicates")

    for class_id in CLASS_IDS:
        assignments = split_underlying_ids(class_id)
        split_by_id = {
            underlying_id: split_name
            for split_name, underlying_ids in assignments.items()
            for underlying_id in underlying_ids
        }
        for underlying_id in range(SIGNALS_PER_CLASS):
            if split is not None and split_by_id[underlying_id] != split:
                continue
            native = generate_native_signal(class_id, underlying_id)
            for rate_hz in rates:
                yield make_rate_copy(native, rate_hz)


def protocol_manifest() -> dict[str, Any]:
    """Return the cheap, generation-independent part of the E1 manifest."""

    split_ids = {
        str(class_id): {
            split_name: list(ids)
            for split_name, ids in split_underlying_ids(class_id).items()
        }
        for class_id in CLASS_IDS
    }
    generator = {
        "protocol_id": PROTOCOL_ID,
        "generator_version": GENERATOR_VERSION,
        "generator_seed": GENERATOR_SEED,
        "random_generator": "numpy.Generator(PCG64(SeedSequence))",
        "signal_seed_sequence": [GENERATOR_SEED, "class_id", "underlying_id"],
        "split_seed_sequence": [GENERATOR_SEED, "class_id", 999],
        "underlying_id_range": [0, SIGNALS_PER_CLASS - 1],
        "native_rate_hz": NATIVE_RATE_HZ,
        "duration_s": DURATION_S,
        "native_points": NATIVE_POINTS,
        "signals_per_class": SIGNALS_PER_CLASS,
        "class_ids": list(CLASS_IDS),
        "split_counts_per_class": dict(SPLIT_COUNTS),
        "evaluation_rates_hz": list(EVALUATION_RATES_HZ),
        "shared_sinusoids_hz": list(SHARED_SINUSOIDS_HZ),
        "shared_amplitude_distribution": {"name": "uniform", "low": 0.2, "high": 1.0},
        "shared_phase_distribution_rad": {
            "name": "uniform",
            "low": 0.0,
            "high_exclusive": 2.0 * np.pi,
        },
        "impulse_frequency_hz_by_class": IMPULSE_FREQUENCY_HZ,
        "impulse_signal_amplitude_distribution": {
            "name": "uniform",
            "low": 0.5,
            "high": 1.5,
        },
        "first_impulse_time_distribution": "uniform_[0,nominal_period)",
        "continuous_event_time_to_sample": "floor(time_s*native_rate_hz)",
        "class_2_interval_jitter": {
            "formula": "nominal_period*(1+epsilon)",
            "epsilon_distribution": {"name": "normal", "mean": 0.0, "sd": 0.05},
            "nonpositive_interval_action": "redraw_from_same_signal_rng",
        },
        "private_resonance_hz_by_class": PRIVATE_RESONANCE_HZ,
        "resonance_kernel": "exp(-1200*n/200000)*sin(2*pi*fc*n/200000)",
        "resonance_kernel_support_samples": NATIVE_POINTS,
        "convolution_rule": "full_then_truncate_first_4000_samples",
        "pre_noise_scaling": "unit_rms",
        "noise_base_draws": "iid_standard_normal",
        "finite_sample_noise_centering": "subtract_realized_sample_mean",
        "finite_sample_noise_scaling": "rescale_realized_rms_to_exact_drawn_snr",
        "finite_sample_noise_distribution_note": (
            "centered_and_rms_constrained_vector_not_iid_gaussian"
        ),
        "noise_snr_db_distribution": {"name": "uniform", "low": 10.0, "high": 30.0},
        "snr_definition": "10*log10(clean_power/noise_power)_before_rate_conversion",
        "noise_realizations_per_underlying_signal": 1,
        "noise_injection_stage": "native_200khz_before_rate_conversion",
        "all_rate_copies_share_same_noisy_native_realization": True,
        "per_rate_noise_redraw_allowed": False,
        "rate_copy_resampling": {
            "implementation": "scipy.signal.resample_poly",
            "ratio_rule": "exact_reduced_rational",
            "window": ["kaiser", KAISER_BETA],
            "padtype": "line",
        },
        "source_shared_band_hz": SOURCE_SHARED_BAND_HZ,
        "hash_array_encoding": {"dtype": "<f8", "order": "C", "shape_framed": True},
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }
    return {
        "generator": generator,
        "generator_sha256": canonical_json_sha256(generator),
        "split_underlying_ids": split_ids,
        "split_underlying_ids_sha256": canonical_json_sha256(split_ids),
    }


def build_bank_manifest(
    split: SplitName | None = None,
    rates_hz: Sequence[int] = EVALUATION_RATES_HZ,
    *,
    include_record_hashes: bool = True,
) -> dict[str, Any]:
    """Generate and hash a requested bank view without writing an artifact."""

    rates = tuple(_validate_rate(rate) for rate in rates_hz)
    identity = {
        "generator_version": GENERATOR_VERSION,
        "split": split,
        "rates_hz": list(rates),
        "ordering": ["class_id", "underlying_id", "rates_hz_argument_order"],
    }
    aggregate = sha256()
    _framed_update(aggregate, _canonical_json_bytes(identity))
    record_hashes: list[dict[str, Any]] = []
    record_count = 0
    sample_count = 0
    for rate_copy in iter_rate_copies(split=split, rates_hz=rates):
        record_digest = rate_copy.record_sha256
        _framed_update(aggregate, bytes.fromhex(record_digest))
        record_count += 1
        sample_count += int(rate_copy.samples.size)
        if include_record_hashes:
            record_hashes.append(
                {
                    "class_id": rate_copy.class_id,
                    "underlying_id": rate_copy.underlying_id,
                    "split": rate_copy.split,
                    "sample_rate_hz": rate_copy.sample_rate_hz,
                    "record_sha256": record_digest,
                }
            )

    manifest: dict[str, Any] = {
        **protocol_manifest(),
        "bank_view": identity,
        "rate_copy_count": record_count,
        "total_sample_count": sample_count,
        "bank_sha256": aggregate.hexdigest(),
    }
    if include_record_hashes:
        manifest["record_hashes"] = record_hashes
    return manifest


__all__ = [
    "CLASS_IDS",
    "DURATION_S",
    "EVALUATION_RATES_HZ",
    "GENERATOR_SEED",
    "GENERATOR_VERSION",
    "NATIVE_POINTS",
    "NATIVE_RATE_HZ",
    "PROTOCOL_ID",
    "RateCopy",
    "SIGNALS_PER_CLASS",
    "SPLIT_COUNTS",
    "NativeSignal",
    "build_bank_manifest",
    "canonical_json_sha256",
    "expected_rate_points",
    "generate_native_signal",
    "generate_rate_copies",
    "iter_rate_copies",
    "make_rate_copy",
    "protocol_manifest",
    "samples_sha256",
    "split_for_underlying",
    "split_underlying_ids",
]
