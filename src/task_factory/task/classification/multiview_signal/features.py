"""Fixed P19 multiview signal coordinates.

Adapted from liq22/phm-agent-benchmark signal_programs.py at commit
76a195133c23149147d204fd2883eebcc2e9289a. The periodic-window DFT, Hilbert,
signed Teager and nonwrapping ACF definitions are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import hilbert


@dataclass(frozen=True)
class SignalFeatureSpec:
    """Four Gaussian bands, three modulation intervals and three fixed lags."""

    sample_rate_hz: float
    bands_hz: tuple[tuple[float, float], ...]
    modulation_bands_hz: tuple[tuple[float, float], ...]
    lags_samples: tuple[int, ...]

    def __post_init__(self) -> None:
        nyquist = self.sample_rate_hz / 2
        if not np.isfinite(nyquist) or nyquist <= 0:
            raise ValueError("sample_rate_hz must be finite and positive")
        if (len(self.bands_hz), len(self.modulation_bands_hz), len(self.lags_samples)) != (4, 3, 3):
            raise ValueError("P19 requires four Gaussian bands, three modulation intervals and three lags: 42 coordinates per channel")
        if any(len(pair) != 2 or not all(np.isfinite(pair)) or
               not (0 < pair[0] < nyquist and 0 < pair[1] < nyquist)
               for pair in self.bands_hz):
            raise ValueError("Gaussian center/width must be finite, positive and below Nyquist")
        if any(len(pair) != 2 or not all(np.isfinite(pair)) or
               not 0 < pair[0] < pair[1] <= nyquist
               for pair in self.modulation_bands_hz):
            raise ValueError("Modulation intervals require 0 < low < high <= Nyquist")
        if any(type(lag) is not int or lag <= 0 for lag in self.lags_samples):
            raise ValueError("Autocorrelation lags must be positive integer sample counts")
        if (len(set(self.bands_hz)) != 4 or len(set(self.modulation_bands_hz)) != 3
                or len(set(self.lags_samples)) != 3):
            raise ValueError("P19 feature coordinates must have distinct bands, intervals and lags")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SignalFeatureSpec":
        expected = {"sample_rate_hz", "bands_hz", "modulation_bands_hz", "lags_samples"}
        if set(value) != expected:
            raise ValueError(f"Program fields must be exactly {sorted(expected)}")
        return cls(float(value["sample_rate_hz"]),
                   tuple(tuple(map(float, pair)) for pair in value["bands_hz"]),
                   tuple(tuple(map(float, pair)) for pair in value["modulation_bands_hz"]),
                   tuple(value["lags_samples"]))

    def to_dict(self) -> dict[str, Any]:
        return dict(sample_rate_hz=self.sample_rate_hz, bands_hz=self.bands_hz,
                    modulation_bands_hz=self.modulation_bands_hz,
                    lags_samples=self.lags_samples)


ProgramSpec = SignalFeatureSpec


def feature_names(spec: SignalFeatureSpec, channels: int = 1) -> list[str]:
    if type(channels) is not int or channels < 1:
        raise ValueError("channels must be a positive integer")
    names: list[str] = []
    for channel in range(channels):
        prefix = f"ch{channel}"
        names.extend((f"{prefix}/raw/kurtosis", f"{prefix}/raw/crest"))
        for index, (center, width) in enumerate(spec.bands_hz):
            band = f"{prefix}/band{index}[{center:g}Hz,{width:g}Hz]"
            names.extend(f"{band}/{stat}" for stat in
                         ("power_fraction", "kurtosis", "envelope_modulation", "teager_signed_ratio"))
            names.extend(f"{band}/acf_lag{lag}[{lag / spec.sample_rate_hz:g}s]"
                         for lag in spec.lags_samples)
            names.extend(f"{band}/envelope_power[{low:g},{high:g}Hz]"
                         for low, high in spec.modulation_bands_hz)
    return names


def view_indices(channels: int = 1) -> dict[str, np.ndarray]:
    """Return disjoint coordinate sets, preserving each channel's original order."""
    if type(channels) is not int or channels < 1:
        raise ValueError("channels must be a positive integer")
    indices: dict[str, list[int]] = dict(transient=[], band=[], modulation=[], recurrence=[])
    for channel in range(channels):
        start = 42 * channel
        indices["transient"].extend((start, start + 1))
        for index in range(4):
            band = start + 2 + 10 * index
            indices["band"].append(band)
            indices["transient"].extend((band + 1, band + 3))
            indices["modulation"].extend((band + 2, band + 7, band + 8, band + 9))
            indices["recurrence"].extend((band + 4, band + 5, band + 6))
    return {name: np.array(sorted(values), dtype=np.int64) for name, values in indices.items()}


def program_features(x: np.ndarray, spec: SignalFeatureSpec) -> tuple[np.ndarray, list[str]]:
    """Return [N,F] explicit features, ordered channel -> raw -> band -> statistic.

    Filtering and Hilbert demodulation use the finite window's periodic DFT.
    Autocorrelation and Teager products use valid, non-wrapping time indices.
    The signed Teager statistic is not clipped and is not physical energy.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or min(x.shape) < 1 or x.shape[1] < 4 or not np.isfinite(x).all():
        raise ValueError('Expected finite, nonempty [N,T,C] windows, T >= 4')
    ntime = x.shape[1]
    if spec.lags_samples and max(spec.lags_samples) >= ntime - 1:
        raise ValueError('Requested lag is not supported by this observation window')
    freq = np.fft.rfftfreq(ntime, d=1 / spec.sample_rate_hz)
    mod_masks = [(freq >= low) & (freq < high)
                 for low, high in spec.modulation_bands_hz]
    if any(not mask.any() for mask in mod_masks):
        raise ValueError('A modulation band contains no observable DFT bin; change the protocol explicitly')
    eps = 1e-12
    columns: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, value: np.ndarray) -> None:
        names.append(name)
        columns.append(value)

    for channel in range(x.shape[2]):
        signal = x[:, :, channel]
        signal = signal - signal.mean(axis=1, keepdims=True)
        energy = np.mean(signal ** 2, axis=1)
        rms = np.sqrt(energy)
        prefix = f'ch{channel}'
        add(f'{prefix}/raw/kurtosis', np.mean(signal ** 4, axis=1) / (energy ** 2 + eps))
        add(f'{prefix}/raw/crest', np.max(np.abs(signal), axis=1) / (rms + eps))
        spectrum = np.fft.rfft(signal, axis=1)
        for index, (center, width) in enumerate(spec.bands_hz):
            response = np.exp(-0.5 * ((freq - center) / width) ** 2)
            filtered = np.fft.irfft(spectrum * response, n=ntime, axis=1)
            power = np.mean(filtered ** 2, axis=1)
            envelope = np.abs(hilbert(filtered, axis=1))
            centered_envelope = envelope - envelope.mean(axis=1, keepdims=True)
            band = f'{prefix}/band{index}[{center:g}Hz,{width:g}Hz]'
            add(f'{band}/power_fraction', power / (energy + eps))
            add(f'{band}/kurtosis', np.mean(filtered ** 4, axis=1) / (power ** 2 + eps))
            add(f'{band}/envelope_modulation', np.mean(centered_envelope ** 2, axis=1) /
                (np.mean(envelope ** 2, axis=1) + eps))
            teager = filtered[:, 1:-1] ** 2 - filtered[:, :-2] * filtered[:, 2:]
            add(f'{band}/teager_signed_ratio', teager.mean(axis=1) / (power + eps))
            denominator = np.sum(filtered ** 2, axis=1) + eps
            for lag in spec.lags_samples:
                add(f'{band}/acf_lag{lag}[{lag / spec.sample_rate_hz:g}s]',
                    np.sum(filtered[:, :-lag] * filtered[:, lag:], axis=1) / denominator)
            mod_power = np.abs(np.fft.rfft(centered_envelope, axis=1)) ** 2
            # One-sided DFT energy: double non-DC/non-Nyquist coefficients.
            correction = np.full(mod_power.shape[1], 2.0)
            correction[0] = 1.0
            if ntime % 2 == 0:
                correction[-1] = 1.0
            mod_power *= correction
            total = mod_power.sum(axis=1) + eps
            for (low, high), mask in zip(spec.modulation_bands_hz, mod_masks, strict=True):
                add(f'{band}/envelope_power[{low:g},{high:g}Hz]',
                    mod_power[:, mask].sum(axis=1) / total)
    values = np.column_stack(columns)
    if not np.isfinite(values).all():
        raise FloatingPointError('Nonfinite signal-program output; no silent replacement')
    return values, names


def extract_signal_features(x: np.ndarray, spec: SignalFeatureSpec) -> np.ndarray:
    """Extract the fixed coordinates from finite windows with shape [N,T,C]."""
    return program_features(x, spec)[0]
