"""Fixed physical-grid projectors for the TII common/increment comparison.

This transform consumes an already selected raw window, not a recording reader.
Band intervals are [low, high) in Hz and use the periodic Fourier basis of the
whole physical window. Fourier resampling truncates unrepresentable frequencies
before changing the grid; it does not invent bandwidth absent from acquisition.
The two outputs retain the declared physical unit. The tokenizer subsequently
divides both by the one source-training RMS.

Qualification owns device-response evidence and freezes bands before target
waveform access. A declared support basis is not authentication of that evidence;
in particular, Nyquist bounds checked here are only necessary conditions.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Sequence

import torch


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite positive number')
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a finite positive number')
    return value


def _samples(duration: float, rate: float, name: str) -> int:
    count = duration * rate
    if not math.isfinite(count) or not math.isclose(count, round(count), abs_tol=1e-6, rel_tol=0):
        raise ValueError(f'duration_s * {name} must be an integer sample count')
    if round(count) < 2:
        raise ValueError(f'duration_s * {name} must contain at least two samples')
    return round(count)


def _bands(values: Sequence[tuple[float, float]], name: str) -> list[tuple[float, float]]:
    if not values:
        raise ValueError(f'{name} must contain a nonempty physical frequency set')
    result = []
    for band in values:
        if len(band) != 2:
            raise ValueError(f'{name} requires [low, high) Hz pairs')
        lo, hi = band
        if (isinstance(lo, bool) or isinstance(hi, bool)
                or not isinstance(lo, Real) or not isinstance(hi, Real)
                or not math.isfinite(lo) or not math.isfinite(hi) or not 0 <= lo < hi):
            raise ValueError(f'{name} requires finite 0 <= low < high in Hz')
        result.append((float(lo), float(hi)))
    return result


def _grid_spectrum(signal: torch.Tensor, count: int) -> torch.Tensor:
    """Resample a periodic physical window without aliasing into lower bins."""
    size = len(signal)
    spectrum = torch.fft.rfft(signal, norm='forward')
    retained = min(size, count) // 2 + 1
    output = spectrum.new_zeros(count // 2 + 1)
    output[:retained] = spectrum[:retained]
    # An even-length Nyquist coefficient represents one real mode. On a larger
    # grid it becomes a conjugate pair; downsampling merges that pair instead.
    if size != count and min(size, count) % 2 == 0:
        output[retained - 1] *= 0.5 if size < count else 2.0
    return output


def project_window(
    raw: torch.Tensor,
    *,
    channel: int,
    native_rate_hz: float,
    effective_rate_hz: float,
    grid_rate_hz: float,
    duration_s: float,
    num_patches: int,
    patch_size: int,
    common_bands_hz: Sequence[tuple[float, float]],
    increment_bands_hz: Sequence[tuple[float, float]],
    common_support: str,
    increment_support: str,
    input_unit: str,
    output_unit: str,
    unit_scale: float,
    support_basis: str,
    support_evidence: str,
) -> dict[str, torch.Tensor]:
    """Project explicit ``raw[L,C]`` into physical-unit ``[K,P]`` blocks.

    Rates describe acquisition, the supplied samples, and the frozen output grid,
    respectively. ``unit_scale`` converts the selected channel to ``output_unit``
    by multiplication; no centering, fitting, padding, or window selection occurs.
    Full common support is mandatory. Incremental support must be fully usable
    or unavailable for the *entire* declared incremental set.

    ``documented_response`` is the natural-support path. ``designed_filter`` and
    ``tensor_fixture`` identify distinct study/test scopes, not natural hardware
    qualification. Evidence must describe more than a sampling-rate calculation.
    """
    if common_support != 'fully_usable':
        raise ValueError('common_support must be fully_usable; missing common cannot be filled')
    if increment_support not in ('fully_usable', 'unavailable'):
        raise ValueError('increment_support must be fully_usable or unavailable; partial/unknown is ineligible')
    if support_basis not in ('documented_response', 'designed_filter', 'tensor_fixture'):
        raise ValueError('support_basis must identify response evidence, designed filtering, or a tensor fixture; Nyquist is insufficient')
    if not isinstance(support_evidence, str) or not support_evidence.strip():
        raise ValueError('support_evidence must explicitly identify the qualification evidence')
    unknown_units = {'unknown', 'none', 'nan', 'arbitrary', 'a.u.', 'au'}
    for name, unit in (('input_unit', input_unit), ('output_unit', output_unit)):
        if not isinstance(unit, str) or not unit.strip() or unit.strip().lower() in unknown_units:
            raise ValueError(f'{name} must be a known physical unit')
    scale = _positive(unit_scale, 'unit_scale')
    native = _positive(native_rate_hz, 'native_rate_hz')
    effective = _positive(effective_rate_hz, 'effective_rate_hz')
    grid = _positive(grid_rate_hz, 'grid_rate_hz')
    duration = _positive(duration_s, 'duration_s')
    for name, size in (('num_patches', num_patches), ('patch_size', patch_size)):
        if isinstance(size, bool) or not isinstance(size, Integral) or size < 1:
            raise ValueError(f'{name} must be a positive integer')
    count = _samples(duration, grid, 'grid_rate_hz')
    if num_patches * patch_size != count:
        raise ValueError('num_patches * patch_size must equal duration_s * grid_rate_hz')
    source_count = _samples(duration, effective, 'effective_rate_hz')
    if not isinstance(raw, torch.Tensor) or raw.ndim != 2 or raw.shape[0] != source_count:
        raise ValueError(f'raw must have shape [{source_count},C] for the declared physical duration')
    if isinstance(channel, bool) or not isinstance(channel, Integral) or not 0 <= channel < raw.shape[1]:
        raise ValueError('channel must identify an existing raw channel explicitly')
    if raw.dtype not in (torch.float32, torch.float64):
        raise ValueError('raw must use float32 or float64 physical samples')
    common_bands = _bands(common_bands_hz, 'common_bands_hz')
    increment_bands = _bands(increment_bands_hz, 'increment_bands_hz')
    all_bands = sorted(common_bands + increment_bands)
    if any(left[1] > right[0] for left, right in zip(all_bands, all_bands[1:])):
        raise ValueError('common and incremental frequency atoms must be mutually disjoint')
    if any(hi > grid / 2 for _, hi in all_bands):
        raise ValueError('all frequency atoms must lie within grid Nyquist')
    available = increment_support == 'fully_usable'
    observed = common_bands + (increment_bands if available else [])
    if any(hi > min(native, effective) / 2 for _, hi in observed):
        raise ValueError('declared usable bands exceed native/effective Nyquist; resampling cannot restore them')
    signal = raw[:, channel] * scale
    if not torch.isfinite(signal).all():
        raise ValueError('the selected physical channel must contain finite samples')
    spectrum = _grid_spectrum(signal, count)
    frequencies = torch.arange(len(spectrum), dtype=torch.float64, device=raw.device) / duration

    def project(bands: list[tuple[float, float]]) -> torch.Tensor:
        mask = torch.zeros_like(frequencies, dtype=torch.bool)
        for lo, hi in bands:
            mask |= (frequencies >= lo) & (frequencies < hi)
        if not mask.any():
            raise ValueError('each support set must contain a Fourier mode on the fixed physical grid')
        return torch.fft.irfft(spectrum * mask, n=count, norm='forward').reshape(num_patches, patch_size)

    common = project(common_bands)
    incremental = project(increment_bands)
    if not available:
        incremental = torch.zeros_like(incremental)
    return {
        'common': common,
        'incremental': incremental,
        'increment_available': torch.tensor(available, device=raw.device),
    }
