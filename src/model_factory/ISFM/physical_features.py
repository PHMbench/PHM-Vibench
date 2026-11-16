"""Fixed physical signal features for an envelope-spectrum linear reference."""
from __future__ import annotations

import math
import torch


def squared_envelope_spectrum(common, incremental, availability, *, source_rms, sample_rate_hz,
                              frequency_band_hz):
    """Return squared-envelope magnitudes on a fixed physical frequency band.

    Both projected coordinates are physical-unit inputs on the same grid. The
    one signal RMS comes from source training; no feature or target statistic is
    fitted here. Output frequencies follow the full-window periodic FFT grid.
    """
    if (common.ndim != 3 or common.shape != incremental.shape or len(common) < 1
            or availability.shape != (len(common),)
            or not ((availability == 0) | (availability == 1)).all()):
        raise ValueError('matching [B,K,P] physical coordinates and binary availability required')
    rms, rate = float(source_rms), float(sample_rate_hz)
    lo, hi = map(float, frequency_band_hz)
    if (not all(math.isfinite(v) for v in (rms, rate, lo, hi))
            or rms <= 0 or rate <= 0 or not 0 <= lo < hi <= rate / 2):
        raise ValueError('positive source RMS/rate and a valid physical frequency band required')
    admitted = torch.where(availability[:, None, None].bool(), incremental,
                           torch.zeros_like(incremental))
    signal = (common + admitted).reshape(len(common), -1) / rms
    if not torch.isfinite(signal).all():
        raise ValueError('admitted union must be finite')
    n = signal.shape[1]
    if n < 2:
        raise ValueError('envelope spectrum requires at least two physical samples')
    multiplier = signal.new_zeros(n)
    multiplier[0] = 1
    multiplier[1:(n + 1) // 2] = 2
    if n % 2 == 0:
        multiplier[n // 2] = 1
    analytic = torch.fft.ifft(torch.fft.fft(signal, dim=-1) * multiplier, dim=-1)
    spectrum = torch.fft.rfft(analytic.abs().square(), dim=-1).abs() / n
    frequency = torch.fft.rfftfreq(n, d=1 / rate, device=signal.device)
    selected = (frequency >= lo) & (frequency < hi)
    if not selected.any():
        raise ValueError('declared envelope frequency band contains no Fourier bins')
    return spectrum[:, selected]
