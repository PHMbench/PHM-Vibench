"""Official FISHER inference preprocessing on source-normalized physical windows."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def log_spectrogram(common, incremental, availability, *, source_rms, sample_rate_hz,
                    band_width, patch_size, allow_time_frame_drop,
                    allow_frequency_bin_drop):
    """Return the official log-STFT input and disclose any discarded coordinates.

    The inputs are the common and available incremental coordinates on the
    caller's declared physical grid. The mean subtraction and fixed constants
    follow the released FISHER inference example. Partial time patches and
    frequency bands follow its implementation only when explicitly allowed.
    """
    if common.ndim != 3 or incremental.shape != common.shape or availability.shape != (len(common),):
        raise ValueError('FISHER needs matching [B,K,P] coordinates and one availability per window')
    if not ((availability == 0) | (availability == 1)).all():
        raise ValueError('FISHER incremental availability must be binary')
    rate, width, patch = int(sample_rate_hz), int(band_width), int(patch_size)
    if rate != sample_rate_hz or rate < 1 or width < 1 or patch < 1 or source_rms <= 0:
        raise ValueError('FISHER needs positive source RMS and integer sample rate, band width and patch size')
    if not isinstance(allow_time_frame_drop, bool) or not isinstance(allow_frequency_bin_drop, bool):
        raise ValueError('FISHER discarded-coordinate permissions must be explicit booleans')
    admitted = torch.where(availability[:, None, None].bool(), incremental, torch.zeros_like(incremental))
    wave = (common + admitted).reshape(len(common), -1) / source_rms
    if not torch.isfinite(wave).all():
        raise ValueError('FISHER admitted waveform must be finite')
    wave = wave - wave.mean(dim=-1, keepdim=True)
    n_fft, hop = 25 * rate // 1000, 10 * rate // 1000
    if n_fft < 2 or hop < 1 or wave.shape[-1] < n_fft:
        raise ValueError('FISHER official STFT needs at least one valid frame and positive hop')
    magnitude = torch.stft(wave, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                           window=torch.hann_window(n_fft, dtype=wave.dtype, device=wave.device),
                           center=False, return_complex=True).abs()
    spec = (torch.log(magnitude + 1e-10).transpose(-2, -1) + 3.017344307886898) / (2.1531635155379805 * 2)
    frames, frequencies = spec.shape[-2:]
    used_frames = min(frames, 1024)
    dropped_frames = frames - used_frames + used_frames % patch
    if dropped_frames and not allow_time_frame_drop:
        raise ValueError(f'FISHER would discard {dropped_frames} STFT time frames')
    spec = spec[:, :1024]
    padded = max(0, width - frequencies)
    if padded:
        spec = F.pad(spec, (0, padded))
    retained_bins_per_band = (width // patch) * patch
    retained_bands = spec.shape[-1] // width
    used_bins = sum(min(max(frequencies - i * width, 0), retained_bins_per_band)
                    for i in range(retained_bands))
    dropped_bins = frequencies - used_bins
    if dropped_bins and not allow_frequency_bin_drop:
        raise ValueError(f'FISHER would discard {dropped_bins} observed STFT frequency bins')
    return spec.unsqueeze(1), dict(sample_rate_hz=rate, stft_n_fft=n_fft, stft_hop=hop,
        stft_frames=frames, stft_frequency_bins=frequencies, frequency_bins_padded=padded,
        time_frames_discarded=dropped_frames, observed_frequency_bins_discarded=dropped_bins,
        complete_frequency_bands=retained_bands)
