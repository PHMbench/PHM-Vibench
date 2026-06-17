from __future__ import annotations

import torch


def _nan_metrics(keys: list[str]) -> dict[str, float]:
    out = {key: float("nan") for key in keys}
    out.update({f"{key}_status_code": 0.0 for key in keys})
    return out


def _resolve_fault_frequency(
    fault_frequency_hz: float | None,
    shaft_rpm: float | None,
) -> float | None:
    if fault_frequency_hz is not None and float(fault_frequency_hz) > 0.0:
        return float(fault_frequency_hz)
    if shaft_rpm is not None and float(shaft_rpm) > 0.0:
        return float(shaft_rpm) / 60.0
    return None


def _frequency_axis(length: int, sampling_rate_hz: float, device) -> torch.Tensor:
    return torch.fft.rfftfreq(length, d=1.0 / float(sampling_rate_hz)).to(device)


def _analytic_envelope(x: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.fft(x.float(), dim=-1)
    h = torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)
    if x.shape[-1] % 2 == 0:
        h[0] = 1.0
        h[x.shape[-1] // 2] = 1.0
        h[1 : x.shape[-1] // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (x.shape[-1] + 1) // 2] = 2.0
    analytic = torch.fft.ifft(spectrum * h.view(*([1] * (x.ndim - 1)), -1), dim=-1)
    return analytic.abs()


def _band_energy(psd: torch.Tensor, freqs: torch.Tensor, low: float, high: float) -> torch.Tensor:
    mask = (freqs >= float(low)) & (freqs < float(high))
    if not bool(mask.any()):
        return torch.full(psd.shape[:-1], float("nan"), device=psd.device)
    return psd[..., mask].sum(dim=-1)


def _peak_frequency(amp: torch.Tensor, freqs: torch.Tensor, center_hz: float) -> torch.Tensor:
    width = max(float(center_hz) * 0.1, float(freqs[1] - freqs[0]) if freqs.numel() > 1 else 1.0)
    mask = (freqs >= center_hz - width) & (freqs <= center_hz + width)
    if not bool(mask.any()):
        return torch.full(amp.shape[:-1], float("nan"), device=amp.device)
    band_amp = amp[..., mask]
    band_freqs = freqs[mask]
    return band_freqs[band_amp.argmax(dim=-1)]


def _phm_spectral_metrics(
    real: torch.Tensor,
    fake: torch.Tensor,
    *,
    sampling_rate_hz: float | None,
    shaft_rpm: float | None,
    fault_frequency_hz: float | None,
) -> dict[str, float]:
    keys = [
        "spectral_phm_band_energy_error",
        "spectral_envelope_spectrum_l1",
        "spectral_fault_characteristic_peak_error",
        "spectral_harmonic_ratio_error",
        "spectral_cross_channel_coherence_error",
    ]
    if sampling_rate_hz is None or float(sampling_rate_hz) <= 0.0:
        return _nan_metrics(keys)

    sample_rate = float(sampling_rate_hz)
    fault_hz = _resolve_fault_frequency(fault_frequency_hz, shaft_rpm)
    freqs = _frequency_axis(real.shape[-1], sample_rate, real.device)
    real_amp = torch.fft.rfft(real.float(), dim=-1).abs()
    fake_amp = torch.fft.rfft(fake.float(), dim=-1).abs()
    real_psd = real_amp.pow(2)
    fake_psd = fake_amp.pow(2)
    nyquist = sample_rate / 2.0
    band_edges = [0.0, nyquist / 4.0, nyquist / 2.0, nyquist]
    real_bands = []
    fake_bands = []
    for low, high in zip(band_edges[:-1], band_edges[1:]):
        real_bands.append(_band_energy(real_psd, freqs, low, high))
        fake_bands.append(_band_energy(fake_psd, freqs, low, high))
    real_band_energy = torch.stack(real_bands, dim=-1)
    fake_band_energy = torch.stack(fake_bands, dim=-1)

    real_env_amp = torch.fft.rfft(_analytic_envelope(real), dim=-1).abs()
    fake_env_amp = torch.fft.rfft(_analytic_envelope(fake), dim=-1).abs()
    coherence_error = float("nan")
    if real.shape[1] >= 2:
        real_cross = torch.fft.rfft(real[:, 0].float(), dim=-1) * torch.conj(
            torch.fft.rfft(real[:, 1].float(), dim=-1)
        )
        fake_cross = torch.fft.rfft(fake[:, 0].float(), dim=-1) * torch.conj(
            torch.fft.rfft(fake[:, 1].float(), dim=-1)
        )
        real_coherence = real_cross.abs() / (
            real_amp[:, 0].clamp_min(1e-8) * real_amp[:, 1].clamp_min(1e-8)
        )
        fake_coherence = fake_cross.abs() / (
            fake_amp[:, 0].clamp_min(1e-8) * fake_amp[:, 1].clamp_min(1e-8)
        )
        coherence_error = float(torch.mean(torch.abs(real_coherence - fake_coherence)).cpu())

    peak_error = float("nan")
    harmonic_error = float("nan")
    if fault_hz is not None:
        real_peak = _peak_frequency(real_amp, freqs, fault_hz)
        fake_peak = _peak_frequency(fake_amp, freqs, fault_hz)
        peak_error = float(torch.nanmean(torch.abs(real_peak - fake_peak)).cpu())
        real_fund = _band_energy(real_psd, freqs, fault_hz * 0.9, fault_hz * 1.1)
        fake_fund = _band_energy(fake_psd, freqs, fault_hz * 0.9, fault_hz * 1.1)
        real_harm = _band_energy(real_psd, freqs, fault_hz * 1.8, fault_hz * 2.2)
        fake_harm = _band_energy(fake_psd, freqs, fault_hz * 1.8, fault_hz * 2.2)
        real_ratio = real_harm / real_fund.clamp_min(1e-8)
        fake_ratio = fake_harm / fake_fund.clamp_min(1e-8)
        harmonic_error = float(torch.nanmean(torch.abs(real_ratio - fake_ratio)).cpu())

    out = {
        "spectral_phm_band_energy_error": float(
            torch.nanmean(torch.abs(real_band_energy - fake_band_energy)).cpu()
        ),
        "spectral_envelope_spectrum_l1": float(
            torch.mean(torch.abs(real_env_amp - fake_env_amp)).cpu()
        ),
        "spectral_fault_characteristic_peak_error": peak_error,
        "spectral_harmonic_ratio_error": harmonic_error,
        "spectral_cross_channel_coherence_error": coherence_error,
    }
    out.update(
        {
            f"{key}_status_code": 1.0 if torch.isfinite(torch.tensor(value)) else 0.0
            for key, value in out.items()
        }
    )
    return out


def spectral_metrics(
    real: torch.Tensor,
    fake: torch.Tensor,
    *,
    sampling_rate_hz: float | None = None,
    shaft_rpm: float | None = None,
    fault_frequency_hz: float | None = None,
) -> dict[str, float]:
    """Eval-only FFT amplitude metrics for `[N, C, L]` tensors."""
    keys = [
        "spectral_fft_l1",
        "spectral_log_l1",
        "spectral_psd_l2",
        "spectral_band_energy_error",
        "spectral_angle",
        "spectral_fault_frequency_preservation",
    ]
    with torch.no_grad():
        if real.ndim != 3 or fake.ndim != 3 or real.shape != fake.shape:
            out = _nan_metrics(keys)
            out.update(
                _phm_spectral_metrics(
                    real,
                    fake,
                    sampling_rate_hz=sampling_rate_hz,
                    shaft_rpm=shaft_rpm,
                    fault_frequency_hz=fault_frequency_hz,
                )
            )
            return out
        real_amp = torch.fft.rfft(real.float(), dim=-1).abs()
        fake_amp = torch.fft.rfft(fake.float(), dim=-1).abs()
        real_psd = real_amp.pow(2)
        fake_psd = fake_amp.pow(2)
        n_bins = real_amp.shape[-1]
        n_bands = min(4, n_bins)
        real_bands = []
        fake_bands = []
        for band in torch.tensor_split(torch.arange(n_bins, device=real_amp.device), n_bands):
            real_bands.append(real_psd.index_select(-1, band).sum(dim=-1))
            fake_bands.append(fake_psd.index_select(-1, band).sum(dim=-1))
        real_band_energy = torch.stack(real_bands, dim=-1)
        fake_band_energy = torch.stack(fake_bands, dim=-1)
        spectral_angle = torch.acos(
            torch.sum(real_amp * fake_amp, dim=-1)
            / (torch.linalg.vector_norm(real_amp, dim=-1) * torch.linalg.vector_norm(fake_amp, dim=-1)).clamp_min(1e-8)
        )
        top_k = min(5, n_bins)
        top_bins = torch.topk(real_amp.mean(dim=0), k=top_k, dim=-1).indices
        fake_top = torch.gather(fake_amp.mean(dim=0), dim=-1, index=top_bins)
        real_top = torch.gather(real_amp.mean(dim=0), dim=-1, index=top_bins).clamp_min(1e-8)
        preservation = torch.mean(torch.minimum(fake_top / real_top, real_top / fake_top.clamp_min(1e-8)))
        out = {
            "spectral_fft_l1": float(torch.mean(torch.abs(real_amp - fake_amp)).cpu()),
            "spectral_log_l1": float(torch.mean(torch.abs(torch.log1p(real_amp) - torch.log1p(fake_amp))).cpu()),
            "spectral_psd_l2": float(torch.sqrt(torch.mean((real_psd - fake_psd).pow(2))).cpu()),
            "spectral_band_energy_error": float(torch.mean(torch.abs(real_band_energy - fake_band_energy)).cpu()),
            "spectral_angle": float(torch.mean(spectral_angle).cpu()),
            "spectral_fault_frequency_preservation": float(preservation.cpu()),
        }
        out.update({f"{key}_status_code": 1.0 for key in keys})
        out.update(
            _phm_spectral_metrics(
                real,
                fake,
                sampling_rate_hz=sampling_rate_hz,
                shaft_rpm=shaft_rpm,
                fault_frequency_hz=fault_frequency_hz,
            )
        )
        return out
