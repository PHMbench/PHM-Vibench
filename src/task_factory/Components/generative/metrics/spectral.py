from __future__ import annotations

import torch


def _nan_metrics(keys: list[str]) -> dict[str, float]:
    out = {key: float("nan") for key in keys}
    out.update({f"{key}_status_code": 0.0 for key in keys})
    return out


def spectral_metrics(real: torch.Tensor, fake: torch.Tensor) -> dict[str, float]:
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
            return _nan_metrics(keys)
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
        return out
