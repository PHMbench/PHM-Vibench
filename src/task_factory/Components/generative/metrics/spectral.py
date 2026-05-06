from __future__ import annotations

import torch


def spectral_metrics(real: torch.Tensor, fake: torch.Tensor) -> dict[str, float]:
    """Eval-only FFT amplitude metrics for `[N, C, L]` tensors."""
    with torch.no_grad():
        real_amp = torch.fft.rfft(real.float(), dim=-1).abs()
        fake_amp = torch.fft.rfft(fake.float(), dim=-1).abs()
        if real_amp.shape != fake_amp.shape:
            return {"spectral_fft_l1": float("nan"), "spectral_log_l1": float("nan")}
        return {
            "spectral_fft_l1": float(torch.mean(torch.abs(real_amp - fake_amp)).cpu()),
            "spectral_log_l1": float(torch.mean(torch.abs(torch.log1p(real_amp) - torch.log1p(fake_amp))).cpu()),
        }

