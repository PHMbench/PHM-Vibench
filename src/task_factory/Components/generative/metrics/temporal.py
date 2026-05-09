from __future__ import annotations

import torch


def _nan_metrics(keys: list[str]) -> dict[str, float]:
    out = {key: float("nan") for key in keys}
    out.update({f"{key}_status_code": 0.0 for key in keys})
    return out


def _safe_std(x: torch.Tensor, dim=None) -> torch.Tensor:
    return torch.std(x, dim=dim, unbiased=False)


def _skew_kurt(x: torch.Tensor, dim) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=dim, keepdim=True)
    std = _safe_std(x, dim=dim).clamp_min(1e-8)
    centered = x - mean
    skew = (centered.pow(3).mean(dim=dim) / std.pow(3))
    kurt = (centered.pow(4).mean(dim=dim) / std.pow(4))
    return skew, kurt


def _autocorr(x: torch.Tensor, max_lag: int) -> torch.Tensor:
    x = x - x.mean(dim=-1, keepdim=True)
    corr = []
    denom = x.pow(2).mean(dim=-1).clamp_min(1e-8)
    for lag in range(max_lag + 1):
        if lag == 0:
            corr.append(torch.ones_like(denom))
        else:
            corr.append((x[..., :-lag] * x[..., lag:]).mean(dim=-1) / denom)
    return torch.stack(corr, dim=-1)


def _cross_channel_corr(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] <= 1:
        return torch.ones(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device)
    flat = x.transpose(0, 1).reshape(x.shape[1], -1)
    flat = flat - flat.mean(dim=1, keepdim=True)
    flat = flat / flat.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-8)
    return (flat @ flat.t()) / max(flat.shape[1] - 1, 1)


def temporal_metrics(real: torch.Tensor, fake: torch.Tensor) -> dict[str, float]:
    """Basic time-domain metrics for `[N, C, L]` tensors."""
    keys = [
        "temporal_mean_abs_error",
        "temporal_std_abs_error",
        "temporal_skew_abs_error",
        "temporal_kurtosis_abs_error",
        "temporal_l1",
        "temporal_l2",
        "temporal_rms_abs_error",
        "temporal_crest_factor_abs_error",
        "temporal_zero_crossing_rate_abs_error",
        "temporal_autocorr_rmse",
        "temporal_cross_channel_corr_error",
    ]
    with torch.no_grad():
        real = real.float()
        fake = fake.float()
        if real.ndim != 3 or fake.ndim != 3 or real.shape != fake.shape:
            return _nan_metrics(keys)

        real_mean = real.mean(dim=(0, 2))
        fake_mean = fake.mean(dim=(0, 2))
        real_std = _safe_std(real, dim=(0, 2))
        fake_std = _safe_std(fake, dim=(0, 2))
        real_skew, real_kurt = _skew_kurt(real, dim=(0, 2))
        fake_skew, fake_kurt = _skew_kurt(fake, dim=(0, 2))
        real_rms = real.pow(2).mean(dim=(0, 2)).sqrt()
        fake_rms = fake.pow(2).mean(dim=(0, 2)).sqrt()
        real_crest = real.abs().amax(dim=(0, 2)) / real_rms.clamp_min(1e-8)
        fake_crest = fake.abs().amax(dim=(0, 2)) / fake_rms.clamp_min(1e-8)
        real_zcr = ((real[..., 1:] * real[..., :-1]) < 0).float().mean(dim=(0, 2))
        fake_zcr = ((fake[..., 1:] * fake[..., :-1]) < 0).float().mean(dim=(0, 2))
        max_lag = min(32, real.shape[-1] - 1)
        autocorr_rmse = torch.sqrt(
            torch.mean((_autocorr(real, max_lag) - _autocorr(fake, max_lag)).pow(2))
        )
        corr_error = torch.mean(torch.abs(_cross_channel_corr(real) - _cross_channel_corr(fake)))

        out = {
            "temporal_mean_abs_error": float((real_mean - fake_mean).abs().mean().cpu()),
            "temporal_std_abs_error": float((real_std - fake_std).abs().mean().cpu()),
            "temporal_skew_abs_error": float((real_skew - fake_skew).abs().mean().cpu()),
            "temporal_kurtosis_abs_error": float((real_kurt - fake_kurt).abs().mean().cpu()),
            "temporal_l1": float(torch.mean(torch.abs(real - fake)).cpu()),
            "temporal_l2": float(torch.sqrt(torch.mean((real - fake).pow(2))).cpu()),
            "temporal_rms_abs_error": float((real_rms - fake_rms).abs().mean().cpu()),
            "temporal_crest_factor_abs_error": float((real_crest - fake_crest).abs().mean().cpu()),
            "temporal_zero_crossing_rate_abs_error": float((real_zcr - fake_zcr).abs().mean().cpu()),
            "temporal_autocorr_rmse": float(autocorr_rmse.cpu()),
            "temporal_cross_channel_corr_error": float(corr_error.cpu()),
        }
        out.update({f"{key}_status_code": 1.0 for key in keys})
        return {
            **out,
            "temporal_mean_abs_error_per_channel_mean": out["temporal_mean_abs_error"],
            "temporal_std_abs_error_per_channel_mean": out["temporal_std_abs_error"],
        }
