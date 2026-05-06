from __future__ import annotations

import torch


def temporal_metrics(real: torch.Tensor, fake: torch.Tensor) -> dict[str, float]:
    """Basic time-domain metrics for `[N, C, L]` tensors."""
    with torch.no_grad():
        real = real.float()
        fake = fake.float()
        return {
            "temporal_mean_abs_error": float((real.mean() - fake.mean()).abs().cpu()),
            "temporal_std_abs_error": float((real.std() - fake.std()).abs().cpu()),
            "temporal_l1": float(torch.mean(torch.abs(real - fake)).cpu()) if real.shape == fake.shape else float("nan"),
        }

