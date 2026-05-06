from __future__ import annotations

import torch


def distribution_metrics(real: torch.Tensor, fake: torch.Tensor) -> dict[str, float]:
    """Lightweight distribution-distance placeholders without extra deps."""
    with torch.no_grad():
        real_flat = real.float().reshape(real.shape[0], -1)
        fake_flat = fake.float().reshape(fake.shape[0], -1)
        return {
            "distribution_mean_distance": float(torch.norm(real_flat.mean(0) - fake_flat.mean(0)).cpu()),
            "distribution_var_distance": float(torch.norm(real_flat.var(0) - fake_flat.var(0)).cpu()),
        }

