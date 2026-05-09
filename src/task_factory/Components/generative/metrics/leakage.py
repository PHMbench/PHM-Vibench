from __future__ import annotations

import torch


def leakage_metrics(
    real: torch.Tensor,
    fake: torch.Tensor,
    *,
    duplicate_threshold: float = 1e-6,
) -> dict[str, float]:
    """Nearest-neighbor leakage placeholder for exploratory synthetic tensors."""
    with torch.no_grad():
        if real.ndim != 3 or fake.ndim != 3:
            return {
                "leakage_nearest_neighbor_l2": float("nan"),
                "leakage_duplicate_rate": float("nan"),
                "leakage_nearest_neighbor_pass": 0.0,
            }
        real_flat = real.float().reshape(real.shape[0], -1)
        fake_flat = fake.float().reshape(fake.shape[0], -1)
        distances = torch.cdist(fake_flat, real_flat)
        nearest = distances.min(dim=1).values
        duplicate_rate = (nearest <= float(duplicate_threshold)).float().mean()
        return {
            "leakage_nearest_neighbor_l2": float(nearest.mean().cpu()),
            "leakage_duplicate_rate": float(duplicate_rate.cpu()),
            "leakage_nearest_neighbor_pass": float((duplicate_rate == 0).cpu()),
        }
