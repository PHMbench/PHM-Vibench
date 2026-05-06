from __future__ import annotations

import torch


def leakage_metrics(real: torch.Tensor, fake: torch.Tensor) -> dict[str, float]:
    """Nearest-neighbor leakage placeholder for exploratory synthetic tensors."""
    with torch.no_grad():
        real_flat = real.float().reshape(real.shape[0], -1)
        fake_flat = fake.float().reshape(fake.shape[0], -1)
        distances = torch.cdist(fake_flat, real_flat)
        return {"leakage_nearest_neighbor_l2": float(distances.min(dim=1).values.mean().cpu())}

