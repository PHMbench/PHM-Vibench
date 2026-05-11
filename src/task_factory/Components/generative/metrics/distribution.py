from __future__ import annotations

import torch


def _flatten_pair(real: torch.Tensor, fake: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return real.float().reshape(real.shape[0], -1), fake.float().reshape(fake.shape[0], -1)


def _pairwise_sq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, y).pow(2)


def _mmd_rbf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xy = torch.cat([x, y], dim=0)
    sq = _pairwise_sq(xy, xy)
    positive = sq[sq > 0]
    bandwidth = positive.median() if positive.numel() else torch.tensor(1.0, device=x.device)
    bandwidth = bandwidth.clamp_min(1e-8)
    k_xx = torch.exp(-_pairwise_sq(x, x) / (2.0 * bandwidth))
    k_yy = torch.exp(-_pairwise_sq(y, y) / (2.0 * bandwidth))
    k_xy = torch.exp(-_pairwise_sq(x, y) / (2.0 * bandwidth))
    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


def _sliced_wasserstein(x: torch.Tensor, y: torch.Tensor, n_projections: int = 64) -> torch.Tensor:
    n = min(x.shape[0], y.shape[0])
    if n == 0:
        return torch.tensor(float("nan"), device=x.device)
    generator = torch.Generator(device=x.device)
    generator.manual_seed(0)
    projections = torch.randn(x.shape[1], n_projections, generator=generator, device=x.device, dtype=x.dtype)
    projections = projections / torch.linalg.vector_norm(projections, dim=0, keepdim=True).clamp_min(1e-8)
    x_proj = torch.sort(x @ projections, dim=0).values[:n]
    y_proj = torch.sort(y @ projections, dim=0).values[:n]
    return torch.mean(torch.abs(x_proj - y_proj))


def _energy_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    d_xy = torch.cdist(x, y).mean()
    d_xx = torch.cdist(x, x).mean()
    d_yy = torch.cdist(y, y).mean()
    return 2.0 * d_xy - d_xx - d_yy


def distribution_metrics(real: torch.Tensor, fake: torch.Tensor) -> dict[str, float]:
    """Lightweight distribution-distance placeholders without extra deps."""
    with torch.no_grad():
        if real.ndim != 3 or fake.ndim != 3:
            return {
                "distribution_mean_distance": float("nan"),
                "distribution_var_distance": float("nan"),
                "distribution_mmd_rbf": float("nan"),
                "distribution_sliced_wasserstein": float("nan"),
                "distribution_energy_distance": float("nan"),
                "distribution_status_code": 0.0,
            }
        real_flat, fake_flat = _flatten_pair(real, fake)
        return {
            "distribution_mean_distance": float(torch.norm(real_flat.mean(0) - fake_flat.mean(0)).cpu()),
            "distribution_var_distance": float(
                torch.norm(real_flat.var(0, unbiased=False) - fake_flat.var(0, unbiased=False)).cpu()
            ),
            "distribution_mmd_rbf": float(_mmd_rbf(real_flat, fake_flat).cpu()),
            "distribution_sliced_wasserstein": float(_sliced_wasserstein(real_flat, fake_flat).cpu()),
            "distribution_energy_distance": float(_energy_distance(real_flat, fake_flat).cpu()),
            "distribution_status_code": 1.0,
        }
