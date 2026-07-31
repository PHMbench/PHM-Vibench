"""Population-level dependency regularization for multichannel windows."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn


def _parse_bandwidths(bandwidths: Sequence[float]) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in bandwidths)
    if not parsed or any(not math.isfinite(value) or value <= 0.0 for value in parsed):
        raise ValueError(
            "population RBF bandwidths must be non-empty, finite, and positive"
        )
    return parsed


def pearson_correlation_vectors(
    windows: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return upper-triangle channel correlations for each ``[N,C,L]`` window."""

    tensor = torch.as_tensor(windows)
    parsed_eps = float(eps)
    if not math.isfinite(parsed_eps) or parsed_eps <= 0.0:
        raise ValueError("population correlation eps must be finite and positive")
    if tensor.ndim != 3:
        raise ValueError(
            f"population windows must be [N,C,L], got {tuple(tensor.shape)}"
        )
    if tensor.shape[0] < 2:
        raise ValueError("population regularization requires batch_size >= 2")
    if tensor.shape[1] < 2:
        raise ValueError("population regularization requires at least two channels")
    if tensor.shape[2] < 2:
        raise ValueError("population regularization requires window length >= 2")
    if not torch.isfinite(tensor).all():
        raise ValueError("population windows contain NaN/Inf")

    centered = tensor - tensor.mean(dim=2, keepdim=True)
    norms = centered.square().sum(dim=2, keepdim=True).sqrt().clamp_min(parsed_eps)
    normalized = centered / norms
    correlations = torch.einsum("ncl,ndl->ncd", normalized, normalized)
    indices = torch.triu_indices(
        tensor.shape[1],
        tensor.shape[1],
        offset=1,
        device=tensor.device,
    )
    return correlations[:, indices[0], indices[1]]


def multi_rbf_mmd(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    bandwidths: Sequence[float],
) -> torch.Tensor:
    """Return biased multi-kernel RBF MMD squared for two feature populations."""

    real = torch.as_tensor(real_features)
    fake = torch.as_tensor(fake_features, device=real.device, dtype=real.dtype)
    parsed_bandwidths = _parse_bandwidths(bandwidths)
    if real.ndim != 2 or fake.ndim != 2:
        raise ValueError("MMD features must both be [N,D]")
    if real.shape[0] < 2 or fake.shape[0] < 2:
        raise ValueError("population MMD requires at least two real and fake samples")
    if real.shape[1] != fake.shape[1]:
        raise ValueError("real and fake population feature dimensions must match")
    if not torch.isfinite(real).all() or not torch.isfinite(fake).all():
        raise ValueError("population MMD features contain NaN/Inf")

    def kernel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        squared_distance = torch.cdist(left, right).square()
        values = [
            torch.exp(-squared_distance / (2.0 * bandwidth * bandwidth))
            for bandwidth in parsed_bandwidths
        ]
        return torch.stack(values).mean(dim=0)

    value = kernel(real, real).mean() + kernel(fake, fake).mean()
    value = value - 2.0 * kernel(real, fake).mean()
    if not torch.isfinite(value):
        raise ValueError("population MMD returned NaN/Inf")
    return value.clamp_min(0.0)


class PopulationCorrelationMMD(nn.Module):
    """Compare distributions of within-window channel correlations."""

    def __init__(self, bandwidths: Sequence[float], eps: float = 1e-8):
        super().__init__()
        self.bandwidths = _parse_bandwidths(bandwidths)
        self.eps = float(eps)
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("population correlation eps must be finite and positive")

    def forward(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        return multi_rbf_mmd(
            pearson_correlation_vectors(real, eps=self.eps),
            pearson_correlation_vectors(fake, eps=self.eps),
            self.bandwidths,
        )
