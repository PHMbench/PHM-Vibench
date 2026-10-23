"""P05 P4 feature-track model shared with the paper's experiment runner.

This is the canonical implementation of the normalized Gaussian classifier.
The caller supplies standardized *training-only* features and labels for KMeans
initialization. Fit the normalization on training units before calling this
module; it does not split data, normalize features, or select an acceptance rule.
The P4 experiment code owns the certificate and calibration procedures.

The direct ``Fuzzy`` interface is intentional: the generic model factory does
not provide the training arrays this initialization requires. This module is not
a waveform model and does not reuse the distinct additive G050 or min/max F0
decision paths.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn


class Fuzzy(nn.Module):
    """Zero-order normalized Gaussian rules with no hidden output path."""

    def __init__(
        self, x: np.ndarray, y: np.ndarray, rules: int, seed: int
    ) -> None:
        super().__init__()
        km = KMeans(n_clusters=rules, n_init=1, random_state=seed).fit(x)
        counts = np.ones((rules, int(y.max()) + 1))
        for j in range(rules):
            counts[j] += np.bincount(y[km.labels_ == j], minlength=counts.shape[1])
        self.centers = nn.Parameter(torch.tensor(km.cluster_centers_, dtype=torch.float32))
        self.log_scales = nn.Parameter(torch.zeros_like(self.centers))
        self.log_q = nn.Parameter(torch.tensor(np.log(counts), dtype=torch.float32))

    def components(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return normalized firing, consequents, distances, and scales."""
        scales = self.log_scales.exp().clamp(.1, 10.)
        distance = (x[:, None, :] - self.centers).abs()
        log_h = -.5 * ((distance / scales) ** 2).sum(dim=2)
        w = log_h.softmax(dim=1)
        q = self.log_q.softmax(dim=1)
        return w, q, distance, scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log probabilities, preserving the P4 checkpoint semantics."""
        w, q, _, _ = self.components(x)
        return (w @ q).clamp_min(1e-12).log()

    def robust_penalty(
        self, x: torch.Tensor, y: torch.Tensor, radius: float
    ) -> torch.Tensor:
        """Return the existing per-row lower-margin training surrogate."""
        _, q, d, scales = self.components(x)
        ll = -.5 * (((d + radius) / scales) ** 2).sum(dim=2)
        lh = -.5 * (((d - radius).clamp_min(0) / scales) ** 2).sum(dim=2)
        shift = lh.max(dim=1, keepdim=True).values.detach()
        lo, hi = (ll - shift).exp(), (lh - shift).exp()
        diff = q[:, y].T[:, :, None] - q[None, :, :]
        lower = torch.where(
            diff >= 0, lo[:, :, None] * diff, hi[:, :, None] * diff
        ).sum(dim=1)
        lower = lower / hi.sum(dim=1, keepdim=True)
        true_class = nn.functional.one_hot(y, q.shape[1]).bool()
        lower = lower.masked_fill(true_class, torch.inf)
        return (.05 - lower.min(dim=1).values).clamp_min(0)
