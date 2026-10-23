"""Compact matched MoE controls and fixed-route reference replacement.

This experimental model consumes explicit physical views [N,K,D], raw features
[N,D], and compatibility scores [N,K]. It does not prepare signal operators or
infer physical metadata. It is distinct from historical M_04/G050 semantics.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, args: Any, metadata: Any = None) -> None:
        super().__init__()
        self.dim = int(args.input_dim)
        self.k = int(args.num_experts)
        self.classes = int(args.num_classes)
        self.width = int(args.width)
        self.arm = str(args.arm)
        if min(self.dim, self.width) < 1 or min(self.k, self.classes) < 2:
            raise ValueError('Require positive input_dim/width and >=2 experts/classes')
        if self.arm not in {'aligned', 'generic', 'shuffled', 'learned_physics', 'uniform', 'no_balance'}:
            raise ValueError(f'Unknown fixed-route control arm: {self.arm}')
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(self.dim, self.width), nn.GELU(), nn.Linear(self.width, self.width))
            for _ in range(self.k)])
        self.router = nn.Sequential(nn.Linear(self.dim, self.width), nn.GELU(), nn.Linear(self.width, self.k))
        self.head = nn.Linear(self.width, self.classes)

    def encode(self, views: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        if raw.ndim != 2 or raw.shape[1] != self.dim or views.shape != (len(raw), self.k, self.dim):
            raise ValueError('Expected raw [N,D] and views [N,K,D] matching configured dimensions')
        if not torch.isfinite(raw).all() or not torch.isfinite(views).all():
            raise ValueError('Raw features and physical views must be finite')
        inputs = raw[:, None, :].expand(-1, self.k, -1) if self.arm == 'generic' else views
        return torch.stack([expert(inputs[:, slot]) for slot, expert in enumerate(self.experts)], dim=1)

    def forward(self, views: torch.Tensor, raw: torch.Tensor,
                compatibility: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encode(views, raw)
        if compatibility.shape != (len(raw), self.k) or not torch.isfinite(compatibility).all():
            raise ValueError('Expected finite compatibility [N,K]')
        if torch.any(compatibility.abs() > 1):
            raise ValueError('Compatibility must be a declared score in [-1,1]')
        logits = self.router(raw)
        if self.arm in {'aligned', 'no_balance'}:
            logits = logits + compatibility
        elif self.arm == 'shuffled':
            logits = logits + compatibility.roll(1, dims=1)
        gates = logits.softmax(-1)
        if self.arm == 'uniform':
            gates = torch.ones_like(gates) / self.k
        return self.head((gates[:, :, None] * features).sum(1)), gates, features

    def fixed_route_replacements(
        self, views: torch.Tensor, raw: torch.Tensor, clean_gates: torch.Tensor,
        clean_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return base probabilities, each single-slot replacement, and change norms.

        Both branches of a matched pair must supply the SAME clean gates/reference.
        The output axes are [N,C], [N,K,C], [N,K]; routing is never recomputed here.
        """
        features = self.encode(views, raw)
        if clean_gates.shape != (len(raw), self.k) or clean_features.shape != features.shape:
            raise ValueError('Clean route/reference axes must match the intervened experts')
        if not torch.isfinite(clean_gates).all() or not torch.isfinite(clean_features).all():
            raise ValueError('Clean route/reference must be finite')
        if torch.any(clean_gates < 0) or not torch.allclose(clean_gates.sum(-1), torch.ones_like(clean_gates[:, 0])):
            raise ValueError('Clean gates must be nonnegative and sum to one')
        mixture = (clean_gates[:, :, None] * features).sum(1)
        replaced = mixture[:, None, :] + clean_gates[:, :, None] * (clean_features - features)
        return (self.head(mixture).softmax(-1), self.head(replaced).softmax(-1),
                (features - clean_features).norm(dim=-1))
