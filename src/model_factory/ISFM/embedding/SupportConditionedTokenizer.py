"""Matched S/U token maps on source-projected physical cells.

Inputs are common and incremental coordinates [B,K,P], before the single
source-fitted RMS division. Availability describes the entire incremental block,
not padded time tokens. Projection and RMS fitting belong to the data protocol.
"""
from __future__ import annotations

import torch
from torch import nn


class SupportConditionedTokenizer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mode = args.token_organization
        self.patch_size = args.patch_size_L
        self.num_patches = args.num_patches
        if self.mode not in ('support', 'ordinary'):
            raise ValueError('token_organization must be support or ordinary')
        if args.output_dim < 2 or args.output_dim % 2:
            raise ValueError('output_dim must be positive and even')
        if min(self.patch_size, self.num_patches, args.d_model) < 1:
            raise ValueError('positive P, K and branch hidden width required')
        # Same family, depth, bias and initialization order in both arms.
        self.common_branch = nn.Sequential(
            nn.Linear(self.patch_size, args.d_model), nn.GELU(),
            nn.Linear(args.d_model, args.output_dim // 2),
        )
        self.increment_branch = nn.Sequential(
            nn.Linear(self.patch_size, args.d_model), nn.GELU(),
            nn.Linear(args.d_model, args.output_dim // 2),
        )
        rms = torch.as_tensor(args.source_rms, dtype=torch.float64)
        if rms.ndim != 0 or not torch.isfinite(rms) or rms <= 0:
            raise ValueError('source_rms must be a finite positive source-training scalar')
        self.register_buffer('source_rms', rms.clone())

    def forward(self, common: torch.Tensor, incremental: torch.Tensor,
                availability: torch.Tensor) -> torch.Tensor:
        expected = (self.num_patches, self.patch_size)
        if common.ndim != 3 or tuple(common.shape[1:]) != expected or common.shape[0] < 1:
            raise ValueError(f'common must have shape [B,{expected[0]},{expected[1]}]')
        if incremental.shape != common.shape or incremental.device != common.device or incremental.dtype != common.dtype:
            raise ValueError('incremental must match common shape, device and dtype')
        gate = torch.as_tensor(availability, device=common.device)
        if gate.shape != (len(common),) or not ((gate == 0) | (gate == 1)).all():
            raise ValueError('availability must contain one binary value per observation')
        gate = gate[:, None, None].bool()
        # Mask before either branch: even unavailable NaN/Inf must not enter
        # a Linear backward or contaminate the ordinary arm's common branch.
        admitted = torch.where(gate, incremental, torch.zeros_like(incremental))
        if not torch.isfinite(common).all() or not torch.isfinite(admitted).all():
            raise ValueError('observed common/incremental coordinates must be finite')
        scale = self.source_rms.to(dtype=common.dtype)
        c, p = common / scale, admitted / scale
        first, second = (c, p) if self.mode == 'support' else (c + p, c + p)
        zc = self.common_branch(first)
        zp = self.increment_branch(second)
        return torch.cat((zc, torch.where(gate, zp, torch.zeros_like(zp))), dim=-1)
