"""P08 shared-head physical conditioning; reuse the maintained HSE verbatim.

Input: x[B,L,1], positive fs[B], source-encoded condition[B,condition_dim].
Output: class logits[B,num_classes]. No file/domain identifier enters prediction.
The Data Factory owns source-fitted condition statistics; this module never fits
or looks up statistics in the supplied metadata. Inactive branches are retained
for shared initialization, not counted as matched effective model capacity.
"""
from __future__ import annotations
from types import SimpleNamespace
import torch
from torch import nn
from .embedding.E_01_HSE import E_01_HSE


class Model(nn.Module):
    def __init__(self, args_model, metadata=None):
        super().__init__()
        for key, expected in [('embedding','E_01_HSE'),('backbone','TransformerEncoderLayer'),('task_head','SharedLinear')]:
            if hasattr(args_model,key) and getattr(args_model,key) != expected:
                raise ValueError(f'P08 {key} must explicitly name {expected}')
        self.coordinates = args_model.coordinates
        self.fusion = args_model.fusion
        if self.coordinates not in {'physical', 'index'}:
            raise ValueError('coordinates must be physical or index')
        if self.fusion not in {'none', 'film', 'token_concat', 'late_concat'}:
            raise ValueError('fusion must be none, film, token_concat or late_concat')
        integers = ['output_dim', 'num_classes', 'condition_dim', 'patch_size_L', 'num_patches', 'nhead']
        for key in integers:
            value = getattr(args_model, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        width = args_model.output_dim
        if width % args_model.nhead or args_model.num_classes < 2:
            raise ValueError('output_dim must be divisible by nhead; num_classes>=2')
        self.condition_dim = args_model.condition_dim
        self.num_classes = args_model.num_classes
        self.task_id = getattr(args_model, 'model_task_id', 'classification')
        # Equal seeds initialize the shared representation before any conditional branch.
        self.embedding = E_01_HSE(SimpleNamespace(patch_size_L=args_model.patch_size_L,
            patch_size_C=1, num_patches=args_model.num_patches, output_dim=width))
        self.norm = nn.LayerNorm(width)
        self.backbone = nn.TransformerEncoderLayer(d_model=width, nhead=args_model.nhead,
            dim_feedforward=2*width, dropout=0., activation='gelu', batch_first=True)
        self.head = nn.Linear(width, self.num_classes)
        self.film = nn.Sequential(nn.Linear(self.condition_dim, width), nn.SiLU(), nn.Linear(width, 2*width))
        self.token_concat = nn.Sequential(nn.Linear(width+self.condition_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.late_concat = nn.Sequential(nn.Linear(width+self.condition_dim, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, x, file_id=None, task_id=None, *, fs=None, condition=None,
                detach_condition=False, start_indices_L=None, start_indices_C=None,
                return_features=False):
        # file_id is accepted only for Factory signature compatibility, never used.
        if task_id is not None and task_id != self.task_id:
            raise ValueError('P08 uses one shared, explicitly configured task head')
        if x.ndim != 3 or x.shape[2] != 1 or x.shape[1] < 2 or not torch.isfinite(x).all():
            raise ValueError('P08 requires finite x[B,L,1] with L>=2; no channel guessing or padding')
        if not isinstance(detach_condition, bool):
            raise TypeError('detach_condition must be a boolean')
        if self.coordinates == 'physical':
            if fs is None:
                raise ValueError('Physical coordinates require measured fs')
            coordinate_rate = fs
        else:
            # Same HSE, j/(L-1) coordinates. Neither resampling nor measured fs changes.
            coordinate_rate = x.shape[1]-1
        h = self.embedding(x, coordinate_rate, start_indices_L=start_indices_L,
                           start_indices_C=start_indices_C)
        v = self.norm(h)
        conditioned = self.fusion != 'none' and not detach_condition
        if self.fusion != 'none':
            if condition is None or condition.shape != (len(x), self.condition_dim):
                raise ValueError('Expected source-encoded condition[B,condition_dim]')
            if condition.device != x.device or condition.dtype != x.dtype or not torch.isfinite(condition).all():
                raise ValueError('condition must share x dtype/device and contain finite values')
        if conditioned:
            if self.fusion == 'film':
                gamma, beta = self.film(condition).chunk(2, dim=-1)
                v = (1+gamma[:, None, :])*v + beta[:, None, :]
            elif self.fusion == 'token_concat':
                p = condition[:, None, :].expand(-1, v.shape[1], -1)
                v = v + self.token_concat(torch.cat((v, p), dim=-1))
        z = self.backbone(v).mean(dim=1)
        if conditioned and self.fusion == 'late_concat':
            z = self.late_concat(torch.cat((z, condition), dim=-1))
        logits = self.head(z)
        return (logits, z) if return_features else logits

    def parameter_counts(self):
        shared = sum(p.numel() for module in (self.embedding, self.norm, self.backbone, self.head) for p in module.parameters())
        branch = {'film': self.film, 'token_concat': self.token_concat, 'late_concat': self.late_concat}
        conditional = 0 if self.fusion == 'none' else sum(p.numel() for p in branch[self.fusion].parameters())
        return {'stored_parameters': sum(p.numel() for p in self.parameters()),
                'active_parameters': shared+conditional, 'active_conditional_parameters': conditional}
