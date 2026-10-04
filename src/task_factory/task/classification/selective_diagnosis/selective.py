"""SelectiveNet objective adapted to P4's feature inputs and unit weights.

This is an independent implementation of the normalized empirical-risk objective
in Geifman and El-Yaniv (ICML 2019), equations (1)--(2):
https://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf
It is not a reproduction of the authors' image-model implementation. In particular,
their public CIFAR-10 code at commit a6d0a8fd33dae61da910b61a2aae93102d2d4869
omits the coverage denominator in the selective loss. Here the paper's denominator
is retained. No code from that repository is copied.

P4 adaptation: a shared ReLU feature layer, separate prediction and auxiliary
heads, and a ReLU/sigmoid selection head. Batch normalization is omitted to keep
the feature backbone consistent with the existing MLP control. Supplied weights
define the empirical distribution; physical-unit weights must be computed before
calling the objective. With mini-batches this is a weighted batch-ratio estimate,
not an unbiased gradient of the complete training-set ratio.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SelectiveMLP(nn.Module):
    """Feature classifier with a learned selector and a training-only class head.

    ``aux_weight`` is the coefficient of auxiliary cross entropy; the coefficient
    of the selective loss is ``1 - aux_weight``. Both default to the paper's 0.5.
    Higher selection scores mean greater propensity to accept, whereas P4's
    risk-score interface uses ``1 - selection``.
    """

    def __init__(self, in_features: int, classes: int, width: int,
                 coverage: float = .9, coverage_penalty: float = 32.,
                 aux_weight: float = .5):
        super().__init__()
        if in_features < 1 or classes < 2 or width < 1:
            raise ValueError('positive feature/hidden dimensions and at least two classes required')
        if not math.isfinite(coverage) or not 0 < coverage <= 1:
            raise ValueError('coverage must be in (0, 1]')
        if not math.isfinite(coverage_penalty) or coverage_penalty <= 0:
            raise ValueError('coverage_penalty must be positive and finite')
        if not math.isfinite(aux_weight) or not 0 < aux_weight < 1:
            raise ValueError('aux_weight must be in (0, 1) to train both prediction heads')
        self.coverage = float(coverage)
        self.coverage_penalty = float(coverage_penalty)
        self.aux_weight = float(aux_weight)
        self.body = nn.Sequential(nn.Linear(in_features, width), nn.ReLU())
        self.prediction_head = nn.Linear(width, classes)
        self.selection_head = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1), nn.Sigmoid())
        self.auxiliary_head = nn.Linear(width, classes)

    def forward_components(self, x: torch.Tensor
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        representation = self.body(x)
        return (self.prediction_head(representation),
                self.selection_head(representation).squeeze(-1),
                self.auxiliary_head(representation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_components(x)[0]

    def objective(self, x: torch.Tensor, y: torch.Tensor,
                  weights: torch.Tensor) -> torch.Tensor:
        """Unit-weighted selective CE / soft coverage + shortfall + auxiliary CE.

        Weights may have any positive total mass. Normalizing them here preserves
        the same objective when callers rescale all weights. Zero accepted mass
        is an explicit numerical failure, not a zero-loss rejection solution.
        """
        if y.ndim != 1 or weights.shape != y.shape or len(x) != len(y) or not len(y):
            raise ValueError('nonempty x, y and weights must have the same sample axis')
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('weights must be finite and nonnegative')
        total = weights.sum()
        if not torch.isfinite(total) or total <= 0:
            raise ValueError('weights must have positive finite total mass')
        logits, selection, auxiliary_logits = self.forward_components(x)
        normalized = weights / total
        soft_coverage = (normalized * selection).sum()
        if not torch.isfinite(soft_coverage) or soft_coverage <= 0:
            raise FloatingPointError('selection has zero or nonfinite weighted mass')
        selective_ce = (normalized * selection * F.cross_entropy(
            logits, y, reduction='none')).sum() / soft_coverage
        shortfall = (self.coverage - soft_coverage).clamp_min(0).square()
        auxiliary_ce = (normalized * F.cross_entropy(
            auxiliary_logits, y, reduction='none')).sum()
        return ((1 - self.aux_weight) * (
            selective_ce + self.coverage_penalty * shortfall)
                + self.aux_weight * auxiliary_ce)
