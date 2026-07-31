"""Gradient-space constraints used by classification tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class GradientConstraintResult:
    norm: torch.Tensor
    scale: torch.Tensor


class FisherGradientConstraint:
    """Cap the diagonal-Fisher entrywise norm using one existing backward pass."""

    def __init__(self, epsilon: float = 2.0):
        self.epsilon = float(epsilon)
        if self.epsilon <= 0.0:
            raise ValueError("FIC epsilon must be positive")

    def apply(self, parameters: Iterable[nn.Parameter]) -> GradientConstraintResult:
        gradients = [
            parameter.grad
            for parameter in parameters
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients:
            zero = torch.tensor(0.0)
            return GradientConstraintResult(norm=zero, scale=torch.ones_like(zero))

        device = gradients[0].device
        norm = torch.zeros((), device=device, dtype=torch.float64)
        for gradient in gradients:
            detached = gradient.detach().to(device=device, dtype=torch.float64)
            if not torch.isfinite(detached).all():
                raise FloatingPointError("FIC received a non-finite gradient")
            norm = norm + detached.square().sum()

        epsilon = torch.tensor(self.epsilon, device=device, dtype=norm.dtype)
        scale = torch.minimum(torch.ones_like(norm), torch.sqrt(epsilon / norm.clamp_min(1e-30)))
        scale_for_grad = scale.to(dtype=gradients[0].dtype)
        if scale.item() < 1.0:
            with torch.no_grad():
                for gradient in gradients:
                    gradient.mul_(scale_for_grad.to(device=gradient.device, dtype=gradient.dtype))
        return GradientConstraintResult(
            norm=norm.to(dtype=torch.float32),
            scale=scale.to(dtype=torch.float32),
        )
