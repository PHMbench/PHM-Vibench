"""Conditional noise predictor using the maintained 1D convolutional backbone."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .phm_cfm_mlp1d import Model as ConditionedConvField


class Model(nn.Module):
    """Predict epsilon in [N,C,L]; the DDPM Task owns diffusion semantics.

    Only the stateless layers are shared with the CFM model. This module has no
    flow interpolation, velocity objective, or ODE sampler.
    """

    def __init__(self, args_model: Any, metadata: Any = None) -> None:
        super().__init__()
        self.noise_predictor = ConditionedConvField(args_model, metadata)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.noise_predictor(x_t, t, condition)
