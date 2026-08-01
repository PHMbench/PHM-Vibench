"""Frozen parameter-matched neural residual for the P05-B1 control arm."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Optional

import torch
import torch.nn as nn


P05_B1_INPUT_DIM = 8
P05_B1_HIDDEN_BY_CLASSES = {2: 29, 4: 26}
P05_B1_PARAMETER_COUNT_BY_CLASSES = {2: 321, 4: 342}


@dataclass(frozen=True)
class P05B1NeuralResidualConfig:
    """Configuration whose only optional value must match the frozen contract."""

    hidden_dim: Optional[int] = None


class P05B1NeuralResidual(nn.Module):
    """``Linear(8,H)-GELU-Linear(H,K)`` residual with no spare parameters."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        cfg: Optional[P05B1NeuralResidualConfig] = None,
    ) -> None:
        super().__init__()
        self.input_dim = self._integer(input_dim, name="input_dim")
        self.num_classes = self._integer(num_classes, name="num_classes")
        if self.input_dim != P05_B1_INPUT_DIM:
            raise ValueError(
                f"P05-B1 requires the same eight-feature input, got {self.input_dim}"
            )
        if self.num_classes not in P05_B1_HIDDEN_BY_CLASSES:
            raise ValueError(
                "P05-B1 is frozen only for XJTU K=2 and CWRU K=4, "
                f"got K={self.num_classes}"
            )

        contract_hidden = P05_B1_HIDDEN_BY_CLASSES[self.num_classes]
        configured_hidden = (cfg or P05B1NeuralResidualConfig()).hidden_dim
        if configured_hidden is None:
            self.hidden_dim = contract_hidden
        else:
            self.hidden_dim = self._integer(configured_hidden, name="hidden_dim")
            if self.hidden_dim != contract_hidden:
                raise ValueError(
                    f"P05-B1 K={self.num_classes} requires H={contract_hidden}, "
                    f"got H={self.hidden_dim}"
                )

        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        expected_count = P05_B1_PARAMETER_COUNT_BY_CLASSES[self.num_classes]
        if self.parameter_count != expected_count:
            raise RuntimeError(
                f"P05-B1 parameter contract drift: expected {expected_count}, "
                f"got {self.parameter_count}"
            )

    @staticmethod
    def _integer(value: object, *, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"P05-B1 {name} must be an integer")
        return int(value)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or int(features.shape[1]) != self.input_dim:
            raise ValueError(
                "P05-B1 neural residual expects features with shape "
                f"(batch, {self.input_dim}), got {tuple(features.shape)}"
            )
        return self.network(features)
