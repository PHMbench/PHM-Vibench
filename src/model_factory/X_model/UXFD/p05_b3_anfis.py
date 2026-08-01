"""Frozen end-to-end ANFIS-like head for the P05-B3 control arm."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


P05_B3_FEATURES = 8
P05_B3_MEMBERSHIPS = 3
P05_B3_RULES = 10
P05_B3_PARAMETER_COUNT_BY_CLASSES = {2: 484, 4: 664}


@dataclass(frozen=True)
class P05B3ANFISConfig:
    """The approved F8/M3/R10 antecedent contract."""

    num_features: int = P05_B3_FEATURES
    num_membership_functions: int = P05_B3_MEMBERSHIPS
    num_rules: int = P05_B3_RULES
    antecedent_temperature: float = 1.0
    min_width: float = 1.0e-4
    firing_epsilon: float = 1.0e-12


@dataclass(frozen=True)
class P05B3ANFISTrace:
    """Reconstruction-only trace; intentionally has no risk-score methods."""

    reduced_features: torch.Tensor
    membership_values: torch.Tensor
    centers: torch.Tensor
    widths: torch.Tensor
    antecedent_probabilities: torch.Tensor
    antecedent_memberships: torch.Tensor
    log_rule_firing: torch.Tensor
    rule_firing: torch.Tensor
    normalized_rule_firing: torch.Tensor
    consequent_coefficients: torch.Tensor
    consequent_bias: torch.Tensor
    rule_outputs: torch.Tensor
    rule_contributions: torch.Tensor
    logits: torch.Tensor

    def reconstruct_logits(self) -> torch.Tensor:
        return self.rule_contributions.sum(dim=1)

    def reconstruction_residual(self) -> torch.Tensor:
        return self.logits - self.reconstruct_logits()


class P05B3ANFISHead(nn.Module):
    """First-order Takagi-Sugeno class head over the same eight features."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        cfg: Optional[P05B3ANFISConfig] = None,
    ) -> None:
        super().__init__()
        self.input_dim = self._integer(input_dim, name="input_dim")
        self.num_classes = self._integer(num_classes, name="num_classes")
        self.cfg = cfg or P05B3ANFISConfig()
        self._validate_contract()

        self.feature_reducer = nn.LayerNorm(P05_B3_FEATURES)
        initial_centers = torch.linspace(
            -1.0,
            1.0,
            steps=P05_B3_MEMBERSHIPS,
        ).repeat(P05_B3_FEATURES, 1)
        self.center_origin = nn.Parameter(initial_centers[:, :1].clone())
        initial_gap = float(2.0 / (P05_B3_MEMBERSHIPS - 1))
        inverse_softplus_gap = math.log(math.expm1(initial_gap))
        self.center_deltas_unconstrained = nn.Parameter(
            torch.full(
                (P05_B3_FEATURES, P05_B3_MEMBERSHIPS - 1),
                inverse_softplus_gap,
            )
        )

        initial_width = 0.75
        inverse_softplus_width = math.log(
            math.expm1(max(initial_width - self.cfg.min_width, 1.0e-6))
        )
        self.widths_unconstrained = nn.Parameter(
            torch.full(
                (P05_B3_FEATURES, P05_B3_MEMBERSHIPS),
                inverse_softplus_width,
            )
        )

        antecedent_logits = torch.full(
            (P05_B3_RULES, P05_B3_FEATURES, P05_B3_MEMBERSHIPS),
            -2.0,
        )
        rule_index = torch.arange(P05_B3_RULES).unsqueeze(1)
        feature_index = torch.arange(P05_B3_FEATURES).unsqueeze(0)
        initial_terms = (rule_index + feature_index) % P05_B3_MEMBERSHIPS
        antecedent_logits.scatter_(2, initial_terms.unsqueeze(-1), 2.0)
        self.antecedent_logits = nn.Parameter(antecedent_logits)

        self.consequent_coefficients = nn.Parameter(
            torch.randn(
                P05_B3_RULES,
                self.num_classes,
                P05_B3_FEATURES,
            )
            * 0.1
        )
        self.consequent_bias = nn.Parameter(
            torch.zeros(P05_B3_RULES, self.num_classes)
        )

        expected_count = P05_B3_PARAMETER_COUNT_BY_CLASSES[self.num_classes]
        if self.parameter_count != expected_count:
            raise RuntimeError(
                f"P05-B3 parameter contract drift: expected {expected_count}, "
                f"got {self.parameter_count}"
            )

    @staticmethod
    def _integer(value: object, *, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"P05-B3 {name} must be an integer")
        return int(value)

    def _validate_contract(self) -> None:
        expected = {
            "num_features": P05_B3_FEATURES,
            "num_membership_functions": P05_B3_MEMBERSHIPS,
            "num_rules": P05_B3_RULES,
            "antecedent_temperature": 1.0,
            "min_width": 1.0e-4,
            "firing_epsilon": 1.0e-12,
        }
        if self.input_dim != P05_B3_FEATURES:
            raise ValueError(
                f"P05-B3 requires the same eight-feature input, got {self.input_dim}"
            )
        if self.num_classes not in P05_B3_PARAMETER_COUNT_BY_CLASSES:
            raise ValueError(
                "P05-B3 is frozen only for XJTU K=2 and CWRU K=4, "
                f"got K={self.num_classes}"
            )
        for name, value in expected.items():
            if getattr(self.cfg, name) != value:
                raise ValueError(
                    f"P05-B3 frozen contract requires {name}={value!r}, "
                    f"got {getattr(self.cfg, name)!r}"
                )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _ordered_centers(self) -> torch.Tensor:
        positive_deltas = F.softplus(self.center_deltas_unconstrained)
        return torch.cat(
            (
                self.center_origin,
                self.center_origin + positive_deltas.cumsum(dim=1),
            ),
            dim=1,
        )

    @staticmethod
    def _membership(
        features: torch.Tensor,
        *,
        centers: torch.Tensor,
        widths: torch.Tensor,
    ) -> torch.Tensor:
        standardized = (
            features.unsqueeze(-1) - centers.unsqueeze(0)
        ) / widths.unsqueeze(0)
        return torch.exp(-0.5 * standardized.square())

    def forward_with_trace(self, features: torch.Tensor) -> P05B3ANFISTrace:
        if features.ndim != 2 or int(features.shape[1]) != self.input_dim:
            raise ValueError(
                "P05-B3 ANFIS head expects features with shape "
                f"(batch, {self.input_dim}), got {tuple(features.shape)}"
            )
        reduced = self.feature_reducer(features)
        centers = self._ordered_centers()
        widths = F.softplus(self.widths_unconstrained) + self.cfg.min_width
        membership = self._membership(reduced, centers=centers, widths=widths)
        antecedent_probabilities = F.softmax(
            self.antecedent_logits / self.cfg.antecedent_temperature,
            dim=-1,
        )
        antecedent_memberships = torch.einsum(
            "bfm,rfm->brf",
            membership,
            antecedent_probabilities,
        )
        log_rule_firing = antecedent_memberships.clamp_min(
            self.cfg.firing_epsilon
        ).log().mean(dim=-1)
        rule_firing = log_rule_firing.exp()
        normalized_rule_firing = F.softmax(log_rule_firing, dim=-1)

        rule_outputs = torch.einsum(
            "bf,rkf->brk",
            reduced,
            self.consequent_coefficients,
        ) + self.consequent_bias.unsqueeze(0)
        rule_contributions = normalized_rule_firing.unsqueeze(-1) * rule_outputs
        logits = rule_contributions.sum(dim=1)
        return P05B3ANFISTrace(
            reduced_features=reduced,
            membership_values=membership,
            centers=centers,
            widths=widths,
            antecedent_probabilities=antecedent_probabilities,
            antecedent_memberships=antecedent_memberships,
            log_rule_firing=log_rule_firing,
            rule_firing=rule_firing,
            normalized_rule_firing=normalized_rule_firing,
            consequent_coefficients=self.consequent_coefficients,
            consequent_bias=self.consequent_bias,
            rule_outputs=rule_outputs,
            rule_contributions=rule_contributions,
            logits=logits,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_with_trace(features).logits
