from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FuzzyConfig:
    num_fuzzy_features: int = 8
    num_membership_functions: int = 3
    num_rules: int = 10
    logit_scale: float = 1.0
    antecedent_temperature: float = 1.0
    min_width: float = 1.0e-4
    firing_epsilon: float = 1.0e-12


@dataclass(frozen=True)
class FuzzyTrace:
    """All tensors needed to audit one fuzzy forward pass.

    The contribution identity is structural:

        fuzzy_logits[b, k] == sum_r rule_contributions[b, r, k]

    No nonlinear layer is applied after the rule sum.
    """

    reduced_features: torch.Tensor
    membership_values: torch.Tensor
    centers: torch.Tensor
    widths: torch.Tensor
    antecedent_probabilities: torch.Tensor
    antecedent_memberships: torch.Tensor
    log_rule_firing: torch.Tensor
    rule_firing: torch.Tensor
    normalized_rule_firing: torch.Tensor
    rule_consequents: torch.Tensor
    rule_contributions: torch.Tensor
    fuzzy_logits: torch.Tensor
    rule_mask: torch.Tensor
    consequent_permutation: torch.Tensor

    def reconstruct_fuzzy_logits(self) -> torch.Tensor:
        return self.rule_contributions.sum(dim=1)

    def reconstruction_residual(self) -> torch.Tensor:
        return self.fuzzy_logits - self.reconstruct_fuzzy_logits()

    def normalized_firing_entropy(self) -> torch.Tensor:
        weights = self.normalized_rule_firing.clamp_min(torch.finfo(self.fuzzy_logits.dtype).tiny)
        entropy = -(weights * weights.log()).sum(dim=1)
        num_rules = int(weights.shape[1])
        if num_rules <= 1:
            return torch.zeros_like(entropy)
        return entropy / math.log(num_rules)

    def top_rule_share(self) -> torch.Tensor:
        return self.normalized_rule_firing.max(dim=1).values


@dataclass(frozen=True)
class P05F0Decision:
    """No-bypass F0 decision and the exact tensors it consumed.

    ``issued_class`` is ``-1`` for abstention. Neural logits and learned vector
    consequents are intentionally absent from this record.
    """

    reduced_features: torch.Tensor
    membership_values: torch.Tensor
    antecedent_probabilities: torch.Tensor
    antecedent_memberships: torch.Tensor
    rule_activations: torch.Tensor
    rule_mask: torch.Tensor
    rule_to_class: torch.Tensor
    class_supports: torch.Tensor
    top_support: torch.Tensor
    second_support: torch.Tensor
    conflict: torch.Tensor
    candidate_class: torch.Tensor
    accepted: torch.Tensor
    issued_class: torch.Tensor
    conflict_threshold: float

    @property
    def abstained(self) -> torch.Tensor:
        return ~self.accepted


class FuzzyReasoner(nn.Module):
    """Additive Takagi--Sugeno-style fuzzy head over learned features.

    Each rule has an explicit soft antecedent over ordered Gaussian membership
    terms for every reduced feature. Rule firing is the geometric mean of the
    selected memberships. Normalized firing weights multiply class-specific
    consequents, so every class logit is exactly the sum of per-rule
    contributions recorded in FuzzyTrace.
    """

    def __init__(self, dim_in: int, num_classes: int, cfg: Optional[FuzzyConfig] = None):
        super().__init__()
        self.cfg = cfg or FuzzyConfig()
        self._validate_config(dim_in=dim_in, num_classes=num_classes)
        self.num_classes = int(num_classes)

        num_features = int(self.cfg.num_fuzzy_features)
        num_memberships = int(self.cfg.num_membership_functions)
        num_rules = int(self.cfg.num_rules)
        num_outputs = int(num_classes)

        if num_features == int(dim_in):
            self.feature_reducer = nn.LayerNorm(num_features)
        else:
            self.feature_reducer = nn.Sequential(
                nn.Linear(int(dim_in), num_features),
                nn.LayerNorm(num_features),
            )

        initial_centers = (
            torch.zeros(1)
            if num_memberships == 1
            else torch.linspace(-1.0, 1.0, steps=num_memberships)
        )
        initial_centers = initial_centers.repeat(num_features, 1)
        self.center_origin = nn.Parameter(initial_centers[:, :1].clone())
        if num_memberships > 1:
            initial_gap = float(2.0 / (num_memberships - 1))
            inverse_softplus_gap = math.log(math.expm1(initial_gap))
            self.center_deltas_unconstrained = nn.Parameter(
                torch.full((num_features, num_memberships - 1), inverse_softplus_gap)
            )
        else:
            self.register_parameter("center_deltas_unconstrained", None)

        initial_width = 0.75
        inverse_softplus_width = math.log(
            math.expm1(max(initial_width - float(self.cfg.min_width), 1.0e-6))
        )
        self.widths_unconstrained = nn.Parameter(
            torch.full((num_features, num_memberships), inverse_softplus_width)
        )

        antecedent_logits = torch.full(
            (num_rules, num_features, num_memberships),
            -2.0,
        )
        rule_index = torch.arange(num_rules).unsqueeze(1)
        feature_index = torch.arange(num_features).unsqueeze(0)
        initial_terms = (rule_index + feature_index) % num_memberships
        antecedent_logits.scatter_(2, initial_terms.unsqueeze(-1), 2.0)
        self.antecedent_logits = nn.Parameter(antecedent_logits)

        self.rule_consequents = nn.Parameter(torch.randn(num_rules, num_outputs) * 0.1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_with_trace(features).fuzzy_logits

    def forward_with_trace(
        self,
        features: torch.Tensor,
        *,
        rule_mask: Optional[torch.Tensor] = None,
        consequent_permutation: Optional[torch.Tensor] = None,
    ) -> FuzzyTrace:
        """Return logits and their complete additive rule trace.

        rule_mask implements rule deletion. It may have shape (R,) or
        (B, R); deleted rules are removed before firing normalization.
        consequent_permutation implements the negative-control shuffle by
        assigning a permutation of the learned consequent rows to the original
        rule firings. It may have shape (R,) for one shared permutation or
        (B, R) for one registered permutation per sample.
        """

        if features.ndim != 2:
            raise ValueError(
                f"FuzzyReasoner expects features with shape (batch, dim), got {tuple(features.shape)}."
            )

        reduced = self.feature_reducer(features)
        centers = self._ordered_centers()
        widths = F.softplus(self.widths_unconstrained) + float(self.cfg.min_width)
        membership = self._compute_membership(reduced, centers=centers, widths=widths)

        temperature = float(self.cfg.antecedent_temperature)
        antecedent_probabilities = F.softmax(self.antecedent_logits / temperature, dim=-1)
        antecedent_memberships = torch.einsum(
            "bfm,rfm->brf",
            membership,
            antecedent_probabilities,
        )
        log_rule_firing = antecedent_memberships.clamp_min(
            float(self.cfg.firing_epsilon)
        ).log().mean(dim=-1)
        rule_firing = log_rule_firing.exp()

        normalized_mask = self._normalize_rule_mask(
            rule_mask,
            batch_size=int(features.shape[0]),
            device=features.device,
        )
        masked_log_firing = log_rule_firing.masked_fill(~normalized_mask, -torch.inf)
        normalized_rule_firing = F.softmax(masked_log_firing, dim=-1)

        permutation = self._normalize_consequent_permutation(
            consequent_permutation,
            batch_size=int(features.shape[0]),
            device=features.device,
        )
        if permutation.ndim == 1:
            consequents = self.rule_consequents.index_select(0, permutation)
            contribution_consequents = consequents.unsqueeze(0)
        else:
            consequents = self.rule_consequents[permutation]
            contribution_consequents = consequents
        rule_contributions = (
            normalized_rule_firing.unsqueeze(-1) * contribution_consequents
        )
        fuzzy_logits = rule_contributions.sum(dim=1)

        return FuzzyTrace(
            reduced_features=reduced,
            membership_values=membership,
            centers=centers,
            widths=widths,
            antecedent_probabilities=antecedent_probabilities,
            antecedent_memberships=antecedent_memberships,
            log_rule_firing=log_rule_firing,
            rule_firing=rule_firing,
            normalized_rule_firing=normalized_rule_firing,
            rule_consequents=consequents,
            rule_contributions=rule_contributions,
            fuzzy_logits=fuzzy_logits,
            rule_mask=normalized_mask,
            consequent_permutation=permutation,
        )

    def forward_f0(
        self,
        features: torch.Tensor,
        *,
        rule_to_class: torch.Tensor,
        conflict_threshold: float,
        rule_mask: Optional[torch.Tensor] = None,
        consequent_override: Optional[torch.Tensor] = None,
    ) -> P05F0Decision:
        """Issue class/abstention using only the declared fuzzy-rule path.

        ``rule_to_class`` is the frozen training-only consequent mapping.
        ``consequent_override`` is an explicit intervention and never a
        fallback. An all-false mask is valid and produces abstention.
        """

        if self.num_classes < 2:
            raise ValueError("P05 F0 requires at least two classes.")
        threshold = self._validate_conflict_threshold(conflict_threshold)

        # This call records the same memberships and learned antecedent mixtures
        # used by the active fuzzy branch. F0 below does not consume its
        # geometric-mean firing, normalization, or learned vector consequents.
        trace = self.forward_with_trace(features)
        batch_size = int(features.shape[0])
        mask = self._normalize_f0_rule_mask(
            rule_mask,
            batch_size=batch_size,
            device=features.device,
        )
        effective_mapping = self._normalize_rule_to_class(
            consequent_override if consequent_override is not None else rule_to_class,
            batch_size=batch_size,
            device=features.device,
        )

        raw_activations = trace.antecedent_memberships.amin(dim=-1)
        rule_activations = raw_activations.masked_fill(~mask, 0.0)
        class_supports = torch.stack(
            [
                rule_activations.masked_fill(
                    effective_mapping.ne(class_id),
                    0.0,
                ).max(dim=1).values
                for class_id in range(self.num_classes)
            ],
            dim=1,
        )

        top_support, candidate_class = class_supports.max(dim=1)
        second_support = class_supports.topk(k=2, dim=1).values[:, 1]
        epsilon = float(self.cfg.firing_epsilon)
        conflict = (second_support + epsilon) / (top_support + epsilon)
        no_support = top_support <= 0.0
        conflict = torch.where(no_support, torch.ones_like(conflict), conflict)
        accepted = (~no_support) & (conflict <= threshold)
        issued_class = torch.where(
            accepted,
            candidate_class,
            torch.full_like(candidate_class, -1),
        )

        return P05F0Decision(
            reduced_features=trace.reduced_features,
            membership_values=trace.membership_values,
            antecedent_probabilities=trace.antecedent_probabilities,
            antecedent_memberships=trace.antecedent_memberships,
            rule_activations=rule_activations,
            rule_mask=mask,
            rule_to_class=effective_mapping,
            class_supports=class_supports,
            top_support=top_support,
            second_support=second_support,
            conflict=conflict,
            candidate_class=candidate_class,
            accepted=accepted,
            issued_class=issued_class,
            conflict_threshold=threshold,
        )

    def _ordered_centers(self) -> torch.Tensor:
        if self.center_deltas_unconstrained is None:
            return self.center_origin
        positive_deltas = F.softplus(self.center_deltas_unconstrained)
        return torch.cat(
            (
                self.center_origin,
                self.center_origin + positive_deltas.cumsum(dim=1),
            ),
            dim=1,
        )

    @staticmethod
    def _compute_membership(
        x: torch.Tensor,
        *,
        centers: torch.Tensor,
        widths: torch.Tensor,
    ) -> torch.Tensor:
        standardized = (x.unsqueeze(-1) - centers.unsqueeze(0)) / widths.unsqueeze(0)
        return torch.exp(-0.5 * standardized.square())

    def _normalize_rule_mask(
        self,
        rule_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        num_rules = int(self.cfg.num_rules)
        if rule_mask is None:
            return torch.ones((batch_size, num_rules), dtype=torch.bool, device=device)

        mask = torch.as_tensor(rule_mask, device=device)
        if mask.ndim == 1:
            if tuple(mask.shape) != (num_rules,):
                raise ValueError(
                    f"rule_mask with one dimension must have shape ({num_rules},), "
                    f"got {tuple(mask.shape)}."
                )
            mask = mask.unsqueeze(0).expand(batch_size, -1)
        elif tuple(mask.shape) != (batch_size, num_rules):
            raise ValueError(
                f"rule_mask must have shape ({num_rules},) or ({batch_size}, {num_rules}), "
                f"got {tuple(mask.shape)}."
            )

        mask = mask.to(dtype=torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("rule_mask must retain at least one rule for every sample.")
        return mask

    def _normalize_f0_rule_mask(
        self,
        rule_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        num_rules = int(self.cfg.num_rules)
        if rule_mask is None:
            return torch.ones(
                (batch_size, num_rules),
                dtype=torch.bool,
                device=device,
            )

        mask = torch.as_tensor(rule_mask, device=device)
        if mask.ndim == 1:
            if tuple(mask.shape) != (num_rules,):
                raise ValueError(
                    "rule_mask with one dimension must have shape "
                    f"({num_rules},), got {tuple(mask.shape)}."
                )
            mask = mask.unsqueeze(0).expand(batch_size, -1)
        elif tuple(mask.shape) != (batch_size, num_rules):
            raise ValueError(
                "rule_mask must have shape "
                f"({num_rules},) or ({batch_size}, {num_rules}), "
                f"got {tuple(mask.shape)}."
            )
        return mask.to(dtype=torch.bool)

    def _normalize_rule_to_class(
        self,
        mapping: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        num_rules = int(self.cfg.num_rules)
        raw_values = torch.as_tensor(mapping, device=device)
        if raw_values.dtype == torch.bool:
            raise TypeError("rule_to_class must contain integer class IDs, not booleans.")
        if raw_values.is_floating_point():
            if not bool(torch.isfinite(raw_values).all()) or not torch.equal(
                raw_values,
                raw_values.round(),
            ):
                raise ValueError("rule_to_class must contain finite integer class IDs.")
        values = raw_values.to(dtype=torch.long)
        if values.ndim == 1:
            if tuple(values.shape) != (num_rules,):
                raise ValueError(
                    f"rule_to_class must have shape ({num_rules},), "
                    f"got {tuple(values.shape)}."
                )
            values = values.unsqueeze(0).expand(batch_size, -1)
        elif tuple(values.shape) != (batch_size, num_rules):
            raise ValueError(
                "rule_to_class must have shape "
                f"({num_rules},) or ({batch_size}, {num_rules}), "
                f"got {tuple(values.shape)}."
            )
        if bool(((values < 0) | (values >= self.num_classes)).any()):
            raise ValueError(
                "rule_to_class must map every rule to a class in "
                f"[0, {self.num_classes})."
            )
        return values

    def _normalize_consequent_permutation(
        self,
        consequent_permutation: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        num_rules = int(self.cfg.num_rules)
        if consequent_permutation is None:
            return torch.arange(num_rules, dtype=torch.long, device=device)

        permutation = torch.as_tensor(
            consequent_permutation,
            dtype=torch.long,
            device=device,
        )
        valid_shape = tuple(permutation.shape) in {
            (num_rules,),
            (batch_size, num_rules),
        }
        if not valid_shape:
            raise ValueError(
                "consequent_permutation must have shape "
                f"({num_rules},) or ({batch_size}, {num_rules}), "
                f"got {tuple(permutation.shape)}."
            )
        expected = torch.arange(num_rules, dtype=torch.long, device=device)
        sorted_permutation = permutation.sort(dim=-1).values
        expected_permutation = (
            expected
            if permutation.ndim == 1
            else expected.unsqueeze(0).expand(batch_size, -1)
        )
        if not torch.equal(sorted_permutation, expected_permutation):
            raise ValueError(
                "each consequent_permutation row must contain every rule index exactly once."
            )
        return permutation

    @staticmethod
    def _validate_conflict_threshold(value: float) -> float:
        if isinstance(value, bool):
            raise TypeError("conflict_threshold must be a finite float in [0, 1].")
        threshold = float(value)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("conflict_threshold must be a finite float in [0, 1].")
        return threshold

    def _validate_config(self, *, dim_in: int, num_classes: int) -> None:
        integer_fields = {
            "dim_in": dim_in,
            "num_classes": num_classes,
            "num_fuzzy_features": self.cfg.num_fuzzy_features,
            "num_membership_functions": self.cfg.num_membership_functions,
            "num_rules": self.cfg.num_rules,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if float(self.cfg.antecedent_temperature) <= 0.0:
            raise ValueError("antecedent_temperature must be positive.")
        if float(self.cfg.min_width) <= 0.0:
            raise ValueError("min_width must be positive.")
        if not 0.0 < float(self.cfg.firing_epsilon) < 1.0:
            raise ValueError("firing_epsilon must lie strictly between zero and one.")
