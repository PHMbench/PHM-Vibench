"""
PromptSelector: Discrete or continuous selection over prompt candidates.

This module implements the selection / weighting logic described in the spec.
It supports:

- **Hard selection** (discrete) via Gumbel-Softmax / straight-through argmax.
- **Soft mixing** (continuous) via softmax attention weights.

Regularisation terms (entropy, balance, sparsity) are exposed for integration
with the overall loss pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


SelectionMode = Literal["none", "hard", "soft"]


@dataclass
class PromptSelectionOutput:
    """Lightweight container for selector outputs."""

    features: torch.Tensor  # (B, N, D)
    weights: Optional[torch.Tensor] = None  # (B, M)
    logits: Optional[torch.Tensor] = None  # (B, M)
    regularization: Optional[Dict[str, torch.Tensor]] = None


class PromptSelector(nn.Module):
    """
    Select or mix prompt-conditioned feature representations.

    Args:
        feature_dim: Input feature dimension (after backbone). Used for pooling MLP.
        hidden_dim: Hidden width for the scoring network.
        mode: ``"none"``, ``"hard"``, or ``"soft"``.
        temperature: Temperature for Gumbel-Softmax (hard) or softmax (soft).
        entropy_weight: Coefficient for entropy regularisation.
        balance_weight: Coefficient for balance regularisation (hard mode).
        sparsity_weight: Coefficient for ℓ₁ sparsity regularisation (soft mode).
        eps: Numerical epsilon for log operations.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: Optional[int] = None,
        mode: SelectionMode = "none",
        temperature: float = 1.0,
        entropy_weight: float = 0.0,
        balance_weight: float = 0.0,
        sparsity_weight: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if mode not in {"none", "hard", "soft"}:
            raise ValueError(f"Unsupported selection mode '{mode}'")

        if hidden_dim is None:
            hidden_dim = max(64, feature_dim // 2)

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.balance_weight = balance_weight
        self.sparsity_weight = sparsity_weight
        self.eps = eps

        self.scorer = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.scorer:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> PromptSelectionOutput:
        """
        Select or mix prompt-conditioned features.

        Args:
            features: Tensor of shape ``(B, M, N, D)`` produced by the backbone.

        Returns:
            PromptSelectionOutput containing the resulting feature tensor and
            selection metadata.
        """
        if self.mode == "none":
            if features.dim() == 4:
                if features.size(1) == 1:
                    squeezed = features.squeeze(1)
                else:
                    squeezed = features[:, 0]
            elif features.dim() == 3:
                squeezed = features
            else:
                raise ValueError("features must have shape (B, M, N, D) or (B, N, D)")
            return PromptSelectionOutput(features=squeezed)

        if features.dim() != 4:
            raise ValueError("features must have shape (B, M, N, D)")

        B, M, N, D = features.shape

        pooled = features.mean(dim=2)  # (B, M, D)
        logits = self.scorer(pooled).squeeze(-1)  # (B, M)

        if self.mode == "hard":
            return self._select_hard(features, logits)
        return self._select_soft(features, logits)

    # ------------------------------------------------------------------
    # Hard selection
    # ------------------------------------------------------------------
    def _select_hard(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
    ) -> PromptSelectionOutput:
        B, M, N, D = features.shape
        if self.training:
            weights = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
        else:
            indices = torch.argmax(logits, dim=-1)
            weights = F.one_hot(indices, num_classes=M).float()

        indices = torch.argmax(weights, dim=-1).view(B, 1, 1, 1)
        gathered = torch.gather(
            features,
            dim=1,
            index=indices.expand(-1, 1, N, D),
        ).squeeze(1)

        regularization = self._hard_regularization(logits, weights)

        return PromptSelectionOutput(
            features=gathered,
            weights=weights,
            logits=logits,
            regularization=regularization,
        )

    def _hard_regularization(
        self,
        logits: torch.Tensor,
        weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        probs = F.softmax(logits, dim=-1)
        reg_terms: Dict[str, torch.Tensor] = {}

        if self.entropy_weight > 0.0:
            entropy = -(probs * torch.log(probs + self.eps)).sum(dim=-1).mean()
            reg_terms["entropy"] = entropy * self.entropy_weight

        if self.balance_weight > 0.0:
            mean_prob = probs.mean(dim=0)
            target = torch.full_like(mean_prob, 1.0 / probs.size(-1))
            balance = torch.abs(mean_prob - target).sum()
            reg_terms["balance"] = balance * self.balance_weight

        return reg_terms

    # ------------------------------------------------------------------
    # Soft selection
    # ------------------------------------------------------------------
    def _select_soft(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
    ) -> PromptSelectionOutput:
        weights = F.softmax(logits / max(self.temperature, 1e-6), dim=-1)
        mixed = torch.einsum("bm,bmnd->bnd", weights, features)

        regularization = self._soft_regularization(weights, logits)

        return PromptSelectionOutput(
            features=mixed,
            weights=weights,
            logits=logits,
            regularization=regularization,
        )

    def _soft_regularization(
        self,
        weights: torch.Tensor,
        logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        reg_terms: Dict[str, torch.Tensor] = {}

        if self.entropy_weight > 0.0:
            entropy = -(weights * torch.log(weights + self.eps)).sum(dim=-1).mean()
            reg_terms["entropy"] = entropy * self.entropy_weight

        if self.sparsity_weight > 0.0:
            sparsity = torch.abs(logits).mean()
            reg_terms["sparsity"] = sparsity * self.sparsity_weight

        return reg_terms


__all__ = ["PromptSelector", "PromptSelectionOutput", "SelectionMode"]
