"""
Contrastive learning loss functions for PHM-Vibench framework.

This module provides various contrastive learning loss functions including
InfoNCE, SimCLR-style losses, and other self-supervised learning objectives.

Author: PHM-Vibench Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss for contrastive learning.

    This implementation supports both symmetric and asymmetric formulations
    of the InfoNCE loss, commonly used in self-supervised representation learning.

    Reference:
        Oord et al. "Representation Learning with Contrastive Predictive Coding"
        Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations"
    """

    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = 'mean',
        symmetric: bool = True,
        eps: float = 1e-8
    ):
        """
        Initialize InfoNCE loss.

        Args:
            temperature: Temperature parameter for scaling similarities
            reduction: Reduction method ('mean', 'sum', 'none')
            symmetric: Whether to use symmetric InfoNCE (both directions)
            eps: Small value for numerical stability
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.symmetric = symmetric
        self.eps = eps

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Unsupported reduction: {reduction}")

    def forward(
        self,
        z_anchor: torch.Tensor,
        z_positive: torch.Tensor,
        z_negative: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            z_anchor: [B, D] anchor representations
            z_positive: [B, D] positive representations
            z_negative: [N, D] negative representations (optional)

        Returns:
            InfoNCE loss tensor
        """
        batch_size = z_anchor.size(0)

        # L2 normalize representations
        z_anchor = F.normalize(z_anchor, dim=1, eps=self.eps)
        z_positive = F.normalize(z_positive, dim=1, eps=self.eps)

        if z_negative is not None:
            z_negative = F.normalize(z_negative, dim=1, eps=self.eps)
            # Concatenate positive and negative samples
            z_all = torch.cat([z_positive, z_negative], dim=0)
        else:
            # Use in-batch negatives (standard approach)
            z_all = z_positive

        # Compute similarity matrix
        similarity_matrix = torch.mm(z_anchor, z_all.t()) / self.temperature

        # Positive pairs are on the diagonal (for in-batch negatives)
        if z_negative is None:
            positive_indices = torch.arange(batch_size, device=z_anchor.device)
            positive_similarities = similarity_matrix[torch.arange(batch_size), positive_indices]
        else:
            # Positive samples are the first batch_size columns
            positive_similarities = similarity_matrix[:, :batch_size].diagonal()

        # Compute InfoNCE loss
        logsumexp = torch.logsumexp(similarity_matrix, dim=1)
        loss = -positive_similarities + logsumexp

        # Symmetric version: compute loss in both directions
        if self.symmetric and z_negative is None:
            similarity_matrix_t = similarity_matrix.t()
            positive_similarities_t = similarity_matrix_t.diagonal()
            logsumexp_t = torch.logsumexp(similarity_matrix_t, dim=1)
            loss_t = -positive_similarities_t + logsumexp_t
            loss = (loss + loss_t) / 2

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SimCLRLoss(InfoNCELoss):
    """
    SimCLR-style contrastive loss.

    This is essentially InfoNCE with symmetric=True and in-batch negatives.
    Provided as a separate class for clarity and potential future extensions.
    """

    def __init__(self, temperature: float = 0.1, **kwargs):
        super().__init__(temperature=temperature, symmetric=True, **kwargs)


class SupConLoss(nn.Module):
    """
    Supervised contrastive loss.

    Extension of InfoNCE for supervised settings where multiple positive
    samples per anchor can be identified using labels.

    Reference:
        Khosla et al. "Supervised Contrastive Learning"
    """

    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = 'mean',
        base_temperature: float = 0.07
    ):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            features: [B, D] normalized feature representations
            labels: [B] ground truth labels
            mask: [B, B] optional mask for valid pairs

        Returns:
            Supervised contrastive loss
        """
        batch_size = features.shape[0]

        if mask is None:
            # Create mask based on labels
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float()

        # Remove diagonal elements (self-similarity)
        mask = mask - torch.eye(batch_size, device=mask.device)

        # Compute similarities
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Compute log probabilities
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# Factory function for easy integration
def get_contrastive_loss(
    loss_type: str,
    temperature: float = 0.07,
    **kwargs
) -> nn.Module:
    """
    Factory function to create contrastive loss functions.

    Args:
        loss_type: Type of contrastive loss ('infonce', 'simclr', 'supcon')
        temperature: Temperature parameter
        **kwargs: Additional arguments passed to loss function

    Returns:
        Contrastive loss module
    """
    loss_type = loss_type.lower()

    if loss_type == 'infonce':
        return InfoNCELoss(temperature=temperature, **kwargs)
    elif loss_type == 'simclr':
        return SimCLRLoss(temperature=temperature, **kwargs)
    elif loss_type == 'supcon':
        return SupConLoss(temperature=temperature, **kwargs)
    else:
        raise ValueError(f"Unsupported contrastive loss type: {loss_type}")


if __name__ == "__main__":
    # Test InfoNCE loss
    batch_size, feature_dim = 4, 128

    z_anchor = F.normalize(torch.randn(batch_size, feature_dim), dim=1)
    z_positive = F.normalize(torch.randn(batch_size, feature_dim), dim=1)

    # Test InfoNCE
    infonce = InfoNCELoss(temperature=0.07)
    loss_infonce = infonce(z_anchor, z_positive)
    print(f"InfoNCE Loss: {loss_infonce:.4f}")

    # Test SimCLR
    simclr = SimCLRLoss(temperature=0.1)
    loss_simclr = simclr(z_anchor, z_positive)
    print(f"SimCLR Loss: {loss_simclr:.4f}")

    # Test SupCon
    labels = torch.randint(0, 3, (batch_size,))
    features = torch.cat([z_anchor, z_positive], dim=0)
    labels_full = torch.cat([labels, labels], dim=0)

    supcon = SupConLoss(temperature=0.07)
    loss_supcon = supcon(features, labels_full)
    print(f"SupCon Loss: {loss_supcon:.4f}")