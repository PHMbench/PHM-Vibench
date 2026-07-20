"""
Semantic Layer Alignment for 1D-2D Fusion Fault Diagnosis

This module implements semantic alignment mechanisms that ensure the 1D time-domain
and 2D frequency-domain representations capture semantically meaningful information
and maintain consistent feature spaces across modalities.

Key concepts:
- Cross-modal contrastive learning
- Feature space alignment
- Semantic similarity preservation
- Class-wise representation consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, List


class SemanticAlignmentLoss(nn.Module):
    """
    Semantic alignment loss that enforces semantic consistency between 1D and 2D
    representations using contrastive learning and feature space alignment.
    """

    def __init__(self,
                 temperature: float = 0.1,
                 contrastive_weight: float = 1.0,
                 alignment_weight: float = 0.5,
                 consistency_weight: float = 0.3,
                 margin: float = 0.5):
        """
        Initialize semantic alignment loss.

        Args:
            temperature: Temperature for contrastive loss
            contrastive_weight: Weight for contrastive learning loss
            alignment_weight: Weight for feature alignment loss
            consistency_weight: Weight for prediction consistency loss
            margin: Margin for triplet loss
        """
        super().__init__()
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.alignment_weight = alignment_weight
        self.consistency_weight = consistency_weight
        self.margin = margin

        # Projection heads for contrastive learning
        self.projection_1d = self._build_projection_head()
        self.projection_2d = self._build_projection_head()

    def _build_projection_head(self) -> nn.Module:
        """Build projection head for contrastive learning."""
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self,
                feat_1d: torch.Tensor,
                feat_2d: torch.Tensor,
                labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute semantic alignment loss.

        Args:
            feat_1d: 1D features [B, C1, L1] or [B, C1]
            feat_2d: 2D features [B, C2, H2, W2] or [B, C2]
            labels: Class labels [B]

        Returns:
            Dictionary containing individual loss components and total loss
        """
        losses = {}

        # Flatten features if needed
        feat_1d_flat = self._flatten_features(feat_1d)
        feat_2d_flat = self._flatten_features(feat_2d)

        # Ensure compatible dimensions for projection
        feat_1d_proj_ready = self._align_dimensions(feat_1d_flat, 512)
        feat_2d_proj_ready = self._align_dimensions(feat_2d_flat, 512)

        # Project features to common space
        proj_1d = self.projection_1d(feat_1d_proj_ready)
        proj_2d = self.projection_2d(feat_2d_proj_ready)

        # Cross-modal contrastive loss
        contrastive_loss = self._cross_modal_contrastive_loss(proj_1d, proj_2d, labels)
        losses['contrastive'] = self.contrastive_weight * contrastive_loss

        # Feature space alignment loss
        alignment_loss = self._feature_alignment_loss(proj_1d, proj_2d, labels)
        losses['alignment'] = self.alignment_weight * alignment_loss

        # Prediction consistency loss
        consistency_loss = self._prediction_consistency_loss(feat_1d_flat, feat_2d_flat, labels)
        losses['consistency'] = self.consistency_weight * consistency_loss

        # Total semantic alignment loss
        losses['total'] = losses['contrastive'] + losses['alignment'] + losses['consistency']

        return losses

    def _flatten_features(self, features: torch.Tensor) -> torch.Tensor:
        """Flatten features to 2D tensor [B, C*]."""
        if features.dim() == 3:  # [B, C, L]
            return features.view(features.size(0), -1)
        elif features.dim() == 4:  # [B, C, H, W]
            return features.view(features.size(0), -1)
        elif features.dim() == 2:  # [B, C]
            return features
        else:
            raise ValueError(f"Unsupported feature dimension: {features.dim()}")

    def _align_dimensions(self, features: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Align feature dimensions to target dimension."""
        current_dim = features.size(-1)
        if current_dim == target_dim:
            return features
        elif current_dim > target_dim:
            return features[..., :target_dim]
        else:
            # Pad with zeros
            padding = target_dim - current_dim
            return F.pad(features, (0, padding))

    def _cross_modal_contrastive_loss(self,
                                    proj_1d: torch.Tensor,
                                    proj_2d: torch.Tensor,
                                    labels: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-modal contrastive loss.
        Positive pairs: same sample across modalities
        Negative pairs: different samples across modalities
        """
        batch_size = proj_1d.size(0)

        # Normalize features
        proj_1d_norm = F.normalize(proj_1d, dim=-1)
        proj_2d_norm = F.normalize(proj_2d, dim=-1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(proj_1d_norm, proj_2d_norm.T)  # [B, B]

        # Apply temperature scaling
        similarity_matrix = similarity_matrix / self.temperature

        # Create positive and negative masks
        labels_expanded = labels.unsqueeze(1).expand(batch_size, batch_size)
        positive_mask = torch.eye(batch_size, device=labels.device).bool()
        negative_mask = (labels_expanded != labels_expanded.T) & (~positive_mask)

        # Compute InfoNCE loss
        # For each 1D feature, match with corresponding 2D feature
        loss_1d_to_2d = 0
        loss_2d_to_1d = 0

        for i in range(batch_size):
            # Positive pair: (1d_i, 2d_i)
            pos_sim = similarity_matrix[i, i]

            # Negative pairs: (1d_i, 2d_j) where j != i and labels[j] != labels[i]
            neg_sims = similarity_matrix[i, negative_mask[i]]

            # InfoNCE loss
            logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
            labels_contrastive = torch.zeros_like(logits, dtype=torch.long)
            loss_1d_to_2d += F.cross_entropy(logits.unsqueeze(0), labels_contrastive.unsqueeze(0))

            # Reverse direction
            pos_sim_rev = similarity_matrix[i, i]
            neg_sims_rev = similarity_matrix[:, i][negative_mask[:, i]]

            logits_rev = torch.cat([pos_sim_rev.unsqueeze(0), neg_sims_rev])
            loss_2d_to_1d += F.cross_entropy(logits_rev.unsqueeze(0), labels_contrastive.unsqueeze(0))

        return (loss_1d_to_2d + loss_2d_to_1d) / (2 * batch_size)

    def _feature_alignment_loss(self,
                              proj_1d: torch.Tensor,
                              proj_2d: torch.Tensor,
                              labels: torch.Tensor) -> torch.Tensor:
        """
        Compute feature space alignment loss using class-wise centroids.
        Features from the same class should have similar centroids across modalities.
        """
        unique_labels = torch.unique(labels)
        alignment_loss = 0

        for label in unique_labels:
            # Get features for this class
            class_mask = (labels == label)
            if torch.sum(class_mask) < 2:  # Skip if not enough samples
                continue

            class_feat_1d = proj_1d[class_mask]
            class_feat_2d = proj_2d[class_mask]

            # Compute class centroids
            centroid_1d = torch.mean(class_feat_1d, dim=0)
            centroid_2d = torch.mean(class_feat_2d, dim=0)

            # Align centroids
            alignment_loss += F.mse_loss(centroid_1d, centroid_2d)

        return alignment_loss / len(unique_labels)

    def _prediction_consistency_loss(self,
                                   feat_1d: torch.Tensor,
                                   feat_2d: torch.Tensor,
                                   labels: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction consistency loss.
        Simple classifiers on both modalities should produce consistent predictions.
        """
        # Simple classifiers (could be external for better performance)
        classifier_1d = nn.Linear(feat_1d.size(-1), len(torch.unique(labels))).to(feat_1d.device)
        classifier_2d = nn.Linear(feat_2d.size(-1), len(torch.unique(labels))).to(feat_2d.device)

        # Get predictions
        pred_1d = classifier_1d(feat_1d)
        pred_2d = classifier_2d(feat_2d)

        # Compute consistency loss (KL divergence between predictions)
        prob_1d = F.softmax(pred_1d, dim=-1)
        prob_2d = F.softmax(pred_2d, dim=-1)

        consistency_loss = F.kl_div(
            torch.log(prob_1d + 1e-8),
            prob_2d,
            reduction='batchmean'
        )

        return consistency_loss


class SemanticProjectionHead(nn.Module):
    """
    Projection head for semantic alignment with configurable architecture.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [256, 128],
                 output_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.projection = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class CrossModalMemoryBank:
    """
    Memory bank for storing cross-modal features for improved contrastive learning.
    """

    def __init__(self,
                 size: int = 10000,
                 feature_dim: int = 64,
                 temperature: float = 0.1):
        self.size = size
        self.feature_dim = feature_dim
        self.temperature = temperature

        # Memory banks for each modality
        self.memory_1d = torch.zeros(size, feature_dim)
        self.memory_2d = torch.zeros(size, feature_dim)
        self.labels = torch.zeros(size, dtype=torch.long)

        self.ptr = 0
        self.filled = False

    def update(self,
               feat_1d: torch.Tensor,
               feat_2d: torch.Tensor,
               labels: torch.Tensor):
        """Update memory bank with new features."""
        batch_size = feat_1d.size(0)

        # Determine indices to update
        if self.ptr + batch_size <= self.size:
            idx = torch.arange(self.ptr, self.ptr + batch_size)
            self.ptr += batch_size
        else:
            # Wrap around
            idx = torch.cat([
                torch.arange(self.ptr, self.size),
                torch.arange(0, (self.ptr + batch_size) % self.size)
            ])
            self.ptr = (self.ptr + batch_size) % self.size
            self.filled = True

        # Update memory
        self.memory_1d[idx] = feat_1d.detach().cpu()
        self.memory_2d[idx] = feat_2d.detach().cpu()
        self.labels[idx] = labels.cpu()

    def get_negative_samples(self,
                           query_feat: torch.Tensor,
                           query_labels: torch.Tensor,
                           modality: str,
                           num_negatives: int = 1024) -> torch.Tensor:
        """Get negative samples from memory bank."""
        if not self.filled and self.ptr < num_negatives:
            num_negatives = self.ptr

        # Get features from opposite modality
        if modality == '1d':
            memory_features = self.memory_2d[:self.ptr if not self.filled else self.size]
        else:
            memory_features = self.memory_1d[:self.ptr if not self.filled else self.size]

        memory_labels = self.labels[:self.ptr if not self.filled else self.size]

        # Filter out same-class samples
        valid_neg_mask = memory_labels.unsqueeze(0) != query_labels.unsqueeze(1)
        negative_candidates = memory_features[valid_neg_mask.any(dim=0)]

        # Randomly sample negatives
        if len(negative_candidates) > num_negatives:
            perm = torch.randperm(len(negative_candidates))[:num_negatives]
            negative_samples = negative_candidates[perm]
        else:
            negative_samples = negative_candidates

        return negative_samples.to(query_feat.device)


def compute_semantic_alignment_metrics(feat_1d: torch.Tensor,
                                     feat_2d: torch.Tensor,
                                     labels: torch.Tensor) -> Dict[str, float]:
    """
    Compute semantic alignment metrics for evaluation.

    Args:
        feat_1d: 1D features
        feat_2d: 2D features
        labels: Class labels

    Returns:
        Dictionary of semantic alignment metrics
    """
    metrics = {}

    # Flatten features
    feat_1d_flat = feat_1d.view(feat_1d.size(0), -1)
    feat_2d_flat = feat_2d.view(feat_2d.size(0), -1)

    # Cross-modal similarity
    feat_1d_norm = F.normalize(feat_1d_flat, dim=-1)
    feat_2d_norm = F.normalize(feat_2d_flat, dim=-1)

    cross_modal_sim = torch.sum(feat_1d_norm * feat_2d_norm, dim=-1)
    metrics['cross_modal_similarity'] = torch.mean(cross_modal_sim).item()

    # Intra-class vs inter-class similarity
    unique_labels = torch.unique(labels)
    intra_similarities = []
    inter_similarities = []

    for label in unique_labels:
        class_mask = (labels == label)
        if torch.sum(class_mask) < 2:
            continue

        class_feat_1d = feat_1d_norm[class_mask]
        class_feat_2d = feat_2d_norm[class_mask]

        # Intra-class similarity
        class_centroid_1d = torch.mean(class_feat_1d, dim=0)
        class_centroid_2d = torch.mean(class_feat_2d, dim=0)
        intra_sim = F.cosine_similarity(class_centroid_1d, class_centroid_2d, dim=0)
        intra_similarities.append(intra_sim.item())

        # Inter-class similarity (with other classes)
        for other_label in unique_labels:
            if other_label == label:
                continue

            other_mask = (labels == other_label)
            if torch.sum(other_mask) < 2:
                continue

            other_feat_1d = feat_1d_norm[other_mask]
            other_feat_2d = feat_2d_norm[other_mask]

            other_centroid_1d = torch.mean(other_feat_1d, dim=0)
            other_centroid_2d = torch.mean(other_feat_2d, dim=0)

            inter_sim_1d = F.cosine_similarity(class_centroid_1d, other_centroid_1d, dim=0)
            inter_sim_2d = F.cosine_similarity(class_centroid_2d, other_centroid_2d, dim=0)
            inter_similarities.extend([inter_sim_1d.item(), inter_sim_2d.item()])

    metrics['intra_class_similarity'] = np.mean(intra_similarities) if intra_similarities else 0.0
    metrics['inter_class_similarity'] = np.mean(inter_similarities) if inter_similarities else 0.0
    metrics['semantic_separation'] = (metrics['intra_class_similarity'] - metrics['inter_class_similarity'])

    return metrics


if __name__ == "__main__":
    # Test the semantic alignment loss
    batch_size, seq_len = 8, 1024
    channels_1d, channels_2d = 64, 32
    height, width = 32, 32
    num_classes = 4

    # Create dummy features and labels
    feat_1d = torch.randn(batch_size, channels_1d, seq_len)
    feat_2d = torch.randn(batch_size, channels_2d, height, width)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Initialize loss
    loss_fn = SemanticAlignmentLoss()

    # Compute loss
    losses = loss_fn(feat_1d, feat_2d, labels)

    print("Semantic Alignment Loss Components:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.6f}")

    # Compute metrics
    metrics = compute_semantic_alignment_metrics(feat_1d, feat_2d, labels)
    print("\nSemantic Alignment Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")