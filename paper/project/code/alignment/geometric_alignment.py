"""
Geometric Layer Alignment for 1D-2D Fusion Fault Diagnosis

This module implements geometric alignment mechanisms that preserve spatial and
structural relationships between 1D time-domain and 2D frequency-domain representations,
ensuring consistent neighborhood structures and topological properties.

Key concepts:
- Local neighborhood preservation
- Manifold alignment
- Topological consistency
- Structural similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform


class GeometricAlignmentLoss(nn.Module):
    """
    Geometric alignment loss that preserves geometric structure between 1D and 2D
    representations using manifold alignment and neighborhood preservation.
    """

    def __init__(self,
                 neighborhood_weight: float = 1.0,
                 manifold_weight: float = 0.5,
                 topology_weight: float = 0.3,
                 n_neighbors: int = 5,
                 temperature: float = 1.0):
        """
        Initialize geometric alignment loss.

        Args:
            neighborhood_weight: Weight for neighborhood preservation loss
            manifold_weight: Weight for manifold alignment loss
            topology_weight: Weight for topological consistency loss
            n_neighbors: Number of neighbors for local structure
            temperature: Temperature for similarity computation
        """
        super().__init__()
        self.neighborhood_weight = neighborhood_weight
        self.manifold_weight = manifold_weight
        self.topology_weight = topology_weight
        self.n_neighbors = n_neighbors
        self.temperature = temperature

    def forward(self,
                feat_1d: torch.Tensor,
                feat_2d: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute geometric alignment loss.

        Args:
            feat_1d: 1D features [B, C1, L1] or [B, C1]
            feat_2d: 2D features [B, C2, H2, W2] or [B, C2]

        Returns:
            Dictionary containing individual loss components and total loss
        """
        losses = {}

        # Flatten features
        feat_1d_flat = self._flatten_features(feat_1d)
        feat_2d_flat = self._flatten_features(feat_2d)

        # Align dimensions
        min_dim = min(feat_1d_flat.size(-1), feat_2d_flat.size(-1))
        feat_1d_aligned = feat_1d_flat[..., :min_dim]
        feat_2d_aligned = feat_2d_flat[..., :min_dim]

        # Neighborhood preservation loss
        neighborhood_loss = self._neighborhood_preservation_loss(
            feat_1d_aligned, feat_2d_aligned
        )
        losses['neighborhood'] = self.neighborhood_weight * neighborhood_loss

        # Manifold alignment loss
        manifold_loss = self._manifold_alignment_loss(feat_1d_aligned, feat_2d_aligned)
        losses['manifold'] = self.manifold_weight * manifold_loss

        # Topological consistency loss
        topology_loss = self._topological_consistency_loss(
            feat_1d_aligned, feat_2d_aligned
        )
        losses['topology'] = self.topology_weight * topology_loss

        # Total geometric alignment loss
        losses['total'] = losses['neighborhood'] + losses['manifold'] + losses['topology']

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

    def _neighborhood_preservation_loss(self,
                                      feat_1d: torch.Tensor,
                                      feat_2d: torch.Tensor) -> torch.Tensor:
        """
        Compute neighborhood preservation loss.
        Neighbors in one modality should remain neighbors in the other modality.
        """
        batch_size = feat_1d.size(0)

        # Compute pairwise distances
        dist_1d = self._pairwise_distances(feat_1d)  # [B, B]
        dist_2d = self._pairwise_distances(feat_2d)  # [B, B]

        # Get k-nearest neighbors for each sample (ensure we don't exceed batch size)
        k = min(self.n_neighbors + 1, batch_size - 1)
        _, neighbors_1d = torch.topk(dist_1d, k, largest=False)
        _, neighbors_2d = torch.topk(dist_2d, k, largest=False)

        # Exclude self (first neighbor)
        neighbors_1d = neighbors_1d[:, 1:]  # [B, k]
        neighbors_2d = neighbors_2d[:, 1:]  # [B, k]

        # Compute neighborhood overlap
        neighborhood_overlap = 0
        valid_samples = 0
        for i in range(batch_size):
            set_1d = set(neighbors_1d[i].tolist())
            set_2d = set(neighbors_2d[i].tolist())
            union_size = len(set_1d.union(set_2d))
            if union_size > 0:
                overlap = len(set_1d.intersection(set_2d)) / union_size
                neighborhood_overlap += overlap
                valid_samples += 1

        # Convert overlap to loss (maximize overlap -> minimize loss)
        if valid_samples > 0:
            neighborhood_loss = 1.0 - (neighborhood_overlap / valid_samples)
        else:
            neighborhood_loss = torch.tensor(1.0, device=feat_1d.device)

        return neighborhood_loss

    def _manifold_alignment_loss(self,
                               feat_1d: torch.Tensor,
                               feat_2d: torch.Tensor) -> torch.Tensor:
        """
        Compute manifold alignment loss using local linear embedding (LLE).
        Preserves local geometric structure on the manifold.
        """
        batch_size = feat_1d.size(0)
        device = feat_1d.device

        # Convert to numpy for sklearn
        feat_1d_np = feat_1d.detach().cpu().numpy()
        feat_2d_np = feat_2d.detach().cpu().numpy()

        # Fit local linear embeddings
        try:
            # Compute reconstruction weights
            weights_1d = self._compute_lle_weights(feat_1d_np, self.n_neighbors)
            weights_2d = self._compute_lle_weights(feat_2d_np, self.n_neighbors)

            # Convert to tensors
            weights_1d = torch.tensor(weights_1d, dtype=torch.float32, device=device)
            weights_2d = torch.tensor(weights_2d, dtype=torch.float32, device=device)

            # Compute reconstruction error
            reconstruction_1d_to_2d = self._compute_reconstruction_error(
                feat_2d, weights_1d
            )
            reconstruction_2d_to_1d = self._compute_reconstruction_error(
                feat_1d, weights_2d
            )

            manifold_loss = reconstruction_1d_to_2d + reconstruction_2d_to_1d

        except Exception:
            # Fallback to simple distance preservation
            manifold_loss = self._distance_preservation_loss(feat_1d, feat_2d)

        return manifold_loss

    def _compute_lle_weights(self, X: np.ndarray, n_neighbors: int) -> np.ndarray:
        """Compute Local Linear Embedding weights."""
        n_samples = X.shape[0]
        weights = np.zeros((n_samples, n_samples))

        # Fit nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
        distances, indices = nbrs.kneighbors(X)

        for i in range(n_samples):
            # Get neighbors (excluding self)
            neighbor_indices = indices[i, 1:]
            neighbors = X[neighbor_indices]

            # Center the data
            centered = neighbors - X[i]

            # Compute local covariance
            C = np.dot(centered, centered.T)

            # Solve for weights with regularization
            try:
                C_inv = np.linalg.inv(C + 1e-3 * np.eye(n_neighbors))
                w = np.sum(C_inv, axis=1) / np.sum(C_inv)
                weights[i, neighbor_indices] = w
            except:
                # Fallback: uniform weights
                weights[i, neighbor_indices] = 1.0 / n_neighbors

        return weights

    def _compute_reconstruction_error(self,
                                    features: torch.Tensor,
                                    weights: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction error using LLE weights."""
        batch_size = features.size(0)
        reconstruction_error = 0

        for i in range(batch_size):
            # Reconstruct sample i from its neighbors
            neighbor_weights = weights[i]
            neighbor_mask = neighbor_weights > 0

            if torch.any(neighbor_mask):
                reconstructed = torch.sum(
                    features[neighbor_mask] * neighbor_weights[neighbor_mask].unsqueeze(-1), dim=0
                )
                reconstruction_error += F.mse_loss(features[i], reconstructed)

        return reconstruction_error / batch_size

    def _distance_preservation_loss(self,
                                  feat_1d: torch.Tensor,
                                  feat_2d: torch.Tensor) -> torch.Tensor:
        """
        Simple distance preservation loss as fallback.
        Relative distances should be preserved between modalities.
        """
        # Compute pairwise distances
        dist_1d = self._pairwise_distances(feat_1d)
        dist_2d = self._pairwise_distances(feat_2d)

        # Normalize distances
        dist_1d_norm = dist_1d / (torch.max(dist_1d) + 1e-8)
        dist_2d_norm = dist_2d / (torch.max(dist_2d) + 1e-8)

        # Compute distance preservation loss
        distance_loss = F.mse_loss(dist_1d_norm, dist_2d_norm)

        return distance_loss

    def _topological_consistency_loss(self,
                                    feat_1d: torch.Tensor,
                                    feat_2d: torch.Tensor) -> torch.Tensor:
        """
        Compute topological consistency loss using persistent homology concepts.
        Preserves connectivity and holes in the data structure.
        """
        # Build connectivity graphs using different thresholds
        thresholds = [0.1, 0.3, 0.5, 0.7]
        topology_loss = 0

        for threshold in thresholds:
            # Compute adjacency matrices
            adj_1d = self._build_adjacency_matrix(feat_1d, threshold)
            adj_2d = self._build_adjacency_matrix(feat_2d, threshold)

            # Compare graph properties
            degree_1d = torch.sum(adj_1d, dim=-1)
            degree_2d = torch.sum(adj_2d, dim=-1)

            topology_loss += F.mse_loss(degree_1d, degree_2d)

        return topology_loss / len(thresholds)

    def _pairwise_distances(self, features: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances."""
        # Expand dimensions for broadcasting
        feat_expanded = features.unsqueeze(1)  # [B, 1, D]
        feat_tiled = features.unsqueeze(0)    # [1, B, D]

        # Compute squared distances
        distances = torch.sum((feat_expanded - feat_tiled) ** 2, dim=-1)  # [B, B]
        return torch.sqrt(distances + 1e-8)

    def _build_adjacency_matrix(self,
                               features: torch.Tensor,
                               threshold: float) -> torch.Tensor:
        """Build adjacency matrix using distance threshold."""
        distances = self._pairwise_distances(features)

        # Normalize distances to [0, 1]
        dist_norm = distances / (torch.max(distances) + 1e-8)

        # Build adjacency matrix (connect if distance < threshold)
        adjacency = (dist_norm < threshold).float()

        # Remove self-connections
        adjacency = adjacency - torch.diag(torch.diag(adjacency))

        return adjacency


class GeometricProjection(nn.Module):
    """
    Geometric projection that aligns features while preserving geometric structure.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 preserve_structure: bool = True):
        super().__init__()
        self.preserve_structure = preserve_structure

        # Standard projection layers
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, output_dim)
        )

        if preserve_structure:
            # Orthogonal constraint layers
            self.orthogonal_regularization = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply geometric projection."""
        projected = self.projection(x)

        if self.preserve_structure and self.training:
            # Apply orthogonal regularization to preserve structure
            weight = self.projection[-1].weight
            orthogonality_loss = torch.mean(
                torch.abs(torch.mm(weight, weight.T) - torch.eye(weight.size(0), device=weight.device))
            )

            # Add to regularization (this would be handled by the training loop)
            if hasattr(self, '_orthogonality_loss'):
                self._orthogonality_loss = orthogonality_loss

        return projected


def compute_geometric_alignment_metrics(feat_1d: torch.Tensor,
                                      feat_2d: torch.Tensor) -> Dict[str, float]:
    """
    Compute geometric alignment metrics for evaluation.

    Args:
        feat_1d: 1D features
        feat_2d: 2D features

    Returns:
        Dictionary of geometric alignment metrics
    """
    metrics = {}

    # Flatten features
    feat_1d_flat = feat_1d.view(feat_1d.size(0), -1)
    feat_2d_flat = feat_2d.view(feat_2d.size(0), -1)

    # Align dimensions
    min_dim = min(feat_1d_flat.size(-1), feat_2d_flat.size(-1))
    feat_1d_aligned = feat_1d_flat[..., :min_dim]
    feat_2d_aligned = feat_2d_flat[..., :min_dim]

    # Convert to numpy for sklearn computations
    feat_1d_np = feat_1d_aligned.detach().cpu().numpy()
    feat_2d_np = feat_2d_aligned.detach().cpu().numpy()

    # Neighborhood preservation
    n_neighbors = 5
    try:
        nbrs_1d = NearestNeighbors(n_neighbors=n_neighbors).fit(feat_1d_np)
        nbrs_2d = NearestNeighbors(n_neighbors=n_neighbors).fit(feat_2d_np)

        _, indices_1d = nbrs_1d.kneighbors(feat_1d_np)
        _, indices_2d = nbrs_2d.kneighbors(feat_2d_np)

        # Excluding self
        indices_1d = indices_1d[:, 1:]
        indices_2d = indices_2d[:, 1:]

        overlap_scores = []
        for i in range(len(indices_1d)):
            set_1d = set(indices_1d[i])
            set_2d = set(indices_2d[i])
            overlap = len(set_1d.intersection(set_2d)) / len(set_1d.union(set_2d))
            overlap_scores.append(overlap)

        metrics['neighborhood_preservation'] = np.mean(overlap_scores)
    except:
        metrics['neighborhood_preservation'] = 0.0

    # Distance correlation
    try:
        # Compute distance matrices
        dist_1d = squareform(pdist(feat_1d_np))
        dist_2d = squareform(pdist(feat_2d_np))

        # Compute distance correlation (simplified)
        dist_1d_flat = dist_1d.flatten()
        dist_2d_flat = dist_2d.flatten()

        # Remove diagonal elements
        mask = ~np.eye(dist_1d.shape[0], dtype=bool)
        dist_1d_flat = dist_1d_flat[mask.flatten()]
        dist_2d_flat = dist_2d_flat[mask.flatten()]

        correlation = np.corrcoef(dist_1d_flat, dist_2d_flat)[0, 1]
        metrics['distance_correlation'] = correlation if not np.isnan(correlation) else 0.0
    except:
        metrics['distance_correlation'] = 0.0

    # Procrustes analysis (shape alignment)
    try:
        # Align shapes using Procrustes analysis
        diff = feat_1d_np - feat_2d_np
        procrustes_error = np.mean(np.sum(diff ** 2, axis=1))
        metrics['procrustes_error'] = procrustes_error
    except:
        metrics['procrustes_error'] = 0.0

    return metrics


if __name__ == "__main__":
    # Test the geometric alignment loss
    batch_size, seq_len = 6, 1024
    channels_1d, channels_2d = 64, 32
    height, width = 32, 32

    # Create dummy features
    feat_1d = torch.randn(batch_size, channels_1d, seq_len)
    feat_2d = torch.randn(batch_size, channels_2d, height, width)

    # Initialize loss
    loss_fn = GeometricAlignmentLoss()

    # Compute loss
    losses = loss_fn(feat_1d, feat_2d)

    print("Geometric Alignment Loss Components:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.6f}")

    # Compute metrics
    metrics = compute_geometric_alignment_metrics(feat_1d, feat_2d)
    print("\nGeometric Alignment Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")