"""
Aligned Fusion Model with Three-Layer Alignment for 1D-2D Fault Diagnosis

This module implements a fusion model that incorporates three-layer alignment mechanisms:
- Physical layer: Energy distribution and spectral consistency
- Semantic layer: Cross-modal contrastive learning
- Geometric layer: Neighborhood preservation and manifold alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

from .one_d_branch import OneDBranch
from .two_d_branch import TwoDBranch, create_spectrogram_from_1d
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alignment import (
    PhysicalAlignmentLoss,
    SemanticAlignmentLoss,
    GeometricAlignmentLoss,
    compute_physical_alignment_metrics,
    compute_semantic_alignment_metrics,
    compute_geometric_alignment_metrics
)


class AlignedFusionModel(nn.Module):
    """
    Fusion model with three-layer alignment mechanisms for 1D-2D fault diagnosis.
    """

    def __init__(self,
                 input_dim_1d: int = 4096,
                 spectrogram_size: Tuple[int, int] = (128, 128),
                 num_classes: int = 10,
                 hidden_dim: int = 128,
                 dropout: float = 0.2,
                 alignment_config: Optional[Dict[str, Any]] = None):
        """
        Initialize aligned fusion model.

        Args:
            input_dim_1d: Input dimension for 1D signals
            spectrogram_size: Size of generated spectrograms
            num_classes: Number of fault classes
            hidden_dim: Hidden dimension for fusion layers
            dropout: Dropout rate
            alignment_config: Configuration for alignment losses
        """
        super(AlignedFusionModel, self).__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.input_dim_1d = input_dim_1d
        self.spectrogram_size = spectrogram_size

        # 1D branch for time series
        self.one_d_branch = OneDBranch(
            input_dim=input_dim_1d,
            in_channels=1,
            out_channels=64,
            num_layers=3,
            dropout=dropout
        )

        # 2D branch for spectrogram
        self.two_d_branch = TwoDBranch(
            input_shape=(1, *spectrogram_size),
            base_channels=32,
            num_layers=3,
            dropout=dropout
        )

        # Alignment losses
        self.physical_aligner = PhysicalAlignmentLoss(
            **(alignment_config.get('physical', {}) if alignment_config else {})
        )
        self.semantic_aligner = SemanticAlignmentLoss(
            **(alignment_config.get('semantic', {}) if alignment_config else {})
        )
        self.geometric_aligner = GeometricAlignmentLoss(
            **(alignment_config.get('geometric', {}) if alignment_config else {})
        )

        # Fusion layers with alignment awareness
        # Input: 64 (1D features) + 64 (2D features) = 128
        self.fusion_layers = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Alignment weights (learnable)
        self.alignment_weights = nn.ParameterDict({
            'physical': nn.Parameter(torch.tensor(1.0)),
            'semantic': nn.Parameter(torch.tensor(1.0)),
            'geometric': nn.Parameter(torch.tensor(1.0))
        })

        # Track alignment metrics
        self.alignment_metrics = {}

    def forward(self,
                signal_1d: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                return_alignment: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with alignment computation.

        Args:
            signal_1d: Input 1D signal tensor [B, L]
            labels: Ground truth labels [B] (required for semantic alignment)
            return_alignment: Whether to return alignment losses and metrics

        Returns:
            Dictionary containing:
            - logits: Classification logits [B, num_classes]
            - features_1d: 1D branch features [B, 64]
            - features_2d: 2D branch features [B, 64]
            - alignment_losses: Alignment losses (if return_alignment=True)
            - alignment_metrics: Alignment metrics (if return_alignment=True)
        """
        outputs = {}

        # Extract 1D features
        features_1d = self.one_d_branch(signal_1d)

        # Create 2D spectrogram from 1D signal
        spectrogram = create_spectrogram_from_1d(signal_1d, target_size=self.spectrogram_size)

        # Extract 2D features
        features_2d = self.two_d_branch(spectrogram)

        # Store features
        outputs['features_1d'] = features_1d
        outputs['features_2d'] = features_2d

        # Early fusion via concatenation
        fused_features = torch.cat([features_1d, features_2d], dim=1)

        # Classification
        logits = self.fusion_layers(fused_features)
        outputs['logits'] = logits

        # Compute alignment losses if requested
        if return_alignment:
            alignment_losses = {}
            alignment_metrics = {}

            # Physical alignment
            physical_losses = self.physical_aligner(
                features_1d, features_2d, signal_1d, spectrogram
            )
            alignment_losses['physical'] = self.alignment_weights['physical'] * physical_losses['total']

            # Semantic alignment (requires labels)
            if labels is not None:
                semantic_losses = self.semantic_aligner(features_1d, features_2d, labels)
                alignment_losses['semantic'] = self.alignment_weights['semantic'] * semantic_losses['total']
            else:
                alignment_losses['semantic'] = torch.tensor(0.0, device=signal_1d.device)

            # Geometric alignment
            geometric_losses = self.geometric_aligner(features_1d, features_2d)
            alignment_losses['geometric'] = self.alignment_weights['geometric'] * geometric_losses['total']

            # Total alignment loss
            alignment_losses['total'] = (alignment_losses['physical'] +
                                        alignment_losses['semantic'] +
                                        alignment_losses['geometric'])

            outputs['alignment_losses'] = alignment_losses

            # Compute alignment metrics
            if labels is not None:
                alignment_metrics.update(compute_semantic_alignment_metrics(
                    features_1d, features_2d, labels
                ))

            alignment_metrics.update(compute_physical_alignment_metrics(features_1d, features_2d))
            alignment_metrics.update(compute_geometric_alignment_metrics(features_1d, features_2d))

            outputs['alignment_metrics'] = alignment_metrics

        return outputs

    def compute_alignment_loss(self,
                             signal_1d: torch.Tensor,
                             labels: torch.Tensor,
                             features_1d: torch.Tensor,
                             features_2d: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute alignment losses separately.

        Args:
            signal_1d: Original 1D signals
            labels: Ground truth labels
            features_1d: 1D branch features
            features_2d: 2D branch features

        Returns:
            Dictionary of alignment losses
        """
        # Create spectrogram for physical alignment
        spectrogram = create_spectrogram_from_1d(signal_1d, target_size=self.spectrogram_size)

        # Physical alignment
        physical_losses = self.physical_aligner(features_1d, features_2d, signal_1d, spectrogram)
        physical_loss = self.alignment_weights['physical'] * physical_losses['total']

        # Semantic alignment
        semantic_losses = self.semantic_aligner(features_1d, features_2d, labels)
        semantic_loss = self.alignment_weights['semantic'] * semantic_losses['total']

        # Geometric alignment
        geometric_losses = self.geometric_aligner(features_1d, features_2d)
        geometric_loss = self.alignment_weights['geometric'] * geometric_losses['total']

        return {
            'physical': physical_loss,
            'semantic': semantic_loss,
            'geometric': geometric_loss,
            'total': physical_loss + semantic_loss + geometric_loss
        }

    def get_embeddings(self, signal_1d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get embeddings without classification.

        Args:
            signal_1d: Input 1D signal tensor

        Returns:
            Tuple of (fused_features, features_1d, features_2d)
        """
        # Extract features from both branches
        features_1d = self.one_d_branch(signal_1d)
        spectrogram = create_spectrogram_from_1d(signal_1d, target_size=self.spectrogram_size)
        features_2d = self.two_d_branch(spectrogram)

        # Concatenate features
        fused_features = torch.cat([features_1d, features_2d], dim=1)

        return fused_features, features_1d, features_2d

    def get_alignment_weights(self) -> Dict[str, float]:
        """Get current alignment weights."""
        return {
            'physical': self.alignment_weights['physical'].item(),
            'semantic': self.alignment_weights['semantic'].item(),
            'geometric': self.alignment_weights['geometric'].item()
        }

    def set_alignment_weights(self, weights: Dict[str, float]):
        """Set alignment weights."""
        for key, value in weights.items():
            if key in self.alignment_weights:
                self.alignment_weights[key].data.fill_(value)


class ProgressiveFusionModel(AlignedFusionModel):
    """
    Progressive fusion model that gradually increases alignment influence during training.
    """

    def __init__(self, *args, warmup_epochs: int = 10, alignment_schedule: str = 'linear', **kwargs):
        """
        Initialize progressive fusion model.

        Args:
            warmup_epochs: Number of epochs before introducing alignment
            alignment_schedule: Schedule for alignment weight progression ('linear', 'cosine', 'exponential')
        """
        super().__init__(*args, **kwargs)
        self.warmup_epochs = warmup_epochs
        self.alignment_schedule = alignment_schedule
        self.current_epoch = 0

    def update_epoch(self, epoch: int):
        """Update current epoch and adjust alignment weights."""
        self.current_epoch = epoch

        if epoch < self.warmup_epochs:
            # No alignment during warmup
            scale = 0.0
        else:
            progress = (epoch - self.warmup_epochs) / max(1, 50 - self.warmup_epochs)  # Assume 50 total epochs

            if self.alignment_schedule == 'linear':
                scale = min(1.0, progress)
            elif self.alignment_schedule == 'cosine':
                scale = 0.5 * (1 + torch.cos(torch.pi * (1 - progress)))
            elif self.alignment_schedule == 'exponential':
                scale = min(1.0, 2 ** progress - 1)
            else:
                scale = progress

        # Scale alignment weights
        for key in self.alignment_weights.keys():
            self.alignment_weights[key].data = torch.tensor(
                self.alignment_weights[key].item() * scale
            )

    def forward(self, *args, **kwargs):
        """Forward pass with automatic epoch update."""
        # Call parent forward
        return super().forward(*args, **kwargs)


def test_aligned_fusion_model():
    """Test function for AlignedFusionModel"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = AlignedFusionModel(
        input_dim_1d=4096,
        spectrogram_size=(128, 128),
        num_classes=10,
        hidden_dim=128
    ).to(device)

    # Test with dummy data
    batch_size = 8
    seq_len = 4096
    x = torch.randn(batch_size, seq_len).to(device)
    labels = torch.randint(0, 10, (batch_size,)).to(device)

    # Forward pass without alignment
    outputs = model(x)
    print(f"Without alignment:")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  1D features shape: {outputs['features_1d'].shape}")
    print(f"  2D features shape: {outputs['features_2d'].shape}")

    # Forward pass with alignment
    outputs_aligned = model(x, labels, return_alignment=True)
    print(f"\nWith alignment:")
    print(f"  Logits shape: {outputs_aligned['logits'].shape}")
    print(f"  Alignment losses: {list(outputs_aligned['alignment_losses'].keys())}")
    print(f"  Total alignment loss: {outputs_aligned['alignment_losses']['total'].item():.6f}")
    print(f"  Alignment metrics: {list(outputs_aligned['alignment_metrics'].keys())}")

    # Test embeddings
    fused_feat, f1d, f2d = model.get_embeddings(x)
    print(f"\nEmbeddings:")
    print(f"  Fused shape: {fused_feat.shape}")
    print(f"  1D shape: {f1d.shape}")
    print(f"  2D shape: {f2d.shape}")

    # Test alignment weights
    weights = model.get_alignment_weights()
    print(f"\nAlignment weights: {weights}")

    return True


if __name__ == "__main__":
    success = test_aligned_fusion_model()
    print(f"Aligned Fusion Model test: {'PASSED' if success else 'FAILED'}")