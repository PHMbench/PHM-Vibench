"""
Early Fusion Model for 1D-2D Feature Combination
Simple concatenation + MLP implementation for minimal demo
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .one_d_branch import OneDBranch
from .two_d_branch import TwoDBranch, create_spectrogram_from_1d


class EarlyFusionModel(nn.Module):
    """
    Early Fusion Model that combines 1D and 2D features via concatenation
    """

    def __init__(self,
                 input_dim_1d=4096,
                 spectrogram_size=(128, 128),
                 num_classes=10,
                 hidden_dim=128,
                 dropout=0.2):
        super(EarlyFusionModel, self).__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

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

        # Fusion layers
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

    def forward(self, signal_1d):
        """
        Forward pass

        Args:
            signal_1d: Input 1D signal tensor of shape (batch_size, seq_len)

        Returns:
            logits: Classification logits of shape (batch_size, num_classes)
            features_1d: 1D branch features
            features_2d: 2D branch features
        """
        # Extract 1D features
        features_1d = self.one_d_branch(signal_1d)

        # Create 2D spectrogram from 1D signal
        spectrogram = create_spectrogram_from_1d(signal_1d, target_size=(128, 128))

        # Extract 2D features
        features_2d = self.two_d_branch(spectrogram)

        # Early fusion via concatenation
        fused_features = torch.cat([features_1d, features_2d], dim=1)

        # Classification
        logits = self.fusion_layers(fused_features)

        return logits, features_1d, features_2d

    def get_embeddings(self, signal_1d):
        """
        Get fused embeddings without classification

        Args:
            signal_1d: Input 1D signal tensor

        Returns:
            fused_features: Concatenated features
        """
        # Extract features from both branches
        features_1d = self.one_d_branch(signal_1d)
        spectrogram = create_spectrogram_from_1d(signal_1d, target_size=(128, 128))
        features_2d = self.two_d_branch(spectrogram)

        # Concatenate features
        fused_features = torch.cat([features_1d, features_2d], dim=1)

        return fused_features, features_1d, features_2d


def test_early_fusion_model():
    """Test function for EarlyFusionModel"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = EarlyFusionModel(
        input_dim_1d=4096,
        spectrogram_size=(128, 128),
        num_classes=10,
        hidden_dim=128
    ).to(device)

    # Test with dummy data
    batch_size = 8
    seq_len = 4096
    x = torch.randn(batch_size, seq_len).to(device)

    # Forward pass
    logits, feat_1d, feat_2d = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"1D features shape: {feat_1d.shape}")
    print(f"2D features shape: {feat_2d.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Test embeddings
    fused_feat, f1d, f2d = model.get_embeddings(x)
    print(f"Fused embeddings shape: {fused_feat.shape}")

    # Test with different sequence length
    x2 = torch.randn(batch_size, 2048).to(device)  # Different length
    try:
        logits2, _, _ = model(x2)
        print(f"Different length test: Logits shape {logits2.shape}")
        flexible_input = True
    except Exception as e:
        print(f"Different length test failed: {e}")
        flexible_input = False

    return (logits.shape == (batch_size, 10) and
            feat_1d.shape == (batch_size, 64) and
            feat_2d.shape == (batch_size, 64) and
            fused_feat.shape == (batch_size, 128))


if __name__ == "__main__":
    success = test_early_fusion_model()
    print(f"Early Fusion Model test: {'PASSED' if success else 'FAILED'}")