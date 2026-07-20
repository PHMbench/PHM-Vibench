"""
1D Time Series Feature Extraction Branch
Simple CNN-based implementation for minimal demo
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OneDBranch(nn.Module):
    """
    Simple 1D CNN branch for time series feature extraction
    """

    def __init__(self,
                 input_dim=4096,
                 in_channels=1,
                 out_channels=64,
                 num_layers=3,
                 kernel_size=7,
                 dropout=0.2):
        super(OneDBranch, self).__init__()

        self.input_dim = input_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        # Convolutional layers
        conv_layers = []
        current_channels = in_channels

        for i in range(num_layers):
            # Calculate output channels for this layer
            if i == num_layers - 1:
                # Last layer outputs out_channels
                layer_out_channels = out_channels
            else:
                # Intermediate layers double channels
                layer_out_channels = min(current_channels * 2, out_channels)

            conv_layers.extend([
                nn.Conv1d(current_channels, layer_out_channels, kernel_size,
                         padding=kernel_size//2, stride=2),
                nn.BatchNorm1d(layer_out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])

            current_channels = layer_out_channels
            kernel_size = max(kernel_size // 2, 3)  # Reduce kernel size in deeper layers

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate flattened feature size after global average pooling
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, input_dim)
            dummy_output = self.conv_layers(dummy_input)
            pooled_output = F.adaptive_avg_pool1d(dummy_output, 1).squeeze(-1)
            self.flattened_size = pooled_output.size(1)

        # Final feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(self.flattened_size, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, seq_len) or (batch_size, 1, seq_len)

        Returns:
            features: Extracted features of shape (batch_size, out_channels)
        """
        # Ensure input has channel dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, seq_len)

        # Apply convolutional layers
        conv_out = self.conv_layers(x)

        # Global average pooling
        pooled = F.adaptive_avg_pool1d(conv_out, 1).squeeze(-1)

        # Final projection
        features = self.feature_proj(pooled)

        return features


def test_one_d_branch():
    """Test function for OneDBranch"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = OneDBranch(
        input_dim=4096,
        in_channels=1,
        out_channels=64,
        num_layers=3
    ).to(device)

    # Test with dummy data
    batch_size = 8
    seq_len = 4096
    x = torch.randn(batch_size, seq_len).to(device)

    # Forward pass
    features = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    return features.shape == (batch_size, 64)


if __name__ == "__main__":
    success = test_one_d_branch()
    print(f"1D Branch test: {'PASSED' if success else 'FAILED'}")