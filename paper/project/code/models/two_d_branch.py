"""
2D Spectrogram Feature Extraction Branch
Simple CNN-based implementation for minimal demo
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TwoDBranch(nn.Module):
    """
    Simple 2D CNN branch for spectrogram feature extraction
    """

    def __init__(self,
                 input_shape=(1, 128, 128),  # (channels, height, width)
                 base_channels=32,
                 num_layers=3,
                 dropout=0.2):
        super(TwoDBranch, self).__init__()

        self.input_shape = input_shape
        self.base_channels = base_channels
        self.num_layers = num_layers

        # Convolutional layers
        conv_layers = []
        current_channels = input_shape[0]

        for i in range(num_layers):
            # Calculate output channels for this layer
            if i == 0:
                layer_out_channels = base_channels
            else:
                layer_out_channels = base_channels * (2 ** min(i, 3))

            conv_layers.extend([
                nn.Conv2d(current_channels, layer_out_channels,
                         kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(layer_out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])

            current_channels = layer_out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate flattened feature size after global average pooling
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape)
            dummy_output = self.conv_layers(dummy_input)
            pooled_output = F.adaptive_avg_pool2d(dummy_output, 1).view(dummy_output.size(0), -1)
            self.flattened_size = pooled_output.size(1)

        # Final feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(self.flattened_size, 64),  # Match 1D branch output
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            features: Extracted features of shape (batch_size, 64)
        """
        # Apply convolutional layers
        conv_out = self.conv_layers(x)

        # Global average pooling
        pooled = F.adaptive_avg_pool2d(conv_out, 1).view(conv_out.size(0), -1)

        # Final projection
        features = self.feature_proj(pooled)

        return features


def stft_spectrogram(signal, n_fft=256, hop_length=128, win_length=256):
    """
    Convert 1D signal to 2D spectrogram using STFT

    Args:
        signal: 1D signal tensor of shape (batch_size, seq_len)
        n_fft: FFT window size
        hop_length: Hop length for STFT
        win_length: Window length for STFT

    Returns:
        spectrogram: Spectrogram of shape (batch_size, 1, freq_bins, time_frames)
    """
    device = signal.device
    seq_len = int(signal.shape[-1])
    effective_n_fft = min(int(n_fft), seq_len)
    effective_win_length = min(int(win_length), effective_n_fft)
    effective_hop_length = min(int(hop_length), max(1, effective_n_fft // 2))

    # Compute STFT with window on correct device
    stft = torch.stft(signal, n_fft=effective_n_fft, hop_length=effective_hop_length,
                      win_length=effective_win_length, window=torch.hann_window(effective_win_length).to(device),
                      return_complex=True)

    # Convert to magnitude (power spectrogram)
    magnitude = torch.abs(stft)

    # Add channel dimension and normalize
    spectrogram = magnitude.unsqueeze(1)  # (batch_size, 1, freq_bins, time_frames)

    # Log-scale for better dynamic range
    spectrogram = torch.log1p(spectrogram)

    return spectrogram


def create_spectrogram_from_1d(signal_1d, target_size=(128, 128)):
    """
    Create 2D spectrogram from 1D signal and resize to target size

    Args:
        signal_1d: 1D signal tensor of shape (batch_size, seq_len)
        target_size: Target size (height, width) for spectrogram

    Returns:
        spectrogram: Resized spectrogram of shape (batch_size, 1, height, width)
    """
    # Create spectrogram
    spec = stft_spectrogram(signal_1d)

    # Resize to target size
    spec_resized = F.interpolate(spec, size=target_size, mode='bilinear', align_corners=False)

    return spec_resized


def test_two_d_branch():
    """Test function for TwoDBranch"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = TwoDBranch(
        input_shape=(1, 128, 128),
        base_channels=32,
        num_layers=3
    ).to(device)

    # Test with dummy spectrogram data
    batch_size = 8
    x = torch.randn(batch_size, 1, 128, 128).to(device)

    # Forward pass
    features = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Test spectrogram creation from 1D signal
    signal_1d = torch.randn(batch_size, 4096).to(device)
    spectrogram = create_spectrogram_from_1d(signal_1d, target_size=(128, 128))
    spec_features = model(spectrogram)

    print(f"1D signal shape: {signal_1d.shape}")
    print(f"Created spectrogram shape: {spectrogram.shape}")
    print(f"Spectrogram features shape: {spec_features.shape}")

    return (features.shape == (batch_size, 64) and
            spec_features.shape == (batch_size, 64))


if __name__ == "__main__":
    success = test_two_d_branch()
    print(f"2D Branch test: {'PASSED' if success else 'FAILED'}")
