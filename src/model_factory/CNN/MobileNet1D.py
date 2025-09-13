"""MobileNet1D with depthwise separable convolutions for time-series analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DepthwiseSeparableConv1D(nn.Module):
    """Depthwise separable convolution for 1D signals.
    
    Parameters
    ----------
    in_channels : int
        Input channels
    out_channels : int
        Output channels
    kernel_size : int
        Convolution kernel size
    stride : int
        Convolution stride
    padding : int
        Convolution padding
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super(DepthwiseSeparableConv1D, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm1d(in_channels)
        
        # Pointwise convolution
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Depthwise convolution
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Pointwise convolution
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x


class InvertedResidual(nn.Module):
    """Inverted residual block (MobileNetV2 style).
    
    Parameters
    ----------
    in_channels : int
        Input channels
    out_channels : int
        Output channels
    stride : int
        Convolution stride
    expand_ratio : int
        Expansion ratio for bottleneck
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expand_ratio: int = 6):
        super(InvertedResidual, self).__init__()
        
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        
        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                     padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        
        # Project
        layers.extend([
            nn.Conv1d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Model(nn.Module):
    """MobileNet1D with depthwise separable convolutions for time-series analysis.
    
    Efficient CNN architecture using depthwise separable convolutions
    to reduce computational cost while maintaining performance.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - width_multiplier : float, width multiplier for channels (default: 1.0)
        - inverted_residual_setting : list, configuration for inverted residual blocks
        - dropout : float, dropout probability (default: 0.2)
        - num_classes : int, number of output classes (for classification)
        - output_dim : int, output dimension (for regression, default: input_dim)
    metadata : Any, optional
        Dataset metadata
        
    Input Shape
    -----------
    x : torch.Tensor
        Input tensor of shape (batch_size, seq_len, input_dim)
        
    Output Shape
    ------------
    torch.Tensor
        For classification: (batch_size, num_classes)
        For regression: (batch_size, seq_len, output_dim)
        
    References
    ----------
    Howard et al. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" arXiv 2017.
    Sandler et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks" CVPR 2018.
    Chollet "Xception: Deep Learning with Depthwise Separable Convolutions" CVPR 2017.
    Adapted for time-series industrial signals with depthwise separable convolutions for efficient temporal modeling.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.width_multiplier = getattr(args, 'width_multiplier', 1.0)
        self.dropout = getattr(args, 'dropout', 0.2)
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        
        # Default inverted residual setting
        # [expand_ratio, channels, num_blocks, stride]
        self.inverted_residual_setting = getattr(args, 'inverted_residual_setting', [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ])
        
        # Initial layer
        input_channel = int(32 * self.width_multiplier)
        self.features = [
            nn.Conv1d(self.input_dim, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(input_channel),
            nn.ReLU6(inplace=True)
        ]
        
        # Inverted residual blocks
        for expand_ratio, channels, num_blocks, stride in self.inverted_residual_setting:
            output_channel = int(channels * self.width_multiplier)
            
            for i in range(num_blocks):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, expand_ratio))
                input_channel = output_channel
        
        # Final layer
        last_channel = int(1280 * self.width_multiplier) if self.width_multiplier > 1.0 else 1280
        self.features.extend([
            nn.Conv1d(input_channel, last_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(last_channel),
            nn.ReLU6(inplace=True)
        ])
        
        self.features = nn.Sequential(*self.features)
        
        # Output layers
        if self.num_classes is not None:
            # Classification: global pooling + classifier
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(last_channel, self.num_classes)
            )
            self.task_type = 'classification'
        else:
            # Regression: upsampling + projection
            self.upsample = nn.Sequential(
                nn.ConvTranspose1d(last_channel, 320, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(320),
                nn.ReLU6(inplace=True),
                nn.ConvTranspose1d(320, 96, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(96),
                nn.ReLU6(inplace=True),
                nn.ConvTranspose1d(96, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU6(inplace=True),
                nn.ConvTranspose1d(32, self.output_dim, kernel_size=4, stride=2, padding=1)
            )
            self.task_type = 'regression'
    
    def forward(self, x: torch.Tensor, data_id=None, task_id=None) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, input_dim)
        data_id : Any, optional
            Data identifier (unused)
        task_id : Any, optional
            Task identifier (unused)
            
        Returns
        -------
        torch.Tensor
            Output tensor shape depends on task type:
            - Classification: (B, num_classes)
            - Regression: (B, L, output_dim)
        """
        # Transpose for convolution: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)
        
        # Apply feature extraction
        x = self.features(x)
        
        if self.task_type == 'classification':
            # Global average pooling and classification
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        else:
            # Upsampling for regression
            x = self.upsample(x)
            # Transpose back: (B, C, L) -> (B, L, C)
            x = x.transpose(1, 2)
        
        return x


if __name__ == "__main__":
    # Test MobileNet1D
    import torch
    from argparse import Namespace
    
    def test_mobilenet1d():
        """Test MobileNet1D with different configurations."""
        print("Testing MobileNet1D...")
        
        # Test classification configuration
        args_cls = Namespace(
            input_dim=3,
            width_multiplier=0.5,  # Smaller for testing
            inverted_residual_setting=[
                [1, 16, 1, 1],
                [6, 24, 1, 2],
                [6, 32, 1, 2],
            ],
            dropout=0.2,
            num_classes=5
        )
        
        model_cls = Model(args_cls)
        print(f"Classification model parameters: {sum(p.numel() for p in model_cls.parameters()):,}")
        
        # Test data
        batch_size = 4
        seq_len = 128
        x = torch.randn(batch_size, seq_len, args_cls.input_dim)
        
        # Forward pass
        with torch.no_grad():
            output_cls = model_cls(x)
        
        print(f"Classification - Input: {x.shape}, Output: {output_cls.shape}")
        assert output_cls.shape == (batch_size, args_cls.num_classes)
        
        # Test with different width multiplier
        args_wide = Namespace(
            input_dim=3,
            width_multiplier=0.25,  # Even smaller
            inverted_residual_setting=[
                [1, 16, 1, 1],
                [6, 24, 1, 2],
            ],
            dropout=0.1,
            num_classes=4
        )
        
        model_wide = Model(args_wide)
        print(f"Wide model parameters: {sum(p.numel() for p in model_wide.parameters()):,}")
        
        with torch.no_grad():
            output_wide = model_wide(x)
        
        print(f"Wide model - Input: {x.shape}, Output: {output_wide.shape}")
        assert output_wide.shape == (batch_size, args_wide.num_classes)
        
        print("✅ MobileNet1D tests passed!")
        return True
    
    test_mobilenet1d()
