"""Multi-scale CNN with Inception-style modules for time-series analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class InceptionModule1D(nn.Module):
    """Inception module for 1D time-series data.
    
    Parameters
    ----------
    in_channels : int
        Input channels
    ch1x1 : int
        Channels for 1x1 convolution branch
    ch3x3red : int
        Channels for 3x3 convolution reduction
    ch3x3 : int
        Channels for 3x3 convolution
    ch5x5red : int
        Channels for 5x5 convolution reduction
    ch5x5 : int
        Channels for 5x5 convolution
    pool_proj : int
        Channels for pooling projection
    """
    
    def __init__(self, in_channels: int, ch1x1: int, ch3x3red: int, ch3x3: int,
                 ch5x5red: int, ch5x5: int, pool_proj: int):
        super(InceptionModule1D, self).__init__()
        
        # 1x1 convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm1d(ch1x1),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolution branch
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm1d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch3x3),
            nn.ReLU(inplace=True)
        )
        
        # 5x5 convolution branch
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm1d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm1d(ch5x5),
            nn.ReLU(inplace=True)
        )
        
        # Max pooling branch
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm1d(pool_proj),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        # Concatenate all branches
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class MultiScaleBlock(nn.Module):
    """Multi-scale block with different kernel sizes.
    
    Parameters
    ----------
    in_channels : int
        Input channels
    out_channels : int
        Output channels
    scales : list
        List of kernel sizes for different scales
    """
    
    def __init__(self, in_channels: int, out_channels: int, scales: List[int] = [3, 5, 7, 9]):
        super(MultiScaleBlock, self).__init__()
        
        self.scales = scales
        self.num_scales = len(scales)
        
        # Each scale gets equal share of output channels
        channels_per_scale = out_channels // self.num_scales
        remaining_channels = out_channels % self.num_scales
        
        self.scale_convs = nn.ModuleList()
        for i, scale in enumerate(scales):
            # Last scale gets any remaining channels
            scale_channels = channels_per_scale + (remaining_channels if i == len(scales) - 1 else 0)
            
            conv = nn.Sequential(
                nn.Conv1d(in_channels, scale_channels, kernel_size=scale, padding=scale//2),
                nn.BatchNorm1d(scale_channels),
                nn.ReLU(inplace=True)
            )
            self.scale_convs.append(conv)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        scale_outputs = []
        for conv in self.scale_convs:
            scale_outputs.append(conv(x))
        
        # Concatenate multi-scale features
        multi_scale = torch.cat(scale_outputs, dim=1)
        
        # Fusion
        output = self.fusion(multi_scale)
        
        return output


class Model(nn.Module):
    """Multi-scale CNN for time-series analysis.
    
    Combines Inception-style modules and multi-scale blocks to capture
    features at different temporal scales simultaneously.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - inception_config : list, configuration for inception modules
        - multiscale_channels : list, channels for multi-scale blocks (default: [64, 128, 256])
        - scales : list, kernel sizes for multi-scale blocks (default: [3, 5, 7])
        - dropout : float, dropout probability (default: 0.1)
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
    Szegedy et al. "Going Deeper with Convolutions" CVPR 2015.
    Szegedy et al. "Rethinking the Inception Architecture for Computer Vision" CVPR 2016.
    LeCun et al. "Gradient-based learning applied to document recognition" Proceedings of the IEEE 1998.
    Adapted for time-series industrial signals with Inception-style multi-scale convolutions for temporal feature extraction.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.multiscale_channels = getattr(args, 'multiscale_channels', [64, 128, 256])
        self.scales = getattr(args, 'scales', [3, 5, 7])
        self.dropout = getattr(args, 'dropout', 0.1)
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.input_dim, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Inception modules
        self.inception3a = InceptionModule1D(64, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule1D(256, 128, 128, 192, 32, 96, 64)
        
        self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = InceptionModule1D(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule1D(512, 160, 112, 224, 24, 64, 64)
        
        # Multi-scale blocks
        self.multiscale_blocks = nn.ModuleList()
        in_channels = 512
        
        for out_channels in self.multiscale_channels:
            block = MultiScaleBlock(in_channels, out_channels, self.scales)
            self.multiscale_blocks.append(block)
            in_channels = out_channels
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Output layers
        final_channels = self.multiscale_channels[-1]
        
        if self.num_classes is not None:
            # Classification
            self.classifier = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(final_channels, self.num_classes)
            )
            self.task_type = 'classification'
        else:
            # Regression: upsampling + projection
            self.upsample = nn.Sequential(
                nn.ConvTranspose1d(final_channels, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose1d(64, self.output_dim, kernel_size=4, stride=2, padding=1)
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
        
        # Initial convolution
        x = self.conv1(x)
        
        # Inception modules
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        
        # Multi-scale blocks
        for block in self.multiscale_blocks:
            x = block(x)
        
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
    # Test Multi-scale CNN
    import torch
    from argparse import Namespace
    
    def test_multiscale_cnn():
        """Test Multi-scale CNN with different configurations."""
        print("Testing Multi-scale CNN...")
        
        # Test classification configuration
        args_cls = Namespace(
            input_dim=3,
            multiscale_channels=[32, 64],  # Smaller for testing
            scales=[3, 5],
            dropout=0.1,
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
        
        # Test with different scales
        args_scales = Namespace(
            input_dim=3,
            multiscale_channels=[32, 64],
            scales=[3, 5, 7, 9],
            dropout=0.1,
            num_classes=4
        )
        
        model_scales = Model(args_scales)
        print(f"Multi-scale model parameters: {sum(p.numel() for p in model_scales.parameters()):,}")
        
        with torch.no_grad():
            output_scales = model_scales(x)
        
        print(f"Multi-scale - Input: {x.shape}, Output: {output_scales.shape}")
        assert output_scales.shape == (batch_size, args_scales.num_classes)
        
        print("✅ Multi-scale CNN tests passed!")
        return True
    
    test_multiscale_cnn()
