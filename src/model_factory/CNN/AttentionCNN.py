"""Attention-enhanced CNN for time-series analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class ChannelAttention(nn.Module):
    """Channel attention module (Squeeze-and-Excitation).
    
    Parameters
    ----------
    in_channels : int
        Number of input channels
    reduction : int
        Reduction ratio for bottleneck
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, L)
            
        Returns
        -------
        torch.Tensor
            Attention weights of shape (B, C, 1)
        """
        b, c, _ = x.size()
        
        # Average pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        
        # Max pooling
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply sigmoid
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1)


class SpatialAttention(nn.Module):
    """Spatial attention module.
    
    Parameters
    ----------
    kernel_size : int
        Convolution kernel size
    """
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, L)
            
        Returns
        -------
        torch.Tensor
            Attention weights of shape (B, 1, L)
        """
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, L)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, L)
        
        # Concatenate and apply convolution
        out = torch.cat([avg_out, max_out], dim=1)  # (B, 2, L)
        out = self.conv(out)  # (B, 1, L)
        
        return self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM).
    
    Parameters
    ----------
    in_channels : int
        Number of input channels
    reduction : int
        Reduction ratio for channel attention
    kernel_size : int
        Kernel size for spatial attention
    """
    
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, L)
            
        Returns
        -------
        torch.Tensor
            Attention-enhanced tensor of shape (B, C, L)
        """
        # Channel attention
        x = x * self.channel_attention(x)
        
        # Spatial attention
        x = x * self.spatial_attention(x)
        
        return x


class AttentionConvBlock(nn.Module):
    """Convolutional block with attention mechanism.
    
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
    use_attention : bool
        Whether to use attention mechanism
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, use_attention: bool = True):
        super(AttentionConvBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM(out_channels)
        
        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_attention:
            out = self.attention(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class Model(nn.Module):
    """Attention-enhanced CNN for time-series analysis.
    
    Combines convolutional layers with attention mechanisms (CBAM)
    to improve feature representation and model performance.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - channels : list, number of channels in each layer (default: [64, 128, 256, 512])
        - use_attention : bool, whether to use attention (default: True)
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
    Woo et al. "CBAM: Convolutional Block Attention Module" ECCV 2018.
    Hu et al. "Squeeze-and-Excitation Networks" CVPR 2018.
    LeCun et al. "Gradient-based learning applied to document recognition" Proceedings of the IEEE 1998.
    Adapted for time-series industrial signals with channel and spatial attention mechanisms for feature enhancement.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.channels = getattr(args, 'channels', [64, 128, 256, 512])
        self.use_attention = getattr(args, 'use_attention', True)
        self.dropout = getattr(args, 'dropout', 0.1)
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        
        # Initial convolution
        self.conv1 = nn.Conv1d(self.input_dim, self.channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Attention-enhanced convolutional blocks
        self.layers = nn.ModuleList()
        
        for i in range(len(self.channels)):
            in_channels = self.channels[i-1] if i > 0 else self.channels[0]
            out_channels = self.channels[i]
            stride = 2 if i > 0 else 1
            
            layer = AttentionConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_attention=self.use_attention
            )
            self.layers.append(layer)
        
        # Output layers
        final_channels = self.channels[-1]
        
        if self.num_classes is not None:
            # Classification: global pooling + classifier
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(final_channels, self.num_classes)
            )
            self.task_type = 'classification'
        else:
            # Regression: upsampling + projection
            self.upsample = nn.Sequential(
                nn.ConvTranspose1d(final_channels, self.channels[-2], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(self.channels[-2]),
                nn.ReLU(inplace=True),
                nn.ConvTranspose1d(self.channels[-2], self.channels[-3], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(self.channels[-3]),
                nn.ReLU(inplace=True),
                nn.ConvTranspose1d(self.channels[-3], self.channels[0], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(self.channels[0]),
                nn.ReLU(inplace=True),
                nn.ConvTranspose1d(self.channels[0], self.output_dim, kernel_size=4, stride=2, padding=1)
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
        
        # Initial convolution and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Apply attention-enhanced convolutional blocks
        for layer in self.layers:
            x = layer(x)
        
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
    # Test Attention-enhanced CNN
    import torch
    from argparse import Namespace
    
    def test_attention_cnn():
        """Test Attention-enhanced CNN with different configurations."""
        print("Testing Attention-enhanced CNN...")
        
        # Test classification configuration
        args_cls = Namespace(
            input_dim=3,
            channels=[32, 64, 128],  # Smaller for testing
            use_attention=True,
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
        
        # Test without attention
        args_no_attn = Namespace(
            input_dim=3,
            channels=[32, 64],
            use_attention=False,
            dropout=0.1,
            num_classes=4
        )
        
        model_no_attn = Model(args_no_attn)
        print(f"No attention model parameters: {sum(p.numel() for p in model_no_attn.parameters()):,}")
        
        with torch.no_grad():
            output_no_attn = model_no_attn(x)
        
        print(f"No attention - Input: {x.shape}, Output: {output_no_attn.shape}")
        assert output_no_attn.shape == (batch_size, args_no_attn.num_classes)
        
        print("✅ Attention-enhanced CNN tests passed!")
        return True
    
    test_attention_cnn()
