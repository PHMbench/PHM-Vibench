"""Wavelet Neural Operator combining wavelet transforms with neural operators."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class WaveletTransform(nn.Module):
    """Discrete Wavelet Transform layer.
    
    Parameters
    ----------
    wavelet_type : str
        Type of wavelet ('haar', 'db4')
    levels : int
        Number of decomposition levels
    """
    
    def __init__(self, wavelet_type: str = 'haar', levels: int = 3):
        super(WaveletTransform, self).__init__()
        
        self.wavelet_type = wavelet_type
        self.levels = levels
        
        # Define wavelet filters
        if wavelet_type == 'haar':
            # Haar wavelet filters
            self.h0 = torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)])  # Low-pass
            self.h1 = torch.tensor([1/math.sqrt(2), -1/math.sqrt(2)])  # High-pass
        elif wavelet_type == 'db4':
            # Daubechies-4 wavelet filters
            self.h0 = torch.tensor([
                0.6830127, 1.1830127, 0.3169873, -0.1830127
            ])
            self.h1 = torch.tensor([
                -0.1830127, -0.3169873, 1.1830127, -0.6830127
            ])
        else:
            raise ValueError(f"Unsupported wavelet type: {wavelet_type}")
        
        # Register as buffers
        self.register_buffer('low_pass', self.h0)
        self.register_buffer('high_pass', self.h1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """Forward wavelet transform.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, L)

        Returns
        -------
        Tuple[torch.Tensor, list]
            Approximation coefficients and list of detail coefficients
        """
        batch_size, channels, seq_len = x.shape
        coeffs = []
        current = x

        # Create filters for all channels
        low_filter = self.low_pass.view(1, 1, -1).repeat(channels, 1, 1)  # (C, 1, filter_len)
        high_filter = self.high_pass.view(1, 1, -1).repeat(channels, 1, 1)  # (C, 1, filter_len)

        for level in range(self.levels):
            # Ensure even length for convolution
            if current.size(-1) % 2 != 0:
                current = F.pad(current, (0, 1), mode='reflect')

            # Apply convolution with wavelet filters (grouped convolution)
            low = F.conv1d(current, low_filter, stride=2, padding=0, groups=channels)
            high = F.conv1d(current, high_filter, stride=2, padding=0, groups=channels)

            coeffs.append(high)
            current = low

        return current, coeffs
    
    def inverse(self, approx: torch.Tensor, details: list) -> torch.Tensor:
        """Inverse wavelet transform.
        
        Parameters
        ----------
        approx : torch.Tensor
            Approximation coefficients
        details : list
            List of detail coefficients
            
        Returns
        -------
        torch.Tensor
            Reconstructed signal
        """
        current = approx
        
        for level in range(self.levels - 1, -1, -1):
            detail = details[level]
            
            # Upsample and convolve
            current_up = F.interpolate(current, scale_factor=2, mode='nearest')
            detail_up = F.interpolate(detail, scale_factor=2, mode='nearest')
            
            # Combine approximation and detail
            current = current_up + detail_up
        
        return current


class WaveletConvLayer(nn.Module):
    """Simplified wavelet-inspired convolution layer.

    Parameters
    ----------
    in_channels : int
        Input channels
    out_channels : int
        Output channels
    kernel_size : int
        Convolution kernel size
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(WaveletConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Multi-scale convolutions to mimic wavelet decomposition
        self.conv_low = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv_high = nn.Conv1d(in_channels, out_channels, kernel_size, padding=2, dilation=2)

        # Learnable combination weights
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, L)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, out_channels, L)
        """
        # Multi-scale convolutions
        low_freq = self.conv_low(x)
        high_freq = self.conv_high(x)

        # Combine with learnable weights
        output = self.alpha * low_freq + self.beta * high_freq

        return output


class Model(nn.Module):
    """Wavelet Neural Operator for time-series analysis.
    
    Combines wavelet transforms with neural operators to capture both
    time-frequency characteristics and operator learning capabilities.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - hidden_dim : int, hidden dimension (default: 64)
        - output_dim : int, output dimension (default: input_dim)
        - num_layers : int, number of wavelet conv layers (default: 4)
        - wavelet_type : str, wavelet type (default: 'haar')
        - levels : int, wavelet decomposition levels (default: 3)
        - dropout : float, dropout probability (default: 0.1)
        - num_classes : int, number of output classes (for classification)
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
    Gupta et al. "Multiwavelet-based Operator Learning for Differential Equations" NeurIPS 2021.
    Mallat "A Wavelet Tour of Signal Processing" Academic Press 1999.
    Li et al. "Fourier Neural Operator for Parametric Partial Differential Equations" ICLR 2021.
    Adapted for time-series industrial signals with multi-scale wavelet decomposition for frequency-domain operator learning.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.hidden_dim = getattr(args, 'hidden_dim', 64)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        self.num_layers = getattr(args, 'num_layers', 4)
        self.wavelet_type = getattr(args, 'wavelet_type', 'haar')
        self.levels = getattr(args, 'levels', 3)
        self.dropout = getattr(args, 'dropout', 0.1)
        self.num_classes = getattr(args, 'num_classes', None)
        
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Wavelet convolution layers
        self.wavelet_layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = WaveletConvLayer(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                kernel_size=3
            )
            self.wavelet_layers.append(layer)
        
        # Normalization and dropout
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(self.dropout) for _ in range(self.num_layers)
        ])
        
        # Output layers
        if self.num_classes is not None:
            # Classification: global pooling + classifier
            self.classifier = nn.Linear(self.hidden_dim, self.num_classes)
            self.task_type = 'classification'
        else:
            # Regression: sequence-to-sequence
            self.output_projection = nn.Linear(self.hidden_dim, self.output_dim)
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
        # Input projection
        x = self.input_projection(x)  # (B, L, hidden_dim)
        
        # Transpose for convolution: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)
        
        # Apply wavelet convolution layers
        for i, (wavelet_layer, norm_layer, dropout_layer) in enumerate(
            zip(self.wavelet_layers, self.norm_layers, self.dropout_layers)
        ):
            residual = x
            
            # Wavelet convolution
            x = wavelet_layer(x)  # (B, hidden_dim, L)
            
            # Residual connection
            x = x + residual
            
            # Transpose for normalization: (B, C, L) -> (B, L, C)
            x = x.transpose(1, 2)
            x = norm_layer(x)
            x = dropout_layer(x)
            
            # Activation
            x = F.gelu(x)
            
            # Transpose back: (B, L, C) -> (B, C, L)
            x = x.transpose(1, 2)
        
        # Final transpose: (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)
        
        if self.task_type == 'classification':
            # Global average pooling and classification
            x = x.mean(dim=1)  # (B, hidden_dim)
            x = self.classifier(x)  # (B, num_classes)
        else:
            # Sequence-to-sequence regression
            x = self.output_projection(x)  # (B, L, output_dim)
        
        return x


if __name__ == "__main__":
    # Test Wavelet Neural Operator
    import torch
    from argparse import Namespace
    
    def test_wavelet_no():
        """Test Wavelet Neural Operator with different configurations."""
        print("Testing Wavelet Neural Operator...")
        
        # Test regression configuration
        args_reg = Namespace(
            input_dim=2,
            hidden_dim=32,
            output_dim=2,
            num_layers=3,
            wavelet_type='haar',
            levels=2,
            dropout=0.1
        )
        
        model_reg = Model(args_reg)
        print(f"Regression model parameters: {sum(p.numel() for p in model_reg.parameters()):,}")
        
        # Test data (use power of 2 length for wavelets)
        batch_size = 4
        seq_len = 64  # Power of 2
        x = torch.randn(batch_size, seq_len, args_reg.input_dim)
        
        # Forward pass
        with torch.no_grad():
            output_reg = model_reg(x)
        
        print(f"Regression - Input: {x.shape}, Output: {output_reg.shape}")
        assert output_reg.shape == (batch_size, seq_len, args_reg.output_dim)
        
        # Test classification configuration
        args_cls = Namespace(
            input_dim=2,
            hidden_dim=32,
            num_layers=3,
            wavelet_type='db4',
            levels=2,
            dropout=0.1,
            num_classes=4
        )
        
        model_cls = Model(args_cls)
        print(f"Classification model parameters: {sum(p.numel() for p in model_cls.parameters()):,}")
        
        with torch.no_grad():
            output_cls = model_cls(x)
        
        print(f"Classification - Input: {x.shape}, Output: {output_cls.shape}")
        assert output_cls.shape == (batch_size, args_cls.num_classes)
        
        print("✅ Wavelet Neural Operator tests passed!")
        return True
    
    test_wavelet_no()
