"""ConvTransformer: Hybrid CNN-Transformer for time-series analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class ConvBlock(nn.Module):
    """Convolutional block for local feature extraction.
    
    Parameters
    ----------
    in_channels : int
        Input channels
    out_channels : int
        Output channels
    kernel_size : int
        Convolution kernel size
    dropout : float
        Dropout probability
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
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
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out = out + residual
        out = self.activation(out)
        
        return out


class MultiScaleConv(nn.Module):
    """Multi-scale convolutional feature extraction.
    
    Parameters
    ----------
    in_channels : int
        Input channels
    out_channels : int
        Output channels
    scales : list
        List of kernel sizes for different scales
    dropout : float
        Dropout probability
    """
    
    def __init__(self, in_channels: int, out_channels: int, scales: list = [3, 5, 7], dropout: float = 0.1):
        super(MultiScaleConv, self).__init__()
        
        self.scales = scales
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels // len(scales), kernel_size=scale, padding=scale//2)
            for scale in scales
        ])
        
        # Additional conv to match output channels if needed
        total_out = (out_channels // len(scales)) * len(scales)
        if total_out != out_channels:
            self.adjust_conv = nn.Conv1d(total_out, out_channels, 1)
        else:
            self.adjust_conv = nn.Identity()
        
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
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
        scale_outputs = []
        for conv in self.scale_convs:
            scale_outputs.append(conv(x))
        
        # Concatenate multi-scale features
        out = torch.cat(scale_outputs, dim=1)
        out = self.adjust_conv(out)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        return out


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class Model(nn.Module):
    """ConvTransformer: Hybrid CNN-Transformer for time-series analysis.
    
    Combines convolutional layers for local feature extraction with
    transformer layers for long-range dependency modeling.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - conv_channels : list, convolutional layer channels (default: [64, 128, 256])
        - d_model : int, transformer model dimension (default: 256)
        - n_heads : int, number of attention heads (default: 8)
        - num_layers : int, number of transformer layers (default: 4)
        - d_ff : int, feed-forward dimension (default: 512)
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
    Liu et al. "A ConvNet for the 2020s" CVPR 2022.
    Vaswani et al. "Attention Is All You Need" NeurIPS 2017.
    Wu et al. "CvT: Introducing Convolutions to Vision Transformers" ICCV 2021.
    Adapted for time-series industrial signals with hybrid CNN-Transformer architecture for local and global feature modeling.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.conv_channels = getattr(args, 'conv_channels', [64, 128, 256])
        self.d_model = getattr(args, 'd_model', 256)
        self.n_heads = getattr(args, 'n_heads', 8)
        self.num_layers = getattr(args, 'num_layers', 4)
        self.d_ff = getattr(args, 'd_ff', 512)
        self.dropout = getattr(args, 'dropout', 0.1)
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        
        # Ensure d_model matches last conv channel
        if self.conv_channels[-1] != self.d_model:
            self.conv_channels[-1] = self.d_model
        
        # Convolutional feature extraction
        self.conv_layers = nn.ModuleList()
        
        # Initial multi-scale convolution
        self.conv_layers.append(
            MultiScaleConv(self.input_dim, self.conv_channels[0], dropout=self.dropout)
        )
        
        # Additional conv blocks
        for i in range(1, len(self.conv_channels)):
            self.conv_layers.append(
                ConvBlock(
                    self.conv_channels[i-1], 
                    self.conv_channels[i], 
                    dropout=self.dropout
                )
            )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Output layers
        if self.num_classes is not None:
            # Classification: global pooling + classifier
            self.classifier = nn.Linear(self.d_model, self.num_classes)
            self.task_type = 'classification'
        else:
            # Regression: sequence-to-sequence
            self.output_projection = nn.Linear(self.d_model, self.output_dim)
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
        
        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Transpose back for transformer: (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer
        x = self.transformer(x)
        
        if self.task_type == 'classification':
            # Global average pooling and classification
            x = x.mean(dim=1)  # (B, d_model)
            x = self.classifier(x)  # (B, num_classes)
        else:
            # Sequence-to-sequence regression
            x = self.output_projection(x)  # (B, L, output_dim)
        
        return x


if __name__ == "__main__":
    # Test ConvTransformer
    import torch
    from argparse import Namespace
    
    def test_conv_transformer():
        """Test ConvTransformer with different configurations."""
        print("Testing ConvTransformer...")
        
        # Test regression configuration
        args_reg = Namespace(
            input_dim=4,
            conv_channels=[32, 64, 128],
            d_model=128,
            n_heads=4,
            num_layers=3,
            d_ff=256,
            dropout=0.1,
            output_dim=4
        )
        
        model_reg = Model(args_reg)
        print(f"Regression model parameters: {sum(p.numel() for p in model_reg.parameters()):,}")
        
        # Test data
        batch_size = 4
        seq_len = 128
        x = torch.randn(batch_size, seq_len, args_reg.input_dim)
        
        # Forward pass
        with torch.no_grad():
            output_reg = model_reg(x)
        
        print(f"Regression - Input: {x.shape}, Output: {output_reg.shape}")
        assert output_reg.shape == (batch_size, seq_len, args_reg.output_dim)
        
        # Test classification configuration
        args_cls = Namespace(
            input_dim=4,
            conv_channels=[32, 64, 128],
            d_model=128,
            n_heads=4,
            num_layers=3,
            d_ff=256,
            dropout=0.1,
            num_classes=6
        )
        
        model_cls = Model(args_cls)
        print(f"Classification model parameters: {sum(p.numel() for p in model_cls.parameters()):,}")
        
        with torch.no_grad():
            output_cls = model_cls(x)
        
        print(f"Classification - Input: {x.shape}, Output: {output_cls.shape}")
        assert output_cls.shape == (batch_size, args_cls.num_classes)
        
        print("✅ ConvTransformer tests passed!")
        return True
    
    test_conv_transformer()
