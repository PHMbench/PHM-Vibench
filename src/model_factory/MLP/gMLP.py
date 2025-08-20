"""gMLP (Gated MLP) architecture for time-series analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpatialGatingUnit(nn.Module):
    """Spatial Gating Unit for gMLP.
    
    Parameters
    ----------
    seq_len : int
        Sequence length
    hidden_dim : int
        Hidden dimension (should be even for splitting)
    """
    
    def __init__(self, seq_len: int, hidden_dim: int):
        super(SpatialGatingUnit, self).__init__()
        
        assert hidden_dim % 2 == 0, "Hidden dimension must be even for gMLP"
        
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.half_dim = hidden_dim // 2
        
        # Layer normalization
        self.norm = nn.LayerNorm(self.half_dim)
        
        # Spatial projection (operates on sequence dimension)
        self.spatial_proj = nn.Linear(seq_len, seq_len)
        
        # Initialize spatial projection to identity
        nn.init.zeros_(self.spatial_proj.weight)
        nn.init.ones_(self.spatial_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, hidden_dim)
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, L, hidden_dim//2)
        """
        # Split input into two halves
        u, v = x.chunk(2, dim=-1)  # Each: (B, L, hidden_dim//2)
        
        # Apply layer normalization to v
        v = self.norm(v)
        
        # Apply spatial gating
        v = v.transpose(1, 2)  # (B, hidden_dim//2, L)
        v = self.spatial_proj(v)  # (B, hidden_dim//2, L)
        v = v.transpose(1, 2)  # (B, L, hidden_dim//2)
        
        # Gated multiplication
        return u * v


class gMLPBlock(nn.Module):
    """gMLP block with spatial gating.
    
    Parameters
    ----------
    seq_len : int
        Sequence length
    hidden_dim : int
        Hidden dimension
    ffn_dim : int
        Feed-forward network dimension
    dropout : float
        Dropout probability
    """
    
    def __init__(self, seq_len: int, hidden_dim: int, ffn_dim: int, dropout: float = 0.1):
        super(gMLPBlock, self).__init__()
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Channel projection (expand)
        self.channel_proj1 = nn.Linear(hidden_dim, ffn_dim)
        
        # Spatial gating unit
        self.sgu = SpatialGatingUnit(seq_len, ffn_dim)
        
        # Channel projection (contract)
        self.channel_proj2 = nn.Linear(ffn_dim // 2, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, hidden_dim)
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, L, hidden_dim)
        """
        residual = x
        
        # Layer normalization
        x = self.norm(x)
        
        # Channel projection (expand)
        x = self.channel_proj1(x)
        x = self.activation(x)
        
        # Spatial gating
        x = self.sgu(x)
        
        # Channel projection (contract)
        x = self.channel_proj2(x)
        x = self.dropout(x)
        
        # Residual connection
        return x + residual


class Model(nn.Module):
    """gMLP (Gated MLP) for time-series analysis.
    
    A gated MLP architecture that uses spatial gating units to enable
    cross-position communication while maintaining the efficiency of MLPs.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - seq_len : int, sequence length (default: 512)
        - hidden_dim : int, hidden dimension (default: 256, must be even)
        - num_layers : int, number of gMLP blocks (default: 6)
        - ffn_dim : int, feed-forward dimension (default: 512, must be even)
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
    Liu et al. "Pay Attention to MLPs" NeurIPS 2021.
    Dauphin et al. "Language Modeling with Gated Convolutional Networks" ICML 2017.
    Ba et al. "Layer Normalization" arXiv 2016.
    Adapted for time-series industrial signals with spatial gating units for temporal modeling.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.seq_len = getattr(args, 'seq_len', 512)
        self.hidden_dim = getattr(args, 'hidden_dim', 256)
        self.num_layers = getattr(args, 'num_layers', 6)
        self.ffn_dim = getattr(args, 'ffn_dim', 512)
        self.dropout = getattr(args, 'dropout', 0.1)
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        
        # Ensure dimensions are even for gating
        if self.hidden_dim % 2 != 0:
            self.hidden_dim += 1
        if self.ffn_dim % 2 != 0:
            self.ffn_dim += 1
        
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # gMLP blocks
        self.gmlp_blocks = nn.ModuleList([
            gMLPBlock(
                seq_len=self.seq_len,
                hidden_dim=self.hidden_dim,
                ffn_dim=self.ffn_dim,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])
        
        # Output layers
        self.norm = nn.LayerNorm(self.hidden_dim)
        
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
            Input tensor of shape (B, L, C)
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
        
        # Apply gMLP blocks
        for gmlp_block in self.gmlp_blocks:
            x = gmlp_block(x)
        
        # Final normalization
        x = self.norm(x)
        
        if self.task_type == 'classification':
            # Global average pooling and classification
            x = x.mean(dim=1)  # (B, hidden_dim)
            x = self.classifier(x)  # (B, num_classes)
        else:
            # Sequence-to-sequence regression
            x = self.output_projection(x)  # (B, L, output_dim)
        
        return x


if __name__ == "__main__":
    # Test gMLP
    import torch
    from argparse import Namespace
    
    def test_gmlp():
        """Test gMLP with different configurations."""
        print("Testing gMLP...")
        
        # Test regression configuration
        args_reg = Namespace(
            input_dim=3,
            seq_len=64,
            hidden_dim=128,  # Will be adjusted to even if needed
            num_layers=4,
            ffn_dim=256,     # Will be adjusted to even if needed
            dropout=0.1,
            output_dim=3
        )
        
        model_reg = Model(args_reg)
        print(f"Regression model parameters: {sum(p.numel() for p in model_reg.parameters()):,}")
        
        # Test data
        batch_size = 8
        x = torch.randn(batch_size, args_reg.seq_len, args_reg.input_dim)
        
        # Forward pass
        with torch.no_grad():
            output_reg = model_reg(x)
        
        print(f"Regression - Input: {x.shape}, Output: {output_reg.shape}")
        assert output_reg.shape == (batch_size, args_reg.seq_len, args_reg.output_dim)
        
        # Test classification configuration
        args_cls = Namespace(
            input_dim=3,
            seq_len=64,
            hidden_dim=128,
            num_layers=4,
            ffn_dim=256,
            dropout=0.1,
            num_classes=7
        )
        
        model_cls = Model(args_cls)
        print(f"Classification model parameters: {sum(p.numel() for p in model_cls.parameters()):,}")
        
        with torch.no_grad():
            output_cls = model_cls(x)
        
        print(f"Classification - Input: {x.shape}, Output: {output_cls.shape}")
        assert output_cls.shape == (batch_size, args_cls.num_classes)
        
        print("✅ gMLP tests passed!")
        return True
    
    test_gmlp()
