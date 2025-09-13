"""MLP-Mixer architecture adapted for time-series analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MixerBlock(nn.Module):
    """MLP-Mixer block with token-mixing and channel-mixing MLPs.
    
    Parameters
    ----------
    seq_len : int
        Sequence length (number of tokens)
    hidden_dim : int
        Hidden dimension (channel dimension)
    tokens_mlp_dim : int
        Hidden dimension for token-mixing MLP
    channels_mlp_dim : int
        Hidden dimension for channel-mixing MLP
    dropout : float
        Dropout probability
    """
    
    def __init__(self, seq_len: int, hidden_dim: int, tokens_mlp_dim: int, 
                 channels_mlp_dim: int, dropout: float = 0.1):
        super(MixerBlock, self).__init__()
        
        # Token-mixing MLP (operates on sequence dimension)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(seq_len, tokens_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tokens_mlp_dim, seq_len),
            nn.Dropout(dropout)
        )
        
        # Channel-mixing MLP (operates on feature dimension)
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, channels_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels_mlp_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, C)
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, L, C)
        """
        # Token-mixing: mix information across sequence dimension
        residual = x
        x = self.token_norm(x)  # (B, L, C)
        x = x.transpose(1, 2)   # (B, C, L)
        x = self.token_mlp(x)   # (B, C, L)
        x = x.transpose(1, 2)   # (B, L, C)
        x = x + residual
        
        # Channel-mixing: mix information across feature dimension
        residual = x
        x = self.channel_norm(x)  # (B, L, C)
        x = self.channel_mlp(x)   # (B, L, C)
        x = x + residual
        
        return x


class Model(nn.Module):
    """MLP-Mixer for time-series analysis.
    
    An architecture based purely on MLPs that alternates between token-mixing
    (across sequence positions) and channel-mixing (across features) operations.
    Adapted from the original MLP-Mixer for time-series data.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - seq_len : int, sequence length (default: 512)
        - hidden_dim : int, hidden dimension (default: 256)
        - num_layers : int, number of mixer blocks (default: 8)
        - tokens_mlp_dim : int, token-mixing MLP hidden dim (default: 512)
        - channels_mlp_dim : int, channel-mixing MLP hidden dim (default: 512)
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
    Tolstikhin et al. "MLP-Mixer: An all-MLP Architecture for Vision" NeurIPS 2021.
    Ba et al. "Layer Normalization" arXiv 2016.
    Hendrycks and Gimpel "Gaussian Error Linear Units (GELUs)" arXiv 2016.
    Adapted for time-series industrial signals with temporal and channel mixing for sequence modeling.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.seq_len = getattr(args, 'seq_len', 512)
        self.hidden_dim = getattr(args, 'hidden_dim', 256)
        self.num_layers = getattr(args, 'num_layers', 8)
        self.tokens_mlp_dim = getattr(args, 'tokens_mlp_dim', 512)
        self.channels_mlp_dim = getattr(args, 'channels_mlp_dim', 512)
        self.dropout = getattr(args, 'dropout', 0.1)
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        
        # Input projection (patch embedding equivalent)
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Mixer blocks
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(
                seq_len=self.seq_len,
                hidden_dim=self.hidden_dim,
                tokens_mlp_dim=self.tokens_mlp_dim,
                channels_mlp_dim=self.channels_mlp_dim,
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
        
        # Apply mixer blocks
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        
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
    # Test MLP-Mixer
    import torch
    from argparse import Namespace
    
    def test_mlp_mixer():
        """Test MLP-Mixer with different configurations."""
        print("Testing MLP-Mixer...")
        
        # Test regression configuration
        args_reg = Namespace(
            input_dim=4,
            seq_len=128,
            hidden_dim=64,
            num_layers=4,
            tokens_mlp_dim=128,
            channels_mlp_dim=128,
            dropout=0.1,
            output_dim=4
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
            input_dim=4,
            seq_len=128,
            hidden_dim=64,
            num_layers=4,
            tokens_mlp_dim=128,
            channels_mlp_dim=128,
            dropout=0.1,
            num_classes=5
        )
        
        model_cls = Model(args_cls)
        print(f"Classification model parameters: {sum(p.numel() for p in model_cls.parameters()):,}")
        
        with torch.no_grad():
            output_cls = model_cls(x)
        
        print(f"Classification - Input: {x.shape}, Output: {output_cls.shape}")
        assert output_cls.shape == (batch_size, args_cls.num_classes)
        
        print("✅ MLP-Mixer tests passed!")
        return True
    
    test_mlp_mixer()
