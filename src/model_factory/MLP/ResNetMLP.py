"""ResNet-style MLP with skip connections for time-series analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualBlock(nn.Module):
    """Residual block with skip connections for MLPs.
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension size
    dropout : float
        Dropout probability
    activation : str
        Activation function name ('relu', 'gelu', 'silu')
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1, activation: str = 'relu'):
        super(ResidualBlock, self).__init__()
        
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.
        
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
        
        # First sub-layer
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Second sub-layer
        x = self.norm2(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        # Residual connection
        return x + residual


class Model(nn.Module):
    """ResNet-style MLP for time-series analysis.
    
    A deep MLP with residual connections that enables training of very deep networks
    while maintaining gradient flow. Suitable for complex time-series modeling tasks.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - hidden_dim : int, hidden layer dimension (default: 256)
        - num_layers : int, number of residual blocks (default: 6)
        - dropout : float, dropout probability (default: 0.1)
        - activation : str, activation function (default: 'relu')
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
    He et al. "Deep Residual Learning for Image Recognition" CVPR 2016.
    Ioffe and Szegedy "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" ICML 2015.
    Srivastava et al. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" JMLR 2014.
    Adapted for time-series industrial signals with residual MLP blocks for deep feature learning.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.hidden_dim = getattr(args, 'hidden_dim', 256)
        self.num_layers = getattr(args, 'num_layers', 6)
        self.dropout = getattr(args, 'dropout', 0.1)
        self.activation = getattr(args, 'activation', 'relu')
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(self.hidden_dim, self.dropout, self.activation)
            for _ in range(self.num_layers)
        ])
        
        # Output layers
        if self.num_classes is not None:
            # Classification: global pooling + classifier
            self.global_pool = nn.AdaptiveAvgPool1d(1)
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
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        if self.task_type == 'classification':
            # Global average pooling and classification
            x = x.transpose(1, 2)  # (B, hidden_dim, L)
            x = self.global_pool(x).squeeze(-1)  # (B, hidden_dim)
            x = self.classifier(x)  # (B, num_classes)
        else:
            # Sequence-to-sequence regression
            x = self.output_projection(x)  # (B, L, output_dim)
        
        return x


if __name__ == "__main__":
    # Test ResNet-style MLP
    import torch
    from argparse import Namespace
    
    def test_resnet_mlp():
        """Test ResNet-style MLP with different configurations."""
        print("Testing ResNet-style MLP...")
        
        # Test regression configuration
        args_reg = Namespace(
            input_dim=3,
            hidden_dim=128,
            num_layers=4,
            dropout=0.1,
            activation='relu',
            output_dim=3
        )
        
        model_reg = Model(args_reg)
        print(f"Regression model parameters: {sum(p.numel() for p in model_reg.parameters()):,}")
        
        # Test data
        batch_size = 16
        seq_len = 256
        x = torch.randn(batch_size, seq_len, args_reg.input_dim)
        
        # Forward pass
        with torch.no_grad():
            output_reg = model_reg(x)
        
        print(f"Regression - Input: {x.shape}, Output: {output_reg.shape}")
        assert output_reg.shape == (batch_size, seq_len, args_reg.output_dim)
        
        # Test classification configuration
        args_cls = Namespace(
            input_dim=3,
            hidden_dim=128,
            num_layers=4,
            dropout=0.1,
            activation='gelu',
            num_classes=10
        )
        
        model_cls = Model(args_cls)
        print(f"Classification model parameters: {sum(p.numel() for p in model_cls.parameters()):,}")
        
        with torch.no_grad():
            output_cls = model_cls(x)
        
        print(f"Classification - Input: {x.shape}, Output: {output_cls.shape}")
        assert output_cls.shape == (batch_size, args_cls.num_classes)
        
        print("✅ ResNet-style MLP tests passed!")
        return True
    
    test_resnet_mlp()
