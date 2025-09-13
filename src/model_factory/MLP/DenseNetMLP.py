"""DenseNet-style MLP with dense connections for time-series analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DenseLayer(nn.Module):
    """Dense layer with growth connections.
    
    Parameters
    ----------
    input_dim : int
        Input dimension
    growth_rate : int
        Growth rate (number of new features added)
    dropout : float
        Dropout probability
    activation : str
        Activation function name
    """
    
    def __init__(self, input_dim: int, growth_rate: int, dropout: float = 0.1, 
                 activation: str = 'relu'):
        super(DenseLayer, self).__init__()
        
        # Bottleneck layer (optional, reduces computation)
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 4 * growth_rate),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Main layer
        self.main = nn.Sequential(
            nn.LayerNorm(4 * growth_rate),
            nn.Linear(4 * growth_rate, growth_rate),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, input_dim)
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, L, input_dim + growth_rate)
        """
        # Apply bottleneck and main layers
        out = self.bottleneck(x)
        out = self.main(out)
        
        # Concatenate with input (dense connection)
        return torch.cat([x, out], dim=-1)


class DenseBlock(nn.Module):
    """Dense block containing multiple dense layers.
    
    Parameters
    ----------
    input_dim : int
        Input dimension
    num_layers : int
        Number of dense layers in the block
    growth_rate : int
        Growth rate for each layer
    dropout : float
        Dropout probability
    activation : str
        Activation function name
    """
    
    def __init__(self, input_dim: int, num_layers: int, growth_rate: int,
                 dropout: float = 0.1, activation: str = 'relu'):
        super(DenseBlock, self).__init__()
        
        self.layers = nn.ModuleList()
        current_dim = input_dim
        
        for i in range(num_layers):
            layer = DenseLayer(current_dim, growth_rate, dropout, activation)
            self.layers.append(layer)
            current_dim += growth_rate
        
        self.output_dim = current_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, input_dim)
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, L, output_dim)
        """
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    """Transition layer to reduce feature dimensions between dense blocks.
    
    Parameters
    ----------
    input_dim : int
        Input dimension
    output_dim : int
        Output dimension
    dropout : float
        Dropout probability
    """
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super(TransitionLayer, self).__init__()
        
        self.transition = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, input_dim)
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, L, output_dim)
        """
        return self.transition(x)


class Model(nn.Module):
    """DenseNet-style MLP for time-series analysis.
    
    A densely connected MLP where each layer receives feature maps from all
    preceding layers, promoting feature reuse and gradient flow.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - growth_rate : int, growth rate for dense layers (default: 32)
        - block_config : List[int], number of layers in each dense block (default: [6, 12, 24, 16])
        - compression : float, compression factor for transition layers (default: 0.5)
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
    Huang et al. "Densely Connected Convolutional Networks" CVPR 2017.
    He et al. "Deep Residual Learning for Image Recognition" CVPR 2016.
    Ioffe and Szegedy "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" ICML 2015.
    Adapted for time-series industrial signals with dense MLP connections for feature reuse and gradient flow.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.growth_rate = getattr(args, 'growth_rate', 32)
        self.block_config = getattr(args, 'block_config', [6, 12, 24, 16])
        self.compression = getattr(args, 'compression', 0.5)
        self.dropout = getattr(args, 'dropout', 0.1)
        self.activation = getattr(args, 'activation', 'relu')
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        
        # Initial feature dimension
        initial_features = 64
        self.input_projection = nn.Linear(self.input_dim, initial_features)
        
        # Build dense blocks and transition layers
        self.features = nn.ModuleList()
        current_features = initial_features
        
        for i, num_layers in enumerate(self.block_config):
            # Dense block
            block = DenseBlock(
                input_dim=current_features,
                num_layers=num_layers,
                growth_rate=self.growth_rate,
                dropout=self.dropout,
                activation=self.activation
            )
            self.features.append(block)
            current_features = block.output_dim
            
            # Transition layer (except for the last block)
            if i != len(self.block_config) - 1:
                transition_features = int(current_features * self.compression)
                transition = TransitionLayer(
                    input_dim=current_features,
                    output_dim=transition_features,
                    dropout=self.dropout
                )
                self.features.append(transition)
                current_features = transition_features
        
        # Final normalization
        self.final_norm = nn.LayerNorm(current_features)
        
        # Output layers
        if self.num_classes is not None:
            # Classification: global pooling + classifier
            self.classifier = nn.Linear(current_features, self.num_classes)
            self.task_type = 'classification'
        else:
            # Regression: sequence-to-sequence
            self.output_projection = nn.Linear(current_features, self.output_dim)
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
        x = self.input_projection(x)  # (B, L, initial_features)
        
        # Apply dense blocks and transition layers
        for layer in self.features:
            x = layer(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        if self.task_type == 'classification':
            # Global average pooling and classification
            x = x.mean(dim=1)  # (B, current_features)
            x = self.classifier(x)  # (B, num_classes)
        else:
            # Sequence-to-sequence regression
            x = self.output_projection(x)  # (B, L, output_dim)
        
        return x


if __name__ == "__main__":
    # Test DenseNet-style MLP
    import torch
    from argparse import Namespace
    
    def test_densenet_mlp():
        """Test DenseNet-style MLP with different configurations."""
        print("Testing DenseNet-style MLP...")
        
        # Test regression configuration
        args_reg = Namespace(
            input_dim=2,
            growth_rate=16,
            block_config=[4, 6, 8],  # Smaller for testing
            compression=0.5,
            dropout=0.1,
            activation='relu',
            output_dim=2
        )
        
        model_reg = Model(args_reg)
        print(f"Regression model parameters: {sum(p.numel() for p in model_reg.parameters()):,}")
        
        # Test data
        batch_size = 4
        seq_len = 64
        x = torch.randn(batch_size, seq_len, args_reg.input_dim)
        
        # Forward pass
        with torch.no_grad():
            output_reg = model_reg(x)
        
        print(f"Regression - Input: {x.shape}, Output: {output_reg.shape}")
        assert output_reg.shape == (batch_size, seq_len, args_reg.output_dim)
        
        # Test classification configuration
        args_cls = Namespace(
            input_dim=2,
            growth_rate=16,
            block_config=[4, 6, 8],
            compression=0.5,
            dropout=0.1,
            activation='gelu',
            num_classes=3
        )
        
        model_cls = Model(args_cls)
        print(f"Classification model parameters: {sum(p.numel() for p in model_cls.parameters()):,}")
        
        with torch.no_grad():
            output_cls = model_cls(x)
        
        print(f"Classification - Input: {x.shape}, Output: {output_cls.shape}")
        assert output_cls.shape == (batch_size, args_cls.num_classes)
        
        print("✅ DenseNet-style MLP tests passed!")
        return True
    
    test_densenet_mlp()
