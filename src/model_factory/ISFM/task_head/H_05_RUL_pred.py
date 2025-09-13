"""
RUL (Remaining Useful Life) Prediction Task Head

This module implements a neural network head specifically designed for RUL prediction tasks.
It takes feature representations from the backbone network and produces continuous RUL values.

Author: PHM-Vibench Team
Date: 2025-09-05
"""

import torch
import torch.nn as nn


class H_05_RUL_pred(nn.Module):
    """
    RUL prediction head for remaining useful life estimation.
    
    This module takes feature representations and outputs positive RUL values
    scaled to the appropriate range for the specific application.
    
    Parameters
    ----------
    args : Namespace
        Configuration containing:
        - output_dim: int, dimension of input features from backbone
        - rul_max_value: float, maximum expected RUL value for scaling (default: 1000.0)
        - hidden_dim: int, hidden layer dimension (default: output_dim // 2)
        - dropout: float, dropout probability (default: 0.1)
        - activation: str, activation function name (default: 'relu')
    """
    
    def __init__(self, args):
        super(H_05_RUL_pred, self).__init__()
        
        # Extract configuration parameters
        self.input_dim = args.output_dim
        self.hidden_dim = getattr(args, 'hidden_dim', args.output_dim // 2)
        self.rul_max_value = getattr(args, 'rul_max_value', 1000.0)
        self.dropout_prob = getattr(args, 'dropout', 0.1)
        self.activation_name = getattr(args, 'activation', 'relu')
        
        # Define activation function
        self.activation = self._get_activation(self.activation_name)
        
        # RUL prediction network
        self.rul_net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            self.activation,
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation,
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.ReLU()  # Ensure positive RUL values
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
        }
        return activations.get(activation_name.lower(), nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x, return_feature=False, **kwargs):
        """
        Forward pass through the RUL prediction head.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features from backbone with shape (B, L, C) or (B, C)
        return_feature : bool, optional
            If True, return intermediate features instead of final predictions
        
        Returns
        -------
        torch.Tensor
            RUL predictions with shape (B, 1), scaled to [0, rul_max_value]
        """
        # Handle input shape: (B, L, C) -> (B, C) via mean pooling
        if x.dim() == 3:
            x = x.mean(dim=1)  # Average pooling over sequence length
        elif x.dim() != 2:
            raise ValueError(f"Expected input with 2 or 3 dimensions, got {x.dim()}")
        
        # Get raw RUL prediction
        rul_raw = self.rul_net(x)
        
        if return_feature:
            return rul_raw  # Return raw values for feature extraction
        
        # Scale to appropriate RUL range
        rul_scaled = rul_raw * self.rul_max_value
        
        return rul_scaled


if __name__ == '__main__':
    """Unit tests for H_05_RUL_pred module."""
    from argparse import Namespace
    
    # Test configuration
    args = Namespace(
        output_dim=512,
        rul_max_value=2000.0,
        hidden_dim=256,
        dropout=0.1,
        activation='relu'
    )
    
    # Create model
    model = H_05_RUL_pred(args)
    print(f"RUL head created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test input shapes
    batch_size = 16
    seq_len = 128
    feature_dim = 512
    
    # Test with 3D input (B, L, C)
    x_3d = torch.randn(batch_size, seq_len, feature_dim)
    print(f"Input shape (3D): {x_3d.shape}")
    
    output_3d = model(x_3d)
    print(f"RUL output shape: {output_3d.shape}")
    assert output_3d.shape == (batch_size, 1), f"Expected (16, 1), got {output_3d.shape}"
    assert torch.all(output_3d >= 0), "RUL values should be non-negative"
    assert torch.all(output_3d <= 2000.0), f"RUL values should be <= max_value, got max={output_3d.max()}"
    
    # Test with 2D input (B, C)
    x_2d = torch.randn(batch_size, feature_dim)
    output_2d = model(x_2d)
    print(f"2D input output shape: {output_2d.shape}")
    assert output_2d.shape == (batch_size, 1), f"Expected (16, 1), got {output_2d.shape}"
    
    # Test feature extraction
    features = model(x_3d, return_feature=True)
    print(f"Feature shape: {features.shape}")
    assert features.shape == (batch_size, 1), f"Expected (16, 1), got {features.shape}"
    
    print("=== All RUL Head Tests Passed! ===")