"""DeepONet (Deep Operator Network) for learning operators between function spaces."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BranchNet(nn.Module):
    """Branch network that encodes the input function.
    
    Parameters
    ----------
    input_dim : int
        Input dimension
    branch_layers : list
        List of hidden layer sizes for branch network
    activation : str
        Activation function name
    dropout : float
        Dropout probability
    """
    
    def __init__(self, input_dim: int, branch_layers: list, 
                 activation: str = 'relu', dropout: float = 0.1):
        super(BranchNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in branch_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU() if activation == 'relu' else nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.output_dim = prev_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, input_dim)
            
        Returns
        -------
        torch.Tensor
            Branch network output of shape (B, L, output_dim)
        """
        return self.network(x)


class TrunkNet(nn.Module):
    """Trunk network that encodes the evaluation coordinates.
    
    Parameters
    ----------
    coord_dim : int
        Coordinate dimension (typically 1 for time-series)
    trunk_layers : list
        List of hidden layer sizes for trunk network
    activation : str
        Activation function name
    dropout : float
        Dropout probability
    """
    
    def __init__(self, coord_dim: int, trunk_layers: list,
                 activation: str = 'relu', dropout: float = 0.1):
        super(TrunkNet, self).__init__()
        
        layers = []
        prev_dim = coord_dim
        
        for hidden_dim in trunk_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU() if activation == 'relu' else nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.output_dim = prev_dim
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        coords : torch.Tensor
            Coordinate tensor of shape (B, L, coord_dim)
            
        Returns
        -------
        torch.Tensor
            Trunk network output of shape (B, L, output_dim)
        """
        return self.network(coords)


class Model(nn.Module):
    """DeepONet for learning operators between function spaces.
    
    DeepONet learns mappings between infinite-dimensional function spaces
    by decomposing the operator into branch and trunk networks.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input function dimension
        - output_dim : int, output function dimension (default: input_dim)
        - branch_layers : list, branch network architecture (default: [128, 128, 128])
        - trunk_layers : list, trunk network architecture (default: [128, 128, 128])
        - coord_dim : int, coordinate dimension (default: 1)
        - activation : str, activation function (default: 'relu')
        - dropout : float, dropout probability (default: 0.1)
        - use_bias : bool, whether to use bias in final layer (default: True)
    metadata : Any, optional
        Dataset metadata
        
    Input Shape
    -----------
    x : torch.Tensor
        Input tensor of shape (batch_size, seq_len, input_dim)
        
    Output Shape
    ------------
    torch.Tensor
        Output tensor of shape (batch_size, seq_len, output_dim)
        
    References
    ----------
    Lu et al. "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators" Nature Machine Intelligence 2021.
    Chen and Chen "Universal Approximation to Nonlinear Operators by Neural Networks with Arbitrary Activation Functions and Its Application to Dynamical Systems" IEEE Transactions on Neural Networks 1995.
    Li et al. "Fourier Neural Operator for Parametric Partial Differential Equations" ICLR 2021.
    Adapted for time-series industrial signals with branch-trunk architecture for operator learning.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        self.branch_layers = getattr(args, 'branch_layers', [128, 128, 128])
        self.trunk_layers = getattr(args, 'trunk_layers', [128, 128, 128])
        self.coord_dim = getattr(args, 'coord_dim', 1)
        self.activation = getattr(args, 'activation', 'relu')
        self.dropout = getattr(args, 'dropout', 0.1)
        self.use_bias = getattr(args, 'use_bias', True)
        
        # Ensure branch and trunk have same output dimension
        if self.branch_layers[-1] != self.trunk_layers[-1]:
            # Adjust trunk layers to match branch output
            self.trunk_layers[-1] = self.branch_layers[-1]
        
        self.latent_dim = self.branch_layers[-1]
        
        # Branch network (encodes input function)
        self.branch_net = BranchNet(
            input_dim=self.input_dim,
            branch_layers=self.branch_layers,
            activation=self.activation,
            dropout=self.dropout
        )
        
        # Trunk network (encodes evaluation coordinates)
        self.trunk_net = TrunkNet(
            coord_dim=self.coord_dim,
            trunk_layers=self.trunk_layers,
            activation=self.activation,
            dropout=self.dropout
        )
        
        # Output projection
        self.output_projection = nn.Linear(self.latent_dim, self.output_dim, bias=self.use_bias)
    
    def _generate_coordinates(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate normalized coordinates for the sequence.
        
        Parameters
        ----------
        seq_len : int
            Sequence length
        device : torch.device
            Device to place coordinates on
            
        Returns
        -------
        torch.Tensor
            Coordinates of shape (seq_len, coord_dim)
        """
        # Generate normalized coordinates [0, 1]
        coords = torch.linspace(0, 1, seq_len, device=device)
        if self.coord_dim == 1:
            coords = coords.unsqueeze(-1)  # (seq_len, 1)
        else:
            # For higher dimensional coordinates, can be extended
            coords = coords.unsqueeze(-1).repeat(1, self.coord_dim)
        
        return coords
    
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
            Output tensor of shape (B, L, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Branch network: encode input function
        branch_output = self.branch_net(x)  # (B, L, latent_dim)
        
        # Generate coordinates and expand for batch
        coords = self._generate_coordinates(seq_len, device)  # (L, coord_dim)
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1)  # (B, L, coord_dim)
        
        # Trunk network: encode coordinates
        trunk_output = self.trunk_net(coords)  # (B, L, latent_dim)
        
        # Combine branch and trunk outputs (element-wise product)
        combined = branch_output * trunk_output  # (B, L, latent_dim)
        
        # Output projection
        output = self.output_projection(combined)  # (B, L, output_dim)
        
        return output


if __name__ == "__main__":
    # Test DeepONet
    import torch
    from argparse import Namespace
    
    def test_deeponet():
        """Test DeepONet with different configurations."""
        print("Testing DeepONet...")
        
        # Test configuration
        args = Namespace(
            input_dim=2,
            output_dim=2,
            branch_layers=[64, 64, 64],
            trunk_layers=[64, 64, 64],
            coord_dim=1,
            activation='relu',
            dropout=0.1,
            use_bias=True
        )
        
        model = Model(args)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test data
        batch_size = 4
        seq_len = 128
        x = torch.randn(batch_size, seq_len, args.input_dim)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        assert output.shape == (batch_size, seq_len, args.output_dim)
        
        # Test with different output dimension
        args.output_dim = 3
        model2 = Model(args)
        
        with torch.no_grad():
            output2 = model2(x)
        
        print(f"Output shape (different output_dim): {output2.shape}")
        assert output2.shape == (batch_size, seq_len, args.output_dim)
        
        print("✅ DeepONet tests passed!")
        return True
    
    test_deeponet()
