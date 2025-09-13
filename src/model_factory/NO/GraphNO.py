"""Graph Neural Operator for structured sensor network data."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class GraphConvLayer(nn.Module):
    """Graph convolution layer for processing sensor network data.
    
    Parameters
    ----------
    in_channels : int
        Input feature dimension
    out_channels : int
        Output feature dimension
    dropout : float
        Dropout probability
    """
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super(GraphConvLayer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Linear transformations
        self.linear_self = nn.Linear(in_channels, out_channels)
        self.linear_neighbor = nn.Linear(in_channels, out_channels)
        
        # Normalization and dropout
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Activation
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (B, N, in_channels)
        adj : torch.Tensor
            Adjacency matrix of shape (N, N) or (B, N, N)
            
        Returns
        -------
        torch.Tensor
            Output features of shape (B, N, out_channels)
        """
        # Self transformation
        x_self = self.linear_self(x)  # (B, N, out_channels)
        
        # Neighbor aggregation
        if adj.dim() == 2:
            # Shared adjacency matrix
            x_neighbor = torch.matmul(adj, x)  # (N, N) @ (B, N, in_channels) -> (B, N, in_channels)
        else:
            # Batch-specific adjacency matrices
            x_neighbor = torch.bmm(adj, x)  # (B, N, N) @ (B, N, in_channels) -> (B, N, in_channels)
        
        x_neighbor = self.linear_neighbor(x_neighbor)  # (B, N, out_channels)
        
        # Combine self and neighbor information
        out = x_self + x_neighbor
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        return out


class SpectralGraphConv(nn.Module):
    """Spectral graph convolution using graph Fourier transform.
    
    Parameters
    ----------
    in_channels : int
        Input feature dimension
    out_channels : int
        Output feature dimension
    num_modes : int
        Number of spectral modes to keep
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_modes: int = 8):
        super(SpectralGraphConv, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_modes = num_modes
        
        # Spectral weights
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, num_modes, dtype=torch.cfloat) * 0.02
        )
    
    def forward(self, x: torch.Tensor, eigenvectors: torch.Tensor) -> torch.Tensor:
        """Forward pass using spectral convolution.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (B, N, in_channels)
        eigenvectors : torch.Tensor
            Graph Laplacian eigenvectors of shape (N, N)

        Returns
        -------
        torch.Tensor
            Output features of shape (B, N, out_channels)
        """
        batch_size, num_nodes, _ = x.shape

        # Reshape for batch matrix multiplication
        x_reshaped = x.view(batch_size * num_nodes, self.in_channels)  # (B*N, in_channels)

        # Graph Fourier Transform for each sample in batch
        x_freq_list = []
        for b in range(batch_size):
            x_b = x[b]  # (N, in_channels)
            x_freq_b = torch.matmul(eigenvectors.T, x_b)  # (N, in_channels)
            x_freq_list.append(x_freq_b)

        x_freq = torch.stack(x_freq_list, dim=0)  # (B, N, in_channels)

        # Apply spectral convolution (only on low frequencies)
        x_freq_complex = torch.complex(x_freq, torch.zeros_like(x_freq))
        out_freq = torch.zeros(batch_size, num_nodes, self.out_channels,
                              dtype=torch.cfloat, device=x.device)

        modes_to_use = min(self.num_modes, num_nodes)
        for i in range(self.in_channels):
            for j in range(self.out_channels):
                out_freq[:, :modes_to_use, j] += (
                    x_freq_complex[:, :modes_to_use, i] * self.weights[i, j, :modes_to_use]
                )

        # Inverse Graph Fourier Transform for each sample in batch
        out_list = []
        for b in range(batch_size):
            out_freq_b = out_freq[b]  # (N, out_channels)
            out_b = torch.matmul(eigenvectors, out_freq_b.real)  # (N, out_channels)
            out_list.append(out_b)

        out = torch.stack(out_list, dim=0)  # (B, N, out_channels)

        return out


class Model(nn.Module):
    """Graph Neural Operator for structured sensor network data.
    
    A neural operator that operates on graph-structured data, suitable for
    sensor networks where spatial relationships between sensors are important.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension per node
        - hidden_dim : int, hidden dimension (default: 64)
        - output_dim : int, output dimension per node (default: input_dim)
        - num_nodes : int, number of nodes in the graph (default: 10)
        - num_layers : int, number of graph conv layers (default: 4)
        - num_modes : int, number of spectral modes (default: 8)
        - use_spectral : bool, whether to use spectral convolution (default: True)
        - dropout : float, dropout probability (default: 0.1)
        - num_classes : int, number of output classes (for classification)
    metadata : Any, optional
        Dataset metadata
        
    Input Shape
    -----------
    x : torch.Tensor
        Input tensor of shape (batch_size, seq_len, input_dim)
        Will be reshaped to (batch_size, num_nodes, features_per_node)
        
    Output Shape
    ------------
    torch.Tensor
        For classification: (batch_size, num_classes)
        For regression: (batch_size, seq_len, output_dim)
        
    References
    ----------
    Li et al. "Neural Operator: Graph Kernel Network for Partial Differential Equations" ICLR 2020 Workshop.
    Kipf and Welling "Semi-Supervised Classification with Graph Convolutional Networks" ICLR 2017.
    Defferrard et al. "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering" NeurIPS 2016.
    Adapted for time-series industrial signals with graph-based spectral convolutions for structured data modeling.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.hidden_dim = getattr(args, 'hidden_dim', 64)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        self.num_nodes = getattr(args, 'num_nodes', 10)
        self.num_layers = getattr(args, 'num_layers', 4)
        self.num_modes = getattr(args, 'num_modes', 8)
        self.use_spectral = getattr(args, 'use_spectral', True)
        self.dropout = getattr(args, 'dropout', 0.1)
        self.num_classes = getattr(args, 'num_classes', None)
        
        # Calculate features per node
        self.features_per_node = self.input_dim // self.num_nodes
        if self.input_dim % self.num_nodes != 0:
            self.features_per_node += 1
        
        # Input projection
        self.input_projection = nn.Linear(self.features_per_node, self.hidden_dim)
        
        # Graph convolution layers
        self.graph_layers = nn.ModuleList()
        for i in range(self.num_layers):
            if self.use_spectral and i % 2 == 0:
                # Use spectral convolution for even layers
                layer = SpectralGraphConv(self.hidden_dim, self.hidden_dim, self.num_modes)
            else:
                # Use spatial convolution for odd layers
                layer = GraphConvLayer(self.hidden_dim, self.hidden_dim, self.dropout)
            self.graph_layers.append(layer)
        
        # Create default adjacency matrix (fully connected)
        self.register_buffer('adj_matrix', self._create_default_adjacency())
        
        # Create graph Laplacian eigenvectors for spectral convolution
        if self.use_spectral:
            self.register_buffer('eigenvectors', self._compute_eigenvectors())
        
        # Output layers
        if self.num_classes is not None:
            # Classification: global pooling + classifier
            self.classifier = nn.Linear(self.hidden_dim, self.num_classes)
            self.task_type = 'classification'
        else:
            # Regression: node-wise output
            self.output_projection = nn.Linear(self.hidden_dim, self.features_per_node)
            self.task_type = 'regression'
    
    def _create_default_adjacency(self) -> torch.Tensor:
        """Create a default adjacency matrix (fully connected)."""
        adj = torch.ones(self.num_nodes, self.num_nodes)
        adj = adj - torch.eye(self.num_nodes)  # Remove self-loops
        # Normalize adjacency matrix
        degree = adj.sum(dim=1, keepdim=True)
        adj = adj / (degree + 1e-8)
        return adj
    
    def _compute_eigenvectors(self) -> torch.Tensor:
        """Compute eigenvectors of the graph Laplacian."""
        # Compute Laplacian
        degree = self.adj_matrix.sum(dim=1)
        laplacian = torch.diag(degree) - self.adj_matrix
        
        # Compute eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        return eigenvectors
    
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
        batch_size, seq_len, _ = x.shape
        
        # Reshape input to graph format
        # Pad input if necessary
        padded_size = self.num_nodes * self.features_per_node
        if self.input_dim < padded_size:
            padding = torch.zeros(batch_size, seq_len, padded_size - self.input_dim, device=x.device)
            x = torch.cat([x, padding], dim=-1)
        
        x = x[:, :, :padded_size]  # Truncate if too large
        x = x.view(batch_size, seq_len, self.num_nodes, self.features_per_node)
        
        # Process each time step
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # (B, num_nodes, features_per_node)
            
            # Input projection
            h = self.input_projection(x_t)  # (B, num_nodes, hidden_dim)
            
            # Apply graph convolution layers
            for i, layer in enumerate(self.graph_layers):
                if isinstance(layer, SpectralGraphConv):
                    h = layer(h, self.eigenvectors)
                else:
                    h = layer(h, self.adj_matrix)
            
            if self.task_type == 'classification':
                # Global pooling for classification
                h_pooled = h.mean(dim=1)  # (B, hidden_dim)
                outputs.append(h_pooled)
            else:
                # Node-wise output for regression
                h_out = self.output_projection(h)  # (B, num_nodes, features_per_node)
                h_out = h_out.view(batch_size, -1)  # (B, num_nodes * features_per_node)
                h_out = h_out[:, :self.output_dim]  # Truncate to output_dim
                outputs.append(h_out)
        
        if self.task_type == 'classification':
            # Use final time step for classification
            final_features = outputs[-1]  # (B, hidden_dim)
            output = self.classifier(final_features)  # (B, num_classes)
        else:
            # Stack all time steps for regression
            output = torch.stack(outputs, dim=1)  # (B, seq_len, output_dim)
        
        return output


if __name__ == "__main__":
    # Test Graph Neural Operator
    import torch
    from argparse import Namespace
    
    def test_graph_no():
        """Test Graph Neural Operator with different configurations."""
        print("Testing Graph Neural Operator...")
        
        # Test regression configuration
        args_reg = Namespace(
            input_dim=20,  # Will be divided among nodes
            hidden_dim=32,
            output_dim=20,
            num_nodes=5,
            num_layers=3,
            num_modes=4,
            use_spectral=True,
            dropout=0.1
        )
        
        model_reg = Model(args_reg)
        print(f"Regression model parameters: {sum(p.numel() for p in model_reg.parameters()):,}")
        
        # Test data
        batch_size = 4
        seq_len = 8
        x = torch.randn(batch_size, seq_len, args_reg.input_dim)
        
        # Forward pass
        with torch.no_grad():
            output_reg = model_reg(x)
        
        print(f"Regression - Input: {x.shape}, Output: {output_reg.shape}")
        assert output_reg.shape == (batch_size, seq_len, args_reg.output_dim)
        
        # Test classification configuration
        args_cls = Namespace(
            input_dim=20,
            hidden_dim=32,
            num_nodes=5,
            num_layers=3,
            num_modes=4,
            use_spectral=False,  # Test without spectral convolution
            dropout=0.1,
            num_classes=3
        )
        
        model_cls = Model(args_cls)
        print(f"Classification model parameters: {sum(p.numel() for p in model_cls.parameters()):,}")
        
        with torch.no_grad():
            output_cls = model_cls(x)
        
        print(f"Classification - Input: {x.shape}, Output: {output_cls.shape}")
        assert output_cls.shape == (batch_size, args_cls.num_classes)
        
        print("✅ Graph Neural Operator tests passed!")
        return True
    
    test_graph_no()
