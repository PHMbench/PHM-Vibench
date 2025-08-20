"""Residual RNN with skip connections for deep recurrent networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ResidualLSTMLayer(nn.Module):
    """LSTM layer with residual connections.
    
    Parameters
    ----------
    input_dim : int
        Input dimension
    hidden_dim : int
        Hidden dimension
    dropout : float
        Dropout probability
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super(ResidualLSTMLayer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Projection layer for residual connection (if dimensions don't match)
        if input_dim != hidden_dim:
            self.projection = nn.Linear(input_dim, hidden_dim)
        else:
            self.projection = nn.Identity()
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, input_dim)
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, L, hidden_dim)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (B, L, hidden_dim)
        
        # Residual connection
        residual = self.projection(x)  # (B, L, hidden_dim)
        output = lstm_out + residual
        
        # Layer normalization and dropout
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output


class ResidualGRULayer(nn.Module):
    """GRU layer with residual connections.
    
    Parameters
    ----------
    input_dim : int
        Input dimension
    hidden_dim : int
        Hidden dimension
    dropout : float
        Dropout probability
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super(ResidualGRULayer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Projection layer for residual connection (if dimensions don't match)
        if input_dim != hidden_dim:
            self.projection = nn.Linear(input_dim, hidden_dim)
        else:
            self.projection = nn.Identity()
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, input_dim)
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, L, hidden_dim)
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)  # (B, L, hidden_dim)
        
        # Residual connection
        residual = self.projection(x)  # (B, L, hidden_dim)
        output = gru_out + residual
        
        # Layer normalization and dropout
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output


class Model(nn.Module):
    """Residual RNN with skip connections for deep recurrent networks.
    
    Implements deep RNNs with residual connections to enable training
    of very deep recurrent networks while maintaining gradient flow.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - hidden_dim : int, hidden dimension (default: 128)
        - num_layers : int, number of RNN layers (default: 6)
        - rnn_type : str, type of RNN ('lstm' or 'gru', default: 'lstm')
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
    He et al. "Deep Residual Learning for Image Recognition" CVPR 2016.
    Hochreiter and Schmidhuber "Long Short-Term Memory" Neural Computation 1997.
    Kim et al. "Residual LSTM: Design of a Deep Recurrent Architecture for Distant Speech Recognition" INTERSPEECH 2017.
    Adapted for time-series industrial signals with residual connections for deep RNN training and gradient flow.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.hidden_dim = getattr(args, 'hidden_dim', 128)
        self.num_layers = getattr(args, 'num_layers', 6)
        self.rnn_type = getattr(args, 'rnn_type', 'lstm').lower()
        self.dropout = getattr(args, 'dropout', 0.1)
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Residual RNN layers
        self.rnn_layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            if self.rnn_type == 'lstm':
                layer = ResidualLSTMLayer(self.hidden_dim, self.hidden_dim, self.dropout)
            elif self.rnn_type == 'gru':
                layer = ResidualGRULayer(self.hidden_dim, self.hidden_dim, self.dropout)
            else:
                raise ValueError(f"Unsupported RNN type: {self.rnn_type}")
            
            self.rnn_layers.append(layer)
        
        # Output layers
        if self.num_classes is not None:
            # Classification: global pooling + classifier
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, self.num_classes)
            )
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
        
        # Apply residual RNN layers
        for rnn_layer in self.rnn_layers:
            x = rnn_layer(x)
        
        if self.task_type == 'classification':
            # Global average pooling and classification
            x = x.mean(dim=1)  # (B, hidden_dim)
            x = self.classifier(x)  # (B, num_classes)
        else:
            # Sequence-to-sequence regression
            x = self.output_projection(x)  # (B, L, output_dim)
        
        return x


if __name__ == "__main__":
    # Test Residual RNN
    import torch
    from argparse import Namespace
    
    def test_residual_rnn():
        """Test Residual RNN with different configurations."""
        print("Testing Residual RNN...")
        
        # Test LSTM regression configuration
        args_lstm_reg = Namespace(
            input_dim=3,
            hidden_dim=64,
            num_layers=4,
            rnn_type='lstm',
            dropout=0.1,
            output_dim=3
        )
        
        model_lstm_reg = Model(args_lstm_reg)
        print(f"LSTM Regression model parameters: {sum(p.numel() for p in model_lstm_reg.parameters()):,}")
        
        # Test data
        batch_size = 4
        seq_len = 32
        x = torch.randn(batch_size, seq_len, args_lstm_reg.input_dim)
        
        # Forward pass
        with torch.no_grad():
            output_lstm_reg = model_lstm_reg(x)
        
        print(f"LSTM Regression - Input: {x.shape}, Output: {output_lstm_reg.shape}")
        assert output_lstm_reg.shape == (batch_size, seq_len, args_lstm_reg.output_dim)
        
        # Test GRU classification configuration
        args_gru_cls = Namespace(
            input_dim=3,
            hidden_dim=64,
            num_layers=4,
            rnn_type='gru',
            dropout=0.1,
            num_classes=5
        )
        
        model_gru_cls = Model(args_gru_cls)
        print(f"GRU Classification model parameters: {sum(p.numel() for p in model_gru_cls.parameters()):,}")
        
        with torch.no_grad():
            output_gru_cls = model_gru_cls(x)
        
        print(f"GRU Classification - Input: {x.shape}, Output: {output_gru_cls.shape}")
        assert output_gru_cls.shape == (batch_size, args_gru_cls.num_classes)
        
        # Test deep network (more layers)
        args_deep = Namespace(
            input_dim=3,
            hidden_dim=32,
            num_layers=8,
            rnn_type='lstm',
            dropout=0.2,
            num_classes=3
        )
        
        model_deep = Model(args_deep)
        print(f"Deep model parameters: {sum(p.numel() for p in model_deep.parameters()):,}")
        
        with torch.no_grad():
            output_deep = model_deep(x)
        
        print(f"Deep model - Input: {x.shape}, Output: {output_deep.shape}")
        assert output_deep.shape == (batch_size, args_deep.num_classes)
        
        print("✅ Residual RNN tests passed!")
        return True
    
    test_residual_rnn()
