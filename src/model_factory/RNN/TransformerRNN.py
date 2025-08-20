"""Transformer-RNN Hybrid for time-series analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


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


class TransformerRNNBlock(nn.Module):
    """Hybrid block combining Transformer and RNN.
    
    Parameters
    ----------
    d_model : int
        Model dimension
    n_heads : int
        Number of attention heads
    d_ff : int
        Feed-forward dimension
    rnn_type : str
        Type of RNN ('lstm' or 'gru')
    dropout : float
        Dropout probability
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, rnn_type: str = 'lstm', dropout: float = 0.1):
        super(TransformerRNNBlock, self).__init__()
        
        self.d_model = d_model
        self.rnn_type = rnn_type.lower()
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # RNN layer
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=1,
                batch_first=True
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=1,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, d_model)
        mask : torch.Tensor, optional
            Attention mask
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, L, d_model)
        """
        # Self-attention with residual connection
        attn_out, _ = self.self_attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # RNN with residual connection
        rnn_out, _ = self.rnn(x)
        x = self.norm2(x + self.dropout(rnn_out))
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm3(x + ff_out)
        
        return x


class Model(nn.Module):
    """Transformer-RNN Hybrid for time-series analysis.
    
    Combines the global attention capabilities of Transformers with
    the sequential modeling strengths of RNNs for enhanced performance.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - d_model : int, model dimension (default: 256)
        - n_heads : int, number of attention heads (default: 8)
        - num_layers : int, number of hybrid layers (default: 4)
        - d_ff : int, feed-forward dimension (default: 512)
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
    Vaswani et al. "Attention Is All You Need" NeurIPS 2017.
    Hochreiter and Schmidhuber "Long Short-Term Memory" Neural Computation 1997.
    Lei et al. "Simple Recurrent Units for Highly Parallelizable Recurrence" EMNLP 2018.
    Adapted for time-series industrial signals with hybrid Transformer-RNN architecture for global and local temporal modeling.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.d_model = getattr(args, 'd_model', 256)
        self.n_heads = getattr(args, 'n_heads', 8)
        self.num_layers = getattr(args, 'num_layers', 4)
        self.d_ff = getattr(args, 'd_ff', 512)
        self.rnn_type = getattr(args, 'rnn_type', 'lstm')
        self.dropout = getattr(args, 'dropout', 0.1)
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        
        # Input embedding
        self.input_embedding = nn.Linear(self.input_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model)
        
        # Transformer-RNN hybrid layers
        self.hybrid_layers = nn.ModuleList([
            TransformerRNNBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                rnn_type=self.rnn_type,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])
        
        # Output layers
        if self.num_classes is not None:
            # Classification: global pooling + classifier
            self.classifier = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model // 2, self.num_classes)
            )
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
        # Input embedding
        x = self.input_embedding(x)  # (B, L, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply hybrid layers
        for layer in self.hybrid_layers:
            x = layer(x)
        
        if self.task_type == 'classification':
            # Global average pooling and classification
            x = x.mean(dim=1)  # (B, d_model)
            x = self.classifier(x)  # (B, num_classes)
        else:
            # Sequence-to-sequence regression
            x = self.output_projection(x)  # (B, L, output_dim)
        
        return x


if __name__ == "__main__":
    # Test Transformer-RNN Hybrid
    import torch
    from argparse import Namespace
    
    def test_transformer_rnn():
        """Test Transformer-RNN Hybrid with different configurations."""
        print("Testing Transformer-RNN Hybrid...")
        
        # Test LSTM regression configuration
        args_lstm_reg = Namespace(
            input_dim=3,
            d_model=64,
            n_heads=4,
            num_layers=3,
            d_ff=128,
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
            d_model=64,
            n_heads=4,
            num_layers=3,
            d_ff=128,
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
        
        # Test with different model dimensions
        args_large = Namespace(
            input_dim=3,
            d_model=128,
            n_heads=8,
            num_layers=2,
            d_ff=256,
            rnn_type='lstm',
            dropout=0.1,
            num_classes=4
        )
        
        model_large = Model(args_large)
        print(f"Large model parameters: {sum(p.numel() for p in model_large.parameters()):,}")
        
        with torch.no_grad():
            output_large = model_large(x)
        
        print(f"Large model - Input: {x.shape}, Output: {output_large.shape}")
        assert output_large.shape == (batch_size, args_large.num_classes)
        
        print("✅ Transformer-RNN Hybrid tests passed!")
        return True
    
    test_transformer_rnn()
