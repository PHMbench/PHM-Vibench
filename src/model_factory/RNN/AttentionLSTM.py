"""Attention-based LSTM for time-series analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class AttentionMechanism(nn.Module):
    """Attention mechanism for LSTM hidden states.
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of LSTM
    attention_dim : int
        Attention mechanism dimension
    """
    
    def __init__(self, hidden_dim: int, attention_dim: int = 128):
        super(AttentionMechanism, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        # Attention layers
        self.attention_linear = nn.Linear(hidden_dim, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention to LSTM outputs.
        
        Parameters
        ----------
        lstm_outputs : torch.Tensor
            LSTM outputs of shape (B, L, hidden_dim)
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Attended output and attention weights
        """
        # Calculate attention scores
        attention_scores = self.attention_linear(lstm_outputs)  # (B, L, attention_dim)
        attention_scores = self.tanh(attention_scores)
        attention_scores = self.context_vector(attention_scores).squeeze(-1)  # (B, L)
        
        # Apply softmax to get attention weights
        attention_weights = self.softmax(attention_scores)  # (B, L)
        
        # Apply attention weights to LSTM outputs
        attended_output = torch.bmm(
            attention_weights.unsqueeze(1),  # (B, 1, L)
            lstm_outputs  # (B, L, hidden_dim)
        ).squeeze(1)  # (B, hidden_dim)
        
        return attended_output, attention_weights


class Model(nn.Module):
    """Attention-based LSTM for time-series analysis.
    
    Combines LSTM with attention mechanism to focus on important
    time steps for better sequence modeling performance.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - hidden_dim : int, LSTM hidden dimension (default: 128)
        - num_layers : int, number of LSTM layers (default: 2)
        - attention_dim : int, attention mechanism dimension (default: 64)
        - dropout : float, dropout probability (default: 0.1)
        - bidirectional : bool, whether to use bidirectional LSTM (default: True)
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
    Bahdanau et al. "Neural Machine Translation by Jointly Learning to Align and Translate" ICLR 2015.
    Hochreiter and Schmidhuber "Long Short-Term Memory" Neural Computation 1997.
    Luong et al. "Effective Approaches to Attention-based Neural Machine Translation" EMNLP 2015.
    Adapted for time-series industrial signals with attention mechanism for focusing on important temporal patterns.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.hidden_dim = getattr(args, 'hidden_dim', 128)
        self.num_layers = getattr(args, 'num_layers', 2)
        self.attention_dim = getattr(args, 'attention_dim', 64)
        self.dropout = getattr(args, 'dropout', 0.1)
        self.bidirectional = getattr(args, 'bidirectional', True)
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        
        # Calculate actual hidden dimension (doubled if bidirectional)
        self.actual_hidden_dim = self.hidden_dim * (2 if self.bidirectional else 1)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = AttentionMechanism(self.actual_hidden_dim, self.attention_dim)
        
        # Output layers
        if self.num_classes is not None:
            # Classification: use attended output
            self.classifier = nn.Sequential(
                nn.Linear(self.actual_hidden_dim, self.actual_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.actual_hidden_dim // 2, self.num_classes)
            )
            self.task_type = 'classification'
        else:
            # Regression: sequence-to-sequence
            self.output_projection = nn.Linear(self.actual_hidden_dim, self.output_dim)
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
        batch_size, seq_len, _ = x.shape
        
        # LSTM forward pass
        lstm_outputs, (hidden, cell) = self.lstm(x)  # (B, L, actual_hidden_dim)
        
        if self.task_type == 'classification':
            # Apply attention for classification
            attended_output, attention_weights = self.attention(lstm_outputs)  # (B, actual_hidden_dim)
            output = self.classifier(attended_output)  # (B, num_classes)
        else:
            # Sequence-to-sequence regression
            output = self.output_projection(lstm_outputs)  # (B, L, output_dim)
        
        return output


if __name__ == "__main__":
    # Test Attention-based LSTM
    import torch
    from argparse import Namespace
    
    def test_attention_lstm():
        """Test Attention-based LSTM with different configurations."""
        print("Testing Attention-based LSTM...")
        
        # Test regression configuration
        args_reg = Namespace(
            input_dim=3,
            hidden_dim=64,
            num_layers=2,
            attention_dim=32,
            dropout=0.1,
            bidirectional=True,
            output_dim=3
        )
        
        model_reg = Model(args_reg)
        print(f"Regression model parameters: {sum(p.numel() for p in model_reg.parameters()):,}")
        
        # Test data
        batch_size = 4
        seq_len = 50
        x = torch.randn(batch_size, seq_len, args_reg.input_dim)
        
        # Forward pass
        with torch.no_grad():
            output_reg = model_reg(x)
        
        print(f"Regression - Input: {x.shape}, Output: {output_reg.shape}")
        assert output_reg.shape == (batch_size, seq_len, args_reg.output_dim)
        
        # Test classification configuration
        args_cls = Namespace(
            input_dim=3,
            hidden_dim=64,
            num_layers=2,
            attention_dim=32,
            dropout=0.1,
            bidirectional=True,
            num_classes=5
        )
        
        model_cls = Model(args_cls)
        print(f"Classification model parameters: {sum(p.numel() for p in model_cls.parameters()):,}")
        
        with torch.no_grad():
            output_cls = model_cls(x)
        
        print(f"Classification - Input: {x.shape}, Output: {output_cls.shape}")
        assert output_cls.shape == (batch_size, args_cls.num_classes)
        
        # Test unidirectional LSTM
        args_uni = Namespace(
            input_dim=3,
            hidden_dim=64,
            num_layers=1,
            attention_dim=32,
            dropout=0.1,
            bidirectional=False,
            num_classes=5
        )
        
        model_uni = Model(args_uni)
        print(f"Unidirectional model parameters: {sum(p.numel() for p in model_uni.parameters()):,}")
        
        with torch.no_grad():
            output_uni = model_uni(x)
        
        print(f"Unidirectional - Input: {x.shape}, Output: {output_uni.shape}")
        assert output_uni.shape == (batch_size, args_uni.num_classes)
        
        print("✅ Attention-based LSTM tests passed!")
        return True
    
    test_attention_lstm()
