"""Attention-based GRU for time-series analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for GRU outputs.
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension
    num_heads : int
        Number of attention heads
    dropout : float
        Dropout probability
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads
        
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.out_projection = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply multi-head attention.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, hidden_dim)
        mask : torch.Tensor, optional
            Attention mask
            
        Returns
        -------
        torch.Tensor
            Attended output of shape (B, L, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.query_projection(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key_projection(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value_projection(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores.masked_fill_(mask.unsqueeze(1).unsqueeze(1), -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_projection(context)
        
        return output


class GRUWithAttention(nn.Module):
    """GRU layer with self-attention mechanism.
    
    Parameters
    ----------
    input_dim : int
        Input dimension
    hidden_dim : int
        Hidden dimension
    num_heads : int
        Number of attention heads
    dropout : float
        Dropout probability
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(GRUWithAttention, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
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
            Output tensor of shape (B, L, hidden_dim)
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)  # (B, L, hidden_dim)
        
        # Self-attention with residual connection
        attn_out = self.attention(gru_out)
        gru_out = self.norm1(gru_out + attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(gru_out)
        output = self.norm2(gru_out + ffn_out)
        
        return output


class Model(nn.Module):
    """Attention-based GRU for time-series analysis.
    
    Combines GRU with multi-head self-attention to capture both
    sequential dependencies and long-range relationships.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - hidden_dim : int, GRU hidden dimension (default: 128)
        - num_layers : int, number of GRU-attention layers (default: 3)
        - num_heads : int, number of attention heads (default: 8)
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
    Cho et al. "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" EMNLP 2014.
    Bahdanau et al. "Neural Machine Translation by Jointly Learning to Align and Translate" ICLR 2015.
    Adapted for time-series industrial signals with multi-head self-attention and GRU for enhanced sequential modeling.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.hidden_dim = getattr(args, 'hidden_dim', 128)
        self.num_layers = getattr(args, 'num_layers', 3)
        self.num_heads = getattr(args, 'num_heads', 8)
        self.dropout = getattr(args, 'dropout', 0.1)
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        
        # Ensure hidden_dim is divisible by num_heads
        if self.hidden_dim % self.num_heads != 0:
            self.hidden_dim = ((self.hidden_dim // self.num_heads) + 1) * self.num_heads
        
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # GRU-Attention layers
        self.gru_attention_layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            input_dim = self.hidden_dim if i > 0 else self.hidden_dim
            layer = GRUWithAttention(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout
            )
            self.gru_attention_layers.append(layer)
        
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
        
        # Apply GRU-Attention layers
        for layer in self.gru_attention_layers:
            x = layer(x)
        
        if self.task_type == 'classification':
            # Global average pooling and classification
            x = x.mean(dim=1)  # (B, hidden_dim)
            x = self.classifier(x)  # (B, num_classes)
        else:
            # Sequence-to-sequence regression
            x = self.output_projection(x)  # (B, L, output_dim)
        
        return x


if __name__ == "__main__":
    # Test Attention-based GRU
    import torch
    from argparse import Namespace
    
    def test_attention_gru():
        """Test Attention-based GRU with different configurations."""
        print("Testing Attention-based GRU...")
        
        # Test regression configuration
        args_reg = Namespace(
            input_dim=4,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            output_dim=4
        )
        
        model_reg = Model(args_reg)
        print(f"Regression model parameters: {sum(p.numel() for p in model_reg.parameters()):,}")
        
        # Test data
        batch_size = 4
        seq_len = 32
        x = torch.randn(batch_size, seq_len, args_reg.input_dim)
        
        # Forward pass
        with torch.no_grad():
            output_reg = model_reg(x)
        
        print(f"Regression - Input: {x.shape}, Output: {output_reg.shape}")
        assert output_reg.shape == (batch_size, seq_len, args_reg.output_dim)
        
        # Test classification configuration
        args_cls = Namespace(
            input_dim=4,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            num_classes=6
        )
        
        model_cls = Model(args_cls)
        print(f"Classification model parameters: {sum(p.numel() for p in model_cls.parameters()):,}")
        
        with torch.no_grad():
            output_cls = model_cls(x)
        
        print(f"Classification - Input: {x.shape}, Output: {output_cls.shape}")
        assert output_cls.shape == (batch_size, args_cls.num_classes)
        
        # Test with different head configuration
        args_heads = Namespace(
            input_dim=4,
            hidden_dim=96,  # Divisible by 8 heads
            num_layers=3,
            num_heads=8,
            dropout=0.1,
            num_classes=3
        )
        
        model_heads = Model(args_heads)
        print(f"Multi-head model parameters: {sum(p.numel() for p in model_heads.parameters()):,}")
        
        with torch.no_grad():
            output_heads = model_heads(x)
        
        print(f"Multi-head - Input: {x.shape}, Output: {output_heads.shape}")
        assert output_heads.shape == (batch_size, args_heads.num_classes)
        
        print("✅ Attention-based GRU tests passed!")
        return True
    
    test_attention_gru()
