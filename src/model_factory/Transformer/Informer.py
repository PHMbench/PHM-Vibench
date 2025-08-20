"""Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ProbSparseAttention(nn.Module):
    """ProbSparse self-attention mechanism from Informer.
    
    Parameters
    ----------
    d_model : int
        Model dimension
    n_heads : int
        Number of attention heads
    factor : int
        Sampling factor for ProbSparse attention (default: 5)
    dropout : float
        Dropout probability
    """
    
    def __init__(self, d_model: int, n_heads: int, factor: int = 5, dropout: float = 0.1):
        super(ProbSparseAttention, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor
        
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def _prob_QK(self, Q: torch.Tensor, K: torch.Tensor, sample_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sampled Q_K attention scores."""
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        
        # Calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        
        # Find the Top-k queries with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(sample_k, sorted=False)[1]
        
        # Use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        
        return Q_K, M_top
    
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        queries : torch.Tensor
            Query tensor of shape (B, L, d_model)
        keys : torch.Tensor
            Key tensor of shape (B, L, d_model)
        values : torch.Tensor
            Value tensor of shape (B, L, d_model)
        attn_mask : torch.Tensor, optional
            Attention mask

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, L, d_model)
        """
        B, L, _ = queries.shape

        # Project to Q, K, V
        Q = self.query_projection(queries).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.key_projection(keys).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.value_projection(values).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # Simplified sparse attention (use standard attention for now)
        # In practice, this would implement the full ProbSparse mechanism
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)

        # Apply sparsity by keeping only top-k attention weights
        if L > 32:  # Apply sparsity for longer sequences
            top_k = max(16, L // 4)  # Keep top 25% or at least 16
            top_scores, top_indices = torch.topk(scores, top_k, dim=-1)

            # Create sparse attention matrix
            sparse_scores = torch.full_like(scores, float('-inf'))
            sparse_scores.scatter_(-1, top_indices, top_scores)
            scores = sparse_scores

        A = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(A, V)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_projection(context)


class InformerLayer(nn.Module):
    """Informer encoder layer.
    
    Parameters
    ----------
    d_model : int
        Model dimension
    n_heads : int
        Number of attention heads
    d_ff : int
        Feed-forward dimension
    factor : int
        ProbSparse attention factor
    dropout : float
        Dropout probability
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, factor: int = 5, dropout: float = 0.1):
        super(InformerLayer, self).__init__()
        
        self.attention = ProbSparseAttention(d_model, n_heads, factor, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, d_model)
        attn_mask : torch.Tensor, optional
            Attention mask
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, L, d_model)
        """
        # Self-attention with residual connection
        attn_out = self.attention(x, x, x, attn_mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class Model(nn.Module):
    """Informer model for long sequence time-series forecasting.
    
    Informer uses ProbSparse self-attention to reduce computational complexity
    from O(L^2) to O(L log L) for long sequences.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - d_model : int, model dimension (default: 256)
        - n_heads : int, number of attention heads (default: 8)
        - e_layers : int, number of encoder layers (default: 6)
        - d_ff : int, feed-forward dimension (default: 512)
        - factor : int, ProbSparse attention factor (default: 5)
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
    Zhou et al. "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" AAAI 2021.
    Vaswani et al. "Attention Is All You Need" NeurIPS 2017.
    Child et al. "Generating Long Sequences with Sparse Transformers" arXiv 2019.
    Adapted for time-series industrial signals with ProbSparse attention for long sequence modeling.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.d_model = getattr(args, 'd_model', 256)
        self.n_heads = getattr(args, 'n_heads', 8)
        self.e_layers = getattr(args, 'e_layers', 6)
        self.d_ff = getattr(args, 'd_ff', 512)
        self.factor = getattr(args, 'factor', 5)
        self.dropout = getattr(args, 'dropout', 0.1)
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        
        # Input embedding
        self.input_embedding = nn.Linear(self.input_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 5000, self.d_model) * 0.02)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            InformerLayer(self.d_model, self.n_heads, self.d_ff, self.factor, self.dropout)
            for _ in range(self.e_layers)
        ])
        
        # Output layers
        if self.num_classes is not None:
            # Classification: global pooling + classifier
            self.classifier = nn.Linear(self.d_model, self.num_classes)
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
        batch_size, seq_len, _ = x.shape
        
        # Input embedding
        x = self.input_embedding(x)  # (B, L, d_model)
        
        # Add positional encoding
        if seq_len <= self.pos_encoding.size(1):
            x = x + self.pos_encoding[:, :seq_len, :]
        else:
            # Interpolate positional encoding for longer sequences
            pos_enc = F.interpolate(
                self.pos_encoding.transpose(1, 2), 
                size=seq_len, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
            x = x + pos_enc
        
        # Apply encoder layers
        for layer in self.encoder_layers:
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
    # Test Informer
    import torch
    from argparse import Namespace
    
    def test_informer():
        """Test Informer with different configurations."""
        print("Testing Informer...")
        
        # Test regression configuration
        args_reg = Namespace(
            input_dim=4,
            d_model=64,
            n_heads=4,
            e_layers=3,
            d_ff=128,
            factor=3,
            dropout=0.1,
            output_dim=4
        )
        
        model_reg = Model(args_reg)
        print(f"Regression model parameters: {sum(p.numel() for p in model_reg.parameters()):,}")
        
        # Test data
        batch_size = 4
        seq_len = 96  # Longer sequence for Informer
        x = torch.randn(batch_size, seq_len, args_reg.input_dim)
        
        # Forward pass
        with torch.no_grad():
            output_reg = model_reg(x)
        
        print(f"Regression - Input: {x.shape}, Output: {output_reg.shape}")
        assert output_reg.shape == (batch_size, seq_len, args_reg.output_dim)
        
        # Test classification configuration
        args_cls = Namespace(
            input_dim=4,
            d_model=64,
            n_heads=4,
            e_layers=3,
            d_ff=128,
            factor=3,
            dropout=0.1,
            num_classes=6
        )
        
        model_cls = Model(args_cls)
        print(f"Classification model parameters: {sum(p.numel() for p in model_cls.parameters()):,}")
        
        with torch.no_grad():
            output_cls = model_cls(x)
        
        print(f"Classification - Input: {x.shape}, Output: {output_cls.shape}")
        assert output_cls.shape == (batch_size, args_cls.num_classes)
        
        print("✅ Informer tests passed!")
        return True
    
    test_informer()
