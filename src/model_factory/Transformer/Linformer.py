"""Linformer: Self-Attention with Linear Complexity."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class LinearAttention(nn.Module):
    """Linear attention mechanism from Linformer.
    
    Parameters
    ----------
    d_model : int
        Model dimension
    n_heads : int
        Number of attention heads
    seq_len : int
        Maximum sequence length
    k : int
        Projected dimension for keys and values
    dropout : float
        Dropout probability
    """
    
    def __init__(self, d_model: int, n_heads: int, seq_len: int, k: int = 256, dropout: float = 0.1):
        super(LinearAttention, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.seq_len = seq_len
        self.k = k
        
        # Standard Q, K, V projections
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        
        # Linear projections for keys and values
        self.E = nn.Parameter(torch.randn(seq_len, k))
        self.F = nn.Parameter(torch.randn(seq_len, k))
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize projections
        nn.init.normal_(self.E, std=0.02)
        nn.init.normal_(self.F, std=0.02)
    
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
        B, L, _ = x.shape
        
        # Project to Q, K, V
        Q = self.query_projection(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, L, d_k)
        K = self.key_projection(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)    # (B, n_heads, L, d_k)
        V = self.value_projection(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, L, d_k)
        
        # Linear projections for K and V
        if L <= self.seq_len:
            E_truncated = self.E[:L, :]  # (L, k)
            F_truncated = self.F[:L, :]  # (L, k)
        else:
            # Interpolate for longer sequences
            E_truncated = F.interpolate(self.E.unsqueeze(0).unsqueeze(0), size=(L, self.k), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            F_truncated = F.interpolate(self.F.unsqueeze(0).unsqueeze(0), size=(L, self.k), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        
        # Apply linear projections: (B, n_heads, L, d_k) -> (B, n_heads, k, d_k)
        # K: (B, n_heads, L, d_k), E_truncated: (L, k) -> K_proj: (B, n_heads, k, d_k)
        K_proj = torch.einsum('bhld,lk->bhkd', K, E_truncated)  # (B, n_heads, k, d_k)
        V_proj = torch.einsum('bhld,lk->bhkd', V, F_truncated)  # (B, n_heads, k, d_k)
        
        # Compute attention scores: (B, n_heads, L, d_k) x (B, n_heads, d_k, k) -> (B, n_heads, L, k)
        scores = torch.matmul(Q, K_proj.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Adjust mask for projected dimension
            mask_proj = mask[:, :, :self.k] if mask.size(-1) >= self.k else F.pad(mask, (0, self.k - mask.size(-1)), value=float('-inf'))
            scores = scores + mask_proj.unsqueeze(1)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values: (B, n_heads, L, k) x (B, n_heads, k, d_k) -> (B, n_heads, L, d_k)
        context = torch.matmul(attn_weights, V_proj)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_projection(context)


class LinformerLayer(nn.Module):
    """Linformer encoder layer.
    
    Parameters
    ----------
    d_model : int
        Model dimension
    n_heads : int
        Number of attention heads
    seq_len : int
        Maximum sequence length
    d_ff : int
        Feed-forward dimension
    k : int
        Projected dimension
    dropout : float
        Dropout probability
    """
    
    def __init__(self, d_model: int, n_heads: int, seq_len: int, d_ff: int, k: int = 256, dropout: float = 0.1):
        super(LinformerLayer, self).__init__()
        
        self.attention = LinearAttention(d_model, n_heads, seq_len, k, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
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
        attn_out = self.attention(x, mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class Model(nn.Module):
    """Linformer: Self-Attention with Linear Complexity.
    
    Linformer reduces the complexity of self-attention from O(n^2) to O(n)
    by projecting keys and values to a lower-dimensional space.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - d_model : int, model dimension (default: 256)
        - n_heads : int, number of attention heads (default: 8)
        - num_layers : int, number of transformer layers (default: 6)
        - d_ff : int, feed-forward dimension (default: 512)
        - seq_len : int, maximum sequence length (default: 512)
        - k : int, projected dimension (default: 256)
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
    Wang et al. "Linformer: Self-Attention with Linear Complexity" arXiv 2020.
    Vaswani et al. "Attention Is All You Need" NeurIPS 2017.
    Katharopoulos et al. "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" ICML 2020.
    Adapted for time-series industrial signals with linear complexity attention for efficient long sequence processing.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.d_model = getattr(args, 'd_model', 256)
        self.n_heads = getattr(args, 'n_heads', 8)
        self.num_layers = getattr(args, 'num_layers', 6)
        self.d_ff = getattr(args, 'd_ff', 512)
        self.seq_len = getattr(args, 'seq_len', 512)
        self.k = getattr(args, 'k', 256)
        self.dropout = getattr(args, 'dropout', 0.1)
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        
        # Input embedding
        self.input_embedding = nn.Linear(self.input_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.seq_len, self.d_model) * 0.02)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            LinformerLayer(self.d_model, self.n_heads, self.seq_len, self.d_ff, self.k, self.dropout)
            for _ in range(self.num_layers)
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
        if seq_len <= self.seq_len:
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
    # Test Linformer
    import torch
    from argparse import Namespace
    
    def test_linformer():
        """Test Linformer with different configurations."""
        print("Testing Linformer...")
        
        # Test regression configuration
        args_reg = Namespace(
            input_dim=3,
            d_model=64,
            n_heads=4,
            num_layers=3,
            d_ff=128,
            seq_len=128,
            k=64,
            dropout=0.1,
            output_dim=3
        )
        
        model_reg = Model(args_reg)
        print(f"Regression model parameters: {sum(p.numel() for p in model_reg.parameters()):,}")
        
        # Test data
        batch_size = 4
        seq_len = 96
        x = torch.randn(batch_size, seq_len, args_reg.input_dim)
        
        # Forward pass
        with torch.no_grad():
            output_reg = model_reg(x)
        
        print(f"Regression - Input: {x.shape}, Output: {output_reg.shape}")
        assert output_reg.shape == (batch_size, seq_len, args_reg.output_dim)
        
        # Test classification configuration
        args_cls = Namespace(
            input_dim=3,
            d_model=64,
            n_heads=4,
            num_layers=3,
            d_ff=128,
            seq_len=128,
            k=64,
            dropout=0.1,
            num_classes=5
        )
        
        model_cls = Model(args_cls)
        print(f"Classification model parameters: {sum(p.numel() for p in model_cls.parameters()):,}")
        
        with torch.no_grad():
            output_cls = model_cls(x)
        
        print(f"Classification - Input: {x.shape}, Output: {output_cls.shape}")
        assert output_cls.shape == (batch_size, args_cls.num_classes)
        
        print("✅ Linformer tests passed!")
        return True
    
    test_linformer()
