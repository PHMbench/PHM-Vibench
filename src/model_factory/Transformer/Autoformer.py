"""Autoformer: Decomposition Transformers with Auto-Correlation for Long-term Series Forecasting."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class AutoCorrelation(nn.Module):
    """Auto-Correlation mechanism from Autoformer.
    
    Parameters
    ----------
    d_model : int
        Model dimension
    n_heads : int
        Number of attention heads
    factor : int
        Top-k factor for auto-correlation
    dropout : float
        Dropout probability
    """
    
    def __init__(self, d_model: int, n_heads: int, factor: int = 1, dropout: float = 0.1):
        super(AutoCorrelation, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor
        
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def time_delay_agg_training(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """Time delay aggregation for training."""
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        
        # Find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        
        # Update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        
        # Aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        
        return delays_agg
    
    def time_delay_agg_inference(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """Time delay aggregation for inference."""
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        
        # Index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        
        # Find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        
        # Update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        
        # Aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        
        return delays_agg
    
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        queries : torch.Tensor
            Query tensor of shape (B, L, d_model)
        keys : torch.Tensor
            Key tensor of shape (B, L, d_model)
        values : torch.Tensor
            Value tensor of shape (B, L, d_model)
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, L, d_model)
        """
        B, L, _ = queries.shape
        
        # Project to Q, K, V
        Q = self.query_projection(queries).view(B, L, self.n_heads, self.d_k)
        K = self.key_projection(keys).view(B, L, self.n_heads, self.d_k)
        V = self.value_projection(values).view(B, L, self.n_heads, self.d_k)
        
        # Transpose for correlation computation
        Q = Q.transpose(1, 2)  # (B, n_heads, L, d_k)
        K = K.transpose(1, 2)  # (B, n_heads, L, d_k)
        V = V.transpose(1, 2)  # (B, n_heads, L, d_k)
        
        # Auto-correlation computation using FFT
        q_fft = torch.fft.rfft(Q.permute(0, 1, 3, 2).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(K.permute(0, 1, 3, 2).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        
        # Time delay aggregation
        if self.training:
            V_agg = self.time_delay_agg_training(V.permute(0, 1, 3, 2).contiguous(), corr)
        else:
            V_agg = self.time_delay_agg_inference(V.permute(0, 1, 3, 2).contiguous(), corr)
        
        V_agg = V_agg.permute(0, 1, 3, 2).contiguous()
        
        # Reshape and project
        V_agg = V_agg.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_projection(V_agg)


class SeriesDecomp(nn.Module):
    """Series decomposition block."""
    
    def __init__(self, kernel_size: int = 25):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        self.kernel_size = kernel_size
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose series into trend and seasonal components.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, C)
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Seasonal and trend components
        """
        # Padding
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        
        # Moving average (trend)
        x_padded = x_padded.permute(0, 2, 1)  # (B, C, L)
        trend = self.moving_avg(x_padded)
        trend = trend.permute(0, 2, 1)  # (B, L, C)
        
        # Seasonal component
        seasonal = x - trend
        
        return seasonal, trend


class AutoformerLayer(nn.Module):
    """Autoformer encoder layer.
    
    Parameters
    ----------
    d_model : int
        Model dimension
    n_heads : int
        Number of attention heads
    d_ff : int
        Feed-forward dimension
    factor : int
        Auto-correlation factor
    dropout : float
        Dropout probability
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, factor: int = 1, dropout: float = 0.1):
        super(AutoformerLayer, self).__init__()
        
        self.attention = AutoCorrelation(d_model, n_heads, factor, dropout)
        self.decomp1 = SeriesDecomp()
        self.decomp2 = SeriesDecomp()
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, d_model)
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, L, d_model)
        """
        # Auto-correlation with decomposition
        new_x, _ = self.decomp1(self.attention(x, x, x) + x)
        
        # Feed-forward with decomposition
        y, _ = self.decomp2(self.feed_forward(new_x) + new_x)
        
        return y


class Model(nn.Module):
    """Autoformer model for long-term series forecasting.
    
    Autoformer introduces auto-correlation mechanism and series decomposition
    to capture long-range dependencies and seasonal patterns.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - d_model : int, model dimension (default: 256)
        - n_heads : int, number of attention heads (default: 8)
        - e_layers : int, number of encoder layers (default: 6)
        - d_ff : int, feed-forward dimension (default: 512)
        - factor : int, auto-correlation factor (default: 1)
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
    Wu et al. "Autoformer: Decomposition Transformers with Auto-Correlation for Long-term Series Forecasting" NeurIPS 2021.
    Vaswani et al. "Attention Is All You Need" NeurIPS 2017.
    Cleveland et al. "STL: A Seasonal-Trend Decomposition Procedure Based on Loess" Journal of Official Statistics 1990.
    Adapted for time-series industrial signals with auto-correlation mechanism and series decomposition for seasonal pattern modeling.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.d_model = getattr(args, 'd_model', 256)
        self.n_heads = getattr(args, 'n_heads', 8)
        self.e_layers = getattr(args, 'e_layers', 6)
        self.d_ff = getattr(args, 'd_ff', 512)
        self.factor = getattr(args, 'factor', 1)
        self.dropout = getattr(args, 'dropout', 0.1)
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        
        # Input embedding
        self.input_embedding = nn.Linear(self.input_dim, self.d_model)
        
        # Series decomposition
        self.decomp = SeriesDecomp()
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            AutoformerLayer(self.d_model, self.n_heads, self.d_ff, self.factor, self.dropout)
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
        # Input embedding
        x = self.input_embedding(x)  # (B, L, d_model)
        
        # Initial decomposition
        seasonal, trend = self.decomp(x)
        
        # Apply encoder layers to seasonal component
        for layer in self.encoder_layers:
            seasonal = layer(seasonal)
        
        # Combine seasonal and trend
        x = seasonal + trend
        
        if self.task_type == 'classification':
            # Global average pooling and classification
            x = x.mean(dim=1)  # (B, d_model)
            x = self.classifier(x)  # (B, num_classes)
        else:
            # Sequence-to-sequence regression
            x = self.output_projection(x)  # (B, L, output_dim)
        
        return x


if __name__ == "__main__":
    # Test Autoformer
    import torch
    from argparse import Namespace
    
    def test_autoformer():
        """Test Autoformer with different configurations."""
        print("Testing Autoformer...")
        
        # Test regression configuration
        args_reg = Namespace(
            input_dim=3,
            d_model=64,
            n_heads=4,
            e_layers=2,  # Smaller for testing
            d_ff=128,
            factor=1,
            dropout=0.1,
            output_dim=3
        )
        
        model_reg = Model(args_reg)
        print(f"Regression model parameters: {sum(p.numel() for p in model_reg.parameters()):,}")
        
        # Test data
        batch_size = 4
        seq_len = 64  # Moderate length for testing
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
            e_layers=2,
            d_ff=128,
            factor=1,
            dropout=0.1,
            num_classes=5
        )
        
        model_cls = Model(args_cls)
        print(f"Classification model parameters: {sum(p.numel() for p in model_cls.parameters()):,}")
        
        with torch.no_grad():
            output_cls = model_cls(x)
        
        print(f"Classification - Input: {x.shape}, Output: {output_cls.shape}")
        assert output_cls.shape == (batch_size, args_cls.num_classes)
        
        print("✅ Autoformer tests passed!")
        return True
    
    test_autoformer()
