"""Temporal Convolutional Network (TCN) with dilated convolutions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class TemporalBlock(nn.Module):
    """Temporal block with dilated convolutions.
    
    Parameters
    ----------
    n_inputs : int
        Number of input channels
    n_outputs : int
        Number of output channels
    kernel_size : int
        Convolution kernel size
    stride : int
        Convolution stride
    dilation : int
        Dilation factor
    padding : int
        Padding size
    dropout : float
        Dropout probability
    """
    
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int,
                 dilation: int, padding: int, dropout: float = 0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove padding from the end of sequence."""
    
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x[:, :, :-self.chomp_size].contiguous()


class Model(nn.Module):
    """Temporal Convolutional Network (TCN) for time-series analysis.
    
    TCN uses dilated convolutions to capture long-range dependencies
    while maintaining computational efficiency.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - num_channels : list, number of channels in each layer (default: [25, 25, 25, 25])
        - kernel_size : int, convolution kernel size (default: 2)
        - dropout : float, dropout probability (default: 0.2)
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
    Bai et al. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" arXiv 2018.
    Yu and Koltun "Multi-Scale Context Aggregation by Dilated Convolutions" ICLR 2016.
    He et al. "Deep Residual Learning for Image Recognition" CVPR 2016.
    Adapted for time-series industrial signals with dilated causal convolutions for long-range temporal dependencies.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.num_channels = getattr(args, 'num_channels', [25, 25, 25, 25])
        self.kernel_size = getattr(args, 'kernel_size', 2)
        self.dropout = getattr(args, 'dropout', 0.2)
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        
        layers = []
        num_levels = len(self.num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = self.input_dim if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            
            layers += [TemporalBlock(in_channels, out_channels, self.kernel_size, stride=1,
                                   dilation=dilation_size,
                                   padding=(self.kernel_size-1) * dilation_size,
                                   dropout=self.dropout)]
        
        self.network = nn.Sequential(*layers)
        
        # Output layers
        final_channels = self.num_channels[-1]
        
        if self.num_classes is not None:
            # Classification: global pooling + classifier
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(final_channels, self.num_classes)
            )
            self.task_type = 'classification'
        else:
            # Regression: sequence-to-sequence
            self.output_projection = nn.Conv1d(final_channels, self.output_dim, kernel_size=1)
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
        # Transpose for convolution: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)
        
        # Apply TCN
        x = self.network(x)
        
        if self.task_type == 'classification':
            # Classification
            x = self.classifier(x)
        else:
            # Regression: project to output dimension
            x = self.output_projection(x)
            # Transpose back: (B, C, L) -> (B, L, C)
            x = x.transpose(1, 2)
        
        return x


if __name__ == "__main__":
    # Test TCN
    import torch
    from argparse import Namespace
    
    def test_tcn():
        """Test TCN with different configurations."""
        print("Testing TCN...")
        
        # Test regression configuration
        args_reg = Namespace(
            input_dim=4,
            num_channels=[16, 32, 64],
            kernel_size=3,
            dropout=0.2,
            output_dim=4
        )
        
        model_reg = Model(args_reg)
        print(f"Regression model parameters: {sum(p.numel() for p in model_reg.parameters()):,}")
        
        # Test data
        batch_size = 4
        seq_len = 100
        x = torch.randn(batch_size, seq_len, args_reg.input_dim)
        
        # Forward pass
        with torch.no_grad():
            output_reg = model_reg(x)
        
        print(f"Regression - Input: {x.shape}, Output: {output_reg.shape}")
        assert output_reg.shape == (batch_size, seq_len, args_reg.output_dim)
        
        # Test classification configuration
        args_cls = Namespace(
            input_dim=4,
            num_channels=[16, 32, 64],
            kernel_size=3,
            dropout=0.2,
            num_classes=6
        )
        
        model_cls = Model(args_cls)
        print(f"Classification model parameters: {sum(p.numel() for p in model_cls.parameters()):,}")
        
        with torch.no_grad():
            output_cls = model_cls(x)
        
        print(f"Classification - Input: {x.shape}, Output: {output_cls.shape}")
        assert output_cls.shape == (batch_size, args_cls.num_classes)
        
        # Test with different kernel size
        args_large_kernel = Namespace(
            input_dim=4,
            num_channels=[8, 16, 32],
            kernel_size=5,
            dropout=0.1,
            num_classes=3
        )
        
        model_large_kernel = Model(args_large_kernel)
        print(f"Large kernel model parameters: {sum(p.numel() for p in model_large_kernel.parameters()):,}")
        
        with torch.no_grad():
            output_large_kernel = model_large_kernel(x)
        
        print(f"Large kernel - Input: {x.shape}, Output: {output_large_kernel.shape}")
        assert output_large_kernel.shape == (batch_size, args_large_kernel.num_classes)
        
        print("✅ TCN tests passed!")
        return True
    
    test_tcn()
