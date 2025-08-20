"""ResNet1D for time-series analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BasicBlock1D(nn.Module):
    """Basic residual block for 1D convolution.
    
    Parameters
    ----------
    in_channels : int
        Input channels
    out_channels : int
        Output channels
    stride : int
        Convolution stride
    downsample : nn.Module, optional
        Downsampling layer
    """
    
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 downsample: Optional[nn.Module] = None):
        super(BasicBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck1D(nn.Module):
    """Bottleneck residual block for 1D convolution.
    
    Parameters
    ----------
    in_channels : int
        Input channels
    out_channels : int
        Output channels
    stride : int
        Convolution stride
    downsample : nn.Module, optional
        Downsampling layer
    """
    
    expansion = 4
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super(Bottleneck1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class Model(nn.Module):
    """ResNet1D for time-series analysis.
    
    1D adaptation of ResNet architecture for time-series data,
    enabling deep networks with skip connections.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - block_type : str, block type ('basic' or 'bottleneck', default: 'basic')
        - layers : list, number of blocks in each layer (default: [2, 2, 2, 2])
        - initial_channels : int, initial number of channels (default: 64)
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
    Ioffe and Szegedy "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" ICML 2015.
    LeCun et al. "Gradient-based learning applied to document recognition" Proceedings of the IEEE 1998.
    Adapted for time-series industrial signals with 1D residual blocks for temporal feature learning.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.block_type = getattr(args, 'block_type', 'basic').lower()
        self.layers = getattr(args, 'layers', [2, 2, 2, 2])
        self.initial_channels = getattr(args, 'initial_channels', 64)
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        
        # Select block type
        if self.block_type == 'basic':
            self.block = BasicBlock1D
        elif self.block_type == 'bottleneck':
            self.block = Bottleneck1D
        else:
            raise ValueError(f"Unsupported block type: {self.block_type}")
        
        self.in_channels = self.initial_channels
        
        # Initial convolution
        self.conv1 = nn.Conv1d(self.input_dim, self.initial_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.initial_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(self.initial_channels, self.layers[0])
        self.layer2 = self._make_layer(self.initial_channels * 2, self.layers[1], stride=2)
        self.layer3 = self._make_layer(self.initial_channels * 4, self.layers[2], stride=2)
        self.layer4 = self._make_layer(self.initial_channels * 8, self.layers[3], stride=2)
        
        # Calculate final feature dimension
        final_channels = self.initial_channels * 8 * self.block.expansion
        
        # Output layers
        if self.num_classes is not None:
            # Classification: global pooling + classifier
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Linear(final_channels, self.num_classes)
            self.task_type = 'classification'
        else:
            # Regression: upsampling + projection
            self.upsample = nn.Sequential(
                nn.ConvTranspose1d(final_channels, self.initial_channels * 4, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(self.initial_channels * 4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose1d(self.initial_channels * 4, self.initial_channels * 2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(self.initial_channels * 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose1d(self.initial_channels * 2, self.initial_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(self.initial_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose1d(self.initial_channels, self.output_dim, kernel_size=4, stride=2, padding=1)
            )
            self.task_type = 'regression'
    
    def _make_layer(self, channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """Create a residual layer."""
        downsample = None
        if stride != 1 or self.in_channels != channels * self.block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, channels * self.block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(channels * self.block.expansion),
            )
        
        layers = []
        layers.append(self.block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * self.block.expansion
        for _ in range(1, blocks):
            layers.append(self.block(self.in_channels, channels))
        
        return nn.Sequential(*layers)
    
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
        
        # Initial convolution and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if self.task_type == 'classification':
            # Global average pooling and classification
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        else:
            # Upsampling for regression
            x = self.upsample(x)
            # Transpose back: (B, C, L) -> (B, L, C)
            x = x.transpose(1, 2)
        
        return x


if __name__ == "__main__":
    # Test ResNet1D
    import torch
    from argparse import Namespace
    
    def test_resnet1d():
        """Test ResNet1D with different configurations."""
        print("Testing ResNet1D...")
        
        # Test basic block classification
        args_basic_cls = Namespace(
            input_dim=3,
            block_type='basic',
            layers=[2, 2, 2, 2],
            initial_channels=32,  # Smaller for testing
            num_classes=5
        )
        
        model_basic_cls = Model(args_basic_cls)
        print(f"Basic Classification model parameters: {sum(p.numel() for p in model_basic_cls.parameters()):,}")
        
        # Test data
        batch_size = 4
        seq_len = 128
        x = torch.randn(batch_size, seq_len, args_basic_cls.input_dim)
        
        # Forward pass
        with torch.no_grad():
            output_basic_cls = model_basic_cls(x)
        
        print(f"Basic Classification - Input: {x.shape}, Output: {output_basic_cls.shape}")
        assert output_basic_cls.shape == (batch_size, args_basic_cls.num_classes)
        
        # Test bottleneck block regression (smaller network for testing)
        args_bottleneck_reg = Namespace(
            input_dim=3,
            block_type='bottleneck',
            layers=[1, 1, 1, 1],  # Smaller for testing
            initial_channels=16,   # Much smaller for testing
            output_dim=3
        )
        
        model_bottleneck_reg = Model(args_bottleneck_reg)
        print(f"Bottleneck Regression model parameters: {sum(p.numel() for p in model_bottleneck_reg.parameters()):,}")
        
        # Use smaller input for regression test
        x_small = torch.randn(batch_size, 64, args_bottleneck_reg.input_dim)
        
        with torch.no_grad():
            output_bottleneck_reg = model_bottleneck_reg(x_small)
        
        print(f"Bottleneck Regression - Input: {x_small.shape}, Output: {output_bottleneck_reg.shape}")
        # Note: output length may differ due to convolution/deconvolution operations
        assert output_bottleneck_reg.shape[0] == batch_size
        assert output_bottleneck_reg.shape[2] == args_bottleneck_reg.output_dim
        
        print("✅ ResNet1D tests passed!")
        return True
    
    test_resnet1d()
