"""
ResNet1D for Time-Series Analysis - Scientific Implementation

This module implements 1D ResNet architecture for time-series classification and regression
following scientific coding standards with proper mathematical notation, comprehensive
documentation, and type safety.

Mathematical Foundation
-----------------------
ResNet introduces skip connections to enable training of very deep networks:

    y = F(x, {W_i}) + x                                    (1)

where F(x, {W_i}) represents the residual mapping to be learned, and the identity
shortcut connection y = x + F(x) is performed by element-wise addition.

For 1D time-series data, we adapt the 2D convolutions to 1D:

    Conv1D: y[n] = Σ(k=0 to K-1) w[k] * x[n-k] + b        (2)

where K is the kernel size, w[k] are the learnable weights, and b is the bias term.

References
----------
He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition.
In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training
by reducing internal covariate shift. In International conference on machine learning (pp. 448-456).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Type aliases for clarity
TensorShape = Tuple[int, ...]
ActivationFunction = nn.Module


class BasicBlock1D(nn.Module):
    """
    Basic residual block for 1D convolution.

    Implements the basic residual block as described in He et al. (2016):

        y = F(x) + x                                        (3)

    where F(x) = Conv1D(ReLU(BN(Conv1D(x))))

    Parameters
    ----------
    in_channels : int
        Number of input channels (C_in)
    out_channels : int
        Number of output channels (C_out)
    stride : int, default=1
        Convolution stride for downsampling
    downsample : Optional[nn.Module], default=None
        Downsampling layer for dimension matching

    Input Shape
    -----------
    x : Tensor[B, C_in, L]
        Input tensor where B=batch_size, C_in=input_channels, L=sequence_length

    Output Shape
    ------------
    Tensor[B, C_out, L//stride]
        Output tensor with potentially reduced sequence length due to stride

    Mathematical Operations
    -----------------------
    1. First convolution: x₁ = Conv1D(x, kernel_size=3, stride=stride)
    2. Batch normalization: x₂ = BatchNorm1D(x₁)
    3. ReLU activation: x₃ = ReLU(x₂)
    4. Second convolution: x₄ = Conv1D(x₃, kernel_size=3, stride=1)
    5. Batch normalization: x₅ = BatchNorm1D(x₄)
    6. Skip connection: y = ReLU(x₅ + downsample(x) if downsample else x₅ + x)
    """

    expansion: int = 1  # Channel expansion factor for this block type

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ) -> None:
        super().__init__()

        # Validate input parameters
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channel dimensions must be positive")
        if stride <= 0:
            raise ValueError("Stride must be positive")

        # Store configuration
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.downsample = downsample

        # First convolution block: Conv1D + BatchNorm1D + ReLU
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False  # Bias is redundant when followed by BatchNorm
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        # Second convolution block: Conv1D + BatchNorm1D
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the basic residual block.

        Parameters
        ----------
        x : Tensor[B, C_in, L]
            Input tensor

        Returns
        -------
        Tensor[B, C_out, L//stride]
            Output tensor after residual connection

        Mathematical Implementation
        ---------------------------
        Following Equation (3), we compute:
        1. Residual path: F(x) = conv2(relu(bn1(conv1(x))))
        2. Identity path: identity = downsample(x) if downsample else x
        3. Output: y = relu(F(x) + identity)
        """
        # Store identity for skip connection
        identity = x

        # First convolution block: x → conv1 → bn1 → relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second convolution block: x → conv2 → bn2
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply downsampling to identity if needed for dimension matching
        if self.downsample is not None:
            identity = self.downsample(x)

        # Skip connection: F(x) + x
        out += identity

        # Final activation
        out = self.relu(out)

        return out

    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        return (f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'stride={self.stride}, expansion={self.expansion}')


class Bottleneck1D(nn.Module):
    """
    Bottleneck residual block for 1D convolution.

    Implements the bottleneck design from ResNet-50/101/152 adapted for 1D signals:

        y = F(x) + x                                        (4)

    where F(x) = Conv1D(1×1) → Conv1D(3×1) → Conv1D(1×1) with batch normalization
    and ReLU activations between each convolution.

    The bottleneck design reduces computational complexity by using 1×1 convolutions
    to reduce and then restore dimensions around the 3×1 convolution.

    Parameters
    ----------
    in_channels : int
        Number of input channels (C_in)
    out_channels : int
        Number of intermediate channels (C_mid)
    stride : int, default=1
        Convolution stride for the 3×1 convolution
    downsample : Optional[nn.Module], default=None
        Downsampling layer for dimension matching

    Input Shape
    -----------
    x : Tensor[B, C_in, L]
        Input tensor

    Output Shape
    ------------
    Tensor[B, C_out, L//stride]
        Output tensor where C_out = out_channels * expansion

    Mathematical Operations
    -----------------------
    1. Dimension reduction: x₁ = Conv1D(x, kernel_size=1)
    2. Spatial convolution: x₂ = Conv1D(x₁, kernel_size=3, stride=stride)
    3. Dimension restoration: x₃ = Conv1D(x₂, kernel_size=1)
    4. Skip connection: y = ReLU(x₃ + downsample(x))
    """

    expansion: int = 4  # Channel expansion factor (C_out = out_channels * 4)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ) -> None:
        super().__init__()

        # Validate input parameters
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channel dimensions must be positive")
        if stride <= 0:
            raise ValueError("Stride must be positive")

        # Store configuration
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.downsample = downsample

        # 1×1 convolution for dimension reduction
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        # 3×1 convolution for spatial feature extraction
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 1×1 convolution for dimension restoration
        self.conv3 = nn.Conv1d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)

        # Activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the bottleneck residual block.

        Parameters
        ----------
        x : Tensor[B, C_in, L]
            Input tensor

        Returns
        -------
        Tensor[B, C_out, L//stride]
            Output tensor after residual connection

        Mathematical Implementation
        ---------------------------
        Following Equation (4), we compute the bottleneck transformation:
        1. Reduce: x₁ = relu(bn1(conv1(x)))     # 1×1 conv to reduce channels
        2. Transform: x₂ = relu(bn2(conv2(x₁))) # 3×1 conv for feature extraction
        3. Expand: x₃ = bn3(conv3(x₂))          # 1×1 conv to expand channels
        4. Skip: y = relu(x₃ + downsample(x))   # Add identity connection
        """
        # Store identity for skip connection
        identity = x

        # 1×1 convolution for dimension reduction
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3×1 convolution for spatial feature extraction
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1×1 convolution for dimension restoration
        out = self.conv3(out)
        out = self.bn3(out)

        # Apply downsampling to identity if needed for dimension matching
        if self.downsample is not None:
            identity = self.downsample(x)

        # Skip connection: F(x) + x
        out += identity

        # Final activation
        out = self.relu(out)

        return out

    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        return (f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'stride={self.stride}, expansion={self.expansion}')


# Type alias for block types
BlockType = Union[BasicBlock1D, Bottleneck1D]


class ResNet1D(nn.Module):
    """
    1D ResNet for time-series analysis with scientific documentation.

    This implementation adapts the ResNet architecture for 1D time-series data,
    enabling deep networks for industrial signal processing tasks such as fault
    diagnosis and remaining useful life prediction.

    Architecture Overview
    ---------------------
    The network follows the standard ResNet structure:

    1. Initial convolution: Conv1D(7×1) + BatchNorm + ReLU + MaxPool
    2. Residual layers: 4 groups of residual blocks with increasing channels
    3. Global pooling: AdaptiveAvgPool1D for classification or upsampling for regression
    4. Output layer: Linear classifier or convolutional decoder

    Mathematical Formulation
    -------------------------
    For input signal x ∈ ℝ^(B×C×L), the network computes:

    1. Feature extraction: h = ResidualLayers(InitialConv(x))     (5)
    2. Classification: ŷ = Classifier(GlobalPool(h))              (6)
    3. Regression: ŷ = Decoder(h)                                 (7)

    where B=batch_size, C=input_channels, L=sequence_length.

    Parameters
    ----------
    config : ModelConfig
        Configuration object containing model hyperparameters:
        - input_dim : int, input feature dimension
        - num_classes : int, number of output classes (for classification)
        - output_dim : int, output feature dimension (for regression)
        - block_type : str, residual block type ('basic' or 'bottleneck')
        - layers : List[int], number of blocks in each layer
        - initial_channels : int, number of channels after initial convolution
        - dropout : float, dropout probability
    metadata : Optional[Any]
        Dataset metadata for automatic parameter inference

    Input Shape
    -----------
    x : Tensor[B, L, C_in]
        Input time-series tensor (will be transposed to [B, C_in, L] internally)

    Output Shape
    ------------
    Classification: Tensor[B, num_classes]
        Class logits for each sample
    Regression: Tensor[B, L_out, C_out]
        Reconstructed or predicted time-series

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> config = SimpleNamespace(
    ...     input_dim=3,
    ...     num_classes=10,
    ...     block_type='basic',
    ...     layers=[2, 2, 2, 2],
    ...     initial_channels=64,
    ...     dropout=0.1
    ... )
    >>> model = ResNet1D(config)
    >>> x = torch.randn(32, 1024, 3)  # Batch of 32 signals, length 1024, 3 channels
    >>> y = model(x)  # Output: [32, 10]

    References
    ----------
    He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition.
    Kiranyaz, S., Avci, O., Abdeljaber, O., Ince, T., Gabbouj, M., & Inman, D. J. (2021).
    1D convolutional neural networks and applications: A survey.
    """

    def __init__(self, config: Any, metadata: Optional[Any] = None) -> None:
        super().__init__()

        # Extract and validate configuration parameters
        self.input_dim = self._validate_positive_int(config.input_dim, "input_dim")
        self.block_type_str = getattr(config, 'block_type', 'basic').lower()
        self.layers = getattr(config, 'layers', [2, 2, 2, 2])
        self.initial_channels = getattr(config, 'initial_channels', 64)
        self.dropout_prob = getattr(config, 'dropout', 0.1)

        # Task-specific parameters
        self.num_classes = getattr(config, 'num_classes', None)
        self.output_dim = getattr(config, 'output_dim', self.input_dim)

        # Validate configuration
        self._validate_configuration()

        # Select block type
        self.block_class = self._get_block_class(self.block_type_str)

        # Initialize channel tracking
        self.in_channels = self.initial_channels

        # Build network layers
        self._build_initial_layers()
        self._build_residual_layers()
        self._build_output_layers()

        # Initialize weights
        self._initialize_weights()

        # Store metadata for debugging
        self.metadata = metadata

    def _validate_positive_int(self, value: int, name: str) -> int:
        """Validate that a parameter is a positive integer."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}")
        return value

    def _validate_configuration(self) -> None:
        """Validate the complete configuration for consistency."""
        if not isinstance(self.layers, (list, tuple)) or len(self.layers) != 4:
            raise ValueError("layers must be a list/tuple of 4 integers")

        if not all(isinstance(x, int) and x > 0 for x in self.layers):
            raise ValueError("All layer counts must be positive integers")

        if not 0 <= self.dropout_prob <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout_prob}")

        if self.num_classes is not None and self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")

        if self.output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {self.output_dim}")

    def _get_block_class(self, block_type: str) -> Type[BlockType]:
        """Get the appropriate block class based on configuration."""
        block_map = {
            'basic': BasicBlock1D,
            'bottleneck': Bottleneck1D
        }

        if block_type not in block_map:
            available = list(block_map.keys())
            raise ValueError(f"Unsupported block_type '{block_type}'. Available: {available}")

        return block_map[block_type]

    def _build_initial_layers(self) -> None:
        """Build the initial convolution and pooling layers."""
        # Initial convolution: large kernel for capturing temporal patterns
        self.conv1 = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.initial_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(self.initial_channels)
        self.relu = nn.ReLU(inplace=True)

        # Max pooling for downsampling
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def _build_residual_layers(self) -> None:
        """Build the residual layers with increasing channel dimensions."""
        # Layer 1: channels = initial_channels
        self.layer1 = self._make_layer(
            channels=self.initial_channels,
            blocks=self.layers[0],
            stride=1
        )

        # Layer 2: channels = initial_channels * 2
        self.layer2 = self._make_layer(
            channels=self.initial_channels * 2,
            blocks=self.layers[1],
            stride=2
        )

        # Layer 3: channels = initial_channels * 4
        self.layer3 = self._make_layer(
            channels=self.initial_channels * 4,
            blocks=self.layers[2],
            stride=2
        )

        # Layer 4: channels = initial_channels * 8
        self.layer4 = self._make_layer(
            channels=self.initial_channels * 8,
            blocks=self.layers[3],
            stride=2
        )