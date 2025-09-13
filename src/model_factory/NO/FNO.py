import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    """
    1D傅里叶谱卷积层。
    通过对输入信号进行FFT，在频域中应用线性变换，然后通过IFFT变换回时域。
    """
    def __init__(self, in_channels, out_channels, modes):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param modes: 要保留的傅里叶模式数。只使用低频模式。
        """
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # 缩放因子，用于保持梯度幅度稳定
        self.scale = (1 / (in_channels * out_channels))
        # 学习权重，形状为 (in_channels, out_channels, modes)，使用复数
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral convolution.

        Args:
            x: 输入张量 ``(B, C, L)``。

        Returns:
            经过谱卷积的张量 ``(B, out_channels, L)``。
        """
        # x - b, c, l
        B, C, L = x.shape
        out_ft = torch.zeros(B, self.out_channels, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        x_ft = torch.fft.rfft(x)
        # Fix for RuntimeError: Slice weights to match input channels C
        # This handles cases where the input feature dimension does not match
        # the dimension the model was initialized with.
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :self.modes], self.weights[:C, :, :])

        x = torch.fft.irfft(out_ft, n=L)
        return x


class Model(nn.Module):
    """1D Fourier Neural Operator (FNO) for time-series analysis.

    A neural operator that learns mappings between function spaces using
    Fourier transforms. Particularly effective for PDEs and time-series modeling.

    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - modes : int, number of Fourier modes to keep (default: 16)
        - width : int, hidden layer width (default: 64)
        - n_layers : int, number of FNO layers (default: 4)
        - output_dim : int, output dimension (default: input_dim)
    metadata : Any, optional
        Dataset metadata

    Input Shape
    -----------
    x : torch.Tensor
        Input tensor of shape (batch_size, seq_len, input_dim)

    Output Shape
    ------------
    torch.Tensor
        Output tensor of shape (batch_size, seq_len, output_dim)

    References
    ----------
    Li et al. "Fourier Neural Operator for Parametric Partial Differential Equations" ICLR 2021.
    Chen and Chen "Universal Approximation to Nonlinear Operators by Neural Networks with Arbitrary Activation Functions and Its Application to Dynamical Systems" IEEE Transactions on Neural Networks 1995.
    Lu et al. "Learning Nonlinear Operators via DeepONet Based on the Universal Approximation Theorem of Operators" Nature Machine Intelligence 2021.
    Adapted for time-series industrial signals with spectral convolutions for operator learning.
    """

    def __init__(self, args, metadata=None):
        """Initialize FNO model.

        Parameters
        ----------
        args : Namespace
            Configuration containing model parameters
        metadata : Any, optional
            Dataset metadata (unused)
        """
        super(Model, self).__init__()

        # Extract parameters from args
        self.input_dim = args.input_dim
        self.modes = getattr(args, 'modes', 16)
        self.width = getattr(args, 'width', 64)
        self.n_layers = getattr(args, 'n_layers', 4)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)

        # Input projection: lift input to hidden dimension
        self.fc0 = nn.Linear(self.input_dim, self.width)

        # FNO spectral convolution layers and skip connections
        self.spectral_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.spectral_layers.append(SpectralConv1d(self.width, self.width, self.modes))
            self.conv_layers.append(nn.Conv1d(self.width, self.width, 1))

        # Output projection: map back to output dimension
        self.fc1 = nn.Linear(self.width, self.output_dim)

    def forward(self, x: torch.Tensor, data_id=None, task_id=None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, C)
        data_id : Any, optional
            Data identifier (unused)
        task_id : Any, optional
            Task identifier (unused)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, L, output_dim)
        """
        # Input tensor shape: (B, L, C)

        # Lift to hidden dimension
        x = self.fc0(x)  # (B, L, C) -> (B, L, width)

        # FNO requires input shape (B, C, L), so permute dimensions
        x = x.permute(0, 2, 1)  # (B, L, width) -> (B, width, L)

        # Apply FNO layers iteratively
        for i in range(self.n_layers):
            x1 = self.spectral_layers[i](x)  # Spectral convolution
            x2 = self.conv_layers[i](x)     # Skip connection
            x = x1 + x2  # Residual connection
            x = F.gelu(x)  # Activation function

        # Permute back to (B, L, C)
        x = x.permute(0, 2, 1)  # (B, width, L) -> (B, L, width)

        # Map to output dimension
        x = self.fc1(x)  # (B, L, width) -> (B, L, output_dim)
        return x

if __name__ == '__main__':
    # Test FNO model
    from argparse import Namespace

    def test_fno():
        """Test FNO model with different configurations."""
        print("Testing FNO model...")

        # Test configuration
        args = Namespace(
            input_dim=3,
            modes=16,
            width=64,
            n_layers=4,
            output_dim=3
        )

        # Create model
        model = Model(args)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test data
        B, L = 8, 512
        input_tensor = torch.randn(B, L, args.input_dim)
        print(f"Input tensor shape: {input_tensor.shape}")

        # Forward pass
        with torch.no_grad():
            output_tensor = model(input_tensor)

        print(f"Output tensor shape: {output_tensor.shape}")

        # Check input/output shape consistency
        expected_shape = (B, L, args.output_dim)
        assert output_tensor.shape == expected_shape, f"Expected {expected_shape}, got {output_tensor.shape}"
        print("✅ FNO model test passed!")

        return True

    test_fno()
