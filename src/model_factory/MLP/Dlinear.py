import torch
import torch.nn as nn
from torch.nn import functional as F
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
    
class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class B_04_Dlinear(nn.Module):
    """DLinear backbone for time-series forecasting.

    Parameters
    ----------
    configs : Namespace
        Provides ``patch_size_L`` and ``patch_size_C``.
    individual : bool, optional
        If ``True`` use an independent linear head for each channel.
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(B_04_Dlinear, self).__init__()
        self.patch_size_L = configs.patch_size_L
        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(65)
        self.individual = individual
        self.channels = configs.patch_size_C

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.patch_size_L, self.patch_size_L))
                self.Linear_Trend.append(nn.Linear(self.patch_size_L, self.patch_size_L))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.patch_size_L) * torch.ones([self.patch_size_L, self.patch_size_L]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.patch_size_L) * torch.ones([self.patch_size_L, self.patch_size_L]))
        else:
            self.Linear_Seasonal = nn.Linear(self.patch_size_L, self.patch_size_L)
            self.Linear_Trend = nn.Linear(self.patch_size_L, self.patch_size_L)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.patch_size_L) * torch.ones([self.patch_size_L, self.patch_size_L]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.patch_size_L) * torch.ones([self.patch_size_L, self.patch_size_L]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, L, C)``.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as input.
        """
        # x: [B, L, C]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)  # [B, C, L]
            
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.patch_size_L],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.patch_size_L],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
            
        output = seasonal_output + trend_output
        
        output = F.silu(output)
        
        return output.permute(0, 2, 1)  # [B, L, C]
    


class Model(nn.Module):
    """DLinear model for time-series analysis.

    A decomposition-based linear model that separates trend and seasonal components
    for effective time-series modeling. Based on the DLinear paper.

    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - seq_len : int, sequence length (default: 512)
        - individual : bool, whether to use individual linear layers per channel (default: False)
        - kernel_size : int, moving average kernel size for decomposition (default: 25)
    metadata : Any, optional
        Dataset metadata (unused in this model)

    Input Shape
    -----------
    x : torch.Tensor
        Input tensor of shape (batch_size, seq_len, input_dim)

    Output Shape
    ------------
    torch.Tensor
        Output tensor of shape (batch_size, seq_len, input_dim)

    References
    ----------
    Zeng et al. "Are Transformers Effective for Time Series Forecasting?" AAAI 2023.
    Cleveland et al. "STL: A Seasonal-Trend Decomposition Procedure Based on Loess" Journal of Official Statistics 1990.
    Hyndman and Athanasopoulos "Forecasting: Principles and Practice" OTexts 2018.
    Implementation includes decomposition-based linear modeling for time-series forecasting with trend and seasonal components.
    """

    def __init__(self, args, metadata=None):
        super(Model, self).__init__()

        # Extract parameters from args
        self.input_dim = getattr(args, 'input_dim', 1)
        self.seq_len = getattr(args, 'seq_len', 512)
        self.individual = getattr(args, 'individual', False)
        kernel_size = getattr(args, 'kernel_size', 25)

        # Series decomposition block
        self.decomposition = series_decomp(kernel_size)

        if self.individual:
            # Individual linear layers for each channel
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.input_dim):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.seq_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.seq_len))

                # Initialize with identity-like weights
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.seq_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.seq_len, self.seq_len]))
        else:
            # Shared linear layers
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.seq_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.seq_len)

            # Initialize with identity-like weights
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.seq_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.seq_len, self.seq_len]))

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
            Output tensor of shape (B, L, C)
        """
        # Decompose into seasonal and trend components
        seasonal_init, trend_init = self.decomposition(x)

        # Transpose for linear operations: (B, L, C) -> (B, C, L)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            # Apply individual linear layers
            seasonal_output = torch.zeros_like(seasonal_init)
            trend_output = torch.zeros_like(trend_init)

            for i in range(self.input_dim):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            # Apply shared linear layers
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        # Combine seasonal and trend components
        output = seasonal_output + trend_output

        # Apply activation and transpose back: (B, C, L) -> (B, L, C)
        output = F.silu(output)
        return output.permute(0, 2, 1)


if __name__ == "__main__":
    # Test DLinear model
    import torch
    from argparse import Namespace

    def test_dlinear_model():
        """Test DLinear model with different configurations."""
        print("Testing DLinear Model...")

        # Test configuration
        args = Namespace(
            input_dim=2,
            seq_len=512,
            individual=False,
            kernel_size=25
        )

        # Create model
        model = Model(args)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test data
        batch_size = 8
        x = torch.randn(batch_size, args.seq_len, args.input_dim)

        # Forward pass
        with torch.no_grad():
            output = model(x)

        # Verify shapes
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"

        # Test individual mode
        args.individual = True
        model_individual = Model(args)
        with torch.no_grad():
            output_individual = model_individual(x)

        print(f"Individual mode output shape: {output_individual.shape}")
        assert output_individual.shape == x.shape

        print("✅ DLinear model tests passed!")
        return True

    test_dlinear_model()
