"""PatchTST: A Time Series Transformer with Patching and Channel-independence."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PatchEmbedding(nn.Module):
    """Patch embedding for time series.

    Parameters
    ----------
    patch_size : int
        Size of each patch
    stride : int
        Stride for patching
    input_dim : int
        Input feature dimension
    d_model : int
        Model dimension
    """

    def __init__(self, patch_size: int, stride: int, input_dim: int, d_model: int):
        super(PatchEmbedding, self).__init__()

        self.patch_size = patch_size
        self.stride = stride
        self.input_dim = input_dim
        self.d_model = d_model

        # Linear projection for each patch
        self.projection = nn.Linear(patch_size * input_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, C)

        Returns
        -------
        torch.Tensor
            Patch embeddings of shape (B, num_patches, d_model)
        """
        B, L, C = x.shape

        # Create patches using unfold
        patches = x.unfold(dimension=1, size=self.patch_size, step=self.stride)  # (B, num_patches, C, patch_size)
        patches = patches.reshape(B, -1, C * self.patch_size)  # (B, num_patches, C * patch_size)

        # Project patches to d_model
        embeddings = self.projection(patches)  # (B, num_patches, d_model)

        return embeddings


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, d_model)

        Returns
        -------
        torch.Tensor
            Input with positional encoding
        """
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class Model(nn.Module):
    """PatchTST: A Time Series Transformer with Patching and Channel-independence.

    PatchTST applies patching to time series and uses channel-independent
    processing to improve efficiency and performance.

    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - patch_size : int, size of each patch (default: 16)
        - stride : int, stride for patching (default: 8)
        - d_model : int, model dimension (default: 256)
        - n_heads : int, number of attention heads (default: 8)
        - num_layers : int, number of transformer layers (default: 6)
        - d_ff : int, feed-forward dimension (default: 512)
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
    Nie et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" ICLR 2023.
    Vaswani et al. "Attention Is All You Need" NeurIPS 2017.
    Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" ICLR 2021.
    Adapted for time-series industrial signals with patching and channel-independent processing for efficient forecasting.
    """

    def __init__(self, args, metadata=None):
        super(Model, self).__init__()

        # Extract parameters
        self.input_dim = args.input_dim
        self.patch_size = getattr(args, 'patch_size', 16)
        self.stride = getattr(args, 'stride', 8)
        self.d_model = getattr(args, 'd_model', 256)
        self.n_heads = getattr(args, 'n_heads', 8)
        self.num_layers = getattr(args, 'num_layers', 6)
        self.d_ff = getattr(args, 'd_ff', 512)
        self.dropout = getattr(args, 'dropout', 0.1)
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)

        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            patch_size=self.patch_size,
            stride=self.stride,
            input_dim=1,  # Channel-independent processing
            d_model=self.d_model
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Output layers
        if self.num_classes is not None:
            # Classification: global pooling + classifier
            self.classifier = nn.Linear(self.d_model, self.num_classes)
            self.task_type = 'classification'
        else:
            # Regression: patch reconstruction
            self.output_projection = nn.Linear(self.d_model, self.patch_size)
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
        B, L, C = x.shape

        # Channel-independent processing
        outputs = []

        for c in range(C):
            # Extract single channel
            x_c = x[:, :, c:c+1]  # (B, L, 1)

            # Patch embedding
            patches = self.patch_embedding(x_c)  # (B, num_patches, d_model)

            # Add positional encoding
            patches = self.pos_encoding(patches)

            # Apply transformer
            encoded = self.transformer(patches)  # (B, num_patches, d_model)

            if self.task_type == 'classification':
                # Global average pooling for classification
                pooled = encoded.mean(dim=1)  # (B, d_model)
                outputs.append(pooled)
            else:
                # Reconstruct patches for regression
                reconstructed = self.output_projection(encoded)  # (B, num_patches, patch_size)

                # Reshape to original sequence length
                num_patches = reconstructed.size(1)
                reconstructed = reconstructed.view(B, -1)  # (B, num_patches * patch_size)

                # Truncate or pad to match original length
                if reconstructed.size(1) > L:
                    reconstructed = reconstructed[:, :L]
                elif reconstructed.size(1) < L:
                    padding = L - reconstructed.size(1)
                    reconstructed = F.pad(reconstructed, (0, padding), mode='replicate')

                outputs.append(reconstructed.unsqueeze(-1))  # (B, L, 1)

        if self.task_type == 'classification':
            # Combine features from all channels
            combined = torch.stack(outputs, dim=1).mean(dim=1)  # (B, d_model)
            output = self.classifier(combined)  # (B, num_classes)
        else:
            # Concatenate all channels
            output = torch.cat(outputs, dim=-1)  # (B, L, input_dim)

            # Project to output dimension if different
            if self.output_dim != self.input_dim:
                output = F.linear(output, torch.randn(self.output_dim, self.input_dim))

        return output


if __name__ == "__main__":
    # Test PatchTST
    import torch
    from argparse import Namespace

    def test_patchtst():
        """Test PatchTST with different configurations."""
        print("Testing PatchTST...")

        # Test regression configuration
        args_reg = Namespace(
            input_dim=2,
            patch_size=8,
            stride=4,
            d_model=64,
            n_heads=4,
            num_layers=3,
            d_ff=128,
            dropout=0.1,
            output_dim=2
        )

        model_reg = Model(args_reg)
        print(f"Regression model parameters: {sum(p.numel() for p in model_reg.parameters()):,}")

        # Test data
        batch_size = 4
        seq_len = 64
        x = torch.randn(batch_size, seq_len, args_reg.input_dim)

        # Forward pass
        with torch.no_grad():
            output_reg = model_reg(x)

        print(f"Regression - Input: {x.shape}, Output: {output_reg.shape}")
        assert output_reg.shape == (batch_size, seq_len, args_reg.output_dim)

        # Test classification configuration
        args_cls = Namespace(
            input_dim=2,
            patch_size=8,
            stride=4,
            d_model=64,
            n_heads=4,
            num_layers=3,
            d_ff=128,
            dropout=0.1,
            num_classes=4
        )

        model_cls = Model(args_cls)
        print(f"Classification model parameters: {sum(p.numel() for p in model_cls.parameters()):,}")

        with torch.no_grad():
            output_cls = model_cls(x)

        print(f"Classification - Input: {x.shape}, Output: {output_cls.shape}")
        assert output_cls.shape == (batch_size, args_cls.num_classes)

        print("✅ PatchTST tests passed!")
        return True

    test_patchtst()

