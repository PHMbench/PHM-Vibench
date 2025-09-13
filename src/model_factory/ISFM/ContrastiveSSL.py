"""Self-Supervised Contrastive Learning for Industrial Signal Foundation Model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
import math


class TimeSeriesAugmentation(nn.Module):
    """Time-series data augmentation for contrastive learning.
    
    Parameters
    ----------
    noise_std : float
        Standard deviation for Gaussian noise
    jitter_std : float
        Standard deviation for jittering
    scaling_range : tuple
        Range for scaling augmentation
    """
    
    def __init__(self, noise_std: float = 0.01, jitter_std: float = 0.03, 
                 scaling_range: tuple = (0.8, 1.2)):
        super(TimeSeriesAugmentation, self).__init__()
        self.noise_std = noise_std
        self.jitter_std = jitter_std
        self.scaling_range = scaling_range
    
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(x) * self.noise_std
        return x + noise
    
    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply jittering (small random shifts)."""
        noise = torch.randn_like(x) * self.jitter_std
        return x + noise
    
    def scaling(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random scaling."""
        scale = torch.rand(x.size(0), 1, 1, device=x.device) * \
                (self.scaling_range[1] - self.scaling_range[0]) + self.scaling_range[0]
        return x * scale
    
    def time_masking(self, x: torch.Tensor, mask_ratio: float = 0.1) -> torch.Tensor:
        """Apply random time masking."""
        B, L, C = x.shape
        mask_len = int(L * mask_ratio)
        
        for i in range(B):
            start_idx = torch.randint(0, L - mask_len + 1, (1,)).item()
            x[i, start_idx:start_idx + mask_len, :] = 0
        
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate two augmented views."""
        # First augmented view
        x1 = self.add_noise(x)
        x1 = self.jitter(x1)
        x1 = self.scaling(x1)
        
        # Second augmented view
        x2 = self.add_noise(x)
        x2 = self.time_masking(x2)
        x2 = self.scaling(x2)
        
        return x1, x2


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension
    hidden_dim : int
        Hidden dimension
    output_dim : int
        Output projection dimension
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 128):
        super(ProjectionHead, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return F.normalize(self.projection(x), dim=1)


class ContrastiveEncoder(nn.Module):
    """Contrastive encoder backbone.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension
    hidden_dim : int
        Hidden dimension
    num_layers : int
        Number of transformer layers
    num_heads : int
        Number of attention heads
    dropout : float
        Dropout probability
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 6,
                 num_heads: int = 8, dropout: float = 0.1):
        super(ContrastiveEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, input_dim)
            
        Returns
        -------
        torch.Tensor
            Encoded features of shape (B, hidden_dim)
        """
        B, L, _ = x.shape
        
        # Input embedding
        x = self.input_embedding(x)  # (B, L, hidden_dim)
        
        # Add positional encoding
        if L <= self.pos_encoding.size(1):
            x = x + self.pos_encoding[:, :L, :]
        else:
            # Interpolate for longer sequences
            pos_enc = F.interpolate(
                self.pos_encoding.transpose(1, 2), 
                size=L, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
            x = x + pos_enc
        
        # Transformer encoding
        x = self.transformer(x)  # (B, L, hidden_dim)
        
        # Global pooling
        x = x.transpose(1, 2)  # (B, hidden_dim, L)
        x = self.global_pool(x).squeeze(-1)  # (B, hidden_dim)
        
        return x


class Model(nn.Module):
    """Self-Supervised Contrastive Learning for Industrial Signal Foundation Model.
    
    Implements contrastive learning with time-series augmentations for
    self-supervised representation learning on industrial signals.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - hidden_dim : int, encoder hidden dimension (default: 256)
        - num_layers : int, number of transformer layers (default: 6)
        - num_heads : int, number of attention heads (default: 8)
        - projection_dim : int, projection head output dimension (default: 128)
        - temperature : float, temperature for contrastive loss (default: 0.1)
        - dropout : float, dropout probability (default: 0.1)
        - num_classes : int, number of output classes (for downstream tasks)
        - output_dim : int, output dimension (for regression tasks)
    metadata : Any, optional
        Dataset metadata
        
    Input Shape
    -----------
    x : torch.Tensor
        Input tensor of shape (batch_size, seq_len, input_dim)
        
    Output Shape
    ------------
    torch.Tensor
        For contrastive learning: (batch_size, projection_dim)
        For downstream tasks: depends on task type
        
    References
    ----------
    Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations" ICML 2020.
    He et al. "Momentum Contrast for Unsupervised Visual Representation Learning" CVPR 2020.
    Grill et al. "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning" NeurIPS 2020.
    Adapted for time-series industrial signals with temporal augmentation strategies for contrastive self-supervised learning.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.hidden_dim = getattr(args, 'hidden_dim', 256)
        self.num_layers = getattr(args, 'num_layers', 6)
        self.num_heads = getattr(args, 'num_heads', 8)
        self.projection_dim = getattr(args, 'projection_dim', 128)
        self.temperature = getattr(args, 'temperature', 0.1)
        self.dropout = getattr(args, 'dropout', 0.1)
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        
        # Data augmentation
        self.augmentation = TimeSeriesAugmentation()
        
        # Encoder backbone
        self.encoder = ContrastiveEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        # Projection head for contrastive learning
        self.projection_head = ProjectionHead(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.projection_dim
        )
        
        # Downstream task heads
        if self.num_classes is not None:
            # Classification head
            self.classifier = nn.Linear(self.hidden_dim, self.num_classes)
            self.task_type = 'classification'
        else:
            # Regression head
            self.regressor = nn.Linear(self.hidden_dim, self.output_dim)
            self.task_type = 'regression'
    
    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss (InfoNCE).
        
        Parameters
        ----------
        z1, z2 : torch.Tensor
            Projected features from two augmented views
            
        Returns
        -------
        torch.Tensor
            Contrastive loss
        """
        B = z1.size(0)
        
        # Concatenate features
        z = torch.cat([z1, z2], dim=0)  # (2B, projection_dim)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)
        
        # Create labels for positive pairs
        labels = torch.cat([torch.arange(B), torch.arange(B)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z.device)
        
        # Mask out self-similarity
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        labels = labels[~mask].view(2 * B, -1)
        sim_matrix = sim_matrix[~mask].view(2 * B, -1)
        
        # Compute loss
        positives = sim_matrix[labels.bool()].view(2 * B, -1)
        negatives = sim_matrix[~labels.bool()].view(2 * B, -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(2 * B, dtype=torch.long, device=z.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def forward(self, x: torch.Tensor, data_id=None, task_id=None, 
                mode: str = 'contrastive') -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, input_dim)
        data_id : Any, optional
            Data identifier (unused)
        task_id : Any, optional
            Task identifier (unused)
        mode : str
            Forward mode: 'contrastive' for self-supervised learning,
            'downstream' for supervised tasks
            
        Returns
        -------
        torch.Tensor or dict
            Output depends on mode and task type
        """
        if mode == 'contrastive':
            # Generate augmented views
            x1, x2 = self.augmentation(x)
            
            # Encode both views
            h1 = self.encoder(x1)
            h2 = self.encoder(x2)
            
            # Project to contrastive space
            z1 = self.projection_head(h1)
            z2 = self.projection_head(h2)
            
            # Compute contrastive loss
            loss = self.contrastive_loss(z1, z2)
            
            return {
                'loss': loss,
                'features': torch.cat([h1, h2], dim=0),
                'projections': torch.cat([z1, z2], dim=0)
            }
        
        else:  # downstream mode
            # Encode input
            features = self.encoder(x)
            
            if self.task_type == 'classification':
                output = self.classifier(features)
            else:
                output = self.regressor(features)
            
            return output


if __name__ == "__main__":
    # Test Contrastive SSL model
    import torch
    from argparse import Namespace
    
    def test_contrastive_ssl():
        """Test Contrastive SSL model."""
        print("Testing Contrastive SSL model...")
        
        # Test configuration
        args = Namespace(
            input_dim=3,
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            projection_dim=64,
            temperature=0.1,
            dropout=0.1,
            num_classes=5
        )
        
        model = Model(args)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test data
        batch_size = 8
        seq_len = 64
        x = torch.randn(batch_size, seq_len, args.input_dim)
        
        # Test contrastive mode
        with torch.no_grad():
            output_contrastive = model(x, mode='contrastive')
        
        print(f"Contrastive mode - Input: {x.shape}")
        print(f"Loss: {output_contrastive['loss'].item():.4f}")
        print(f"Features: {output_contrastive['features'].shape}")
        print(f"Projections: {output_contrastive['projections'].shape}")
        
        # Test downstream mode
        with torch.no_grad():
            output_downstream = model(x, mode='downstream')
        
        print(f"Downstream mode - Input: {x.shape}, Output: {output_downstream.shape}")
        assert output_downstream.shape == (batch_size, args.num_classes)
        
        print("✅ Contrastive SSL model tests passed!")
        return True
    
    test_contrastive_ssl()
