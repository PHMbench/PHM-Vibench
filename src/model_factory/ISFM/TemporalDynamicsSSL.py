"""Self-Supervised Temporal Dynamics Learning for Industrial Signals."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import math


class TemporalAugmentation(nn.Module):
    """Temporal augmentation strategies for self-supervised learning.
    
    Parameters
    ----------
    crop_ratio : float
        Ratio for temporal cropping
    permute_segments : int
        Number of segments for permutation
    """
    
    def __init__(self, crop_ratio: float = 0.8, permute_segments: int = 4):
        super(TemporalAugmentation, self).__init__()
        self.crop_ratio = crop_ratio
        self.permute_segments = permute_segments
    
    def temporal_crop(self, x: torch.Tensor) -> torch.Tensor:
        """Random temporal cropping."""
        B, L, C = x.shape
        crop_len = int(L * self.crop_ratio)
        
        start_idx = torch.randint(0, L - crop_len + 1, (B,))
        cropped = torch.zeros(B, crop_len, C, device=x.device)
        
        for i in range(B):
            cropped[i] = x[i, start_idx[i]:start_idx[i] + crop_len]
        
        return cropped
    
    def temporal_permutation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Temporal segment permutation and return permutation order."""
        B, L, C = x.shape
        segment_len = L // self.permute_segments
        
        # Reshape into segments
        x_segments = x[:, :segment_len * self.permute_segments].view(B, self.permute_segments, segment_len, C)
        
        # Generate random permutation for each sample
        permuted_x = torch.zeros_like(x_segments)
        permutation_orders = torch.zeros(B, self.permute_segments, dtype=torch.long, device=x.device)
        
        for i in range(B):
            perm = torch.randperm(self.permute_segments)
            permutation_orders[i] = perm
            permuted_x[i] = x_segments[i, perm]
        
        # Reshape back
        permuted_x = permuted_x.view(B, segment_len * self.permute_segments, C)
        
        return permuted_x, permutation_orders
    
    def temporal_masking(self, x: torch.Tensor, mask_ratio: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random temporal masking."""
        B, L, C = x.shape
        
        # Generate random mask
        mask = torch.rand(B, L, device=x.device) < mask_ratio
        
        # Apply mask
        masked_x = x.clone()
        masked_x[mask] = 0
        
        return masked_x, mask
    
    def forward(self, x: torch.Tensor, augmentation_type: str = 'crop') -> Tuple[torch.Tensor, Any]:
        """Apply temporal augmentation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, C)
        augmentation_type : str
            Type of augmentation ('crop', 'permute', 'mask')
            
        Returns
        -------
        Tuple[torch.Tensor, Any]
            Augmented tensor and auxiliary information
        """
        if augmentation_type == 'crop':
            return self.temporal_crop(x), None
        elif augmentation_type == 'permute':
            return self.temporal_permutation(x)
        elif augmentation_type == 'mask':
            return self.temporal_masking(x)
        else:
            raise ValueError(f"Unknown augmentation type: {augmentation_type}")


class TemporalEncoder(nn.Module):
    """Temporal encoder with multi-scale processing.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension
    hidden_dim : int
        Hidden dimension
    num_layers : int
        Number of layers
    num_heads : int
        Number of attention heads
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 6, num_heads: int = 8):
        super(TemporalEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Multi-scale convolutions
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, input_dim)
            
        Returns
        -------
        torch.Tensor
            Encoded features of shape (B, L, hidden_dim)
        """
        B, L, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)  # (B, L, hidden_dim)
        
        # Multi-scale convolutions
        x_conv = x.transpose(1, 2)  # (B, hidden_dim, L)
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = F.relu(conv(x_conv))
            conv_outputs.append(conv_out)
        
        # Concatenate and fuse
        multi_scale = torch.cat(conv_outputs, dim=1)  # (B, hidden_dim * 3, L)
        multi_scale = multi_scale.transpose(1, 2)  # (B, L, hidden_dim * 3)
        x = self.fusion(multi_scale)  # (B, L, hidden_dim)
        
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
        
        # Output projection
        x = self.output_projection(x)
        
        return x


class TemporalPredictionHead(nn.Module):
    """Prediction head for temporal dynamics tasks.
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension
    task_type : str
        Type of prediction task
    """
    
    def __init__(self, hidden_dim: int, task_type: str = 'next_step'):
        super(TemporalPredictionHead, self).__init__()
        
        self.task_type = task_type
        
        if task_type == 'next_step':
            # Predict next time step
            self.predictor = nn.Linear(hidden_dim, hidden_dim)
        elif task_type == 'permutation':
            # Predict permutation order
            self.predictor = nn.Linear(hidden_dim, 4)  # For 4 segments
        elif task_type in ['mask_reconstruction', 'mask']:
            # Reconstruct masked regions
            self.predictor = nn.Linear(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.predictor(x)


class Model(nn.Module):
    """Self-Supervised Temporal Dynamics Learning for Industrial Signals.
    
    Learns temporal representations through various self-supervised tasks
    including next-step prediction, temporal permutation, and masked reconstruction.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - hidden_dim : int, hidden dimension (default: 256)
        - num_layers : int, number of encoder layers (default: 6)
        - num_heads : int, number of attention heads (default: 8)
        - ssl_tasks : list, self-supervised tasks (default: ['next_step', 'permutation', 'mask'])
        - crop_ratio : float, temporal cropping ratio (default: 0.8)
        - permute_segments : int, number of permutation segments (default: 4)
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
    torch.Tensor or dict
        For self-supervised learning: dict with losses and features
        For downstream tasks: depends on task type
        
    References
    ----------
    Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" NAACL 2019.
    Vaswani et al. "Attention Is All You Need" NeurIPS 2017.
    Oord et al. "Representation Learning with Contrastive Predictive Coding" arXiv 2018.
    Adapted for time-series industrial signals with temporal dynamics learning through next-step prediction, permutation, and masked reconstruction tasks.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.hidden_dim = getattr(args, 'hidden_dim', 256)
        self.num_layers = getattr(args, 'num_layers', 6)
        self.num_heads = getattr(args, 'num_heads', 8)
        self.ssl_tasks = getattr(args, 'ssl_tasks', ['next_step', 'permutation', 'mask'])
        self.crop_ratio = getattr(args, 'crop_ratio', 0.8)
        self.permute_segments = getattr(args, 'permute_segments', 4)
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', 1)
        
        # Temporal augmentation
        self.augmentation = TemporalAugmentation(
            crop_ratio=self.crop_ratio,
            permute_segments=self.permute_segments
        )
        
        # Temporal encoder
        self.encoder = TemporalEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads
        )
        
        # SSL prediction heads
        self.ssl_heads = nn.ModuleDict()
        for task in self.ssl_tasks:
            self.ssl_heads[task] = TemporalPredictionHead(self.hidden_dim, task)
        
        # Downstream task heads
        if self.num_classes is not None:
            # Classification head
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim // 2, self.num_classes)
            )
            self.task_type = 'classification'
        else:
            # Regression head
            self.regressor = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim // 2, self.output_dim)
            )
            self.task_type = 'regression'
    
    def compute_ssl_losses(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute self-supervised learning losses.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, input_dim)
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of SSL losses
        """
        losses = {}
        
        for task in self.ssl_tasks:
            if task == 'next_step':
                # Next step prediction
                features = self.encoder(x)  # (B, L, hidden_dim)
                pred = self.ssl_heads[task](features[:, :-1])  # Predict next step
                target = features[:, 1:]  # Target is next step
                losses[task] = F.mse_loss(pred, target)
            
            elif task == 'permutation':
                # Temporal permutation prediction
                permuted_x, perm_orders = self.augmentation(x, 'permute')
                features = self.encoder(permuted_x)  # (B, L, hidden_dim)
                
                # Global pooling and prediction
                pooled = features.mean(dim=1)  # (B, hidden_dim)
                pred_order = self.ssl_heads[task](pooled)  # (B, num_segments)
                
                # Convert permutation to classification targets
                target_order = perm_orders.float()
                losses[task] = F.mse_loss(pred_order, target_order)
            
            elif task == 'mask':
                # Masked reconstruction
                masked_x, mask = self.augmentation(x, 'mask')
                features = self.encoder(masked_x)  # (B, L, hidden_dim)
                
                # Reconstruct original features
                original_features = self.encoder(x)
                pred_features = self.ssl_heads[task](features)
                
                # Loss only on masked positions
                mask_expanded = mask.unsqueeze(-1).expand_as(pred_features)
                losses[task] = F.mse_loss(
                    pred_features[mask_expanded], 
                    original_features[mask_expanded]
                )
        
        return losses
    
    def forward(self, x: torch.Tensor, data_id=None, task_id=None, 
                mode: str = 'ssl') -> torch.Tensor:
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
            Forward mode: 'ssl' for self-supervised learning,
            'downstream' for supervised tasks
            
        Returns
        -------
        torch.Tensor or dict
            Output depends on mode and task type
        """
        if mode == 'ssl':
            # Self-supervised learning
            ssl_losses = self.compute_ssl_losses(x)
            total_loss = sum(ssl_losses.values())
            
            # Get features for analysis
            features = self.encoder(x)
            
            return {
                'total_loss': total_loss,
                'ssl_losses': ssl_losses,
                'features': features
            }
        
        else:  # downstream mode
            # Encode features
            features = self.encoder(x)  # (B, L, hidden_dim)
            
            if self.task_type == 'classification':
                # Transpose for adaptive pooling
                features = features.transpose(1, 2)  # (B, hidden_dim, L)
                output = self.classifier(features)
            else:
                # Transpose for adaptive pooling
                features = features.transpose(1, 2)  # (B, hidden_dim, L)
                output = self.regressor(features)
            
            return output


if __name__ == "__main__":
    # Test Temporal Dynamics SSL model
    import torch
    from argparse import Namespace
    
    def test_temporal_dynamics_ssl():
        """Test Temporal Dynamics SSL model."""
        print("Testing Temporal Dynamics SSL model...")
        
        # Test configuration
        args = Namespace(
            input_dim=3,
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            ssl_tasks=['next_step', 'permutation', 'mask'],
            crop_ratio=0.8,
            permute_segments=4,
            num_classes=5
        )
        
        model = Model(args)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test data
        batch_size = 4
        seq_len = 64
        x = torch.randn(batch_size, seq_len, args.input_dim)
        
        # Test SSL mode
        with torch.no_grad():
            output_ssl = model(x, mode='ssl')
        
        print(f"SSL mode - Input: {x.shape}")
        print(f"Total loss: {output_ssl['total_loss'].item():.4f}")
        print(f"SSL losses: {[f'{k}: {v.item():.4f}' for k, v in output_ssl['ssl_losses'].items()]}")
        print(f"Features: {output_ssl['features'].shape}")
        
        # Test downstream mode
        with torch.no_grad():
            output_downstream = model(x, mode='downstream')
        
        print(f"Downstream mode - Input: {x.shape}, Output: {output_downstream.shape}")
        assert output_downstream.shape == (batch_size, args.num_classes)
        
        print("✅ Temporal Dynamics SSL model tests passed!")
        return True
    
    test_temporal_dynamics_ssl()
