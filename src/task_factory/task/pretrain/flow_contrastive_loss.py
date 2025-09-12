"""
Flow Contrastive Loss for Joint Flow-Contrastive Pretraining

This module implements a joint loss function that combines Flow reconstruction loss 
with contrastive learning objectives for enhanced representation learning in
industrial vibration signal analysis.

Author: PHM-Vibench Team
Date: 2025-09-02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Import ContrastiveSSL components with better error handling
try:
    from ....model_factory.ISFM.ContrastiveSSL import TimeSeriesAugmentation, ProjectionHead
except (ImportError, ModuleNotFoundError):
    try:
        from src.model_factory.ISFM.ContrastiveSSL import TimeSeriesAugmentation, ProjectionHead
    except (ImportError, ModuleNotFoundError):
        # Fallback: Define minimal versions locally
        print("⚠️  ContrastiveSSL components not available, using fallback implementations")
        
        class TimeSeriesAugmentation(nn.Module):
            """Fallback TimeSeriesAugmentation implementation."""
            def __init__(self, noise_std=0.01, jitter_std=0.03, scaling_range=(0.8, 1.2)):
                super().__init__()
                self.noise_std = noise_std
                self.jitter_std = jitter_std
                self.scaling_range = scaling_range
            
            def forward(self, x):
                """Generate two augmented views."""
                # Simple noise-based augmentation
                x1 = x + torch.randn_like(x) * self.noise_std
                x2 = x + torch.randn_like(x) * self.noise_std
                return x1, x2
        
        class ProjectionHead(nn.Module):
            """Fallback ProjectionHead implementation."""
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.projection = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
            
            def forward(self, x):
                return F.normalize(self.projection(x), dim=-1)


class FlowContrastiveLoss(nn.Module):
    """
    Joint Flow-Contrastive loss function for pretraining.
    
    Combines Flow reconstruction loss with contrastive learning to learn
    robust representations. Supports configurable loss weighting and 
    gradient balancing mechanisms.
    
    Key Features:
    - Configurable loss weights (λ_flow, λ_contrastive)
    - Time-series augmentation for contrastive pairs
    - Gradient balancing for stable joint training
    - InfoNCE contrastive loss implementation
    """
    
    def __init__(
        self,
        flow_weight: float = 1.0,
        contrastive_weight: float = 0.1,
        contrastive_temperature: float = 0.1,
        projection_dim: int = 128,
        hidden_dim: int = 256,
        use_gradient_balancing: bool = True,
        augmentation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize FlowContrastiveLoss.
        
        Parameters
        ----------
        flow_weight : float
            Weight for Flow reconstruction loss (λ_flow)
        contrastive_weight : float
            Weight for contrastive learning loss (λ_contrastive)  
        contrastive_temperature : float
            Temperature parameter for InfoNCE loss
        projection_dim : int
            Dimension of contrastive projection space
        hidden_dim : int
            Hidden dimension for projection head
        use_gradient_balancing : bool
            Whether to use gradient balancing between losses
        augmentation_config : Optional[Dict[str, Any]]
            Configuration for time-series augmentation
        """
        super().__init__()
        
        # Loss weights
        self.flow_weight = flow_weight
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.use_gradient_balancing = use_gradient_balancing
        
        # Time-series augmentation for contrastive pairs
        aug_config = augmentation_config or {}
        self.augmentation = TimeSeriesAugmentation(
            noise_std=aug_config.get('noise_std', 0.01),
            jitter_std=aug_config.get('jitter_std', 0.03),
            scaling_range=aug_config.get('scaling_range', (0.8, 1.2))
        )
        
        # Projection head for contrastive learning
        # Note: input_dim will be set when we know the feature dimension
        self.projection_head = None
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        
        # Gradient balancing parameters
        self.flow_loss_scale = nn.Parameter(torch.tensor(1.0))
        self.contrastive_loss_scale = nn.Parameter(torch.tensor(1.0))
        
        print(f"🔗 初始化FlowContrastiveLoss:")
        print(f"   - Flow权重: {self.flow_weight}")
        print(f"   - 对比学习权重: {self.contrastive_weight}")
        print(f"   - 温度参数: {self.contrastive_temperature}")
        print(f"   - 投影维度: {self.projection_dim}")
    
    def _initialize_projection_head(self, input_dim: int):
        """Initialize projection head when feature dimension is known."""
        if self.projection_head is None:
            self.projection_head = ProjectionHead(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.projection_dim
            )
            print(f"   ✅ 投影头初始化完成 (输入维度: {input_dim})")
    
    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.
        
        Parameters
        ----------
        z1, z2 : torch.Tensor
            Projected features from two augmented views (B, projection_dim)
            
        Returns
        -------
        torch.Tensor
            Contrastive loss value
        """
        B = z1.size(0)
        
        # Concatenate features from both views
        z = torch.cat([z1, z2], dim=0)  # (2B, projection_dim)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.contrastive_temperature  # (2B, 2B)
        
        # Create labels for positive pairs
        labels = torch.cat([torch.arange(B), torch.arange(B)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z.device)
        
        # Mask out diagonal (self-similarity)
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        labels = labels[~mask].view(2 * B, -1)
        sim_matrix = sim_matrix[~mask].view(2 * B, -1)
        
        # Separate positive and negative similarities
        positives = sim_matrix[labels.bool()].view(2 * B, -1)
        negatives = sim_matrix[~labels.bool()].view(2 * B, -1)
        
        # Compute InfoNCE loss
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(2 * B, dtype=torch.long, device=z.device)
        
        contrastive_loss = F.cross_entropy(logits, labels)
        return contrastive_loss
    
    def compute_joint_loss(
        self,
        flow_outputs: Dict[str, torch.Tensor],
        features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute joint Flow-Contrastive loss.
        
        Parameters
        ----------
        flow_outputs : Dict[str, torch.Tensor]
            Flow model outputs containing flow_loss and features
        features : Optional[torch.Tensor]
            Pre-extracted features for contrastive learning (B, feature_dim)
            If None, will extract from flow_outputs
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing individual and combined losses
        """
        # Extract Flow loss
        flow_loss = flow_outputs.get('flow_loss', flow_outputs.get('loss'))
        if flow_loss is None:
            raise ValueError("Flow loss not found in flow_outputs")
        
        # Extract or compute features for contrastive learning
        if features is None:
            # Try to extract features from flow_outputs
            # This might be velocity field or encoded features
            features = flow_outputs.get('velocity', flow_outputs.get('features'))
            if features is None:
                # Fallback: use original input
                features = flow_outputs.get('x_original')
                if features is None:
                    raise ValueError("No suitable features found for contrastive learning")
        
        # Flatten features if needed (B, L, C) -> (B, L*C)
        if len(features.shape) == 3:
            B, L, C = features.shape
            features = features.view(B, L * C)
        
        # Initialize projection head if needed
        self._initialize_projection_head(features.shape[-1])
        
        # Generate augmented views for contrastive learning
        x_original = flow_outputs.get('x_original')
        if x_original is not None and len(x_original.shape) == 3:
            # Use original input for augmentation
            x1, x2 = self.augmentation(x_original)
            
            # Extract features from augmented views (simplified - normally would re-encode)
            # For now, apply same augmentation to features
            features1 = features + torch.randn_like(features) * 0.01  # Simplified augmentation
            features2 = features + torch.randn_like(features) * 0.01
        else:
            # Direct feature augmentation
            features1 = features + torch.randn_like(features) * 0.01
            features2 = features + torch.randn_like(features) * 0.01
        
        # Project features to contrastive space
        z1 = self.projection_head(features1)
        z2 = self.projection_head(features2)
        
        # Compute contrastive loss
        contrastive_loss_value = self.contrastive_loss(z1, z2)
        
        # Apply gradient balancing if enabled
        if self.use_gradient_balancing:
            # Scale losses to balance gradients
            scaled_flow_loss = flow_loss * torch.abs(self.flow_loss_scale)
            scaled_contrastive_loss = contrastive_loss_value * torch.abs(self.contrastive_loss_scale)
        else:
            scaled_flow_loss = flow_loss
            scaled_contrastive_loss = contrastive_loss_value
        
        # Compute weighted joint loss
        joint_loss = (
            self.flow_weight * scaled_flow_loss + 
            self.contrastive_weight * scaled_contrastive_loss
        )
        
        return {
            'total_loss': joint_loss,
            'flow_loss': flow_loss,
            'contrastive_loss': contrastive_loss_value,
            'flow_weight': self.flow_weight,
            'contrastive_weight': self.contrastive_weight,
            'projections_1': z1,
            'projections_2': z2,
            'features_1': features1,
            'features_2': features2
        }
    
    def update_weights(self, flow_weight: float, contrastive_weight: float):
        """
        Update loss weights during training.
        
        Parameters
        ----------
        flow_weight : float
            New Flow loss weight
        contrastive_weight : float
            New contrastive loss weight
        """
        old_flow = self.flow_weight
        old_contrastive = self.contrastive_weight
        
        self.flow_weight = flow_weight
        self.contrastive_weight = contrastive_weight
        
        print(f"🔄 损失权重更新:")
        print(f"   Flow: {old_flow:.3f} -> {flow_weight:.3f}")
        print(f"   Contrastive: {old_contrastive:.3f} -> {contrastive_weight:.3f}")
    
    def forward(
        self,
        flow_outputs: Dict[str, torch.Tensor],
        features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute joint loss.
        
        Parameters
        ----------
        flow_outputs : Dict[str, torch.Tensor]
            Flow model outputs
        features : Optional[torch.Tensor]
            Optional pre-extracted features
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Loss computation results
        """
        return self.compute_joint_loss(flow_outputs, features)