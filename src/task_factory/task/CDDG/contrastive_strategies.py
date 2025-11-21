"""
Contrastive Learning Strategies for HSE Tasks

This module implements strategy pattern for different contrastive learning approaches,
providing a unified interface for various contrastive learning methods while maintaining
flexibility and extensibility.

Key Features:
1. Strategy Pattern: Decouples contrastive loss computation from task logic
2. Multi-view Support: Handles different data augmentation requirements
3. Ensemble Support: Combines multiple contrastive losses with configurable weights
4. HSE Integration: Specialized handling for prompt-guided contrastive learning
5. Performance Optimization: Memory-efficient computation for large-scale training

Authors: PHM-Vibench Team
Date: 2025-01-20
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

from ...Components.contrastive_losses import (
    InfoNCELoss, TripletLoss, SupConLoss, PrototypicalLoss,
    BarlowTwinsLoss, VICRegLoss
)

logger = logging.getLogger(__name__)


class MultiModalAttentionFusion(nn.Module):
    """
    Multi-modal attention fusion for integrating features and prompts.

    This module implements cross-attention between features and prompts,
    allowing the model to dynamically focus on relevant prompt information
    for each feature dimension.
    """

    def __init__(self, feature_dim: int, prompt_dim: int, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.prompt_dim = prompt_dim
        self.num_heads = num_heads

        # Multi-head attention components
        self.head_dim = feature_dim // num_heads
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        self.feature_proj = nn.Linear(feature_dim, feature_dim)
        self.prompt_proj = nn.Linear(prompt_dim, feature_dim)
        self.output_proj = nn.Linear(feature_dim, feature_dim)

        # Layer normalization
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )

    def forward(self, features: torch.Tensor, prompts: torch.Tensor,
                system_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for multi-modal attention fusion.

        Args:
            features: Feature tensor [batch_size, feature_dim]
            prompts: Prompt tensor [batch_size, prompt_dim]
            system_ids: Optional system IDs for system-aware processing

        Returns:
            Enhanced features with prompt integration
        """
        batch_size = features.shape[0]

        # Project prompts to feature dimension
        prompt_features = self.prompt_proj(prompts)  # [batch_size, feature_dim]

        # Multi-head self-attention with prompts as keys/values
        q = self.feature_proj(features).view(batch_size, self.num_heads, self.head_dim)
        k = prompt_features.view(batch_size, self.num_heads, self.head_dim)
        v = prompt_features.view(batch_size, self.num_heads, self.head_dim)

        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention
        attended_features = torch.matmul(attention_weights, v)
        attended_features = attended_features.view(batch_size, self.feature_dim)

        # Residual connection and layer norm
        attended_features = self.norm1(features + attended_features)

        # Feed-forward network
        ffn_output = self.ffn(attended_features)
        output = self.norm2(attended_features + ffn_output)

        return self.output_proj(output)


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism for controlled prompt integration.

    This module uses learnable gates to control the influence of prompts
    on the feature representations, allowing adaptive fusion based on
    the input characteristics.
    """

    def __init__(self, feature_dim: int, prompt_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.prompt_dim = prompt_dim

        # Projection layers
        self.feature_proj = nn.Linear(feature_dim, feature_dim)
        self.prompt_proj = nn.Linear(prompt_dim, feature_dim)

        # Gating mechanism
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim + prompt_dim, feature_dim),
            nn.Sigmoid()
        )

        # Output projection
        self.output_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, features: torch.Tensor, prompts: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for gated fusion.

        Args:
            features: Feature tensor [batch_size, feature_dim]
            prompts: Prompt tensor [batch_size, prompt_dim]

        Returns:
            Gated-fused features
        """
        # Project inputs
        proj_features = self.feature_proj(features)
        proj_prompts = self.prompt_proj(prompts)

        # Compute gate
        gate_input = torch.cat([features, prompts], dim=-1)
        gate = self.gate_net(gate_input)

        # Gated fusion
        fused_features = gate * proj_features + (1 - gate) * proj_prompts

        return self.output_proj(fused_features)


class AdaptivePromptWeightScheduler:
    """
    Adaptive scheduler for prompt weight during training.

    This scheduler adjusts the influence of prompts based on training progress
    and performance metrics, enabling curriculum learning for prompt integration.
    """

    def __init__(self, initial_weight: float = 0.1, final_weight: float = 0.3,
                 warmup_epochs: int = 10, total_epochs: int = 100):
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def get_weight(self, epoch: int, performance_metric: Optional[float] = None) -> float:
        """
        Get current prompt weight based on training progress.

        Args:
            epoch: Current training epoch
            performance_metric: Optional performance metric for adaptive adjustment

        Returns:
            Current prompt weight
        """
        if epoch < self.warmup_epochs:
            # Linear warmup
            weight = self.initial_weight + (self.final_weight - self.initial_weight) * \
                     (epoch / self.warmup_epochs)
        else:
            # Gradual decay based on progress
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            weight = self.final_weight * (1 - 0.5 * progress)

        # Adjust based on performance if provided
        if performance_metric is not None:
            # Higher performance -> higher prompt weight
            weight *= (1 + 0.1 * performance_metric)

        return max(0.01, min(1.0, weight))  # Clamp to reasonable range


class ContrastiveStrategy(ABC):
    """
    Base class for contrastive learning strategies.

    This abstract class defines the interface for different contrastive learning approaches,
    enabling flexible implementation of various SOTA contrastive methods while maintaining
    consistency with the HSE framework.
    """

    def __init__(self, strategy_config: Dict[str, Any]):
        """
        Initialize contrastive strategy.

        Args:
            strategy_config: Configuration dictionary containing strategy-specific parameters
        """
        self.config = strategy_config
        self._validate_config()

    @abstractmethod
    def compute_loss(
        self,
        features: torch.Tensor,     # Backbone output features
        projections: torch.Tensor,  # Projection head output (critical for contrastive learning!)
        prompts: Optional[torch.Tensor],      # HSE prompt vectors
        labels: Optional[torch.Tensor],       # Ground truth labels
        system_ids: Optional[torch.Tensor]    # System/domain information
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive loss.

        Args:
            features: Raw features from backbone network
            projections: Projected features for contrastive computation
            prompts: HSE prompt vectors for system-aware learning
            labels: Class labels for supervised contrastive learning
            system_ids: System/domain identifiers for cross-domain learning

        Returns:
            Dictionary containing:
            - 'loss': Total contrastive loss value
            - 'components': Dictionary of individual loss components
            - 'metrics': Additional metrics for logging
        """
        pass

    @property
    def requires_multiple_views(self) -> bool:
        """
        Check if strategy requires multiple data views.

        Returns:
            True if strategy needs multiple augmented views, False otherwise
        """
        return False

    @property
    def requires_prompts(self) -> bool:
        """
        Check if strategy requires HSE prompt vectors.

        Returns:
            True if strategy uses prompt-guided learning, False otherwise
        """
        return True

    @property
    def requires_labels(self) -> bool:
        """
        Check if strategy requires ground truth labels.

        Returns:
            True if strategy needs labels (e.g., supervised contrastive), False otherwise
        """
        return False

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate strategy configuration parameters."""
        pass

    def _check_input_requirements(
        self,
        features: torch.Tensor,
        projections: torch.Tensor,
        prompts: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        system_ids: Optional[torch.Tensor] = None
    ) -> None:
        """Validate input tensor requirements."""
        if features is None:
            raise ValueError("Features cannot be None")
        if projections is None:
            raise ValueError("Projections cannot be None")

        # Check dimension consistency
        batch_size = features.size(0)
        if projections.size(0) != batch_size:
            raise ValueError(f"Batch size mismatch: features {batch_size}, projections {projections.size(0)}")

        if self.requires_prompts and prompts is None:
            raise ValueError("This strategy requires prompt vectors")

        if self.requires_labels and labels is None:
            raise ValueError("This strategy requires ground truth labels")


class SingleContrastiveStrategy(ContrastiveStrategy):
    """
    Single contrastive learning strategy using one loss function.

    This strategy uses a single contrastive loss function (e.g., InfoNCE, Triplet, SupCon)
    and is suitable for standard contrastive learning scenarios.
    """

    def __init__(self, strategy_config: Dict[str, Any]):
        super().__init__(strategy_config)
        self.loss_type = strategy_config.get('loss_type', 'INFONCE')
        self.contrastive_loss = self._create_loss()

    def compute_loss(
        self,
        features: torch.Tensor,
        projections: torch.Tensor,
        prompts: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        system_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute single contrastive loss with enhanced HSE prompt integration."""
        self._check_input_requirements(features, projections, prompts, labels, system_ids)

        # Enhanced prompt integration: fuse prompts with projections for contrastive computation
        contrastive_features = self._integrate_prompts_with_features(projections, prompts, system_ids)

        try:
            if self.loss_type in ['INFONCE', 'BARLOWTWINS', 'VICREG']:
                # Self-supervised losses don't need labels
                if self.loss_type == 'INFONCE':
                    loss_value = self._compute_infonce_with_prompts(contrastive_features, prompts, system_ids, labels)
                elif self.loss_type == 'BARLOWTWINS':
                    # BarlowTwins needs two views - create second view by augmentation
                    view2 = self._create_augmented_view(contrastive_features)
                    view2_prompts = self._integrate_prompts_with_features(view2, prompts, system_ids)
                    loss_value = self.contrastive_loss(contrastive_features, view2_prompts)
                elif self.loss_type == 'VICREG':
                    # VICReg needs two views
                    view2 = self._create_augmented_view(contrastive_features)
                    view2_prompts = self._integrate_prompts_with_features(view2, prompts, system_ids)
                    loss_value = self.contrastive_loss(contrastive_features, view2_prompts)

            elif self.loss_type in ['SUPCON', 'TRIPLET', 'PROTOTYPICAL']:
                # Supervised losses need labels
                if labels is None:
                    raise ValueError(f"{self.loss_type} requires labels")

                if self.loss_type == 'SUPCON':
                    loss_value = self._compute_supcon_with_prompts(contrastive_features, prompts, system_ids, labels)
                elif self.loss_type == 'TRIPLET':
                    loss_value = self._compute_triplet_with_prompts(contrastive_features, prompts, system_ids, labels)
                elif self.loss_type == 'PROTOTYPICAL':
                    loss_value = self._compute_prototypical_with_prompts(contrastive_features, prompts, system_ids, labels)

            else:
                raise ValueError(f"Unsupported loss type: {self.loss_type}")

            # Add prompt regularization loss for better training
            prompt_regularization = self._compute_prompt_regularization(prompts)

            total_loss = loss_value + prompt_regularization

            return {
                'loss': total_loss,
                'components': {
                    self.loss_type: loss_value,
                    'prompt_regularization': prompt_regularization
                },
                'metrics': {
                    f'{self.loss_type.lower()}_loss': loss_value.item(),
                    'prompt_regularization_loss': prompt_regularization.item()
                }
            }

        except Exception as e:
            logger.error(f"Error in {self.loss_type} computation: {e}")
            return {
                'loss': torch.tensor(0.0, device=features.device),
                'components': {self.loss_type: torch.tensor(0.0, device=features.device)},
                'metrics': {}
            }

    @property
    def requires_multiple_views(self) -> bool:
        """Check if loss requires multiple views."""
        return self.loss_type in ['BARLOWTWINS', 'VICREG']

    @property
    def requires_labels(self) -> bool:
        """Check if loss requires labels."""
        return self.loss_type in ['SUPCON', 'TRIPLET', 'PROTOTYPICAL']

    def _validate_config(self) -> None:
        """Validate single strategy configuration."""
        required_keys = ['loss_type']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        supported_losses = ['INFONCE', 'TRIPLET', 'SUPCON', 'PROTOTYPICAL', 'BARLOWTWINS', 'VICREG']
        if self.config['loss_type'] not in supported_losses:
            raise ValueError(f"Unsupported loss type: {self.config['loss_type']}")

    def _create_loss(self) -> nn.Module:
        """Create contrastive loss instance based on configuration."""
        loss_type = self.config['loss_type']

        loss_mapping = {
            'INFONCE': lambda: InfoNCELoss(
                temperature=self.config.get('temperature', 0.07)
            ),
            'TRIPLET': lambda: TripletLoss(
                margin=self.config.get('margin', 0.3)
            ),
            'SUPCON': lambda: SupConLoss(
                temperature=self.config.get('temperature', 0.07)
            ),
            'PROTOTYPICAL': lambda: PrototypicalLoss(
                distance_fn=self.config.get('distance_fn', 'euclidean')
            ),
            'BARLOWTWINS': lambda: BarlowTwinsLoss(
                lambda_param=self.config.get('barlow_lambda', 5e-3)
            ),
            'VICREG': lambda: VICRegLoss(
                lambda_inv=self.config.get('vicreg_lambda_inv', 25.0),
                mu_var=self.config.get('vicreg_mu_var', 25.0),
                nu_cov=self.config.get('vicreg_nu_cov', 1.0)
            )
        }

        return loss_mapping[loss_type]()

    def _create_augmented_view(self, features: torch.Tensor) -> torch.Tensor:
        """
        Create augmented view for self-supervised losses.

        Args:
            features: Original feature tensor

        Returns:
            Augmented feature tensor
        """
        # Simple augmentation: add Gaussian noise
        noise_std = self.config.get('augmentation_noise_std', 0.1)
        noise = torch.randn_like(features) * noise_std
        return features + noise

    def _integrate_prompts_with_features(
        self,
        features: torch.Tensor,
        prompts: Optional[torch.Tensor] = None,
        system_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Enhanced HSE prompt integration with multiple fusion strategies.

        Args:
            features: Input features [batch_size, feature_dim]
            prompts: HSE prompt vectors [batch_size, prompt_dim]
            system_ids: System IDs for system-aware processing

        Returns:
            Enhanced features with prompt integration
        """
        if prompts is None:
            return features

        batch_size, feature_dim = features.shape
        prompt_dim = prompts.shape[-1]

        # Get prompt fusion configuration
        fusion_type = self.config.get('prompt_fusion', 'attention')
        prompt_weight = self.config.get('prompt_weight', 0.1)

        if fusion_type == 'add':
            # Element-wise addition with learnable weight
            if prompt_dim == feature_dim:
                enhanced_features = features + prompt_weight * prompts
            else:
                # Use linear projection to match dimensions
                if not hasattr(self, 'prompt_projector'):
                    self.prompt_projector = nn.Linear(prompt_dim, feature_dim).to(features.device)
                enhanced_features = features + prompt_weight * self.prompt_projector(prompts)

        elif fusion_type == 'concat':
            # Concatenation followed by projection
            if not hasattr(self, 'fusion_projector'):
                self.fusion_projector = nn.Linear(feature_dim + prompt_dim, feature_dim).to(features.device)
            enhanced_features = self.fusion_projector(torch.cat([features, prompts], dim=-1))

        elif fusion_type == 'attention':
            # Attention-based fusion
            if not hasattr(self, 'attention_fusion'):
                self.attention_fusion = MultiModalAttentionFusion(feature_dim, prompt_dim).to(features.device)
            enhanced_features = self.attention_fusion(features, prompts, system_ids)

        elif fusion_type == 'gate':
            # Gated fusion
            if not hasattr(self, 'gate_fusion'):
                self.gate_fusion = GatedFusion(feature_dim, prompt_dim).to(features.device)
            enhanced_features = self.gate_fusion(features, prompts)

        else:
            # Default to simple addition
            logger.warning(f"Unknown fusion type: {fusion_type}, using simple addition")
            enhanced_features = features + prompt_weight * prompts

        return enhanced_features

    def _compute_infonce_with_prompts(
        self,
        features: torch.Tensor,
        prompts: Optional[torch.Tensor] = None,
        system_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss with enhanced prompt integration and system-aware sampling.
        """
        # System-aware positive/negative sampling
        if system_ids is not None and self.config.get('use_system_sampling', True):
            sampling_result = self._apply_system_aware_sampling(features, system_ids, prompts)
            features = sampling_result['features']

        # Compute standard InfoNCE loss
        return self.contrastive_loss(features, labels)

    def _compute_supcon_with_prompts(
        self,
        features: torch.Tensor,
        prompts: Optional[torch.Tensor] = None,
        system_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute SupCon loss with prompt-enhanced feature representations.
        """
        # Apply prompt-aware contrastive sampling
        if prompts is not None:
            # Use prompts to enhance same-class identification
            features = self._enhance_with_prompts_for_supcon(features, prompts, labels)

        return self.contrastive_loss(features, labels)

    def _compute_triplet_with_prompts(
        self,
        features: torch.Tensor,
        prompts: Optional[torch.Tensor] = None,
        system_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Triplet loss with prompt-guided sample mining.
        """
        if prompts is not None:
            # Use prompts for better anchor/positive/negative selection
            features = self._prompt_guided_triplet_mining(features, prompts, labels, system_ids)

        return self.contrastive_loss(features, labels)

    def _compute_prototypical_with_prompts(
        self,
        features: torch.Tensor,
        prompts: Optional[torch.Tensor] = None,
        system_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Prototypical loss with prompt-enhanced class prototypes.
        """
        if prompts is not None:
            # Enhance class prototypes using prompt information
            features = self._enhance_prototypes_with_prompts(features, prompts, labels, system_ids)

        return self.contrastive_loss(features, labels)

    def _compute_prompt_regularization(self, prompts: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute prompt regularization loss for better training stability.
        """
        if prompts is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        reg_loss = torch.tensor(0.0, device=prompts.device)

        # L2 regularization
        if self.config.get('prompt_l2_reg', True):
            reg_loss += 0.01 * torch.mean(prompts ** 2)

        # Orthogonality regularization (encourage diverse prompts)
        if self.config.get('prompt_orthogonal_reg', False) and prompts.shape[0] > 1:
            prompt_similarities = torch.mm(prompts, prompts.t())
            identity = torch.eye(prompt_similarities.shape[0], device=prompts.device)
            reg_loss += 0.01 * torch.mean((prompt_similarities - identity) ** 2)

        return reg_loss

    def _apply_system_aware_sampling(
        self,
        features: torch.Tensor,
        system_ids: torch.Tensor,
        prompts: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Apply advanced system-aware sampling for better cross-domain generalization.

        This method implements sophisticated sampling strategies that:
        1. Balance intra-system and inter-system contrastive learning
        2. Perform hard negative mining across different systems
        3. Apply system-adaptive temperature scaling
        4. Enable progressive domain mixing for better generalization

        Args:
            features: Feature vectors [batch_size, feature_dim]
            system_ids: System IDs [batch_size]
            prompts: Optional prompt vectors [batch_size, prompt_dim]

        Returns:
            Dictionary containing enhanced features and sampling metadata
        """
        if not self.config.get('enable_cross_system_contrast', True):
            return {'features': features, 'sampling_info': {}}

        batch_size, feature_dim = features.shape
        device = features.device

        # 1. Analyze system distribution
        unique_systems = torch.unique(system_ids)
        num_systems = len(unique_systems)

        if num_systems <= 1:
            return {'features': features, 'sampling_info': {'single_system': True}}

        # 2. Compute system relationships
        system_relationships = self._compute_system_relationships(features, system_ids, prompts)

        # 3. Apply adaptive sampling strategy
        sampling_strategy = self.config.get('system_sampling_strategy', 'balanced')

        if sampling_strategy == 'balanced':
            enhanced_features = self._balanced_system_sampling(features, system_ids, system_relationships)
        elif sampling_strategy == 'hard_negative':
            enhanced_features = self._hard_negative_system_sampling(features, system_ids, system_relationships)
        elif sampling_strategy == 'progressive_mixing':
            enhanced_features = self._progressive_system_mixing(features, system_ids, prompts)
        else:
            # Default to balanced sampling
            enhanced_features = self._balanced_system_sampling(features, system_ids, system_relationships)

        # 4. Apply system-aware temperature scaling
        if self.config.get('adaptive_temperature', True):
            enhanced_features = self._apply_adaptive_temperature(enhanced_features, system_ids)

        # 5. Collect sampling metadata
        sampling_info = {
            'num_systems': num_systems,
            'system_distribution': {int(sys_id): int((system_ids == sys_id).sum()) for sys_id in unique_systems},
            'sampling_strategy': sampling_strategy,
            'system_relationships': system_relationships
        }

        return {
            'features': enhanced_features,
            'sampling_info': sampling_info
        }

    def _compute_system_relationships(
        self,
        features: torch.Tensor,
        system_ids: torch.Tensor,
        prompts: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute relationships between different systems for informed sampling.

        Args:
            features: Feature vectors [batch_size, feature_dim]
            system_ids: System IDs [batch_size]
            prompts: Optional prompt vectors [batch_size, prompt_dim]

        Returns:
            Dictionary containing system relationship matrices
        """
        unique_systems = torch.unique(system_ids)
        num_systems = len(unique_systems)
        device = features.device

        # Initialize relationship matrices
        system_similarity_matrix = torch.zeros(num_systems, num_systems, device=device)
        system_distance_matrix = torch.zeros(num_systems, num_systems, device=device)

        # Compute system-level representations
        system_representations = []
        for i, system_id in enumerate(unique_systems):
            system_mask = (system_ids == system_id)
            system_features = features[system_mask]

            # Use mean pooling to get system representation
            system_repr = torch.mean(system_features, dim=0)
            system_representations.append(system_repr)

        system_representations = torch.stack(system_representations)  # [num_systems, feature_dim]

        # Compute pairwise system similarities and distances
        for i in range(num_systems):
            for j in range(num_systems):
                if i != j:
                    # Cosine similarity between systems
                    sim = F.cosine_similarity(
                        system_representations[i].unsqueeze(0),
                        system_representations[j].unsqueeze(0)
                    )
                    system_similarity_matrix[i, j] = sim

                    # Euclidean distance between systems
                    dist = torch.norm(system_representations[i] - system_representations[j])
                    system_distance_matrix[i, j] = dist

        # Add prompt-level relationships if available
        prompt_relationships = {}
        if prompts is not None:
            prompt_similarities = torch.zeros(num_systems, num_systems, device=device)
            for i, system_id in enumerate(unique_systems):
                for j in range(num_systems):
                    if i != j:
                        system_i_mask = (system_ids == system_id)
                        system_j_mask = (system_ids == unique_systems[j])

                        system_i_prompts = prompts[system_i_mask]
                        system_j_prompts = prompts[system_j_mask]

                        # Average prompt similarity between systems
                        prompt_sim = F.cosine_similarity(
                            torch.mean(system_i_prompts, dim=0).unsqueeze(0),
                            torch.mean(system_j_prompts, dim=0).unsqueeze(0)
                        )
                        prompt_similarities[i, j] = prompt_sim

            prompt_relationships['prompt_similarity_matrix'] = prompt_similarities

        return {
            'feature_similarity_matrix': system_similarity_matrix,
            'distance_matrix': system_distance_matrix,
            **prompt_relationships
        }

    def _balanced_system_sampling(
        self,
        features: torch.Tensor,
        system_ids: torch.Tensor,
        system_relationships: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Perform balanced sampling across systems to ensure uniform representation.
        """
        unique_systems = torch.unique(system_ids)
        enhanced_features = []
        target_samples_per_system = min(
            [(system_ids == sys_id).sum() for sys_id in unique_systems]
        )

        for system_id in unique_systems:
            system_mask = (system_ids == system_id)
            system_features = features[system_mask]

            # Randomly sample or upsample to target size
            if len(system_features) >= target_samples_per_system:
                # Random sampling
                indices = torch.randperm(len(system_features))[:target_samples_per_system]
                sampled_features = system_features[indices]
            else:
                # Upsample with random duplication
                indices = torch.randint(0, len(system_features), (target_samples_per_system,))
                sampled_features = system_features[indices]

            enhanced_features.append(sampled_features)

        return torch.cat(enhanced_features, dim=0)

    def _hard_negative_system_sampling(
        self,
        features: torch.Tensor,
        system_ids: torch.Tensor,
        system_relationships: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Perform hard negative mining across different systems.
        """
        unique_systems = torch.unique(system_ids)
        enhanced_features = []

        for i, system_id in enumerate(unique_systems):
            system_mask = (system_ids == system_id)
            system_features = features[system_mask]

            # Find most dissimilar systems (hard negatives)
            similarity_matrix = system_relationships['feature_similarity_matrix']
            dissimilar_systems = torch.argsort(similarity_matrix[i])  # Most dissimilar first

            # Sample from hard negative systems
            hard_neg_features = []
            for j in dissimilar_systems[:2]:  # Take 2 most dissimilar systems
                if j != i:
                    neg_system_id = unique_systems[j]
                    neg_mask = (system_ids == neg_system_id)
                    neg_features = features[neg_mask]

                    # Sample a subset of hard negatives
                    num_neg_samples = min(len(neg_features), len(system_features) // 2)
                    if num_neg_samples > 0:
                        indices = torch.randperm(len(neg_features))[:num_neg_samples]
                        hard_neg_features.append(neg_features[indices])

            # Combine positive and hard negative features
            if hard_neg_features:
                combined_features = torch.cat([system_features] + hard_neg_features, dim=0)
            else:
                combined_features = system_features

            enhanced_features.append(combined_features)

        return torch.cat(enhanced_features, dim=0)

    def _progressive_system_mixing(
        self,
        features: torch.Tensor,
        system_ids: torch.Tensor,
        prompts: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply progressive domain mixing for better cross-domain adaptation.
        """
        unique_systems = torch.unique(system_ids)
        mixing_ratio = self.config.get('mixing_ratio', 0.3)

        enhanced_features = []
        for i, system_id in enumerate(unique_systems):
            system_mask = (system_ids == system_id)
            system_features = features[system_mask]

            # Mix with other systems
            mixed_features = [system_features]

            for j, other_system_id in enumerate(unique_systems):
                if i != j:
                    other_mask = (system_ids == other_system_id)
                    other_features = features[other_mask]

                    # Randomly select samples for mixing
                    num_mix_samples = min(len(system_features), len(other_features)) // 4
                    if num_mix_samples > 0:
                        system_indices = torch.randperm(len(system_features))[:num_mix_samples]
                        other_indices = torch.randperm(len(other_features))[:num_mix_samples]

                        # Linear interpolation between systems
                        alpha = torch.rand(num_mix_samples, 1, device=features.device) * mixing_ratio
                        mixed_samples = (
                            (1 - alpha) * system_features[system_indices] +
                            alpha * other_features[other_indices]
                        )
                        mixed_features.append(mixed_samples)

            enhanced_features.append(torch.cat(mixed_features, dim=0))

        return torch.cat(enhanced_features, dim=0)

    def _apply_adaptive_temperature(
        self,
        features: torch.Tensor,
        system_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply system-aware temperature scaling to features.
        """
        unique_systems = torch.unique(system_ids)
        enhanced_features = []

        for system_id in unique_systems:
            system_mask = (system_ids == system_id)
            system_features = features[system_mask]

            # Compute system-specific temperature based on feature variance
            feature_variance = torch.var(system_features, dim=0).mean()
            # Higher variance -> lower temperature (sharper distribution)
            system_temperature = 1.0 / (1.0 + feature_variance)

            # Apply temperature scaling
            scaled_features = system_features * system_temperature
            enhanced_features.append(scaled_features)

        return torch.cat(enhanced_features, dim=0)

    def _enhance_with_prompts_for_supcon(
        self,
        features: torch.Tensor,
        prompts: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Enhance features for SupCon using prompt-based similarity.
        """
        # Compute prompt similarity matrix
        prompt_similarities = torch.mm(prompts, prompts.t())
        prompt_similarities = F.normalize(prompt_similarities, p=2, dim=1)

        # Enhance feature similarities with prompt information
        feature_similarities = torch.mm(features, features.t())
        feature_similarities = F.normalize(feature_similarities, p=2, dim=1)

        # Combine feature and prompt similarities
        alpha = self.config.get('prompt_feature_fusion_weight', 0.3)
        combined_similarities = (1 - alpha) * feature_similarities + alpha * prompt_similarities

        # Use combined similarities to enhance features
        enhanced_features = torch.mm(combined_similarities, features)

        return enhanced_features

    def _prompt_guided_triplet_mining(
        self,
        features: torch.Tensor,
        prompts: torch.Tensor,
        labels: torch.Tensor,
        system_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform prompt-guided triplet mining for better sample selection.
        """
        # This is a simplified implementation - in practice, you'd implement
        # sophisticated mining strategies using prompt similarities
        return features

    def _enhance_prototypes_with_prompts(
        self,
        features: torch.Tensor,
        prompts: torch.Tensor,
        labels: torch.Tensor,
        system_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Enhance class prototypes using prompt information.
        """
        # Compute class prototypes
        unique_labels = torch.unique(labels)
        enhanced_features = features.clone()

        for label in unique_labels:
            class_mask = (labels == label)
            class_features = features[class_mask]
            class_prompts = prompts[class_mask]

            # Compute prompt-enhanced prototype
            prompt_prototype = torch.mean(class_prompts, dim=0)
            feature_prototype = torch.mean(class_features, dim=0)

            # Combine prototypes
            alpha = self.config.get('prototype_fusion_weight', 0.3)
            enhanced_prototype = (1 - alpha) * feature_prototype + alpha * prompt_prototype

            # Move class features toward enhanced prototype
            enhanced_features[class_mask] = (
                enhanced_features[class_mask] + 0.1 * (enhanced_prototype - class_features)
            )

        return enhanced_features


class EnsembleContrastiveStrategy(ContrastiveStrategy):
    """
    Ensemble contrastive learning strategy combining multiple loss functions.

    This strategy combines multiple contrastive losses with configurable weights,
    enabling sophisticated multi-objective contrastive learning scenarios.
    """

    def __init__(self, strategy_config: Dict[str, Any]):
        super().__init__(strategy_config)
        self.losses_config = strategy_config.get('losses', [])
        self.weight_normalization = strategy_config.get('weight_normalization', True)

        if self.weight_normalization:
            self._normalize_weights()

        self.strategies = self._create_strategies()

    def compute_loss(
        self,
        features: torch.Tensor,
        projections: torch.Tensor,
        prompts: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        system_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute ensemble contrastive loss."""
        self._check_input_requirements(features, projections, prompts, labels, system_ids)

        total_loss = torch.tensor(0.0, device=features.device)
        components = {}
        metrics = {}

        for loss_config in self.losses_config:
            strategy_name = loss_config['name']
            weight = loss_config['weight']

            # Create temporary single strategy for this loss
            temp_strategy = SingleContrastiveStrategy({
                'loss_type': strategy_name,
                **loss_config
            })

            try:
                result = temp_strategy.compute_loss(
                    features, projections, prompts, labels, system_ids
                )

                loss_value = result['loss'] * weight
                total_loss += loss_value

                components[strategy_name] = loss_value

                # Update metrics
                for key, value in result.get('metrics', {}).items():
                    metrics[f'{strategy_name}_{key}'] = value

            except Exception as e:
                logger.warning(f"Error computing {strategy_name} loss: {e}")
                components[strategy_name] = torch.tensor(0.0, device=features.device)

        return {
            'loss': total_loss,
            'components': components,
            'metrics': metrics
        }

    @property
    def requires_multiple_views(self) -> bool:
        """Check if any component requires multiple views."""
        return any(
            loss_config['name'] in ['BARLOWTWINS', 'VICREG']
            for loss_config in self.losses_config
        )

    @property
    def requires_labels(self) -> bool:
        """Check if any component requires labels."""
        return any(
            loss_config['name'] in ['SUPCON', 'TRIPLET', 'PROTOTYPICAL']
            for loss_config in self.losses_config
        )

    def _validate_config(self) -> None:
        """Validate ensemble strategy configuration."""
        if 'losses' not in self.config:
            raise ValueError("Ensemble strategy requires 'losses' configuration")

        if not self.config['losses']:
            raise ValueError("Losses list cannot be empty")

        # Validate each loss configuration
        for i, loss_config in enumerate(self.config['losses']):
            if not isinstance(loss_config, dict):
                raise ValueError(f"Loss config {i} must be a dictionary")

            required_keys = ['name', 'weight']
            for key in required_keys:
                if key not in loss_config:
                    raise ValueError(f"Loss config {i} missing required key: {key}")

            if not isinstance(loss_config['weight'], (int, float)):
                raise ValueError(f"Weight in loss config {i} must be numeric")

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        total_weight = sum(loss_config['weight'] for loss_config in self.losses_config)

        if abs(total_weight - 1.0) > 1e-6:
            if total_weight == 0:
                raise ValueError("Total weight cannot be zero")

            for loss_config in self.losses_config:
                loss_config['weight'] = loss_config['weight'] / total_weight

    def _create_strategies(self) -> Dict[str, SingleContrastiveStrategy]:
        """Create single strategies for ensemble components."""
        strategies = {}

        for loss_config in self.losses_config:
            name = loss_config['name']
            strategies[name] = SingleContrastiveStrategy({
                'loss_type': name,
                **{k: v for k, v in loss_config.items() if k != 'name' and k != 'weight'}
            })

        return strategies


class ContrastiveStrategyManager:
    """
    Manager class for contrastive learning strategies.

    This class provides a unified interface for creating and managing different
    contrastive learning strategies, supporting dynamic strategy switching and
    ensemble combinations.
    """

    def __init__(self):
        """Initialize strategy manager."""
        self.current_strategy = None
        self.strategy_history = []

    def create_strategy(self, strategy_config: Dict[str, Any]) -> ContrastiveStrategy:
        """
        Create contrastive strategy from configuration.

        Args:
            strategy_config: Strategy configuration dictionary

        Returns:
            Configured contrastive strategy instance
        """
        strategy_type = strategy_config.get('type', 'single')

        if strategy_type == 'single':
            strategy = SingleContrastiveStrategy(strategy_config)
        elif strategy_type == 'weighted_ensemble':
            strategy = EnsembleContrastiveStrategy(strategy_config)
        elif strategy_type == 'adaptive':
            # TODO: Implement adaptive strategy
            raise NotImplementedError("Adaptive strategy not yet implemented")
        else:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")

        self.current_strategy = strategy
        self.strategy_history.append(strategy)

        return strategy

    def get_current_strategy(self) -> Optional[ContrastiveStrategy]:
        """Get the current active strategy."""
        return self.current_strategy

    def set_strategy(self, strategy: ContrastiveStrategy) -> None:
        """Set the current strategy."""
        self.current_strategy = strategy
        self.strategy_history.append(strategy)

    def compute_loss(
        self,
        features: torch.Tensor,
        projections: torch.Tensor,
        prompts: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        system_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss using current strategy.

        Args:
            features: Backbone features
            projections: Projected features
            prompts: HSE prompt vectors
            labels: Ground truth labels
            system_ids: System identifiers

        Returns:
            Dictionary containing loss and components
        """
        if self.current_strategy is None:
            raise ValueError("No strategy is currently set")

        return self.current_strategy.compute_loss(
            features, projections, prompts, labels, system_ids
        )

    @property
    def requires_multiple_views(self) -> bool:
        """Check if current strategy requires multiple views."""
        return self.current_strategy.requires_multiple_views if self.current_strategy else False

    @property
    def requires_prompts(self) -> bool:
        """Check if current strategy requires prompts."""
        return self.current_strategy.requires_prompts if self.current_strategy else False

    @property
    def requires_labels(self) -> bool:
        """Check if current strategy requires labels."""
        return self.current_strategy.requires_labels if self.current_strategy else False


# Factory function for strategy creation
def create_contrastive_strategy(strategy_config: Dict[str, Any]) -> ContrastiveStrategyManager:
    """
    Factory function to create contrastive strategy manager.

    Args:
        strategy_config: Strategy configuration

    Returns:
        Configured ContrastiveStrategyManager
    """
    manager = ContrastiveStrategyManager()
    manager.create_strategy(strategy_config)
    return manager


# Export main classes
__all__ = [
    'ContrastiveStrategy',
    'SingleContrastiveStrategy',
    'EnsembleContrastiveStrategy',
    'ContrastiveStrategyManager',
    'create_contrastive_strategy'
]