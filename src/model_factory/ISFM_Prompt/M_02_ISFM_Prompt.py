"""
M_02_ISFM_Prompt: Simplified Prompt-guided Industrial Signal Foundation Model

This model implements a simplified version of prompt-guided industrial signal processing
with HSE (Heterogeneous Signal Embedding) and lightweight system-specific learnable prompts.

Key Features:
- Heterogeneous Signal Embedding with system prompts
- Simple Dataset_id → learnable prompt mapping
- Direct signal + prompt combination (add/concat)
- Two-stage training support (pretrain/finetune)
- Full backward compatibility with non-prompt modes
- Integration with existing PHM-Vibench components

Simplified from original complex design:
- Removed complex prompt library and selector
- Removed multi-level prompt encoding
- Kept core HSE + prompt functionality
- Lightweight and easy to understand

Author: PHM-Vibench Team
Date: 2025-01-23
License: MIT
"""

from importlib import import_module
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.model_factory.ISFM.system_utils import resolve_batch_metadata

# Config-facing component IDs resolve to one exact module and symbol. Loading only
# the selected component avoids importing unrelated optional backbones.
PromptEmbedding_dict = {
    'HSE_prompt': (
        'src.model_factory.ISFM_Prompt.embedding.HSE_prompt',
        'HSE_prompt',
    ),
    'E_01_HSE': (
        'src.model_factory.ISFM.embedding.E_01_HSE',
        'E_01_HSE',
    ),
    'E_01_HSE_v2': (
        'src.model_factory.ISFM_Prompt.embedding.E_01_HSE_v2',
        'E_01_HSE_v2',
    ),
}

PromptBackbone_dict = {
    name: (f'src.model_factory.ISFM.backbone.{name}', name)
    for name in (
        'B_01_basic_transformer',
        'B_04_Dlinear',
        'B_05_Mamba',
        'B_06_TimesNet',
        'B_08_PatchTST',
        'B_09_FNO',
        'B_11_MomentumEncoder',
    )
}

PromptTaskHead_dict = {
    name: (f'src.model_factory.ISFM.task_head.{name}', name)
    for name in (
        'H_01_Linear_cla',
        'H_02_distance_cla',
        'H_03_Linear_pred',
        'H_09_multiple_task',
        'H_10_ProjectionHead',
    )
}


def _required_component_id(args_m: Any, field: str) -> str:
    component_id = getattr(args_m, field, None)
    if not isinstance(component_id, str) or not component_id.strip():
        raise ValueError(f"model.{field} must be explicitly configured")
    return component_id


def _load_component(
    components: dict[str, tuple[str, str]],
    component_id: str,
    kind: str,
) -> Any:
    try:
        module_path, symbol = components[component_id]
    except KeyError as exc:
        available = ", ".join(sorted(components))
        raise ValueError(
            f"Unknown ISFM_Prompt {kind} {component_id!r}. "
            f"Available values: {available}"
        ) from exc

    module = import_module(module_path)
    return getattr(module, symbol)


class Model(nn.Module):
    """
    Simplified Prompt-guided Industrial Signal Foundation Model (M_02_ISFM_Prompt).

    This model integrates lightweight system-specific learnable prompts with heterogeneous
    signal embedding for enhanced cross-system generalization in industrial fault diagnosis.

    Simplified Architecture:
    1. HSE_prompt: Process heterogeneous signals with system prompts
    2. Backbone Network: Process embeddings through transformer/CNN architectures
    3. Task Head: Generate task-specific outputs (classification/prediction)

    Key Simplifications:
    - Removed complex prompt library and selector
    - Simplified to Dataset_id → learnable prompt mapping
    - Direct signal + prompt combination (add/concat)
    - Lightweight and easy to understand
    """
    
    def __init__(self, args_m, metadata=None):
        """
        Initialize simplified M_02_ISFM_Prompt model.

        Args:
            args_m: Configuration object with model parameters
                Required attributes:
                - embedding: Embedding layer type (e.g., 'HSE_prompt')
                - backbone: Backbone network type (e.g., 'B_08_PatchTST')
                - task_head: Task head type (e.g., 'H_01_Linear_cla')

                Optional prompt-related attributes:
                - use_prompt: Enable prompt functionality (default: True)
                - training_stage: Training stage ('pretrain'/'finetune', default: 'pretrain')

            metadata: Dataset metadata accessor for system information lookup
        """
        super().__init__()

        if metadata is None:
            raise ValueError(
                "M_02_ISFM_Prompt requires metadata with Dataset_id and "
                "Sample_rate for every file_id"
            )

        self.metadata = metadata
        self.args_m = args_m

        # Simplified configuration
        self.use_prompt = getattr(args_m, 'use_prompt', True)
        self.training_stage = getattr(args_m, 'training_stage', 'pretrain')
        self.freeze_prompt = getattr(args_m, 'freeze_prompt', False)
        
        embedding_id = _required_component_id(args_m, 'embedding')
        backbone_id = _required_component_id(args_m, 'backbone')
        task_head_id = _required_component_id(args_m, 'task_head')

        embedding_cls = _load_component(
            PromptEmbedding_dict, embedding_id, 'embedding'
        )
        backbone_cls = _load_component(PromptBackbone_dict, backbone_id, 'backbone')
        task_head_cls = _load_component(
            PromptTaskHead_dict, task_head_id, 'task head'
        )

        self.embedding = embedding_cls(args_m)
        self.backbone = backbone_cls(args_m)
        
        # Get number of classes from metadata (following M_01_ISFM pattern)
        # self.num_classes = get_num_classes(self.metadata)  # Simplified: use config value
        # args_m.num_classes = self.num_classes
        
        self.task_head = task_head_cls(args_m)
        
        # Simplified: No complex prompt components
        self.last_prompt_vector: Optional[torch.Tensor] = None

        # Set training stage
        self.set_training_stage(self.training_stage)
    
    # def get_num_classes(self):
    #     """
    #     Extract number of classes per dataset from metadata (following M_01_ISFM pattern).

    #     Returns:
    #         Dictionary mapping dataset IDs to number of classes
    #     """
    #     if self.metadata is None:
    #         # Fallback for testing scenarios
    #         return {0: 10}  # Default single dataset with 10 classes, keep integer key

    #     return get_num_classes(self.metadata)
    
    def set_training_stage(self, stage: str):
        """
        Set training stage and configure prompt freezing.

        Args:
            stage: Training stage ('pretrain'/'pretraining' or 'finetune')
        """
        # Normalize stage name for consistency
        stage = stage.lower()
        if stage in {"pretraining", "pretrain"}:
            stage = "pretrain"
        elif stage in {"finetuning", "finetune"}:
            stage = "finetune"

        self.training_stage = stage

        # For simplified version, HSE_prompt handles its own prompt freezing
        if hasattr(self.embedding, 'set_training_stage'):
            self.embedding.set_training_stage(stage)

    def _resolve_metadata(
        self,
        x: torch.Tensor,
        file_id: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if file_id is None:
            raise ValueError(
                "file_id is required to resolve Dataset_id and Sample_rate metadata"
            )

        system_ids, sample_rates = resolve_batch_metadata(
            self.metadata, file_id_batch=file_id, device=x.device
        )
        batch_size = x.shape[0]
        if system_ids.numel() == 1:
            system_ids = system_ids.expand(batch_size)
        elif system_ids.numel() != batch_size:
            raise ValueError(
                "file_id metadata must resolve to one system ID or one ID per "
                f"sample; got {system_ids.numel()} for batch_size={batch_size}"
            )
        if sample_rates.numel() == 1:
            sample_rates = sample_rates.expand(batch_size)
        elif sample_rates.numel() != batch_size:
            raise ValueError(
                "file_id metadata must resolve to one sampling frequency or one "
                f"per sample; got {sample_rates.numel()} for batch_size={batch_size}"
            )
        return system_ids, sample_rates
    
    def _embed(self, x: torch.Tensor, file_id: Optional[Any] = None) -> torch.Tensor:
        """
        Signal embedding stage with simplified prompt integration.

        Args:
            x: Input signal tensor (B, L, C)
            file_id: File identifier for metadata lookup

        Returns:
            Embedded signal tensor (B, num_patches, signal_dim)
        """
        system_ids, sample_rates = self._resolve_metadata(x, file_id)
        if self.args_m.embedding == 'HSE_prompt':
            return self.embedding(x, sample_rates, system_ids)
        if self.args_m.embedding == 'E_01_HSE':
            return self.embedding(x, sample_rates)
        if self.args_m.embedding == 'E_01_HSE_v2':
            raise ValueError(
                "E_01_HSE_v2 requires explicit Domain_id metadata and is not "
                "supported by the simplified M_02_ISFM_Prompt contract; use "
                "HSE_prompt or a model that declares the full metadata interface"
            )
        raise ValueError(f"Unsupported embedding {self.args_m.embedding!r}")

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Backbone encoding stage.
        
        Args:
            x: Input features (B, num_patches, feature_dim)
            
        Returns:
            Encoded features from backbone network
        """
        return self.backbone(x)
    
    def _head(self, 
             x: torch.Tensor, 
             file_id: Optional[Any] = None, 
             task_id: Optional[str] = None, 
             return_feature: bool = False) -> torch.Tensor:
        """
        Task head stage (following M_01_ISFM pattern).
        
        Args:
            x: Encoded features
            file_id: File identifier for system information
            task_id: Task type identifier
            return_feature: Return features instead of final outputs
            
        Returns:
            Task-specific outputs or features
        """
        system_ids, _ = self._resolve_metadata(x, file_id)
        
        if task_id == 'classification':
            return self.task_head(
                x,
                system_id=system_ids,
                return_feature=return_feature,
                task_id=task_id,
            )
        elif task_id == 'prediction':
            shape = (self.shape[1], self.shape[2]) if len(self.shape) > 2 else (self.shape[1],)
            return self.task_head(x, return_feature=return_feature, task_id=task_id, shape=shape)
        return self.task_head(
            x,
            system_id=system_ids,
            return_feature=return_feature,
            task_id=task_id,
        )
    
    def forward(self,
                x: torch.Tensor,
                file_id: Optional[Any] = None,
                task_id: Optional[str] = None,
                return_feature: bool = False) -> torch.Tensor:
        """
        Simplified forward pass through M_02_ISFM_Prompt model.

        Args:
            x: Input signal tensor (B, L, C)
            file_id: File identifier for metadata lookup
            task_id: Task type ('classification', 'prediction', etc.)
            return_feature: Return intermediate features instead of final outputs

        Returns:
            Model output tensor or (output, features) if return_feature=True
        """
        self.shape = x.shape  # Store for prediction tasks

        # Stage 1: Signal embedding with simplified prompt integration
        signal_emb = self._embed(x, file_id)

        # Stage 2: Backbone encoding
        encoded_features = self._encode(signal_emb)

        # Stage 3: Task-specific head
        task_output = self._head(encoded_features, file_id, task_id, return_feature)

        # Return based on requirements
        if return_feature:
            # 检查task_head是否返回了tuple，避免嵌套构造
            if isinstance(task_output, tuple):
                task_logits, task_features = task_output
                # 使用backbone特征而不是task_features，避免嵌套
                if encoded_features.ndim > 2:
                    backbone_features = encoded_features.mean(dim=1)
                else:
                    backbone_features = encoded_features
                return task_logits, backbone_features
            else:
                # task_head不支持return_feature，构造特征
                if encoded_features.ndim > 2:
                    backbone_features = encoded_features.mean(dim=1)
                else:
                    backbone_features = encoded_features
                return task_output, backbone_features

        return task_output
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get simplified model information.

        Returns:
            Dictionary with model configuration and statistics
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = {
            'model_name': 'M_02_ISFM_Prompt_Simplified',
            'use_prompt': self.use_prompt,
            'training_stage': self.training_stage,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'components': {
                'embedding': self.args_m.embedding,
                'backbone': getattr(self.args_m, 'backbone', 'None'),
                'task_head': getattr(self.args_m, 'task_head', 'None')
            }
        }

        # Add embedding-specific info for HSE_prompt
        if self.args_m.embedding == 'HSE_prompt' and hasattr(self.embedding, 'get_model_info'):
            embedding_info = self.embedding.get_model_info()
            info['prompt_config'] = {
                'prompt_dim': embedding_info.get('prompt_dim', 'unknown'),
                'max_dataset_ids': embedding_info.get('max_dataset_ids', 'unknown'),
                'prompt_combination': embedding_info.get('prompt_combination', 'unknown'),
                'prompt_parameters': embedding_info.get('prompt_parameters', 0)
            }

        return info


# For backward compatibility and factory registration
def create_model(args_m, metadata=None):
    """Factory function to create M_02_ISFM_Prompt model."""
    return Model(args_m, metadata)
