"""Prompt-guided industrial signal model used by the P08 method.

The HSE_prompt path uses physical-duration tokens, explicit Nyquist-aware
shared/private bands, and continuous acquisition metadata.  Dataset identity is
not passed to the embedding and the P08 configuration uses one unified head.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union

# Import existing PHM-Vibench components for reuse
from src.model_factory.ISFM.embedding import *
from src.model_factory.ISFM.backbone import *
from src.model_factory.ISFM.task_head import *
from src.model_factory.ISFM.system_utils import resolve_batch_metadata

# Import simplified prompt components
from .embedding.HSE_prompt import HSE_prompt
from .embedding.E_01_HSE_v2 import E_01_HSE_v2


# Define available components for the simplified Prompt-guided ISFM
PromptEmbedding_dict = {
    'HSE_prompt': HSE_prompt,
    'E_01_HSE': E_01_HSE,                       # Fallback to original HSE
    'E_01_HSE_v2': E_01_HSE_v2,                 # Enhanced HSE with prompt support
}

# Reuse existing backbones - they work with any embedding output
PromptBackbone_dict = {
    'B_01_basic_transformer': B_01_basic_transformer,
    'B_04_Dlinear': B_04_Dlinear,
    'B_05_Mamba': B_05_Mamba,
    'B_06_TimesNet': B_06_TimesNet,
    'B_08_PatchTST': B_08_PatchTST,            # Recommended for Prompt fusion
    'B_09_FNO': B_09_FNO,
    'B_11_MomentumEncoder': B_11_MomentumEncoder,  # For contrastive learning
}

# Reuse existing task heads + add contrastive learning projection head
PromptTaskHead_dict = {
    'H_01_Linear_cla': H_01_Linear_cla,         # Standard classification
    'H_02_distance_cla': H_02_distance_cla,     # Distance-based classification
    'H_03_Linear_pred': H_03_Linear_pred,       # Prediction head
    'H_09_multiple_task': H_09_multiple_task,   # Multi-task head
    'H_10_ProjectionHead': H_10_ProjectionHead,  # Contrastive learning projection
    'H_11_Unified_cla': H_11_Unified_cla,       # Dataset-identity-free classification
}


class Model(nn.Module):
    """
    Simplified Prompt-guided Industrial Signal Foundation Model (M_02_ISFM_Prompt).

    This model integrates continuous physical-metadata prompts with heterogeneous
    signal embedding for cross-system industrial fault diagnosis.

    Simplified Architecture:
    1. HSE_prompt: Process fixed-duration signal support and observable bands
    2. Backbone Network: Process embeddings through transformer/CNN architectures
    3. Task Head: Generate task-specific outputs (classification/prediction)

    For the P08 path, sampling rate is looked up from metadata while the returned
    dataset ID is discarded.  Classification uses ``H_11_Unified_cla`` so no
    system selector enters the decision path.
    """
    
    def __init__(self, args_m, metadata=None):
        """
        Initialize simplified M_02_ISFM_Prompt model.

        Args:
            args_m: Configuration object with model parameters
                Required attributes:
                - embedding: Embedding layer type (e.g., 'HSE_prompt')
                - backbone: Backbone network type (e.g., 'B_08_PatchTST')
                - task_head: Task head type (P08 uses 'H_11_Unified_cla')

                Optional prompt-related attributes:
                - use_prompt: Enable prompt functionality (default: True)
                - training_stage: Training stage ('pretrain'/'finetune', default: 'pretrain')

            metadata: Dataset metadata accessor for system information lookup
        """
        super().__init__()

        self.metadata = metadata
        self.args_m = args_m

        # Simplified configuration
        self.use_prompt = getattr(args_m, 'use_prompt', True)
        self.training_stage = getattr(args_m, 'training_stage', 'pretrain')
        self.freeze_prompt = getattr(args_m, 'freeze_prompt', False)
        
        # Initialize core ISFM components following PHM-Vibench pattern
        self.embedding = PromptEmbedding_dict[args_m.embedding](args_m)
        
        # Initialize backbone (works with any embedding output)
        if hasattr(args_m, 'backbone') and args_m.backbone:
            self.backbone = PromptBackbone_dict[args_m.backbone](args_m)
        else:
            self.backbone = nn.Identity()
        
        # Get number of classes from metadata (following M_01_ISFM pattern)
        # self.num_classes = get_num_classes(self.metadata)  # Simplified: use config value
        # args_m.num_classes = self.num_classes
        
        # Initialize task head
        if hasattr(args_m, 'task_head') and args_m.task_head:
            self.task_head = PromptTaskHead_dict[args_m.task_head](args_m)
        else:
            self.task_head = nn.Identity()
        
        # No separate prompt library or dataset-specific prompt state.
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

    def _normalize_single_file_id(self, file_id: Optional[Any]) -> Optional[Any]:
        """
        将批量或张量形式的 file_id 归一化为单个标量 key，
        用于 metadata 索引。

        支持:
        - list/tuple: 取第一个元素（Same_system_Sampler 下 batch 内同一系统）;
        - torch.Tensor: 取第一个元素并转为 Python 标量（避免 CUDA→NumPy 错误）;
        - 其他标量类型: 原样返回。
        """
        if file_id is None:
            return None

        import torch

        # 批量场景：DataLoader 默认会把 file_id 聚合为 list
        if isinstance(file_id, (list, tuple)):
            if not file_id:
                return None
            fid0 = file_id[0]
            if isinstance(fid0, torch.Tensor):
                return fid0.view(-1)[0].item()
            return fid0

        # 单个张量 ID（可能已被 Lightning 移到 CUDA）
        if isinstance(file_id, torch.Tensor):
            return file_id.view(-1)[0].item()

        # 其他标量（int/str 等）
        return file_id
    
    def _embed(
        self,
        x: torch.Tensor,
        file_id: Optional[Any] = None,
        sampling_rate_hz: Optional[Union[torch.Tensor, list, tuple]] = None,
    ) -> torch.Tensor:
        """
        Signal embedding stage with simplified prompt integration.

        Args:
            x: Input signal tensor (B, L, C)
            file_id: File identifier for metadata lookup

        Returns:
            Embedded signal tensor (B, num_patches, signal_dim)
        """
        if self.args_m.embedding == 'HSE_prompt':
            if sampling_rate_hz is not None:
                sample_rates = torch.as_tensor(
                    sampling_rate_hz, device=x.device, dtype=x.dtype
                ).reshape(-1)
                if sample_rates.numel() != x.shape[0]:
                    raise ValueError(
                        "explicit sampling_rate_hz length must equal batch size; "
                        f"got {sample_rates.numel()} for batch {x.shape[0]}"
                    )
                signal_emb = self.embedding(x, sample_rates)
            elif file_id is not None and self.metadata is not None:
                # Resolve the measurable sampling rate only.  The dataset-ID
                # return value is deliberately discarded.
                _, sample_rates = resolve_batch_metadata(
                    self.metadata, file_id_batch=file_id, device=x.device
                )
                signal_emb = self.embedding(x, sample_rates)
            else:
                default_fs = getattr(self.args_m, 'default_sampling_rate_hz', None)
                if default_fs is None:
                    raise ValueError(
                        "HSE_prompt requires per-sample sampling-rate metadata; "
                        "set default_sampling_rate_hz only for an explicit synthetic test"
                    )
                signal_emb = self.embedding(x, float(default_fs))

        elif self.args_m.embedding == 'E_01_HSE':
            # Traditional HSE embeddings need sampling frequency
            if file_id is not None and self.metadata is not None:
                _, sample_rates = resolve_batch_metadata(
                    self.metadata, file_id_batch=file_id, device=x.device
                )
                fs = sample_rates  # [B]
            else:
                fs = 1000.0  # Default sampling frequency

            signal_emb = self.embedding(x, fs)

        else:
            # Other embedding types
            signal_emb = self.embedding(x)

        return signal_emb

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
        unified_head = getattr(self.args_m, 'task_head', None) == 'H_11_Unified_cla'
        if unified_head:
            system_id = None
        elif file_id is not None and self.metadata is not None:
            # 使用统一的批量解析逻辑，避免 file_id 为 CUDA tensor 时触发 NumPy 错误
            system_ids_tensor, _ = resolve_batch_metadata(
                self.metadata, file_id_batch=file_id, device=x.device
            )
            # 当前实现假设 Same_system_Sampler 保证单系统 per batch，取第一个 system_id
            system_id = int(system_ids_tensor[0].item())
        else:
            # Use a valid default system_id from common target systems
            system_id = 1  # Default to CWRU (system_id 1) instead of 0
        
        if task_id == 'classification':
            return self.task_head(x, system_id=system_id, return_feature=return_feature, task_id=task_id)
        elif task_id == 'prediction':
            shape = (self.shape[1], self.shape[2]) if len(self.shape) > 2 else (self.shape[1],)
            return self.task_head(x, return_feature=return_feature, task_id=task_id, shape=shape)
        else:
            # Default behavior for other task types
            if hasattr(self.task_head, 'forward'):
                try:
                    return self.task_head(x, system_id=system_id, return_feature=return_feature, task_id=task_id)
                except TypeError:
                    # Fallback if task head doesn't support all arguments
                    return self.task_head(x)
            else:
                return x
    
    def forward(self,
                x: torch.Tensor,
                file_id: Optional[Any] = None,
                task_id: Optional[str] = None,
                return_feature: bool = False,
                sampling_rate_hz: Optional[Union[torch.Tensor, list, tuple]] = None,
                ) -> torch.Tensor:
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
        signal_emb = self._embed(
            x, file_id=file_id, sampling_rate_hz=sampling_rate_hz
        )

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
            'model_name': 'M_02_ISFM_Prompt_PhysicalNyquist',
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
                'prompt_combination': embedding_info.get('prompt_combination', 'unknown'),
                'prompt_parameters': embedding_info.get('prompt_parameters', 0),
                'prompt_features': embedding_info.get('prompt_features', []),
                'dataset_identity_consumed': embedding_info.get(
                    'dataset_identity_consumed', True
                ),
                'physical_patch_duration_s': embedding_info.get(
                    'physical_patch_duration_s'
                ),
                'shared_band_hz': embedding_info.get('shared_band_hz'),
            }

        return info


# For backward compatibility and factory registration
def create_model(args_m, metadata=None):
    """Factory function to create M_02_ISFM_Prompt model."""
    return Model(args_m, metadata)
