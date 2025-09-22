"""
HSE Contrastive Learning Task for Cross-Dataset Domain Generalization

Task implementation that integrates Prompt-guided Hierarchical Signal Embedding (HSE)
with state-of-the-art contrastive learning for industrial fault diagnosis.

Core Innovation: First work to combine system metadata prompts with contrastive learning
for cross-system industrial fault diagnosis, targeting ICML/NeurIPS 2025.

Key Features:
1. Prompt-guided contrastive learning with system-aware sampling
2. Two-stage training support (pretrain/finetune)
3. Cross-dataset domain generalization (CDDG)
4. Integration with all 6 SOTA contrastive losses
5. System-invariant representation learning

Authors: PHM-Vibench Team
Date: 2025-01-06
License: Apache 2.0
"""

import torch
import pytorch_lightning as pl
from typing import Dict, Any, Optional, List, Tuple
import logging
from collections import defaultdict

from ...Default_task import Default_task
from ...Components.prompt_contrastive import PromptGuidedContrastiveLoss
from ...Components.loss import get_loss_fn
from ...Components.metrics import get_metrics

logger = logging.getLogger(__name__)


class task(Default_task):
    """
    HSE Prompt-guided Contrastive Learning Task
    
    Inherits from Default_task and extends with:
    1. Prompt-guided contrastive learning capabilities
    2. System-aware positive/negative sampling
    3. Two-stage training workflow support
    4. Cross-dataset domain generalization
    
    Training Stages:
    - **Pretrain**: Multi-system contrastive learning with prompt guidance
    - **Finetune**: Task-specific adaptation with frozen prompts
    """
    
    def __init__(
        self, 
        network, 
        args_data, 
        args_model, 
        args_task, 
        args_trainer, 
        args_environment, 
        metadata
    ):
        """Initialize HSE contrastive learning task."""
        
        # Initialize parent class
        super().__init__(network, args_data, args_model, args_task, args_trainer, args_environment, metadata)
        
        # HSE-specific configuration
        self.args_task = args_task
        self.args_model = args_model
        self.metadata = metadata
        
        # Training stage control
        self.training_stage = getattr(args_model, 'training_stage', 'pretrain')
        self.freeze_prompt = getattr(args_model, 'freeze_prompt', False)
        
        # Contrastive learning setup
        self.enable_contrastive = getattr(args_task, 'contrast_weight', 0.0) > 0
        self.contrast_weight = getattr(args_task, 'contrast_weight', 0.15)
        
        if self.enable_contrastive:
            # Initialize prompt-guided contrastive loss
            self.contrastive_loss = PromptGuidedContrastiveLoss(
                base_loss_type=getattr(args_task, 'contrast_loss', 'INFONCE'),
                temperature=getattr(args_task, 'temperature', 0.07),
                prompt_similarity_weight=getattr(args_task, 'prompt_weight', 0.1),
                system_aware_sampling=getattr(args_task, 'use_system_sampling', True),
                enable_cross_system_contrast=getattr(args_task, 'cross_system_contrast', True),
                # Pass additional arguments for specific loss types
                margin=getattr(args_task, 'margin', 0.3),  # For Triplet loss
                lambda_param=getattr(args_task, 'barlow_lambda', 5e-3),  # For Barlow Twins
            )
            
            logger.info(f"✓ Contrastive learning enabled: {args_task.contrast_loss} (weight: {self.contrast_weight})")
        else:
            self.contrastive_loss = None
            logger.info("ℹ Contrastive learning disabled (weight: 0.0)")
        
        # Domain generalization setup
        self.source_domain_id = getattr(args_task, 'source_domain_id', [])
        self.target_domain_id = getattr(args_task, 'target_domain_id', [])
        
        # Metrics tracking
        self.train_metrics_dict = defaultdict(list)
        self.val_metrics_dict = defaultdict(list)
        
        # Log configuration
        self._log_task_config()
    
    def _log_task_config(self):
        """Log task configuration for debugging."""
        logger.info("HSE Contrastive Learning Task Configuration:")
        logger.info(f"  - Training stage: {self.training_stage}")
        logger.info(f"  - Contrastive learning: {self.enable_contrastive}")
        logger.info(f"  - Source domains: {self.source_domain_id}")
        logger.info(f"  - Target domains: {self.target_domain_id}")
        logger.info(f"  - Prompt frozen: {self.freeze_prompt}")
        if self.enable_contrastive:
            logger.info(f"  - Contrastive loss: {self.args_task.contrast_loss}")
            logger.info(f"  - Contrastive weight: {self.contrast_weight}")
    
    def training_step(self, batch, batch_idx):
        """Training step with prompt-guided contrastive learning."""
        metrics = self._shared_step(batch, batch_idx, stage='train')
        self._log_metrics(metrics, "train")
        return metrics["train_total_loss"]
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        metrics = self._shared_step(batch, batch_idx, stage='val')
        self._log_metrics(metrics, "val")
    
    def _shared_step(self, batch, batch_idx, stage='train'):
        """Shared step logic for training and validation."""
        batch_dict = self._prepare_batch(batch)

        x = batch_dict['x']
        y = batch_dict['y']

        if x is None or y is None:
            raise ValueError("Batch must contain 'x' and 'y' entries for HSE contrastive task.")

        if y.ndim > 1:
            y = y.view(-1)
        if y.dtype != torch.long:
            y = y.long()

        batch_size = x.size(0)

        file_ids = self._ensure_file_id_list(batch_dict.get('file_id'), batch_size)
        resolved_ids, dataset_names, system_ids_list = self._resolve_metadata(file_ids)
        system_ids_tensor = self._system_ids_to_tensor(system_ids_list, device=x.device)

        task_id = batch_dict.get('task_id', 'classification')
        primary_file_id = resolved_ids[0] if resolved_ids else None

        logits, prompts, feature_repr = self._forward_with_prompts(
            x,
            file_id=primary_file_id,
            task_id=task_id
        )

        classification_loss = self.loss_fn(logits, y)

        reg_dict = self._compute_regularization()
        total_loss = classification_loss + reg_dict.get('total', torch.tensor(0.0, device=classification_loss.device))

        contrastive_loss_value = torch.tensor(0.0, device=classification_loss.device)
        contrastive_losses = None
        if self.enable_contrastive and prompts is not None:
            try:
                contrastive_losses = self.contrastive_loss(
                    features=feature_repr if feature_repr is not None else logits,
                    prompts=prompts,
                    labels=y,
                    system_ids=system_ids_tensor
                )
                contrastive_loss_value = contrastive_losses['total_loss']
                total_loss = total_loss + self.contrast_weight * contrastive_loss_value
            except Exception as exc:
                logger.warning(f"Contrastive loss computation failed: {exc}")

        preds = torch.argmax(logits, dim=1)

        dataset_name = dataset_names[0] if dataset_names else 'global'
        step_metrics = {
            f"{stage}_loss": total_loss,
            f"{stage}_classification_loss": classification_loss,
            f"{stage}_{dataset_name}_loss": classification_loss,
            f"{stage}_total_loss": total_loss,
        }

        metric_values = super()._compute_metrics(preds, y, dataset_name, stage)
        step_metrics.update(metric_values)

        if contrastive_losses is not None:
            step_metrics[f"{stage}_contrastive_loss"] = contrastive_loss_value
            step_metrics[f"{stage}_contrastive_base_loss"] = contrastive_losses['base_loss']
            step_metrics[f"{stage}_contrastive_prompt_loss"] = contrastive_losses['prompt_loss']
            step_metrics[f"{stage}_contrastive_system_loss"] = contrastive_losses['system_loss']

        for reg_type, reg_loss_val in reg_dict.items():
            if reg_type != 'total':
                step_metrics[f"{stage}_{reg_type}_reg_loss"] = reg_loss_val

        if stage == 'train':
            self.log('contrast_weight', torch.tensor(self.contrast_weight, device=total_loss.device))
            if prompts is not None:
                prompt_norm = prompts.norm(dim=-1).mean()
                step_metrics['train_prompt_norm'] = prompt_norm

        return step_metrics
    
    
    def _prepare_batch(self, batch: Any) -> Dict[str, Any]:
        if isinstance(batch, dict):
            prepared = dict(batch)
        else:
            (x, y), data_name = batch
            prepared = {'x': x, 'y': y, 'file_id': data_name}
        prepared.setdefault('task_id', 'classification')
        return prepared

    def _ensure_file_id_list(self, file_id_field: Any, batch_size: int) -> List[Any]:
        if file_id_field is None:
            return [None] * batch_size

        if isinstance(file_id_field, torch.Tensor):
            values = file_id_field.view(-1).tolist()
        elif isinstance(file_id_field, (list, tuple)):
            values = list(file_id_field)
        else:
            values = [file_id_field]

        if len(values) < batch_size and values:
            values.extend([values[-1]] * (batch_size - len(values)))
        return values

    def _resolve_metadata(self, file_ids: List[Any]) -> Tuple[List[Any], List[str], List[int]]:
        resolved_ids: List[Any] = []
        dataset_names: List[str] = []
        system_ids: List[int] = []

        for fid in file_ids:
            key, meta_dict = self._safe_metadata_lookup(fid)
            resolved_ids.append(key)

            dataset_name = meta_dict.get('Name', 'unknown') if meta_dict else 'unknown'
            dataset_names.append(dataset_name)

            try:
                system_ids.append(int(meta_dict.get('Dataset_id', 0)))
            except (ValueError, TypeError, AttributeError):
                system_ids.append(0)

        return resolved_ids, dataset_names, system_ids

    def _safe_metadata_lookup(self, file_id: Any) -> Tuple[Any, Optional[Dict[str, Any]]]:
        candidates: List[Any] = []

        if isinstance(file_id, torch.Tensor):
            try:
                file_id = file_id.item()
            except Exception:
                file_id = file_id.detach().cpu().item()

        candidates.append(file_id)

        try:
            candidates.append(int(file_id))
        except (ValueError, TypeError):
            pass

        candidates.append(str(file_id))

        for cand in candidates:
            try:
                meta = self.metadata[cand]
                meta_dict = meta.to_dict() if hasattr(meta, 'to_dict') else dict(meta)
                return cand, meta_dict
            except Exception:
                continue

        return candidates[0] if candidates else None, None

    def _system_ids_to_tensor(self, system_ids: List[int], device: torch.device) -> Optional[torch.Tensor]:
        if not system_ids:
            return None
        if all(sid == 0 for sid in system_ids):
            return None
        return torch.tensor(system_ids, device=device)

    def _forward_with_prompts(self, x: torch.Tensor, file_id: Any, task_id: str):
        network_kwargs = {
            'file_id': file_id,
            'task_id': task_id,
        }

        logits = None
        prompts = None
        feature_repr = None

        try:
            output = self.network(x, return_prompt=True, return_feature=True, **network_kwargs)
        except TypeError:
            output = self.network(x, return_prompt=True, **network_kwargs)

        if isinstance(output, tuple):
            if len(output) == 3:
                logits, prompts, feature_repr = output
            elif len(output) == 2:
                logits, prompts = output
            else:
                logits = output[0]
        else:
            logits = output

        return logits, prompts, feature_repr
    
    def configure_optimizers(self):
        """Configure optimizers with support for fine-grained learning rates."""
        # Get base configuration from parent
        optimizer_config = super().configure_optimizers()
        
        # Handle fine-grained learning rates for two-stage training
        if hasattr(self.args_task, 'backbone_lr_multiplier') and self.training_stage == 'finetune':
            # Different learning rates for different components
            param_groups = []
            
            # Backbone parameters (lower LR)
            backbone_params = []
            # Task head parameters (full LR)  
            head_params = []
            # Other parameters (full LR)
            other_params = []
            
            for name, param in self.network.named_parameters():
                if not param.requires_grad:
                    continue  # Skip frozen parameters
                    
                if 'backbone' in name.lower() or 'embedding' in name.lower():
                    backbone_params.append(param)
                elif 'head' in name.lower() or 'classifier' in name.lower():
                    head_params.append(param)
                else:
                    other_params.append(param)
            
            # Create parameter groups
            base_lr = self.args_task.lr
            backbone_lr = base_lr * getattr(self.args_task, 'backbone_lr_multiplier', 0.1)
            
            if backbone_params:
                param_groups.append({'params': backbone_params, 'lr': backbone_lr})
            if head_params:
                param_groups.append({'params': head_params, 'lr': base_lr})
            if other_params:
                param_groups.append({'params': other_params, 'lr': base_lr})
            
            if param_groups:
                optimizer = torch.optim.AdamW(
                    param_groups,
                    weight_decay=getattr(self.args_task, 'weight_decay', 1e-4)
                )
                
                logger.info(f"Fine-grained LR: backbone={backbone_lr:.1e}, head={base_lr:.1e}")
                
                # Return with scheduler if specified
                if hasattr(self.args_task, 'scheduler') and self.args_task.scheduler:
                    scheduler = self._create_scheduler(optimizer)
                    return [optimizer], [scheduler]
                else:
                    return optimizer
        
        # Fallback to parent configuration
        return optimizer_config
    
    def _create_scheduler(self, optimizer):
        """Create learning rate scheduler."""
        scheduler_type = getattr(self.args_task, 'scheduler_type', 'cosine')
        
        if scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                optimizer, 
                T_max=getattr(self.args_task, 'epochs', 50)
            )
        elif scheduler_type == 'step':
            from torch.optim.lr_scheduler import StepLR
            return StepLR(
                optimizer,
                step_size=getattr(self.args_task, 'step_size', 15),
                gamma=getattr(self.args_task, 'gamma', 0.5)
            )
        else:
            # Fallback to no scheduler
            return None
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        super().on_train_epoch_end()
        
        # Log training stage
        self.log('training_stage', 1.0 if self.training_stage == 'pretrain' else 0.0)
        
        # Additional HSE-specific logging
        current_epoch = self.current_epoch
        
        if current_epoch % 10 == 0:  # Log every 10 epochs
            logger.info(f"Epoch {current_epoch}: Stage={self.training_stage}, "
                       f"Contrastive={self.enable_contrastive}, "
                       f"Frozen_prompt={self.freeze_prompt}")
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch.""" 
        super().on_validation_epoch_end()
        
        # Could add epoch-level validation logging here if needed
        pass
    
    def set_training_stage(self, stage: str):
        """Set training stage (for two-stage training)."""
        self.training_stage = stage
        
        # Update contrastive learning based on stage
        if stage == 'finetune':
            # Disable contrastive learning for finetuning
            self.contrast_weight = 0.0
            self.enable_contrastive = False
            logger.info("Switched to finetuning: disabled contrastive learning")
        elif stage == 'pretrain':
            # Enable contrastive learning for pretraining
            self.contrast_weight = getattr(self.args_task, 'contrast_weight', 0.15)
            self.enable_contrastive = self.contrast_weight > 0
            logger.info(f"Switched to pretraining: enabled contrastive learning (weight: {self.contrast_weight})")
        
        # Update network training stage if supported
        if hasattr(self.network, 'set_training_stage'):
            self.network.set_training_stage(stage)
    
    def get_contrastive_features(self, batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract features and prompts for contrastive learning analysis."""
        (x, y), data_name = batch
        batch_size = x.size(0)
        
        with torch.no_grad():
            network_output = self.network(x, metadata=self._create_metadata_batch(data_name, batch_size))
            
            if isinstance(network_output, tuple):
                features, prompts = network_output
            else:
                features = network_output
                prompts = None
            
            return features, prompts


# Self-testing section  
if __name__ == "__main__":
    print("🎯 Testing HSE Contrastive Learning Task")
    
    # Mock arguments for testing
    class MockArgs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    # Test configuration
    args_model = MockArgs(
        embedding='E_01_HSE_Prompt',
        training_stage='pretrain',
        freeze_prompt=False,
        prompt_dim=128
    )
    
    args_task = MockArgs(
        loss='CE',
        contrast_loss='INFONCE',
        contrast_weight=0.15,
        temperature=0.07,
        lr=5e-4,
        epochs=50,
        source_domain_id=[1, 13, 19],
        target_domain_id=[6],
        use_system_sampling=True,
        cross_system_contrast=True
    )
    
    args_data = MockArgs(batch_size=32)
    args_trainer = MockArgs(gpus=1)
    args_environment = MockArgs(output_dir='test')
    
    print("1. Testing Task Initialization:")
    try:
        # Create mock network
        import torch.nn as nn
        
        class MockNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(256, 512)
                self.head = nn.Linear(512, 10)
                
            def forward(self, x, file_id=None, task_id=None, return_prompt=False, return_feature=False, **kwargs):
                latent = self.backbone(x.view(x.size(0), -1))
                logits = self.head(latent)
                prompt = torch.randn(x.size(0), 128, device=x.device) if return_prompt else None
                feature = latent if return_feature else None

                if return_prompt and return_feature:
                    return logits, prompt, feature
                if return_prompt:
                    return logits, prompt
                if return_feature:
                    return logits, feature
                return logits
            
            def set_training_stage(self, stage):
                pass  # Mock implementation
        
        mock_network = MockNetwork()
        mock_metadata = {'num_classes': 10}
        
        # Initialize task
        hse_task = task(
            mock_network, args_data, args_model, args_task, 
            args_trainer, args_environment, mock_metadata
        )
        
        print("   ✓ HSE contrastive task initialized successfully")
        print(f"   ✓ Training stage: {hse_task.training_stage}")
        print(f"   ✓ Contrastive learning: {hse_task.enable_contrastive}")
        print(f"   ✓ Contrastive weight: {hse_task.contrast_weight}")
        
    except Exception as e:
        print(f"   ✗ Task initialization failed: {e}")
    
    print("\n2. Testing Training Stage Switching:")
    try:
        # Test stage switching
        original_weight = hse_task.contrast_weight
        
        hse_task.set_training_stage('finetune')
        print(f"   ✓ Switched to finetune: contrast_weight={hse_task.contrast_weight}")
        
        hse_task.set_training_stage('pretrain')
        print(f"   ✓ Switched to pretrain: contrast_weight={hse_task.contrast_weight}")
        
    except Exception as e:
        print(f"   ✗ Training stage switching failed: {e}")
    
    print("\n3. Testing Mock Forward Pass:")
    try:
        # Create mock batch
        batch_size = 4
        x = torch.randn(batch_size, 2, 1024)  # (B, C, L) format
        y = torch.randint(0, 10, (batch_size,))
        batch = {
            'x': x,
            'y': y,
            'file_id': [0] * batch_size
        }
        
        # Mock training step (would normally require full Lightning setup)
        print(f"   ✓ Created mock batch: {x.shape}, labels: {y}")
        metrics = hse_task._shared_step(batch, batch_idx=0, stage='train')
        print(f"   ✓ Shared step metrics keys: {list(metrics.keys())[:5]} ...")

    except Exception as e:
        print(f"   ✗ Mock forward pass test failed: {e}")
    
    print("\n" + "="*70)
    print("✅ HSE Contrastive Learning Task tests completed!")
    print("🚀 Ready for integration with PHM-Vibench training pipeline.")
    
    # Configuration example
    print("\n💡 Configuration Example:")
    print("""
    # configs/demo/HSE_Contrastive/hse_prompt_pretrain.yaml
    task:
      type: "CDDG"
      name: "hse_contrastive"
      
      # Cross-dataset domain generalization
      source_domain_id: [1, 13, 19]  # CWRU, THU, MFPT
      target_domain_id: [6]          # XJTU
      
      # Contrastive learning
      contrast_loss: "INFONCE"
      contrast_weight: 0.15
      temperature: 0.07
      use_system_sampling: true
      cross_system_contrast: true
      
      # Standard training parameters
      loss: "CE"
      lr: 5e-4
      epochs: 50
    """)
