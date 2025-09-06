"""
Multi-task PHM implementation for In_distribution tasks.

Refactored from multi_task_lightning.py to follow standard task factory patterns.
Inherits from Default_task to reuse optimizer, scheduler, and logging infrastructure.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any
import pytorch_lightning as pl

from ...Components.loss import get_loss_fn


# @register_task("In_distribution", "multi_task_phm")  
class task(pl.LightningModule):
    """
    Multi-task PHM implementation for In_distribution tasks.
    
    Supports simultaneous training on:
    - Fault classification
    - Anomaly detection
    - Signal prediction
    - RUL prediction
    
    Implements its own initialization to avoid Default_task's single-loss dependency.
    """
    
    def __init__(
        self,
        network: nn.Module,
        args_data: Any,
        args_model: Any,
        args_task: Any,
        args_trainer: Any,
        args_environment: Any,
        metadata: Any
    ):
        # Initialize LightningModule directly (skip Default_task's single-loss logic)
        super().__init__()
        
        # Set required attributes (copied from Default_task logic)
        self.network = network.cuda() if args_trainer.gpus else network
        self.args_task = args_task
        self.args_model = args_model
        self.args_data = args_data
        self.args_trainer = args_trainer
        self.args_environment = args_environment
        self.metadata = metadata
        
        # Multi-task specific initialization (no single loss_fn needed)
        self.enabled_tasks = self._get_enabled_tasks()
        self.task_weights = self._get_task_weights()
        self.task_loss_fns = self._initialize_task_losses()
        
        # Save hyperparameters (copied from Default_task)
        hparams_dict = {
            **vars(self.args_task),
            **vars(self.args_model),
            **vars(self.args_data),
            **vars(self.args_trainer),
            **vars(self.args_environment),
        }
        self.save_hyperparameters(hparams_dict, ignore=['network', 'metadata'])
        
    def _get_enabled_tasks(self) -> List[str]:
        """Get enabled tasks from configuration."""
        default_tasks = ['classification', 'anomaly_detection', 
                        'signal_prediction', 'rul_prediction']
        return getattr(self.args_task, 'enabled_tasks', default_tasks)
    
    def _get_task_weights(self) -> Dict[str, float]:
        """Get task weights for loss balancing."""
        default_weights = {
            'classification': 1.0,
            'anomaly_detection': 0.6,
            'signal_prediction': 0.7,
            'rul_prediction': 0.8
        }
        config_weights = getattr(self.args_task, 'task_weights', {})
        # Handle both dict and Namespace for config_weights
        if hasattr(config_weights, '__dict__'):
            config_weights = vars(config_weights)
        
        # Only use weights for enabled tasks
        weights = {}
        for task in self.enabled_tasks:
            if isinstance(config_weights, dict):
                weight = config_weights.get(task, default_weights.get(task, 1.0))
            else:
                weight = getattr(config_weights, task, default_weights.get(task, 1.0))
            weights[task] = weight
        return weights
    
    def _initialize_task_losses(self) -> Dict[str, nn.Module]:
        """Initialize loss functions for each enabled task."""
        loss_mapping = {
            'classification': 'CE',
            'anomaly_detection': 'BCE', 
            'signal_prediction': 'MSE',
            'rul_prediction': 'MSE'
        }
        
        task_losses = {}
        for task in self.enabled_tasks:
            loss_name = loss_mapping.get(task, 'MSE')
            task_losses[task] = get_loss_fn(loss_name)
        
        return task_losses
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler (copied from Default_task)."""
        optimizer_name = getattr(self.args_task, 'optimizer', 'adam').lower()
        lr = getattr(self.args_task, 'lr', 1e-3)
        weight_decay = getattr(self.args_task, 'weight_decay', 0.0)

        # Choose optimizer
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = getattr(self.args_task, 'momentum', 0.9)
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        # Simple scheduler configuration
        scheduler_config = getattr(self.args_task, 'scheduler', None)
        if scheduler_config and isinstance(scheduler_config, dict) and scheduler_config.get('name'):
            scheduler_name = scheduler_config['name'].lower()
            if scheduler_name == 'cosine':
                T_max = scheduler_config.get('options', {}).get('T_max', 100)
                eta_min = scheduler_config.get('options', {}).get('eta_min', 1e-6)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
                return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        
        return optimizer
    
    def _build_task_labels(self, y, metadata):
        """Build task-specific labels from single label and metadata."""
        y_dict = {}
        
        # Classification task - use original label
        if 'classification' in self.enabled_tasks:
            y_dict['classification'] = y
        
        # Anomaly detection - convert to binary (0=normal, 1=anomaly) 
        if 'anomaly_detection' in self.enabled_tasks:
            # Assume label 0 is normal, others are anomaly
            y_dict['anomaly_detection'] = (y > 0).float()
        
        # RUL prediction - get from metadata if available
        if 'rul_prediction' in self.enabled_tasks:
            rul_value = metadata.get('RUL_label', 0)
            if isinstance(rul_value, (int, float)):
                y_dict['rul_prediction'] = torch.tensor(rul_value, dtype=torch.float32).to(y.device)
            else:
                # Skip this task if RUL data not available
                pass
        
        # Signal prediction - no label needed (reconstruction task)
        if 'signal_prediction' in self.enabled_tasks:
            # Use the input signal as target for reconstruction
            y_dict['signal_prediction'] = None  # Will be handled in _compute_task_loss
            
        return y_dict
    
    def _shared_step(self, batch, batch_idx, mode='train'):
        """Shared logic for training/validation/testing steps."""
        # batch_idx is kept for PyTorch Lightning interface compatibility
        _ = batch_idx  # Acknowledge parameter for linting
        
        # Extract data from batch dict (correct format from IdIncludedDataset)
        x = batch['x']
        y = batch['y']
        file_id = batch['file_id'][0].item()
        
        # Get metadata for this sample
        metadata = self.metadata[file_id]
        
        # Single forward pass with enabled tasks list
        outputs = self.network(x, file_id, task_id=self.enabled_tasks)
        
        # Build task-specific labels from metadata
        y_dict = self._build_task_labels(y, metadata)
        
        # Compute loss for each enabled task
        total_loss = 0.0
        
        for task_name in self.enabled_tasks:
            if task_name in y_dict:
                try:
                    task_loss = self._compute_task_loss(task_name, outputs, y_dict[task_name], x)
                    if task_loss is not None:
                        weighted_loss = self.task_weights[task_name] * task_loss
                        total_loss += weighted_loss
                        
                        # Log individual task loss with mode-specific parameters
                        on_step = (mode == 'train')
                        self.log(f'{mode}_{task_name}_loss', task_loss, 
                                on_step=on_step, on_epoch=True)
                except Exception as e:
                    # Continue if individual task fails
                    print(f'WARNING: {task_name} {mode} failed: {e}')
                    continue
        
        # Log total loss with mode-specific parameters
        on_step = (mode == 'train')
        self.log(f'{mode}_loss', total_loss, on_step=on_step, on_epoch=True)
        return total_loss
    
    def training_step(self, batch, batch_idx):
        """Training step using shared logic."""
        return self._shared_step(batch, batch_idx, mode='train')
    
    def validation_step(self, batch, batch_idx):
        """Validation step using shared logic."""
        return self._shared_step(batch, batch_idx, mode='val')
    
    def test_step(self, batch, batch_idx):
        """Test step using shared logic."""
        return self._shared_step(batch, batch_idx, mode='test')
    
    def _compute_task_loss(self, task_name: str, outputs: Any, targets: torch.Tensor, x: torch.Tensor = None) -> torch.Tensor:
        """Compute loss for a specific task."""
        loss_fn = self.task_loss_fns[task_name]
        
        # Handle both dictionary and object-style outputs
        if isinstance(outputs, dict):
            # Dictionary format from MultiTaskHead
            task_output = outputs.get(task_name, None)
            if task_output is None:
                print(f'WARNING: No output found for task {task_name} in outputs: {list(outputs.keys())}')
                return None
        else:
            # Object/attribute format - try to get specific task output
            task_output = getattr(outputs, task_name, 
                                getattr(outputs, f'{task_name}_logits', outputs))
        
        if task_name == 'classification':
            return loss_fn(task_output, targets.long())
        
        elif task_name == 'anomaly_detection':
            # Binary classification for anomaly detection
            return loss_fn(task_output, targets.float())
        
        elif task_name == 'signal_prediction':
            # Signal prediction task - use original signal as target
            if targets is None and x is not None:
                # For reconstruction tasks, use input as target
                targets = x
            elif targets is None:
                # Skip this task if no target available
                return None
            return loss_fn(task_output, targets.float())
        
        elif task_name == 'rul_prediction':
            # RUL regression task
            return loss_fn(task_output.squeeze(), targets.float())
        
        else:
            # Default: use task output directly
            return loss_fn(task_output, targets)


# Export for task factory compatibility
# task = task  # Already defined as class name