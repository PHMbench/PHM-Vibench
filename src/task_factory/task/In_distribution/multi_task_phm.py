"""
Multi-task PHM implementation for In_distribution tasks.

Refactored from multi_task_lightning.py to follow standard task factory patterns.
Inherits from Default_task to reuse optimizer, scheduler, and logging infrastructure.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from ...task_factory import register_task
from ...Default_task import Default_task
from ...Components.loss import get_loss_fn
from ...Components.metrics import get_metrics


@register_task("In_distribution", "multi_task_phm")
class MultiTaskPHM(Default_task):
    """
    Multi-task PHM implementation inheriting Default_task infrastructure.
    
    Supports simultaneous training on:
    - Fault classification
    - Anomaly detection
    - Signal prediction
    - RUL prediction
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
        # Initialize parent class (gets optimizer, scheduler, logging)
        super().__init__(
            network, args_data, args_model, args_task, 
            args_trainer, args_environment, metadata
        )
        
        # Multi-task specific initialization
        self.enabled_tasks = self._get_enabled_tasks()
        self.task_weights = self._get_task_weights()
        self.task_loss_fns = self._initialize_task_losses()
        
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
    
    def training_step(self, batch, batch_idx):
        """Override training step for multi-task training."""
        (x, y_dict), data_name = batch
        
        # Single forward pass
        outputs = self.network(x)
        
        # Compute loss for each enabled task
        total_loss = 0.0
        task_losses = {}
        
        for task_name in self.enabled_tasks:
            if task_name in y_dict and y_dict[task_name] is not None:
                try:
                    task_loss = self._compute_task_loss(task_name, outputs, y_dict[task_name])
                    weighted_loss = self.task_weights[task_name] * task_loss
                    task_losses[task_name] = task_loss
                    total_loss += weighted_loss
                    
                    # Log individual task loss
                    self.log(f'train_{task_name}_loss', task_loss, on_step=True, on_epoch=True)
                except Exception as e:
                    # Continue training if individual task fails
                    self.log(f'WARNING: {task_name} loss computation failed: {e}')
                    continue
        
        # Log total loss
        self.log('train_loss', total_loss, on_step=True, on_epoch=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Override validation step for multi-task validation."""
        (x, y_dict), data_name = batch
        
        # Single forward pass
        outputs = self.network(x)
        
        # Compute validation loss for each enabled task
        total_val_loss = 0.0
        
        for task_name in self.enabled_tasks:
            if task_name in y_dict and y_dict[task_name] is not None:
                try:
                    task_loss = self._compute_task_loss(task_name, outputs, y_dict[task_name])
                    weighted_loss = self.task_weights[task_name] * task_loss
                    total_val_loss += weighted_loss
                    
                    # Log individual validation loss
                    self.log(f'val_{task_name}_loss', task_loss, on_step=False, on_epoch=True)
                except Exception as e:
                    self.log(f'WARNING: {task_name} validation failed: {e}')
                    continue
        
        # Log total validation loss
        self.log('val_loss', total_val_loss, on_step=False, on_epoch=True)
        return total_val_loss
    
    def _compute_task_loss(self, task_name: str, outputs: Any, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss for a specific task."""
        loss_fn = self.task_loss_fns[task_name]
        
        if task_name == 'classification':
            # Assume outputs has classification_logits or similar
            logits = getattr(outputs, 'classification_logits', outputs)
            return loss_fn(logits, targets.long())
        
        elif task_name == 'anomaly_detection':
            # Binary classification for anomaly detection
            anomaly_logits = getattr(outputs, 'anomaly_logits', outputs)
            return loss_fn(anomaly_logits, targets.float())
        
        elif task_name == 'signal_prediction':
            # Signal prediction task
            pred_outputs = getattr(outputs, 'signal_prediction', outputs)
            return loss_fn(pred_outputs, targets.float())
        
        elif task_name == 'rul_prediction':
            # RUL regression task
            rul_outputs = getattr(outputs, 'rul_prediction', outputs)
            return loss_fn(rul_outputs.squeeze(), targets.float())
        
        else:
            # Default: use outputs directly
            return loss_fn(outputs, targets)


# Export for task factory compatibility
task = MultiTaskPHM