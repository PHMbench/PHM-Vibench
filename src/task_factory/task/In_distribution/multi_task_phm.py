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
from ...Components.metrics import get_metrics
import torchmetrics


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
        # Store device for metrics initialization (use custom name to avoid conflict with Lightning)
        try:
            self._custom_device = next(self.network.parameters()).device
        except StopIteration:
            # Handle case where network has no parameters (e.g., mock networks in tests)
            self._custom_device = torch.device('cuda' if args_trainer.gpus else 'cpu')
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
        self.task_metrics = self._initialize_task_metrics()
        
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
    
    def _initialize_task_metrics(self) -> Dict[str, Dict[str, torchmetrics.Metric]]:
        """Initialize metrics for each enabled task."""
        # Define metrics for each task type
        task_metric_mapping = {
            'classification': ['acc', 'f1', 'precision', 'recall'],
            'anomaly_detection': ['acc', 'f1', 'precision', 'recall', 'auroc'],
            'signal_prediction': ['mse', 'mae', 'r2'],
            'rul_prediction': ['mse', 'mae', 'r2', 'mape']
        }
        
        task_metrics = {}
        for task in self.enabled_tasks:
            metrics_list = task_metric_mapping.get(task, ['mse', 'mae'])
            
            # Create metrics for each stage (train, val, test)
            stage_metrics = {}
            for stage in ['train', 'val', 'test']:
                stage_metrics[stage] = nn.ModuleDict()
                for metric_name in metrics_list:
                    metric_key = f"{stage}_{metric_name}"
                    
                    # Classification metrics need special parameters
                    if metric_name in ['acc', 'f1', 'precision', 'recall']:
                        # Map metric names to torchmetrics class names
                        metric_class_names = {
                            'acc': 'Accuracy',
                            'f1': 'F1Score', 
                            'precision': 'Precision',
                            'recall': 'Recall'
                        }
                        metric_class_name = metric_class_names[metric_name]
                        
                        if task == 'anomaly_detection':
                            # Binary classification
                            stage_metrics[stage][metric_key] = getattr(torchmetrics, metric_class_name)(
                                task='binary'
                            ).to(self._custom_device)
                        else:
                            # Multi-class classification - determine num_classes from metadata
                            labels = [
                                item.get('Label', 0) for item in self.metadata.values() 
                                if isinstance(item, dict) and 'Label' in item
                            ]
                            max_classes = int(max(labels)) + 1 if labels else 2
                            stage_metrics[stage][metric_key] = getattr(torchmetrics, metric_class_name)(
                                task='multiclass',
                                num_classes=int(max(max_classes, 2))
                            ).to(self._custom_device)
                    elif metric_name == 'auroc':
                        stage_metrics[stage][metric_key] = torchmetrics.AUROC(task='binary').to(self._custom_device)
                    else:
                        # Regression metrics
                        regression_class_names = {
                            'mse': 'MeanSquaredError',
                            'mae': 'MeanAbsoluteError', 
                            'r2': 'R2Score',
                            'mape': 'MeanAbsolutePercentageError'
                        }
                        metric_class_name = regression_class_names.get(metric_name, metric_name.upper())
                        stage_metrics[stage][metric_key] = getattr(torchmetrics, metric_class_name)().to(self._custom_device)
            
            task_metrics[task] = stage_metrics
        
        return task_metrics
    
    def _compute_task_metrics(self, task_name: str, task_output: torch.Tensor, targets: torch.Tensor, mode: str) -> Dict[str, torch.Tensor]:
        """Compute task-specific metrics using single dispatch to task-specific methods."""
        if task_name not in self.task_metrics:
            return {}
        
        # Single dispatch to task-specific method
        if task_name == 'classification':
            return self._compute_classification_metrics(task_name, task_output, targets, mode)
        elif task_name == 'anomaly_detection':
            return self._compute_anomaly_metrics(task_name, task_output, targets, mode)
        elif task_name in ['signal_prediction', 'rul_prediction']:
            return self._compute_regression_metrics(task_name, task_output, targets, mode)
        return {}
    
    def _compute_classification_metrics(self, task_name: str, task_output: torch.Tensor, targets: torch.Tensor, mode: str) -> Dict[str, torch.Tensor]:
        """Compute classification metrics with task-specific logic."""
        metric_values = {}
        stage_metrics = self.task_metrics[task_name][mode]
        
        # Prepare classification predictions
        preds = torch.argmax(task_output, dim=-1)
        targets = targets.long()
        
        # Compute all classification metrics
        for metric_key, metric_fn in stage_metrics.items():
            try:
                value = metric_fn(preds, targets)
                metric_values[f"{task_name}_{metric_key}"] = value
            except Exception as e:
                print(f"Warning: Failed to compute {metric_key} for {task_name}: {e}")
                continue
        
        return metric_values
    
    def _compute_anomaly_metrics(self, task_name: str, task_output: torch.Tensor, targets: torch.Tensor, mode: str) -> Dict[str, torch.Tensor]:
        """Compute anomaly detection metrics with task-specific logic."""
        metric_values = {}
        stage_metrics = self.task_metrics[task_name][mode]
        
        # Prepare anomaly detection outputs
        prob_preds = torch.sigmoid(task_output).squeeze(-1)
        binary_preds = (prob_preds > 0.5).int()
        targets = targets.float()
        
        # Compute all anomaly detection metrics
        for metric_key, metric_fn in stage_metrics.items():
            try:
                if 'auroc' in metric_key:
                    # AUROC needs probabilities
                    value = metric_fn(prob_preds, targets.int())
                elif any(clf_metric in metric_key for clf_metric in ['acc', 'f1', 'precision', 'recall']):
                    # Binary classification metrics need integer predictions
                    value = metric_fn(binary_preds, targets.int())
                else:
                    # Other metrics use probabilities
                    value = metric_fn(prob_preds, targets)
                    
                metric_values[f"{task_name}_{metric_key}"] = value
            except Exception as e:
                print(f"Warning: Failed to compute {metric_key} for {task_name}: {e}")
                continue
        
        return metric_values
    
    def _compute_regression_metrics(self, task_name: str, task_output: torch.Tensor, targets: torch.Tensor, mode: str) -> Dict[str, torch.Tensor]:
        """Compute regression metrics with task-specific logic for signal_prediction and rul_prediction."""
        metric_values = {}
        stage_metrics = self.task_metrics[task_name][mode]
        
        # Skip if no targets available
        if targets is None:
            return {}
        
        # Prepare regression predictions
        preds = task_output
        targets = targets.float()
        
        # Task-specific dimension handling
        if task_name == 'rul_prediction':
            # Handle RUL predictions - squeeze until dimensions match
            while preds.dim() > targets.dim():
                preds = preds.squeeze(-1)
            # Final squeeze if last dimension is 1
            if preds.shape[-1:] == torch.Size([1]) and len(targets.shape) == 1:
                preds = preds.squeeze(-1)
                
            # FIXED: Filter out NaN values for RUL metrics
            valid_mask = ~torch.isnan(targets)
            if valid_mask.sum() == 0:
                # No valid targets, skip metrics
                return {}
            preds = preds[valid_mask]
            targets = targets[valid_mask]
            
        elif task_name == 'signal_prediction' and preds.dim() > 2:
            # For signal prediction with 3D tensors, flatten for metric computation
            preds = preds.reshape(-1, preds.size(-1)).contiguous()
            targets = targets.reshape(-1, targets.size(-1)).contiguous()

        # Compute all regression metrics
        for metric_key, metric_fn in stage_metrics.items():
            try:
                value = metric_fn(preds, targets)
                metric_values[f"{task_name}_{metric_key}"] = value
            except Exception as e:
                print(f"Warning: Failed to compute {metric_key} for {task_name}: {e}")
                continue
        
        return metric_values
    
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
    

    
    def _build_task_labels_batch(self, y, file_ids):
        """Build task-specific labels for batch processing with individual metadata per sample."""
        y_dict = {}
        batch_size = y.shape[0] if hasattr(y, 'shape') else len(y)
        
        # Convert file_ids to consistent format
        if hasattr(file_ids, 'tolist'):
            file_ids = file_ids.tolist()
        elif isinstance(file_ids, torch.Tensor):
            file_ids = [fid.item() if isinstance(fid, torch.Tensor) else fid for fid in file_ids]
        
        # Ensure we have the right number of file_ids
        if len(file_ids) != batch_size:
            # Handle case where all samples might have same file_id (single file batch)
            if len(file_ids) == 1:
                file_ids = file_ids * batch_size
            else:
                raise ValueError(f"Mismatch: batch_size={batch_size}, file_ids={len(file_ids)}")
        
        # Classification task - use original labels
        if 'classification' in self.enabled_tasks:
            y_dict['classification'] = y
        
        # Anomaly detection - convert to binary (0=normal, 1=anomaly) 
        if 'anomaly_detection' in self.enabled_tasks:
            # Assume label 0 is normal, others are anomaly
            y_dict['anomaly_detection'] = (y > 0).float()
        
        # RUL prediction - get from metadata for each sample
        if 'rul_prediction' in self.enabled_tasks:
            rul_values = []
            valid_rul_count = 0
            
            for file_id in file_ids:
                metadata = self.metadata.get(file_id, {})
                rul_value = metadata.get('RUL_label', None)
                
                if rul_value is not None and isinstance(rul_value, (int, float)) and not torch.isnan(torch.tensor(rul_value)):
                    # Valid RUL value found
                    rul_values.append(float(rul_value))
                    valid_rul_count += 1
                else:
                    # FIXED: Skip samples with missing RUL instead of using default values
                    # This prevents training on fake/default data that degrades performance
                    rul_values.append(float('nan'))  # Mark as invalid
                    
                    # Log warning only once per file to avoid spam
                    if not hasattr(self, '_rul_warnings'):
                        self._rul_warnings = set()
                    if file_id not in self._rul_warnings:
                        print(f"Warning: Missing/invalid RUL label for file {file_id}, sample will be ignored in RUL task")
                        self._rul_warnings.add(file_id)
            
            # Only create RUL targets if we have valid samples
            if valid_rul_count > 0:
                y_dict['rul_prediction'] = torch.tensor(rul_values, dtype=torch.float32, device=y.device)
            else:
                # Skip RUL task entirely for this batch if no valid labels
                print(f"Warning: No valid RUL labels in current batch, skipping RUL task")
                y_dict['rul_prediction'] = None
        
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
        file_ids = batch['file_id']  # Get all file IDs in the batch
        
        # Handle both tensor and list formats for file_ids
        if isinstance(file_ids, torch.Tensor):
            file_ids = file_ids.tolist()
        
        # FIXED: Pass full file_ids list to support batch-level system_id processing
        # The model (M_01_ISFM) now supports both single file_id and batch file_ids
        outputs = self.network(x, file_ids, task_id=self.enabled_tasks)
        
        # Build task-specific labels for each sample using its own metadata
        y_dict = self._build_task_labels_batch(y, file_ids)
        
        # Compute loss and metrics for each enabled task
        total_loss = 0.0
        
        for task_name in self.enabled_tasks:
            if task_name in y_dict:
                try:
                    # Extract task-specific output with improved validation
                    task_output = None
                    if isinstance(outputs, dict):
                        task_output = outputs.get(task_name)
                    else:
                        # Handle object with attributes
                        task_output = getattr(outputs, task_name, None)
                    
                    if task_output is None:
                        # Log warning and skip this task
                        print(f'WARNING: No output found for task {task_name} in model outputs')
                        self.log(f'{mode}_{task_name}_skipped', 1.0, on_step=False, on_epoch=True)
                        continue
                        
                    # Validate output shape and type
                    if not isinstance(task_output, torch.Tensor):
                        print(f'WARNING: Task {task_name} output is not a tensor (type: {type(task_output)}), skipping')
                        self.log(f'{mode}_{task_name}_invalid_output', 1.0, on_step=False, on_epoch=True)
                        continue
                    
                    # Compute task loss with extracted output
                    task_loss = self._compute_task_loss(task_name, task_output, y_dict[task_name], x)
                    if task_loss is not None:
                        weighted_loss = self.task_weights[task_name] * task_loss
                        total_loss += weighted_loss
                        
                        # Log individual task loss with mode-specific parameters
                        on_step = (mode == 'train')
                        self.log(f'{mode}_{task_name}_loss', task_loss, 
                                on_step=on_step, on_epoch=True)
                        
                        # Compute and log task-specific metrics
                        # Use input signal as target for signal_prediction reconstruction metrics
                        metric_targets = x if task_name == 'signal_prediction' and y_dict[task_name] is None else y_dict[task_name]
                        task_metrics = self._compute_task_metrics(task_name, task_output, metric_targets, mode)
                        for metric_name, metric_value in task_metrics.items():
                            # Log metrics only on epoch for cleaner logging
                            self.log(metric_name, metric_value, 
                                    on_step=False, on_epoch=True, prog_bar=True)
                            
                except Exception as e:
                    # Log detailed error information and continue
                    error_msg = f'ERROR: {task_name} {mode} step failed - {type(e).__name__}: {str(e)}'
                    print(error_msg)
                    
                    # Log error metrics for monitoring
                    self.log(f'{mode}_{task_name}_error_count', 1.0, on_step=False, on_epoch=True)
                    
                    # Add shape information if available for debugging
                    if 'task_output' in locals() and hasattr(task_output, 'shape'):
                        print(f'  Task output shape: {task_output.shape}')
                    if task_name in y_dict and hasattr(y_dict[task_name], 'shape'):
                        print(f'  Target shape: {y_dict[task_name].shape}')
                    
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
    
    def _compute_task_loss(self, task_name: str, task_output: torch.Tensor, targets: torch.Tensor, x: torch.Tensor = None) -> torch.Tensor:
        """Compute loss for a specific task using task-specific output directly."""
        loss_fn = self.task_loss_fns[task_name]
        
        if task_name == 'classification':
            return loss_fn(task_output, targets.long())
        
        elif task_name == 'anomaly_detection':
            # Binary classification for anomaly detection
            # Fix dimension mismatch: BCE expects [batch_size] for both input and target
            task_output_squeezed = task_output.squeeze(-1) if task_output.dim() > 1 and task_output.shape[-1] == 1 else task_output
            targets_squeezed = targets.squeeze() if targets.dim() > 1 else targets
            return loss_fn(task_output_squeezed, targets_squeezed.float())
        
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
            if targets is None:
                # Skip this task if no valid RUL targets in batch
                return None
                
            # FIXED: Handle NaN values in RUL targets
            targets_float = targets.float()
            valid_mask = ~torch.isnan(targets_float)
            
            if valid_mask.sum() == 0:
                # No valid RUL targets in this batch
                return None
            
            # Only compute loss on valid (non-NaN) samples
            valid_predictions = task_output.squeeze()[valid_mask]
            valid_targets = targets_float[valid_mask]
            
            if len(valid_predictions) > 0:
                return loss_fn(valid_predictions, valid_targets)
            else:
                return None
        
        else:
            # Default: use task output directly
            return loss_fn(task_output, targets)


# Export for task factory compatibility
# task = task  # Already defined as class name