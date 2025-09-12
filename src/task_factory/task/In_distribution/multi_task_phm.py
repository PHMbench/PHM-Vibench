"""
Multi-task PHM implementation for In_distribution tasks.

Refactored from multi_task_lightning.py to follow standard task factory patterns.
Inherits from Default_task to reuse optimizer, scheduler, and logging infrastructure.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any
import pytorch_lightning as pl
import pandas as pd
import os

from ...Components.loss import get_loss_fn
from ...Components.metrics import get_metrics
from ...Components.system_metrics_tracker import SystemMetricsTracker
from ...Components.metrics_markdown_reporter import MetricsMarkdownReporter
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
        
        # Performance optimization flags
        self.validation_verbose = getattr(args_trainer, 'validation_verbose', False)
        self.warning_counts = {'truncate_channels': 0, 'no_rul_labels': 0, 'output_mismatch': 0, 
                               'missing_output': 0, 'invalid_output': 0, 'task_error': 0}
        self.max_warnings_per_type = 5  # Only show first N warnings of each type
        
        # Debug output: confirm configuration
        model_max_out = getattr(args_model, 'max_out', 'NOT_SET')
        print(f"[DEBUG] Model config: max_out={model_max_out}, validation_verbose={self.validation_verbose}")
        print(f"[DEBUG] Warning suppression initialized: max_warnings_per_type={self.max_warnings_per_type}")
        
        # Multi-task specific initialization (no single loss_fn needed)
        self.enabled_tasks = self._get_enabled_tasks()
        self.task_weights = self._get_task_weights()
        self.task_loss_fns = self._initialize_task_losses()
        self.task_metrics = self._initialize_task_metrics()
        
        # System-specific metrics tracking initialization
        self.val_system_tracker = SystemMetricsTracker()
        self.test_system_tracker = SystemMetricsTracker()
        
        # Initialize metrics reporter
        reports_dir = os.path.join(getattr(args_trainer, 'default_root_dir', './'), 'metrics_reports')
        self.metrics_reporter = MetricsMarkdownReporter(save_dir=reports_dir)
        
        # Configuration for system metrics tracking (disabled by default for performance)
        self.track_system_metrics = getattr(args_task, 'track_system_metrics', False)  # Changed to False for performance
        self.system_metrics_verbose = getattr(args_task, 'system_metrics_verbose', False)  # Changed to False for performance
        
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
        """Get enabled tasks from configuration with metadata validation."""
        default_tasks = ['classification', 'anomaly_detection', 
                        'signal_prediction', 'rul_prediction']
        config_tasks = getattr(self.args_task, 'enabled_tasks', default_tasks)
        
        # Configuration options for task validation
        enable_validation = getattr(self.args_task, 'enable_task_validation', True)
        validation_mode = getattr(self.args_task, 'validation_mode', 'warn')  # 'warn', 'error', 'ignore'
        force_enable_tasks = getattr(self.args_task, 'force_enable_tasks', [])
        
        # Skip validation if disabled
        if not enable_validation:
            print("Info: Task validation disabled, using configured tasks as-is")
            return config_tasks
            
        # Filter tasks based on dataset capabilities from metadata
        supported_tasks = self._get_dataset_supported_tasks()
        validated_tasks = []
        
        for task in config_tasks:
            # Check if task is force-enabled (skip validation)
            if task in force_enable_tasks:
                validated_tasks.append(task)
                print(f"Info: Task '{task}' force-enabled, skipping validation")
                continue
                
            # Check if task is supported by dataset
            if task in supported_tasks:
                validated_tasks.append(task)
            else:
                # Handle validation failure based on validation_mode
                if validation_mode == 'error':
                    raise ValueError(f"Task '{task}' not supported by current dataset(s)")
                elif validation_mode == 'warn':
                    print(f"Warning: Task '{task}' disabled - not supported by current dataset(s)")
                elif validation_mode == 'ignore':
                    # Silently skip unsupported tasks
                    pass
        
        # Fallback handling
        if not validated_tasks:
            if validation_mode == 'error':
                raise ValueError("No valid tasks found after validation")
            else:
                print("Warning: No valid tasks found, falling back to classification")
                validated_tasks = ['classification']  # Fallback to basic classification
            
        return validated_tasks
    
    def _get_dataset_supported_tasks(self) -> List[str]:
        """
        Check which tasks are supported by the current dataset(s) based on metadata fields.
        
        Returns list of supported task names based on metadata fields:
        - Fault_Diagnosis=1/TRUE → classification task
        - Anomaly_Detection=1/TRUE → anomaly_detection task  
        - Remaining_Life=1/TRUE → rul_prediction task
        - signal_prediction is always supported (reconstruction task)
        """
        supported_tasks = set()
        
        # Signal prediction is always supported (reconstruction task)
        supported_tasks.add('signal_prediction')
        
        # Check each dataset in metadata for task capability fields
        for file_id, meta in self.metadata.items():
            if not (isinstance(meta, dict) or isinstance(meta, pd.Series)):
                continue
                
            # Check Fault Diagnosis capability
            fault_diagnosis = meta.get('Fault_Diagnosis', False)
            if self._is_capability_enabled(fault_diagnosis):
                supported_tasks.add('classification')
                
            # Check Anomaly Detection capability
            anomaly_detection = meta.get('Anomaly_Detection', False)
            if self._is_capability_enabled(anomaly_detection):
                supported_tasks.add('anomaly_detection')
                
            # Check Remaining Life (RUL) capability
            remaining_life = meta.get('Remaining_Life', False)
            if self._is_capability_enabled(remaining_life):
                supported_tasks.add('rul_prediction')
        
        supported_list = list(supported_tasks)
        
        # Log supported tasks for debugging
        print(f"Dataset supported tasks: {supported_list}")
        
        return supported_list
    
    def _is_capability_enabled(self, value) -> bool:
        """
        Check if a capability field indicates the task is enabled.
        Handles various formats: 1, 'TRUE', 'true', True, etc.
        """
        if value is None:
            return False
            
        # Handle different value types and formats
        if isinstance(value, bool):
            return value
        elif isinstance(value, (int, float)):
            return value > 0
        elif isinstance(value, str):
            return value.upper() in ['TRUE', '1', 'YES', 'ENABLED']
        else:
            return False
    
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
        # task_metric_mapping = {
        #     'classification': ['acc', 'f1', 'precision', 'recall'],
        #     'anomaly_detection': ['acc', 'f1', 'precision', 'recall', 'auroc'],
        #     'signal_prediction': ['mse', 'mae', 'r2'],
        #     'rul_prediction': ['mse', 'mae', 'r2', 'mape']
        # }
        task_metric_mapping = {
            'classification': ['acc'],
            'anomaly_detection': ['acc', 'auroc'],
            'signal_prediction': ['mse', 'mae'],
            'rul_prediction': ['mse', 'mae',]
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
                                task='binary',
                                sync_on_compute=False  # Optimize validation performance
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
                                num_classes=int(max(max_classes, 2)),
                                sync_on_compute=False  # Optimize validation performance
                            ).to(self._custom_device)
                    elif metric_name == 'auroc':
                        stage_metrics[stage][metric_key] = torchmetrics.AUROC(
                            task='binary', 
                            sync_on_compute=False  # Optimize validation performance
                        ).to(self._custom_device)
                    else:
                        # Regression metrics
                        regression_class_names = {
                            'mse': 'MeanSquaredError',
                            'mae': 'MeanAbsoluteError', 
                            'r2': 'R2Score',
                            'mape': 'MeanAbsolutePercentageError'
                        }
                        metric_class_name = regression_class_names.get(metric_name, metric_name.upper())
                        stage_metrics[stage][metric_key] = getattr(torchmetrics, metric_class_name)(
                            sync_on_compute=False  # Optimize validation performance
                        ).to(self._custom_device)
            
            task_metrics[task] = stage_metrics
        
        return task_metrics
    
    def _log_warning(self, warning_type: str, message: str, force: bool = False):
        """Controlled warning logging to reduce validation slowdown."""
        # Initialize warning type counter if not exists
        if warning_type not in self.warning_counts:
            self.warning_counts[warning_type] = 0
            
        if force or self.validation_verbose or self.warning_counts[warning_type] < self.max_warnings_per_type:
            print(message)
            self.warning_counts[warning_type] += 1
            if self.warning_counts[warning_type] == self.max_warnings_per_type and not force:
                print(f"[INFO] Suppressing further '{warning_type}' warnings (showing first {self.max_warnings_per_type})")
    
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
            
        elif task_name == 'signal_prediction':
            # Handle channel mismatches using shared utility
            preds, targets = self._handle_channel_mismatch(preds, targets, task_name)
            
            if preds.dim() > 2:
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
                self._log_warning('no_rul_labels', "Warning: No valid RUL labels in current batch, skipping RUL task")
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
                        self._log_warning('missing_output', f'WARNING: No output found for task {task_name} in model outputs')
                        self.log(f'{mode}_{task_name}_skipped', 1.0, on_step=False, on_epoch=True)
                        continue
                        
                    # Validate output shape and type
                    if not isinstance(task_output, torch.Tensor):
                        self._log_warning('invalid_output', f'WARNING: Task {task_name} output is not a tensor (type: {type(task_output)}), skipping')
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
                            # Log metrics only on epoch for cleaner logging, disable prog_bar for validation
                            prog_bar_enabled = (mode == 'train')  # Only show progress bar during training
                            self.log(metric_name, metric_value, 
                                    on_step=False, on_epoch=True, prog_bar=prog_bar_enabled)
                            
                except Exception as e:
                    # Log detailed error information and continue
                    error_msg = f'ERROR: {task_name} {mode} step failed - {type(e).__name__}: {str(e)}'
                    self._log_warning('task_error', error_msg, force=True)  # Force log errors
                    
                    # Log error metrics for monitoring
                    self.log(f'{mode}_{task_name}_error_count', 1.0, on_step=False, on_epoch=True)
                    
                    # Add shape information if available for debugging (only if verbose)
                    if self.validation_verbose:
                        if 'task_output' in locals() and hasattr(task_output, 'shape'):
                            print(f'  Task output shape: {task_output.shape}')
                        if task_name in y_dict and hasattr(y_dict[task_name], 'shape'):
                            print(f'  Target shape: {y_dict[task_name].shape}')
                    
                    continue
        
        # System-specific metrics tracking (for validation and test phases)
        if self.track_system_metrics and mode in ['test'] and file_ids:
            system_id = self._extract_system_id_from_batch(file_ids)
            if system_id is not None:
                # Collect batch-level metrics for this system
                batch_metrics = {}
                
                # Re-compute metrics for system tracking (without logging to avoid duplication)
                for task_name in self.enabled_tasks:
                    if task_name in y_dict:
                        try:
                            task_output = outputs.get(task_name) if isinstance(outputs, dict) else getattr(outputs, task_name, None)
                            if task_output is not None and isinstance(task_output, torch.Tensor):
                                metric_targets = x if task_name == 'signal_prediction' and y_dict[task_name] is None else y_dict[task_name]
                                task_metrics = self._compute_task_metrics(task_name, task_output, metric_targets, mode)
                                
                                # Add task-specific metrics to batch metrics
                                for metric_name, metric_value in task_metrics.items():
                                    batch_metrics[metric_name] = metric_value
                        except Exception as e:
                            # Skip this task for system tracking if it fails
                            continue
                
                # Update system tracker
                if batch_metrics:
                    tracker = self.val_system_tracker if mode == 'val' else self.test_system_tracker
                    tracker.update(system_id, batch_metrics)
                    
                    # Log system-specific metrics with system ID prefix
                    for metric_name, metric_value in batch_metrics.items():
                        self.log(f'{mode}_sys{system_id}_{metric_name}', metric_value,
                                on_step=False, on_epoch=True, add_dataloader_idx=False)
        
        # Log total loss with epoch-only logging to improve training speed
        on_step = False  # Disable step-level logging for faster training
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
    
    def _extract_system_id_from_batch(self, file_ids: List[int]):
        """Extract system ID from batch file IDs.
        
        Since Same_system_Sampler ensures all samples in a batch come from the same system,
        we can use the first file_id to get the system ID.
        
        Args:
            file_ids: List of file IDs in the batch
            
        Returns:
            System ID or None if not found
        """
        if not file_ids:
            return None
            
        first_file_id = file_ids[0]
        if first_file_id not in self.metadata:
            return None
            
        metadata_row = self.metadata[first_file_id]
        
        # Try Dataset_id first, then System_id as fallback
        system_id = metadata_row.get('Dataset_id')
        if system_id is None:
            system_id = metadata_row.get('System_id')
            
        return system_id
    
    def _handle_channel_mismatch(self, task_output: torch.Tensor, targets: torch.Tensor, task_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Handle channel dimension mismatches between model outputs and targets.
        Used by both loss and metrics computation to ensure consistency.
        
        Args:
            task_output: Model prediction tensor
            targets: Ground truth tensor
            task_name: Name of the task for logging
            
        Returns:
            Tuple of (adjusted_output, adjusted_targets)
        """
        if task_output.shape[-1] != targets.shape[-1]:
            target_channels = targets.shape[-1]
            output_channels = task_output.shape[-1]
            
            if output_channels < target_channels:
                # Truncate target to match output channels (memory-constrained scenario)
                self._log_warning('truncate_channels', 
                    f"Info: Truncating target channels from {target_channels} to {output_channels} for {task_name}")
                targets = targets[..., :output_channels]
            else:
                # Pad output to match target channels (shouldn't happen with current logic)
                self._log_warning('output_mismatch', 
                    f"Warning: Output channels ({output_channels}) > target channels ({target_channels}) for {task_name}, truncating output")
                task_output = task_output[..., :target_channels]
        
        return task_output, targets

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
            
            # Handle dimension mismatches using shared utility
            task_output, targets = self._handle_channel_mismatch(task_output, targets, task_name)
            
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
    
    def on_validation_epoch_end(self):
        """Validation epoch结束时汇总系统指标"""
        if not self.track_system_metrics:
            return
            
        if self.val_system_tracker.get_system_count() > 0:
            # 计算系统级epoch指标
            epoch_metrics = self.val_system_tracker.compute_epoch_metrics()
            
            # 记录每个系统的平均指标
            for sys_id, sys_metrics in epoch_metrics.items():
                for metric_name, metric_value in sys_metrics.items():
                    self.log(f'val_epoch_sys{sys_id}_{metric_name}', metric_value,
                            on_step=False, on_epoch=True)
            
            # 打印系统性能对比
            if self.system_metrics_verbose:
                self._print_system_comparison('Validation', epoch_metrics)
            
            # 清空追踪器准备下一epoch
            self.val_system_tracker.clear()

    def on_test_epoch_end(self):
        """Test epoch结束时生成完整报告"""
        if not self.track_system_metrics:
            return
            
        if self.test_system_tracker.get_system_count() > 0:
            # 计算系统级epoch指标
            epoch_metrics = self.test_system_tracker.compute_epoch_metrics()
            
            # 收集全局指标
            global_metrics = {}
            if hasattr(self.trainer, 'logged_metrics'):
                for key, value in self.trainer.logged_metrics.items():
                    if key.startswith('test_') and 'sys' not in key:
                        clean_key = key.replace('test_', '')
                        global_metrics[clean_key] = float(value) if torch.is_tensor(value) else value
            
            # 准备配置信息
            config_info = {
                'model': getattr(self.args_model, 'name', 'Unknown'),
                'enabled_tasks': ', '.join(self.enabled_tasks),
                'batch_size': getattr(self.args_data, 'batch_size', 'Unknown'),
                'systems_tested': len(epoch_metrics)
            }
            
            try:
                # 生成Markdown报告
                report_path = self.metrics_reporter.generate_report(
                    system_metrics=epoch_metrics,
                    global_metrics=global_metrics,
                    phase='test',
                    experiment_name=getattr(self.args_task, 'name', 'multi_task_phm'),
                    config_info=config_info
                )
                
                # 记录报告路径
                self.log('metrics_report_path', str(report_path), on_step=False, on_epoch=True)
                
                if self.system_metrics_verbose:
                    print(f"\n📊 System metrics report generated: {report_path}")
                
            except Exception as e:
                print(f"Warning: Failed to generate metrics report: {e}")
            
            # 打印系统性能对比
            if self.system_metrics_verbose:
                self._print_system_comparison('Test', epoch_metrics)
            
            # 清空追踪器
            self.test_system_tracker.clear()

    def _print_system_comparison(self, phase: str, epoch_metrics: Dict[str, Dict[str, float]]):
        """打印系统间性能对比表"""
        if not epoch_metrics:
            return
            
        print(f"\n{phase} System-Level Metrics:")
        print("-" * 80)
        
        # 组织成表格格式
        all_metrics = set()
        for sys_metrics in epoch_metrics.values():
            all_metrics.update(sys_metrics.keys())
        
        # 选择前5个最重要的指标显示
        priority_metrics = []
        metric_priorities = ['classification_acc', 'classification_f1', 'anomaly_auroc', 'rul_mae', 'signal_r2']
        
        for priority in metric_priorities:
            if priority in all_metrics:
                priority_metrics.append(priority)
        
        # 添加其他指标直到5个
        remaining_metrics = [m for m in sorted(all_metrics) if m not in priority_metrics]
        display_metrics = priority_metrics + remaining_metrics[:max(0, 5 - len(priority_metrics))]
        
        # 打印表头
        header = f"{'System':<15}"
        for metric in display_metrics:
            short_name = metric.replace('classification_', '').replace('anomaly_', '').replace('signal_', '')[:12]
            header += f"{short_name:<15}"
        print(header)
        print("-" * 80)
        
        # 打印每个系统的指标
        for sys_id, sys_metrics in sorted(epoch_metrics.items()):
            row = f"{'sys_' + str(sys_id):<15}"
            for metric in display_metrics:
                value = sys_metrics.get(metric, 'N/A')
                if isinstance(value, float):
                    row += f"{value:<15.4f}"
                else:
                    row += f"{str(value):<15}"
            print(row)
        print("-" * 80)


# Export for task factory compatibility
# task = task  # Already defined as class name