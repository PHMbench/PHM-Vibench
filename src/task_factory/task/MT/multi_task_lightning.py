"""
Multi-Task Lightning Module for PHM Foundation Model

This module extends PyTorch Lightning's LightningModule to handle multi-task training
for Prognostics and Health Management (PHM) applications. It supports simultaneous
training on three tasks:
1. Fault Classification
2. Remaining Useful Life (RUL) Prediction  
3. Anomaly Detection

The module implements combined loss functions, separate metrics tracking for each task,
and proper training/validation/test step implementations with configurable loss weights.

Author: PHM-Vibench Team
Date: 2025-08-18
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np

# Import existing components
from ...Components.loss import get_loss_fn
from ...Components.metrics import get_metrics
from ...Components.regularization import calculate_regularization


class MultiTaskLightningModule(pl.LightningModule):
    """
    Multi-task PyTorch Lightning module for PHM foundation model training.
    
    This module handles simultaneous training on fault classification, RUL prediction,
    and anomaly detection tasks with configurable loss weights and separate metrics
    tracking for each task.
    
    Parameters
    ----------
    network : nn.Module
        The backbone network (e.g., ISFM model)
    args_data : Any
        Data configuration arguments
    args_model : Any
        Model configuration arguments
    args_task : Any
        Task configuration arguments
    args_trainer : Any
        Trainer configuration arguments
    args_environment : Any
        Environment configuration arguments
    metadata : Any
        Dataset metadata
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
        super().__init__()
        
        # Store configuration
        self.network = network.cuda() if args_trainer.gpus else network
        self.args_task = args_task
        self.args_model = args_model
        self.args_data = args_data
        self.args_trainer = args_trainer
        self.args_environment = args_environment
        self.metadata = metadata
        
        # Multi-task configuration
        self.task_weights = self._get_task_weights()
        self.enabled_tasks = self._get_enabled_tasks()
        
        # Initialize loss functions for each task
        self.loss_functions = self._initialize_loss_functions()
        
        # Initialize metrics for each task
        self.metrics = self._initialize_metrics()
        
        # Save hyperparameters
        hparams_dict = {
            **vars(self.args_task),
            **vars(self.args_model),
            **vars(self.args_data),
            **vars(self.args_trainer),
            **vars(self.args_environment),
        }
        self.save_hyperparameters(hparams_dict, ignore=['network', 'metadata'])
    
    def _get_task_weights(self) -> Dict[str, float]:
        """Get loss weights for each task from configuration."""
        default_weights = {
            'classification': 1.0,
            'rul_prediction': 1.0,
            'anomaly_detection': 1.0
        }
        
        # Override with user-specified weights if available
        if hasattr(self.args_task, 'task_weights'):
            default_weights.update(self.args_task.task_weights)
        
        return default_weights
    
    def _get_enabled_tasks(self) -> List[str]:
        """Get list of enabled tasks from configuration."""
        if hasattr(self.args_task, 'enabled_tasks'):
            return self.args_task.enabled_tasks
        elif hasattr(self.args_task, 'task_list'):
            return self.args_task.task_list
        else:
            # Default to all tasks
            return ['classification', 'rul_prediction', 'anomaly_detection']
    
    def _initialize_loss_functions(self) -> Dict[str, nn.Module]:
        """Initialize loss functions for each task."""
        loss_functions = {}
        
        if 'classification' in self.enabled_tasks:
            # Use CrossEntropy for multi-class classification
            loss_functions['classification'] = nn.CrossEntropyLoss()
        
        if 'rul_prediction' in self.enabled_tasks:
            # Use MSE for regression
            rul_loss_type = getattr(self.args_task, 'rul_loss', 'MSE')
            if rul_loss_type.upper() == 'MAE':
                loss_functions['rul_prediction'] = nn.L1Loss()
            else:
                loss_functions['rul_prediction'] = nn.MSELoss()
        
        if 'anomaly_detection' in self.enabled_tasks:
            # Use BCE with logits for binary classification
            loss_functions['anomaly_detection'] = nn.BCEWithLogitsLoss()
        
        return loss_functions
    
    def _initialize_metrics(self) -> Dict[str, Dict[str, torchmetrics.Metric]]:
        """Initialize metrics for each task and dataset."""
        all_metrics = {}
        
        # Get unique dataset names from metadata
        dataset_names = set()
        for item_id, item_data in self.metadata.items():
            if "Name" in item_data:
                dataset_names.add(item_data["Name"])
        
        for dataset_name in dataset_names:
            dataset_metrics = {}
            
            # Classification metrics
            if 'classification' in self.enabled_tasks:
                # Get number of classes for this dataset
                n_classes = self._get_num_classes_for_dataset(dataset_name)
                task_type = "multiclass" if n_classes > 2 else "binary"
                
                for stage in ["train", "val", "test"]:
                    dataset_metrics[f"{stage}_cls_acc"] = torchmetrics.Accuracy(
                        task=task_type, num_classes=n_classes
                    )
                    dataset_metrics[f"{stage}_cls_f1"] = torchmetrics.F1Score(
                        task=task_type, num_classes=n_classes
                    )
            
            # RUL prediction metrics
            if 'rul_prediction' in self.enabled_tasks:
                for stage in ["train", "val", "test"]:
                    dataset_metrics[f"{stage}_rul_mse"] = torchmetrics.MeanSquaredError()
                    dataset_metrics[f"{stage}_rul_mae"] = torchmetrics.MeanAbsoluteError()
                    dataset_metrics[f"{stage}_rul_r2"] = torchmetrics.R2Score()
            
            # Anomaly detection metrics
            if 'anomaly_detection' in self.enabled_tasks:
                for stage in ["train", "val", "test"]:
                    dataset_metrics[f"{stage}_anom_acc"] = torchmetrics.Accuracy(task="binary")
                    dataset_metrics[f"{stage}_anom_f1"] = torchmetrics.F1Score(task="binary")
                    dataset_metrics[f"{stage}_anom_auroc"] = torchmetrics.AUROC(task="binary")
            
            all_metrics[dataset_name] = nn.ModuleDict(dataset_metrics)
        
        return nn.ModuleDict(all_metrics)
    
    def _get_num_classes_for_dataset(self, dataset_name: str) -> int:
        """Get number of classes for a specific dataset."""
        max_label = 0
        for item_id, item_data in self.metadata.items():
            if item_data.get("Name") == dataset_name and "Label" in item_data:
                max_label = max(max_label, item_data["Label"])
        return max_label + 1
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the network."""
        x = batch['x']
        file_id = batch['file_id']
        
        # Get all task outputs
        outputs = self.network(x, file_id, task_id='all')
        return outputs
    
    def _compute_task_losses(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute losses for each enabled task."""
        losses = {}
        
        if 'classification' in self.enabled_tasks and 'classification' in predictions:
            cls_pred = predictions['classification']
            cls_target = targets['classification'].long()
            losses['classification'] = self.loss_functions['classification'](cls_pred, cls_target)
        
        if 'rul_prediction' in self.enabled_tasks and 'rul_prediction' in predictions:
            rul_pred = predictions['rul_prediction'].squeeze(-1)
            rul_target = targets['rul_prediction'].float()
            losses['rul_prediction'] = self.loss_functions['rul_prediction'](rul_pred, rul_target)
        
        if 'anomaly_detection' in self.enabled_tasks and 'anomaly_detection' in predictions:
            anom_pred = predictions['anomaly_detection'].squeeze(-1)
            anom_target = targets['anomaly_detection'].float()
            losses['anomaly_detection'] = self.loss_functions['anomaly_detection'](anom_pred, anom_target)
        
        return losses
    
    def _compute_total_loss(self, task_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted total loss across all tasks."""
        total_loss = torch.tensor(0.0, device=self.device)
        
        for task_name, loss_value in task_losses.items():
            weight = self.task_weights.get(task_name, 1.0)
            total_loss += weight * loss_value
        
        return total_loss
    
    def _update_metrics(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor],
        dataset_name: str,
        stage: str
    ) -> Dict[str, torch.Tensor]:
        """Update metrics for all tasks."""
        metric_values = {}
        
        if dataset_name not in self.metrics:
            return metric_values
        
        dataset_metrics = self.metrics[dataset_name]
        
        # Classification metrics
        if 'classification' in predictions and 'classification' in targets:
            cls_pred = predictions['classification']
            cls_target = targets['classification'].long()
            cls_pred_labels = torch.argmax(cls_pred, dim=1)
            
            for metric_name in [f"{stage}_cls_acc", f"{stage}_cls_f1"]:
                if metric_name in dataset_metrics:
                    value = dataset_metrics[metric_name](cls_pred_labels, cls_target)
                    metric_values[f"{metric_name}_{dataset_name}"] = value
        
        # RUL prediction metrics
        if 'rul_prediction' in predictions and 'rul_prediction' in targets:
            rul_pred = predictions['rul_prediction'].squeeze(-1)
            rul_target = targets['rul_prediction'].float()
            
            for metric_name in [f"{stage}_rul_mse", f"{stage}_rul_mae", f"{stage}_rul_r2"]:
                if metric_name in dataset_metrics:
                    value = dataset_metrics[metric_name](rul_pred, rul_target)
                    metric_values[f"{metric_name}_{dataset_name}"] = value
        
        # Anomaly detection metrics
        if 'anomaly_detection' in predictions and 'anomaly_detection' in targets:
            anom_pred = predictions['anomaly_detection'].squeeze(-1)
            anom_target = targets['anomaly_detection'].float()
            anom_pred_probs = torch.sigmoid(anom_pred)
            anom_pred_labels = (anom_pred_probs > 0.5).float()
            
            for metric_name in [f"{stage}_anom_acc", f"{stage}_anom_f1"]:
                if metric_name in dataset_metrics:
                    value = dataset_metrics[metric_name](anom_pred_labels, anom_target)
                    metric_values[f"{metric_name}_{dataset_name}"] = value
            
            # AUROC uses probabilities, not labels
            auroc_metric_name = f"{stage}_anom_auroc"
            if auroc_metric_name in dataset_metrics:
                value = dataset_metrics[auroc_metric_name](anom_pred_probs, anom_target.long())
                metric_values[f"{auroc_metric_name}_{dataset_name}"] = value
        
        return metric_values

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> Dict[str, torch.Tensor]:
        """Shared step for training, validation, and testing."""
        try:
            # Extract batch information
            file_id = batch['file_id'][0].item()
            dataset_name = self.metadata[file_id]['Name']
            batch.update({'file_id': file_id})
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error processing batch: {e}")

        # Forward pass
        predictions = self.forward(batch)

        # Prepare targets
        targets = {}
        if 'y' in batch:  # Classification target
            targets['classification'] = batch['y']
        if 'rul' in batch:  # RUL target
            targets['rul_prediction'] = batch['rul']
        if 'anomaly' in batch:  # Anomaly target
            targets['anomaly_detection'] = batch['anomaly']

        # Compute task losses
        task_losses = self._compute_task_losses(predictions, targets)

        # Compute total loss
        total_loss = self._compute_total_loss(task_losses)

        # Update metrics
        metric_values = self._update_metrics(predictions, targets, dataset_name, stage)

        # Prepare step metrics
        step_metrics = {f"{stage}_total_loss": total_loss}

        # Add individual task losses
        for task_name, loss_value in task_losses.items():
            step_metrics[f"{stage}_{task_name}_loss"] = loss_value
            step_metrics[f"{stage}_{task_name}_loss_{dataset_name}"] = loss_value

        # Add computed metrics
        step_metrics.update(metric_values)

        # Add regularization if configured
        reg_dict = self._compute_regularization()
        for reg_type, reg_loss_val in reg_dict.items():
            if reg_type != 'total':
                step_metrics[f"{stage}_{reg_type}_reg_loss"] = reg_loss_val

        # Update total loss with regularization
        total_loss_with_reg = total_loss + reg_dict.get('total', torch.tensor(0.0, device=total_loss.device))
        step_metrics[f"{stage}_total_loss_with_reg"] = total_loss_with_reg

        return step_metrics

    def _compute_regularization(self) -> Dict[str, torch.Tensor]:
        """Compute regularization losses."""
        return calculate_regularization(
            getattr(self.args_task, 'regularization', {}),
            self.parameters()
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        metrics = self._shared_step(batch, "train")
        self._log_metrics(metrics, "train")
        return metrics["train_total_loss_with_reg"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        metrics = self._shared_step(batch, "val")
        self._log_metrics(metrics, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step."""
        metrics = self._shared_step(batch, "test")
        self._log_metrics(metrics, "test")

    def _log_metrics(self, metrics: Dict[str, torch.Tensor], stage: str) -> None:
        """Log metrics to tensorboard/wandb."""
        log_dict = {}
        prog_bar_metrics = {}

        for k, v in metrics.items():
            if k.startswith(stage):
                log_dict[k] = v

                # Select key metrics for progress bar
                if any(key in k for key in ['total_loss', 'cls_acc', 'rul_mse', 'anom_f1']):
                    if 'total_loss' in k or any(metric in k for metric in ['_acc_', '_mse_', '_f1_']):
                        prog_bar_metrics[k] = v

        self.log_dict(
            log_dict,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer_name = self.args_task.optimizer.lower()
        lr = self.args_task.lr
        weight_decay = getattr(self.args_task, 'weight_decay', 0.0)

        # Select optimizer
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = getattr(self.args_task, 'momentum', 0.9)
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Configure scheduler if specified
        scheduler_config = getattr(self.args_task, 'scheduler', None)
        if not scheduler_config or not isinstance(scheduler_config, dict) or not scheduler_config.get('name'):
            return optimizer

        scheduler_name = scheduler_config['name'].lower()
        scheduler_options = scheduler_config.get('options', {})

        if scheduler_name == 'reduceonplateau':
            monitor_metric = getattr(self.args_task, 'monitor', 'val_total_loss')
            patience = scheduler_options.get('patience', getattr(self.args_task, 'patience', 10) // 2)
            factor = scheduler_options.get('factor', 0.1)
            mode = scheduler_options.get('mode', 'min')

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, factor=factor, patience=patience
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': monitor_metric,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        elif scheduler_name == 'cosine':
            max_epochs = getattr(self.trainer, 'max_epochs', None) or getattr(self.args_task, 'max_epochs', 100)
            t_max = scheduler_options.get('T_max', max_epochs)
            eta_min = scheduler_options.get('eta_min', 0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        elif scheduler_name == 'step':
            step_size = scheduler_options.get('step_size', 10)
            gamma = scheduler_options.get('gamma', 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}]


if __name__ == '__main__':
    """Unit tests for MultiTaskLightningModule."""
    from argparse import Namespace
    import torch

    # Mock network
    class MockNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 64)

        def forward(self, x, file_id, task_id):
            features = self.linear(x.mean(dim=1))  # (B, 64)
            return {
                'classification': torch.randn(x.size(0), 5),
                'rul_prediction': torch.randn(x.size(0), 1),
                'anomaly_detection': torch.randn(x.size(0), 1)
            }

    # Mock metadata
    metadata = {
        0: {'Name': 'dataset1', 'Label': 4},
        1: {'Name': 'dataset2', 'Label': 2}
    }

    # Test configuration
    args_task = Namespace(
        enabled_tasks=['classification', 'rul_prediction', 'anomaly_detection'],
        task_weights={'classification': 1.0, 'rul_prediction': 0.5, 'anomaly_detection': 0.3},
        optimizer='adam',
        lr=0.001,
        weight_decay=0.0001,
        regularization={'l2': 1e-5}
    )

    args_model = Namespace(output_dim=64)
    args_data = Namespace(batch_size=32)
    args_trainer = Namespace(gpus=0)
    args_environment = Namespace(project='test')

    # Create module
    network = MockNetwork()
    module = MultiTaskLightningModule(
        network=network,
        args_data=args_data,
        args_model=args_model,
        args_task=args_task,
        args_trainer=args_trainer,
        args_environment=args_environment,
        metadata=metadata
    )

    # Test batch
    batch = {
        'x': torch.randn(4, 100, 10),
        'file_id': torch.tensor([0]),
        'y': torch.randint(0, 5, (4,)),
        'rul': torch.randn(4),
        'anomaly': torch.randint(0, 2, (4,)).float()
    }

    print("=== Testing MultiTaskLightningModule ===")

    # Test forward pass
    outputs = module.forward(batch)
    print(f"Forward pass outputs: {list(outputs.keys())}")
    assert 'classification' in outputs
    assert 'rul_prediction' in outputs
    assert 'anomaly_detection' in outputs

    # Test training step
    loss = module.training_step(batch, 0)
    print(f"Training loss: {loss}")
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad

    # Test validation step
    module.validation_step(batch, 0)
    print("Validation step completed")

    # Test test step
    module.test_step(batch, 0)
    print("Test step completed")

    # Test optimizer configuration
    optimizer = module.configure_optimizers()
    print(f"Optimizer configured: {type(optimizer)}")

    print("=== All tests passed! ===")
