"""
Masked Reconstruction Task for Pretraining PHM Foundation Models

This task implements masked signal reconstruction for unsupervised pretraining
of PHM foundation models. It follows the PHM-Vibench framework's task factory pattern.

Author: PHM-Vibench Team
Date: 2025-08-18
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Tuple, Optional
import numpy as np

from ...Default_task import Default_task
from ...Components.prediction_loss import Signal_mask_Loss
from ....utils.masking import add_mask
from ... import register_task


@register_task("masked_reconstruction", "pretrain")
class MaskedReconstructionTask(Default_task):
    """
    Masked reconstruction task for unsupervised pretraining.
    
    This task implements a masked autoencoder-style pretraining approach where
    portions of input signals are masked and the model learns to reconstruct them.
    
    The task extends Default_task and integrates with the PHM-Vibench framework's
    factory pattern for consistent task management.
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
        """
        Initialize the masked reconstruction task.
        
        Parameters
        ----------
        network : nn.Module
            The backbone network to be trained
        args_data : Any
            Data configuration namespace
        args_model : Any
            Model configuration namespace
        args_task : Any
            Task configuration namespace
        args_trainer : Any
            Trainer configuration namespace
        args_environment : Any
            Environment configuration namespace
        metadata : Any
            Dataset metadata
        """
        super().__init__(
            network=network,
            args_data=args_data,
            args_model=args_model,
            args_task=args_task,
            args_trainer=args_trainer,
            args_environment=args_environment,
            metadata=metadata
        )
        
        # Pretraining configuration
        self.mask_ratio = getattr(args_task, 'mask_ratio', 0.15)
        self.forecast_part = getattr(args_task, 'forecast_part', 0.1)
        
        # Loss function for reconstruction
        self.reconstruction_loss = nn.MSELoss()
        
        # Initialize prediction loss if available
        pred_cfg = getattr(args_task, 'pred_cfg', args_task)
        try:
            self.pred_loss_fn = Signal_mask_Loss(pred_cfg)
            self.use_signal_mask_loss = True
        except Exception:
            # Fallback to simple MSE if Signal_mask_Loss fails
            self.use_signal_mask_loss = False
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the network for reconstruction.
        
        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Input batch containing 'x' and 'file_id'
            
        Returns
        -------
        torch.Tensor
            Reconstructed signal
        """
        x = batch['x']
        file_id = batch['file_id']
        
        # Use prediction task_id to get reconstruction output
        return self.network(x, file_id, task_id='prediction')
    
    def _compute_reconstruction_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss on masked positions.
        
        Parameters
        ----------
        predictions : torch.Tensor
            Predicted signal values
        targets : torch.Tensor
            Ground truth signal values
        mask : torch.Tensor
            Boolean mask indicating masked positions
            
        Returns
        -------
        torch.Tensor
            Reconstruction loss
        """
        if mask.sum() > 0:
            return self.reconstruction_loss(predictions[mask], targets[mask])
        else:
            # Return zero loss with gradients if no masked positions
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
    def _compute_additional_metrics(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute additional metrics for pretraining evaluation.
        
        Parameters
        ----------
        predictions : torch.Tensor
            Predicted signal values
        targets : torch.Tensor
            Ground truth signal values
        mask : torch.Tensor
            Boolean mask indicating masked positions
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of computed metrics
        """
        metrics = {}
        
        with torch.no_grad():
            # Signal correlation on masked regions
            if mask.sum() > 0:
                pred_masked = predictions[mask].flatten()
                true_masked = targets[mask].flatten()
                
                if len(pred_masked) > 1:
                    # Compute correlation coefficient
                    pred_mean = pred_masked.mean()
                    true_mean = true_masked.mean()
                    
                    pred_centered = pred_masked - pred_mean
                    true_centered = true_masked - true_mean
                    
                    numerator = (pred_centered * true_centered).sum()
                    denominator = torch.sqrt(
                        (pred_centered ** 2).sum() * (true_centered ** 2).sum()
                    )
                    
                    if denominator > 1e-8:
                        correlation = numerator / denominator
                    else:
                        correlation = torch.tensor(0.0, device=predictions.device)
                    
                    if torch.isnan(correlation):
                        correlation = torch.tensor(0.0, device=predictions.device)
                else:
                    correlation = torch.tensor(0.0, device=predictions.device)
            else:
                correlation = torch.tensor(0.0, device=predictions.device)
            
            # Masking statistics
            mask_fraction = mask.float().mean()
            
            metrics['signal_correlation'] = correlation
            metrics['mask_fraction'] = mask_fraction
        
        return metrics
    
    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> Dict[str, torch.Tensor]:
        """
        Shared step for pretraining with masked reconstruction.
        
        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Input batch
        stage : str
            Training stage ('train', 'val', 'test')
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of computed metrics and losses
        """
        try:
            # Extract batch information
            file_id = batch['file_id'][0].item() if isinstance(batch['file_id'], torch.Tensor) else batch['file_id']
            batch.update({'file_id': file_id})
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error processing batch: {e}")
        
        # Get original signal
        signal = batch['x']  # (B, L, C)
        
        # Apply masking
        x_masked, total_mask = add_mask(signal, self.forecast_part, self.mask_ratio)
        
        # Create masked batch
        masked_batch = batch.copy()
        masked_batch['x'] = x_masked
        
        # Forward pass for reconstruction
        x_reconstructed = self.forward(masked_batch)
        
        # Compute reconstruction loss
        reconstruction_loss = self._compute_reconstruction_loss(x_reconstructed, signal, total_mask)
        
        # Compute additional metrics
        additional_metrics = self._compute_additional_metrics(x_reconstructed, signal, total_mask)
        
        # Prepare step metrics
        step_metrics = {
            f"{stage}_reconstruction_loss": reconstruction_loss,
            f"{stage}_loss": reconstruction_loss,  # Main loss for framework compatibility
        }
        
        # Add additional metrics with stage prefix
        for metric_name, metric_value in additional_metrics.items():
            step_metrics[f"{stage}_{metric_name}"] = metric_value
        
        return step_metrics
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for pretraining."""
        metrics = self._shared_step(batch, "train")
        self._log_metrics(metrics, "train")
        return metrics["train_loss"]
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step for pretraining."""
        metrics = self._shared_step(batch, "val")
        self._log_metrics(metrics, "val")
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step for pretraining."""
        metrics = self._shared_step(batch, "test")
        self._log_metrics(metrics, "test")
    
    def _log_metrics(self, metrics: Dict[str, torch.Tensor], stage: str) -> None:
        """
        Log metrics to tensorboard/wandb.
        
        Parameters
        ----------
        metrics : Dict[str, torch.Tensor]
            Metrics to log
        stage : str
            Training stage
        """
        log_dict = {}
        
        for k, v in metrics.items():
            if k.startswith(stage):
                log_dict[k] = v
        
        self.log_dict(
            log_dict,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )


# Alias for backward compatibility
task = MaskedReconstructionTask
