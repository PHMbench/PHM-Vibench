"""
Flow Pretraining Task for PHM Foundation Models

This task implements Flow-based generative pretraining for unsupervised learning
of robust representations in industrial vibration signal analysis. It integrates
the M_04_ISFM_Flow model with PHM-Vibench's task factory framework.

Author: PHM-Vibench Team  
Date: 2025-09-02
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint

from ...Default_task import Default_task
from ... import register_task
from .flow_contrastive_loss import FlowContrastiveLoss
from .flow_metrics import FlowMetrics


@register_task("flow_pretrain", "pretrain")
class FlowPretrainTask(Default_task):
    """
    Flow-based pretraining task for unsupervised foundation model training.
    
    This task implements Flow generative modeling (RectifiedFlow) for learning
    robust signal representations. It extends Default_task and integrates with 
    the PHM-Vibench framework's factory pattern.
    
    Key features:
    - Flow-based generative modeling with RectifiedFlow
    - Optional joint training with contrastive learning
    - Support for conditional and unconditional generation
    - Multi-scale validation (small/medium/full datasets)
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
        Initialize the Flow pretraining task.
        
        Parameters
        ----------
        network : nn.Module
            The M_04_ISFM_Flow model to be trained
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
        
        # Flow pretraining configuration - enhanced parsing
        self.use_conditional = getattr(args_task, 'use_conditional', True)
        self.generation_mode = getattr(args_task, 'generation_mode', 'conditional')  # 'conditional' or 'unconditional'
        
        # Training parameters with validation
        self.num_steps = max(getattr(args_task, 'num_steps', 1000), 100)  # Flow sampling steps (min 100)
        self.sigma_min = max(getattr(args_task, 'sigma_min', 0.001), 1e-6)
        self.sigma_max = min(getattr(args_task, 'sigma_max', 1.0), 10.0)
        
        # Flow model specific parameters
        self.flow_lr = getattr(args_task, 'flow_lr', 1e-4)  # Flow model learning rate
        self.use_ema = getattr(args_task, 'use_ema', False)  # Exponential moving average
        self.ema_decay = getattr(args_task, 'ema_decay', 0.999) if self.use_ema else None
        
        # Validation configuration
        self.validate_generation = getattr(args_task, 'validate_generation', True)
        self.generation_samples = getattr(args_task, 'generation_samples', 10)
        
        # Contrastive learning configuration
        self.use_contrastive = getattr(args_task, 'use_contrastive', False)
        self.flow_weight = getattr(args_task, 'flow_weight', 1.0)  # λ_flow
        self.contrastive_weight = getattr(args_task, 'contrastive_weight', 0.1)  # λ_contrastive
        self.contrastive_temperature = getattr(args_task, 'contrastive_temperature', 0.1)
        self.use_gradient_balancing = getattr(args_task, 'use_gradient_balancing', True)
        
        # Initialize FlowContrastiveLoss if enabled
        self.flow_contrastive_loss = None
        if self.use_contrastive:
            augmentation_config = {
                'noise_std': getattr(args_task, 'augmentation_noise_std', 0.01),
                'jitter_std': getattr(args_task, 'augmentation_jitter_std', 0.03),
                'scaling_range': getattr(args_task, 'augmentation_scaling_range', (0.8, 1.2))
            }
            
            self.flow_contrastive_loss = FlowContrastiveLoss(
                flow_weight=self.flow_weight,
                contrastive_weight=self.contrastive_weight,
                contrastive_temperature=self.contrastive_temperature,
                projection_dim=getattr(args_task, 'projection_dim', 128),
                hidden_dim=getattr(args_task, 'contrastive_hidden_dim', 256),
                use_gradient_balancing=self.use_gradient_balancing,
                augmentation_config=augmentation_config
            )
        
        # Initialize FlowMetrics for comprehensive monitoring
        self.flow_metrics = FlowMetrics(
            device=self.device,
            enable_visualization=getattr(args_task, 'enable_visualization', True),
            save_plots=getattr(args_task, 'save_plots', False),
            plot_dir=getattr(args_task, 'plot_dir', "./flow_metrics_plots"),
            track_memory=getattr(args_task, 'track_memory', True),
            track_gradients=getattr(args_task, 'track_gradients', True)
        )
        
        # Task registration validation
        if not hasattr(self, 'network'):
            raise ValueError("FlowPretrainTask requires a network (M_04_ISFM_Flow model)")
        
        # Verify Flow model compatibility
        if hasattr(self.network, 'flow_model'):
            print(f"   ✅ 检测到Flow模型组件")
        else:
            print(f"   ⚠️  警告: 网络可能不是Flow模型")
        
        print(f"🚀 初始化FlowPretrainTask:")
        print(f"   - 条件生成: {self.use_conditional}")
        print(f"   - 生成模式: {self.generation_mode}")
        print(f"   - Flow步数: {self.num_steps}")
        print(f"   - 对比学习: {self.use_contrastive}")
        if self.use_contrastive:
            print(f"   - Flow权重: {self.flow_weight}")
            print(f"   - 对比权重: {self.contrastive_weight}")
            print(f"   - 梯度平衡: {self.use_gradient_balancing}")
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the M_04_ISFM_Flow model for training.
        
        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Input batch containing 'x' and 'file_id'
            Expected format:
            - 'x': Input signals (B, L, C) 
            - 'file_id': List of file identifiers for conditional encoding
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Flow model outputs including:
            - 'velocity': Predicted velocity field
            - 'flow_loss': Flow reconstruction loss
            - 'x_original': Original input
            - 'condition_features': Encoded conditional features
        """
        x = batch['x']  # Shape: (B, L, C)
        file_ids = batch['file_id']  # List of file identifiers
        
        # Store last file_ids for quality assessment
        self.last_file_ids = file_ids
        
        # Validate input shape
        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D input (B, L, C), got shape {x.shape}")
        
        # Forward through M_04_ISFM_Flow model with conditional generation support
        flow_outputs = self.network(
            x=x, 
            file_ids=file_ids if self.use_conditional else None,
            return_loss=True  # Always compute loss during training
        )
        
        # Add task-specific information
        flow_outputs.update({
            'generation_mode': self.generation_mode,
            'use_conditional': self.use_conditional,
            'batch_size': x.shape[0]
        })
        
        return flow_outputs
    
    def training_step(self, batch, batch_idx):
        """
        Training step for Flow pretraining with optional contrastive learning.
        
        Supports both pure Flow training and joint Flow-Contrastive training
        based on use_contrastive configuration.
        """
        # Track training performance
        self.flow_metrics.track_training_speed()
        self.flow_metrics.track_gpu_memory()
        
        # Forward pass through Flow model
        flow_outputs = self.forward(batch)
        
        if self.use_contrastive and self.flow_contrastive_loss is not None:
            # Joint Flow-Contrastive training
            loss_results = self.flow_contrastive_loss(flow_outputs)
            
            # Extract individual losses
            total_loss = loss_results['total_loss']
            flow_loss = loss_results['flow_loss']
            contrastive_loss = loss_results['contrastive_loss']
            
            # Log all losses
            self.log('train_total_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log('train_flow_loss', flow_loss, prog_bar=False, on_step=True, on_epoch=True)
            self.log('train_contrastive_loss', contrastive_loss, prog_bar=False, on_step=True, on_epoch=True)
            self.log('train_loss', total_loss, prog_bar=True)  # For compatibility
            
            # Update FlowMetrics tracking
            self.flow_metrics.update_loss_tracking(
                flow_loss=flow_loss,
                contrastive_loss=contrastive_loss,
                total_loss=total_loss
            )
            
            # Log loss weights for monitoring
            if batch_idx % 200 == 0:
                self.log('flow_weight', self.flow_weight, on_step=True)
                self.log('contrastive_weight', self.contrastive_weight, on_step=True)
            
            return total_loss
            
        else:
            # Pure Flow training (original behavior)
            flow_loss = flow_outputs.get('flow_loss', flow_outputs.get('loss', None))
            
            if flow_loss is None:
                # Fallback: compute simple MSE loss between input and reconstructed
                x_original = flow_outputs.get('x_original', batch['x'])
                velocity = flow_outputs.get('velocity', x_original)
                flow_loss = nn.MSELoss()(velocity, x_original)
            
            # Log Flow loss
            self.log('train_flow_loss', flow_loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log('train_loss', flow_loss, prog_bar=True)  # For compatibility
            
            # Update FlowMetrics tracking for Flow-only training
            self.flow_metrics.update_loss_tracking(flow_loss=flow_loss)
            
            return flow_loss
        
        # Log batch information periodically
        if batch_idx % 100 == 0:
            batch_size = flow_outputs.get('batch_size', batch['x'].shape[0])
            self.log('batch_size', float(batch_size), on_step=True)
            if self.use_contrastive:
                self.log('training_mode', 1.0, on_step=True)  # 1.0 for joint training
            else:
                self.log('training_mode', 0.0, on_step=True)  # 0.0 for Flow-only
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for Flow pretraining with optional contrastive learning.
        
        Computes validation loss and basic generation quality metrics.
        Supports both pure Flow validation and joint Flow-Contrastive validation.
        """
        # Forward pass through Flow model
        flow_outputs = self.forward(batch)
        
        if self.use_contrastive and self.flow_contrastive_loss is not None:
            # Joint Flow-Contrastive validation
            with torch.no_grad():
                loss_results = self.flow_contrastive_loss(flow_outputs)
            
            # Extract individual losses
            total_val_loss = loss_results['total_loss']
            flow_val_loss = loss_results['flow_loss']
            contrastive_val_loss = loss_results['contrastive_loss']
            
            # Log all validation losses
            self.log('val_total_loss', total_val_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('val_flow_loss', flow_val_loss, prog_bar=False, on_step=False, on_epoch=True)
            self.log('val_contrastive_loss', contrastive_val_loss, prog_bar=False, on_step=False, on_epoch=True)
            self.log('val_loss', total_val_loss, prog_bar=True)  # For compatibility
            
            return {
                'val_loss': total_val_loss,
                'val_flow_loss': flow_val_loss,
                'val_contrastive_loss': contrastive_val_loss,
                'flow_outputs': flow_outputs,
                'loss_results': loss_results
            }
            
        else:
            # Pure Flow validation (original behavior)
            val_loss = flow_outputs.get('flow_loss', flow_outputs.get('loss', None))
            
            if val_loss is None:
                # Fallback: compute simple MSE loss
                x_original = flow_outputs.get('x_original', batch['x'])
                velocity = flow_outputs.get('velocity', x_original)
                val_loss = nn.MSELoss()(velocity, x_original)
            
            # Log validation metrics
            self.log('val_flow_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('val_loss', val_loss, prog_bar=True)  # For compatibility
            
            return {'val_loss': val_loss, 'flow_outputs': flow_outputs}
    
    def on_after_backward(self):
        """Hook called after backward pass - track gradient norms."""
        super().on_after_backward()
        if hasattr(self, 'flow_metrics'):
            self.flow_metrics.track_gradient_norms(self.network)
    
    def validation_epoch_end(self, outputs):
        """
        Enhanced validation epoch end with quality assessment.
        
        Periodically generates samples and computes quality metrics.
        """
        super().validation_epoch_end(outputs) if hasattr(super(), 'validation_epoch_end') else None
        
        # Perform quality assessment every N epochs
        if self.current_epoch % getattr(self, 'quality_assessment_frequency', 10) == 0:
            try:
                # Generate samples for quality assessment
                with torch.no_grad():
                    batch_size = min(32, getattr(self, 'generation_samples', 10))
                    
                    # Get a real batch for comparison (use last validation batch)
                    if outputs and 'flow_outputs' in outputs[-1]:
                        real_samples = outputs[-1]['flow_outputs'].get('x_original')
                        if real_samples is not None and real_samples.shape[0] >= batch_size:
                            real_samples = real_samples[:batch_size]
                            
                            # Generate corresponding samples
                            file_ids = None  # Use unconditional generation for assessment
                            if self.use_conditional and hasattr(self, 'last_file_ids'):
                                file_ids = self.last_file_ids[:batch_size]
                            
                            generated_samples = self.generate_samples(
                                batch_size=batch_size,
                                file_ids=file_ids,
                                num_steps=min(self.num_steps, 50)  # Use fewer steps for efficiency
                            )
                            
                            # Compute quality metrics
                            ks_stat, ks_p_value = self.flow_metrics.compute_ks_test(
                                real_samples, generated_samples
                            )
                            
                            spectral_sim = self.flow_metrics.compute_spectral_similarity(
                                real_samples, generated_samples
                            )
                            
                            snr_score = self.flow_metrics.compute_snr_score(
                                real_samples, generated_samples
                            )
                            
                            diversity_score = self.flow_metrics.compute_diversity_score(
                                generated_samples
                            )
                            
                            # Log quality metrics
                            self.log('val_ks_statistic', ks_stat, on_epoch=True)
                            self.log('val_spectral_similarity', spectral_sim, on_epoch=True)
                            self.log('val_snr_score', snr_score, on_epoch=True)
                            self.log('val_diversity_score', diversity_score, on_epoch=True)
                            
                            # Create visualization
                            if self.current_epoch % (getattr(self, 'quality_assessment_frequency', 10) * 2) == 0:
                                fig = self.flow_metrics.create_comparison_plots(
                                    real_samples, generated_samples, num_samples=3
                                )
                                if fig is not None:
                                    plt.close(fig)  # Clean up
                            
                            print(f"📊 质量评估 (Epoch {self.current_epoch}):")
                            print(f"   KS统计量: {ks_stat:.4f}")
                            print(f"   频谱相似度: {spectral_sim:.4f}")
                            print(f"   SNR分数: {snr_score:.4f}")
                            print(f"   多样性分数: {diversity_score:.4f}")
                
            except Exception as e:
                print(f"⚠️  质量评估失败: {e}")
        
        # Log performance summary periodically
        if self.current_epoch % 20 == 0:
            performance_summary = self.flow_metrics.get_performance_summary()
            for key, value in performance_summary.items():
                if isinstance(value, (int, float)):
                    self.log(f'perf_{key}', float(value), on_epoch=True)
    
    def generate_samples(
        self, 
        batch_size: int, 
        file_ids: Optional[List[str]] = None,
        mode: Optional[str] = None,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate samples using Flow model with conditional/unconditional modes.
        
        Parameters
        ----------
        batch_size : int
            Number of samples to generate
        file_ids : Optional[List[str]]
            File identifiers for conditional generation (ignored in unconditional mode)
        mode : Optional[str]
            Generation mode ('conditional' or 'unconditional'). 
            If None, uses self.generation_mode
        num_steps : Optional[int]
            Number of sampling steps. If None, uses self.num_steps
            
        Returns
        -------
        torch.Tensor
            Generated samples with shape (batch_size, sequence_length, channels)
        """
        # Determine generation mode
        gen_mode = mode if mode is not None else self.generation_mode
        num_steps = num_steps if num_steps is not None else self.num_steps
        
        # Validate generation mode
        if gen_mode not in ['conditional', 'unconditional']:
            raise ValueError(f"Generation mode must be 'conditional' or 'unconditional', got {gen_mode}")
        
        # Set up conditional features based on mode
        if gen_mode == 'conditional':
            if not self.use_conditional:
                print("⚠️  警告: 尝试条件生成但模型未启用条件编码，切换到无条件模式")
                conditional_file_ids = None
            elif file_ids is None:
                print("⚠️  警告: 条件生成模式但未提供file_ids，切换到无条件模式")
                conditional_file_ids = None
            else:
                conditional_file_ids = file_ids
        else:  # unconditional
            conditional_file_ids = None
        
        # Generate samples using Flow model
        with torch.no_grad():
            samples = self.network.sample(
                batch_size=batch_size,
                file_ids=conditional_file_ids,
                num_steps=num_steps,
                device=self.device
            )
        
        return samples
    
    def set_generation_mode(self, mode: str) -> None:
        """
        Set the generation mode for the task.
        
        Parameters
        ----------
        mode : str
            Generation mode ('conditional' or 'unconditional')
        """
        if mode not in ['conditional', 'unconditional']:
            raise ValueError(f"Mode must be 'conditional' or 'unconditional', got {mode}")
        
        old_mode = self.generation_mode
        self.generation_mode = mode
        print(f"🔄 生成模式切换: {old_mode} -> {mode}")
        
        # Validate mode compatibility
        if mode == 'conditional' and not self.use_conditional:
            print("⚠️  警告: 设置为条件生成模式但未启用条件编码")
    
    def validate_generation_capability(self) -> Dict[str, bool]:
        """
        Validate the model's generation capabilities.
        
        Returns
        -------
        Dict[str, bool]
            Capability validation results
        """
        capabilities = {
            'conditional_generation': False,
            'unconditional_generation': False,
            'flow_model_available': False,
            'condition_encoder_available': False
        }
        
        # Check if Flow model is available
        if hasattr(self.network, 'flow_model'):
            capabilities['flow_model_available'] = True
            capabilities['unconditional_generation'] = True  # Flow model always supports unconditional
        
        # Check conditional generation capability
        if self.use_conditional and hasattr(self.network, 'condition_encoder'):
            capabilities['condition_encoder_available'] = True
            if self.network.condition_encoder is not None:
                capabilities['conditional_generation'] = True
        
        return capabilities
    
    def update_contrastive_weights(self, flow_weight: float, contrastive_weight: float):
        """
        Update loss weights for Flow and contrastive components.
        
        Parameters
        ----------
        flow_weight : float
            New Flow loss weight (λ_flow)
        contrastive_weight : float
            New contrastive loss weight (λ_contrastive)
        """
        if not self.use_contrastive:
            print("⚠️  警告: 对比学习未启用，权重更新无效")
            return
        
        old_flow = self.flow_weight
        old_contrastive = self.contrastive_weight
        
        self.flow_weight = flow_weight
        self.contrastive_weight = contrastive_weight
        
        # Update FlowContrastiveLoss weights if available
        if self.flow_contrastive_loss is not None:
            self.flow_contrastive_loss.update_weights(flow_weight, contrastive_weight)
        
        print(f"🔄 损失权重已更新:")
        print(f"   Flow: {old_flow:.3f} -> {flow_weight:.3f}")
        print(f"   Contrastive: {old_contrastive:.3f} -> {contrastive_weight:.3f}")
    
    def get_contrastive_status(self) -> Dict[str, Any]:
        """
        Get current contrastive learning configuration status.
        
        Returns
        -------
        Dict[str, Any]
            Status information including weights, settings, and capabilities
        """
        status = {
            'use_contrastive': self.use_contrastive,
            'flow_weight': self.flow_weight,
            'contrastive_weight': self.contrastive_weight,
            'contrastive_temperature': self.contrastive_temperature,
            'use_gradient_balancing': self.use_gradient_balancing,
            'flow_contrastive_loss_available': self.flow_contrastive_loss is not None
        }
        
        if self.flow_contrastive_loss is not None:
            status.update({
                'projection_dim': self.flow_contrastive_loss.projection_dim,
                'hidden_dim': self.flow_contrastive_loss.hidden_dim,
                'projection_head_initialized': self.flow_contrastive_loss.projection_head is not None
            })
        
        return status
    
    def configure_optimizers(self):
        """
        Configure optimizer for Flow pretraining.
        Uses Adam optimizer with configurable learning rate.
        """
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=getattr(self.args_task, 'lr', 1e-4),
            weight_decay=getattr(self.args_task, 'weight_decay', 1e-5)
        )
        
        # Optional learning rate scheduler
        if getattr(self.args_task, 'use_scheduler', False):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=getattr(self.args_task, 'max_epochs', 100)
            )
            return [optimizer], [scheduler]
        
        return optimizer
    
    # ========================================================
    # Pipeline_02_pretrain_fewshot Compatibility Methods
    # ========================================================
    
    def get_pretrained_backbone(self) -> nn.Module:
        """
        Extract pretrained backbone for Pipeline_02_pretrain_fewshot compatibility.
        
        Returns the pretrained network/backbone that can be loaded into 
        subsequent few-shot learning tasks.
        
        Returns
        -------
        nn.Module
            The pretrained backbone network
        """
        return self.network
    
    def save_pretrained_state(self, filepath: str) -> Dict[str, Any]:
        """
        Save pretrained state for Pipeline_02_pretrain_fewshot compatibility.
        
        Saves both the model state and task-specific configuration that
        can be loaded for few-shot learning stages.
        
        Parameters
        ----------
        filepath : str
            Path to save the pretrained state
            
        Returns
        -------
        Dict[str, Any]
            Saved state information including paths and metadata
        """
        # Prepare state dictionary with task-specific information
        pretrained_state = {
            'model_state_dict': self.network.state_dict(),
            'task_type': 'flow_pretrain',
            'task_config': {
                'use_conditional': self.use_conditional,
                'generation_mode': self.generation_mode,
                'num_steps': self.num_steps,
                'sigma_min': self.sigma_min,
                'sigma_max': self.sigma_max,
                'use_contrastive': self.use_contrastive,
                'flow_weight': self.flow_weight,
                'contrastive_weight': self.contrastive_weight,
            },
            'hyperparameters': dict(self.hparams),
            'metadata': {
                'training_epoch': self.current_epoch,
                'model_architecture': self.args_model.name if hasattr(self.args_model, 'name') else 'M_04_ISFM_Flow',
                'data_config': vars(self.args_data) if hasattr(self, 'args_data') else {},
                'capabilities': self.validate_generation_capability()
            }
        }
        
        # Save the state
        torch.save(pretrained_state, filepath)
        
        print(f"🔄 Flow预训练状态已保存:")
        print(f"   路径: {filepath}")
        print(f"   训练轮次: {self.current_epoch}")
        print(f"   模型架构: {pretrained_state['metadata']['model_architecture']}")
        print(f"   生成模式: {self.generation_mode}")
        
        return {
            'filepath': filepath,
            'state_info': pretrained_state['metadata'],
            'success': True
        }
    
    def load_pretrained_state(self, filepath: str, strict: bool = True) -> Dict[str, Any]:
        """
        Load pretrained state for Pipeline_02_pretrain_fewshot compatibility.
        
        Loads model state and task configuration from a pretrained checkpoint,
        enabling continuation in few-shot learning pipelines.
        
        Parameters
        ----------
        filepath : str
            Path to the pretrained state file
        strict : bool, default=True
            Whether to strictly enforce state_dict key matching
            
        Returns
        -------
        Dict[str, Any]
            Loading results including success status and loaded configuration
        """
        try:
            # Load the pretrained state
            pretrained_state = torch.load(filepath, map_location=self.device, weights_only=False)
            
            # Verify compatibility
            if pretrained_state.get('task_type') != 'flow_pretrain':
                print(f"⚠️  警告: 加载的检查点不是Flow预训练任务类型 (实际: {pretrained_state.get('task_type')})")
            
            # Load model state
            missing_keys, unexpected_keys = self.network.load_state_dict(
                pretrained_state['model_state_dict'], 
                strict=strict
            )
            
            # Restore task configuration
            task_config = pretrained_state.get('task_config', {})
            for key, value in task_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            # Update FlowContrastiveLoss if needed and available
            if self.use_contrastive and self.flow_contrastive_loss is not None:
                self.flow_contrastive_loss.update_weights(
                    task_config.get('flow_weight', self.flow_weight),
                    task_config.get('contrastive_weight', self.contrastive_weight)
                )
            
            load_info = {
                'success': True,
                'source_epoch': pretrained_state['metadata']['training_epoch'],
                'model_architecture': pretrained_state['metadata']['model_architecture'],
                'missing_keys': len(missing_keys),
                'unexpected_keys': len(unexpected_keys),
                'loaded_config': task_config
            }
            
            print(f"✅ Flow预训练状态加载成功:")
            print(f"   源文件: {filepath}")
            print(f"   源训练轮次: {load_info['source_epoch']}")
            print(f"   模型架构: {load_info['model_architecture']}")
            print(f"   缺失键: {load_info['missing_keys']}")
            print(f"   意外键: {load_info['unexpected_keys']}")
            
            return load_info
            
        except Exception as e:
            error_info = {
                'success': False,
                'error': str(e),
                'filepath': filepath
            }
            print(f"❌ Flow预训练状态加载失败: {e}")
            return error_info
    
    def prepare_for_fewshot_transfer(self) -> Dict[str, Any]:
        """
        Prepare the Flow model for few-shot transfer learning.
        
        Configures the model for effective transfer to few-shot learning tasks
        by freezing/unfreezing appropriate layers and preparing transfer metadata.
        
        Returns
        -------
        Dict[str, Any]
            Transfer preparation information including frozen layers and recommendations
        """
        transfer_info = {
            'frozen_layers': [],
            'unfrozen_layers': [],
            'transfer_recommendations': {},
            'model_capabilities': self.validate_generation_capability()
        }
        
        # Determine freezing strategy based on model architecture
        if hasattr(self.network, 'flow_model'):
            # Flow model layers - typically freeze Flow-specific components
            # Keep encoder unfrozen for adaptation
            flow_params = []
            encoder_params = []
            other_params = []
            
            for name, param in self.network.named_parameters():
                if 'flow_model' in name or 'velocity_network' in name:
                    flow_params.append(name)
                elif 'encoder' in name or 'backbone' in name:
                    encoder_params.append(name)
                else:
                    other_params.append(name)
            
            # Default strategy: freeze Flow-specific components, unfreeze encoders
            freeze_flow_layers = getattr(self.args_task, 'freeze_flow_in_fewshot', True)
            freeze_encoder_layers = getattr(self.args_task, 'freeze_encoder_in_fewshot', False)
            
            if freeze_flow_layers:
                for name in flow_params:
                    if hasattr(self.network, name.split('.')[0]):
                        param = dict(self.network.named_parameters())[name]
                        param.requires_grad = False
                        transfer_info['frozen_layers'].append(name)
            
            if freeze_encoder_layers:
                for name in encoder_params:
                    if hasattr(self.network, name.split('.')[0]):
                        param = dict(self.network.named_parameters())[name]
                        param.requires_grad = False
                        transfer_info['frozen_layers'].append(name)
            else:
                transfer_info['unfrozen_layers'].extend(encoder_params)
            
            # Keep other layers unfrozen by default
            transfer_info['unfrozen_layers'].extend(other_params)
        
        # Transfer recommendations
        transfer_info['transfer_recommendations'] = {
            'suggested_lr_reduction': 0.1,  # Reduce learning rate for fine-tuning
            'suggested_weight_decay': getattr(self.args_task, 'weight_decay', 1e-5) * 0.1,
            'suggested_epochs': 20,  # Fewer epochs for fine-tuning
            'use_conditional_generation': self.use_conditional,
            'generation_quality_metrics': ['spectral_similarity', 'snr_score', 'diversity_score']
        }
        
        print(f"🔧 Few-shot转移准备完成:")
        print(f"   冻结层数: {len(transfer_info['frozen_layers'])}")
        print(f"   可训练层数: {len(transfer_info['unfrozen_layers'])}")
        print(f"   建议学习率缩放: {transfer_info['transfer_recommendations']['suggested_lr_reduction']}")
        
        return transfer_info
    
    def get_pipeline_checkpoint_callback(self) -> ModelCheckpoint:
        """
        Get optimized checkpoint callback for Pipeline_02_pretrain_fewshot.
        
        Returns a configured ModelCheckpoint callback that saves checkpoints
        in a format compatible with the multi-stage pipeline.
        
        Returns
        -------
        ModelCheckpoint
            Configured checkpoint callback for pipeline compatibility
        """
        # Create checkpoint callback optimized for pipeline transfer
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',  # Monitor validation loss for best model
            mode='min',
            save_top_k=1,  # Save only the best checkpoint
            save_last=True,  # Also save the last checkpoint
            filename='flow_pretrain-{epoch:02d}-{val_loss:.4f}',
            auto_insert_metric_name=False,
            save_weights_only=False,  # Save full checkpoint for pipeline compatibility
            verbose=True
        )
        
        return checkpoint_callback
    
    def extract_feature_representations(self, dataloader) -> Dict[str, torch.Tensor]:
        """
        Extract feature representations for pipeline analysis.
        
        Extracts and returns feature representations from the pretrained Flow model,
        useful for analyzing transfer learning effectiveness.
        
        Parameters
        ----------
        dataloader
            DataLoader to extract features from
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing extracted features and metadata
        """
        self.eval()
        features = {
            'encoded_features': [],
            'flow_features': [],
            'condition_features': [],
            'file_ids': [],
            'original_signals': []
        }
        
        with torch.no_grad():
            for batch in dataloader:
                # Get features from different network components
                x = batch['x'].to(self.device)
                file_ids = batch['file_id']
                
                # Store original signals
                features['original_signals'].append(x.cpu())
                features['file_ids'].extend(file_ids if isinstance(file_ids, list) else [file_ids])
                
                # Extract encoder features if available
                if hasattr(self.network, 'encoder') or hasattr(self.network, 'backbone'):
                    try:
                        # Try different encoder access patterns
                        if hasattr(self.network, 'encode'):
                            encoded = self.network.encode(x, file_ids)
                        elif hasattr(self.network, 'encoder'):
                            encoded = self.network.encoder(x)
                        else:
                            encoded = x  # Fallback to input if no encoder found
                        
                        features['encoded_features'].append(encoded.cpu())
                    except Exception as e:
                        print(f"⚠️  警告: 特征提取失败: {e}")
                        features['encoded_features'].append(x.cpu())
                
                # Extract condition features if conditional model
                if self.use_conditional and hasattr(self.network, 'condition_encoder'):
                    try:
                        condition_feats = self.network.condition_encoder(file_ids)
                        features['condition_features'].append(condition_feats.cpu())
                    except Exception:
                        pass
                
                # Limit extraction to avoid memory issues
                if len(features['original_signals']) * x.shape[0] > 1000:  # Limit to ~1000 samples
                    break
        
        # Concatenate collected features
        for key in ['encoded_features', 'flow_features', 'original_signals']:
            if features[key]:
                features[key] = torch.cat(features[key], dim=0)
        
        if features['condition_features']:
            features['condition_features'] = torch.cat(features['condition_features'], dim=0)
        
        print(f"📊 特征提取完成:")
        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape}")
            elif isinstance(value, list):
                print(f"   {key}: {len(value)} items")
        
        return features
    
    def validate_pipeline_compatibility(self) -> Dict[str, Any]:
        """
        Validate compatibility with Pipeline_02_pretrain_fewshot.
        
        Performs comprehensive checks to ensure the Flow pretraining task
        can work seamlessly with the multi-stage pipeline.
        
        Returns
        -------
        Dict[str, Any]
            Validation results including compatibility status and recommendations
        """
        validation_results = {
            'compatible': True,
            'issues': [],
            'warnings': [],
            'recommendations': [],
            'capabilities': {}
        }
        
        # Check model architecture compatibility
        if not hasattr(self.network, 'state_dict'):
            validation_results['issues'].append("Network lacks state_dict method")
            validation_results['compatible'] = False
        
        # Check checkpoint saving/loading capability
        try:
            # Test state dict serialization
            state_dict = self.network.state_dict()
            validation_results['capabilities']['state_dict_serializable'] = True
        except Exception as e:
            validation_results['issues'].append(f"State dict serialization failed: {e}")
            validation_results['compatible'] = False
        
        # Check Flow model specific requirements
        if hasattr(self.network, 'flow_model'):
            validation_results['capabilities']['flow_model_available'] = True
        else:
            validation_results['warnings'].append("Flow model component not clearly identified")
        
        # Check conditional generation capability
        validation_results['capabilities'].update(self.validate_generation_capability())
        
        # Check optimization configuration
        if hasattr(self, 'configure_optimizers'):
            validation_results['capabilities']['optimizer_configurable'] = True
        else:
            validation_results['warnings'].append("Optimizer configuration method missing")
        
        # Pipeline-specific recommendations
        validation_results['recommendations'] = [
            "Use val_loss monitoring for checkpoint selection",
            "Enable save_weights_only=False for full pipeline compatibility", 
            "Consider freezing Flow-specific layers during few-shot transfer",
            "Validate generation quality before pipeline transitions",
            f"Current generation mode: {self.generation_mode}",
            f"Contrastive learning enabled: {self.use_contrastive}"
        ]
        
        # Overall compatibility assessment
        if len(validation_results['issues']) == 0:
            validation_results['status'] = 'FULLY_COMPATIBLE'
        elif len(validation_results['issues']) <= 2:
            validation_results['status'] = 'MOSTLY_COMPATIBLE'
        else:
            validation_results['status'] = 'COMPATIBILITY_ISSUES'
            validation_results['compatible'] = False
        
        print(f"🔍 Pipeline兼容性验证: {validation_results['status']}")
        if validation_results['issues']:
            print(f"   问题: {len(validation_results['issues'])} 个")
            for issue in validation_results['issues']:
                print(f"     - {issue}")
        if validation_results['warnings']:
            print(f"   警告: {len(validation_results['warnings'])} 个")
            for warning in validation_results['warnings']:
                print(f"     - {warning}")
        
        return validation_results


# Backward compatibility alias for task factory
task = FlowPretrainTask