"""
Lightning模块实现，用于将模型和任务包装为PyTorch Lightning模块
"""
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Union
import torch.optim as optim

from src.utils.metrics_utils import compute_metrics


class TaskLightningModule(pl.LightningModule):
    """任务Lightning模块，将模型和任务包装为PyTorch Lightning模块
    
    支持:
    - 多种任务类型（分类、异常检测、RUL预测）
    - 自定义损失函数和评估指标
    - 灵活的优化器和学习率调度器配置
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        task=None,
        loss_fn=None,
        task_type: str = 'classification',
        metrics: Optional[List[str]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        scheduler_config: Optional[Dict[str, Any]] = None
    ):
        """初始化模块
        
        Args:
            model: 基础模型
            task: 任务实例
            loss_fn: 损失函数
            task_type: 任务类型，可选'classification', 'anomaly', 'rul'
            metrics: 要计算的指标列表
            optimizer_config: 优化器配置
            scheduler_config: 学习率调度器配置
        """
        super().__init__()
        self.model = model
        self.task = task
        self.task_type = task_type
        self.metrics = metrics or []
        self.optimizer_config = optimizer_config or {'name': 'adam', 'lr': 0.001}
        self.scheduler_config = scheduler_config or {'use_scheduler': False}
        
        # 保存超参数
        self.save_hyperparameters(ignore=["model", "task", "loss_fn"])
        
        # 设置损失函数
        self.loss_fn = loss_fn
        if self.loss_fn is None:
            # 根据任务类型设置默认损失函数
            if task_type == 'classification':
                self.loss_fn = torch.nn.CrossEntropyLoss()
            elif task_type == 'anomaly':
                self.loss_fn = torch.nn.BCEWithLogitsLoss()
            elif task_type == 'rul':
                self.loss_fn = torch.nn.MSELoss()
            else:
                self.loss_fn = torch.nn.MSELoss()
        
        # 如果任务实例提供了损失函数，优先使用它
        if self.task is not None and hasattr(self.task, 'get_loss_fn'):
            task_loss_fn = self.task.get_loss_fn()
            if task_loss_fn is not None:
                self.loss_fn = task_loss_fn
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入数据
            
        Returns:
            模型输出
        """
        return self.model(x)
    
    def _get_predictions(self, outputs):
        """根据任务类型转换模型输出为预测结果
        
        Args:
            outputs: 模型输出
            
        Returns:
            预测结果
        """
        if self.task_type == 'classification':
            if outputs.dim() > 1 and outputs.size(-1) > 1:
                # 多分类情况，使用argmax获取预测类别
                return torch.argmax(outputs, dim=1)
            else:
                # 二分类情况，通过阈值获取预测类别
                return (torch.sigmoid(outputs) > 0.5).long()
        elif self.task_type == 'anomaly':
            # 异常检测，通过阈值获取预测类别
            return (torch.sigmoid(outputs) > 0.5).long()
        else:
            # 回归问题，直接返回模型输出
            return outputs
    
    def _get_probabilities(self, outputs):
        """根据任务类型转换模型输出为概率分布
        
        Args:
            outputs: 模型输出
            
        Returns:
            概率分布
        """
        if self.task_type == 'classification':
            if outputs.dim() > 1 and outputs.size(-1) > 1:
                # 多分类情况，使用softmax获取概率
                return F.softmax(outputs, dim=1)
            else:
                # 二分类情况，使用sigmoid获取概率
                return torch.sigmoid(outputs)
        elif self.task_type == 'anomaly':
            # 异常检测，使用sigmoid获取异常分数
            return torch.sigmoid(outputs)
        else:
            # 回归问题，直接返回模型输出
            return outputs
    
    def training_step(self, batch, batch_idx):
        """训练步骤
        
        Args:
            batch: 批次数据
            batch_idx: 批次索引
            
        Returns:
            损失值字典
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # 记录训练损失
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # 如果任务实例定义了自定义的训练步骤
        if self.task is not None and hasattr(self.task, 'training_step'):
            task_output = self.task.training_step(x, y_hat, y)
            if task_output is not None and isinstance(task_output, dict):
                # 记录任务返回的指标
                for key, value in task_output.items():
                    if key != 'loss':  # 避免覆盖主损失
                        self.log(f"train_{key}", value, on_step=False, on_epoch=True)
                
                # 如果任务返回了损失，使用它替代或组合当前损失
                if 'loss' in task_output:
                    loss = task_output['loss']
        
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        """验证步骤
        
        Args:
            batch: 批次数据
            batch_idx: 批次索引
            
        Returns:
            验证结果字典
        """
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)
        
        # 记录验证损失
        self.log("val_loss", val_loss, prog_bar=True)
        
        # 计算预测和概率
        y_pred = self._get_predictions(y_hat)
        y_prob = self._get_probabilities(y_hat)
        
        # 如果任务实例定义了自定义的验证步骤
        if self.task is not None and hasattr(self.task, 'validation_step'):
            task_output = self.task.validation_step(x, y_hat, y)
            if task_output is not None and isinstance(task_output, dict):
                # 记录任务返回的指标
                for key, value in task_output.items():
                    self.log(f"val_{key}", value)
        
        return {
            "val_loss": val_loss, 
            "y_pred": y_pred, 
            "y_true": y,
            "y_prob": y_prob
        }
    
    def test_step(self, batch, batch_idx):
        """测试步骤
        
        Args:
            batch: 批次数据
            batch_idx: 批次索引
            
        Returns:
            测试结果字典
        """
        x, y = batch
        y_hat = self(x)
        test_loss = self.loss_fn(y_hat, y)
        
        # 记录测试损失
        self.log("test_loss", test_loss)
        
        # 计算预测和概率
        y_pred = self._get_predictions(y_hat)
        y_prob = self._get_probabilities(y_hat)
        
        # 基本指标
        result = {"test_loss": test_loss}
        
        # 如果任务实例定义了自定义的测试步骤
        if self.task is not None and hasattr(self.task, 'test_step'):
            task_output = self.task.test_step(x, y_hat, y)
            if task_output is not None and isinstance(task_output, dict):
                # 记录任务返回的指标
                for key, value in task_output.items():
                    self.log(f"test_{key}", value)
                    result[f"test_{key}"] = value
        
        # 使用指标计算功能计算指标
        # 这里不在模块内计算详细指标，因为一次需要所有批次的结果
        # 详细指标计算将在trainer中进行
        
        return {
            **result,
            "y_pred": y_pred, 
            "y_true": y,
            "y_prob": y_prob
        }
    
    def predict_step(self, batch, batch_idx):
        """预测步骤
        
        Args:
            batch: 批次数据
            batch_idx: 批次索引
            
        Returns:
            预测结果元组 (预测标签, 实际标签, 预测概率)
        """
        x, y = batch
        y_hat = self(x)
        
        # 计算预测和概率
        y_pred = self._get_predictions(y_hat)
        y_prob = self._get_probabilities(y_hat)
        
        return y_pred, y, y_prob
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器
        
        Returns:
            优化器配置
        """
        # 获取优化器参数
        optimizer_name = self.optimizer_config.get('name', 'adam').lower()
        lr = self.optimizer_config.get('lr', 0.001)
        weight_decay = self.optimizer_config.get('weight_decay', 0)
        
        # 创建优化器
        if optimizer_name == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = self.optimizer_config.get('momentum', 0.9)
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 检查是否需要学习率调度器
        if self.scheduler_config.get('use_scheduler', False):
            scheduler_type = self.scheduler_config.get('scheduler_type', 'step').lower()
            
            if scheduler_type == 'step':
                step_size = self.scheduler_config.get('step_size', 10)
                gamma = self.scheduler_config.get('gamma', 0.5)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            elif scheduler_type == 'cosine':
                t_max = self.scheduler_config.get('t_max', 50)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
            elif scheduler_type == 'plateau':
                patience = self.scheduler_config.get('patience', 5)
                factor = self.scheduler_config.get('factor', 0.5)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                              factor=factor, patience=patience)
                # ReduceLROnPlateau需要特殊配置
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",
                    }
                }
            else:
                return optimizer
                
            # 其他调度器的返回格式
            return [optimizer], [scheduler]
        
        # 如果不使用调度器，只返回优化器
        return optimizer