import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, List, Tuple, Callable
import wandb

class BasicLightningModel(pl.LightningModule):
    """基础 PyTorch Lightning 模型封装，用于训练和评估
    
    该类封装了普通的 PyTorch 模型，提供训练、验证和测试逻辑
    """
    
    def __init__(
        self, 
        model: nn.Module,
        args_t: Any,
        args_m: Any,
        args_d: Any,
        loss_fn: Optional[Callable] = None
    ):
        """初始化基础 Lightning 模型
        
        Args:
            model: PyTorch 模型实例
            args_t: 训练相关参数
            args_m: 模型相关参数
            args_d: 数据集相关参数
            loss_fn: 可选的损失函数
        """
        super().__init__()
        self.network = model
        self.args_t = args_t
        self.args_m = args_m
        self.args_d = args_d
        
        # 如果提供了损失函数则使用，否则使用默认的 MSE
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
        
        # 自动设置日志参数
        self.save_hyperparameters(ignore=["network"])
        
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            模型输出
        """
        return self.network(x)
    
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
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
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
        return {"val_loss": val_loss}
    
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
        self.log("test_loss", test_loss, prog_bar=True)
        return {"test_loss": test_loss}
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器
        
        Returns:
            优化器配置
        """
        # 从参数中获取优化器设置
        lr = getattr(self.args_t, "lr", 0.001)
        weight_decay = getattr(self.args_t, "weight_decay", 0)
        
        # 创建优化器
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 检查是否需要学习率调度器
        if hasattr(self.args_t, "scheduler") and self.args_t.scheduler:
            scheduler_type = getattr(self.args_t, "scheduler_type", "step")
            
            if scheduler_type == "step":
                step_size = getattr(self.args_t, "step_size", 10)
                gamma = getattr(self.args_t, "gamma", 0.5)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            elif scheduler_type == "reduce_on_plateau":
                patience = getattr(self.args_t, "patience", 5)
                factor = getattr(self.args_t, "factor", 0.5)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                              factor=factor, patience=patience)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",
                    },
                }
            else:
                return optimizer
                
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }
            
        return optimizer