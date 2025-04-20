import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List

from src.task_factory import register_task
from src.task_factory.base_task import BaseTask

@register_task('DummyClassificationTask')
class DummyClassificationTask(BaseTask):
    """示例分类任务类，用于框架测试
    
    实现了基本的分类任务，包括训练、验证和测试
    """
    
    def __init__(self, 
                model=None, 
                num_classes=2, 
                class_weights=None,
                learning_rate=0.001,
                **kwargs):
        """初始化任务
        
        Args:
            model: 模型实例
            num_classes: 类别数量
            class_weights: 类别权重
            learning_rate: 学习率
            **kwargs: 其他参数
        """
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # 设置损失函数
        if class_weights is not None:
            if isinstance(class_weights, list):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入数据
            
        Returns:
            模型输出
        """
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """训练步骤
        
        Args:
            batch: 批量数据
            batch_idx: 批次索引
            
        Returns:
            损失值字典
        """
        x, y = batch
        y_hat = self.forward(x)
        
        if y.dim() == 2 and y.size(1) == 1:
            y = y.squeeze(1)  # [B, 1] -> [B]
            
        loss = self.loss_fn(y_hat, y)
        
        # 计算准确率
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        
        # 记录指标
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return {'loss': loss, 'acc': acc}
    
    def validation_step(self, batch, batch_idx):
        """验证步骤
        
        Args:
            batch: 批量数据
            batch_idx: 批次索引
            
        Returns:
            验证结果字典
        """
        x, y = batch
        y_hat = self.forward(x)
        
        if y.dim() == 2 and y.size(1) == 1:
            y = y.squeeze(1)  # [B, 1] -> [B]
            
        loss = self.loss_fn(y_hat, y)
        
        # 计算准确率
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        
        # 记录指标
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def test_step(self, batch, batch_idx):
        """测试步骤
        
        Args:
            batch: 批量数据
            batch_idx: 批次索引
            
        Returns:
            测试结果字典
        """
        x, y = batch
        y_hat = self.forward(x)
        
        if y.dim() == 2 and y.size(1) == 1:
            y = y.squeeze(1)  # [B, 1] -> [B]
            
        loss = self.loss_fn(y_hat, y)
        
        # 计算准确率
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        
        # 记录指标
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        
        return {'test_loss': loss, 'test_acc': acc}
    
    def configure_optimizers(self):
        """配置优化器
        
        Returns:
            优化器
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    @staticmethod
    def add_task_specific_args(parent_parser):
        """添加任务特定的命令行参数
        
        Args:
            parent_parser: 父解析器
            
        Returns:
            更新后的解析器
        """
        parser = parent_parser.add_argument_group("DummyClassificationTask")
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        return parent_parser


if __name__ == '__main__':
    """测试入口点"""
    import argparse
    import numpy as np
    import pytorch_lightning as pl
    from torch.utils.data import TensorDataset, DataLoader
    
    # 创建解析器
    parser = argparse.ArgumentParser(description='DummyClassificationTask 测试')
    parser = DummyClassificationTask.add_task_specific_args(parser)
    args = parser.parse_args()
    
    # 创建一个简单的模型
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # 实例化模型
    model = SimpleModel()
    
    # 创建任务实例
    task = DummyClassificationTask(
        model=model,
        num_classes=args.num_classes,
        learning_rate=args.learning_rate
    )
    
    # 创建假数据
    batch_size = 16
    input_dim = 10
    
    # 生成随机特征和标签
    x_train = torch.randn(100, input_dim)
    y_train = torch.randint(0, args.num_classes, (100,))
    
    x_val = torch.randn(20, input_dim)
    y_val = torch.randint(0, args.num_classes, (20,))
    
    # 创建数据集和数据加载器
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 训练模型
    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(task, train_loader, val_loader)
    
    print("\n任务测试成功!")