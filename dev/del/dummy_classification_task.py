import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from torch.utils.data import Dataset, Subset

from src.task_factory import register_task
from Vbench.src.task_factory.task_factory import BaseTask

@register_task('DummyclassificationTask')
class DummyclassificationTask(BaseTask):
    """示例分类任务类，用于框架测试
    
    实现了基本的分类任务，包括训练、验证和测试
    """
    
    def __init__(self, 
                model=None, 
                dataset=None,
                num_classes=2, 
                class_weights=None,
                learning_rate=0.001,
                batch_size=32,
                **kwargs):
        """初始化任务
        
        Args:
            model: 模型实例
            dataset: 数据集实例
            num_classes: 类别数量
            class_weights: 类别权重
            learning_rate: 学习率
            batch_size: 批处理大小
            **kwargs: 其他参数
        """
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # 设置损失函数
        if class_weights is not None:
            if isinstance(class_weights, list):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        
        # 调用父类初始化，包装数据集
        super().__init__(model=model, dataset=dataset, **kwargs)
    
    def _wrap_dataset(self, dataset, **kwargs):
        """包装数据集以适应分类任务
        
        Args:
            dataset: 原始数据集
            **kwargs: 其他参数
        """
        # 首先调用父类方法获取基本划分
        super()._wrap_dataset(dataset, **kwargs)
        
        # 如果原始数据集没有分割，则进行任务特定的分割
        if self.train_dataset is None and hasattr(dataset, 'data'):
            # 根据任务需求自定义分割逻辑
            print("[INFO] 按照分类任务要求重新包装数据集")
            # 这里可以根据分类任务的特定需求进行数据转换
            # 例如，确保标签是整数类别而非浮点数等
            
            # 获取所有数据
            all_data = dataset.data if hasattr(dataset, 'data') else dataset
            data_size = len(all_data)
            
            # 分割数据集
            train_size = int(0.7 * data_size)
            val_size = int(0.15 * data_size)
            test_size = data_size - train_size - val_size
            
            indices = torch.randperm(data_size).tolist()
            self.train_dataset = Subset(all_data, indices[:train_size])
            self.val_dataset = Subset(all_data, indices[train_size:train_size+val_size])
            self.test_dataset = Subset(all_data, indices[train_size+val_size:])
    
    def get_loss_fn(self) -> nn.Module:
        """获取损失函数
        
        Returns:
            损失函数实例
        """
        return self.loss_fn
    
    def calculate_accuracy(self, y_pred, y_true):
        """计算分类准确率
        
        Args:
            y_pred: 预测输出 [batch_size, num_classes]
            y_true: 真实标签 [batch_size]
            
        Returns:
            准确率
        """
        # 确保y_true的形状正确
        if y_true.dim() > 1 and y_true.size(1) == 1:
            y_true = y_true.squeeze(1)
        
        # 获取预测类别
        preds = torch.argmax(y_pred, dim=1)
        # 计算准确率
        correct = (preds == y_true).sum().item()
        total = y_true.size(0)
        return correct / total
    
    def calculate_metrics(self, y_pred, y_true) -> Dict[str, Any]:
        """计算分类任务相关的指标
        
        Args:
            y_pred: 预测输出
            y_true: 真实标签
            
        Returns:
            指标字典
        """
        metrics = {}
        
        # 计算准确率
        accuracy = self.calculate_accuracy(y_pred, y_true)
        metrics['accuracy'] = accuracy
        
        # 这里可以添加更多分类指标，如精确度、召回率、F1分数等
        
        return metrics
    
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
    
    def train(self, train_loader, val_loader=None, **kwargs):
        """训练模型
        
        实现基类的抽象方法
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            **kwargs: 其他训练参数
            
        Returns:
            训练结果
        """
        # 在实际应用中，可能会将训练逻辑委托给PyTorch Lightning等框架
        # 这里只是一个示例实现
        epochs = kwargs.get('epochs', 10)
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(device)
        self.model.train()
        
        optimizer = self.configure_optimizers()
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch in train_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                y_hat = self.model(x)
                
                if y.dim() == 2 and y.size(1) == 1:
                    y = y.squeeze(1)
                    
                loss = self.loss_fn(y_hat, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # 计算准确率
                preds = torch.argmax(y_hat, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            
            # 计算平均损失和准确率
            avg_loss = total_loss / len(train_loader)
            avg_acc = correct / total
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}')
            
            # 如果提供了验证集，则进行验证
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, is_val=True, device=device)
                print(f'Val Loss: {val_metrics["val_loss"]:.4f}, Val Acc: {val_metrics["val_acc"]:.4f}')
        
        return {
            'train_loss': avg_loss,
            'train_acc': avg_acc
        }
    
    def evaluate(self, test_loader, **kwargs):
        """评估模型
        
        实现基类的抽象方法
        
        Args:
            test_loader: 测试数据加载器
            **kwargs: 其他参数
            
        Returns:
            评估结果字典
        """
        is_val = kwargs.get('is_val', False)
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(device)
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                
                y_hat = self.model(x)
                
                if y.dim() == 2 and y.size(1) == 1:
                    y = y.squeeze(1)
                    
                loss = self.loss_fn(y_hat, y)
                
                total_loss += loss.item()
                
                # 计算准确率
                preds = torch.argmax(y_hat, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(test_loader)
        avg_acc = correct / total
        
        prefix = 'val' if is_val else 'test'
        return {
            f'{prefix}_loss': avg_loss,
            f'{prefix}_acc': avg_acc
        }
        
    @staticmethod
    def add_task_specific_args(parent_parser):
        """添加任务特定的命令行参数
        
        Args:
            parent_parser: 父解析器
            
        Returns:
            更新后的解析器
        """
        parser = parent_parser.add_argument_group("DummyclassificationTask")
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--batch_size", type=int, default=32)
        return parent_parser


if __name__ == '__main__':
    """测试入口点"""
    import argparse
    import numpy as np
    import pytorch_lightning as pl
    from torch.utils.data import TensorDataset, DataLoader
    
    # 创建解析器
    parser = argparse.ArgumentParser(description='DummyclassificationTask 测试')
    parser = DummyclassificationTask.add_task_specific_args(parser)
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
    task = DummyclassificationTask(
        model=model,
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )
    
    # 创建假数据
    batch_size = args.batch_size
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