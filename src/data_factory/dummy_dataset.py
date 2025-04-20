import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Optional, Union, Tuple

from src.data_factory import register_dataset
from src.data_factory.base_dataset import BaseDataset

@register_dataset('DummyDataset')
class DummyDataset(BaseDataset):
    """示例数据集类，用于框架测试
    
    生成随机数据，支持分类和回归任务
    """
    
    def __init__(self,
                task_type='classification',
                num_samples=1000,
                feature_dim=10,
                num_classes=2,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                batch_size=32,
                seed=42,
                **kwargs):
        """初始化数据集
        
        Args:
            task_type: 任务类型，'classification' 或 'regression'
            num_samples: 样本总数
            feature_dim: 特征维度
            num_classes: 类别数量（仅在分类任务中使用）
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            batch_size: 批次大小
            seed: 随机种子
            **kwargs: 其他参数
        """
        super().__init__()
        
        self.task_type = task_type
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        
        # 检查比例总和是否为 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例总和必须为 1"
        
        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 生成数据
        self.generate_data()
    
    def generate_data(self):
        """生成随机数据"""
        # 生成特征
        X = np.random.randn(self.num_samples, self.feature_dim)
        
        # 生成标签
        if self.task_type == 'classification':
            y = np.random.randint(0, self.num_classes, size=(self.num_samples,))
        else:  # 回归任务
            y = np.random.randn(self.num_samples, 1)
        
        # 划分数据集
        train_size = int(self.num_samples * self.train_ratio)
        val_size = int(self.num_samples * self.val_ratio)
        
        self.X_train = X[:train_size]
        self.y_train = y[:train_size]
        
        self.X_val = X[train_size:train_size+val_size]
        self.y_val = y[train_size:train_size+val_size]
        
        self.X_test = X[train_size+val_size:]
        self.y_test = y[train_size+val_size:]
        
        # 创建数据集对象
        self.train_dataset = _CustomDataset(self.X_train, self.y_train)
        self.val_dataset = _CustomDataset(self.X_val, self.y_val)
        self.test_dataset = _CustomDataset(self.X_test, self.y_test)
    
    def get_train_loader(self):
        """获取训练数据加载器"""
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0
        )
    
    def get_val_loader(self):
        """获取验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
    
    def get_test_loader(self):
        """获取测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
    
    def get_data_loaders(self, batch_size=None):
        """获取所有数据加载器
        
        Args:
            batch_size: 批量大小，如果为None则使用默认值
            
        Returns:
            训练、验证和测试数据加载器
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        train_loader = self.get_train_loader()
        val_loader = self.get_val_loader()
        test_loader = self.get_test_loader()
        
        return train_loader, val_loader, test_loader
    
    @staticmethod
    def add_dataset_specific_args(parent_parser):
        """添加数据集特定的命令行参数
        
        Args:
            parent_parser: 父解析器
            
        Returns:
            更新后的解析器
        """
        parser = parent_parser.add_argument_group("DummyDataset")
        parser.add_argument("--task_type", type=str, default='classification', choices=['classification', 'regression'])
        parser.add_argument("--num_samples", type=int, default=1000)
        parser.add_argument("--feature_dim", type=int, default=10)
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument("--train_ratio", type=float, default=0.7)
        parser.add_argument("--val_ratio", type=float, default=0.15)
        parser.add_argument("--test_ratio", type=float, default=0.15)
        parser.add_argument("--seed", type=int, default=42)
        return parent_parser


class _CustomDataset(Dataset):
    """自定义数据集类
    
    用于封装特征和标签，提供给 DataLoader 使用
    """
    
    def __init__(self, features, labels):
        """初始化数据集
        
        Args:
            features: 特征
            labels: 标签
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long if labels.ndim == 1 else torch.float32)
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.features)
    
    def __getitem__(self, idx):
        """获取指定索引的样本
        
        Args:
            idx: 样本索引
            
        Returns:
            特征和标签
        """
        return self.features[idx], self.labels[idx]


if __name__ == '__main__':
    """测试入口点"""
    import argparse
    
    # 创建解析器
    parser = argparse.ArgumentParser(description='DummyDataset 测试')
    parser = DummyDataset.add_dataset_specific_args(parser)
    args = parser.parse_args()
    
    # 创建数据集实例
    dataset = DummyDataset(
        task_type=args.task_type,
        num_samples=args.num_samples,
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = dataset.get_data_loaders(batch_size=32)
    
    # 打印数据集信息
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 检查一个批次
    x_batch, y_batch = next(iter(train_loader))
    print(f"批次特征形状: {x_batch.shape}")
    print(f"批次标签形状: {y_batch.shape}")
    
    print("\n数据集测试成功!")