"""
任务基类
"""
import torch.nn as nn
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Dict, Any, Union, Optional, List, Tuple




class BaseTask(ABC):

    """任务基类，定义通用接口
    
    所有自定义任务类必须继承该类，并实现必要的方法
    """
    
    def __init__(self, model=None, dataset=None, **kwargs):
        """初始化任务
        
        Args:
            model: 模型实例
            dataset: 数据集实例，将在任务中进行包装
            **kwargs: 其他参数
        """
        self.model = model
        self.dataset = dataset
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # 如果提供了原始数据集，则进行任务特定的包装
        if dataset is not None:
            self._wrap_dataset(dataset, **kwargs)
    
    def _wrap_dataset(self, dataset, **kwargs):
        """包装原始数据集，使其适应特定任务
        
        子类应重写此方法以根据任务类型进行特定包装
        
        Args:
            dataset: 原始数据集
            **kwargs: 可能包含分割比例等参数
        """
        # 默认实现：直接使用原始数据集的分割
        # 子类应根据具体任务需求重写此方法
        self.train_dataset = dataset.train_dataset if hasattr(dataset, 'train_dataset') else None
        self.val_dataset = dataset.val_dataset if hasattr(dataset, 'val_dataset') else None
        self.test_dataset = dataset.test_dataset if hasattr(dataset, 'test_dataset') else None
    
    def get_train_loader(self, batch_size=32, shuffle=True, num_workers=4, **kwargs) -> Optional[DataLoader]:
        """获取训练数据加载器
        
        Args:
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 数据加载的工作进程数
            **kwargs: 其他数据加载器参数
            
        Returns:
            训练数据加载器
        """
        if self.train_dataset is None:
            return None
        
        return DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            **kwargs
        )
    
    def get_val_loader(self, batch_size=32, shuffle=False, num_workers=4, **kwargs) -> Optional[DataLoader]:
        """获取验证数据加载器
        
        Args:
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 数据加载的工作进程数
            **kwargs: 其他数据加载器参数
            
        Returns:
            验证数据加载器
        """
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            **kwargs
        )
    
    def get_test_loader(self, batch_size=32, shuffle=False, num_workers=4, **kwargs) -> Optional[DataLoader]:
        """获取测试数据加载器
        
        Args:
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 数据加载的工作进程数
            **kwargs: 其他数据加载器参数
            
        Returns:
            测试数据加载器
        """
        if self.test_dataset is None:
            return None
        
        return DataLoader(
            self.test_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            **kwargs
        )
    
    def get_loss_function(self) -> Optional[nn.Module]:
        """获取损失函数
        
        Returns:
            损失函数实例
        """
        if hasattr(self, 'loss_fn'):
            return self.loss_fn
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取评估指标函数
        
        Returns:
            包含评估指标名称和函数的字典
        """
        # 默认实现，子类可以重写添加更多指标
        metrics = {}
        if hasattr(self, 'calculate_accuracy'):
            metrics['accuracy'] = self.calculate_accuracy
        return metrics
    
    def calculate_accuracy(self, y_pred, y_true):
        """计算准确率
        
        Args:
            y_pred: 预测输出
            y_true: 真实标签
            
        Returns:
            准确率
        """
        raise NotImplementedError("子类必须实现 calculate_accuracy 方法")
    
    def calculate_metrics(self, y_pred, y_true) -> Dict[str, Any]:
        """计算评估指标
        
        Args:
            y_pred: 预测输出
            y_true: 真实标签
            
        Returns:
            包含多种评估指标的字典
        """
        metrics = {}
        return metrics
    
    @abstractmethod
    def train(self, 
             train_loader: DataLoader, 
             val_loader: Optional[DataLoader] = None,
             **kwargs) -> Dict[str, Any]:
        """训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            **kwargs: 其他训练参数
            
        Returns:
            训练结果字典
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_loader: DataLoader, **kwargs) -> Dict[str, Any]:
        """评估模型
        
        Args:
            test_loader: 测试数据加载器
            **kwargs: 其他评估参数
            
        Returns:
            评估结果字典
        """
        pass

def task_reader(config: Dict[str, Any]) -> 'BaseTask':
    pass