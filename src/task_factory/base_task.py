"""
任务基类
"""
import torch.nn as nn
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from typing import Dict, Any, Union, Optional

class BaseTask(ABC):
    """任务基类，定义通用接口
    
    所有自定义任务类必须继承该类，并实现必要的方法
    """
    
    def __init__(self, model=None, **kwargs):
        """初始化任务
        
        Args:
            model: 模型实例
            **kwargs: 其他参数
        """
        self.model = model
    
    def get_loss_fn(self) -> Optional[nn.Module]:
        """获取损失函数
        
        Returns:
            损失函数实例
        """
        return None
    
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