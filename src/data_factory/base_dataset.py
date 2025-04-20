"""
数据集基类
"""
from abc import ABC, abstractmethod
from typing import Tuple, Any

class BaseDataset(ABC):
    """数据集基类，定义通用接口
    
    所有自定义数据集类必须继承该类，并实现必要的方法
    """
    
    @abstractmethod
    def get_train_loader(self):
        """获取训练数据加载器"""
        pass
    
    @abstractmethod
    def get_val_loader(self):
        """获取验证数据加载器"""
        pass
    
    @abstractmethod
    def get_test_loader(self):
        """获取测试数据加载器"""
        pass
    
    def get_data_loaders(self):
        """获取所有数据加载器
        
        Returns:
            训练、验证和测试数据加载器的元组
        """
        train_loader = self.get_train_loader()
        val_loader = self.get_val_loader()
        test_loader = self.get_test_loader()
        
        return train_loader, val_loader, test_loader