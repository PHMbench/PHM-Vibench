"""
模型基类
"""
import torch
import torch.nn as nn
from abc import ABC

class BaseModel(nn.Module, ABC):
    """模型基类，定义通用接口和功能
    
    所有自定义模型类必须继承该类
    """
    
    def __init__(self):
        """初始化基类"""
        super().__init__()
    
    def summary(self):
        """返回模型概要信息
        
        Returns:
            模型结构描述
        """
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"Model: {self.__class__.__name__}, Parameters: {num_params:,}"
    
    def freeze(self):
        """冻结所有模型参数"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """解冻所有模型参数"""
        for param in self.parameters():
            param.requires_grad = True
    
    def save_weights(self, path):
        """保存模型权重
        
        Args:
            path: 权重保存路径
        """
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path):
        """加载模型权重
        
        Args:
            path: 权重文件路径
        """
        self.load_state_dict(torch.load(path))
    
    def get_trainable_params(self):
        """获取可训练参数数量
        
        Returns:
            可训练参数数量
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)