"""
训练器基类
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseTrainer(ABC):
    """训练器基类，定义通用接口
    
    所有自定义训练器类必须继承该类，并实现必要的方法
    """
    
    def __init__(self, **kwargs):
        """初始化训练器
        
        Args:
            **kwargs: 训练相关参数
        """
        self.config = kwargs
    
    @abstractmethod
    def __call__(self, configs, save_path, iteration=0) -> Dict[str, Any]:
        """训练及评估流程
        
        Args:
            configs: 完整配置字典
            save_path: 结果保存路径
            iteration: 当前迭代次数
            
        Returns:
            评估结果字典
        """
        pass