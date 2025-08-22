"""
训练器基类
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union

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
    def __call__(self, 
                dataset=None,
                model=None, 
                task=None,
                train_loader=None,
                val_loader=None,
                test_loader=None,
                configs=None, 
                args_t=None,
                args_m=None,
                args_d=None,
                args_task=None,
                save_path=None, 
                iteration=0) -> Dict[str, Any]:
        """训练及评估流程
        
        Args:
            dataset: 数据集实例
            model: 模型实例
            task: 任务实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            configs: 完整配置字典
            args_t: 训练器参数命名空间
            args_m: 模型参数命名空间
            args_d: 数据集参数命名空间
            args_task: 任务参数命名空间
            save_path: 结果保存路径
            iteration: 当前迭代次数
            
        Returns:
            评估结果字典
        """
        pass