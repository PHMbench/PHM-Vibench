"""
训练器工厂模块
"""
import importlib
import os
import glob
from typing import Dict, Any
from .trainer_factory import trainer_factory


def build_trainer(
        args_environment,
        args_trainer,  # 训练参数 (Namespace)
    args_data,     # 数据参数 (Namespace)
    path) -> Any:
    """根据配置构建训练器实例
    
    Args:
        config: 训练器配置字典，包含 "name" 和 "args" 字段
        
    Returns:
        训练器实例
    """
    return trainer_factory(
        args_environment,  # 环境参数 (Namespace)
        args_trainer,  # 训练参数 (Namespace)
    args_data,     # 数据参数 (Namespace)
    path)



# 导出公共API
__all__ = ["register_trainer", "build_trainer"]