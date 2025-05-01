"""
模型工厂模块
"""
import importlib
import os
import glob
from typing import Dict, Any
from .model_factory import model_factory

def build_model(args: Dict[str, Any]) -> Any:
    """根据配置构建模型实例
    
    Args:
        config: 模型配置字典，包含 "name" 和 "args" 字段
        
    Returns:
        模型实例
    """

    return model_factory(args)



# 导出公共API
__all__ = ["build_model",
           'Transformer',
           'CNN',
           'MLP',
           'RNN',
           'ISFM',
           'Other']