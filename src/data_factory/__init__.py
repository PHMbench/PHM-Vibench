"""
数据集工厂模块
"""
import importlib
import os
import glob
from functools import partial
from typing import Dict, Any, Optional
from .data_factory import data_factory


def build_data(args_data,args_task) -> Any:
    """根据配置构建数据集实例
    
    Args:
        config: 数据集配置字典，包含 "name" 和 "args" 字段
        
    Returns:
        数据集实例

        
    """
    return data_factory(args_data, args_task)



# 导出公共API
__all__ = ["register_dataset", "build_dataset", "_dataset_registry"]