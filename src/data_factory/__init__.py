"""
数据集工厂模块
"""
import importlib
import os
import glob
from functools import partial
from typing import Dict, Any, Optional
from .data_reader import data_reader


def build_data(args) -> Any:
    """根据配置构建数据集实例
    
    Args:
        config: 数据集配置字典，包含 "name" 和 "args" 字段
        
    Returns:
        数据集实例

        
    """
    metadata, data_dict = data_reader(args)
    return metadata, data_dict


# 导出公共API
__all__ = ["register_dataset", "build_dataset", "_dataset_registry"]