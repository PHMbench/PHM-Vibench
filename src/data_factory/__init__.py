"""
数据集工厂模块
"""
import importlib
import os
import glob
from functools import partial
from typing import Dict, Any, Optional
from .data_factory import data_factory
from .dataset_task.Dataset_cluster import IdIncludedDataset

def build_data(args_data,args_task) -> Any:
    """根据配置构建数据集实例
    
    Args:
        config: 数据集配置字典，包含 "name" 和 "args" 字段
        
    Returns:
        数据集实例

        
    """
    return data_factory(args_data, args_task)



# 导出公共API
# 当前模块仅暴露 `build_data` 方法。原先的 `register_dataset` 等接口
# 在早期版本中已被移除，继续暴露这些不存在的名称会导致 `import *`
# 时出现 `ImportError`。因此将 `__all__` 更新为实际可用的函数列表。
__all__ = ["build_data", "IdIncludedDataset"]
