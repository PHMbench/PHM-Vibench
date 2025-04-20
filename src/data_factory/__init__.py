"""
数据集工厂模块
"""
import importlib
import os
import glob
from functools import partial
from typing import Dict, Any, Optional

# 用于存储已注册的数据集类
_dataset_registry = {}

def register_dataset(name):
    """注册数据集类的装饰器
    
    Args:
        name: 数据集类的名称
        
    Returns:
        装饰器函数
    """
    def decorator(cls):
        _dataset_registry[name] = cls
        return cls
    return decorator

def build_dataset(config: Dict[str, Any]) -> Any:
    """根据配置构建数据集实例
    
    Args:
        config: 数据集配置字典，包含 "name" 和 "args" 字段
        
    Returns:
        数据集实例
    """
    name = config["name"]
    args = config.get("args", {})
    
    if name in _dataset_registry:
        return _dataset_registry[name](**args)
    else:
        raise ValueError(f"未知的数据集类型: {name}")

# 动态加载当前目录下的所有模块
current_dir = os.path.dirname(__file__)
for file in glob.glob(os.path.join(current_dir, "*.py")):
    if not os.path.basename(file).startswith("_") and not file.endswith("__init__.py"):
        module_name = os.path.basename(file)[:-3]  # 去除 .py 扩展名
        importlib.import_module(f"src.data_factory.{module_name}")

# 导出公共API
__all__ = ["register_dataset", "build_dataset", "_dataset_registry"]