"""
任务工厂模块
"""
import importlib
import os
import glob
from typing import Dict, Any

# 用于存储已注册的任务类
_task_registry = {}

def register_task(name):
    """注册任务类的装饰器
    
    Args:
        name: 任务类的名称
        
    Returns:
        装饰器函数
    """
    def decorator(cls):
        _task_registry[name] = cls
        return cls
    return decorator

def build_task(config: Dict[str, Any]) -> Any:
    """根据配置构建任务实例
    
    Args:
        config: 任务配置字典，包含 "name" 和 "args" 字段
        
    Returns:
        任务实例
    """
    name = config["name"]
    args = config.get("args", {})
    
    if name in _task_registry:
        return _task_registry[name](**args)
    else:
        raise ValueError(f"未知的任务类型: {name}")

# 动态加载当前目录下的所有模块
current_dir = os.path.dirname(__file__)
for file in glob.glob(os.path.join(current_dir, "*.py")):
    if not os.path.basename(file).startswith("_") and not file.endswith("__init__.py"):
        module_name = os.path.basename(file)[:-3]  # 去除 .py 扩展名
        importlib.import_module(f"src.task_factory.{module_name}")

# 导出公共API
__all__ = ["register_task", "build_task"]