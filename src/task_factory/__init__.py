"""
任务工厂模块
"""
import importlib
import os
import glob
from typing import Dict, Any
from .task_reader import task_reader



def build_task(config: Dict[str, Any]) -> Any:
    """根据配置构建任务实例
    
    Args:
        config: 任务配置字典，包含 "name" 和 "args" 字段
        
    Returns:
        任务实例
    """
    return task_reader(config)



# 导出公共API
__all__ = ["build_task"]