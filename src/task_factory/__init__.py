"""
任务工厂模块
"""
import importlib
import os
import glob
from typing import Dict, Any
from .task_factory import task_factory
import torch.nn as nn
from argparse import Namespace


def build_task(    args_task: Namespace,      # Task config (Namespace)
    network: nn.Module,
    args_data: Namespace,      # Data args (Namespace)
    args_model: Namespace,     # Model args (Namespace)
    args_trainer: Namespace,   # Training args (Namespace) - Renamed from args_t
    args_environment: Namespace, # Environment args (Namespace)
    metadata: Any         ) -> Any:
    """根据配置构建任务实例
    
    Args:
        config: 任务配置字典，包含 "name" 和 "args" 字段
        
    Returns:
        任务实例
    """
    return task_factory(
        args_task=args_task,      # Task config (Namespace)
        network=network,
        args_data=args_data,      # Data args (Namespace)
        args_model=args_model,     # Model args (Namespace)
        args_trainer=args_trainer,   # Training args (Namespace) - Renamed from args_t
        args_environment=args_environment, # Environment args (Namespace)
        metadata=metadata
    )



# 导出公共API
__all__ = ["build_task"]