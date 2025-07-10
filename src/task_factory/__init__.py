"""任务工厂模块"""

from .task_factory import task_factory
import torch.nn as nn
from argparse import Namespace
from typing import Any


def build_task(
    args_task: Namespace,
    network: nn.Module,
    args_data: Namespace,
    args_model: Namespace,
    args_trainer: Namespace,
    args_environment: Namespace,
    metadata: Any,
) -> Any:
    """根据配置构建任务实例"""
    return task_factory(
        args_task=args_task,
        network=network,
        args_data=args_data,
        args_model=args_model,
        args_trainer=args_trainer,
        args_environment=args_environment,
        metadata=metadata,
    )

__all__ = ["build_task"]
