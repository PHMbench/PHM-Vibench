"""Factory utilities for creating task modules."""

from __future__ import annotations

import importlib
from argparse import Namespace
from typing import Any, Optional

import pytorch_lightning as pl
import torch.nn as nn
from ..utils.registry import Registry

TASK_REGISTRY = Registry()

def register_task(task_type: str, name: str):
    """Decorator to register a task implementation."""
    return TASK_REGISTRY.register(f"{task_type}.{name}")

def resolve_task_module(args_task: Namespace) -> str:
    """Return the Python import path for the task module."""
    task_name = args_task.name
    task_type = args_task.type
    if task_type == "Default_task" or task_name == "Default_task":
        return f"src.task_factory.{task_name}"
    # if task_name == "multitask":
    #     composed = "_".join(args_task.task_list)
    #     return f"src.task_factory.task.{task_type}.{composed}"
    # Support In_distribution tasks
    if task_type == "In_distribution":
        return f"src.task_factory.task.In_distribution.{task_name}"
    return f"src.task_factory.task.{task_type}.{task_name}"


def task_factory(
    args_task: Namespace,
    network: nn.Module,
    args_data: Namespace,
    args_model: Namespace,
    args_trainer: Namespace,
    args_environment: Namespace,
    metadata: Any,
) -> Optional[pl.LightningModule]:
    """Instantiate a task module using configuration namespaces."""
    key = f"{args_task.type}.{args_task.name}"
    try:
        task_cls = TASK_REGISTRY.get(key)
    except KeyError:
        module_path = resolve_task_module(args_task)
        try:
            task_module = importlib.import_module(module_path)
            # 智能检测类名，支持多种命名约定
            task_cls = None

            # 优先级1：查找与文件名相同的类名（如 hse_contrastive.py 中的 HseContrastiveTask）
            class_name_from_file = module_path.split('.')[-1].replace('_', ' ').title().replace(' ', '')
            if hasattr(task_module, class_name_from_file):
                task_cls = getattr(task_module, class_name_from_file)

            # 优先级2：查找标准化的类名（Task后缀）
            task_name = args_task.name
            standard_name = task_name.replace('_', ' ').title().replace(' ', '') + 'Task'
            if hasattr(task_module, standard_name):
                task_cls = getattr(task_module, standard_name)

            # 优先级3：向后兼容 - 查找 'task' 类名
            if hasattr(task_module, 'task'):
                task_cls = getattr(task_module, 'task')

            if task_cls is None:
                raise AttributeError(f"No task class found in {module_path} (tried: {class_name_from_file}, {standard_name}, task)")

        except Exception as exc:  # pragma: no cover - runtime safeguard
            print(f"Failed to import task from {module_path}: {exc}")
            return None

    try:
        return task_cls(
            network=network,
            args_data=args_data,
            args_model=args_model,
            args_task=args_task,
            args_trainer=args_trainer,
            args_environment=args_environment,
            metadata=metadata,
        )
    except Exception as exc:  # pragma: no cover - runtime safeguard
        print(f"Failed to create task {key}: {exc}")
        return None





