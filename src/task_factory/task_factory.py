"""Factory utilities for creating task modules."""

from __future__ import annotations

import importlib
from argparse import Namespace
from typing import Any

import pytorch_lightning as pl
import torch.nn as nn

from ..utils.registry import Registry

TASK_REGISTRY = Registry()


def register_task(task_type: str, name: str):
    """Decorator to register a task implementation."""

    return TASK_REGISTRY.register(f"{task_type}.{name}")


def resolve_task_module(args_task: Namespace) -> str:
    """Return the historical Python import path for one task configuration."""

    task_name = args_task.name
    task_type = args_task.type
    if task_type == "Default_task" or task_name == "Default_task":
        return f"src.task_factory.{task_name}"
    if task_name == "multitask":
        composed = "_".join(args_task.task_list)
        return f"src.task_factory.task.{task_type}.{composed}"
    return f"src.task_factory.task.{task_type}.{task_name}"


def _resolve_task_class(args_task: Namespace):
    """Resolve one task class from the registry or historical ``task`` symbol."""

    key = f"{args_task.type}.{args_task.name}"
    try:
        return TASK_REGISTRY.get(key)
    except KeyError:
        pass

    module_path = resolve_task_module(args_task)
    try:
        task_module = importlib.import_module(module_path)
    except Exception as exc:
        raise ImportError(
            f"Cannot import task {key!r} from {module_path!r}: {exc}. "
            "Check task.type, task.name, the module path, and optional "
            "dependencies."
        ) from exc

    # Importing a module may execute its @register_task decorator.
    try:
        return TASK_REGISTRY.get(key)
    except KeyError:
        pass

    task_class = getattr(task_module, "task", None)
    if task_class is None:
        raise AttributeError(
            f"Task module {module_path!r} does not register {key!r} and does "
            "not expose the historical class name 'task'. Register the class "
            "with @register_task or export 'task'."
        )
    return task_class


def task_factory(
    args_task: Namespace,
    network: nn.Module,
    args_data: Namespace,
    args_model: Namespace,
    args_trainer: Namespace,
    args_environment: Namespace,
    metadata: Any,
) -> pl.LightningModule:
    """Instantiate one task while preserving constructor failures."""

    task_class = _resolve_task_class(args_task)
    return task_class(
        network=network,
        args_data=args_data,
        args_model=args_model,
        args_task=args_task,
        args_trainer=args_trainer,
        args_environment=args_environment,
        metadata=metadata,
    )


__all__ = [
    "TASK_REGISTRY",
    "register_task",
    "resolve_task_module",
    "task_factory",
]
