"""Public API for the task factory package."""

from __future__ import annotations

import importlib
from argparse import Namespace
from typing import Any


def _factory_module():
    return importlib.import_module(f"{__name__}.task_factory")


def register_task(task_type: str, name: str):
    """Register a task implementation without importing the factory at package import time."""
    return _factory_module().register_task(task_type, name)


def resolve_task_module(args_task: Namespace) -> str:
    return _factory_module().resolve_task_module(args_task)


def build_task(
    args_task: Namespace,
    network: Any,
    args_data: Namespace,
    args_model: Namespace,
    args_trainer: Namespace,
    args_environment: Namespace,
    metadata: Any,
) -> Any:
    """Instantiate a task module using :mod:`task_factory`.

    Parameters
    ----------
    args_task : Namespace
        Task configuration namespace.
    network : nn.Module
        Model backbone to be wrapped by the task.
    args_data : Namespace
        Dataset related configuration.
    args_model : Namespace
        Model configuration namespace.
    args_trainer : Namespace
        Trainer configuration namespace.
    args_environment : Namespace
        Runtime environment configuration.
    metadata : Any
        Dataset metadata passed to the task.

    Returns
    -------
    Any
        Instantiated LightningModule or ``None`` on failure.
    """
    return _factory_module().task_factory(
        args_task=args_task,
        network=network,
        args_data=args_data,
        args_model=args_model,
        args_trainer=args_trainer,
        args_environment=args_environment,
        metadata=metadata,
    )


def __getattr__(name: str):
    if name == "TASK_REGISTRY":
        return _factory_module().TASK_REGISTRY
    raise AttributeError(name)


__all__ = [
    "build_task",
    "resolve_task_module",
    "register_task",
    "TASK_REGISTRY",
]
