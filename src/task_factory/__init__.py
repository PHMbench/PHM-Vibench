"""Public API for the task factory package."""

from argparse import Namespace
from typing import Any

import torch.nn as nn

from .task_factory import (
    TASK_REGISTRY,
    register_task,
    resolve_task_module,
    task_factory,
)


def build_task(
    args_task: Namespace,
    network: nn.Module,
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
    return task_factory(
        args_task=args_task,
        network=network,
        args_data=args_data,
        args_model=args_model,
        args_trainer=args_trainer,
        args_environment=args_environment,
        metadata=metadata,
    )

__all__ = [
    "build_task",
    "resolve_task_module",
    "register_task",
    "TASK_REGISTRY",
]
