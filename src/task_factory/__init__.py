"""Public API for task construction and registration."""

from argparse import Namespace
from typing import Any

import pytorch_lightning as pl
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
) -> pl.LightningModule:
    """Build one task; import and construction failures are raised."""

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
