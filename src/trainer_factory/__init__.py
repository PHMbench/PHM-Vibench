"""Public API for trainer construction and registration."""

from argparse import Namespace

import pytorch_lightning as pl

from .trainer_factory import (
    TRAINER_REGISTRY,
    register_trainer,
    resolve_trainer_module,
    trainer_factory,
)


def build_trainer(
    args_environment: Namespace,
    args_trainer: Namespace,
    args_data: Namespace,
    path: str,
) -> pl.Trainer:
    """Build one trainer; import and construction failures are raised."""

    return trainer_factory(
        args_environment,
        args_trainer,
        args_data,
        path,
    )


__all__ = [
    "build_trainer",
    "resolve_trainer_module",
    "register_trainer",
    "TRAINER_REGISTRY",
]
