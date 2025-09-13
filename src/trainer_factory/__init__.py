"""Public API for the trainer factory package."""

from argparse import Namespace
from typing import Any
from . import Default_trainer
from .trainer_factory import (
    # TRAINER_REGISTRY,
    # register_trainer,
    resolve_trainer_module,
    trainer_factory,

)


def build_trainer(
    args_environment: Namespace,
    args_trainer: Namespace,
    args_data: Namespace,
    path: str,
) -> Any:
    """Instantiate a trainer via :mod:`trainer_factory`.

    Parameters
    ----------
    args_environment : Namespace
        Environment configuration.
    args_trainer : Namespace
        Trainer configuration namespace.
    args_data : Namespace
        Dataset configuration.
    path : str
        Output directory for checkpoints/logs.

    Returns
    -------
    Any
        Instantiated trainer object or ``None`` on failure.
    """
    return trainer_factory(
        args_environment,
        args_trainer,
        args_data,
        path,
    )



# public exports
__all__ = [
    "build_trainer",
    "resolve_trainer_module",
    "register_trainer",
    "TRAINER_REGISTRY",
    "Default_trainer"
]
