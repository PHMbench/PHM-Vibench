"""Factory utilities for creating trainer objects."""

from __future__ import annotations

import importlib
from argparse import Namespace
from typing import Any, Optional

import pytorch_lightning as pl
from ..utils.registry import Registry

# TRAINER_REGISTRY = Registry()

# def register_trainer(name: str):
#     """Decorator to register a trainer implementation."""
#     return TRAINER_REGISTRY.register(name)

def resolve_trainer_module(args_trainer: Namespace) -> str:
    """Return the Python import path for the trainer module."""
    trainer_name = getattr(args_trainer, "trainer_name", "Default_trainer")
    return f"src.trainer_factory.{trainer_name}"


def trainer_factory(
    args_environment: Namespace,
    args_trainer: Namespace,
    args_data: Namespace,
    path: str,
) -> Optional[pl.Trainer]:
    """Instantiate a trainer using configuration namespaces."""
    name = getattr(args_trainer, "name", "Default_trainer")
    # try:
    #     trainer_fn = TRAINER_REGISTRY.get(name)
    # except KeyError:
    module_path = resolve_trainer_module(args_trainer)
    try:
        trainer_module = importlib.import_module(module_path)
        trainer_fn = trainer_module.trainer
    except Exception as exc:  # pragma: no cover - runtime safeguard
        print(f"Failed to import trainer {module_path}: {exc}")
        return None

    try:
        return trainer_fn(
            args_e=args_environment,
            args_t=args_trainer,
            args_d=args_data,
            path=path,
        )
    except Exception as exc:  # pragma: no cover - runtime safeguard
        print(f"Failed to create trainer {name}: {exc}")
        return None
