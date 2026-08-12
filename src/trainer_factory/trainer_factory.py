"""Factory utilities for creating trainer objects."""

from __future__ import annotations

import importlib
from argparse import Namespace

import pytorch_lightning as pl

from ..utils.registry import Registry
from .p05_runtime import p05_evidence_mode_enabled

TRAINER_REGISTRY = Registry()


def register_trainer(name: str):
    """Decorator to register a trainer implementation."""

    return TRAINER_REGISTRY.register(name)


def resolve_trainer_module(args_trainer: Namespace) -> str:
    """Return the historical Python import path for one trainer configuration."""

    trainer_name = getattr(
        args_trainer,
        "name",
        getattr(args_trainer, "trainer_name", "Default_trainer"),
    )
    return f"src.trainer_factory.{trainer_name}"


def _resolve_trainer_function(args_trainer: Namespace):
    """Resolve one trainer builder from the registry or module ``trainer`` symbol."""

    name = getattr(
        args_trainer,
        "name",
        getattr(args_trainer, "trainer_name", "Default_trainer"),
    )
    try:
        return name, TRAINER_REGISTRY.get(name)
    except KeyError:
        pass

    module_path = resolve_trainer_module(args_trainer)
    try:
        trainer_module = importlib.import_module(module_path)
    except Exception as exc:
        raise ImportError(
            f"Cannot import trainer {name!r} from {module_path!r}: {exc}. "
            "Check trainer.name, the module path, and optional dependencies."
        ) from exc

    # Importing a module may execute its @register_trainer decorator.
    try:
        return name, TRAINER_REGISTRY.get(name)
    except KeyError:
        pass

    trainer_function = getattr(trainer_module, "trainer", None)
    if trainer_function is None:
        raise AttributeError(
            f"Trainer module {module_path!r} does not register {name!r} and "
            "does not expose the historical function name 'trainer'. Register "
            "the builder with @register_trainer or export 'trainer'."
        )
    return name, trainer_function


def trainer_factory(
    args_environment: Namespace,
    args_trainer: Namespace,
    args_data: Namespace,
    path: str,
) -> pl.Trainer:
    """Instantiate one trainer or raise at the trainer factory boundary."""
    p05_evidence_mode = p05_evidence_mode_enabled(args_trainer)

    try:
        name, trainer_function = _resolve_trainer_function(args_trainer)
    except (ImportError, AttributeError) as exc:
        if p05_evidence_mode:
            name = getattr(
                args_trainer,
                "name",
                getattr(args_trainer, "trainer_name", "Default_trainer"),
            )
            raise RuntimeError(
                f"P05 evidence mode failed to import trainer {name!r}: {exc}"
            ) from exc
        raise
    try:
        return trainer_function(
            args_e=args_environment,
            args_t=args_trainer,
            args_d=args_data,
            path=path,
        )
    except Exception as exc:
        function_name = getattr(
            trainer_function,
            "__name__",
            type(trainer_function).__name__,
        )
        if p05_evidence_mode:
            raise RuntimeError(
                f"P05 evidence mode failed to create trainer {name!r}: {exc}"
            ) from exc
        raise RuntimeError(
            f"Cannot construct trainer {name!r} with {function_name}: {exc}. "
            "Check trainer settings, device availability, and output path."
        ) from exc


__all__ = [
    "TRAINER_REGISTRY",
    "register_trainer",
    "resolve_trainer_module",
    "trainer_factory",
]
