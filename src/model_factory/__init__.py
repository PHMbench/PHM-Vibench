"""Public API for model construction.

Importing a concrete model module must not initialize the full training stack.
Factory internals are therefore loaded only when a factory function is called.
"""

from __future__ import annotations

from typing import Any


def resolve_model_module(args_model: Any) -> str:
    """Return the import path for the requested model module."""
    from .model_factory import resolve_model_module as _resolve_model_module

    return _resolve_model_module(args_model)


def model_factory(args_model: Any, metadata: Any = None) -> Any:
    """Instantiate a configured model without eager package initialization."""
    from .model_factory import model_factory as _model_factory

    return _model_factory(args_model, metadata=metadata)


def build_model(args: Any, metadata: Any = None) -> Any:
    """Instantiate a configured model, optionally using dataset metadata."""
    return model_factory(args, metadata=metadata)


__all__ = ["build_model", "model_factory", "resolve_model_module"]
