"""Public API for model construction.

Importing a concrete model module must not initialize the full training stack.
Factory internals are therefore loaded only when a public function is called.
"""

from __future__ import annotations

from typing import Any


def resolve_model_module(args_model: Any) -> str:
    """Return the import path for the requested model module."""
    from .model_factory import resolve_model_module as implementation

    return implementation(args_model)


def build_model(args: Any, metadata: Any = None) -> Any:
    """Instantiate a configured model, optionally using dataset metadata."""
    from .model_factory import model_factory as implementation

    return implementation(args, metadata=metadata)


__all__ = ["build_model", "resolve_model_module"]
