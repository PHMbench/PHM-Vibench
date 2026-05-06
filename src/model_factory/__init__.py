"""Public API for the model factory package."""

from __future__ import annotations

import importlib
from typing import Any


def _factory_module():
    return importlib.import_module(f"{__name__}.model_factory")


def resolve_model_module(args: Any) -> str:
    return _factory_module().resolve_model_module(args)


def build_model(args: Any, metadata: Any = None) -> Any:
    """Instantiate a model from configuration.

    Parameters
    ----------
    args : Any
        Namespace or dictionary with model options.
    metadata : Any, optional
        Dataset metadata used to compute ``num_classes``.

    Returns
    -------
    Any
        Instantiated model object.
    """

    return _factory_module().model_factory(args, metadata=metadata)



# public exports
__all__ = [
    "build_model",
    "resolve_model_module",
]
