"""Public API for the model factory package."""

from typing import Any

from .model_factory import (
    model_factory,
    resolve_model_module,
)


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

    return model_factory(args, metadata=metadata)



# public exports
__all__ = [
    "build_model",
    "resolve_model_module",
]
