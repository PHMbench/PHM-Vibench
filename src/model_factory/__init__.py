"""Public API for the model factory package."""

from typing import Any, Callable, Dict, Tuple, TypeVar

from .model_factory import (
    model_factory,
    resolve_model_module,
)

T = TypeVar("T")
MODEL_CLASS_REGISTRY: Dict[Tuple[str, str], Any] = {}


def register_model(model_type: str, model_name: str) -> Callable[[T], T]:
    """Register a model class for legacy decorator-based modules."""

    def decorator(model_cls: T) -> T:
        MODEL_CLASS_REGISTRY[(model_type, model_name)] = model_cls
        return model_cls

    return decorator


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
    "MODEL_CLASS_REGISTRY",
    "register_model",
    "resolve_model_module",
]
