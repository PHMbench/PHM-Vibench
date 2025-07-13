"""Public API for the data factory package."""

from typing import Any

from .data_factory import (
    DATA_FACTORY_REGISTRY,
    data_factory,
    register_data_factory,
)
from .dataset_task.Dataset_cluster import IdIncludedDataset
from .id_data_factory import IDDataFactory


def resolve_data_factory_class(name: str):
    """Return the factory class registered as ``name``.

    Parameters
    ----------
    name : str
        Factory identifier from configuration.

    Returns
    -------
    type
        Data factory class corresponding to ``name``.
    """
    try:
        return DATA_FACTORY_REGISTRY.get(name)
    except KeyError:
        # default fallback
        if name == "default":
            return data_factory
        raise


def build_data(args_data: Any, args_task: Any) -> Any:
    """Instantiate a dataset using the configured factory.

    Parameters
    ----------
    args_data : Any
        Data related configuration namespace.
    args_task : Any
        Task configuration used during dataset creation.

    Returns
    -------
    Any
        Instantiated dataset factory.
    """
    name = getattr(args_data, "factory_name", "default")
    factory_cls = resolve_data_factory_class(name)
    return factory_cls(args_data, args_task)


# public exports
__all__ = [
    "build_data",
    "resolve_data_factory_class",
    "register_data_factory",
    "DATA_FACTORY_REGISTRY",
    "IdIncludedDataset",
    "IDDataFactory",
]
