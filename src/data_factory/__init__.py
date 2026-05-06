"""Public API for the data factory package."""

from __future__ import annotations

import importlib
from typing import Any


def _factory_module():
    return importlib.import_module(f"{__name__}.data_factory")


def register_data_factory(name: str):
    return _factory_module().register_data_factory(name)




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
    module = _factory_module()
    try:
        return module.DATA_FACTORY_REGISTRY.get(name)
    except KeyError:
        # default fallback
        if name == "default":
            return module.data_factory
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


def __getattr__(name: str):
    if name in {"DATA_FACTORY_REGISTRY", "data_factory"}:
        return getattr(_factory_module(), name)
    if name == "IdIncludedDataset":
        return importlib.import_module(f"{__name__}.dataset_task.Dataset_cluster").IdIncludedDataset
    if name == "id_data_factory":
        return importlib.import_module(f"{__name__}.id_data_factory").id_data_factory
    raise AttributeError(name)


# public exports
__all__ = [
    "build_data",
    "resolve_data_factory_class",
    "register_data_factory",
    "DATA_FACTORY_REGISTRY",
    "IdIncludedDataset",
    "id_data_factory",
]
