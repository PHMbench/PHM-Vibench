"""Public API for PHMFactory data construction and dataset extensions."""

from typing import Any

from .data_factory import (
    DATA_FACTORY_REGISTRY,
    data_factory,
    register_data_factory,
)
from .dataset_task.Dataset_cluster import IdIncludedDataset
from .dataset_task.adapters import (
    DATASET_ADAPTERS,
    register_dataset_adapter,
    resolve_dataset_adapter,
)
from .explicit_data_factory import ExplicitDataFactory
from .id_data_factory import id_data_factory
from .phm_data_factory import PHMDataFactory
from .standalone import build_data_repository


def resolve_data_factory_class(name: str):
    """Return one explicit data factory class.

    ``default`` uses :class:`ExplicitDataFactory`, which resolves the dataset
    implementation from the registered ``(task.type, task.name)`` contract. Other
    factory names must be present in ``DATA_FACTORY_REGISTRY``.
    """

    requested = str(name)
    if requested == "default":
        return ExplicitDataFactory

    try:
        return DATA_FACTORY_REGISTRY.get(requested)
    except KeyError as exc:
        available = ["default", "department", "id"]
        raise ValueError(
            f"Unknown data.factory_name={requested!r}. "
            f"Available factories: {', '.join(available)}"
        ) from exc


def build_data(args_data: Any, args_task: Any) -> Any:
    """Instantiate the configured data factory for one task contract."""

    name = getattr(args_data, "factory_name", "default")
    factory_cls = resolve_data_factory_class(name)
    return factory_cls(args_data, args_task)


__all__ = [
    "build_data",
    "build_data_repository",
    "resolve_data_factory_class",
    "register_data_factory",
    "DATA_FACTORY_REGISTRY",
    "data_factory",
    "ExplicitDataFactory",
    "DATASET_ADAPTERS",
    "register_dataset_adapter",
    "resolve_dataset_adapter",
    "IdIncludedDataset",
    "id_data_factory",
    "PHMDataFactory",
]
