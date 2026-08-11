"""Explicit dataset adapter registry keyed by ``(task.type, task.name)``."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any


@dataclass(frozen=True)
class DatasetAdapterSpec:
    """Import location for one dataset implementation."""

    module: str
    attribute: str = "set_dataset"


DATASET_ADAPTERS: dict[tuple[str, str], DatasetAdapterSpec] = {
    ("Default_task", "Default_task"): DatasetAdapterSpec(
        "src.data_factory.dataset_task.Default_dataset",
        "Default_dataset",
    ),
    ("Default_task", "ID_task"): DatasetAdapterSpec(
        "src.data_factory.dataset_task.ID.Classification_dataset"
    ),
    ("DG", "classification"): DatasetAdapterSpec(
        "src.data_factory.dataset_task.DG.Classification_dataset"
    ),
    ("CDDG", "classification"): DatasetAdapterSpec(
        "src.data_factory.dataset_task.CDDG.classification_dataset"
    ),
    ("FS", "classification"): DatasetAdapterSpec(
        "src.data_factory.dataset_task.FS.Classification_dataset"
    ),
    ("FS", "prototypical_network"): DatasetAdapterSpec(
        "src.data_factory.dataset_task.FS.Classification_dataset"
    ),
    ("FS", "matching_network"): DatasetAdapterSpec(
        "src.data_factory.dataset_task.FS.Classification_dataset"
    ),
    ("FS", "knn_feature"): DatasetAdapterSpec(
        "src.data_factory.dataset_task.FS.Classification_dataset"
    ),
    ("FS", "finetuning"): DatasetAdapterSpec(
        "src.data_factory.dataset_task.FS.Classification_dataset"
    ),
    ("GFS", "classification"): DatasetAdapterSpec(
        "src.data_factory.dataset_task.GFS.Classification_dataset"
    ),
    ("GFS", "matching"): DatasetAdapterSpec(
        "src.data_factory.dataset_task.GFS.Classification_dataset"
    ),
    ("pretrain", "classification"): DatasetAdapterSpec(
        "src.data_factory.dataset_task.Pretrain.Classification_dataset"
    ),
    ("pretrain", "hse_contrastive"): DatasetAdapterSpec(
        "src.data_factory.dataset_task.Pretrain.Classification_dataset"
    ),
    ("pretrain", "masked_reconstruction"): DatasetAdapterSpec(
        "src.data_factory.dataset_task.Pretrain.Classification_dataset"
    ),
    ("pretrain", "prediction"): DatasetAdapterSpec(
        "src.data_factory.dataset_task.Pretrain.Classification_dataset"
    ),
    ("pretrain", "classification_prediction"): DatasetAdapterSpec(
        "src.data_factory.dataset_task.Pretrain.Classification_dataset"
    ),
    ("generative", "conditional_flow_matching"): DatasetAdapterSpec(
        "src.data_factory.dataset_task.Default_dataset",
        "Default_dataset",
    ),
    ("In_distribution", "multi_task_phm"): DatasetAdapterSpec(
        "src.data_factory.dataset_task.Default_dataset",
        "Default_dataset",
    ),
}


def register_dataset_adapter(
    task_type: str,
    task_name: str,
    module: str,
    attribute: str = "set_dataset",
) -> None:
    """Register one explicit dataset adapter for an extension package."""

    key = (str(task_type), str(task_name))
    if key in DATASET_ADAPTERS:
        raise ValueError(f"dataset adapter already registered for {key!r}")
    DATASET_ADAPTERS[key] = DatasetAdapterSpec(module, attribute)


def resolve_dataset_adapter(task_type: Any, task_name: Any):
    """Return the dataset class for one exact task combination.

    Unknown combinations fail with the supported keys. Import errors retain their
    original cause instead of falling back to ``Default_dataset``.
    """

    key = (str(task_type), str(task_name))
    spec = DATASET_ADAPTERS.get(key)
    if spec is None:
        supported = ", ".join(
            f"{item_type}/{item_name}"
            for item_type, item_name in sorted(DATASET_ADAPTERS)
        )
        raise ValueError(
            f"No dataset adapter is registered for task {key[0]}/{key[1]}. "
            f"Registered combinations: {supported}. Add an explicit adapter "
            "before running this task."
        )

    try:
        module = importlib.import_module(spec.module)
    except ImportError as exc:
        raise ImportError(
            f"Dataset adapter for {key[0]}/{key[1]} could not import "
            f"{spec.module}: {exc}"
        ) from exc

    try:
        dataset_class = getattr(module, spec.attribute)
    except AttributeError as exc:
        raise ImportError(
            f"Dataset adapter for {key[0]}/{key[1]} expected "
            f"{spec.module}.{spec.attribute}, but that attribute does not exist."
        ) from exc

    return dataset_class


__all__ = [
    "DATASET_ADAPTERS",
    "DatasetAdapterSpec",
    "register_dataset_adapter",
    "resolve_dataset_adapter",
]
