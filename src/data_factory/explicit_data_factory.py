"""Default data factory with explicit task-to-dataset adapter resolution."""

from __future__ import annotations

from tqdm import tqdm

from .data_factory import data_factory
from .dataset_task.Dataset_cluster import IdIncludedDataset
from .dataset_task.adapters import resolve_dataset_adapter


class ExplicitDataFactory(data_factory):
    """Build datasets only from registered task adapters.

    Cache construction, ID selection, samplers, and DataLoader creation remain in
    the historical ``data_factory`` implementation. This subclass replaces only
    the ambiguous dataset-module lookup and its silent fallback.
    """

    def _init_dataset(self):
        task_type = str(self.args_task.type)
        task_name = str(self.args_task.name)
        dataset_cls = resolve_dataset_adapter(task_type, task_name)

        train_dataset = {}
        val_dataset = {}
        test_dataset = {}
        train_val_ids, test_ids = self.search_id()

        print(
            "Initializing datasets with explicit adapter "
            f"{dataset_cls.__module__}.{dataset_cls.__name__} "
            f"for {task_type}/{task_name}."
        )
        for file_id in tqdm(
            train_val_ids,
            desc="Creating train/val datasets",
        ):
            file_data = {file_id: self.data[file_id]}
            train_dataset[file_id] = dataset_cls(
                file_data,
                self.target_metadata,
                self.args_data,
                self.args_task,
                "train",
            )
            val_dataset[file_id] = dataset_cls(
                file_data,
                self.target_metadata,
                self.args_data,
                self.args_task,
                "val",
            )

        for file_id in tqdm(test_ids, desc="Creating test datasets"):
            test_dataset[file_id] = dataset_cls(
                {file_id: self.data[file_id]},
                self.target_metadata,
                self.args_data,
                self.args_task,
                "test",
            )

        return (
            IdIncludedDataset(train_dataset, self.target_metadata),
            IdIncludedDataset(val_dataset, self.target_metadata),
            IdIncludedDataset(test_dataset, self.target_metadata),
        )


__all__ = ["ExplicitDataFactory"]
