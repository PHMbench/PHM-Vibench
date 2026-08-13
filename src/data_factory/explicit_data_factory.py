"""Default data factory with explicit adapters and atomic cache publication."""

from __future__ import annotations

from typing import Any

from tqdm import tqdm

from .contracts import format_loader_summary, require_nonempty_dataloaders
from .data_factory import data_factory
from .dataset_task.Dataset_cluster import IdIncludedDataset
from .dataset_task.adapters import resolve_dataset_adapter
from .splitting import resolve_data_splits


def _plain_values(value: Any) -> set[Any]:
    """Return scalar metadata values without imposing a new metadata schema."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    values = value if isinstance(value, (list, tuple, set)) else [value]

    result: set[Any] = set()
    for item in values:
        if hasattr(item, "item"):
            item = item.item()
        result.add(item)
    return result


def _metadata_values(
    metadata: Any,
    file_ids: set[Any],
    field: str,
) -> set[Any] | None:
    """Return split metadata values, or ``None`` when the optional fact is unavailable."""
    values: set[Any] = set()
    for file_id in file_ids:
        try:
            raw = metadata[file_id][field]
        except (KeyError, TypeError, IndexError):
            return None
        values.update(_plain_values(raw))
    return values


def _records(dataset_map: dict[Any, Any]) -> list[tuple[Any, int, int]] | None:
    records: list[tuple[Any, int, int]] = []
    for file_id, dataset in dataset_map.items():
        intervals = getattr(dataset, "window_intervals", None)
        if intervals is None:
            return None
        records.extend(
            (file_id, int(start), int(end)) for start, end in intervals
        )
    return records


def _raw_interval_overlap(
    left: list[tuple[Any, int, int]] | None,
    right: list[tuple[Any, int, int]] | None,
) -> bool | None:
    """Report raw-sample overlap for windows belonging to the same source file."""
    if left is None or right is None:
        return None

    right_by_file: dict[Any, list[tuple[int, int]]] = {}
    for file_id, start, end in right:
        right_by_file.setdefault(file_id, []).append((start, end))

    for file_id, left_start, left_end in left:
        for right_start, right_end in right_by_file.get(file_id, ()):
            if max(left_start, right_start) < min(left_end, right_end):
                return True
    return False


def _set_overlap(
    left: set[Any] | None,
    right: set[Any] | None,
) -> list[Any] | None:
    if left is None or right is None:
        return None
    return sorted(left & right, key=str)


def _summarize_split_assignments(
    split_maps: dict[str, dict[Any, Any]],
    metadata: Any,
) -> dict[str, Any]:
    """Summarize actual file/domain/class and raw-window relationships."""
    split_files = {
        split: set(dataset_map.keys())
        for split, dataset_map in split_maps.items()
    }
    split_domains = {
        split: _metadata_values(metadata, file_ids, "Domain_id")
        for split, file_ids in split_files.items()
    }
    split_labels = {
        split: _metadata_values(metadata, file_ids, "Label")
        for split, file_ids in split_files.items()
    }
    split_records = {
        split: _records(dataset_map)
        for split, dataset_map in split_maps.items()
    }

    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    raw_overlap = {
        f"{left}_{right}": _raw_interval_overlap(
            split_records[left],
            split_records[right],
        )
        for left, right in pairs
    }
    file_overlap = {
        f"{left}_{right}": sorted(
            split_files[left] & split_files[right],
            key=str,
        )
        for left, right in pairs
    }
    domain_overlap = {
        f"{left}_{right}": _set_overlap(
            split_domains[left],
            split_domains[right],
        )
        for left, right in pairs
    }
    classes = {
        split: None if values is None else sorted(values, key=str)
        for split, values in split_labels.items()
    }
    train_labels = split_labels["train"]
    test_labels = split_labels["test"]
    test_classes_seen = (
        None
        if train_labels is None or test_labels is None
        else test_labels.issubset(train_labels)
    )

    return {
        "raw_interval_overlap": raw_overlap,
        "file_overlap": file_overlap,
        "domain_overlap": domain_overlap,
        "classes": classes,
        "test_classes_seen_in_train": test_classes_seen,
    }


def _format_split_summary(summary: dict[str, Any]) -> str:
    raw = summary["raw_interval_overlap"]
    files = summary["file_overlap"]
    return (
        "raw-overlap "
        f"train/val={raw['train_val']}, "
        f"train/test={raw['train_test']}, "
        f"val/test={raw['val_test']}; "
        "file-overlap "
        f"train/val={files['train_val']}, "
        f"train/test={files['train_test']}; "
        "test-classes-seen-in-train="
        f"{summary['test_classes_seen_in_train']}"
    )


class ExplicitDataFactory(data_factory):
    """Build data through explicit adapters and publish only usable data stacks.

    Reader behavior, ID selection, windowing, samplers and DataLoaders remain in
    their existing modules. This class owns user-visible boundaries for explicit
    adapters, complete caches, non-empty loaders, and observable split facts.
    """

    def __init__(self, args_data, args_task):
        super().__init__(args_data, args_task)
        counts = require_nonempty_dataloaders(
            self,
            args_task,
            args_data,
        )
        print(f"[SUCCESS] 数据加载器可用: {format_loader_summary(counts)}")

    def _init_dataset(self):
        task_type = str(self.args_task.type)
        task_name = str(self.args_task.name)
        dataset_cls = resolve_dataset_adapter(task_type, task_name)

        train_dataset = {}
        val_dataset = {}
        test_dataset = {}
        train_val_ids, task_test_ids = self.search_id()
        self.split_result = resolve_data_splits(
            self.target_metadata,
            self.args_data,
            self.args_task,
            train_val_ids,
            task_test_ids,
        )

        print(
            "Initializing datasets with explicit adapter "
            f"{dataset_cls.__module__}.{dataset_cls.__name__} "
            f"for {task_type}/{task_name}."
        )
        for file_id in tqdm(
            self.split_result.train_ids,
            desc="Creating train datasets",
        ):
            file_data = {file_id: self.data[file_id]}
            train_dataset[file_id] = dataset_cls(
                file_data,
                self.target_metadata,
                self.args_data,
                self.args_task,
                "train",
            )
        for file_id in tqdm(
            self.split_result.val_ids,
            desc="Creating val datasets",
        ):
            val_dataset[file_id] = dataset_cls(
                {file_id: self.data[file_id]},
                self.target_metadata,
                self.args_data,
                self.args_task,
                "val",
            )

        for file_id in tqdm(
            self.split_result.test_ids,
            desc="Creating test datasets",
        ):
            test_dataset[file_id] = dataset_cls(
                {file_id: self.data[file_id]},
                self.target_metadata,
                self.args_data,
                self.args_task,
                "test",
            )

        self.split_summary = _summarize_split_assignments(
            {
                "train": train_dataset,
                "val": val_dataset,
                "test": test_dataset,
            },
            self.target_metadata,
        )
        print(f"[DATA SPLIT] {_format_split_summary(self.split_summary)}")

        return (
            IdIncludedDataset(train_dataset, self.target_metadata),
            IdIncludedDataset(val_dataset, self.target_metadata),
            IdIncludedDataset(test_dataset, self.target_metadata),
        )


__all__ = ["ExplicitDataFactory"]
