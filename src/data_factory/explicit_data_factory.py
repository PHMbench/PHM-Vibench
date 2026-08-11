"""Default data factory with explicit adapters and atomic cache publication."""

from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
import shutil
from typing import Any

import h5py
from tqdm import tqdm

from .contracts import format_loader_summary, require_nonempty_dataloaders
from .data_factory import data_factory
from .dataset_task.Dataset_cluster import IdIncludedDataset
from .dataset_task.adapters import resolve_dataset_adapter


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

    def _update_name_cache(self, name, ids, args_data, max_workers):
        """Read all requested IDs and atomically update one dataset cache."""
        if not ids:
            return

        id_meta_pairs = []
        missing_metadata = []
        for file_id in ids:
            meta = self.metadata[file_id]
            if not meta.get("File"):
                missing_metadata.append(str(file_id))
                continue
            id_meta_pairs.append((file_id, meta))

        if missing_metadata:
            raise RuntimeError(
                f"Cannot build cache for dataset {name!r}: metadata is missing "
                f"File for ID(s) {', '.join(missing_metadata)}. Fix the metadata "
                "before rerunning."
            )

        results = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = [
                executor.submit(
                    self._read_single_data,
                    file_id,
                    meta,
                    args_data,
                )
                for file_id, meta in id_meta_pairs
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"并行读取 {name}",
            ):
                results.append(future.result())

        failures = [
            (str(file_id), error or "reader returned no data")
            for file_id, data, error in results
            if data is None
        ]
        if failures:
            details = "; ".join(
                f"ID {file_id}: {reason}" for file_id, reason in failures
            )
            raise RuntimeError(
                f"Cannot publish cache for dataset {name!r}; raw-data reading "
                f"failed. {details}"
            )

        cache_path = Path(args_data.data_dir) / f"{name}.h5"
        temp_path = cache_path.with_name(f".{cache_path.name}.tmp")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.unlink(missing_ok=True)

        try:
            if cache_path.is_file():
                shutil.copy2(cache_path, temp_path)
            with h5py.File(temp_path, "a") as h5_file:
                for file_id, data, _ in results:
                    key = str(file_id)
                    if key in h5_file:
                        del h5_file[key]
                    h5_file.create_dataset(key, data=data)

                missing_ids = [
                    str(file_id)
                    for file_id in ids
                    if str(file_id) not in h5_file
                ]
                if missing_ids:
                    raise RuntimeError(
                        f"Temporary cache {temp_path} is missing ID(s) "
                        f"{', '.join(missing_ids)}."
                    )

            os.replace(temp_path, cache_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def _build_final_cache(self, task_meta, args_data, use_cache):
        """Reuse a complete task cache or rebuild it before atomic publication."""
        expected_ids = list(task_meta.keys())
        if not expected_ids:
            raise ValueError(
                "The selected task contains no data IDs. Check task.target_system_id, "
                "domain selection, labels, and metadata."
            )

        expected_keys = {str(file_id) for file_id in expected_ids}
        cache_path = Path(args_data.data_dir) / "cache.h5"
        temp_path = cache_path.with_name(".cache.h5.tmp")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if use_cache and cache_path.is_file():
            try:
                with h5py.File(cache_path, "r") as published_cache:
                    if expected_keys.issubset(published_cache.keys()):
                        return str(cache_path)
            except OSError as exc:
                raise RuntimeError(
                    f"Existing cache cannot be opened: {cache_path}. Delete this "
                    "cache and rerun so PHMFactory can rebuild it."
                ) from exc

        temp_path.unlink(missing_ok=True)
        missing = []

        try:
            with h5py.File(temp_path, "w") as output_cache:
                for file_id in tqdm(
                    expected_ids,
                    desc="整合 cache.h5",
                ):
                    meta = self.metadata[file_id]
                    dataset_name = meta.get("Name")
                    if not dataset_name:
                        missing.append(
                            (str(file_id), "metadata field Name is missing")
                        )
                        continue

                    source_path = (
                        Path(args_data.data_dir) / f"{dataset_name}.h5"
                    )
                    if not source_path.is_file():
                        missing.append(
                            (str(file_id), f"dataset cache not found: {source_path}")
                        )
                        continue

                    key = str(file_id)
                    with h5py.File(source_path, "r") as source_cache:
                        if key not in source_cache:
                            missing.append(
                                (
                                    key,
                                    f"ID is absent from dataset cache {source_path}",
                                )
                            )
                            continue
                        source_cache.copy(key, output_cache, name=key)

            if missing:
                details = "; ".join(
                    f"ID {file_id}: {reason}"
                    for file_id, reason in missing
                )
                raise RuntimeError(
                    "Cannot publish cache.h5 because the selected data is "
                    f"incomplete. {details}"
                )

            os.replace(temp_path, cache_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        return str(cache_path)

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
