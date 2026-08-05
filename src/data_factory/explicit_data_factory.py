"""Default data factory with explicit adapters and atomic cache publication."""

from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
import shutil

import h5py
from tqdm import tqdm

from .data_factory import data_factory
from .dataset_task.Dataset_cluster import IdIncludedDataset
from .dataset_task.adapters import resolve_dataset_adapter


class ExplicitDataFactory(data_factory):
    """Build data through explicit adapters and publish only complete caches.

    Reader behavior, ID selection, windowing, samplers and DataLoaders remain in
    their existing modules. This class owns two user-visible boundaries:

    - task-to-dataset selection is explicit;
    - a cache path is replaced only after every requested ID is present.
    """

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
        """Build the current task cache completely, then publish it atomically."""
        del use_cache  # The final cache is rebuilt from validated dataset caches.

        expected_ids = list(task_meta.keys())
        if not expected_ids:
            raise ValueError(
                "The selected task contains no data IDs. Check task.target_system_id, "
                "domain selection, labels, and metadata."
            )

        cache_path = Path(args_data.data_dir) / "cache.h5"
        temp_path = cache_path.with_name(".cache.h5.tmp")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
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

        return (
            IdIncludedDataset(train_dataset, self.target_metadata),
            IdIncludedDataset(val_dataset, self.target_metadata),
            IdIncludedDataset(test_dataset, self.target_metadata),
        )


__all__ = ["ExplicitDataFactory"]
