"""Task-aware batch sampler selection."""

from collections.abc import Mapping

import pandas as pd

from .Sampler import HierarchicalFewShotSampler, Same_system_Sampler


def _missing_dataset_id(metadata_entry) -> bool:
    if not isinstance(metadata_entry, Mapping) or "Dataset_id" not in metadata_entry:
        return True
    value = metadata_entry["Dataset_id"]
    if isinstance(value, (list, tuple, set, dict)):
        return True
    return bool(pd.isna(value))


def _require_metadata_coverage(dataset) -> None:
    """Require every selected sample to resolve one explicit Dataset_id."""

    windows = getattr(dataset, "file_windows_list", None)
    if not isinstance(windows, list) or not windows:
        raise ValueError("Sampler requires a non-empty IdIncludedDataset population.")
    metadata = getattr(dataset, "metadata", None)
    if metadata is None:
        raise ValueError("Sampler requires metadata for every selected file ID.")

    selected_file_ids = {item["file_id"] for item in windows}
    missing_rows = [file_id for file_id in selected_file_ids if file_id not in metadata]
    if missing_rows:
        raise ValueError(
            "Sampler metadata is missing selected file ID(s) "
            f"{sorted(missing_rows, key=str)}."
        )

    missing_systems = [
        file_id
        for file_id in selected_file_ids
        if _missing_dataset_id(metadata[file_id])
    ]
    if missing_systems:
        raise ValueError(
            "Sampler metadata is missing Dataset_id for selected file ID(s) "
            f"{sorted(missing_systems, key=str)}."
        )


def _evaluation_sampler(args_data, dataset):
    """Keep every validation/test sample, including a final short batch."""
    return Same_system_Sampler(
        dataset=dataset,
        batch_size=args_data.batch_size,
        shuffle=False,
        drop_last=False,
    )


def _get_gfs_sampler(args_task, args_data, dataset, mode):
    if mode == "train":
        return HierarchicalFewShotSampler(
            dataset=dataset,
            num_episodes=args_task.num_episodes,
            num_systems_per_episode=args_task.num_systems,
            num_domains_per_system=args_task.num_domains,
            num_labels_per_domain_task=args_task.num_labels,
            num_support_per_label=args_task.num_support,
            num_query_per_label=args_task.num_query,
        )
    if mode in {"val", "test"}:
        return _evaluation_sampler(args_data, dataset)
    raise ValueError(f"Unknown mode for GFS sampler: {mode}")


def _get_standard_sampler(args_data, dataset, mode, task_name):
    if mode == "train":
        return Same_system_Sampler(
            dataset=dataset,
            batch_size=args_data.batch_size,
            shuffle=True,
            drop_last=False,
        )
    if mode in {"val", "test"}:
        return _evaluation_sampler(args_data, dataset)
    raise ValueError(f"Unknown mode for {task_name} sampler: {mode}")


def Get_sampler(args_task, args_data, dataset, mode="train"):
    """Return the sampler for one explicit task type and complete sample population."""

    _require_metadata_coverage(dataset)
    task_type = args_task.type

    if task_type == "GFS":
        return _get_gfs_sampler(args_task, args_data, dataset, mode)
    if task_type == "FS":
        return _get_standard_sampler(args_data, dataset, mode, "FS")
    if task_type in {"pretrain", "generative"}:
        return _get_standard_sampler(args_data, dataset, mode, "Pretrain")
    if task_type == "CDDG":
        return _get_standard_sampler(args_data, dataset, mode, "CDDG")
    if task_type == "DG":
        return _get_standard_sampler(args_data, dataset, mode, "DG")
    if task_type == "multi_task":
        return _get_standard_sampler(args_data, dataset, mode, "multi_task")
    if task_type == "In_distribution":
        return _get_standard_sampler(
            args_data,
            dataset,
            mode,
            "In_distribution",
        )

    raise ValueError(f"Unknown task type for sampler: {task_type}")
