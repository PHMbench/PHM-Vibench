from __future__ import annotations

import pandas as pd

from ..data_utils import MetadataAccessor
from .Get_id import Get_CDDG_ids, Get_DG_ids


def _config_get(container, key, default=None):
    """Read a key from dict-, namespace-, or OmegaConf-like containers."""
    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(key, default)
    getter = getattr(container, "get", None)
    if callable(getter):
        value = getter(key, default)
        return default if value is None else value
    return getattr(container, key, default)


def _integer_mapping(container):
    if container is None:
        return {}
    items = container.items() if hasattr(container, "items") else []
    return {int(key): int(value) for key, value in items}


def apply_label_ontology(frame, args_task):
    """Map dataset-local labels into a prespecified shared semantic space."""
    ontology = _config_get(args_task, "label_ontology")
    if ontology is None:
        return frame

    mappings = _config_get(ontology, "mappings")
    if mappings is None:
        raise ValueError("label_ontology.mappings is required")
    excluded = _config_get(ontology, "excluded_labels", {})
    num_classes = int(_config_get(ontology, "num_classes", 0))
    if num_classes < 2:
        raise ValueError("label_ontology.num_classes must be at least 2")

    mapped = frame.copy()
    mapped["Raw_Label"] = mapped["Label"].astype(int)
    keep_mask = pd.Series(True, index=mapped.index)
    common_labels = pd.Series(index=mapped.index, dtype="float64")

    for dataset_id in sorted(mapped["Dataset_id"].astype(int).unique()):
        dataset_mask = mapped["Dataset_id"].astype(int) == dataset_id
        raw_to_common = _integer_mapping(_config_get(mappings, str(dataset_id)))
        if not raw_to_common:
            raw_to_common = _integer_mapping(_config_get(mappings, dataset_id))
        if not raw_to_common:
            raise ValueError(
                f"label ontology has no mapping for Dataset_id {dataset_id}"
            )

        excluded_values = _config_get(excluded, str(dataset_id), [])
        if not excluded_values:
            excluded_values = _config_get(excluded, dataset_id, [])
        excluded_set = {int(value) for value in excluded_values}
        if excluded_set:
            keep_mask &= ~(
                dataset_mask & mapped["Raw_Label"].isin(excluded_set)
            )

        eligible = dataset_mask & keep_mask
        common_labels.loc[eligible] = mapped.loc[eligible, "Raw_Label"].map(
            raw_to_common
        )

    mapped = mapped.loc[keep_mask].copy()
    mapped_labels = common_labels.loc[keep_mask]
    if mapped_labels.isna().any():
        missing_rows = mapped.loc[mapped_labels.isna(), ["Dataset_id", "Raw_Label"]]
        missing = sorted(
            {tuple(row) for row in missing_rows.astype(int).to_numpy().tolist()}
        )
        raise ValueError(f"unmapped raw labels in shared ontology: {missing}")

    mapped["Label"] = mapped_labels.astype(int)
    invalid = mapped[(mapped["Label"] < 0) | (mapped["Label"] >= num_classes)]
    if not invalid.empty:
        pairs = sorted(
            {
                tuple(row)
                for row in invalid[["Dataset_id", "Label"]]
                .astype(int)
                .to_numpy()
                .tolist()
            }
        )
        raise ValueError(f"mapped labels outside configured ontology: {pairs}")
    return mapped


_ALL_ID_TASK_TYPES = frozenset(
    {
        "FS",
        "GFS",
        "pretrain",
        "Pretrain",
        "generative",
        "multi_task",
        "In_distribution",
        "Default_task",
    }
)
_TASKS_ALLOWING_ALL_SYSTEMS = frozenset(
    {"pretrain", "Pretrain", "generative", "Default_task"}
)
_SUPERVISED_TASK_TYPES = frozenset(
    {
        "DG",
        "CDDG",
        "FS",
        "GFS",
        "multi_task",
        "In_distribution",
        "Default_task",
    }
)


def search_ids_for_task(metadata_accessor, args_task):
    """Return the exact train/validation and test IDs for one known task type."""

    task_type = str(getattr(args_task, "type", "") or "").strip()
    if not task_type:
        raise ValueError("task.type must be a non-empty string before selecting data IDs.")

    target_system_id = getattr(args_task, "target_system_id", None)
    if not target_system_id and task_type not in _TASKS_ALLOWING_ALL_SYSTEMS:
        raise ValueError(
            f"task.type={task_type!r} requires a non-empty task.target_system_id. "
            "Declare the intended system population instead of using all metadata rows."
        )

    if task_type == "DG":
        train_val_ids, test_ids = Get_DG_ids(metadata_accessor, args_task)
    elif task_type == "CDDG":
        train_val_ids, test_ids = Get_CDDG_ids(metadata_accessor, args_task)
    elif task_type in _ALL_ID_TASK_TYPES:
        # These maintained compatibility paths intentionally reuse the selected
        # metadata population. Their scientific support/query semantics are reviewed
        # separately; this function must not invent another split.
        selected_ids = list(metadata_accessor.keys())
        train_val_ids, test_ids = selected_ids, list(selected_ids)
    else:
        supported = sorted({"DG", "CDDG", *_ALL_ID_TASK_TYPES})
        raise ValueError(
            f"Unknown task.type={task_type!r} for data ID selection. "
            f"Supported task types: {', '.join(supported)}."
        )

    if not train_val_ids:
        raise ValueError(
            f"task.type={task_type!r} selected no training/validation IDs. "
            "Check target_system_id, domain settings, labels, and metadata."
        )
    if task_type in {"DG", "CDDG"} and not test_ids:
        raise ValueError(
            f"task.type={task_type!r} selected no test IDs. "
            "The requested domain protocol cannot be executed."
        )

    return train_val_ids, test_ids


def search_target_dataset_metadata(metadata_accessor, args_task):
    """Select the declared system population without dropping invalid rows silently."""

    task_type = str(getattr(args_task, "type", "") or "").strip()
    target_system_id = getattr(args_task, "target_system_id", None)

    if not target_system_id:
        if task_type in _TASKS_ALLOWING_ALL_SYSTEMS:
            return metadata_accessor
        raise ValueError(
            f"task.type={task_type!r} requires task.target_system_id before metadata "
            "selection."
        )

    if "Dataset_id" not in metadata_accessor.df.columns:
        raise ValueError("Metadata must contain a 'Dataset_id' column.")

    filtered_df = metadata_accessor.df[
        metadata_accessor.df["Dataset_id"].isin(target_system_id)
    ].copy()

    if filtered_df.empty:
        raise ValueError(
            f"task.target_system_id={list(target_system_id)!r} matched no metadata rows."
        )

    selected_files = getattr(args_task, "selected_files", None)
    grouped_split = getattr(args_task, "grouped_split", None)
    grouped_enabled = bool(getattr(grouped_split, "enabled", False))
    if selected_files is None and grouped_enabled:
        raw_groups = getattr(grouped_split, "groups", None)
        if not isinstance(raw_groups, (list, tuple)) or not raw_groups:
            raise ValueError(
                "task.grouped_split.groups must declare the selected files."
            )
        selected_files = []
        for group in raw_groups:
            files = getattr(group, "files", None)
            if not isinstance(files, (list, tuple)) or not files:
                raise ValueError(
                    "Every task.grouped_split group must declare files."
                )
            selected_files.extend(files)
    if selected_files is not None:
        if isinstance(selected_files, (str, bytes)):
            selected_files = [selected_files]
        else:
            selected_files = list(selected_files)
        if not selected_files:
            raise ValueError("task.selected_files must contain at least one file.")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in selected_files
        ):
            raise ValueError(
                "task.selected_files must contain only non-empty file names."
            )
        if len(set(selected_files)) != len(selected_files):
            raise ValueError("task.selected_files contains duplicate file names.")
        if "File" not in filtered_df.columns:
            raise ValueError(
                "task.selected_files requires metadata column 'File'."
            )

        selected_set = set(selected_files)
        available_counts = filtered_df["File"].value_counts(dropna=False)
        missing = sorted(selected_set - set(available_counts.index), key=str)
        if missing:
            raise ValueError(
                "task.selected_files contains file(s) absent from the selected "
                f"system metadata: {missing}."
            )
        ambiguous = sorted(
            value for value in selected_set if int(available_counts[value]) != 1
        )
        if ambiguous:
            raise ValueError(
                "task.selected_files must identify exactly one metadata row per "
                f"file; ambiguous file(s): {ambiguous}."
            )
        filtered_df = filtered_df[filtered_df["File"].isin(selected_set)].copy()

    label_policy = str(getattr(args_task, "label_policy", "native"))
    if label_policy == "binary_fault":
        if "Label" not in filtered_df.columns:
            raise ValueError("binary_fault label policy requires metadata column 'Label'.")
        filtered_df = filtered_df[
            filtered_df["Label"].notna() & (filtered_df["Label"] != -1)
        ].copy()
        filtered_df["Label"] = (
            filtered_df["Label"].astype(float) > 0.0
        ).astype(int)
    elif label_policy != "native":
        raise ValueError(f"Unknown task.label_policy: {label_policy}")

    if task_type in _SUPERVISED_TASK_TYPES:
        if "Label" not in filtered_df.columns:
            raise ValueError(
                f"Supervised task.type={task_type!r} requires metadata column 'Label'."
            )
        invalid = filtered_df[filtered_df["Label"].isna() | (filtered_df["Label"] == -1)]
        if not invalid.empty:
            invalid_ids = (
                invalid["Id"].tolist()
                if "Id" in invalid.columns
                else invalid.index.tolist()
            )
            raise ValueError(
                "Supervised metadata contains missing or -1 labels for ID(s) "
                f"{invalid_ids}. Fix the metadata instead of dropping rows."
            )

    filtered_df = apply_label_ontology(filtered_df, args_task)
    filtered_df.reset_index(drop=True, inplace=True)
    return MetadataAccessor(filtered_df, key_column=metadata_accessor.key_column)
