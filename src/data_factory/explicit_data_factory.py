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


def _window_count(dataset_map: dict[Any, Any]) -> int | None:
    total = 0
    for dataset in dataset_map.values():
        try:
            total += len(dataset)
            continue
        except TypeError:
            intervals = getattr(dataset, "window_intervals", None)
        if intervals is None:
            return None
        total += len(intervals)
    return total


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


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _required_config_text(config: Any, name: str) -> str:
    value = _config_value(config, name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"task.grouped_split.{name} must be a non-empty string.")
    return value.strip()


def _grouped_partition_ids(
    train_val_ids: list[Any],
    test_ids: list[Any],
    metadata: Any,
    args_task: Any,
    args_data: Any,
) -> tuple[dict[str, list[Any]], dict[Any, str] | None, dict[str, Any] | None]:
    """Resolve one explicit physical-group protocol without fallback splitting."""

    protocol = getattr(args_task, "grouped_split", None)
    enabled = bool(_config_value(protocol, "enabled", False)) if protocol else False
    if not enabled:
        return (
            {
                "train": list(train_val_ids),
                "val": list(train_val_ids),
                "test": list(test_ids),
            },
            None,
            None,
        )

    if str(getattr(args_task, "type", "")) != "DG":
        raise ValueError(
            "task.grouped_split is currently admitted only for task.type='DG'."
        )
    normalization = str(getattr(args_data, "normalization", "")).lower()
    if normalization != "none":
        raise ValueError(
            "Grouped P01 data requires data.normalization='none' until an explicit "
            "train-fitted normalization state exists; per-window evaluation fitting "
            "is not admissible."
        )

    group_key = _required_config_text(protocol, "group_key")
    group_meaning = _required_config_text(protocol, "group_meaning")
    inferential_unit = _required_config_text(protocol, "inferential_unit")
    verified_run_identity = _required_config_text(protocol, "verified_run_identity")
    observation_hierarchy = _required_config_text(protocol, "observation_hierarchy")
    identity_limit = _required_config_text(protocol, "identity_limit")
    target_label_access_boundary = _required_config_text(
        protocol,
        "target_label_access_boundary",
    )
    endpoint = _required_config_text(protocol, "endpoint")
    excluded_label_0_reason = _required_config_text(
        protocol,
        "excluded_label_0_reason",
    )
    official_sources_config = _config_value(protocol, "official_sources")
    official_sources = {
        name: _required_config_text(official_sources_config, name)
        for name in ("overview", "apparatus", "drive_end_12k", "fan_end_12k")
    }
    admitted_labels = list(_config_value(protocol, "admitted_labels", []) or [])
    if not admitted_labels or len(set(admitted_labels)) != len(admitted_labels):
        raise ValueError(
            "task.grouped_split.admitted_labels must contain unique labels."
        )
    non_authoritative_fields = list(
        _config_value(protocol, "non_authoritative_metadata_fields", []) or []
    )
    if any(
        not isinstance(field, str) or not field.strip()
        for field in non_authoritative_fields
    ):
        raise ValueError(
            "task.grouped_split.non_authoritative_metadata_fields must contain "
            "non-empty field names."
        )
    non_authoritative_fields = [field.strip() for field in non_authoritative_fields]
    metadata_limit = _required_config_text(protocol, "metadata_limit")
    raw_groups = _config_value(protocol, "groups")
    if not isinstance(raw_groups, (list, tuple)) or not raw_groups:
        raise ValueError("task.grouped_split.groups must be a non-empty list.")
    domain_order = list(_config_value(protocol, "domain_order", []) or [])
    if not domain_order or len(set(domain_order)) != len(domain_order):
        raise ValueError(
            "task.grouped_split.domain_order must contain unique domain IDs."
        )
    expected_sample_rate = _config_value(protocol, "expected_sample_rate")

    valid_partitions = {"train", "val", "test"}
    file_to_group: dict[str, str] = {}
    file_expected_domain: dict[str, Any] = {}
    group_partition: dict[str, str] = {}
    group_expected_label: dict[str, Any] = {}
    condition_to_group: dict[tuple[Any, ...], str] = {}
    official_record_keys: set[tuple[str, str]] = set()
    official_mapping: dict[str, dict[str, Any]] = {}
    partition_groups = {name: [] for name in ("train", "val", "test")}
    label_by_fault_location = {"inner_race": 1, "ball": 2, "outer_race": 3}
    for entry in raw_groups:
        group_id = _required_config_text(entry, "group_id")
        partition = _required_config_text(entry, "partition").lower()
        if partition not in valid_partitions:
            raise ValueError(
                f"Physical group {group_id!r} has invalid partition {partition!r}."
            )
        if group_id in group_partition:
            raise ValueError(f"Duplicate physical group_id {group_id!r}.")
        expected_label = _config_value(entry, "expected_label")
        if expected_label is None:
            raise ValueError(
                f"Physical group {group_id!r} must declare expected_label."
            )
        condition_config = _config_value(entry, "official_condition")
        condition = {
            "bearing_end": _required_config_text(
                condition_config,
                "bearing_end",
            ),
            "fault_location": _required_config_text(
                condition_config,
                "fault_location",
            ),
            "fault_diameter_mils": _config_value(
                condition_config,
                "fault_diameter_mils",
            ),
            "outer_race_position": _required_config_text(
                condition_config,
                "outer_race_position",
            ),
        }
        if condition["bearing_end"] not in {"drive_end", "fan_end"}:
            raise ValueError(
                f"Condition block {group_id!r} has invalid bearing_end."
            )
        fault_location = condition["fault_location"]
        if fault_location not in label_by_fault_location:
            raise ValueError(
                f"Condition block {group_id!r} has invalid fault_location."
            )
        try:
            condition["fault_diameter_mils"] = int(
                condition["fault_diameter_mils"]
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Condition block {group_id!r} has invalid fault_diameter_mils."
            ) from error
        if condition["fault_diameter_mils"] <= 0:
            raise ValueError(
                f"Condition block {group_id!r} has invalid fault_diameter_mils."
            )
        outer_position = condition["outer_race_position"]
        if fault_location == "outer_race":
            if outer_position not in {"3_oclock", "6_oclock", "12_oclock"}:
                raise ValueError(
                    f"Condition block {group_id!r} must declare an outer-race "
                    "clock position."
                )
        elif outer_position != "not_applicable":
            raise ValueError(
                f"Condition block {group_id!r} must use outer_race_position="
                "'not_applicable'."
            )
        if expected_label != label_by_fault_location[fault_location]:
            raise ValueError(
                f"Condition block {group_id!r} label does not match its official "
                f"fault_location={fault_location!r}."
            )
        condition_key = tuple(condition[field] for field in (
            "bearing_end",
            "fault_location",
            "fault_diameter_mils",
            "outer_race_position",
        ))
        if condition_key in condition_to_group:
            raise ValueError(
                f"Official condition {condition!r} is assigned to both "
                f"{condition_to_group[condition_key]!r} and {group_id!r}."
            )
        condition_to_group[condition_key] = group_id

        raw_official_records = _config_value(entry, "official_records")
        if not isinstance(raw_official_records, (list, tuple)):
            raise ValueError(
                f"Condition block {group_id!r} must declare official_records."
            )
        if len(raw_official_records) != len(domain_order):
            raise ValueError(
                f"Condition block {group_id!r} must declare one official record "
                f"for every domain in domain_order={domain_order}."
            )
        official_records = []
        for domain, raw_record in zip(domain_order, raw_official_records):
            if not isinstance(raw_record, str) or not raw_record.strip():
                raise ValueError(
                    f"Condition block {group_id!r} contains an invalid official record."
                )
            record = raw_record.strip()
            if not record.endswith(f"_{domain}"):
                raise ValueError(
                    f"Official record {record!r} for {group_id!r} does not match "
                    f"domain {domain!r}."
                )
            record_key = (condition["bearing_end"], record)
            if record_key in official_record_keys:
                raise ValueError(
                    f"Official record {record_key!r} is assigned more than once."
                )
            official_record_keys.add(record_key)
            official_records.append(record)
        raw_files = _config_value(entry, "files")
        if not isinstance(raw_files, (list, tuple)) or not raw_files:
            raise ValueError(
                f"Physical group {group_id!r} must declare a non-empty files list."
            )
        if len(raw_files) != len(domain_order):
            raise ValueError(
                f"Physical group {group_id!r} must declare one file for every "
                f"domain in domain_order={domain_order}."
            )
        group_partition[group_id] = partition
        group_expected_label[group_id] = expected_label
        partition_groups[partition].append(group_id)
        official_mapping[group_id] = {
            "partition": partition,
            "expected_label": expected_label,
            "condition": condition,
            "source_url": official_sources[
                "drive_end_12k"
                if condition["bearing_end"] == "drive_end"
                else "fan_end_12k"
            ],
            "records": [],
        }
        for domain, raw_file, official_record in zip(
            domain_order,
            raw_files,
            official_records,
        ):
            if not isinstance(raw_file, str) or not raw_file.strip():
                raise ValueError(
                    f"Physical group {group_id!r} contains an invalid file name."
                )
            file_name = raw_file.strip()
            if file_name in file_to_group:
                raise ValueError(
                    f"File {file_name!r} is assigned to more than one physical group."
                )
            file_to_group[file_name] = group_id
            file_expected_domain[file_name] = domain
            official_mapping[group_id]["records"].append(
                {
                    "domain": domain,
                    "official_record": official_record,
                    "file": file_name,
                }
            )

    empty_partitions = [name for name, groups in partition_groups.items() if not groups]
    if empty_partitions:
        raise ValueError(
            f"Grouped split has no physical groups for partition(s) {empty_partitions}."
        )

    metadata_files: dict[str, Any] = {}
    group_by_id: dict[Any, str] = {}
    for file_id in metadata.keys():
        entry = metadata[file_id]
        file_name = entry.get("File")
        if not isinstance(file_name, str) or not file_name.strip():
            raise ValueError(
                f"Physical group identity is ambiguous for ID {file_id!r}: File is missing."
            )
        file_name = file_name.strip()
        if file_name in metadata_files:
            raise ValueError(
                f"Physical group identity is ambiguous: File {file_name!r} maps to "
                "multiple metadata IDs."
            )
        metadata_files[file_name] = file_id
        if file_name not in file_to_group:
            raise ValueError(
                f"Physical group identity is missing for selected File {file_name!r}."
            )
        group_by_id[file_id] = file_to_group[file_name]
        group_id = group_by_id[file_id]
        observed_label = entry.get("Label")
        expected_label = group_expected_label[group_id]
        if observed_label != expected_label:
            raise ValueError(
                f"Metadata label mismatch for File {file_name!r}: observed "
                f"{observed_label!r}, expected {expected_label!r}."
            )
        observed_domain = entry.get("Domain_id")
        expected_domain = file_expected_domain[file_name]
        if observed_domain != expected_domain:
            raise ValueError(
                f"Metadata domain mismatch for File {file_name!r}: observed "
                f"{observed_domain!r}, expected {expected_domain!r}."
            )
        if expected_sample_rate is not None:
            observed_rate = entry.get("Sample_rate")
            if observed_rate != expected_sample_rate:
                raise ValueError(
                    f"Metadata sample-rate mismatch for File {file_name!r}: "
                    f"observed {observed_rate!r}, expected {expected_sample_rate!r}."
                )

    stale_files = sorted(set(file_to_group) - set(metadata_files), key=str)
    if stale_files:
        raise ValueError(
            "task.grouped_split declares file(s) absent from selected metadata: "
            f"{stale_files}."
        )

    source_ids = set(train_val_ids)
    target_ids = set(test_ids)
    ordered_ids = list(metadata.keys())
    split_ids = {
        "train": [
            file_id
            for file_id in ordered_ids
            if file_id in source_ids
            and group_partition[group_by_id[file_id]] == "train"
        ],
        "val": [
            file_id
            for file_id in ordered_ids
            if file_id in source_ids
            and group_partition[group_by_id[file_id]] == "val"
        ],
        "test": [
            file_id
            for file_id in ordered_ids
            if file_id in target_ids
            and group_partition[group_by_id[file_id]] == "test"
        ],
    }
    empty = [name for name, ids in split_ids.items() if not ids]
    if empty:
        raise ValueError(f"Grouped split produced empty partition(s) {empty}.")

    split_group_sets = {
        name: {group_by_id[file_id] for file_id in ids}
        for name, ids in split_ids.items()
    }
    for name in ("train", "val", "test"):
        expected = set(partition_groups[name])
        if split_group_sets[name] != expected:
            missing = sorted(expected - split_group_sets[name])
            raise ValueError(
                f"Grouped split partition {name!r} has no admissible row for "
                f"physical group(s) {missing}."
            )
    if (
        split_group_sets["train"] & split_group_sets["val"]
        or split_group_sets["train"] & split_group_sets["test"]
        or split_group_sets["val"] & split_group_sets["test"]
    ):
        raise ValueError("Physical groups cross train/validation/test partitions.")

    group_labels: dict[str, set[Any]] = {}
    for file_id, group_id in group_by_id.items():
        group_labels.setdefault(group_id, set()).add(metadata[file_id].get("Label"))
    inconsistent = sorted(
        group_id for group_id, labels in group_labels.items() if len(labels) != 1
    )
    if inconsistent:
        raise ValueError(
            "Each physical group must have one fault label; inconsistent group(s): "
            f"{inconsistent}."
        )

    all_labels = sorted(
        {next(iter(labels)) for labels in group_labels.values()},
        key=str,
    )
    if all_labels != sorted(admitted_labels, key=str):
        raise ValueError(
            "Observed grouped labels do not equal task.grouped_split.admitted_labels: "
            f"observed={all_labels}, admitted={sorted(admitted_labels, key=str)}."
        )
    source_domains = list(getattr(args_task, "source_domain_id", []) or [])
    target_domains = list(getattr(args_task, "target_domain_id", []) or [])
    expected_domains = {
        "train": source_domains,
        "val": source_domains,
        "test": target_domains,
    }
    minimum_support = int(_config_value(protocol, "min_groups_per_class_domain", 1))
    if minimum_support < 1:
        raise ValueError(
            "task.grouped_split.min_groups_per_class_domain must be positive."
        )

    support: dict[str, dict[str, dict[str, int]]] = {}
    for split, ids in split_ids.items():
        support[split] = {}
        for domain in expected_domains[split]:
            support[split][str(domain)] = {}
            for label in all_labels:
                groups = {
                    group_by_id[file_id]
                    for file_id in ids
                    if metadata[file_id].get("Domain_id") == domain
                    and metadata[file_id].get("Label") == label
                }
                support[split][str(domain)][str(label)] = len(groups)
                if len(groups) < minimum_support:
                    raise ValueError(
                        "Grouped split lacks independent-unit support for "
                        f"split={split}, domain={domain}, label={label}: "
                        f"{len(groups)} group(s), require {minimum_support}."
                    )

    selected_ids = set(group_by_id)
    used_ids = set().union(*(set(ids) for ids in split_ids.values()))
    non_authoritative_observations = {
        group_id: {
            field: sorted(
                {
                    metadata[file_id].get(field)
                    for file_id, observed_group in group_by_id.items()
                    if observed_group == group_id
                },
                key=str,
            )
            for field in non_authoritative_fields
        }
        for group_id in sorted(group_partition)
    }
    facts = {
        "enabled": True,
        "assignment_method": _config_value(protocol, "assignment_method"),
        "group_key": group_key,
        "group_meaning": group_meaning,
        "inferential_unit": inferential_unit,
        "verified_run_identity": verified_run_identity,
        "observation_hierarchy": observation_hierarchy,
        "identity_limit": identity_limit,
        "official_sources": official_sources,
        "official_file_condition_mapping": official_mapping,
        "non_authoritative_metadata_fields": non_authoritative_fields,
        "non_authoritative_metadata_observations": non_authoritative_observations,
        "metadata_limit": metadata_limit,
        "endpoint": endpoint,
        "admitted_labels": admitted_labels,
        "excluded_label_0_reason": excluded_label_0_reason,
        "target_label_access_boundary": target_label_access_boundary,
        "partition_groups": {
            name: list(partition_groups[name]) for name in ("train", "val", "test")
        },
        "source_domains": source_domains,
        "target_domains": target_domains,
        "source_environment_meaning": _config_value(
            protocol,
            "source_environment_meaning",
        ),
        "target_environment_meaning": _config_value(
            protocol,
            "target_environment_meaning",
        ),
        "domain_order": domain_order,
        "expected_sample_rate": expected_sample_rate,
        "selected_file_count": len(selected_ids),
        "used_file_count": len(used_ids),
        "excluded_file_ids": sorted(selected_ids - used_ids, key=str),
        "selected_group_count": len(group_partition),
        "min_groups_per_class_domain": minimum_support,
        "class_domain_group_support": support,
        "normalization_fitting_boundary": (
            "none: raw windows are not fitted or normalized from any partition"
        ),
        "windowing": {
            "window_size": int(getattr(args_data, "window_size")),
            "sampling_strategy": str(
                getattr(args_data, "window_sampling_strategy")
            ),
            "num_window_per_file": int(getattr(args_data, "num_window")),
            "sampling_seed": int(getattr(args_data, "window_sampling_seed", 0)),
            "dtype": str(getattr(args_data, "dtype", "")),
        },
    }
    return split_ids, group_by_id, facts


def _summarize_split_assignments(
    split_maps: dict[str, dict[Any, Any]],
    metadata: Any,
    physical_group_by_id: dict[Any, str] | None = None,
    grouped_protocol: dict[str, Any] | None = None,
    normalization: str | None = None,
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

    result = {
        "raw_interval_overlap": raw_overlap,
        "file_overlap": file_overlap,
        "domain_overlap": domain_overlap,
        "classes": classes,
        "test_classes_seen_in_train": test_classes_seen,
        "window_counts": {
            split: _window_count(dataset_map)
            for split, dataset_map in split_maps.items()
        },
        "normalization": {
            "method": normalization,
            "fitting_boundary": (
                "none: no data-derived normalization statistics"
                if normalization == "none"
                else "per-window or unspecified; not a train-fitted dataset statistic"
            ),
        },
    }

    if physical_group_by_id is not None:
        split_groups = {
            split: {
                physical_group_by_id[file_id]
                for file_id in file_ids
            }
            for split, file_ids in split_files.items()
        }
        group_overlap = {
            f"{left}_{right}": sorted(split_groups[left] & split_groups[right])
            for left, right in pairs
        }
        class_support: dict[str, dict[str, int]] = {}
        for split, file_ids in split_files.items():
            label_groups: dict[str, set[str]] = {}
            for file_id in file_ids:
                label = str(metadata[file_id]["Label"])
                label_groups.setdefault(label, set()).add(
                    physical_group_by_id[file_id]
                )
            class_support[split] = {
                label: len(groups) for label, groups in sorted(label_groups.items())
            }
        result["physical_groups"] = {
            "group_key": (
                None if grouped_protocol is None else grouped_protocol["group_key"]
            ),
            "group_meaning": (
                None if grouped_protocol is None else grouped_protocol["group_meaning"]
            ),
            "memberships": {
                split: sorted(groups) for split, groups in split_groups.items()
            },
            "counts": {
                split: len(groups) for split, groups in split_groups.items()
            },
            "overlap": group_overlap,
            "class_support": class_support,
        }
        result["grouped_protocol"] = grouped_protocol

    return result


def _format_split_summary(summary: dict[str, Any]) -> str:
    raw = summary["raw_interval_overlap"]
    files = summary["file_overlap"]
    base = (
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
    groups = summary.get("physical_groups")
    if groups is None:
        return base
    overlap = groups["overlap"]
    return (
        f"{base}; physical-groups train={groups['counts']['train']}, "
        f"val={groups['counts']['val']}, test={groups['counts']['test']}; "
        "group-overlap "
        f"train/val={overlap['train_val']}, "
        f"train/test={overlap['train_test']}, "
        f"val/test={overlap['val_test']}"
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
        split_ids, physical_group_by_id, grouped_protocol = _grouped_partition_ids(
            train_val_ids,
            test_ids,
            self.target_metadata,
            self.args_task,
            self.args_data,
        )
        train_ids = split_ids["train"]
        val_ids = split_ids["val"]
        test_ids = split_ids["test"]
        self.physical_group_by_id = physical_group_by_id
        self.grouped_protocol = grouped_protocol

        print(
            "Initializing datasets with explicit adapter "
            f"{dataset_cls.__module__}.{dataset_cls.__name__} "
            f"for {task_type}/{task_name}."
        )
        for file_id in tqdm(train_ids, desc="Creating train datasets"):
            file_data = {file_id: self.data[file_id]}
            train_dataset[file_id] = dataset_cls(
                file_data,
                self.target_metadata,
                self.args_data,
                self.args_task,
                "train",
            )
        for file_id in tqdm(val_ids, desc="Creating validation datasets"):
            file_data = {file_id: self.data[file_id]}
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
            physical_group_by_id,
            grouped_protocol,
            str(getattr(self.args_data, "normalization", "")),
        )
        print(f"[DATA SPLIT] {_format_split_summary(self.split_summary)}")

        return (
            IdIncludedDataset(
                train_dataset,
                self.target_metadata,
                physical_group_by_id,
            ),
            IdIncludedDataset(
                val_dataset,
                self.target_metadata,
                physical_group_by_id,
            ),
            IdIncludedDataset(
                test_dataset,
                self.target_metadata,
                physical_group_by_id,
            ),
        )


__all__ = ["ExplicitDataFactory"]
