from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data_factory.ID.Id_searcher import search_target_dataset_metadata
from src.data_factory.data_utils import MetadataAccessor
from src.data_factory.dataset_task.Dataset_cluster import IdIncludedDataset
from src.data_factory.dataset_task.Default_dataset import Default_dataset
from src.data_factory.explicit_data_factory import (
    _grouped_partition_ids,
    _summarize_split_assignments,
)


def _synthetic_protocol():
    rows = []
    groups = []
    next_id = 1
    partition_groups = {"train": [], "val": [], "test": []}

    for partition_index, partition in enumerate(("train", "val", "test")):
        for label in (1, 2, 3):
            for replicate in range(2):
                group_id = f"{partition}-label{label}-group{replicate}"
                files = []
                partition_groups[partition].append(group_id)
                for domain in (0, 1, 2, 3):
                    file_name = f"{group_id}-domain{domain}.mat"
                    files.append(file_name)
                    rows.append(
                        {
                            "Id": next_id,
                            "Dataset_id": 1,
                            "File": file_name,
                            "Label": label,
                            "Domain_id": domain,
                            "Sample_rate": 12000,
                        }
                    )
                    next_id += 1
                groups.append(
                    SimpleNamespace(
                        group_id=group_id,
                        partition=partition,
                        expected_label=label,
                        official_condition=SimpleNamespace(
                            bearing_end="drive_end",
                            fault_location={
                                1: "inner_race",
                                2: "ball",
                                3: "outer_race",
                            }[label],
                            fault_diameter_mils=(partition_index * 2 + replicate + 1),
                            outer_race_position=(
                                "6_oclock" if label == 3 else "not_applicable"
                            ),
                        ),
                        official_records=[
                            f"{group_id}_{domain}" for domain in (0, 1, 2, 3)
                        ],
                        files=files,
                    )
                )

    metadata = MetadataAccessor(pd.DataFrame(rows), key_column="Id")
    protocol = SimpleNamespace(
        enabled=True,
        group_key="official_fault_condition",
        group_meaning="one seeded fault-bearing condition across motor loads",
        inferential_unit="documented_fault_condition_block",
        verified_run_identity="metadata.File",
        observation_hierarchy="window < File/run < condition block",
        identity_limit="synthetic fixture has no specimen serial identity",
        target_label_access_boundary="protocol validation only",
        endpoint="fault_location_labels_1_2_3_only",
        excluded_label_0_reason="one group cannot support three partitions",
        official_sources=SimpleNamespace(
            overview="https://example.test/overview",
            apparatus="https://example.test/apparatus",
            drive_end_12k="https://example.test/drive-end",
            fan_end_12k="https://example.test/fan-end",
        ),
        admitted_labels=[1, 2, 3],
        non_authoritative_metadata_fields=["Fault_level"],
        metadata_limit="Fault_level is not used by this synthetic protocol.",
        domain_order=[0, 1, 2, 3],
        expected_sample_rate=12000,
        min_groups_per_class_domain=2,
        groups=groups,
    )
    args_task = SimpleNamespace(
        type="DG",
        source_domain_id=[0, 1],
        target_domain_id=[2, 3],
        grouped_split=protocol,
    )
    args_data = SimpleNamespace(
        normalization="none",
        window_size=4096,
        window_sampling_strategy="evenly_spaced",
        num_window=64,
        window_sampling_seed=31,
        dtype="float32",
    )
    source_ids = [
        file_id
        for file_id in metadata.keys()
        if metadata[file_id]["Domain_id"] in {0, 1}
    ]
    target_ids = [
        file_id
        for file_id in metadata.keys()
        if metadata[file_id]["Domain_id"] in {2, 3}
    ]
    return metadata, args_task, args_data, source_ids, target_ids, partition_groups


def test_explicit_group_membership_is_disjoint_and_reproducible() -> None:
    metadata, args_task, args_data, source_ids, target_ids, expected = (
        _synthetic_protocol()
    )

    first = _grouped_partition_ids(
        source_ids, target_ids, metadata, args_task, args_data
    )
    second = _grouped_partition_ids(
        source_ids, target_ids, metadata, args_task, args_data
    )

    assert first == second
    split_ids, group_by_id, facts = first
    assert {name: len(ids) for name, ids in split_ids.items()} == {
        "train": 12,
        "val": 12,
        "test": 12,
    }
    assert facts is not None
    assert facts["partition_groups"] == expected
    assert facts["selected_group_count"] == 18
    assert facts["selected_file_count"] == 72
    assert facts["used_file_count"] == 36
    assert len(facts["excluded_file_ids"]) == 36
    assert facts["admitted_labels"] == [1, 2, 3]
    assert facts["inferential_unit"] == "documented_fault_condition_block"
    assert len(facts["official_file_condition_mapping"]) == 18
    assert group_by_id is not None
    split_groups = {
        split: {group_by_id[file_id] for file_id in ids}
        for split, ids in split_ids.items()
    }
    assert split_groups["train"].isdisjoint(split_groups["val"])
    assert split_groups["train"].isdisjoint(split_groups["test"])
    assert split_groups["val"].isdisjoint(split_groups["test"])
    for split in ("train", "val", "test"):
        for domain_support in facts["class_domain_group_support"][split].values():
            assert set(domain_support.values()) == {2}


def test_grouped_protocol_rejects_missing_identity_and_metadata_drift() -> None:
    metadata, args_task, args_data, source_ids, target_ids, _ = (
        _synthetic_protocol()
    )
    args_task.grouped_split.groups[0].files[0] = "absent.mat"
    with pytest.raises(ValueError, match="identity is missing"):
        _grouped_partition_ids(
            source_ids, target_ids, metadata, args_task, args_data
        )


def test_grouped_protocol_rejects_official_condition_mapping_drift() -> None:
    metadata, args_task, args_data, source_ids, target_ids, _ = (
        _synthetic_protocol()
    )
    args_task.grouped_split.groups[0].official_records[0] = "wrong-domain-record"

    with pytest.raises(ValueError, match="does not match domain"):
        _grouped_partition_ids(
            source_ids, target_ids, metadata, args_task, args_data
        )

    metadata, args_task, args_data, source_ids, target_ids, _ = (
        _synthetic_protocol()
    )
    first_id = metadata.keys()[0]
    metadata.df.loc[first_id, "Label"] = 99
    with pytest.raises(ValueError, match="Metadata label mismatch"):
        _grouped_partition_ids(
            source_ids, target_ids, metadata, args_task, args_data
        )


def test_grouped_protocol_rejects_test_adaptive_normalization() -> None:
    metadata, args_task, args_data, source_ids, target_ids, _ = (
        _synthetic_protocol()
    )
    args_data.normalization = "per_window_standardization"

    with pytest.raises(ValueError, match="normalization='none'"):
        _grouped_partition_ids(
            source_ids, target_ids, metadata, args_task, args_data
        )


def test_selected_files_is_explicit_before_supervised_label_validation() -> None:
    metadata = MetadataAccessor(
        pd.DataFrame(
            [
                {
                    "Id": 1,
                    "Dataset_id": 1,
                    "File": "admitted.mat",
                    "Label": 1,
                    "Domain_id": 0,
                },
                {
                    "Id": 2,
                    "Dataset_id": 1,
                    "File": "invalid.mat",
                    "Label": np.nan,
                    "Domain_id": np.nan,
                },
            ]
        ),
        key_column="Id",
    )
    args = SimpleNamespace(
        type="DG",
        target_system_id=[1],
        selected_files=["admitted.mat"],
    )

    selected = search_target_dataset_metadata(metadata, args)

    assert selected.keys() == [1]


def test_ambiguous_selected_file_identity_fails_closed() -> None:
    metadata = MetadataAccessor(
        pd.DataFrame(
            [
                {
                    "Id": 1,
                    "Dataset_id": 1,
                    "File": "duplicate.mat",
                    "Label": 1,
                    "Domain_id": 0,
                },
                {
                    "Id": 2,
                    "Dataset_id": 1,
                    "File": "duplicate.mat",
                    "Label": 1,
                    "Domain_id": 1,
                },
            ]
        ),
        key_column="Id",
    )
    args = SimpleNamespace(
        type="DG",
        target_system_id=[1],
        selected_files=["duplicate.mat"],
    )

    with pytest.raises(ValueError, match="ambiguous file"):
        search_target_dataset_metadata(metadata, args)


def _dataset_args() -> SimpleNamespace:
    return SimpleNamespace(
        window_size=4,
        num_window=4,
        window_sampling_strategy="evenly_spaced",
        stride=None,
        window_sampling_seed=0,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        unused_ratio=0.0,
        normalization="none",
        train_noise_snr=None,
        evaluation_noise_snr=None,
        evaluation_noise_seed=0,
        dtype="float32",
    )


def test_grouped_partition_uses_complete_window_set_and_exposes_group_id() -> None:
    signal = np.arange(32, dtype=np.float32).reshape(16, 2)
    metadata = {1: {"Label": 1}}
    grouped_task = SimpleNamespace(
        type="DG",
        grouped_split=SimpleNamespace(enabled=True),
    )
    legacy_task = SimpleNamespace(type="DG")

    grouped = Default_dataset(
        {1: signal}, metadata, _dataset_args(), grouped_task, mode="train"
    )
    legacy = Default_dataset(
        {1: signal}, metadata, _dataset_args(), legacy_task, mode="train"
    )
    wrapped = IdIncludedDataset(
        {1: grouped}, metadata, physical_group_by_id={1: "bearing-a"}
    )

    assert len(grouped) == 4
    assert len(legacy) == 2
    assert wrapped[0]["file_id"] == 1
    assert wrapped[0]["physical_group_id"] == "bearing-a"


def test_machine_readable_summary_contains_group_and_fitting_boundaries() -> None:
    metadata, args_task, args_data, source_ids, target_ids, _ = (
        _synthetic_protocol()
    )
    split_ids, group_by_id, facts = _grouped_partition_ids(
        source_ids, target_ids, metadata, args_task, args_data
    )

    class Stub:
        def __init__(self) -> None:
            self.window_intervals = [(0, 4), (4, 8)]

        def __len__(self) -> int:
            return 2

    summary = _summarize_split_assignments(
        {
            split: {file_id: Stub() for file_id in ids}
            for split, ids in split_ids.items()
        },
        metadata,
        group_by_id,
        facts,
        normalization="none",
    )

    assert summary["physical_groups"]["counts"] == {
        "train": 6,
        "val": 6,
        "test": 6,
    }
    assert summary["physical_groups"]["overlap"] == {
        "train_val": [],
        "train_test": [],
        "val_test": [],
    }
    assert summary["window_counts"] == {"train": 24, "val": 24, "test": 24}
    assert summary["normalization"]["fitting_boundary"].startswith("none:")
    assert summary["grouped_protocol"]["admitted_labels"] == [1, 2, 3]
    assert summary["grouped_protocol"]["target_label_access_boundary"]
