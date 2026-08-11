from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.data_factory.ID.Get_id import Get_CDDG_ids, Get_DG_ids
from src.data_factory.ID.Id_searcher import (
    search_ids_for_task,
    search_target_dataset_metadata,
)
from src.data_factory.data_utils import MetadataAccessor


def _metadata(rows) -> MetadataAccessor:
    return MetadataAccessor(pd.DataFrame(rows).copy(), key_column="Id")


def _single_system_rows():
    return [
        {"Id": 1, "Dataset_id": 10, "Domain_id": 0, "Label": 0},
        {"Id": 2, "Dataset_id": 10, "Domain_id": 1, "Label": 1},
        {"Id": 3, "Dataset_id": 10, "Domain_id": 2, "Label": 0},
    ]


def test_unknown_task_type_does_not_default_to_all_ids() -> None:
    metadata = _metadata(_single_system_rows())
    args = SimpleNamespace(type="unknown", target_system_id=[10])

    with pytest.raises(ValueError, match="Unknown task.type"):
        search_ids_for_task(metadata, args)


def test_supervised_task_requires_explicit_target_system() -> None:
    metadata = _metadata(_single_system_rows())
    args = SimpleNamespace(type="DG", target_system_id=None)

    with pytest.raises(ValueError, match="requires task.target_system_id"):
        search_target_dataset_metadata(metadata, args)


def test_pretrain_may_explicitly_use_complete_metadata_population() -> None:
    metadata = _metadata(_single_system_rows())
    args = SimpleNamespace(type="pretrain", target_system_id=None)

    assert search_target_dataset_metadata(metadata, args) is metadata


def test_target_system_must_match_metadata_rows() -> None:
    metadata = _metadata(_single_system_rows())
    args = SimpleNamespace(type="DG", target_system_id=[999])

    with pytest.raises(ValueError, match="matched no metadata rows"):
        search_target_dataset_metadata(metadata, args)


@pytest.mark.parametrize("invalid_label", [None, -1])
def test_supervised_metadata_rejects_invalid_labels(invalid_label) -> None:
    rows = _single_system_rows()
    rows[1]["Label"] = invalid_label
    metadata = _metadata(rows)
    args = SimpleNamespace(type="DG", target_system_id=[10])

    with pytest.raises(ValueError, match="Fix the metadata instead of dropping rows"):
        search_target_dataset_metadata(metadata, args)


def test_valid_dynamic_dg_split_keeps_one_source_domain() -> None:
    metadata = _metadata(_single_system_rows())
    args = SimpleNamespace(
        type="DG",
        target_system_id=[10],
        target_domain_num=1,
    )

    train_ids, test_ids = Get_DG_ids(metadata, args)

    assert train_ids == [1, 2]
    assert test_ids == [3]


def test_impossible_dynamic_dg_request_fails_instead_of_train_only() -> None:
    metadata = _metadata(
        [{"Id": 1, "Dataset_id": 10, "Domain_id": 0, "Label": 0}]
    )
    args = SimpleNamespace(
        type="DG",
        target_system_id=[10],
        target_domain_num=1,
    )

    with pytest.raises(ValueError, match="cannot be satisfied"):
        Get_DG_ids(metadata, args)


def test_explicit_dg_domains_must_be_disjoint_and_available() -> None:
    metadata = _metadata(_single_system_rows())

    overlapping = SimpleNamespace(
        type="DG",
        target_system_id=[10],
        target_domain_num=0,
        source_domain_id=[0, 1],
        target_domain_id=[1, 2],
    )
    with pytest.raises(ValueError, match="must be disjoint"):
        Get_DG_ids(metadata, overlapping)

    missing = SimpleNamespace(
        type="DG",
        target_system_id=[10],
        target_domain_num=0,
        source_domain_id=[0, 1],
        target_domain_id=[99],
    )
    with pytest.raises(ValueError, match="are absent"):
        Get_DG_ids(metadata, missing)


def test_valid_cddg_reserves_each_systems_last_domain() -> None:
    rows = []
    next_id = 1
    for dataset_id in (10, 20):
        for domain_id in (0, 1, 2):
            rows.append(
                {
                    "Id": next_id,
                    "Dataset_id": dataset_id,
                    "Domain_id": domain_id,
                    "Label": domain_id % 2,
                }
            )
            next_id += 1
    metadata = _metadata(rows)
    args = SimpleNamespace(
        type="CDDG",
        target_system_id=[10, 20],
        target_domain_num=1,
    )

    train_ids, test_ids = Get_CDDG_ids(metadata, args)

    assert train_ids == [1, 2, 4, 5]
    assert test_ids == [3, 6]


def test_cddg_fails_when_any_system_cannot_keep_a_training_domain() -> None:
    metadata = _metadata(
        [
            {"Id": 1, "Dataset_id": 10, "Domain_id": 0, "Label": 0},
            {"Id": 2, "Dataset_id": 20, "Domain_id": 0, "Label": 0},
            {"Id": 3, "Dataset_id": 20, "Domain_id": 1, "Label": 1},
        ]
    )
    args = SimpleNamespace(
        type="CDDG",
        target_system_id=[10, 20],
        target_domain_num=1,
    )

    with pytest.raises(ValueError, match="at least one training domain must remain"):
        Get_CDDG_ids(metadata, args)
