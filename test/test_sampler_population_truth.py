from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch.utils.data import Dataset

from src.data_factory.dataset_task.Dataset_cluster import IdIncludedDataset
from src.data_factory.samplers.Get_sampler import Get_sampler


class DictDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        return {"x": index, "y": index % 2}


def _args(batch_size: int = 2):
    return (
        SimpleNamespace(type="DG"),
        SimpleNamespace(batch_size=batch_size),
    )


def test_id_included_dataset_rejects_empty_population() -> None:
    with pytest.raises(ValueError, match="non-empty mapping"):
        IdIncludedDataset({}, metadata={})


def test_id_included_dataset_rejects_missing_dataset_object() -> None:
    with pytest.raises(ValueError, match="has no dataset object"):
        IdIncludedDataset({1: None}, metadata={1: {"Dataset_id": 10}})


def test_id_included_dataset_rejects_zero_sample_file() -> None:
    with pytest.raises(ValueError, match="produced zero samples"):
        IdIncludedDataset(
            {1: DictDataset(0)},
            metadata={1: {"Dataset_id": 10}},
        )


def test_sampler_rejects_missing_metadata_row() -> None:
    dataset = IdIncludedDataset({1: DictDataset(2)}, metadata={})
    args_task, args_data = _args()

    with pytest.raises(ValueError, match="missing selected file ID"):
        Get_sampler(args_task, args_data, dataset, mode="train")


def test_sampler_rejects_missing_dataset_id() -> None:
    dataset = IdIncludedDataset({1: DictDataset(2)}, metadata={1: {}})
    args_task, args_data = _args()

    with pytest.raises(ValueError, match="missing Dataset_id"):
        Get_sampler(args_task, args_data, dataset, mode="train")


def test_training_sampler_keeps_final_short_batch() -> None:
    dataset = IdIncludedDataset(
        {1: DictDataset(3)},
        metadata={1: {"Dataset_id": 10}},
    )
    args_task, args_data = _args(batch_size=2)

    batches = list(Get_sampler(args_task, args_data, dataset, mode="train"))

    assert sorted(len(batch) for batch in batches) == [1, 2]
    assert sorted(index for batch in batches for index in batch) == [0, 1, 2]


def test_sampler_represents_every_selected_system() -> None:
    dataset = IdIncludedDataset(
        {
            1: DictDataset(1),
            2: DictDataset(2),
        },
        metadata={
            1: {"Dataset_id": 10},
            2: {"Dataset_id": 20},
        },
    )
    args_task, args_data = _args(batch_size=4)

    batches = list(Get_sampler(args_task, args_data, dataset, mode="train"))
    represented_file_ids = {
        dataset.get_file_id(index)
        for batch in batches
        for index in batch
    }

    assert represented_file_ids == {1, 2}
    assert sum(len(batch) for batch in batches) == len(dataset)
