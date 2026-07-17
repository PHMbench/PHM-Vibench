from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset

from src.data_factory.dataset_task.Dataset_cluster import IdIncludedDataset
from src.data_factory.samplers.Get_sampler import Get_sampler


@pytest.mark.parametrize(
    "task_type",
    ["DG", "CDDG", "pretrain", "FS", "GFS", "multi_task", "In_distribution"],
)
@pytest.mark.parametrize("mode", ["val", "test"])
def test_evaluation_sampler_keeps_incomplete_batches(task_type, mode):
    dataset = IdIncludedDataset(
        {1: TensorDataset(torch.arange(2))},
        metadata={1: {"Dataset_id": 0}},
    )
    sampler = Get_sampler(
        SimpleNamespace(type=task_type),
        SimpleNamespace(batch_size=4),
        dataset,
        mode=mode,
    )

    assert sampler.drop_last is False
    assert len(sampler) == 1
    assert list(sampler) == [[0, 1]]
