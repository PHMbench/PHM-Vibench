from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset

from src.data_factory.dataset_task.Dataset_cluster import IdIncludedDataset
from src.data_factory.samplers.Get_sampler import Get_sampler


EVAL_TASK_TYPES = (
    "DG",
    "CDDG",
    "pretrain",
    "FS",
    "GFS",
    "generative",
    "multi_task",
    "In_distribution",
)
SAME_SYSTEM_TRAIN_TASK_TYPES = tuple(
    task_type for task_type in EVAL_TASK_TYPES if task_type != "GFS"
)


def _two_sample_dataset() -> IdIncludedDataset:
    return IdIncludedDataset(
        {1: TensorDataset(torch.arange(2))},
        metadata={1: {"Dataset_id": 0}},
    )


@pytest.mark.parametrize("task_type", EVAL_TASK_TYPES)
@pytest.mark.parametrize("mode", ("val", "test"))
def test_evaluation_sampler_keeps_incomplete_batches(task_type, mode):
    sampler = Get_sampler(
        SimpleNamespace(type=task_type),
        SimpleNamespace(batch_size=4),
        _two_sample_dataset(),
        mode=mode,
    )

    assert sampler.drop_last is False
    assert len(sampler) == 1
    assert list(sampler) == [[0, 1]]


@pytest.mark.parametrize("task_type", SAME_SYSTEM_TRAIN_TASK_TYPES)
def test_training_sampler_still_drops_incomplete_batches(task_type):
    sampler = Get_sampler(
        SimpleNamespace(type=task_type),
        SimpleNamespace(batch_size=4),
        _two_sample_dataset(),
        mode="train",
    )

    assert sampler.drop_last is True
    assert len(sampler) == 0
    assert list(sampler) == []
