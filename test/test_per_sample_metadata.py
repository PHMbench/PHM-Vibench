from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.model_factory.ISFM.system_utils import normalize_fs, resolve_batch_metadata
from src.model_factory.ISFM.task_head.H_01_Linear_cla import H_01_Linear_cla
from src.task_factory.Default_task import Default_task


_METADATA = {
    101: {
        "Name": "Dummy",
        "Dataset_id": 1,
        "Domain_id": 0,
        "Sample_Rate": 12_000,
    },
    102: {
        "Name": "Dummy",
        "Dataset_id": 1,
        "Domain_id": 1,
        "Sample_Rate": 48_000,
    },
}


class _TaskHarness:
    metadata = _METADATA

    def __init__(self):
        self.network = SimpleNamespace()
        self.seen_file_ids = None
        self.seen_sample_rates = None

    def forward(self, batch):
        self.seen_file_ids = batch["file_id"].clone()
        _, sample_rates = resolve_batch_metadata(
            self.metadata,
            batch["file_id"],
            device=batch["x"].device,
        )
        self.seen_sample_rates = sample_rates
        return torch.tensor(
            [[2.0, -1.0], [-1.0, 2.0]],
            dtype=torch.float32,
            requires_grad=True,
        )

    @staticmethod
    def _compute_loss(y_hat, y):
        return F.cross_entropy(y_hat, y)

    @staticmethod
    def _compute_metrics(y_hat, y, data_name, stage):
        return {}

    @staticmethod
    def _compute_regularization():
        return {"total": torch.tensor(0.0)}


def test_default_task_preserves_per_sample_file_ids_and_sampling_rates():
    task = _TaskHarness()
    batch = {
        "x": torch.zeros(2, 8, 1),
        "y": torch.tensor([0, 1]),
        "file_id": torch.tensor([101, 102]),
    }

    metrics = Default_task._shared_step(task, batch, "train")

    assert torch.equal(batch["file_id"], torch.tensor([101, 102]))
    assert torch.equal(task.seen_file_ids, torch.tensor([101, 102]))
    assert torch.equal(task.seen_sample_rates, torch.tensor([12_000.0, 48_000.0]))
    assert metrics["train_total_loss"].requires_grad


def test_default_task_rejects_partial_file_id_vectors():
    task = _TaskHarness()
    batch = {
        "x": torch.zeros(3, 8, 1),
        "y": torch.tensor([0, 1, 0]),
        "file_id": torch.tensor([101, 102]),
    }

    with pytest.raises(ValueError, match="one ID or one ID per sample"):
        Default_task._shared_step(task, batch, "train")


def test_normalize_fs_preserves_vectors_and_rejects_wrong_lengths():
    fs = normalize_fs(
        torch.tensor([12_000.0, 48_000.0]),
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert torch.equal(fs, torch.tensor([12_000.0, 48_000.0]))

    with pytest.raises(ValueError, match="received 2 values for batch_size=3"):
        normalize_fs(
            torch.tensor([12_000.0, 48_000.0]),
            batch_size=3,
            device=torch.device("cpu"),
        )


def test_linear_head_requires_one_system_per_batch():
    head = H_01_Linear_cla(
        SimpleNamespace(
            num_classes={1: 2, 2: 2},
            output_dim=4,
        )
    )
    features = torch.randn(2, 4)

    logits = head(features, system_id=torch.tensor([1, 1]))
    assert logits.shape == (2, 2)

    with pytest.raises(ValueError, match="single Dataset_id per batch"):
        head(features, system_id=torch.tensor([1, 2]))
