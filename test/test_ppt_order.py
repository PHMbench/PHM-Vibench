from types import SimpleNamespace

import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_factory.data_utils import MetadataAccessor
from src.model_factory.ISFM.M_01_ISFM import Model as ISFMModel
from src.task_factory.Components.ppt_time_order import PPTOrderConfig, PPTOrderLoss
from src.task_factory.task.pretrain.ppt_order import PptOrderTask


def _objective(**overrides):
    values = {
        "num_patches": 8,
        "num_channels": 4,
        "embedding_dim": 6,
        "order_axes": ("time", "channel"),
        "weak_swaps": 1,
        "strong_swaps": 4,
        "channel_weak_swaps": 1,
        "channel_strong_swaps": 2,
        "bank_size": 32,
        "seed": 3,
        "temperature": 0.1,
        "weighting": "fixed",
    }
    values.update(overrides)
    return PPTOrderLoss(PPTOrderConfig(**values))


def test_multiaxis_ppt_loss_is_finite_and_backpropagates():
    objective = _objective()
    grid = torch.randn(3, 4, 8, 6, requires_grad=True)

    loss, stats = objective(grid, offset=7)

    assert torch.isfinite(loss)
    assert {
        "time_consistency_loss",
        "time_contrastive_loss",
        "channel_consistency_loss",
        "channel_contrastive_loss",
    }.issubset(stats)
    loss.backward()
    assert grid.grad is not None
    assert torch.isfinite(grid.grad).all()


def test_multiaxis_ppt_permutations_are_seeded_and_uncertainty_is_learnable():
    first = _objective(weighting="uncertainty")
    second = _objective(weighting="uncertainty")
    assert torch.equal(first.weak_banks["time"].indices, second.weak_banks["time"].indices)
    assert torch.equal(
        first.strong_banks["channel"].indices,
        second.strong_banks["channel"].indices,
    )

    loss, _ = first(torch.randn(2, 4, 8, 6))
    loss.backward()
    assert all(parameter.grad is not None for parameter in first.log_variances.values())


def test_channel_order_requires_three_channels():
    with pytest.raises(ValueError, match="at least three channels"):
        _objective(num_channels=2)


class _GridNetwork(nn.Module):
    def __init__(self, channels=4, patches=8, embedding_dim=6, classes=3):
        super().__init__()
        self.channels = channels
        self.patches = patches
        self.projection = nn.Linear(1, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, classes)

    def encode_patch_grid(self, x, file_id=None):
        del file_id
        channel_first = x.transpose(1, 2)
        patches = channel_first.reshape(
            x.shape[0],
            self.channels,
            self.patches,
            -1,
        ).mean(dim=-1, keepdim=True)
        return self.projection(patches)

    def classify_encoded(self, sequence, file_id=None):
        del file_id
        return self.classifier(sequence.mean(dim=1))


def _task_args(mode="ssl", weighting="fixed"):
    args_data = SimpleNamespace(window_size=64)
    args_model = SimpleNamespace(
        embedding="E_03_Patch",
        backbone="B_08_PatchTST",
        channel_independent=True,
        input_dim=4,
        window_size=64,
        patch_size_L=8,
        num_patches=8,
        output_dim=6,
    )
    args_task = SimpleNamespace(
        optimizer="adamw",
        lr=1e-3,
        weight_decay=0.0,
        ppt=SimpleNamespace(
            mode=mode,
            order_axes=["time", "channel"],
            weighting=weighting,
            weak_swaps=1,
            strong_swaps=4,
            channel_weak_swaps=1,
            channel_strong_swaps=2,
            permutation_bank_size=16,
            permutation_seed=2,
            temperature=0.1,
            consistency_weight=1.0,
            contrastive_weight=1.0,
            classification_weight=1.0,
        ),
    )
    return args_data, args_model, args_task


@pytest.mark.parametrize("mode", ["ssl", "supervised"])
def test_ppt_order_task_supports_ssl_and_supervised(mode, monkeypatch):
    args_data, args_model, args_task = _task_args(mode=mode, weighting="uncertainty")
    task = PptOrderTask(
        _GridNetwork(),
        args_data,
        args_model,
        args_task,
        SimpleNamespace(),
        SimpleNamespace(),
        None,
    )
    monkeypatch.setattr(task, "log_dict", lambda *args, **kwargs: None)
    batch = {
        "x": torch.randn(3, 64, 4),
        "file_id": torch.ones(3, dtype=torch.long),
        "y": torch.tensor([0, 1, 2]),
    }

    loss = task.training_step(batch, batch_idx=0)
    loss.backward()

    assert torch.isfinite(loss)
    assert task.network.projection.weight.grad is not None
    if mode == "supervised":
        assert task.network.classifier.weight.grad is not None
        assert task.classification_log_variance.grad is not None


def test_real_isfm_channel_grid_assembly_backpropagates(monkeypatch):
    metadata = MetadataAccessor(
        pd.DataFrame(
            [
                {"Id": 1, "Dataset_id": 0, "Label": 0, "Sample_rate": 1000},
                {"Id": 2, "Dataset_id": 0, "Label": 1, "Sample_rate": 1000},
            ]
        )
    )
    args_data, args_model, args_task = _task_args()
    args_model.task_head = "H_01_Linear_cla"
    args_model.factor = 5
    args_model.dropout = 0.0
    args_model.num_heads = 2
    args_model.d_ff = 12
    args_model.e_layers = 1
    args_model.activation = "gelu"
    args_model.d_model = 8
    network = ISFMModel(args_model, metadata)
    task = PptOrderTask(
        network,
        args_data,
        args_model,
        args_task,
        SimpleNamespace(),
        SimpleNamespace(),
        metadata,
    )
    monkeypatch.setattr(task, "log_dict", lambda *args, **kwargs: None)

    loss = task.training_step(
        {"x": torch.randn(3, 64, 4), "file_id": torch.ones(3, dtype=torch.long)},
        batch_idx=0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert network.embedding.proj[0].weight.grad is not None
    assert next(network.backbone.parameters()).grad is not None


def test_ppt_order_one_epoch_lightning_smoke():
    args_data, args_model, args_task = _task_args()
    task = PptOrderTask(
        _GridNetwork(),
        args_data,
        args_model,
        args_task,
        SimpleNamespace(),
        SimpleNamespace(),
        None,
    )
    samples = [
        {"x": torch.randn(64, 4), "file_id": torch.tensor(1)}
        for _ in range(4)
    ]
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    trainer.fit(task, train_dataloaders=DataLoader(samples, batch_size=4))

    assert trainer.global_step == 1
