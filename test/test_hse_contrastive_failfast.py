from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.task_factory.Components.contrastive_strategies import create_contrastive_strategy
from src.task_factory.task.pretrain.hse_contrastive import HseContrastiveTask


def _minimal_hse_task(pairing: str = "simclr_2view"):
    task = HseContrastiveTask.__new__(HseContrastiveTask)
    object.__setattr__(
        task,
        "strategy_manager",
        create_contrastive_strategy({"type": "single", "loss_type": "INFONCE", "temperature": 0.07}),
    )
    object.__setattr__(task, "contrastive_pairing", pairing)
    object.__setattr__(
        task,
        "args_task",
        SimpleNamespace(
            augmentation_type="noise",
            augmentation_noise_std=0.01,
            augmentation_dropout_p=0.0,
            augmentation_scale_std=0.0,
        ),
    )
    object.__setattr__(task, "ce_loss_fn", F.cross_entropy)
    return task


def test_hse_contrastive_flow_has_nonzero_signal() -> None:
    torch.manual_seed(0)
    hse_task = _minimal_hse_task()
    features = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 0, 1])

    loss = hse_task._run_contrastive_flow(features, labels)

    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_hse_classification_invalid_labels_raise() -> None:
    hse_task = _minimal_hse_task()
    logits = torch.randn(2, 2)
    labels = torch.tensor([0, 3])

    with pytest.raises(ValueError, match="out of range"):
        hse_task._run_classification_flow(logits, labels)


def test_hse_label_pairing_rejects_negative_labels() -> None:
    hse_task = _minimal_hse_task(pairing="labels")
    features = torch.randn(2, 8)
    labels = torch.tensor([0, -1])

    with pytest.raises(ValueError, match="Negative labels"):
        hse_task._run_contrastive_flow(features, labels)
