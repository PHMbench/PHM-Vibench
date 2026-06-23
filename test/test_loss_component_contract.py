from __future__ import annotations

import pytest
import torch

from src.task_factory.Components.contrastive_losses import (
    BarlowTwinsLoss,
    InfoNCELoss,
    PrototypicalLoss,
    SupConLoss,
    TripletLoss,
    VICRegLoss,
)
from src.task_factory.Components.loss import get_loss_fn
from src.task_factory.Components.metrics import get_metrics


def test_documented_supervised_and_contrastive_loss_keys_resolve() -> None:
    for key in [
        "CE",
        "MSE",
        "BCE",
        "INFONCE",
        "SUPCON",
        "TRIPLET",
        "PROTOTYPICAL",
        "BARLOWTWINS",
        "VICREG",
    ]:
        assert get_loss_fn(key) is not None


def test_unknown_loss_key_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="不支持的损失函数类型"):
        get_loss_fn("NOT_A_LOSS")


def test_documented_metric_keys_resolve() -> None:
    metadata = {
        1: {"Name": "demo", "Label": 0},
        2: {"Name": "demo", "Label": 1},
        3: {"Name": "demo", "Label": 2},
    }

    metrics = get_metrics(["acc", "f1", "precision", "recall", "mse", "mae", "r2"], metadata)

    for key in ["acc", "f1", "precision", "recall", "mse", "mae", "r2"]:
        assert f"train_{key}" in metrics["demo"]


def test_infonce_rejects_impossible_pairings() -> None:
    with pytest.raises(ValueError, match="two-view batch"):
        InfoNCELoss()(torch.randn(3, 8))

    with pytest.raises(ValueError, match="no positive pairs"):
        InfoNCELoss()(torch.randn(4, 8), torch.arange(4))


def test_two_view_losses_reject_singleton_or_mismatched_views() -> None:
    with pytest.raises(ValueError, match="at least two paired samples"):
        BarlowTwinsLoss()(torch.randn(1, 8), torch.randn(1, 8))

    with pytest.raises(ValueError, match="same shape"):
        BarlowTwinsLoss()(torch.randn(2, 8), torch.randn(2, 4))

    with pytest.raises(ValueError, match="at least two paired samples"):
        VICRegLoss()(torch.randn(1, 8), torch.randn(1, 8))

    with pytest.raises(ValueError, match="same shape"):
        VICRegLoss()(torch.randn(2, 8), torch.randn(2, 4))


def test_metric_contrastive_losses_reject_impossible_batches() -> None:
    features = torch.randn(4, 8)

    with pytest.raises(ValueError, match="positive pair and one negative pair"):
        TripletLoss()(features, torch.arange(4))

    with pytest.raises(ValueError, match="positive pair and one negative pair"):
        TripletLoss()(features, torch.zeros(4, dtype=torch.long))

    with pytest.raises(ValueError, match="no positive pairs"):
        SupConLoss()(features, torch.arange(4))

    with pytest.raises(ValueError, match="at least two classes"):
        PrototypicalLoss()(features, torch.zeros(4, dtype=torch.long))
