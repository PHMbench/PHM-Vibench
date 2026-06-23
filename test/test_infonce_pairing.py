from __future__ import annotations

import pytest
import torch

from src.task_factory.Components.contrastive_losses import InfoNCELoss


def test_unlabeled_infonce_uses_two_view_pairs() -> None:
    torch.manual_seed(0)
    base = torch.randn(4, 8)
    two_view_features = torch.cat([base, base + 0.01 * torch.randn_like(base)], dim=0)

    loss = InfoNCELoss(temperature=0.07)(two_view_features)

    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_unlabeled_infonce_rejects_odd_batch() -> None:
    features = torch.randn(5, 8)

    with pytest.raises(ValueError, match="two-view batch"):
        InfoNCELoss()(features)


def test_supervised_infonce_rejects_no_positive_pairs() -> None:
    features = torch.randn(4, 8)
    labels = torch.arange(4)

    with pytest.raises(ValueError, match="no positive pairs"):
        InfoNCELoss()(features, labels)
