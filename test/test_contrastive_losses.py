import importlib.util
from pathlib import Path

import torch


def _load_infonce_loss():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "task_factory"
        / "Components"
        / "contrastive_losses.py"
    )
    spec = importlib.util.spec_from_file_location("contrastive_losses", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.InfoNCELoss


InfoNCELoss = _load_infonce_loss()


def test_unsupervised_infonce_uses_simclr_pairs():
    loss_fn = InfoNCELoss(temperature=0.5)
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.1],
            [0.1, 1.0],
        ]
    )

    loss = loss_fn(features)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_unsupervised_infonce_odd_batch_falls_back_to_zero():
    loss_fn = InfoNCELoss(temperature=0.5)
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.1],
        ]
    )

    loss = loss_fn(features)

    assert torch.isclose(loss, torch.tensor(0.0))
