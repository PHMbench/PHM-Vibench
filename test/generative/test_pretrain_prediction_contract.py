from argparse import Namespace

import pytest
import torch
import torch.nn as nn

from src.task_factory.Components.prediction_loss import Signal_mask_Loss
from src.task_factory.task.pretrain.prediction import task as PredictionTask


class _IdentityPredictionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.seen_file_id = None
        self.seen_input = None

    def forward(self, x: torch.Tensor, file_id=None, task_id=None) -> torch.Tensor:
        assert task_id == "prediction"
        self.seen_file_id = file_id
        self.seen_input = x.detach().clone()
        return x + self.anchor


class _ShortPredictionModel(_IdentityPredictionModel):
    def forward(self, x: torch.Tensor, file_id=None, task_id=None) -> torch.Tensor:
        super().forward(x, file_id=file_id, task_id=task_id)
        return x[:, :-1, :]


class _OnesPredictionModel(_IdentityPredictionModel):
    def forward(self, x: torch.Tensor, file_id=None, task_id=None) -> torch.Tensor:
        super().forward(x, file_id=file_id, task_id=task_id)
        return torch.ones_like(x)


class _MetaPredictionModel(_IdentityPredictionModel):
    def forward(self, x: torch.Tensor, file_id=None, task_id=None) -> torch.Tensor:
        super().forward(x, file_id=file_id, task_id=task_id)
        return torch.empty_like(x, device="meta")


def _namespace(**kwargs):
    return Namespace(**kwargs)


def test_pretrain_prediction_shared_step_reports_prediction_loss() -> None:
    network = _IdentityPredictionModel()
    args_task = _namespace(
        loss="CE",
        metrics=["acc"],
        mask_ratio=0.0,
        forecast_part=0.5,
        alpha_prediction=0.25,
        regularization={},
    )
    task = PredictionTask(
        network=network,
        args_data=_namespace(),
        args_model=_namespace(),
        args_task=args_task,
        args_trainer=_namespace(gpus=0),
        args_environment=_namespace(),
        metadata={7: {"Name": "Dummy_Data", "Label": 0}},
    )
    batch = {
        "x": torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]),
        "y": torch.tensor([0]),
        "file_id": torch.tensor([7]),
    }

    metrics = task._shared_step(batch, "train")

    assert "train_Dummy_Data_pred_loss" in metrics
    assert "train_total_loss" in metrics
    assert torch.allclose(metrics["train_Dummy_Data_pred_loss"], torch.tensor(12.5))
    assert torch.allclose(metrics["train_total_loss"], torch.tensor(3.125))
    assert torch.equal(network.seen_file_id, torch.tensor([7]))
    assert torch.equal(network.seen_input, torch.tensor([[[1.0], [2.0], [0.0], [0.0]]]))


def test_pretrain_prediction_requires_reconstruction_shape_match() -> None:
    args_task = _namespace(
        loss="CE",
        metrics=["acc"],
        mask_ratio=0.0,
        forecast_part=0.5,
        alpha_prediction=1.0,
        regularization={},
    )
    task = PredictionTask(
        network=_ShortPredictionModel(),
        args_data=_namespace(),
        args_model=_namespace(),
        args_task=args_task,
        args_trainer=_namespace(gpus=0),
        args_environment=_namespace(),
        metadata={7: {"Name": "Dummy_Data", "Label": 0}},
    )
    batch = {
        "x": torch.randn(2, 8, 2),
        "y": torch.tensor([0, 1]),
        "file_id": torch.tensor([7, 7]),
    }

    with pytest.raises(ValueError, match="output shape must match input signal shape"):
        task._shared_step(batch, "train")


def test_signal_mask_loss_uses_configured_relative_l2() -> None:
    cfg = _namespace(mask_ratio=0.0, forecast_part=1.0, loss_type="rel_l2")
    loss_fn = Signal_mask_Loss(cfg)
    model = _OnesPredictionModel()
    batch = {"x": torch.full((1, 4, 1), 2.0), "file_id": None}

    loss = loss_fn(model, batch)

    assert torch.allclose(loss, torch.tensor(0.5))


def test_signal_mask_loss_rejects_device_mismatch() -> None:
    cfg = _namespace(mask_ratio=0.0, forecast_part=1.0, loss_type="mse")
    loss_fn = Signal_mask_Loss(cfg)
    model = _MetaPredictionModel()
    batch = {"x": torch.ones(1, 4, 1), "file_id": None}

    with pytest.raises(ValueError, match="output device must match input signal device"):
        loss_fn(model, batch)
