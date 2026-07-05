from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.task_factory.Components.generative.losses.ot_nfm import OTNFMLoss
from src.task_factory.task.generative.ot_nfm import OtNfmTask


class TinyMap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(2, 2, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.conv(x)


def _task() -> OtNfmTask:
    args_task = SimpleNamespace(
        type="generative",
        name="ot_nfm",
        lr=1e-4,
        weight_decay=0.0,
        optimizer="adamw",
        generative=SimpleNamespace(experimental=True, num_steps=1, validity_status="exploratory"),
    )
    args_model = SimpleNamespace(type="generative_model", name="tiny", in_channels=2)
    args_data = SimpleNamespace(normalization="standardization", normalization_scope="window")
    metadata = {
        0: {"Label": 0, "Domain_id": 0},
        1: {"Label": 1, "Domain_id": 1},
    }
    return OtNfmTask(TinyMap(), args_data, args_model, args_task, SimpleNamespace(), SimpleNamespace(), metadata)


def test_ot_nfm_loss_is_finite_and_backward() -> None:
    loss_fn = OTNFMLoss()
    z = torch.randn(3, 2, 8)
    x1 = torch.randn(3, 2, 8)
    pred = torch.randn(3, 2, 8, requires_grad=True)

    loss_dict = loss_fn(pred, x1, z)
    loss_dict["loss"].backward()

    assert torch.isfinite(loss_dict["loss"])
    assert pred.grad is not None
    assert loss_dict["pairing_indices"].unique().numel() == 3


def test_ot_nfm_loss_rejects_small_batch() -> None:
    with pytest.raises(ValueError, match="batch size >=2"):
        OTNFMLoss()(torch.randn(1, 2, 8), torch.randn(1, 2, 8), torch.randn(1, 2, 8))


def test_ot_nfm_task_train_and_sample_contract() -> None:
    task = _task()
    batch = {
        "x": torch.randn(2, 2, 16),
        "file_id": torch.tensor([0, 1]),
        "fault_label": torch.tensor([0, 1]),
        "domain_id": torch.tensor([0, 1]),
    }

    loss = task.training_step(batch)
    loss.backward()
    sample = task.sample(
        {"fault_label": torch.tensor([0]), "domain_id": torch.tensor([0])},
        num_samples=2,
        length=16,
        channels=2,
        num_steps=1,
        device="cpu",
    )

    assert torch.isfinite(loss)
    assert sample.shape == (2, 2, 16)
    assert task.loss_id == "ot_nfm"
    assert task.sampler_metadata()["method_fidelity"] == "experimental_method_specific_ot_nfm"
