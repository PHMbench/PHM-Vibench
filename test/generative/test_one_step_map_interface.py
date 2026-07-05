from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.task_factory.Components.generative.samplers.one_step_map import sample_one_step_map
from src.task_factory.task.generative.base_one_step_map import BaseOneStepMapTask


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
        return self.conv(x) + t.view(-1, 1, 1) * 0.0


class DummyOneStepTask(BaseOneStepMapTask):
    method_id = "dummy_one_step_map"

    def training_step(self, batch: dict, *args, **kwargs) -> torch.Tensor:
        x1 = self._to_ncl(batch["x"])
        condition = self._extract_condition(batch)
        z = torch.randn_like(x1)
        pred = self.map_forward(z, condition)
        loss = torch.nn.functional.mse_loss(pred, x1)
        self.log("train_loss", loss, batch_size=x1.shape[0])
        return loss


def _task() -> DummyOneStepTask:
    args_task = SimpleNamespace(
        type="generative",
        name="dummy",
        lr=1e-4,
        weight_decay=0.0,
        optimizer="adamw",
        generative=SimpleNamespace(experimental=True, num_steps=1, validity_status="exploratory"),
    )
    args_model = SimpleNamespace(type="generative_model", name="tiny", in_channels=2)
    args_data = SimpleNamespace(normalization="standardization", normalization_scope="window")
    args_trainer = SimpleNamespace()
    args_environment = SimpleNamespace()
    metadata = {
        0: {"Label": 0, "Domain_id": 0},
        1: {"Label": 1, "Domain_id": 1},
    }
    return DummyOneStepTask(
        TinyMap(),
        args_data,
        args_model,
        args_task,
        args_trainer,
        args_environment,
        metadata,
    )


def test_one_step_map_task_train_and_sample_are_finite() -> None:
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
    assert task.sampler_metadata()["nfe"] == 1


def test_one_step_map_sampler_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="noise must be"):
        sample_one_step_map(TinyMap(), torch.randn(2, 16), {"fault_label": torch.tensor([0, 1])})


def test_one_step_map_sampler_rejects_condition_batch_mismatch() -> None:
    with pytest.raises(ValueError, match="batch size mismatch"):
        sample_one_step_map(
            TinyMap(),
            torch.randn(2, 2, 16),
            {"fault_label": torch.tensor([0]), "domain_id": torch.tensor([0, 1])},
        )
