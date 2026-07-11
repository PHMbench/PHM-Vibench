from typing import Optional

import torch
import torch.nn as nn

import src.task_factory.Components.flow as flow_components
from src.task_factory.Components.flow import FlowLoss
from src.task_factory.Components.mean_flow_loss import MeanFlow


class _ZeroVelocity(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class _ConstantVelocity(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self.value)


class _IdentityImageVelocity(nn.Module):
    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return z


class _ZeroImageVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.zeros_like(z)


def test_flow_loss_trains_against_target_velocity(monkeypatch) -> None:
    loss_fn = FlowLoss(target_channels=3, z_channels=2, depth=1, width=8, num_sampling_steps=4)
    loss_fn.net = _ZeroVelocity()

    target = torch.tensor([[2.0, 4.0, 6.0]])
    z = torch.tensor([[0.5, -0.5]])
    noise = torch.tensor([[1.0, 1.0, 1.0]])

    monkeypatch.setattr(flow_components.torch, "randn_like", lambda _: noise)
    monkeypatch.setattr(
        flow_components.torch,
        "rand",
        lambda *shape, **kwargs: torch.full(shape, 0.25, device=kwargs.get("device")),
    )

    loss = loss_fn(target, z)

    target_velocity = target - noise
    weights = torch.tensor([1.0, 0.5, 1.0 / 3.0])
    expected = (weights * target_velocity.square()).sum()
    assert torch.allclose(loss, expected)


def test_flow_loss_sampler_integrates_predicted_velocity(monkeypatch) -> None:
    loss_fn = FlowLoss(target_channels=2, z_channels=3, depth=1, width=8, num_sampling_steps=4)
    loss_fn.net = _ConstantVelocity(0.5)

    monkeypatch.setattr(flow_components.torch, "randn", lambda *shape, **kwargs: torch.zeros(*shape))

    z = torch.ones(1, 3)
    samples = loss_fn.sample(z, num_samples=3)

    assert samples.shape == (1, 3, 2)
    assert torch.allclose(samples, torch.full_like(samples, 0.5))


def test_mean_flow_unconditional_loss_is_finite() -> None:
    loss_fn = MeanFlow(channels=1, image_size=4, num_classes=None, flow_ratio=0.0)
    model = _IdentityImageVelocity()
    x = torch.rand(2, 1, 4, 4)

    loss, mse = loss_fn.loss(model, x, c=None)

    assert torch.isfinite(loss)
    assert torch.isfinite(mse)


def test_mean_flow_sampler_defaults_to_model_device() -> None:
    loss_fn = MeanFlow(channels=1, image_size=4, num_classes=2)
    model = _ZeroImageVelocity()

    samples = loss_fn.sample_each_class(model, n_per_class=1, sample_steps=1)

    assert samples.shape == (2, 1, 4, 4)
    assert samples.device == model.anchor.device
