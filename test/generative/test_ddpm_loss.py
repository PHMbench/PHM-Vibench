import pytest
import torch

from src.task_factory.Components.generative.losses.ddpm import DDPMEpsilonPredictionLoss
from src.task_factory.Components.generative.schedulers.ddpm import DDPMScheduler


def test_ddpm_epsilon_loss_and_q_sample_contract():
    scheduler = DDPMScheduler(num_train_timesteps=10)
    loss_fn = DDPMEpsilonPredictionLoss(scheduler=scheduler)
    x0 = torch.randn(4, 2, 16)
    epsilon = torch.randn_like(x0)
    t = torch.tensor([0, 1, 5, 9])

    x_t = loss_fn.q_sample(x0, epsilon, t)
    out = loss_fn(epsilon, x0, epsilon, t)

    assert x_t.shape == x0.shape
    assert torch.isclose(out["loss"], torch.tensor(0.0))


def test_ddpm_scheduler_finite_monotonic_and_bounds():
    scheduler = DDPMScheduler(num_train_timesteps=10)

    assert torch.isfinite(scheduler.betas).all()
    assert torch.all(scheduler.alpha_bars[1:] < scheduler.alpha_bars[:-1])
    with pytest.raises(ValueError, match="out of range"):
        scheduler.alpha_bar_at(torch.tensor([10]), device="cpu", dtype=torch.float32)


def test_ddpm_loss_rejects_shape_mismatch():
    loss_fn = DDPMEpsilonPredictionLoss(DDPMScheduler(num_train_timesteps=4))
    with pytest.raises(ValueError, match="shape mismatch"):
        loss_fn(torch.randn(2, 2, 8), torch.randn(2, 2, 8), torch.randn(2, 2, 7), torch.tensor([0, 1]))

