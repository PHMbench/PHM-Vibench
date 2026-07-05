import pytest
import torch

from src.task_factory.Components.generative.samplers.ddpm import sample
from src.task_factory.Components.generative.schedulers.ddpm import DDPMScheduler


class ZeroEpsilonModel(torch.nn.Module):
    def forward(self, x_t, t, condition):
        return torch.zeros_like(x_t)


class NaNEpsilonModel(torch.nn.Module):
    def forward(self, x_t, t, condition):
        return torch.full_like(x_t, float("nan"))


def test_ddpm_sampler_shape_and_fixed_seed_smoke():
    noise = torch.randn(2, 2, 16)
    condition = {
        "fault_label": torch.tensor([0, 1]),
        "domain_id": torch.tensor([0, 1]),
    }
    scheduler = DDPMScheduler(num_train_timesteps=8)

    out1 = sample(ZeroEpsilonModel(), noise, condition, scheduler, num_steps=4, seed=7)
    out2 = sample(ZeroEpsilonModel(), noise, condition, scheduler, num_steps=4, seed=7)

    assert out1.shape == noise.shape
    assert torch.allclose(out1, out2)


def test_ddpm_sampler_rejects_nan_model_output():
    noise = torch.randn(1, 2, 8)
    condition = {"fault_label": torch.tensor([0]), "domain_id": torch.tensor([0])}

    with pytest.raises(ValueError, match="epsilon contains NaN/Inf"):
        sample(NaNEpsilonModel(), noise, condition, DDPMScheduler(num_train_timesteps=4), num_steps=2)

