from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.model_factory.generative_model.phm_ddpm_mlp1d import Model
from src.task_factory.task.generative.conditional_ddpm import ConditionalDDPMTask


METADATA = {1: {"Label": 0, "Domain_id": 0}, 2: {"Label": 1, "Domain_id": 1}}


def _task(network=None) -> ConditionalDDPMTask:
    model_args = SimpleNamespace(
        in_channels=2, hidden_dim=8, condition_dim=4,
        num_fault_classes=2, num_domains=2,
    )
    task = ConditionalDDPMTask(
        network=network if network is not None else Model(model_args, METADATA),
        args_data=SimpleNamespace(normalization="standardization"),
        args_model=model_args,
        args_task=SimpleNamespace(diffusion_steps=4, beta_start=0.1, beta_end=0.4),
        args_trainer=SimpleNamespace(), args_environment=SimpleNamespace(),
        metadata=METADATA,
    )
    task.log = lambda *args, **kwargs: None
    return task


class _PointMassNoise(nn.Module):
    """Exact epsilon predictor when every clean waveform equals a known value."""

    def __init__(self, alpha_bars: torch.Tensor, clean_value: float, bias: float = 0.0):
        super().__init__()
        self.register_buffer("alpha_bars", alpha_bars)
        self.bias = nn.Parameter(torch.tensor(bias))
        self.clean_value = clean_value
        self.times = []

    def forward(self, x_t, time, condition):
        self.times.append(time.detach().clone())
        assert condition["fault_label"].shape == (x_t.shape[0],)
        steps = (time * len(self.alpha_bars)).round().long() - 1
        alpha_bar = self.alpha_bars.to(x_t)[steps].view(-1, 1, 1)
        return (x_t - alpha_bar.sqrt() * self.clean_value) / (1.0 - alpha_bar).sqrt() + self.bias


def test_reverse_mean_matches_bayesian_posterior_and_recovers_clean_endpoint():
    task = _task()
    clean = torch.tensor([[[3.0, -2.0]], [[1.5, 4.0]]], dtype=torch.float64)
    noise = torch.tensor([[[0.4, -0.1]], [[-0.5, 0.2]]], dtype=torch.float64)
    steps = torch.tensor([0, 2])
    noised = task._forward_diffusion(clean, noise, steps)
    mean, variance = task._reverse_parameters(noised, noise, steps)

    # Independent Eq. (7) posterior in terms of x0, not epsilon prediction.
    expected_means, expected_variances = [], []
    for row, step in enumerate(steps.tolist()):
        beta = task.betas[step]
        alpha = task.alphas[step]
        alpha_bar = task.alpha_bars[step]
        previous = task.alpha_bars[step - 1] if step else torch.tensor(1.0)
        expected_means.append(
            previous.sqrt() * beta / (1.0 - alpha_bar) * clean[row]
            + alpha.sqrt() * (1.0 - previous) / (1.0 - alpha_bar) * noised[row]
        )
        expected_variances.append(beta * (1.0 - previous) / (1.0 - alpha_bar))
    torch.testing.assert_close(mean, torch.stack(expected_means), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(variance.flatten(), torch.stack(expected_variances))
    torch.testing.assert_close(mean[0], clean[0], rtol=1e-12, atol=1e-12)
    assert variance[0].item() == 0.0


@pytest.mark.parametrize("bias", [0.0, 0.3])
def test_training_objective_is_epsilon_mse(bias):
    task = _task()
    task.network = _PointMassNoise(task.alpha_bars.clone(), clean_value=3.0, bias=bias)
    loss = task.training_step({
        "x": torch.full((2, 16, 2), 3.0),
        "y": torch.tensor([0, 1]), "file_id": torch.tensor([1, 2]),
    })
    torch.testing.assert_close(loss, torch.tensor(bias**2), atol=1e-6, rtol=1e-6)
    loss.backward()
    torch.testing.assert_close(task.network.bias.grad, torch.tensor(2.0 * bias), atol=1e-6, rtol=1e-6)


def test_ancestral_oracle_preserves_amplitude_and_omits_final_random_draw():
    task = _task()
    task.network = _PointMassNoise(task.alpha_bars.clone(), clean_value=3.0)
    task.network.train()
    torch.manual_seed(109)
    samples = task.sample(
        {"fault_label": torch.tensor([0]), "domain_id": torch.tensor([1])},
        num_samples=2, channels=2, length=16, num_steps=4, device="cpu",
    )
    following_draw = torch.randn(3)

    torch.testing.assert_close(samples, torch.full_like(samples, 3.0), rtol=1e-6, atol=1e-6)
    assert task.network.training
    assert not samples.requires_grad
    torch.testing.assert_close(
        torch.stack(task.network.times)[:, 0], torch.tensor([1.0, 0.75, 0.5, 0.25]),
    )
    # One initial Gaussian plus T-1 reverse noises; no draw at the final step.
    torch.manual_seed(109)
    for _ in range(4):
        torch.randn(2, 2, 16)
    torch.testing.assert_close(following_draw, torch.randn(3))
    assert task.sampler_metadata()["nfe"] == len(task.network.times) == 4


def test_sampler_rejects_a_different_schedule_before_denoising():
    task = _task()
    task.network = _PointMassNoise(task.alpha_bars.clone(), clean_value=3.0)
    with pytest.raises(ValueError, match="num_steps == diffusion_steps"):
        task.sample(
            {"fault_label": torch.tensor([0]), "domain_id": torch.tensor([0])},
            num_samples=1, channels=2, length=16, num_steps=2,
        )
    assert task.network.times == []


def test_actual_noise_model_receives_gradients_and_generates_finite_waveforms():
    task = _task()
    loss = task.training_step({
        "x": torch.randn(2, 16, 2), "y": torch.tensor([0, 1]),
        "file_id": torch.tensor([1, 2]),
    })
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in task.network.parameters())
    samples = task.sample(
        {"fault_label": torch.tensor([0, 1]), "domain_id": torch.tensor([0, 1])},
        num_samples=2, channels=2, length=16, num_steps=4,
    )
    assert samples.shape == (2, 2, 16)
    assert torch.isfinite(samples).all()
