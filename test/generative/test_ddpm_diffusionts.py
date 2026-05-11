from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from src.task_factory.Components.generative.losses.ddpm import (
    DDPMEpsilonPredictionLoss,
    ddpm_sampler_metadata,
)
from src.task_factory.Components.generative.samplers.ddpm import sample as sample_ddpm


REPO = Path(__file__).resolve().parents[2]


class _ZeroEpsilon(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: dict[str, torch.Tensor]) -> torch.Tensor:
        assert t.min().item() >= 0.0
        assert t.max().item() <= 1.0
        return torch.zeros_like(x)


def _condition(n: int = 2) -> dict[str, torch.Tensor]:
    return {
        "fault_label": torch.zeros(n, dtype=torch.long),
        "domain_id": torch.zeros(n, dtype=torch.long),
    }


def test_ddpm_loss_q_sample_and_epsilon_contract_are_finite() -> None:
    loss_fn = DDPMEpsilonPredictionLoss(num_train_timesteps=10)
    x0 = torch.randn(2, 2, 16)
    epsilon = torch.randn_like(x0)
    t = loss_fn.sample_timesteps(x0.shape[0], x0.device)
    x_t = loss_fn.q_sample(x0, epsilon, t)

    loss = loss_fn(epsilon, x0, epsilon, t)

    assert x_t.shape == x0.shape
    assert torch.isfinite(x_t).all()
    assert loss["loss"].item() == 0.0


def test_ddpm_sampler_returns_finite_shape() -> None:
    loss_fn = DDPMEpsilonPredictionLoss(num_train_timesteps=10)
    noise = torch.randn(2, 2, 16)

    sample = sample_ddpm(_ZeroEpsilon(), noise, _condition(), loss_fn.scheduler, num_steps=4, seed=0)

    assert sample.shape == noise.shape
    assert torch.isfinite(sample).all()


def test_ddpm_task_records_sampler_metadata() -> None:
    metadata = ddpm_sampler_metadata(DDPMEpsilonPredictionLoss(num_train_timesteps=10).scheduler)

    assert metadata["scheduler"] == "ddpm_linear_beta"
    assert metadata["num_train_timesteps"] == 10
    assert metadata["prediction_type"] == "epsilon"


def test_ddpm_demo_preflight_passes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--config",
            "configs/demo/10_generative/dummy_generative_ddpm.yaml",
            "--preflight-only",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Pipeline_06_generative" in result.stdout
