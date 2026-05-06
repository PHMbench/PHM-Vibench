from __future__ import annotations

import torch

from src.task_factory.Components.generative.samplers.euler_ode import sample_euler_ode


def generate_samples(
    model,
    *,
    noise: torch.Tensor,
    condition: dict[str, torch.Tensor],
    num_steps: int,
) -> torch.Tensor:
    """Thin task-level wrapper for CFM Euler sampling."""
    return sample_euler_ode(model, noise, condition, num_steps=num_steps)

