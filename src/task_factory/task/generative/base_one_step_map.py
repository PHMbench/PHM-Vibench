from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.task_factory.Components.generative.samplers.one_step_map import sample_one_step_map
from src.task_factory.task.generative.conditional_flow_matching import (
    ConditionalFlowMatchingTask,
)


class BaseOneStepMapTask(ConditionalFlowMatchingTask):
    """Shared one-step transport-map task contract for `[N, C, L]` PHM signals."""

    method_id = "one_step_map_base"

    def __init__(
        self,
        network: nn.Module,
        args_data: Any,
        args_model: Any,
        args_task: Any,
        args_trainer: Any,
        args_environment: Any,
        metadata: Any,
    ) -> None:
        gen_cfg = getattr(args_task, "generative", None)
        if gen_cfg is None or not bool(getattr(gen_cfg, "experimental", False)):
            raise ValueError(f"{self.method_id} requires task.generative.experimental=true")
        if int(getattr(gen_cfg, "num_steps", 0)) != 1:
            raise ValueError(f"{self.method_id} requires task.generative.num_steps=1")
        if str(getattr(gen_cfg, "validity_status", "exploratory")) == "benchmark-valid":
            raise ValueError(f"{self.method_id} cannot be benchmark-valid before promotion")
        super().__init__(
            network=network,
            args_data=args_data,
            args_model=args_model,
            args_task=args_task,
            args_trainer=args_trainer,
            args_environment=args_environment,
            metadata=metadata,
        )
        self.loss_id = self.method_id
        self.loss_fn = None
        self.sampler_id = "one_step_map"

    def map_forward(
        self,
        z: torch.Tensor,
        condition: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError(f"z must be [N, C, L], got shape={tuple(z.shape)}")
        t = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        return self.forward(z, t, condition)

    @torch.no_grad()
    def sample(
        self,
        condition: dict[str, torch.Tensor],
        *,
        num_samples: int,
        length: int,
        channels: int,
        num_steps: int,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if int(num_steps) != 1:
            raise ValueError(f"{self.method_id} one-step sampler requires num_steps=1")
        sample_device = torch.device(device or self.device)
        noise = torch.randn(num_samples, channels, length, device=sample_device)
        expanded_condition = {}
        for key, value in condition.items():
            value = value.to(sample_device).long().view(-1)
            if value.numel() == 1 and num_samples > 1:
                value = value.repeat(num_samples)
            if value.numel() != num_samples:
                raise ValueError(
                    f"condition {key} must have 1 or num_samples values; "
                    f"got {value.numel()} for num_samples={num_samples}"
                )
            expanded_condition[key] = value
        return sample_one_step_map(self.network.to(sample_device), noise, expanded_condition)

    def sampler_metadata(self) -> dict[str, Any]:
        return {
            "experimental": True,
            "method_id": self.method_id,
            "one_step": True,
            "nfe": 1,
            "sampler_id": self.sampler_id,
            "promotion_required_for_benchmark_valid": True,
        }
