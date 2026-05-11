from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.task_factory.Components.generative.losses.ddpm import (
    DDPMEpsilonPredictionLoss,
    ddpm_sampler_metadata,
)
from src.task_factory.Components.generative.samplers.ddpm import sample as sample_ddpm
from src.task_factory.task.generative.conditional_flow_matching import (
    ConditionalFlowMatchingTask,
)


class DdpmEpsilonTask(ConditionalFlowMatchingTask):
    """DDPM epsilon-prediction task for PHM `[N,C,L]` windows."""

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
        super().__init__(
            network=network,
            args_data=args_data,
            args_model=args_model,
            args_task=args_task,
            args_trainer=args_trainer,
            args_environment=args_environment,
            metadata=metadata,
        )
        self.loss_id = "ddpm_epsilon"
        self.sampler_id = "ddpm"
        self.loss_fn = DDPMEpsilonPredictionLoss(
            num_train_timesteps=int(getattr(args_task, "num_train_timesteps", 1000))
        )

    def sampler_metadata(self) -> dict[str, Any]:
        return ddpm_sampler_metadata(self.loss_fn.scheduler)

    def _shared_step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        x0 = self._to_ncl(batch["x"])
        condition = self._extract_condition(batch)
        epsilon = torch.randn_like(x0)
        t_idx = self.loss_fn.sample_timesteps(x0.shape[0], x0.device)
        x_t = self.loss_fn.q_sample(x0, epsilon, t_idx)
        model_t = t_idx.float() / float(max(self.loss_fn.scheduler.num_train_timesteps - 1, 1))
        pred_epsilon = self.forward(x_t, model_t, condition)
        loss_dict = self.loss_fn(pred_epsilon, x0, epsilon, t_idx)
        loss = loss_dict["loss"]
        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=x0.shape[0],
        )
        self.log(
            f"{stage}_mse_epsilon",
            loss_dict["mse_epsilon"],
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=x0.shape[0],
        )
        return loss

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
        return sample_ddpm(
            self.network.to(sample_device),
            noise,
            expanded_condition,
            self.loss_fn.scheduler,
            num_steps=num_steps,
        )


task = DdpmEpsilonTask
