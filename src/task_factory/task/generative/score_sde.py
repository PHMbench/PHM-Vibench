from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.task_factory.Components.generative.losses.score_sde import ScoreSDEResearchLoss
from src.task_factory.Components.generative.samplers.score_sde import (
    sample_score_sde_annealed_langevin,
)
from src.task_factory.task.generative.conditional_flow_matching import (
    ConditionalFlowMatchingTask,
)


def _cfg_attr(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class ScoreSdeTask(ConditionalFlowMatchingTask):
    """Exploratory score-SDE task with explicit stochastic sampler settings."""

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
        self.loss_id = "score_sde_dsm"
        self.sampler_id = "score_sde_annealed_langevin"
        self.loss_fn = ScoreSDEResearchLoss()
        gen_cfg = getattr(args_task, "generative", None)
        if gen_cfg is None:
            raise ValueError("score_sde requires task.generative stochastic settings")
        self.sigma_min = float(_cfg_attr(gen_cfg, "sigma_min"))
        self.sigma_max = float(_cfg_attr(gen_cfg, "sigma_max"))
        self.stochastic_step_size = float(_cfg_attr(gen_cfg, "stochastic_step_size"))
        if str(_cfg_attr(gen_cfg, "stochastic_sampler", "")) != "annealed_langevin":
            raise ValueError("score_sde requires task.generative.stochastic_sampler=annealed_langevin")
        if not 0.0 < self.sigma_min < self.sigma_max:
            raise ValueError("score_sde requires 0 < sigma_min < sigma_max")
        if self.stochastic_step_size <= 0.0:
            raise ValueError("score_sde requires stochastic_step_size > 0")

    def sampler_metadata(self) -> dict[str, Any]:
        return {
            "stochastic": True,
            "stochastic_sampler": "annealed_langevin",
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "stochastic_step_size": self.stochastic_step_size,
        }

    def _shared_step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        x0 = self._to_ncl(batch["x"])
        condition = self._extract_condition(batch)
        u = torch.rand(x0.shape[0], device=x0.device)
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** u
        noise = torch.randn_like(x0)
        sigma_view = sigma.view(-1, 1, 1)
        x_t = x0 + sigma_view * noise
        target_score = -noise / sigma_view.clamp_min(1e-8)
        pred_score = self.forward(x_t, sigma, condition)
        loss_dict = self.loss_fn(pred_score, target_score)
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
            f"{stage}_mse_score",
            loss_dict["mse_score"],
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
        return sample_score_sde_annealed_langevin(
            self.network.to(sample_device),
            noise,
            expanded_condition,
            num_steps=num_steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            step_size=self.stochastic_step_size,
        )


task = ScoreSdeTask
