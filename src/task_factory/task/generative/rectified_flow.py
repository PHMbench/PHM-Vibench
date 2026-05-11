from __future__ import annotations

from typing import Any

import torch.nn as nn

from src.task_factory.Components.generative.losses.rectified_flow import RectifiedFlowLoss
from src.task_factory.task.generative.conditional_flow_matching import (
    ConditionalFlowMatchingTask,
)


class RectifiedFlowTask(ConditionalFlowMatchingTask):
    """Factory-integrated Rectified Flow task for PHM velocity models."""

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
        self.loss_id = "rectified_flow"
        self.loss_fn = RectifiedFlowLoss(eps=float(getattr(args_task, "t_eps", 1e-3)))


task = RectifiedFlowTask
