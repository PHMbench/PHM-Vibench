from __future__ import annotations

from typing import Any

import torch.nn as nn

from src.task_factory.task.generative.rectified_flow import RectifiedFlowTask


class ExperimentalOneStepFlowTask(RectifiedFlowTask):
    """Shared one-step exploratory flow task.

    These methods reuse the rectified-flow velocity contract until a promotion
    goal supplies method-specific benchmark evidence.
    """

    method_id = "experimental_one_step_flow"

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
        self.sampler_id = "one_step_euler"

    def sampler_metadata(self) -> dict[str, Any]:
        return {
            "experimental": True,
            "method_id": self.method_id,
            "one_step": True,
            "promotion_required_for_benchmark_valid": True,
        }
