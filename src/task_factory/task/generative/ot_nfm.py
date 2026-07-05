from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.task_factory.Components.generative.losses.ot_nfm import OTNFMLoss
from src.task_factory.task.generative.base_one_step_map import BaseOneStepMapTask


class OtNfmTask(BaseOneStepMapTask):
    method_id = "ot_nfm"

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
        self.loss_id = "ot_nfm"
        self.loss_fn = OTNFMLoss()

    def _shared_step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        x1 = self._to_ncl(batch["x"])
        condition = self._extract_condition(batch)
        z = torch.randn_like(x1)
        pred_map = self.map_forward(z, condition)
        loss_dict = self.loss_fn(pred_map, x1, z)
        loss = loss_dict["loss"]
        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=x1.shape[0],
        )
        self.log(
            f"{stage}_mse_map",
            loss_dict["mse_map"],
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=x1.shape[0],
        )
        self.log(
            f"{stage}_pairing_cost",
            loss_dict["pairing_cost"],
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=x1.shape[0],
        )
        return loss

    def sampler_metadata(self) -> dict[str, Any]:
        metadata = super().sampler_metadata()
        metadata.update(
            {
                "method_fidelity": "experimental_method_specific_ot_nfm",
                "pairing": "minibatch_flattened_l2_assignment",
                "cost": "torch.cdist(flatten(z), flatten(x1), p=2)",
            }
        )
        return metadata


task = OtNfmTask
