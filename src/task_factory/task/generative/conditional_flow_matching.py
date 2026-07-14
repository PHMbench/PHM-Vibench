from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.task_factory.Components.generative import (
    ConditionalFlowMatchingLoss,
    sample_euler_ode,
)


def _flatten_values(value: Any) -> list[Any]:
    if torch.is_tensor(value):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten_values(item))
        return flattened
    return [value]


class ConditionalFlowMatchingTask(pl.LightningModule):
    """Lightning task for the first PHM Conditional Flow Matching slice."""

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
        super().__init__()
        self.network = network
        self.args_data = args_data
        self.args_model = args_model
        self.args_task = args_task
        self.args_trainer = args_trainer
        self.args_environment = args_environment
        self.metadata = metadata
        self.loss_id = "conditional_flow_matching"
        self.sampler_id = "euler_ode"
        self.loss_fn = ConditionalFlowMatchingLoss(
            eps=float(getattr(args_task, "t_eps", 1e-3))
        )
        self.save_hyperparameters(
            {
                "task_type": getattr(args_task, "type", "generative"),
                "task_name": getattr(
                    args_task,
                    "name",
                    "conditional_flow_matching",
                ),
                "model_type": getattr(
                    args_model,
                    "type",
                    "generative_model",
                ),
                "model_name": getattr(
                    args_model,
                    "name",
                    "phm_cfm_mlp1d",
                ),
                "lr": float(getattr(args_task, "lr", 1e-4)),
                "weight_decay": float(
                    getattr(args_task, "weight_decay", 1e-4)
                ),
            }
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.network(x_t, t, condition)

    def _to_ncl(self, x: Any) -> torch.Tensor:
        tensor = torch.as_tensor(x, device=self.device).float()
        if tensor.ndim != 3:
            raise ValueError(
                "expected x as [N,L,C] or [N,C,L], got "
                f"{tuple(tensor.shape)}"
            )
        channels = int(getattr(self.args_model, "in_channels", 2))
        if tensor.shape[1] == channels:
            return tensor.contiguous()
        if tensor.shape[2] == channels:
            return tensor.transpose(1, 2).contiguous()
        raise ValueError(
            f"cannot infer channel axis for shape={tuple(tensor.shape)} "
            f"and in_channels={channels}"
        )

    def _metadata_row(self, file_id: Any) -> Mapping[str, Any]:
        candidates = [file_id]
        try:
            candidates.append(int(file_id))
        except (TypeError, ValueError):
            pass
        candidates.append(str(file_id))

        for key in candidates:
            try:
                row = self.metadata[key]
            except (KeyError, TypeError, IndexError):
                continue
            if isinstance(row, Mapping):
                return row
        raise ValueError(f"file_id={file_id!r} is missing from metadata")

    def _condition_values(
        self,
        batch: Mapping[str, Any],
        *,
        batch_key: str,
        metadata_key: str,
        file_ids: list[Any],
        fallback_batch_key: str | None = None,
    ) -> torch.Tensor:
        if batch_key in batch:
            values = _flatten_values(batch[batch_key])
        elif fallback_batch_key and fallback_batch_key in batch:
            values = _flatten_values(batch[fallback_batch_key])
        else:
            values = []
            for file_id in file_ids:
                row = self._metadata_row(file_id)
                if metadata_key not in row:
                    raise ValueError(
                        f"metadata for file_id={file_id!r} is missing "
                        f"required key {metadata_key}"
                    )
                values.append(row[metadata_key])

        if len(values) != len(file_ids):
            raise ValueError(
                f"{batch_key} length mismatch: expected {len(file_ids)}, "
                f"got {len(values)}"
            )
        return torch.as_tensor(
            values,
            device=self.device,
            dtype=torch.long,
        ).reshape(-1)

    def _extract_condition(
        self,
        batch: Mapping[str, Any],
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        if "file_id" not in batch:
            raise ValueError(
                "generative batches must contain file_id for condition provenance"
            )
        file_ids = _flatten_values(batch["file_id"])
        if len(file_ids) != batch_size:
            raise ValueError(
                f"file_id length mismatch: expected {batch_size}, "
                f"got {len(file_ids)}"
            )
        fault_label = self._condition_values(
            batch,
            batch_key="fault_label",
            metadata_key="Label",
            file_ids=file_ids,
            fallback_batch_key="y",
        )
        domain_id = self._condition_values(
            batch,
            batch_key="domain_id",
            metadata_key="Domain_id",
            file_ids=file_ids,
        )
        return {"fault_label": fault_label, "domain_id": domain_id}

    def _shared_step(
        self,
        batch: Mapping[str, Any],
        stage: str,
    ) -> torch.Tensor:
        if "x" not in batch:
            raise ValueError("generative batch is missing x")
        x1 = self._to_ncl(batch["x"])
        condition = self._extract_condition(batch, x1.shape[0])
        noise = torch.randn_like(x1)
        t = self.loss_fn.sample_t(
            x1.shape[0],
            x1.device,
            dtype=x1.dtype,
        )
        x_t = self.loss_fn.sample_xt(x1, noise, t)
        predicted_velocity = self.forward(x_t, t, condition)
        loss_values = self.loss_fn(predicted_velocity, x1, noise, t)
        loss = loss_values["loss"]
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
            f"{stage}_mse_v",
            loss_values["mse_v"],
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=x1.shape[0],
        )
        return loss

    def training_step(
        self,
        batch: Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(
        self,
        batch: Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(
        self,
        batch: Mapping[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer_name = str(
            getattr(self.args_task, "optimizer", "adamw")
        ).lower()
        lr = float(getattr(self.args_task, "lr", 1e-4))
        weight_decay = float(getattr(self.args_task, "weight_decay", 1e-4))
        if optimizer_name == "adam":
            return torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        if optimizer_name == "adamw":
            return torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        raise ValueError(
            f"unsupported optimizer for generative task: {optimizer_name}"
        )

    def sampler_metadata(self) -> dict[str, Any]:
        return {"sampler_id": self.sampler_id, "integration": "explicit_euler"}

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
        if int(num_samples) <= 0 or int(length) <= 0 or int(channels) <= 0:
            raise ValueError(
                "num_samples, length, and channels must all be positive"
            )
        if int(channels) != int(getattr(self.args_model, "in_channels", 2)):
            raise ValueError(
                f"sample channel mismatch: requested {channels}, model expects "
                f"{getattr(self.args_model, 'in_channels', 2)}"
            )
        noise = torch.randn(
            int(num_samples),
            int(channels),
            int(length),
            device=sample_device,
        )
        return sample_euler_ode(
            self.network.to(sample_device),
            noise,
            condition,
            num_steps=int(num_steps),
        )


# Backward-compatible task-factory discovery alias.
task = ConditionalFlowMatchingTask
