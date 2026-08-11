from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.task_factory.Components.generative import (
    ConditionalFlowMatchingLoss,
    PopulationCorrelationMMD,
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
        self.method_id = "conditional_flow_matching"
        self.sampler_id = "euler_ode"
        self.loss_fn = ConditionalFlowMatchingLoss(
            eps=float(getattr(args_task, "t_eps", 1e-3))
        )
        population = getattr(args_task, "population_regularization", None)
        if isinstance(population, Mapping):
            population_get = population.get
        else:
            population_get = lambda key, default=None: getattr(
                population, key, default
            )
        self.population_regularization_enabled = bool(
            population_get("enabled", False) if population is not None else False
        )
        self.population_weight = 0.0
        self.population_loss = None
        if self.population_regularization_enabled:
            dependency = str(population_get("dependency", "pearson_correlation"))
            estimator = str(population_get("estimator", "biased"))
            same_time = bool(population_get("same_time_per_batch", True))
            self.population_weight = float(population_get("weight", 0.1))
            bandwidths = population_get(
                "rbf_bandwidths",
                [0.1, 0.5, 1.0, 2.0],
            )
            if dependency != "pearson_correlation":
                raise ValueError(
                    "population_regularization.dependency must be "
                    "pearson_correlation"
                )
            if estimator != "biased":
                raise ValueError(
                    "population_regularization.estimator must be biased"
                )
            if not same_time:
                raise ValueError(
                    "population_regularization requires same_time_per_batch=true"
                )
            if self.population_weight <= 0.0:
                raise ValueError(
                    "population_regularization.weight must be positive"
                )
            self.population_loss = PopulationCorrelationMMD(bandwidths)
            self.method_id = "population_aware_cfm"
            self.loss_id = "conditional_flow_matching+population_correlation_mmd"
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
                "population_regularization_enabled": (
                    self.population_regularization_enabled
                ),
                "population_weight": self.population_weight,
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
        time_batch_size = (
            1 if self.population_regularization_enabled else x1.shape[0]
        )
        t = self.loss_fn.sample_t(
            time_batch_size,
            x1.device,
            dtype=x1.dtype,
        )
        if self.population_regularization_enabled:
            t = t.expand(x1.shape[0]).clone()
        x_t = self.loss_fn.sample_xt(x1, noise, t)
        predicted_velocity = self.forward(x_t, t, condition)
        loss_values = self.loss_fn(predicted_velocity, x1, noise, t)
        loss = loss_values["loss"]
        population_mmd = None
        if self.population_regularization_enabled:
            t_view = t.to(device=x1.device, dtype=x1.dtype).view(-1, 1, 1)
            predicted_clean = x_t + (1.0 - t_view) * predicted_velocity
            population_mmd = self.population_loss(x1, predicted_clean)
            loss = loss + self.population_weight * population_mmd
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
        if population_mmd is not None:
            self.log(
                f"{stage}_population_correlation_mmd",
                population_mmd,
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
