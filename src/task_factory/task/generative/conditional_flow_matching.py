from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.task_factory.Components.generative.losses.flow_matching import (
    ConditionalFlowMatchingLoss,
)
from src.task_factory.Components.generative.manifests.synthetic_data_manifest import (
    build_synthetic_data_manifest,
    write_synthetic_data_manifest,
)
from src.task_factory.Components.generative.samplers.euler_ode import sample_euler_ode


class ConditionalFlowMatchingTask(pl.LightningModule):
    """Lightning wrapper for PHM Conditional Flow Matching."""

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
        self.loss_fn = ConditionalFlowMatchingLoss(eps=float(getattr(args_task, "t_eps", 1e-3)))
        self.save_hyperparameters(
            {
                "task_type": getattr(args_task, "type", "generative"),
                "task_name": getattr(args_task, "name", "conditional_flow_matching"),
                "model_type": getattr(args_model, "type", "generative_model"),
                "model_name": getattr(args_model, "name", "phm_cfm_mlp1d"),
                "lr": float(getattr(args_task, "lr", 1e-4)),
                "weight_decay": float(getattr(args_task, "weight_decay", 1e-4)),
            }
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.network(x_t, t, condition)

    def _to_ncl(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.device).float()
        if x.ndim != 3:
            raise ValueError(f"Expected x as [N, L, C] or [N, C, L], got {tuple(x.shape)}")
        expected_channels = int(
            getattr(self.args_model, "in_channels", getattr(self.args_model, "channels", x.shape[-1]))
        )
        if x.shape[1] == expected_channels:
            return x.contiguous()
        if x.shape[2] == expected_channels:
            return x.transpose(1, 2).contiguous()
        raise ValueError(
            f"Cannot infer channel axis from shape={tuple(x.shape)} and channels={expected_channels}"
        )

    def _flatten_values(self, value: Any) -> list[Any]:
        if torch.is_tensor(value):
            return value.detach().cpu().view(-1).tolist()
        if isinstance(value, (list, tuple)):
            out = []
            for item in value:
                if torch.is_tensor(item):
                    out.extend(item.detach().cpu().view(-1).tolist())
                else:
                    out.append(item)
            return out
        return [value]

    def _metadata_row(self, file_id: Any) -> dict[str, Any]:
        candidates = [file_id]
        try:
            candidates.append(int(file_id))
        except (TypeError, ValueError):
            pass
        candidates.append(str(file_id))
        for key in candidates:
            try:
                return self.metadata[key]
            except KeyError:
                continue
        raise ValueError(f"file_id={file_id!r} is missing from metadata")

    def _tensor_from_batch_or_metadata(
        self,
        batch: dict[str, Any],
        batch_key: str,
        metadata_key: str,
        file_ids: list[Any],
    ) -> torch.Tensor:
        if batch_key in batch:
            values = self._flatten_values(batch[batch_key])
        else:
            values = []
            for file_id in file_ids:
                row = self._metadata_row(file_id)
                if metadata_key not in row:
                    raise ValueError(
                        f"metadata for file_id={file_id!r} is missing required key {metadata_key}"
                    )
                values.append(row[metadata_key])
        if len(values) != len(file_ids):
            raise ValueError(
                f"{batch_key} length mismatch: expected {len(file_ids)}, got {len(values)}"
            )
        return torch.as_tensor(values, device=self.device, dtype=torch.long).view(-1)

    def _extract_condition(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if "file_id" not in batch:
            raise ValueError("generative batch must contain file_id for domain traceability")
        file_ids = self._flatten_values(batch["file_id"])
        fault_label = self._tensor_from_batch_or_metadata(batch, "fault_label", "Label", file_ids)
        if "fault_label" not in batch and "y" in batch:
            y_values = self._flatten_values(batch["y"])
            if len(y_values) == len(file_ids):
                fault_label = torch.as_tensor(y_values, device=self.device, dtype=torch.long).view(-1)
        domain_id = self._tensor_from_batch_or_metadata(batch, "domain_id", "Domain_id", file_ids)
        return {"fault_label": fault_label, "domain_id": domain_id}

    def _shared_step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        x1 = self._to_ncl(batch["x"])
        condition = self._extract_condition(batch)
        z = torch.randn_like(x1)
        t = self.loss_fn.sample_t(x1.shape[0], x1.device)
        x_t = self.loss_fn.sample_xt(x1, z, t)
        pred_velocity = self.forward(x_t, t, condition)
        loss_dict = self.loss_fn(pred_velocity, x1, z, t)
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
            f"{stage}_mse_v",
            loss_dict["mse_v"],
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=x1.shape[0],
        )
        return loss

    def training_step(self, batch: dict[str, Any], *args: Any, **kwargs: Any) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], *args: Any, **kwargs: Any) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, Any], *args: Any, **kwargs: Any) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer_name = str(getattr(self.args_task, "optimizer", "adamw")).lower()
        lr = float(getattr(self.args_task, "lr", 1e-4))
        weight_decay = float(getattr(self.args_task, "weight_decay", 1e-4))
        if optimizer_name == "adam":
            return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        if optimizer_name == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        raise ValueError(f"Unsupported optimizer for generative task: {optimizer_name}")

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
        return sample_euler_ode(
            self.network.to(sample_device),
            noise,
            expanded_condition,
            num_steps=num_steps,
        )

    def write_sample_manifest(
        self,
        *,
        output_path: str | Path,
        synthetic_dataset_id: str,
        checkpoint_path: str,
        generator_run_id: str,
        source_split: str,
        domain_map_path: str,
        domain_map_hash: str,
        sampler_id: str,
        num_steps: int,
        seed: int,
        num_samples: int,
        shape: list[int] | tuple[int, ...],
        status: str = "exploratory",
    ) -> dict[str, Any]:
        normalization = {
            "method": str(getattr(self.args_data, "normalization", "standardization")),
            "scope": "window",
        }
        params_artifact = getattr(self.args_data, "normalization_params_path", None)
        params_hash = getattr(self.args_data, "normalization_params_hash", None)
        if params_artifact and params_hash:
            normalization["params_artifact"] = str(params_artifact)
            normalization["params_hash"] = str(params_hash)

        manifest = build_synthetic_data_manifest(
            synthetic_dataset_id=synthetic_dataset_id,
            model_type=str(getattr(self.args_model, "type", "generative_model")),
            model_name=str(getattr(self.args_model, "name", "phm_cfm_mlp1d")),
            loss_id="conditional_flow_matching",
            checkpoint_path=checkpoint_path,
            generator_run_id=generator_run_id,
            source_split=source_split,
            domain_map_path=domain_map_path,
            domain_map_hash=domain_map_hash,
            normalization=normalization,
            sampler_id=sampler_id,
            num_steps=num_steps,
            seed=seed,
            num_samples=num_samples,
            shape=shape,
            status=status,
        )
        write_synthetic_data_manifest(output_path, manifest)
        return manifest


task = ConditionalFlowMatchingTask
