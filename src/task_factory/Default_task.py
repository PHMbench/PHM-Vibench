from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.task_factory import register_task

from .Components.gradient_constraints import FisherGradientConstraint
from .Components.loss import get_loss_fn
from .Components.metrics import get_metrics
from .Components.regularization import calculate_regularization


@register_task("Default_task", "Default_task")
class Default_task(pl.LightningModule):
    """General Lightning task with explicit objective and metric semantics."""

    def __init__(
        self,
        network: nn.Module,
        args_data: Any,
        args_model: Any,
        args_task: Any,
        args_trainer: Any,
        args_environment: Any,
        metadata: Any,
    ):
        super().__init__()

        # Device placement belongs exclusively to Trainer Factory. Task
        # construction preserves the model returned by Model Factory.
        self.network = network
        self.args_task = args_task
        self.args_model = args_model
        self.args_data = args_data
        self.metadata = metadata
        self.args_trainer = args_trainer
        self.args_environment = args_environment

        configured_task_id = getattr(
            args_task,
            "model_task_id",
            getattr(args_task, "name", None),
        )
        if not isinstance(configured_task_id, str) or not configured_task_id.strip():
            raise ValueError(
                "task.model_task_id or task.name must explicitly identify the "
                "model task head"
            )
        self.model_task_id = configured_task_id.strip()

        gradient_constraint = getattr(self.args_task, "gradient_constraint", None)
        self.gradient_constraint = None
        if gradient_constraint:
            if isinstance(gradient_constraint, dict):
                constraint_name = gradient_constraint.get("name")
                epsilon = gradient_constraint.get("epsilon", 2.0)
            else:
                constraint_name = getattr(gradient_constraint, "name", None)
                epsilon = getattr(gradient_constraint, "epsilon", 2.0)
            if str(constraint_name).lower() != "fic":
                raise ValueError(
                    f"unsupported task.gradient_constraint.name {constraint_name!r}"
                )
            if str(getattr(self.args_task, "loss", "")).upper() != "CE":
                raise ValueError("FIC gradient_constraint currently requires task.loss=CE")
            self.gradient_constraint = FisherGradientConstraint(epsilon=float(epsilon))

        self.loss_fn = get_loss_fn(self.args_task.loss)
        self.metrics = get_metrics(self.args_task.metrics, self.metadata)

        hparams_dict = {
            **vars(self.args_task),
            **vars(self.args_model),
            **vars(self.args_data),
            **vars(self.args_trainer),
            **vars(self.args_environment),
        }
        self.save_hyperparameters(hparams_dict, ignore=["network", "metadata"])

    def _resolve_model_task_id(self, batch: Mapping[str, Any]) -> str:
        """Return the configured task head and reject batch-level overrides."""

        if "task_id" not in batch:
            return self.model_task_id

        raw_task_id = batch["task_id"]
        if isinstance(raw_task_id, str):
            observed = [raw_task_id]
        elif isinstance(raw_task_id, (list, tuple)):
            observed = list(raw_task_id)
        else:
            raise TypeError(
                "batch['task_id'] must be a string or a sequence of strings, "
                f"got {type(raw_task_id).__name__}"
            )

        if not observed or any(not isinstance(value, str) for value in observed):
            raise TypeError("batch['task_id'] must contain non-empty string values")
        unique = {value.strip() for value in observed if value.strip()}
        if len(unique) != 1:
            raise ValueError(
                f"one batch cannot mix model task IDs, observed={sorted(unique)}"
            )
        observed_task_id = next(iter(unique))
        if observed_task_id != self.model_task_id:
            raise ValueError(
                "batch task identity conflicts with the configured Task Factory "
                f"semantics: configured={self.model_task_id!r}, "
                f"observed={observed_task_id!r}"
            )
        return self.model_task_id

    def forward(self, batch):
        """Forward one batch through the explicitly configured model task head."""

        if not isinstance(batch, Mapping):
            raise TypeError(f"task batch must be a mapping, got {type(batch).__name__}")
        x = batch["x"]
        file_id = batch["file_id"]
        task_id = self._resolve_model_task_id(batch)
        if getattr(self.network, "requires_physical_metadata", False):
            canonical_fields = (
                "sample_rate_hz",
                "rotation_speed_rpm",
                "load_hp",
            )
            explicit_metadata = {
                field: batch[field] for field in canonical_fields if field in batch
            }
            return self.network(
                x,
                file_id,
                task_id,
                physical_metadata=explicit_metadata or None,
            )
        return self.network(x, file_id, task_id)

    def _compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(y_hat, y.long() if y.dtype != torch.long else y)

    def _compute_metrics(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        data_name: str,
        stage: str,
    ) -> Dict[str, torch.Tensor]:
        if data_name not in self.metrics:
            available = sorted(self.metrics.keys())
            raise KeyError(
                f"No metric lifecycle exists for dataset Name={data_name!r}; "
                f"available={available}. The batch cannot be evaluated with a "
                "different dataset's metrics."
            )

        metric_values = {}
        for metric_key, metric_fn in self.metrics[data_name].items():
            if metric_key.startswith(stage):
                metric_values[f"{metric_key}_{data_name}"] = metric_fn(y_hat, y)
        return metric_values

    def _compute_regularization(self) -> Dict[str, torch.Tensor]:
        return calculate_regularization(
            getattr(self.args_task, "regularization", {}),
            self.parameters(),
        )

    @staticmethod
    def _file_id_values(raw_file_ids: Any) -> list[Any]:
        if isinstance(raw_file_ids, torch.Tensor):
            return [value.item() for value in raw_file_ids.view(-1)]
        if isinstance(raw_file_ids, (list, tuple)):
            return list(raw_file_ids)
        return [raw_file_ids]

    def _resolve_batch_identity(
        self,
        batch: Mapping[str, Any],
        batch_size: int,
    ) -> tuple[str, Any, list[Any]]:
        """Require one metadata dataset identity for the complete batch."""

        file_ids = self._file_id_values(batch["file_id"])
        if len(file_ids) not in (1, batch_size):
            raise ValueError(
                "batch['file_id'] must contain one ID or one ID per sample: "
                f"received {len(file_ids)} IDs for batch_size={batch_size}."
            )

        names: list[Any] = []
        dataset_ids: list[Any] = []
        for current_file_id in file_ids:
            try:
                row = self.metadata[current_file_id]
                names.append(row["Name"])
                dataset_ids.append(row["Dataset_id"])
            except (KeyError, IndexError, TypeError) as exc:
                raise KeyError(
                    "Unable to resolve metadata Name and Dataset_id for "
                    f"file_id={current_file_id!r}."
                ) from exc

        unique_names = {str(name) for name in names}
        unique_dataset_ids = {str(dataset_id) for dataset_id in dataset_ids}
        if len(unique_names) != 1 or len(unique_dataset_ids) != 1:
            raise ValueError(
                "one batch cannot mix dataset identities because model heads and "
                "metric states would be ambiguous: "
                f"Names={sorted(unique_names)}, "
                f"Dataset_ids={sorted(unique_dataset_ids)}. Use a "
                "dataset-homogeneous sampler."
            )
        return str(names[0]), dataset_ids[0], file_ids

    def _shared_step(
        self,
        batch: Tuple,
        stage: str,
        task_id=False,
    ) -> Dict[str, torch.Tensor]:
        del task_id
        if not isinstance(batch, Mapping):
            raise TypeError(f"task batch must be a mapping, got {type(batch).__name__}")

        batch_size = int(batch["x"].shape[0])
        data_name, _, _ = Default_task._resolve_batch_identity(
            self,
            batch,
            batch_size,
        )

        # Preserve the original per-sample file IDs. The task does not rewrite
        # the batch or inject an implicit classification task.
        y_hat = self.forward(batch)

        y = batch["y"]
        loss = self._compute_loss(y_hat, y)
        y_argmax = torch.argmax(y_hat, dim=1) if y_hat.ndim > 1 else y_hat

        step_metrics = {
            f"{stage}_loss": loss,
            f"{stage}_{data_name}_loss": loss,
        }
        step_metrics.update(
            self._compute_metrics(y_argmax, y, data_name, stage)
        )

        reg_dict = self._compute_regularization()
        for reg_type, reg_loss_val in reg_dict.items():
            if reg_type != "total":
                step_metrics[f"{stage}_{reg_type}_reg_loss"] = reg_loss_val

        model_auxiliary = {}
        consume_auxiliary = getattr(self.network, "consume_auxiliary_losses", None)
        if callable(consume_auxiliary):
            model_auxiliary = consume_auxiliary()
            if not isinstance(model_auxiliary, dict):
                raise TypeError("network.consume_auxiliary_losses() must return a dict")
            for name, value in model_auxiliary.items():
                if not isinstance(value, torch.Tensor) or value.ndim != 0:
                    raise ValueError(
                        f"model auxiliary loss {name!r} must be a scalar tensor"
                    )
                if not torch.isfinite(value):
                    raise ValueError(f"model auxiliary loss {name!r} is not finite")
                step_metrics[f"{stage}_{name}_loss"] = value

        auxiliary_total = sum(
            model_auxiliary.values(),
            torch.tensor(0.0, device=loss.device),
        )
        total_loss = (
            loss
            + reg_dict.get("total", torch.tensor(0.0, device=loss.device))
            + auxiliary_total
        )
        step_metrics[f"{stage}_total_loss"] = total_loss
        return step_metrics

    def training_step(self, batch: dict, *args, **kwargs) -> torch.Tensor:
        metrics = self._shared_step(batch, "train")
        self._log_metrics(metrics, "train")
        return metrics["train_total_loss"]

    def validation_step(self, batch: dict, *args, **kwargs) -> None:
        metrics = self._shared_step(batch, "val")
        self._log_metrics(metrics, "val")

    def test_step(self, batch: dict, *args, **kwargs) -> None:
        metrics = self._shared_step(batch, "test")
        self._log_metrics(metrics, "test")

    def _log_metrics(self, metrics: Dict[str, torch.Tensor], stage: str) -> None:
        log_dict = {
            key: value
            for key, value in metrics.items()
            if key.startswith(stage) and "batch_size" not in key
        }
        self.log_dict(
            log_dict,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer_name = self.args_task.optimizer.lower()
        lr = self.args_task.lr
        weight_decay = getattr(self.args_task, "weight_decay", 0.0)

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            momentum = getattr(self.args_task, "momentum", 0.9)
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")

        scheduler_config = getattr(self.args_task, "scheduler", None)
        if (
            not scheduler_config
            or not isinstance(scheduler_config, dict)
            or not scheduler_config.get("name")
        ):
            return optimizer

        scheduler_name = scheduler_config["name"].lower()
        scheduler_options = scheduler_config.get("options", {})

        if scheduler_name == "reduceonplateau":
            monitor_metric = getattr(self.args_task, "monitor", "val_total_loss")
            patience = scheduler_options.get(
                "patience",
                getattr(self.args_task, "patience", 10) // 2
                if hasattr(self.args_task, "patience")
                else 5,
            )
            factor = scheduler_options.get("factor", 0.1)
            mode = scheduler_options.get("mode", "min")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor_metric,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        if scheduler_name == "cosine":
            max_epochs = getattr(self.trainer, "max_epochs", None) or getattr(
                self.args_task,
                "max_epochs",
                100,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_options.get("T_max", max_epochs),
                eta_min=scheduler_options.get("eta_min", 0),
            )
        elif scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_options.get("step_size", 10),
                gamma=scheduler_options.get("gamma", 0.1),
            )
        else:
            raise ValueError(f"不支持的调度器: {scheduler_name}")

        return [optimizer], [
            {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        ]

    def on_before_optimizer_step(self, optimizer) -> None:
        del optimizer
        if self.gradient_constraint is None:
            return
        result = self.gradient_constraint.apply(self.parameters())
        self.log(
            "train_fic_norm",
            result.norm,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train_fic_scale",
            result.scale,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
