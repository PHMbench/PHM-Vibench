"""HSE feature-level contrastive pretraining task."""

import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from ...Components.contrastive_strategies import create_contrastive_strategy
from ...Components.loss import get_loss_fn
from ...Default_task import Default_task


logger = logging.getLogger(__name__)


class task(Default_task):
    """Feature-level HSE contrastive learning with an optional CE objective."""

    def __init__(
        self,
        network,
        args_data,
        args_model,
        args_task,
        args_trainer,
        args_environment,
        metadata,
    ):
        super().__init__(
            network,
            args_data,
            args_model,
            args_task,
            args_trainer,
            args_environment,
            metadata,
        )

        self.args_task = args_task
        self.args_model = args_model
        self.args_data = args_data
        self.metadata = metadata

        self.contrast_weight = self._validated_weight(
            getattr(args_task, "contrast_weight", 1.0),
            "contrast_weight",
        )
        self.classification_weight = self._validated_weight(
            getattr(args_task, "classification_weight", 0.0),
            "classification_weight",
        )
        if self.contrast_weight == 0.0 and self.classification_weight == 0.0:
            raise ValueError(
                "hse_contrastive requires at least one positive objective weight: "
                "set task.contrast_weight or task.classification_weight above zero."
            )

        self.ce_loss_fn = get_loss_fn("CE")
        self.strategy_manager = None
        if self.contrast_weight > 0:
            loss_type = getattr(args_task, "contrast_loss", "INFONCE")
            contrastive_config = {
                "type": "single",
                "loss_type": loss_type,
                "temperature": getattr(args_task, "temperature", 0.07),
                "margin": getattr(args_task, "margin", 0.3),
                "barlow_lambda": getattr(args_task, "barlow_lambda", 5e-3),
            }
            try:
                self.strategy_manager = create_contrastive_strategy(contrastive_config)
            except Exception as exc:
                raise RuntimeError(
                    "Unable to initialize the configured HSE contrastive objective "
                    f"{loss_type!r}. Check task.contrast_loss and its parameters."
                ) from exc
            logger.info("[hse_contrastive] Enabled contrastive strategy: %s", loss_type)

    @staticmethod
    def _validated_weight(value: Any, name: str) -> float:
        weight = float(value)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"task.{name} must be a finite non-negative number, got {value!r}.")
        return weight

    @staticmethod
    def _require_valid_loss(loss: torch.Tensor, name: str, stage: str) -> None:
        if not torch.is_tensor(loss):
            raise TypeError(f"{name} must return a torch.Tensor, got {type(loss).__name__}.")
        if loss.numel() != 1:
            raise ValueError(f"{name} must return one scalar loss, got shape {tuple(loss.shape)}.")
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"{name} produced a non-finite loss during {stage}.")
        if stage == "train" and not loss.requires_grad:
            raise RuntimeError(
                f"{name} is enabled but its training loss does not require gradients."
            )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        metrics = self._shared_step(batch, batch_idx, stage="train")
        self._log_simple_metrics(metrics, stage="train")
        return metrics["train_total_loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        metrics = self._shared_step(batch, batch_idx, stage="val")
        self._log_simple_metrics(metrics, stage="val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        metrics = self._shared_step(batch, batch_idx, stage="test")
        self._log_simple_metrics(metrics, stage="test")

    def _shared_step(
        self,
        batch: Any,
        batch_idx: int,
        stage: str,
    ) -> Dict[str, torch.Tensor]:
        batch_dict = self._prepare_batch(batch)
        x: torch.Tensor = batch_dict["x"]
        y: torch.Tensor = batch_dict["y"]
        file_id: Any = batch_dict.get("file_id")
        task_id: str = batch_dict.get("task_id", "classification")

        system_ids: List[int] = []
        if self.classification_weight > 0:
            system_ids = self._infer_system_ids(file_id)
            if len(system_ids) > 1:
                raise ValueError(
                    "hse_contrastive classification requires one Dataset_id per batch, "
                    f"but batch_idx={batch_idx} in {stage} contains {system_ids}."
                )

        logits, features = self._forward_backbone(x, file_id, task_id)

        classification_loss = x.new_zeros(())
        classification_acc = x.new_zeros(())
        if self.classification_weight > 0:
            classification_loss, classification_acc = self._run_classification_flow(
                logits,
                y,
                system_ids=system_ids,
            )
            self._require_valid_loss(classification_loss, "classification objective", stage)

        contrastive_loss = x.new_zeros(())
        if self.contrast_weight > 0:
            contrastive_loss = self._run_contrastive_flow(features, y)
            self._require_valid_loss(contrastive_loss, "contrastive objective", stage)

        total_loss = (
            self.classification_weight * classification_loss
            + self.contrast_weight * contrastive_loss
        )
        self._require_valid_loss(total_loss, "total HSE objective", stage)

        metrics: Dict[str, torch.Tensor] = {
            f"{stage}_total_loss": total_loss,
            f"{stage}_classification_loss": classification_loss,
            f"{stage}_contrastive_loss": contrastive_loss,
            f"{stage}_classification_weight": torch.tensor(
                self.classification_weight,
                device=x.device,
            ),
            f"{stage}_contrast_weight": torch.tensor(
                self.contrast_weight,
                device=x.device,
            ),
        }
        if self.classification_weight > 0:
            metrics[f"{stage}_acc"] = classification_acc
        return metrics

    def _infer_system_ids(self, file_id: Any) -> List[int]:
        """Resolve the unique Dataset IDs for a classification batch."""
        if file_id is None:
            raise ValueError("hse_contrastive classification requires batch['file_id'].")
        if self.metadata is None:
            raise ValueError("hse_contrastive classification requires metadata.")

        if isinstance(file_id, torch.Tensor):
            ids_iter = [value.item() for value in file_id.view(-1)]
        elif isinstance(file_id, (list, tuple)):
            ids_iter = list(file_id)
        else:
            ids_iter = [file_id]

        if not ids_iter:
            raise ValueError("batch['file_id'] must contain at least one ID.")

        system_ids: List[int] = []
        for fid in ids_iter:
            try:
                sid = self.metadata[fid]["Dataset_id"]
            except (KeyError, IndexError, TypeError) as exc:
                raise KeyError(
                    f"Unable to resolve Dataset_id metadata for file_id={fid!r}."
                ) from exc
            if hasattr(sid, "iloc"):
                sid = sid.iloc[0]
            system_ids.append(int(sid))
        return sorted(set(system_ids))

    def _prepare_batch(self, batch: Any) -> Dict[str, Any]:
        if isinstance(batch, dict):
            prepared = dict(batch)
        else:
            (x, y), data_name = batch
            prepared = {"x": x, "y": y, "file_id": data_name}
        prepared.setdefault("task_id", "classification")
        return prepared

    def _forward_backbone(
        self,
        x: torch.Tensor,
        file_id: Any,
        task_id: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.network(
            x,
            file_id=file_id,
            task_id=task_id,
            return_feature=True,
        )
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise TypeError(
                "hse_contrastive requires model.forward(..., return_feature=True) "
                "to return (logits, features)."
            )
        logits, features = output[0], output[1]
        if not torch.is_tensor(logits) or not torch.is_tensor(features):
            raise TypeError("hse_contrastive model outputs must both be torch.Tensor values.")
        return logits, self._flatten_features(features)

    @staticmethod
    def _flatten_features(features: torch.Tensor) -> torch.Tensor:
        if features.ndim < 2:
            raise ValueError(
                "hse_contrastive features must include batch and feature dimensions, "
                f"got shape {tuple(features.shape)}."
            )
        if features.ndim > 2:
            features = features.mean(dim=1)
        return features

    def _run_classification_flow(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        system_ids: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(logits) or not torch.is_tensor(y):
            raise TypeError("classification logits and labels must be torch.Tensor values.")
        if logits.ndim != 2:
            raise ValueError(
                f"classification logits must have shape [B, C], got {tuple(logits.shape)}."
            )
        if y.ndim != 1:
            raise ValueError(f"classification labels must have shape [B], got {tuple(y.shape)}.")
        if logits.device != y.device:
            y = y.to(logits.device)
        if logits.shape[0] != y.shape[0]:
            raise ValueError(
                "classification batch size mismatch: "
                f"logits={logits.shape[0]}, labels={y.shape[0]}."
            )
        if not torch.isfinite(logits).all():
            raise FloatingPointError("classification logits contain NaN or Inf values.")
        if not torch.isfinite(y).all():
            raise FloatingPointError("classification labels contain NaN or Inf values.")

        y = y.long()
        num_classes = logits.shape[1]
        if num_classes <= 0:
            raise ValueError("classification logits must contain at least one class.")
        y_min = int(y.min().item())
        y_max = int(y.max().item())
        if y_min < 0 or y_max >= num_classes:
            suffix = "" if not system_ids else f"; system_ids={system_ids}"
            raise ValueError(
                "classification labels are outside the logits class range: "
                f"received [{y_min}, {y_max}], expected [0, {num_classes - 1}]"
                f"{suffix}."
            )

        loss = self.ce_loss_fn(logits, y)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y).float().mean()
        return loss, accuracy

    def _run_contrastive_flow(
        self,
        features: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        if self.strategy_manager is None:
            raise RuntimeError(
                "The contrastive objective is enabled but no strategy was initialized."
            )
        if not torch.is_tensor(features) or not torch.is_tensor(y):
            raise TypeError("contrastive features and labels must be torch.Tensor values.")

        features = self._flatten_features(features)
        target_device = features.device
        if y.device != target_device:
            y = y.to(target_device)
        if features.shape[0] != y.shape[0]:
            raise ValueError(
                "contrastive batch size mismatch: "
                f"features={features.shape[0]}, labels={y.shape[0]}."
            )
        if not torch.isfinite(features).all():
            raise FloatingPointError("contrastive features contain NaN or Inf values.")
        if not torch.isfinite(y).all():
            raise FloatingPointError("contrastive labels contain NaN or Inf values.")

        labels_ext: Optional[torch.Tensor] = None
        if getattr(self.strategy_manager, "requires_labels", False):
            if y.ndim != 1:
                raise ValueError(
                    f"label-aware contrastive loss requires 1D labels, got {y.ndim}D."
                )
            y = y.long()
            if int(y.min().item()) < 0:
                raise ValueError(
                    f"contrastive labels must be non-negative, got minimum {int(y.min().item())}."
                )
            labels_ext = torch.cat([y, y], dim=0)

        z1 = features
        z2 = self._create_augmented_view(features)
        z = torch.cat([z1, z2], dim=0)

        result = self.strategy_manager.compute_loss(
            features=z,
            projections=z,
            prompts=None,
            labels=labels_ext,
            system_ids=None,
        )
        if not isinstance(result, Mapping) or "loss" not in result:
            raise TypeError(
                "The contrastive strategy must return a mapping containing a 'loss' tensor."
            )
        return result["loss"]

    def _create_augmented_view(self, features: torch.Tensor) -> torch.Tensor:
        aug_type = str(getattr(self.args_task, "augmentation_type", "noise")).lower()
        allowed = {"none", "noise", "scaling", "dropout", "mixed"}
        if aug_type not in allowed:
            raise ValueError(
                f"Unknown task.augmentation_type {aug_type!r}. "
                f"Available values: {', '.join(sorted(allowed))}."
            )

        noise_std = float(getattr(self.args_task, "augmentation_noise_std", 0.1))
        dropout_p = float(getattr(self.args_task, "augmentation_dropout_p", 0.1))
        scale_std = float(getattr(self.args_task, "augmentation_scale_std", 0.1))
        if not math.isfinite(noise_std) or noise_std < 0:
            raise ValueError("task.augmentation_noise_std must be finite and non-negative.")
        if not math.isfinite(scale_std) or scale_std < 0:
            raise ValueError("task.augmentation_scale_std must be finite and non-negative.")
        if not math.isfinite(dropout_p) or not 0 <= dropout_p < 1:
            raise ValueError("task.augmentation_dropout_p must satisfy 0 <= p < 1.")

        if aug_type == "none":
            augmented = features.clone()
        else:
            if aug_type == "mixed":
                candidates = ("noise", "scaling", "dropout")
                index = torch.randint(len(candidates), (1,), device=features.device).item()
                aug_type = candidates[index]

            if aug_type == "dropout":
                if dropout_p == 0:
                    augmented = features.clone()
                else:
                    mask = (torch.rand_like(features) >= dropout_p).to(features.dtype)
                    augmented = features * mask
            elif aug_type == "scaling":
                if scale_std == 0:
                    augmented = features.clone()
                else:
                    scale = 1.0 + torch.randn_like(features) * scale_std
                    augmented = features * scale
            else:
                if noise_std == 0:
                    augmented = features.clone()
                else:
                    augmented = features + torch.randn_like(features) * noise_std

        if not torch.isfinite(augmented).all():
            raise FloatingPointError("HSE augmentation produced NaN or Inf values.")
        return augmented

    def _log_simple_metrics(
        self,
        metrics: Dict[str, torch.Tensor],
        stage: str,
    ) -> None:
        for key, value in metrics.items():
            if not key.startswith(stage):
                continue
            self.log(
                key,
                value,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=key.endswith("total_loss"),
                logger=True,
                sync_dist=True,
            )

        if stage == "val":
            total_loss_key = f"{stage}_total_loss"
            if total_loss_key in metrics:
                self.log(
                    "val_loss",
                    metrics[total_loss_key],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )

    def configure_optimizers(self):
        return super().configure_optimizers()


HseContrastiveTask = task
HSEContrastiveTask = task
