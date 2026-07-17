"""Experimental multivariate PPT pretraining and supervised task."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import register_task
from ...Components.ppt_time_order import PPTOrderConfig, PPTOrderLoss


@register_task("pretrain", "ppt_order")
class PptOrderTask(pl.LightningModule):
    """Apply time/channel patch-order objectives to channel-independent ISFM grids."""

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
        del args_trainer, args_environment, metadata
        self.network = network
        self.args_task = args_task

        if getattr(args_model, "embedding", None) != "E_03_Patch":
            raise ValueError("ppt_order requires model.embedding=E_03_Patch")
        if getattr(args_model, "backbone", None) != "B_08_PatchTST":
            raise ValueError("ppt_order requires model.backbone=B_08_PatchTST")
        if not bool(getattr(args_model, "channel_independent", False)):
            raise ValueError("ppt_order requires model.channel_independent=true")
        if not hasattr(network, "encode_patch_grid"):
            raise ValueError("ppt_order requires model.encode_patch_grid(x, file_id)")

        ppt = getattr(args_task, "ppt", None)
        if ppt is None:
            raise ValueError("ppt_order requires task.ppt configuration")
        get = (
            ppt.get
            if isinstance(ppt, dict)
            else lambda key, default=None: getattr(ppt, key, default)
        )
        self.mode = str(get("mode", "ssl"))
        if self.mode not in {"ssl", "supervised"}:
            raise ValueError("task.ppt.mode must be ssl or supervised")
        self.weighting = str(get("weighting", "fixed"))
        self.classification_weight = float(get("classification_weight", 1.0))
        if self.classification_weight < 0.0:
            raise ValueError("task.ppt.classification_weight must be non-negative")
        if self.mode == "supervised" and not hasattr(network, "classify_encoded"):
            raise ValueError("supervised ppt_order requires model.classify_encoded")

        data_window = int(getattr(args_data, "window_size", 0))
        model_window = int(getattr(args_model, "window_size", 0))
        patch_size = int(getattr(args_model, "patch_size_L", 0))
        num_patches = int(getattr(args_model, "num_patches", 0))
        num_channels = int(getattr(args_model, "input_dim", 0))
        if data_window <= 0 or model_window != data_window:
            raise ValueError("model.window_size must equal the positive data.window_size")
        if patch_size <= 0 or data_window % patch_size != 0:
            raise ValueError("data.window_size must be divisible by model.patch_size_L")
        if num_patches != data_window // patch_size:
            raise ValueError("model.num_patches must equal window_size // patch_size_L")

        axes = tuple(get("order_axes", ["time", "channel"]))
        self.objective = PPTOrderLoss(
            PPTOrderConfig(
                num_patches=num_patches,
                num_channels=num_channels,
                embedding_dim=int(getattr(args_model, "output_dim", 0)),
                order_axes=axes,
                weak_swaps=int(get("weak_swaps", 1)),
                strong_swaps=int(get("strong_swaps", 5)),
                channel_weak_swaps=int(get("channel_weak_swaps", 1)),
                channel_strong_swaps=int(get("channel_strong_swaps", 2)),
                bank_size=int(get("permutation_bank_size", 256)),
                seed=int(get("permutation_seed", 42)),
                temperature=float(get("temperature", 0.1)),
                consistency_weight=float(get("consistency_weight", 1.0)),
                contrastive_weight=float(get("contrastive_weight", 1.0)),
                weighting=self.weighting,
            )
        )
        self.classification_log_variance = None
        if self.mode == "supervised" and self.weighting == "uncertainty":
            self.classification_log_variance = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, file_id: Any = None) -> torch.Tensor:
        grid = self.network.encode_patch_grid(x, file_id=file_id)
        if grid.ndim != 4:
            raise RuntimeError(
                "model.encode_patch_grid must return [batch, channels, patches, dim]"
            )
        return grid

    def _shared_step(self, batch: dict[str, Any], batch_idx: int, stage: str):
        if not isinstance(batch, dict) or "x" not in batch:
            raise TypeError("ppt_order requires a dict batch containing 'x'")
        x = batch["x"]
        file_id = batch.get("file_id")
        grid = self.forward(x, file_id=file_id)
        offset = int(self.current_epoch) * 1_000_003 + batch_idx * x.shape[0]
        total, raw_stats = self.objective(grid, offset=offset)

        if self.mode == "supervised":
            if "y" not in batch:
                raise ValueError("supervised ppt_order requires batch['y']")
            sequence = grid.mean(dim=1)
            logits = self.network.classify_encoded(sequence, file_id=file_id)
            classification = F.cross_entropy(logits, batch["y"].long())
            if self.weighting == "fixed":
                total = total + self.classification_weight * classification
            else:
                total = total + (
                    torch.exp(-self.classification_log_variance) * classification
                    + self.classification_log_variance
                )
            raw_stats["classification_loss"] = classification

        raw_stats["loss"] = total
        stats = {f"{stage}_{name}": value for name, value in raw_stats.items()}
        self.log_dict(
            stats,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=x.shape[0],
        )
        return total, stats

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _ = self._shared_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        name = str(getattr(self.args_task, "optimizer", "adamw")).lower()
        lr = float(getattr(self.args_task, "lr", 1e-3))
        weight_decay = float(getattr(self.args_task, "weight_decay", 0.0))
        if name == "adam":
            return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        if name == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        if name == "sgd":
            momentum = float(getattr(self.args_task, "momentum", 0.9))
            return torch.optim.SGD(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
            )
        raise ValueError(f"unsupported ppt_order optimizer {name!r}")
