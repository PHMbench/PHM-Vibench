"""Experimental univariate PPT time-order pretraining task."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn

from ... import register_task
from ...Components.ppt_time_order import PPTTimeOrderConfig, PPTTimeOrderLoss


@register_task("pretrain", "ppt_time_order")
class PptTimeOrderTask(pl.LightningModule):
    """Train E_03_Patch + B_08_PatchTST with PPT's time-order objectives only."""

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
            raise ValueError("ppt_time_order requires model.embedding=E_03_Patch")
        if getattr(args_model, "backbone", None) != "B_08_PatchTST":
            raise ValueError("ppt_time_order requires model.backbone=B_08_PatchTST")
        if int(getattr(args_model, "input_dim", 0)) != 1:
            raise ValueError("the initial ppt_time_order task supports one input channel only")
        if not hasattr(network, "encode_sequence"):
            raise ValueError("ppt_time_order requires a model.encode_sequence(x, file_id) contract")

        data_window = int(getattr(args_data, "window_size", 0))
        model_window = int(getattr(args_model, "window_size", 0))
        patch_size = int(getattr(args_model, "patch_size_L", 0))
        num_patches = int(getattr(args_model, "num_patches", 0))
        if data_window <= 0 or model_window != data_window:
            raise ValueError("model.window_size must equal the positive data.window_size")
        if patch_size <= 0 or data_window % patch_size != 0:
            raise ValueError("data.window_size must be divisible by model.patch_size_L")
        if num_patches != data_window // patch_size:
            raise ValueError("model.num_patches must equal window_size // patch_size_L")

        self.objective = PPTTimeOrderLoss(
            PPTTimeOrderConfig(
                num_patches=num_patches,
                embedding_dim=int(getattr(args_model, "output_dim", 0)),
                weak_swaps=int(getattr(args_task, "weak_swaps", 1)),
                strong_swaps=int(getattr(args_task, "strong_swaps", 5)),
                bank_size=int(getattr(args_task, "permutation_bank_size", 256)),
                seed=int(getattr(args_task, "permutation_seed", 42)),
                temperature=float(getattr(args_task, "temperature", 0.1)),
                consistency_weight=float(getattr(args_task, "consistency_weight", 1.0)),
                contrastive_weight=float(getattr(args_task, "contrastive_weight", 1.0)),
            )
        )

    def forward(self, x: torch.Tensor, file_id: Any = None) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 1:
            raise ValueError(
                "ppt_time_order expects univariate input shaped [batch, length, 1]"
            )
        sequence = self.network.encode_sequence(x, file_id=file_id)
        if sequence.ndim != 3:
            raise RuntimeError("model.encode_sequence must return [batch, patches, embedding_dim]")
        return sequence

    def _shared_step(self, batch: dict[str, Any], batch_idx: int, stage: str):
        if not isinstance(batch, dict) or "x" not in batch:
            raise TypeError("ppt_time_order requires a dict batch containing 'x'")
        x = batch["x"]
        sequence = self.forward(x, file_id=batch.get("file_id"))
        offset = int(self.current_epoch) * 1_000_003 + batch_idx * x.shape[0]
        loss, raw_stats = self.objective(sequence, offset=offset)
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
        return loss, stats

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
        raise ValueError(f"unsupported ppt_time_order optimizer {name!r}")
