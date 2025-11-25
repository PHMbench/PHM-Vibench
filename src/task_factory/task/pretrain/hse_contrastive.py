"""
Simplified HSE contrastive pretraining task.

设计目标
--------
- 只做“特征级对比预训练”：
  - Prompt 已在 embedding / 模型内部融合，这里不再显式处理 prompt 向量。
  - 统一通过 ContrastiveStrategyManager 计算对比损失。
- 支持一个可选的辅助分类头（CE），用于稳定训练或简单监控。
- 保持对现有配置的兼容：
  - Experiment 2 第一阶段：`type: "pretrain", name: "hse_contrastive"`.
  - 仍然通过 `task.contrast_loss` / `contrast_weight` / `classification_weight` 控制行为。
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ...Default_task import Default_task
from ...Components.loss import get_loss_fn
from ...Components.contrastive_strategies import create_contrastive_strategy

import logging

logger = logging.getLogger(__name__)


class task(Default_task):
    """
    HSE 对比预训练任务（简化版）

    - 特征级对比学习（feature-level），不在 task 中显式处理 prompt。
    - 可选分类辅助：CE loss 作为对比损失的“辅助项”，通过 `classification_weight` 控制。
    """

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
        # 调用父类初始化（负责 optimizer / metrics 等通用配置）
        super().__init__(network, args_data, args_model, args_task, args_trainer, args_environment, metadata)

        self.args_task = args_task
        self.args_model = args_model
        self.args_data = args_data
        self.metadata = metadata

        # 对比学习权重与分类权重
        self.contrast_weight: float = float(getattr(args_task, "contrast_weight", 1.0))
        self.classification_weight: float = float(getattr(args_task, "classification_weight", 0.0))

        # 独立的分类损失（统一使用 CE），避免与 args_task.loss 冲突
        self.ce_loss_fn = get_loss_fn("CE")

        # 初始化对比策略
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
                logger.info(f"[hse_contrastive] Enabled contrastive strategy: {loss_type}")
            except Exception as exc:  # pragma: no cover - runtime safeguard
                logger.error(f"[hse_contrastive] Failed to init contrastive strategy: {exc}")
                self.strategy_manager = None
                self.contrast_weight = 0.0

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 核心 shared_step
    # ------------------------------------------------------------------
    def _shared_step(self, batch: Any, batch_idx: int, stage: str) -> Dict[str, torch.Tensor]:
        batch_dict = self._prepare_batch(batch)
        x: torch.Tensor = batch_dict["x"]
        y: torch.Tensor = batch_dict["y"]
        file_id: Any = batch_dict.get("file_id", None)
        task_id: str = batch_dict.get("task_id", "classification")

        # 1. 获取 logits 与特征向量
        logits, features = self._forward_backbone(x, file_id, task_id)

        # 2. 分类流（可选）
        classification_loss = torch.tensor(0.0, device=x.device)
        classification_acc = torch.tensor(0.0, device=x.device)
        if self.classification_weight > 0:
            classification_loss, classification_acc = self._run_classification_flow(logits, y)

        # 3. 对比流（可选）
        contrastive_loss = torch.tensor(0.0, device=x.device)
        if self.contrast_weight > 0 and self.strategy_manager is not None:
            contrastive_loss = self._run_contrastive_flow(features, y)

        # 4. 总损失
        total_loss = self.classification_weight * classification_loss + self.contrast_weight * contrastive_loss

        metrics: Dict[str, torch.Tensor] = {
            f"{stage}_total_loss": total_loss,
            f"{stage}_classification_loss": classification_loss,
            f"{stage}_contrastive_loss": contrastive_loss,
            f"{stage}_classification_weight": torch.tensor(self.classification_weight, device=x.device),
            f"{stage}_contrast_weight": torch.tensor(self.contrast_weight, device=x.device),
        }

        # 5. 简单分类指标（仅在分类流开启时记录）
        if self.classification_weight > 0:
            metrics[f"{stage}_acc"] = classification_acc

        return metrics

    # ------------------------------------------------------------------
    # 子流程：batch 解析 + backbone 特征
    # ------------------------------------------------------------------
    def _prepare_batch(self, batch: Any) -> Dict[str, Any]:
        """统一 batch 格式为 dict，至少包含 x / y / file_id / task_id。"""
        if isinstance(batch, dict):
            prepared = dict(batch)
        else:
            # 兼容 ((x, y), data_name) 格式
            (x, y), data_name = batch
            prepared = {"x": x, "y": y, "file_id": data_name}

        prepared.setdefault("task_id", "classification")
        return prepared

    def _forward_backbone(
        self, x: torch.Tensor, file_id: Any, task_id: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        统一从网络获取 logits 和特征向量。
        优先调用 return_feature=True；若不支持，则使用 logits 作为特征。
        """
        network_kwargs = {"file_id": file_id, "task_id": task_id}

        # 1) 尝试带 return_feature 的调用
        logits: torch.Tensor
        features: torch.Tensor
        try:
            output = self.network(x, return_feature=True, **network_kwargs)
        except TypeError:
            try:
                output = self.network(x, return_feature=True)
            except TypeError:
                # 最终回退：只拿 logits，用 logits 作为特征
                logits = self.network(x)
                features = logits
                return logits, self._flatten_features(features)

        # 处理不同输出格式
        if isinstance(output, tuple):
            # 约定：第一个是 logits，第二个是特征
            if len(output) >= 2:
                logits, features = output[0], output[1]
            else:
                logits, features = output[0], output[0]
        else:
            logits, features = output, output

        return logits, self._flatten_features(features)

    @staticmethod
    def _flatten_features(features: torch.Tensor) -> torch.Tensor:
        """将特征展平到 [B, D] 形式（必要时对 patch 维度做 mean pooling）。"""
        if features.ndim > 2:
            # 例如 [B, T, D] 或 [B, C, L] -> 对第二维求均值
            features = features.mean(dim=1)
        return features

    # ------------------------------------------------------------------
    # 子流程：分类与对比损失
    # ------------------------------------------------------------------
    def _run_classification_flow(
        self, logits: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用 CE 计算分类损失，并返回简单准确率。"""
        loss = self.ce_loss_fn(logits, y.long() if y.dtype != torch.long else y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        return loss, acc

    def _run_contrastive_flow(self, features: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """构造两视图特征并调用策略管理器计算对比损失。"""
        if self.strategy_manager is None:
            return torch.tensor(0.0, device=features.device)

        # z1: 原始特征；z2: 简单增强视图
        z1 = features
        z2 = self._create_augmented_view(features)

        # 拼接视图以兼容 InfoNCE / SupCon 等实现
        z = torch.cat([z1, z2], dim=0)

        # 部分损失需要标签
        requires_labels = getattr(self.strategy_manager, "requires_labels", False)
        labels_ext: Optional[torch.Tensor] = None
        if requires_labels:
            labels_ext = torch.cat([y, y], dim=0)

        try:
            result = self.strategy_manager.compute_loss(
                features=z,
                projections=z,
                prompts=None,
                labels=labels_ext,
                system_ids=None,
            )
            return result["loss"]
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logger.error(f"[hse_contrastive] Contrastive loss computation failed: {exc}")
            return torch.tensor(0.0, device=features.device)

    def _create_augmented_view(self, features: torch.Tensor) -> torch.Tensor:
        """简单的特征级增强：加高斯噪声。"""
        noise_std = float(getattr(self.args_task, "augmentation_noise_std", 0.1))
        noise = torch.randn_like(features) * noise_std
        return features + noise

    # ------------------------------------------------------------------
    # Logging 简化
    # ------------------------------------------------------------------
    def _log_simple_metrics(self, metrics: Dict[str, torch.Tensor], stage: str) -> None:
        """使用 Lightning 的 self.log 接口记录少量关键指标。"""
        for k, v in metrics.items():
            if not k.startswith(stage):
                continue
            on_step = stage == "train"
            # 只在进度条上显示总 loss
            prog_bar = k.endswith("total_loss")
            self.log(
                k,
                v,
                on_step=on_step,
                on_epoch=True,
                prog_bar=prog_bar,
                logger=True,
                sync_dist=True,
            )

        # 兼容性别名：为val_total_loss提供val_loss别名
        if stage == "val":
            total_loss_key = f"{stage}_total_loss"
            if total_loss_key in metrics:
                self.log("val_loss", metrics[total_loss_key],
                        on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    # 仍然使用 Default_task 的优化器配置逻辑
    def configure_optimizers(self):
        return super().configure_optimizers()


# Alias for backward compatibility and registration
HseContrastiveTask = task
HSEContrastiveTask = task

