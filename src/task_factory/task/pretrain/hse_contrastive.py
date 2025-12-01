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

from typing import Any, Dict, Optional, Tuple, List

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

         # 尝试解析当前 batch 对应的 system_id（仅用于调试/日志）
        system_ids: List[int] = self._infer_system_ids(file_id)
        if len(system_ids) > 1:
            logger.error(
                f"[hse_contrastive] Multi-system batch detected in {stage} "
                f"stage (batch_idx={batch_idx}): system_ids={system_ids}. "
                "Classification head assumes single system per batch."
            )

        # 1. 获取 logits 与特征向量
        logits, features = self._forward_backbone(x, file_id, task_id)

        # 2. 分类流（可选）
        classification_loss = torch.tensor(0.0, device=x.device)
        classification_acc = torch.tensor(0.0, device=x.device)
        if self.classification_weight > 0:
            classification_loss, classification_acc = self._run_classification_flow(
                logits, y, system_ids=system_ids
            )

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
    def _infer_system_ids(self, file_id: Any) -> List[int]:
        """根据 file_id 从 metadata 推断当前 batch 的 system_id 列表（仅用于调试）。

        设计为“尽量不报错”，解析失败时返回空列表。
        """
        if file_id is None or self.metadata is None:
            return []

        try:
            import pandas as pd  # 局部导入避免不必要依赖

            # DataLoader 默认会把多个 file_id 组成 list；单个时可能是标量
            if isinstance(file_id, (list, tuple)):
                ids_iter = file_id
            else:
                ids_iter = [file_id]

            system_ids: List[int] = []
            for fid in ids_iter:
                row = self.metadata[fid]
                sid = row["Dataset_id"]
                if isinstance(sid, pd.Series):
                    # Default_dataset / IdIncludedDataset 场景下，一般是全相同的一列
                    sid = sid.iloc[0]
                system_ids.append(int(sid))

            # 去重并排序，方便阅读
            return sorted(set(system_ids))
        except Exception as exc:  # pragma: no cover - 仅用于保护日志
            logger.debug(f"[hse_contrastive] Failed to infer system_ids from file_id: {exc}")
            return []

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
        要求网络 forward 支持签名：

            forward(x, file_id=..., task_id=..., return_feature=True)

        若不支持，应在模型侧显式修复，而不是在此处吞掉异常。
        """
        network_kwargs = {"file_id": file_id, "task_id": task_id, "return_feature": True}

        # 直接调用网络，若签名不兼容或内部实现有问题，让异常完整抛出以便调试
        logits: torch.Tensor
        features: torch.Tensor
        output = self.network(x, **network_kwargs)

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
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        system_ids: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用 CE 计算分类损失，并返回简单准确率，包含GPU安全检查。"""
        # 设备一致性检查
        if logits.device != y.device:
            logger.warning(f"[hse_contrastive] Moving y from {y.device} to {logits.device}")
            y = y.to(logits.device)

        # 确保logits和y的batch维度匹配
        if logits.shape[0] != y.shape[0]:
            logger.error(f"[hse_contrastive] Batch size mismatch: logits={logits.shape[0]}, y={y.shape[0]}")
            return torch.tensor(0.0, device=logits.device), torch.tensor(0.0, device=logits.device)

        # 检查数值稳定性
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.error("[hse_contrastive] Logits contain NaN or Inf values")
            return torch.tensor(0.0, device=logits.device), torch.tensor(0.0, device=logits.device)

        if torch.isnan(y).any() or torch.isinf(y).any():
            logger.error("[hse_contrastive] Labels contain NaN or Inf values")
            return torch.tensor(0.0, device=logits.device), torch.tensor(0.0, device=logits.device)

        # 标签值范围检查
        num_classes = logits.shape[1]
        y_min, y_max = y.min().item(), y.max().item()
        if y_min < 0 or y_max >= num_classes:
            logger.error(
                f"[hse_contrastive] Label values out of range: "
                f"[{y_min}, {y_max}], expected [0, {num_classes-1}]"
                f"{'' if not system_ids else f', system_ids={system_ids}'}"
            )
            return torch.tensor(0.0, device=logits.device), torch.tensor(0.0, device=logits.device)

        # 确保标签是long类型
        if y.dtype != torch.long:
            logger.debug(f"[hse_contrastive] Converting y from {y.dtype} to long")
            y = y.long()

        # 安全的分类损失计算
        try:
            loss = self.ce_loss_fn(logits, y)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()
            return loss, acc
        except RuntimeError as exc:
            if "device-side assert" in str(exc):
                logger.error(f"[hse_contrastive] CUDA device-side assert in CE loss, logits shape: {logits.shape}, y shape: {y.shape}")
                logger.error(f"[hse_contrastive] Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}], y range: [{y_min}, {y_max}]")
            return torch.tensor(0.0, device=logits.device), torch.tensor(0.0, device=logits.device)

    def _run_contrastive_flow(self, features: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """构造两视图特征并调用策略管理器计算对比损失。"""
        if self.strategy_manager is None:
            return torch.tensor(0.0, device=features.device)

        # 增强的GPU设备一致性检查
        if not torch.is_tensor(features):
            raise TypeError(f"[hse_contrastive] features must be tensor, got {type(features)}")
        if not torch.is_tensor(y):
            raise TypeError(f"[hse_contrastive] y must be tensor, got {type(y)}")

        # 确保设备一致性
        target_device = features.device
        if y.device != target_device:
            logger.warning(f"[hse_contrastive] Moving y from {y.device} to {target_device}")
            y = y.to(target_device)

        # GPU环境下的额外安全检查
        if target_device.type == 'cuda':
            # 检查CUDA内存状态
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(target_device)
                memory_total = torch.cuda.get_device_properties(target_device).total_memory
                if memory_used > memory_total * 0.9:
                    logger.error(f"[hse_contrastive] GPU memory nearly full: {memory_used/memory_total:.1%}")
                    return torch.tensor(0.0, device=target_device)

            # 检查数值稳定性
            if torch.isnan(features).any() or torch.isinf(features).any():
                logger.error("[hse_contrastive] Features contain NaN or Inf values")
                return torch.tensor(0.0, device=target_device)
            if torch.isnan(y).any() or torch.isinf(y).any():
                logger.error("[hse_contrastive] Labels contain NaN or Inf values")
                return torch.tensor(0.0, device=target_device)

        # 确保特征展平为 [B, D]
        features = self._flatten_features(features)
        B = features.shape[0]

        # 部分损失需要标签，先做最小一致性检查，避免在 CUDA 中触发 device-side assert
        requires_labels = getattr(self.strategy_manager, "requires_labels", False)
        labels_ext: Optional[torch.Tensor] = None
        if requires_labels:
            if y is None:
                logger.error("[hse_contrastive] Contrastive strategy requires labels, but y is None")
                return torch.tensor(0.0, device=features.device)
            if y.ndim != 1:
                logger.error(f"[hse_contrastive] Expected 1D labels, got {y.ndim}D")
                return torch.tensor(0.0, device=features.device)
            if y.shape[0] != B:
                logger.error(
                    f"[hse_contrastive] Label length mismatch: features={B}, labels={y.shape[0]}"
                )
                return torch.tensor(0.0, device=features.device)

            # 标签数据类型和范围检查
            if y.dtype not in [torch.long, torch.int32, torch.int64]:
                logger.warning(f"[hse_contrastive] Converting y from {y.dtype} to long")
                y = y.long()

            # 检查标签值的合理性
            if y.min() < 0:
                logger.error(f"[hse_contrastive] Negative labels found: min={y.min()}")
                return torch.tensor(0.0, device=features.device)

            labels_ext = torch.cat([y, y], dim=0)
            # 确保扩展标签也在正确设备上
            if labels_ext.device != target_device:
                labels_ext = labels_ext.to(target_device)

        # z1: 原始特征；z2: 简单增强视图
        z1 = features
        z2 = self._create_augmented_view(features)

        # 拼接视图以兼容 InfoNCE / SupCon 等实现
        z = torch.cat([z1, z2], dim=0)

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
        """特征级数据增强，支持多种轻量策略以适配不同对比学习需求。

        通过 `args_task.augmentation_type` 控制增强方式：
        - "noise":   加性高斯噪声（默认）
        - "scaling": 随机缩放
        - "dropout": 随机元素丢弃
        - "mixed":   在上述三种中随机选择一种
        - "none":    不做增强，返回原始副本
        """
        device = features.device
        dtype = features.dtype

        aug_type = getattr(self.args_task, "augmentation_type", "noise").lower()

        if aug_type == "none":
            return features.clone()

        if aug_type == "mixed":
            # 使用 GPU 友好的随机选择，避免 Python random 开销
            candidates = ("noise", "scaling", "dropout")
            idx = torch.randint(len(candidates), (1,), device=device).item()
            aug_type = candidates[idx]

        noise_std = float(getattr(self.args_task, "augmentation_noise_std", 0.1))
        dropout_p = float(getattr(self.args_task, "augmentation_dropout_p", 0.1))
        scale_std = float(getattr(self.args_task, "augmentation_scale_std", 0.1))

        # 防止参数过大导致数值不稳定
        if noise_std > 1.0:
            logger.warning(f"[hse_contrastive] Large noise_std={noise_std}, clamping to 1.0")
            noise_std = 1.0
        if dropout_p < 0.0 or dropout_p > 0.9:
            dropout_p = min(max(dropout_p, 0.0), 0.9)
        if scale_std > 1.0:
            scale_std = 1.0

        if aug_type == "dropout" and dropout_p > 0.0:
            mask = (torch.rand_like(features, device=device) >= dropout_p).to(dtype)
            augmented = features * mask
        elif aug_type == "scaling" and scale_std > 0.0:
            scale = 1.0 + torch.randn_like(features, device=device, dtype=dtype) * scale_std
            augmented = features * scale
        else:
            # 默认或退化到噪声增强
            if noise_std <= 0.0:
                return features.clone()
            noise = torch.randn_like(features, device=device, dtype=dtype) * noise_std
            augmented = features + noise

        # 检查增强后的数值稳定性
        if device.type == "cuda":
            if torch.isnan(augmented).any() or torch.isinf(augmented).any():
                logger.warning("[hse_contrastive] Augmented features contain NaN/Inf, returning original")
                return features.clone()

        return augmented

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
