import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from src.task_factory import register_task
from typing import Dict, List, Optional, Any, Tuple

# 导入解耦后的组件
from .Components.loss import get_loss_fn
from .Components.metrics import get_metrics
from .Components.regularization import calculate_regularization
from .Components.gradient_constraints import FisherGradientConstraint
from .p05_epoch_metrics import (
    WeightedEpochConfusionMatrix,
    WeightedEpochLoss,
    weighted_mean_loss,
)


@register_task("Default_task", "Default_task")
class Default_task(pl.LightningModule):
    """
    通用 PyTorch Lightning 任务模块 (已重构)

    Features:
    - 通过组件配置损失函数和评估指标
    - 通过组件配置正则化方法
    - 灵活的优化器和调度器配置
    - 期望 batch 格式为 ((x, y), data_name)
    """

    def __init__(
        self,
        network: nn.Module,
        args_data: Any,  # Data args (Namespace)
        args_model: Any,  # Model args (Namespace)
        args_task: Any,  # Training args (Namespace)
        args_trainer: Any,  # Trainer args (Namespace)
        args_environment: Any,  # Environment args (Namespace)
        metadata: Any # Metadata object/dict
    ):
        """
        初始化训练模块

        :param network: 待训练的主干网络
        :param args_t: 训练参数配置对象 (Namespace)
        :param args_m: 模型参数配置对象 (Namespace)
        :param args_d: 数据参数配置对象 (Namespace)
        :param metadata: 数据元信息
        """
        super().__init__()

        evidence_mode = getattr(args_task, "p05_evidence_mode", False)
        if not isinstance(evidence_mode, bool):
            raise TypeError("task.p05_evidence_mode must be a boolean")
        self.p05_evidence_mode = evidence_mode

        # 兼容旧配置：为 gpus 提供合理默认值，避免缺少属性导致崩溃
        gpus = getattr(args_trainer, "gpus", None)
        if gpus is None:
            gpus = getattr(args_trainer, "devices", 1)
            setattr(args_trainer, "gpus", gpus)

        if self.p05_evidence_mode:
            if getattr(args_model, "device", None) != "cuda":
                raise RuntimeError("P05 evidence mode requires model.device='cuda'")
            runtime_identity = getattr(args_trainer, "p05_runtime_identity", None)
            if not isinstance(runtime_identity, dict) or runtime_identity.get("evidence_mode") is not True:
                raise RuntimeError(
                    "P05 evidence task requires a completed fail-closed runtime preflight"
                )
            if gpus != 1:
                raise RuntimeError("P05 evidence task requires exactly one GPU")
            if not hasattr(network, "cuda"):
                raise RuntimeError("P05 evidence network cannot be moved to CUDA")
            try:
                self.network = network.cuda()
            except Exception as exc:
                raise RuntimeError(
                    "P05 evidence network CUDA placement failed; CPU fallback is forbidden"
                ) from exc
        else:
            # Legacy behavior: use CUDA when available, otherwise keep CPU.
            use_cuda = bool(gpus) and torch.cuda.is_available()
            if use_cuda and hasattr(network, "cuda"):
                self.network = network.cuda()
            else:
                self.network = network
        self.args_task = args_task
        self.args_model = args_model
        self.args_data = args_data
        self.metadata = metadata # 存储 metadata
        self.args_trainer = args_trainer
        self.args_environment = args_environment

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

        # 使用组件配置损失和指标
        if self.p05_evidence_mode:
            if str(self.args_task.loss).upper() not in {"CE", "CE_WEIGHTED"}:
                raise ValueError(
                    "P05 evidence mode requires task.loss=CE_weighted "
                    "(legacy CE spelling is also accepted)"
                )
            self.loss_fn = get_loss_fn(self.args_task.loss, reduction="none")
        else:
            self.loss_fn = get_loss_fn(self.args_task.loss)

        configured_metrics = list(getattr(self.args_task, "metrics", []))
        if self.p05_evidence_mode:
            if any(str(name).lower() == "f1" for name in configured_metrics):
                raise ValueError(
                    "P05 evidence mode forbids batch-aggregated metric 'f1'; "
                    "use 'f1_macro'"
                )
            configured_metrics = [
                name
                for name in configured_metrics
                if str(name).lower() not in {"f1_macro", "acc", "accuracy"}
            ]
            num_classes = self._resolve_p05_num_classes(args_model, metadata)
            self.p05_epoch_metrics = nn.ModuleDict(
                {
                    f"{stage}_epoch": WeightedEpochConfusionMatrix(num_classes)
                    for stage in ("train", "val", "test")
                }
            )
            self.p05_epoch_losses = nn.ModuleDict(
                {
                    f"{stage}_epoch": WeightedEpochLoss()
                    for stage in ("train", "val", "test")
                }
            )
        # 假设 get_metrics 需要数据配置来确定任务类型和类别数
        self.metrics = get_metrics(configured_metrics, self.metadata)

        # 保存超参数 (确保 Namespace 可以转换为字典)
        hparams_dict = {**vars(self.args_task),
                            **vars(self.args_model),
                            **vars(self.args_data),
                            **vars(self.args_trainer),
                            **vars(self.args_environment),
                            # metadata 可能包含复杂对象，选择性保存或忽略
                            # 'metadata': metadata
                            }
        self.save_hyperparameters(hparams_dict, ignore=['network', 'metadata'])


    def forward(self, batch):
        """模型前向传播"""
        x = batch['x']
        file_id = batch['file_id']
        task_id = batch['task_id'] if 'task_id' in batch else None

        return self.network(x, file_id, task_id)

    # def _forward_pass(self, batch) -> torch.Tensor:
    #     """执行前向传播"""
    #     return self(batch)

    @staticmethod
    def _resolve_p05_num_classes(args_model: Any, metadata: Any) -> int:
        configured = getattr(args_model, "num_classes", None)
        if configured is None:
            labels = [item["Label"] for item in metadata.values() if "Label" in item]
            if not labels:
                raise ValueError("P05 evidence mode requires model.num_classes or metadata labels")
            configured = max(int(label) for label in labels) + 1
        if isinstance(configured, bool) or int(configured) != configured or configured < 2:
            raise ValueError("P05 evidence num_classes must be an integer >= 2")
        return int(configured)

    def _p05_sample_weight(self, batch: dict, stage: str) -> torch.Tensor:
        specific_key = getattr(self.args_task, f"{stage}_sample_weight_key", None)
        weight_key = specific_key or getattr(
            self.args_task,
            "sample_weight_key",
            "sample_weight",
        )
        if not isinstance(weight_key, str) or not weight_key:
            raise ValueError(f"P05 {stage} sample-weight key must be a non-empty string")
        if weight_key not in batch:
            raise KeyError(
                f"P05 evidence {stage} batch is missing sample-weight field {weight_key!r}"
            )
        return batch[weight_key]

    def _compute_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """计算任务损失"""
        # 确保 y 是 long 类型用于分类损失
        targets = y.long() if y.dtype != torch.long else y
        loss = self.loss_fn(y_hat, targets)
        if not self.p05_evidence_mode:
            return loss
        return weighted_mean_loss(loss, sample_weight)

    def _p05_unreduced_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        if not self.p05_evidence_mode:
            raise RuntimeError("unreduced P05 loss is available only in evidence mode")
        targets = y.long() if y.dtype != torch.long else y
        return self.loss_fn(y_hat, targets)

    def _compute_metrics(self, y_hat: torch.Tensor, y: torch.Tensor, data_name: str, stage: str) -> Dict[str, torch.Tensor]:
        """计算并更新评估指标"""
        metric_values = {}
        # print(f"计算 {stage} 阶段的指标: {data_name}")
        if data_name in self.metrics:
            for metric_key, metric_fn in self.metrics[data_name].items():
                if metric_key.startswith(stage):
                    # metric_fn 是 torchmetrics 对象，调用会更新内部状态并返回值
                    value = metric_fn(y_hat, y)
                    # 记录当前 step 的值 (注意：torchmetrics 通常在 epoch 结束时计算最终值)
                    # 为了日志记录，我们可能需要记录瞬时值或累计值
                    # 这里记录瞬时值，log_dict 会在 epoch 结束时聚合
                    metric_values[f"{metric_key}_{data_name}"] = value
        else:
            # 仅在第一次遇到未知 data_name 时打印警告，避免刷屏
            if not hasattr(self, '_warned_missing_metrics') or data_name not in self._warned_missing_metrics:
                 print(f"警告: 在 metrics 中未找到数据名称 '{data_name}' 的指标配置。")
                 if not hasattr(self, '_warned_missing_metrics'):
                     self._warned_missing_metrics = set()
                 self._warned_missing_metrics.add(data_name)

        return metric_values

    def _compute_regularization(self) -> Dict[str, torch.Tensor]:
        """计算正则化损失"""
        return calculate_regularization(
            getattr(self.args_task, 'regularization', {}),
            self.parameters() # 只对当前 LightningModule 的参数计算正则化
        )

    def _shared_step(self, batch: Tuple,
                     stage: str,
                     task_id = False) -> Dict[str, torch.Tensor]:
        """
        通用处理步骤 (已重构)
        期望 batch 格式: ((x, y), data_name)
        """
        try:
            # x, y, id = batch['x'], batch['y'], batch['id']
            # Ensure a default task identifier if not provided
            batch.setdefault('task_id', 'classification')
            # Convert tensor-based ID to a Python int for indexing metadata
            file_id = batch['file_id'][0].item()
            data_name = self.metadata[file_id]['Name']# .values
            # dataset_id = self.metadata[file_id]['Dataset_id'].item() 
            batch.update({'file_id': file_id})
        except (ValueError, TypeError) as e:
            raise ValueError(f" Error: {e}")

        sample_weight = self._p05_sample_weight(batch, stage) if self.p05_evidence_mode else None

        # 1. 前向传播
        y_hat = self.forward(batch)

        # 2. 计算任务损失
        y = batch['y']
        if self.p05_evidence_mode:
            per_sample_loss = self._p05_unreduced_loss(y_hat, y)
            loss = weighted_mean_loss(per_sample_loss, sample_weight)
            self.p05_epoch_losses[f"{stage}_epoch"].update(
                per_sample_loss,
                sample_weight,
            )
        else:
            loss = self._compute_loss(y_hat, y, sample_weight=sample_weight)
        y_argmax = torch.argmax(y_hat, dim=1) if y_hat.ndim > 1 else y_hat

        if self.p05_evidence_mode:
            self.p05_epoch_metrics[f"{stage}_epoch"].update(y_argmax, y, sample_weight)

        # 3. 计算和记录指标
        step_metrics = {f"{stage}_loss": loss}
        step_metrics[f"{stage}_{data_name}_loss"] = loss # 记录特定数据集的损失
        metric_values = self._compute_metrics(y_argmax, y, data_name, stage)
        step_metrics.update(metric_values)

        # 4. 计算正则化损失
        reg_dict = self._compute_regularization()
        for reg_type, reg_loss_val in reg_dict.items():
            if reg_type != 'total':
                step_metrics[f"{stage}_{reg_type}_reg_loss"] = reg_loss_val

        # 5. 计算总损失
        total_loss = loss + reg_dict.get('total', torch.tensor(0.0, device=loss.device))
        step_metrics[f"{stage}_total_loss"] = total_loss

        # 添加 batch size 用于日志记录
        # step_metrics[f"{stage}_batch_size"] = torch.tensor(x.shape[0], dtype=torch.float, device=loss.device)

        return step_metrics

    def training_step(self, batch: dict, *args, **kwargs) -> torch.Tensor:
        """训练步骤"""
        metrics = self._shared_step(batch, "train")
        # 使用 _log_metrics 记录 (确保 batch_size 传递正确)
      
        self._log_metrics(metrics, "train")
        # 返回用于反向传播的总损失
        return metrics["train_total_loss"]

    def validation_step(self, batch: dict, *args, **kwargs) -> None:
        """验证步骤"""
        metrics = self._shared_step(batch, "val")
      
        self._log_metrics(metrics, "val")
        # validation_step 通常不返回损失

    def test_step(self, batch: dict, *args, **kwargs) -> None:
        """测试步骤"""
        metrics = self._shared_step(batch, "test")
        
        self._log_metrics(metrics, "test")
        # test_step 通常不返回损失

    def _reset_p05_epoch_metric(self, stage: str) -> None:
        if self.p05_evidence_mode:
            self.p05_epoch_metrics[f"{stage}_epoch"].reset()
            self.p05_epoch_losses[f"{stage}_epoch"].reset()

    def _log_p05_epoch_statistics(self, stage: str) -> None:
        if not self.p05_evidence_mode:
            return
        confusion = self.p05_epoch_metrics[f"{stage}_epoch"]
        values = {
            f"{stage}_loss": self.p05_epoch_losses[f"{stage}_epoch"].compute(),
            f"{stage}_acc": confusion.compute_accuracy(),
            f"{stage}_f1_macro": confusion.compute_macro_f1(),
        }
        for name, value in values.items():
            self.log(
                name,
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=(stage == "val" and name in {"val_loss", "val_f1_macro"}),
                logger=True,
                sync_dist=False,
            )

    def on_train_epoch_start(self) -> None:
        self._reset_p05_epoch_metric("train")

    def on_validation_epoch_start(self) -> None:
        self._reset_p05_epoch_metric("val")

    def on_test_epoch_start(self) -> None:
        self._reset_p05_epoch_metric("test")

    def on_train_epoch_end(self) -> None:
        self._log_p05_epoch_statistics("train")

    def on_validation_epoch_end(self) -> None:
        self._log_p05_epoch_statistics("val")

    def on_test_epoch_end(self) -> None:
        self._log_p05_epoch_statistics("test")

    def _log_metrics(self, metrics: Dict[str, torch.Tensor], stage: str) -> None:
        """统一日志记录"""
        if self.p05_evidence_mode:
            # Validation/checkpoint metrics are emitted exactly once from the
            # complete float64 epoch accumulators.  A step-only training value
            # remains useful for diagnostics but cannot select a checkpoint.
            if stage == "train":
                self.log(
                    "train_step_loss",
                    metrics["train_loss"],
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                )
            return
        log_dict = {}
        prog_bar_metrics = {}
        for k, v in metrics.items():
            # 过滤掉非当前阶段或 batch_size 的指标
            if k.startswith(stage) and "batch_size" not in k:
                log_dict[k] = v
                # 选择要在进度条上显示的指标
                if any(prog_key in k for prog_key in ['loss', 'acc', 'f1']): # 简化进度条显示
                    # 只显示不带数据集名称的总指标或第一个数据集的指标
                    if f"{stage}_loss" == k or f"{stage}_acc_" in k or f"{stage}_f1_" in k:
                         prog_bar_metrics[k.replace(f"_{stage}", "")] = v # 简化显示名称


        self.log_dict(
            log_dict,
            on_step= (stage == "train"), # 训练时可以记录 step 级别的 loss
            on_epoch=True,
            prog_bar=False, # 单独控制进度条
            logger=True,
            sync_dist=True,
        )
        # 单独记录需要在进度条显示的指标 (只在 epoch 结束时显示聚合值)
        # self.log_dict(
        #     prog_bar_metrics,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=False, # 避免重复记录
        #     sync_dist=True,
        # )

    def configure_optimizers(self):
        """配置优化器和学习率调度器 (保持不变或根据需要调整)"""
        optimizer_name = self.args_task.optimizer.lower()
        lr = self.args_task.lr
        weight_decay = getattr(self.args_task, 'weight_decay', 0.0) # 提供默认值

        # 选择优化器
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = getattr(self.args_task, 'momentum', 0.9) # SGD momentum
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")

        # 配置学习率调度器 (如果指定)
        scheduler_config = getattr(self.args_task, 'scheduler', None)
        if not scheduler_config or not isinstance(scheduler_config, dict) or not scheduler_config.get('name'):
            return optimizer # 只返回优化器

        scheduler_name = scheduler_config['name'].lower()
        scheduler_options = scheduler_config.get('options', {}) # 获取调度器特定参数

        if scheduler_name == 'reduceonplateau':
            # 确保 monitor 指标存在
            monitor_metric = getattr(self.args_task, 'monitor', 'val_total_loss')
            # 可以在这里添加检查，确保 monitor_metric 会被记录
            # if monitor_metric not in self.metrics... (但这比较复杂，因为指标是动态生成的)
            patience = scheduler_options.get('patience', getattr(self.args_task, 'patience', 10) // 2 if hasattr(self.args_task, 'patience') else 5)
            factor = scheduler_options.get('factor', 0.1)
            mode = scheduler_options.get('mode', 'min')
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, factor=factor, patience=patience
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': monitor_metric, # 指定监控的指标
                    'interval': 'epoch', # 通常在 epoch 结束时调整
                    'frequency': 1
                }
            }
        elif scheduler_name == 'cosine':
            # 尝试从 trainer 获取 max_epochs，否则从 args_task 获取
            max_epochs = getattr(self.trainer, 'max_epochs', None) or getattr(self.args_task, 'max_epochs', 100)
            t_max = scheduler_options.get('T_max', max_epochs)
            eta_min = scheduler_options.get('eta_min', 0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        elif scheduler_name == 'step':
            step_size = scheduler_options.get('step_size', 10)
            gamma = scheduler_options.get('gamma', 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            raise ValueError(f"不支持的调度器: {scheduler_name}")

        # 对于非 ReduceLROnPlateau 的调度器，返回列表形式
        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}]

    def on_before_optimizer_step(self, optimizer) -> None:
        """Apply an optional post-backward gradient constraint before the update."""
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
