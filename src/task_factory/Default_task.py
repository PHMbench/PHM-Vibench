import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from src.task_factory import register_task
from typing import Dict, List, Optional, Any, Mapping, Tuple

# 导入解耦后的组件
from .Components.loss import get_loss_fn
from .Components.metrics import get_metrics
from .Components.regularization import calculate_regularization
from .Components.gradient_constraints import FisherGradientConstraint


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

        # 兼容旧配置：为 gpus 提供合理默认值，避免缺少属性导致崩溃
        gpus = getattr(args_trainer, "gpus", None)
        if gpus is None:
            gpus = getattr(args_trainer, "devices", 1)
            setattr(args_trainer, "gpus", gpus)

        # 将网络移动到 GPU（仅在配置明确请求 CUDA/GPU 且 CUDA 可用时）。
        # ``gpus`` 在旧配置里也被 Lightning 当作 CPU devices 使用，因此
        # 不能仅凭其非零就覆盖 ``trainer.device: cpu``。
        requested_device = str(getattr(args_trainer, "device", "cpu")).lower()
        use_cuda = (
            requested_device in {"cuda", "gpu"}
            and bool(gpus)
            and torch.cuda.is_available()
        )
        if requested_device in {"cuda", "gpu"} and bool(gpus) and not use_cuda:
            raise RuntimeError(
                "CUDA was explicitly requested but is unavailable; evidence-bearing "
                "runs must not fall back to CPU"
            )
        if use_cuda and hasattr(network, "cuda"):
            self.network = network.cuda()
        else:
            self.network = network  # 在当前环境（无 GPU）下保持 CPU 训练
        self.args_task = args_task
        self.args_model = args_model
        self.args_data = args_data
        self.metadata = metadata # 存储 metadata
        self.args_trainer = args_trainer
        self.args_environment = args_environment

        self._raw_label_order = self._build_label_contract()
        self._raw_label_to_index = {
            raw_label: index
            for index, raw_label in enumerate(self._raw_label_order or ())
        }
        grouped_evaluation = getattr(self.args_task, "grouped_evaluation", None)
        self._grouped_evaluation_enabled = bool(
            getattr(grouped_evaluation, "enabled", False)
            if grouped_evaluation is not None
            else False
        )
        self._grouped_test_records: list[dict[str, Any]] = []

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
        if (
            bool(getattr(self.network, "uses_alignment_objective", False))
            and self.gradient_constraint is not None
        ):
            raise ValueError(
                "P01 M5 forbids a post-backward gradient constraint outside its "
                "frozen classification + physical + semantic + geometric objective"
            )

        # 使用组件配置损失和指标
        self.loss_fn = get_loss_fn(self.args_task.loss)
        metric_num_classes = (
            len(self._raw_label_order)
            if self._raw_label_order is not None
            else None
        )
        self.metrics = get_metrics(
            self.args_task.metrics,
            self.metadata,
            num_classes=metric_num_classes,
            average=getattr(self.args_task, "metric_average", None),
        )

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

    def _build_label_contract(self) -> tuple[int, ...] | None:
        """Validate an optional raw-label to contiguous-index contract."""
        contract = getattr(self.args_task, "label_contract", None)
        if contract is None:
            return None
        raw_labels = getattr(contract, "raw_labels", None)
        if not isinstance(raw_labels, (list, tuple)) or len(raw_labels) < 2:
            raise ValueError(
                "task.label_contract.raw_labels must contain at least two labels"
            )
        try:
            ordered = tuple(int(label) for label in raw_labels)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "task.label_contract.raw_labels must contain integer labels"
            ) from exc
        if len(set(ordered)) != len(ordered):
            raise ValueError(
                "task.label_contract.raw_labels must not contain duplicates"
            )

        grouped_split = getattr(self.args_task, "grouped_split", None)
        admitted = (
            getattr(grouped_split, "admitted_labels", None)
            if grouped_split is not None
            else None
        )
        if admitted is not None and tuple(int(label) for label in admitted) != ordered:
            raise ValueError(
                "task.label_contract.raw_labels must exactly match "
                "task.grouped_split.admitted_labels in canonical order"
            )

        output_classes = getattr(self.network, "num_classes", None)
        if output_classes is None or int(output_classes) != len(ordered):
            raise ValueError(
                "network.num_classes must equal the length of "
                "task.label_contract.raw_labels"
            )
        return ordered

    def label_contract_identity(self) -> dict[str, Any] | None:
        """Return the pre-outcome label mapping used by the training objective."""
        if self._raw_label_order is None:
            return None
        return {
            "raw_labels": list(self._raw_label_order),
            "training_indices": list(range(len(self._raw_label_order))),
            "raw_to_training_index": dict(self._raw_label_to_index),
        }

    def encode_raw_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Map raw dataset labels to contiguous indices without mutating metadata."""
        if getattr(self, "_raw_label_order", None) is None:
            return labels
        encoded = torch.empty_like(labels, dtype=torch.long)
        matched = torch.zeros_like(labels, dtype=torch.bool)
        for raw_label, index in self._raw_label_to_index.items():
            mask = labels == raw_label
            encoded[mask] = index
            matched |= mask
        if not bool(matched.all().item()):
            unknown = torch.unique(labels[~matched]).detach().cpu().tolist()
            raise ValueError(
                "Batch contains raw label(s) outside task.label_contract: "
                f"{unknown}"
            )
        return encoded

    def decode_training_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Invert the configured mapping for result reporting."""
        if self._raw_label_order is None:
            return indices
        if indices.numel() and (
            int(indices.min().item()) < 0
            or int(indices.max().item()) >= len(self._raw_label_order)
        ):
            raise ValueError("Training index is outside the configured label contract")
        raw = torch.as_tensor(
            self._raw_label_order,
            dtype=torch.long,
            device=indices.device,
        )
        return raw[indices.long()]


    def forward(self, batch):
        """模型前向传播"""
        x = batch['x']
        file_id = batch['file_id']
        task_id = batch['task_id'] if 'task_id' in batch else None
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

    # def _forward_pass(self, batch) -> torch.Tensor:
    #     """执行前向传播"""
    #     return self(batch)

    def _compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算任务损失"""
        # 确保 y 是 long 类型用于分类损失        
        return self.loss_fn(y_hat, y.long() if y.dtype != torch.long else y)

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
        batch.setdefault('task_id', 'classification')

        batch_size = int(batch['x'].shape[0])
        raw_file_ids = batch['file_id']
        if isinstance(raw_file_ids, torch.Tensor):
            file_ids = [value.item() for value in raw_file_ids.view(-1)]
        elif isinstance(raw_file_ids, (list, tuple)):
            file_ids = list(raw_file_ids)
        else:
            file_ids = [raw_file_ids]

        if len(file_ids) not in (1, batch_size):
            raise ValueError(
                "batch['file_id'] must contain one ID or one ID per sample: "
                f"received {len(file_ids)} IDs for batch_size={batch_size}."
            )

        first_file_id = file_ids[0]
        try:
            data_name = self.metadata[first_file_id]['Name']
        except (KeyError, IndexError, TypeError) as exc:
            raise KeyError(
                f"Unable to resolve metadata Name for file_id={first_file_id!r}."
            ) from exc
        if getattr(self.network, "requires_physical_metadata", False):
            names = []
            for current_file_id in file_ids:
                try:
                    names.append(self.metadata[current_file_id]['Name'])
                except (KeyError, IndexError, TypeError) as exc:
                    raise KeyError(
                        "Unable to resolve metadata Name for "
                        f"file_id={current_file_id!r}."
                    ) from exc
            if any(name != data_name for name in names):
                raise ValueError(
                    "decisive P04 batches cannot mix dataset Names because metric "
                    "and physical metadata authority would be ambiguous"
                )

        raw_y = batch['y']
        y = Default_task.encode_raw_labels(self, raw_y)

        # 1. 前向传播。M5 在训练阶段显式返回对齐项；验证/测试从不把
        # 标签送入对齐目标。保留原始逐样本 file_id，不得用首个文件覆盖整批。
        alignment_losses: Optional[Mapping[str, torch.Tensor]] = None
        network = getattr(self, "network", None)
        if stage == "train" and bool(
            getattr(network, "uses_alignment_objective", False)
        ):
            forward_with_alignment = getattr(
                network, "forward_with_alignment", None
            )
            if not callable(forward_with_alignment):
                raise RuntimeError(
                    "An alignment-enabled network must expose forward_with_alignment"
                )
            y_hat, alignment_losses = forward_with_alignment(
                batch['x'],
                y,
                data_id=batch['file_id'],
                task_id=batch['task_id'],
            )
        else:
            y_hat = self.forward(batch)

        raw_label_order = getattr(self, "_raw_label_order", None)
        if raw_label_order is not None and (
            y_hat.ndim != 2 or y_hat.shape[1] != len(raw_label_order)
        ):
            raise ValueError(
                "Model output width does not match task.label_contract: "
                f"shape={tuple(y_hat.shape)}, labels={raw_label_order}"
            )

        # 2. 计算任务损失
        loss = self._compute_loss(y_hat, y)
        y_argmax = torch.argmax(y_hat, dim=1) if y_hat.ndim > 1 else y_hat

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

        # 5. Consume optional model-defined auxiliary losses exactly once.
        # Models without this explicit hook retain the existing behavior.
        model_auxiliary = {}
        consume_auxiliary = getattr(self.network, 'consume_auxiliary_losses', None)
        if callable(consume_auxiliary):
            model_auxiliary = consume_auxiliary()
            if not isinstance(model_auxiliary, dict):
                raise TypeError("network.consume_auxiliary_losses() must return a dict")
            for name, value in model_auxiliary.items():
                if not isinstance(value, torch.Tensor) or value.ndim != 0:
                    raise ValueError(
                        f"model auxiliary loss {name!r} must be a scalar tensor"
                    )
                if not bool(torch.isfinite(value).item()):
                    raise ValueError(f"model auxiliary loss {name!r} is not finite")
                step_metrics[f"{stage}_{name}_loss"] = value

        # 6. M5's sole allowed scalar is classification plus the three
        # a_k * lambda_k * L_k terms. Other models retain generic regularization
        # and their explicitly exposed auxiliary losses.
        regularization_total = reg_dict.get(
            'total', torch.tensor(0.0, device=loss.device)
        )
        if alignment_losses is not None:
            if int(torch.count_nonzero(regularization_total.detach()).item()) != 0:
                raise ValueError(
                    "P01 M5 forbids generic regularization outside the frozen "
                    "classification + physical + semantic + geometric objective"
                )
            if model_auxiliary:
                raise ValueError(
                    "P01 M5 forbids model auxiliary losses outside the frozen "
                    "classification + physical + semantic + geometric objective"
                )
            compose_objective = getattr(
                network, "compose_training_objective", None
            )
            if not callable(compose_objective):
                raise RuntimeError(
                    "An alignment-enabled network must expose compose_training_objective"
                )
            objective = compose_objective(loss, alignment_losses)
            total_loss = objective["total"]
            for objective_name, objective_value in objective.items():
                if objective_name == "total":
                    continue
                step_metrics[
                    f"{stage}_{objective_name}_loss"
                ] = objective_value
        else:
            auxiliary_total = sum(
                model_auxiliary.values(), torch.tensor(0.0, device=loss.device)
            )
            total_loss = loss + regularization_total + auxiliary_total
        step_metrics[f"{stage}_total_loss"] = total_loss

        if stage == "test" and bool(
            getattr(self, "_grouped_evaluation_enabled", False)
        ):
            step_metrics["_grouped_logits"] = y_hat.detach()
            step_metrics["_grouped_labels"] = y.detach()

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
        grouped_logits = metrics.pop("_grouped_logits", None)
        grouped_labels = metrics.pop("_grouped_labels", None)
        if grouped_logits is not None:
            if grouped_labels is None:
                raise RuntimeError("Grouped evaluation is missing encoded labels")
            self._record_grouped_test_batch(
                batch,
                logits=grouped_logits,
                encoded_labels=grouped_labels,
            )
        
        self._log_metrics(metrics, "test")
        # test_step 通常不返回损失

    @staticmethod
    def _batch_values(value: Any, *, name: str, batch_size: int) -> list[Any]:
        if isinstance(value, torch.Tensor):
            values = value.detach().cpu().reshape(-1).tolist()
        elif isinstance(value, np.ndarray):
            values = value.reshape(-1).tolist()
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            values = [value]
        if len(values) != batch_size:
            raise ValueError(
                f"Grouped evaluation requires one {name} per sample; "
                f"got {len(values)} for batch_size={batch_size}"
            )
        return values

    def on_test_epoch_start(self) -> None:
        if self._grouped_evaluation_enabled:
            self._grouped_test_records.clear()

    def _record_grouped_test_batch(
        self,
        batch: Mapping[str, Any],
        *,
        logits: torch.Tensor,
        encoded_labels: torch.Tensor,
    ) -> None:
        if not self._grouped_evaluation_enabled:
            return
        if logits.ndim != 2 or not bool(torch.isfinite(logits).all().item()):
            raise ValueError("Grouped evaluation requires finite rank-2 logits")
        batch_size = int(logits.shape[0])
        file_ids = self._batch_values(
            batch.get("file_id"), name="file_id", batch_size=batch_size
        )
        group_ids = self._batch_values(
            batch.get("physical_group_id"),
            name="physical_group_id",
            batch_size=batch_size,
        )
        raw_labels = self._batch_values(
            batch.get("y"), name="raw label", batch_size=batch_size
        )
        training_labels = self._batch_values(
            encoded_labels, name="training label", batch_size=batch_size
        )
        logits_cpu = logits.detach().cpu()

        for index, (file_id, group_id, raw_label, training_label) in enumerate(
            zip(file_ids, group_ids, raw_labels, training_labels)
        ):
            try:
                metadata_row = self.metadata[file_id]
                metadata_label = int(metadata_row["Label"])
                domain_id = int(metadata_row["Domain_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Grouped evaluation cannot resolve metadata for file_id={file_id!r}"
                ) from exc
            if not isinstance(group_id, str) or not group_id:
                raise ValueError(
                    "Grouped evaluation requires a non-empty physical_group_id"
                )
            if int(raw_label) != metadata_label:
                raise ValueError(
                    "Batch raw label differs from immutable metadata for "
                    f"file_id={file_id!r}"
                )
            self._grouped_test_records.append(
                {
                    "file_id": file_id,
                    "physical_group_id": group_id,
                    "domain_id": domain_id,
                    "raw_label": metadata_label,
                    "training_label": int(training_label),
                    "logits": logits_cpu[index].tolist(),
                }
            )

    def grouped_evaluation_records(self) -> list[dict[str, Any]]:
        """Return a copy of post-checkpoint test records for P01 aggregation."""
        return [dict(record) for record in self._grouped_test_records]

    def _log_metrics(self, metrics: Dict[str, torch.Tensor], stage: str) -> None:
        """统一日志记录"""
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
