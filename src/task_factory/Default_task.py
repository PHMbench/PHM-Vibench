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
            requested_device = str(getattr(args_trainer, "device", "cpu")).lower()
            use_cuda = (
                requested_device in {"cuda", "gpu"}
                and bool(gpus)
                and torch.cuda.is_available()
            )
            if requested_device in {"cuda", "gpu"} and bool(gpus) and not use_cuda:
                raise RuntimeError(
                    "CUDA was explicitly requested but is unavailable; "
                    "evidence-bearing runs must not fall back to CPU"
                )
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
        self._alignment_target_control = self._build_alignment_target_control()
        self._alignment_training_sums: dict[str, float] = {}
        self._alignment_training_sample_count = 0
        self._alignment_training_batch_count = 0
        self._last_alignment_target_permutation: torch.Tensor | None = None
        self._alignment_target_derived_seeds: list[int] = []
        self._p01_view_gradient_observation: dict[str, Any] | None = None
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
        metric_num_classes = (
            len(self._raw_label_order)
            if self._raw_label_order is not None
            else None
        )
        self.metrics = get_metrics(
            configured_metrics,
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

    def _build_alignment_target_control(self) -> dict[str, Any] | None:
        control = getattr(self.args_task, "alignment_target_control", None)
        if control is None:
            return None
        if not bool(getattr(self.network, "uses_alignment_objective", False)):
            raise ValueError(
                "task.alignment_target_control requires an alignment-enabled model"
            )
        grouped = getattr(
            getattr(self, "args_task", None), "grouped_evaluation", None
        )
        if (
            grouped is None
            or str(getattr(grouped, "goal_id", "")) not in {"C04", "C05", "C06"}
            or str(getattr(grouped, "condition_id", "")) != "C2"
        ):
            raise ValueError(
                "task.alignment_target_control is admitted only for C04/C05/C06 "
                "condition C2"
            )
        mode = str(getattr(control, "mode", ""))
        expected_mode = "seeded_sattolo_derangement_after_batching"
        if mode != expected_mode:
            raise ValueError(
                f"C2 alignment target mode must be {expected_mode!r}"
            )
        seed = getattr(control, "seed", None)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("C2 alignment target seed must be a non-negative integer")
        seed_key = str(getattr(control, "seed_key", ""))
        expected_seed_key = "base_seed_plus_epoch_times_1000003_plus_batch_index"
        if seed_key != expected_seed_key:
            raise ValueError(f"C2 alignment target seed_key must be {expected_seed_key!r}")
        return {
            "mode": mode,
            "seed": seed,
            "algorithm": "sattolo_single_cycle",
            "stage": "train_after_batching",
            "operand": "alignment_target_z2_only",
            "affected_terms": [
                "physical_energy",
                "physical_spectral",
                "semantic",
                "geometric",
            ],
            "unaffected_terms": ["classification", "physical_parseval"],
            "classification_pairing": "synchronized_original_views",
            "semantic_mask_basis": "original_label_and_index_slots",
            "seed_key": seed_key,
            "rng_scope": "dedicated_cpu_generator_no_global_rng_mutation",
            "fixed_point_policy": "forbidden",
        }

    def alignment_target_control_identity(self) -> dict[str, Any] | None:
        return (
            None
            if self._alignment_target_control is None
            else dict(self._alignment_target_control)
        )

    def _alignment_target_permutation(
        self,
        *,
        batch_size: int,
        device: torch.device,
        batch_index: int,
    ) -> torch.Tensor | None:
        control = self._alignment_target_control
        if control is None:
            return None
        if batch_size < 2:
            raise ValueError("C2 target derangement requires batch_size >= 2")
        if isinstance(batch_index, bool) or not isinstance(batch_index, int):
            raise TypeError("C2 target derangement batch_index must be an integer")
        if batch_index < 0:
            raise ValueError("C2 target derangement batch_index must be non-negative")
        epoch = int(getattr(getattr(self, "_trainer", None), "current_epoch", 0))
        derived_seed = int(control["seed"]) + epoch * 1_000_003 + batch_index
        generator = torch.Generator(device="cpu")
        generator.manual_seed(derived_seed)
        values = list(range(batch_size))
        for index in range(batch_size - 1, 0, -1):
            swap_index = int(
                torch.randint(index, (1,), generator=generator).item()
            )
            values[index], values[swap_index] = values[swap_index], values[index]
        permutation = torch.tensor(values, dtype=torch.long, device=device)
        expected = torch.arange(batch_size, device=device)
        if bool(torch.eq(permutation, expected).any().item()):
            raise RuntimeError("Sattolo target construction produced a fixed point")
        self._last_alignment_target_permutation = permutation.detach().cpu()
        self._alignment_target_derived_seeds.append(derived_seed)
        return permutation

    def _record_p01_training_objective(
        self,
        objective: Mapping[str, torch.Tensor],
        *,
        batch_size: int,
    ) -> None:
        grouped = getattr(
            getattr(self, "args_task", None), "grouped_evaluation", None
        )
        goal_id = str(getattr(grouped, "goal_id", ""))
        if goal_id not in {"C04", "C05", "C06"}:
            return
        names = (
            "classification",
            "physical",
            "semantic",
            "geometric",
            "weighted_physical",
            "weighted_semantic",
            "weighted_geometric",
            "total",
            "physical_energy",
            "physical_spectral",
            "physical_parseval",
        )
        observed = {
            name: float(objective[name].detach().cpu().item())
            for name in names
            if name in objective
        }
        if not observed:
            raise RuntimeError(f"{goal_id} training objective summary is empty")
        if self._alignment_training_sums and set(observed) != set(
            self._alignment_training_sums
        ):
            raise RuntimeError(f"{goal_id} objective fields changed between batches")
        for name, value in observed.items():
            if not np.isfinite(value):
                raise FloatingPointError(
                    f"{goal_id} training summary field {name!r} is not finite"
                )
            self._alignment_training_sums[name] = (
                self._alignment_training_sums.get(name, 0.0)
                + value * batch_size
            )
        self._alignment_training_sample_count += batch_size
        self._alignment_training_batch_count += 1

    def on_train_start(self) -> None:
        """Start fresh current-fit objective and decisive P01 view-use observations."""
        self._alignment_training_sums.clear()
        self._alignment_training_sample_count = 0
        self._alignment_training_batch_count = 0
        self._last_alignment_target_permutation = None
        self._alignment_target_derived_seeds.clear()
        self._p01_view_gradient_observation = None

    def on_after_backward(self) -> None:
        """Fail closed when a required P01 branch is unused on the first batch."""
        grouped = getattr(self.args_task, "grouped_evaluation", None)
        goal_id = str(getattr(grouped, "goal_id", ""))
        if (
            goal_id not in {"C05", "C06"}
            or self._p01_view_gradient_observation is not None
        ):
            return
        condition_id = str(
            getattr(grouped, "condition_id", getattr(self.args_model, "condition", ""))
        )
        required_groups = {
            "M1": {
                "waveform_1d": ("encoder_1d.", "project_1d."),
                "classifier_head": ("head.",),
            },
            "M2": {
                "time_frequency_2d": ("encoder_2d.", "project_2d."),
                "classifier_head": ("head.",),
            },
            "M3": {
                "waveform_1d": ("encoder_1d.", "project_1d."),
                "time_frequency_2d": ("encoder_2d.", "project_2d."),
                "classifier_head": ("head.",),
            },
            "M4": {
                "waveform_1d": ("encoder_1d.", "project_1d."),
                "time_frequency_2d": ("encoder_2d.", "project_2d."),
                "fusion_attention": ("attention.",),
                "classifier_head": ("head.",),
            },
            "M5": {
                "waveform_1d": ("encoder_1d.", "project_1d."),
                "time_frequency_2d": ("encoder_2d.", "project_2d."),
                "classifier_head": ("head.",),
            },
            "C1": {
                "waveform_1d": ("encoder_1d.", "project_1d."),
                "time_frequency_2d": ("encoder_2d.", "project_2d."),
                "fusion_attention": ("attention.",),
                "classifier_head": ("head.",),
            },
            "C2": {
                "waveform_1d": ("encoder_1d.", "project_1d."),
                "time_frequency_2d": ("encoder_2d.", "project_2d."),
                "classifier_head": ("head.",),
            },
            "C3": {
                "time_frequency_2d": ("encoder_2d.", "project_2d."),
                "duplicate_time_frequency_2d": (
                    "encoder_duplicate_2d.",
                    "project_duplicate_2d.",
                ),
                "classifier_head": ("head.",),
            },
        }.get(condition_id)
        if required_groups is None:
            raise RuntimeError(
                f"{goal_id} has no view-gradient contract for {condition_id!r}"
            )

        named_parameters = tuple(self.network.named_parameters())
        threshold = 1.0e-12
        norms: dict[str, float] = {}
        for group_name, prefixes in required_groups.items():
            parameters = [
                parameter
                for name, parameter in named_parameters
                if parameter.requires_grad and name.startswith(prefixes)
            ]
            if not parameters:
                raise RuntimeError(
                    f"{goal_id} {condition_id} gradient group {group_name!r} "
                    "has no parameters"
                )
            squared_norm: torch.Tensor | None = None
            for parameter in parameters:
                if parameter.grad is None:
                    continue
                term = parameter.grad.detach().float().square().sum()
                squared_norm = term if squared_norm is None else squared_norm + term
            norm = 0.0 if squared_norm is None else float(squared_norm.sqrt().cpu().item())
            if not np.isfinite(norm) or norm <= threshold:
                raise RuntimeError(
                    f"{goal_id} {condition_id} required gradient group {group_name!r} "
                    f"has norm {norm}, threshold {threshold}"
                )
            norms[group_name] = norm
        self._p01_view_gradient_observation = {
            "scope": "first_source_training_batch_after_backward_before_optimizer_step",
            "condition_id": condition_id,
            "required_gradient_norm_threshold": threshold,
            "gradient_norms": dict(sorted(norms.items())),
            "status": "passed",
        }

    def view_gradient_summary(self) -> dict[str, Any] | None:
        observation = self._p01_view_gradient_observation
        if observation is None:
            return None
        return {
            **observation,
            "gradient_norms": dict(observation["gradient_norms"]),
        }

    def training_objective_summary(self) -> dict[str, Any] | None:
        if self._alignment_training_sample_count == 0:
            return None
        count = self._alignment_training_sample_count
        means = {
            name: value / count
            for name, value in sorted(self._alignment_training_sums.items())
        }
        reconstructed = means["classification"] + sum(
            means.get(name, 0.0)
            for name in (
                "weighted_physical",
                "weighted_semantic",
                "weighted_geometric",
            )
        )
        summary: dict[str, Any] = {
            "scope": "source_train_current_fit_not_checkpoint_persistent",
            "aggregation": "batch_scalar_mean_weighted_by_batch_size",
            "observed_samples": count,
            "observed_batches": self._alignment_training_batch_count,
            "means": means,
            "objective_reconstruction_residual": means["total"] - reconstructed,
            "alignment_coefficients": getattr(
                self.network, "alignment_identity", lambda: None
            )(),
        }
        if self._alignment_target_control is not None:
            seeds = self._alignment_target_derived_seeds
            summary["target_permutation_observation"] = {
                "observed_permutations": len(seeds),
                "observed_fixed_points": 0,
                "derived_seed_min": min(seeds) if seeds else None,
                "derived_seed_max": max(seeds) if seeds else None,
                "unique_derived_seeds": len(set(seeds)),
            }
        return summary

    def alignment_training_summary(self) -> dict[str, Any] | None:
        """Backward-compatible reader for the P01 objective summary."""
        return self.training_objective_summary()

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

    def _compute_auxiliary_loss(
        self, reference_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Collect and explicitly weight model-provided representation losses."""

        provider = getattr(self.network, "get_auxiliary_losses", None)
        if not callable(provider):
            return reference_loss.new_zeros(()), {}
        components = provider()
        if not isinstance(components, dict):
            raise ValueError("get_auxiliary_losses must return a dict")
        configured = getattr(self.args_task, "auxiliary_loss_weights", None)
        if isinstance(configured, dict):
            weights = configured
        elif hasattr(configured, "__dict__"):
            weights = vars(configured)
        elif components:
            raise ValueError(
                "A model with auxiliary losses requires task.auxiliary_loss_weights"
            )
        else:
            return reference_loss.new_zeros(()), {}
        missing = sorted(set(components) - set(weights))
        unknown = sorted(set(weights) - set(components))
        if missing or unknown:
            raise ValueError(
                f"Auxiliary loss/weight mismatch: missing={missing}, unknown={unknown}"
            )
        total = reference_loss.new_zeros(())
        checked = {}
        for name, component in components.items():
            if not isinstance(component, torch.Tensor) or component.numel() != 1:
                raise ValueError(f"Auxiliary loss '{name}' must be a scalar tensor")
            if not bool(torch.isfinite(component).all()):
                raise FloatingPointError(f"Auxiliary loss '{name}' is not finite")
            weight = float(weights[name])
            if weight < 0.0:
                raise ValueError(
                    f"Auxiliary loss weight '{name}' must be non-negative"
                )
            checked[name] = component
            total = total + weight * component
        return total, checked

    def _shared_step(self, batch: Tuple,
                     stage: str,
                     task_id = False,
                     *,
                     batch_index: int = 0) -> Dict[str, torch.Tensor]:
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
            alignment_target_permutation = self._alignment_target_permutation(
                batch_size=batch_size,
                device=batch['x'].device,
                batch_index=batch_index,
            )
            y_hat, alignment_losses = forward_with_alignment(
                batch['x'],
                y,
                data_id=batch['file_id'],
                task_id=batch['task_id'],
                alignment_target_permutation=alignment_target_permutation,
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
        if self.p05_evidence_mode:
            sample_weight = self._p05_sample_weight(batch, stage)
            per_sample_loss = self._p05_unreduced_loss(y_hat, y)
            loss = weighted_mean_loss(per_sample_loss, sample_weight)
            self.p05_epoch_losses[f"{stage}_epoch"].update(
                per_sample_loss,
                sample_weight,
            )
        else:
            sample_weight = None
            loss = self._compute_loss(y_hat, y)
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
            Default_task._record_p01_training_objective(
                self,
                objective,
                batch_size=batch_size,
            )
            for objective_name, objective_value in objective.items():
                if objective_name == "total":
                    continue
                step_metrics[
                    f"{stage}_{objective_name}_loss"
                ] = objective_value
        else:
            if model_auxiliary:
                auxiliary_total = sum(
                    model_auxiliary.values(),
                    torch.tensor(0.0, device=loss.device),
                )
            else:
                auxiliary_total, auxiliary_components = (
                    self._compute_auxiliary_loss(loss)
                )
                for name, component in auxiliary_components.items():
                    step_metrics[f"{stage}_aux_{name}_loss"] = component
                if auxiliary_components:
                    step_metrics[f"{stage}_aux_total_loss"] = auxiliary_total
            total_loss = loss + regularization_total + auxiliary_total
            if stage == "train":
                Default_task._record_p01_training_objective(
                    self,
                    {"classification": loss, "total": total_loss},
                    batch_size=batch_size,
                )
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
        raw_batch_index = args[0] if args else kwargs.get("batch_idx", 0)
        if isinstance(raw_batch_index, bool) or not isinstance(raw_batch_index, int):
            raise TypeError("training_step batch_idx must be an integer")
        metrics = self._shared_step(
            batch,
            "train",
            batch_index=raw_batch_index,
        )
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
        self._reset_p05_epoch_metric("test")

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
