"""Support-only reliability-conditioned generalized few-shot adaptation.

The module deliberately separates two concerns:

* :class:`SupportReliabilityConditioner` is a feature-level, query-blind method
  that can be tested without a dataset or trainer.
* :class:`task` binds the method to the existing ViBench Lightning task surface.

The implementation has no permissive fallback.  Missing classes, unexpected
labels, non-finite features, and incompatible classifier dimensions are hard
errors because any of those conditions would invalidate a P09 comparison.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_factory.ISFM.system_utils import resolve_batch_metadata
from src.task_factory import register_task

from ...Default_task import Default_task


RELIABILITY_FEATURE_NAMES = (
    "count",
    "compactness",
    "inlier_weight",
    "class_balance",
)


def _config_value(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _as_unique_class_ids(
    values: Iterable[int] | torch.Tensor,
    *,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    ids = torch.as_tensor(list(values) if not isinstance(values, torch.Tensor) else values, device=device)
    ids = ids.to(dtype=torch.long).reshape(-1)
    if ids.numel() == 0:
        raise ValueError(f"{name} must contain at least one class id")
    if torch.unique(ids).numel() != ids.numel():
        raise ValueError(f"{name} must contain unique class ids")
    return ids


@dataclass(frozen=True)
class SupportCondition:
    """Immutable support-derived adaptation state.

    Every tensor in this object is a function of labeled support features,
    frozen base weights, and fixed method parameters. Query features are not an
    input to :meth:`SupportReliabilityConditioner.condition`.
    """

    novel_class_ids: torch.Tensor
    robust_prototypes: torch.Tensor
    base_priors: torch.Tensor
    adapted_prototypes: torch.Tensor
    reliability_features: torch.Tensor
    reliability: torch.Tensor
    temperature: torch.Tensor
    adapter_gate: torch.Tensor
    abstention_threshold: torch.Tensor

    @property
    def feature_names(self) -> tuple[str, ...]:
        return RELIABILITY_FEATURE_NAMES


class SupportReliabilityConditioner(nn.Module):
    """Closed-form reliability controller with a bounded trainable adapter.

    The four reliability features are combined by a geometric mean.  The
    resulting class reliability controls three quantities monotonically:

    * support weight in an empirical-Bayes prototype shrinkage rule;
    * class-specific predictive temperature;
    * a conservative, global gate on a source-trained low-rank adapter.

    The adapter residual is clipped per sample so that its pre-normalization
    norm cannot exceed ``adapter_gate * adapter_relative_bound * ||x||``.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        adapter_rank: int = 16,
        count_prior: float = 2.0,
        reliability_floor: float = 0.05,
        reliability_ceiling: float = 0.98,
        temperature_min: float = 0.75,
        temperature_max: float = 2.0,
        prior_temperature: float = 0.25,
        adapter_max_gate: float = 0.25,
        adapter_relative_bound: float = 0.10,
        abstention_min: float = 0.45,
        abstention_max: float = 0.85,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if adapter_rank <= 0 or adapter_rank > feature_dim:
            raise ValueError("adapter_rank must be in [1, feature_dim]")
        if count_prior <= 0:
            raise ValueError("count_prior must be positive")
        if not 0.0 < reliability_floor <= reliability_ceiling <= 1.0:
            raise ValueError("reliability bounds must satisfy 0 < floor <= ceiling <= 1")
        if not 0.0 < temperature_min <= temperature_max:
            raise ValueError("temperature bounds must satisfy 0 < min <= max")
        if prior_temperature <= 0:
            raise ValueError("prior_temperature must be positive")
        if not 0.0 <= adapter_max_gate <= 1.0:
            raise ValueError("adapter_max_gate must be in [0, 1]")
        if not 0.0 <= adapter_relative_bound <= 1.0:
            raise ValueError("adapter_relative_bound must be in [0, 1]")
        if not 0.0 <= abstention_min <= abstention_max <= 1.0:
            raise ValueError("abstention bounds must satisfy 0 <= min <= max <= 1")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.feature_dim = int(feature_dim)
        self.count_prior = float(count_prior)
        self.reliability_floor = float(reliability_floor)
        self.reliability_ceiling = float(reliability_ceiling)
        self.temperature_min = float(temperature_min)
        self.temperature_max = float(temperature_max)
        self.prior_temperature = float(prior_temperature)
        self.adapter_max_gate = float(adapter_max_gate)
        self.adapter_relative_bound = float(adapter_relative_bound)
        self.abstention_min = float(abstention_min)
        self.abstention_max = float(abstention_max)
        self.eps = float(eps)

        self.adapter_down = nn.Linear(self.feature_dim, adapter_rank, bias=False)
        self.adapter_up = nn.Linear(adapter_rank, self.feature_dim, bias=False)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_up.weight)

    @classmethod
    def from_config(cls, feature_dim: int, config: Any) -> "SupportReliabilityConditioner":
        keys = (
            "adapter_rank",
            "count_prior",
            "reliability_floor",
            "reliability_ceiling",
            "temperature_min",
            "temperature_max",
            "prior_temperature",
            "adapter_max_gate",
            "adapter_relative_bound",
            "abstention_min",
            "abstention_max",
            "eps",
        )
        defaults = dict(cls.__init__.__kwdefaults__ or {})
        defaults["adapter_rank"] = min(int(defaults["adapter_rank"]), int(feature_dim))
        kwargs = {key: _config_value(config, key, defaults[key]) for key in keys}
        return cls(feature_dim=feature_dim, **kwargs)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @property
    def contract(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "query_blind_conditioning": True,
                "base_logits_use_unadapted_features": True,
                "base_logits_equal_frozen_linear_head": True,
                "adapter_max_gate": self.adapter_max_gate,
                "adapter_relative_bound": self.adapter_relative_bound,
                "reliability_feature_names": RELIABILITY_FEATURE_NAMES,
            }
        )

    def _feature_matrix(self, value: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError(f"{name} must be a rank-2 tensor")
        if value.shape[0] == 0 or value.shape[1] != self.feature_dim:
            raise ValueError(
                f"{name} must have shape [N, {self.feature_dim}] with N > 0"
            )
        if not torch.is_floating_point(value):
            raise ValueError(f"{name} must use a floating-point dtype")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
        return value

    def _robust_summary(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        median = features.median(dim=0).values
        radii = torch.linalg.vector_norm(features - median, dim=1)
        median_radius = radii.median()
        mad_radius = (radii - median_radius).abs().median()
        scale = median_radius + 1.4826 * mad_radius + self.eps
        weights = 1.0 / (1.0 + (radii / scale).square())
        prototype = (weights[:, None] * features).sum(dim=0) / weights.sum().clamp_min(self.eps)
        centered_radius = torch.linalg.vector_norm(features - prototype, dim=1)
        within = (weights * centered_radius).sum() / weights.sum().clamp_min(self.eps)
        return prototype, within, weights.mean()

    def _class_reliability(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        novel_class_ids: torch.Tensor,
        base_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prototypes: list[torch.Tensor] = []
        within_values: list[torch.Tensor] = []
        inlier_values: list[torch.Tensor] = []
        counts: list[int] = []

        for class_id in novel_class_ids:
            class_features = support_features[support_labels == class_id]
            if class_features.shape[0] == 0:
                raise ValueError(f"support is missing novel class {int(class_id.item())}")
            prototype, within, inlier = self._robust_summary(class_features)
            prototypes.append(prototype)
            within_values.append(within)
            inlier_values.append(inlier)
            counts.append(int(class_features.shape[0]))

        robust_prototypes = torch.stack(prototypes)
        within = torch.stack(within_values)
        inlier_weight = torch.stack(inlier_values)
        count_tensor = torch.tensor(
            counts,
            dtype=support_features.dtype,
            device=support_features.device,
        )

        if robust_prototypes.shape[0] > 1:
            pairwise = torch.cdist(robust_prototypes, robust_prototypes)
            pairwise.fill_diagonal_(torch.inf)
            separation = pairwise.min(dim=1).values
        else:
            separation = torch.cdist(robust_prototypes, base_weights).min(dim=1).values

        count_score = count_tensor / (count_tensor + self.count_prior)
        compactness = separation / (separation + within + self.eps)
        class_balance = count_tensor / count_tensor.max().clamp_min(1.0)
        feature_vector = torch.stack(
            (count_score, compactness, inlier_weight, class_balance),
            dim=1,
        ).clamp(min=self.eps, max=1.0)
        reliability = torch.exp(torch.log(feature_vector).mean(dim=1)).clamp(
            min=self.reliability_floor,
            max=self.reliability_ceiling,
        )
        return robust_prototypes, feature_vector, reliability

    def apply_adapter(self, features: torch.Tensor, gate: torch.Tensor | float) -> torch.Tensor:
        features = self._feature_matrix(features, "features")
        gate_tensor = torch.as_tensor(gate, dtype=features.dtype, device=features.device)
        if gate_tensor.numel() != 1 or not torch.isfinite(gate_tensor):
            raise ValueError("adapter gate must be one finite scalar")
        gate_tensor = gate_tensor.reshape(()).clamp(min=0.0, max=self.adapter_max_gate)

        residual = self.adapter_up(F.gelu(self.adapter_down(features)))
        feature_norm = torch.linalg.vector_norm(features, dim=1, keepdim=True)
        residual_norm = torch.linalg.vector_norm(residual, dim=1, keepdim=True)
        allowed_norm = self.adapter_relative_bound * feature_norm
        clip_scale = (allowed_norm / residual_norm.clamp_min(self.eps)).clamp(max=1.0)
        return features + gate_tensor * clip_scale * residual

    def condition(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        base_weights: torch.Tensor,
        novel_class_ids: Iterable[int] | torch.Tensor,
    ) -> SupportCondition:
        """Build an immutable adaptation state from support data only."""
        support_features = self._feature_matrix(support_features, "support_features")
        base_weights = self._feature_matrix(base_weights, "base_weights")
        if not isinstance(support_labels, torch.Tensor) or support_labels.ndim != 1:
            raise ValueError("support_labels must be a rank-1 tensor")
        if support_labels.shape[0] != support_features.shape[0]:
            raise ValueError("support_labels length must match support_features")
        support_labels = support_labels.to(device=support_features.device, dtype=torch.long)
        expected_ids = _as_unique_class_ids(
            novel_class_ids,
            device=support_features.device,
            name="novel_class_ids",
        )
        observed_ids = torch.unique(support_labels, sorted=True)
        if not torch.equal(torch.sort(expected_ids).values, observed_ids):
            raise ValueError(
                "support labels must exactly match novel_class_ids; "
                f"expected={torch.sort(expected_ids).values.tolist()}, observed={observed_ids.tolist()}"
            )

        detached_base = base_weights.detach()
        robust, feature_vector, reliability = self._class_reliability(
            support_features,
            support_labels,
            expected_ids,
            detached_base,
        )
        robust_unit = F.normalize(robust, dim=1)
        base_unit = F.normalize(detached_base, dim=1)
        prior_attention = torch.softmax(
            robust_unit @ base_unit.transpose(0, 1) / self.prior_temperature,
            dim=1,
        )
        base_priors = F.normalize(prior_attention @ base_unit, dim=1)
        shrunk = F.normalize(
            reliability[:, None] * robust_unit
            + (1.0 - reliability[:, None]) * base_priors,
            dim=1,
        )

        global_reliability = reliability.numel() / torch.sum(1.0 / reliability.clamp_min(self.eps))
        adapter_gate = self.adapter_max_gate * global_reliability
        temperature = self.temperature_min + (
            self.temperature_max - self.temperature_min
        ) * (1.0 - reliability)
        abstention_threshold = self.abstention_min + (
            self.abstention_max - self.abstention_min
        ) * (1.0 - global_reliability)
        adapted_prototypes = F.normalize(self.apply_adapter(shrunk, adapter_gate), dim=1)

        return SupportCondition(
            novel_class_ids=expected_ids.detach().clone(),
            robust_prototypes=robust.detach().clone(),
            base_priors=base_priors.detach().clone(),
            adapted_prototypes=adapted_prototypes,
            reliability_features=feature_vector.detach().clone(),
            reliability=reliability.detach().clone(),
            temperature=temperature.detach().clone(),
            adapter_gate=adapter_gate.detach().clone(),
            abstention_threshold=abstention_threshold.detach().clone(),
        )

    def predict(
        self,
        query_features: torch.Tensor,
        base_weights: torch.Tensor,
        base_class_ids: Iterable[int] | torch.Tensor,
        condition: SupportCondition,
        *,
        base_bias: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Score base and novel classes jointly without adapting base logits."""
        query_features = self._feature_matrix(query_features, "query_features")
        base_weights = self._feature_matrix(base_weights, "base_weights")
        base_ids = _as_unique_class_ids(
            base_class_ids,
            device=query_features.device,
            name="base_class_ids",
        )
        if base_ids.numel() != base_weights.shape[0]:
            raise ValueError("base_class_ids length must match base_weights")
        if set(base_ids.tolist()) & set(condition.novel_class_ids.tolist()):
            raise ValueError("base and novel class ids must be disjoint")
        if condition.adapted_prototypes.device != query_features.device:
            raise ValueError("condition and query features must be on the same device")

        detached_base = base_weights.detach()
        base_logits = query_features @ detached_base.transpose(0, 1)
        if base_bias is not None:
            bias = torch.as_tensor(base_bias, dtype=base_logits.dtype, device=base_logits.device).reshape(-1)
            if bias.numel() != base_logits.shape[1] or not torch.isfinite(bias).all():
                raise ValueError("base_bias must be finite and match the number of base classes")
            base_logits = base_logits + bias.detach()

        adapted_query = self.apply_adapter(query_features, condition.adapter_gate)
        base_weight_scale = torch.linalg.vector_norm(detached_base, dim=1).median().clamp_min(self.eps)
        scaled_novel_weights = condition.adapted_prototypes * base_weight_scale
        novel_logits = (
            adapted_query @ scaled_novel_weights.transpose(0, 1)
        ) / condition.temperature[None, :]
        joint_logits = torch.cat((base_logits, novel_logits), dim=1)
        probabilities = torch.softmax(joint_logits, dim=1)
        confidence, prediction_index = probabilities.max(dim=1)
        joint_class_ids = torch.cat((base_ids, condition.novel_class_ids.to(base_ids.device)))

        return {
            "base_logits": base_logits,
            "novel_logits": novel_logits,
            "joint_logits": joint_logits,
            "probabilities": probabilities,
            "confidence": confidence,
            "accepted": confidence >= condition.abstention_threshold,
            "prediction_index": prediction_index,
            "prediction_label": joint_class_ids[prediction_index],
            "joint_class_ids": joint_class_ids,
        }


@register_task("GFS", "reliability_conditioned")
class task(Default_task):
    """Lightning binding for support-reliability-conditioned GFS episodes."""

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
        super().__init__(
            network,
            args_data,
            args_model,
            args_task,
            args_trainer,
            args_environment,
            metadata,
        )
        self.base_class_ids = tuple(int(value) for value in _config_value(args_task, "base_class_ids", ()))
        self.novel_class_ids = tuple(int(value) for value in _config_value(args_task, "novel_class_ids", ()))
        if not self.base_class_ids or not self.novel_class_ids:
            raise ValueError("GFS reliability task requires non-empty base_class_ids and novel_class_ids")
        if set(self.base_class_ids) & set(self.novel_class_ids):
            raise ValueError("base_class_ids and novel_class_ids must be disjoint")

        self.num_support = int(_config_value(args_task, "num_support", 0))
        if self.num_support <= 0:
            raise ValueError("task.num_support must be positive")
        self.freeze_encoder_base = bool(_config_value(args_task, "freeze_encoder_base", True))
        if not self.freeze_encoder_base:
            raise ValueError("P09 method contract requires freeze_encoder_base=true")
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)
        self.network.eval()

        feature_dim = int(_config_value(args_model, "output_dim", 0))
        conditioner_config = _config_value(args_task, "reliability_conditioner", {})
        self.conditioner = SupportReliabilityConditioner.from_config(feature_dim, conditioner_config)

    def train(self, mode: bool = True) -> "task":
        super().train(mode)
        if self.freeze_encoder_base:
            self.network.eval()
        return self

    def _single_file_id(self, batch: Mapping[str, torch.Tensor]) -> int:
        if "file_id" not in batch:
            raise ValueError("batch is missing file_id")
        values = torch.as_tensor(batch["file_id"]).reshape(-1)
        unique = torch.unique(values)
        if unique.numel() != 1:
            raise ValueError("each reliability-conditioned episode must contain one file/system id")
        return int(unique[0].item())

    def _extract_features(self, x: torch.Tensor, file_id: int) -> torch.Tensor:
        context = torch.no_grad() if self.freeze_encoder_base else nullcontext()
        with context:
            output = self.network(x, file_id, "classification", return_feature=True)
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("network must return (logits, features) when return_feature=true")
        features = output[1]
        if features.ndim == 3:
            features = features.mean(dim=1)
        elif features.ndim > 3:
            features = features.flatten(start_dim=1)
        if features.ndim != 2 or features.shape[1] != self.conditioner.feature_dim:
            raise RuntimeError(
                "pooled network feature dimension does not match task conditioner: "
                f"observed={tuple(features.shape)}, expected_dim={self.conditioner.feature_dim}"
            )
        return features.detach() if self.freeze_encoder_base else features

    def _base_classifier(self, file_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
        task_head = getattr(self.network, "task_head", None)
        heads = getattr(task_head, "mutiple_fc", None)
        if heads is None:
            raise RuntimeError("network task head does not expose frozen per-system linear classifiers")
        system_ids, _ = resolve_batch_metadata(self.metadata, file_id, device=device)
        key = str(int(system_ids.reshape(-1)[0].item()))
        if key not in heads:
            raise KeyError(f"missing frozen base classifier for system {key}")
        head = heads[key]
        indices = torch.tensor(self.base_class_ids, dtype=torch.long, device=head.weight.device)
        if indices.min() < 0 or indices.max() >= head.weight.shape[0]:
            raise ValueError("base_class_ids exceed the frozen classifier output range")
        weights = head.weight.index_select(0, indices).detach()
        bias = head.bias.index_select(0, indices).detach() if head.bias is not None else None
        return weights, bias

    def _episode_indices(self, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        labels = labels.to(dtype=torch.long).reshape(-1)
        expected = set(self.base_class_ids) | set(self.novel_class_ids)
        observed = set(torch.unique(labels).tolist())
        if observed != expected:
            raise ValueError(
                "episode labels must exactly match the configured base/novel partition; "
                f"expected={sorted(expected)}, observed={sorted(observed)}"
            )

        support_indices: list[torch.Tensor] = []
        query_indices: list[torch.Tensor] = []
        for class_id in self.base_class_ids:
            class_indices = torch.nonzero(labels == class_id, as_tuple=False).reshape(-1)
            if class_indices.numel() == 0:
                raise ValueError(f"episode has no base query for class {class_id}")
            query_indices.append(class_indices)
        for class_id in self.novel_class_ids:
            class_indices = torch.nonzero(labels == class_id, as_tuple=False).reshape(-1)
            if class_indices.numel() <= self.num_support:
                raise ValueError(
                    f"novel class {class_id} requires at least num_support + 1 samples"
                )
            support_indices.append(class_indices[: self.num_support])
            query_indices.append(class_indices[self.num_support :])
        return torch.cat(support_indices), torch.cat(query_indices)

    def _joint_targets(self, labels: torch.Tensor) -> torch.Tensor:
        joint_ids = self.base_class_ids + self.novel_class_ids
        targets = torch.empty_like(labels, dtype=torch.long)
        for index, class_id in enumerate(joint_ids):
            targets[labels == class_id] = index
        return targets

    def _shared_step(self, batch: Mapping[str, torch.Tensor], stage: str, task_id: bool = False) -> dict[str, torch.Tensor]:
        del task_id
        if "x" not in batch or "y" not in batch:
            raise ValueError("episode batch must contain x and y")
        file_id = self._single_file_id(batch)
        labels = torch.as_tensor(batch["y"], device=batch["x"].device).to(dtype=torch.long).reshape(-1)
        support_indices, query_indices = self._episode_indices(labels)
        features = self._extract_features(batch["x"], file_id)
        base_weights, base_bias = self._base_classifier(file_id, features.device)

        condition = self.conditioner.condition(
            features.index_select(0, support_indices),
            labels.index_select(0, support_indices),
            base_weights,
            self.novel_class_ids,
        )
        prediction = self.conditioner.predict(
            features.index_select(0, query_indices),
            base_weights,
            self.base_class_ids,
            condition,
            base_bias=base_bias,
        )
        query_labels = labels.index_select(0, query_indices)
        targets = self._joint_targets(query_labels)
        loss = F.cross_entropy(prediction["joint_logits"], targets)
        correct = prediction["prediction_index"] == targets
        base_mask = targets < len(self.base_class_ids)
        novel_mask = ~base_mask
        base_accuracy = correct[base_mask].float().mean()
        novel_accuracy = correct[novel_mask].float().mean()
        harmonic_mean = (
            2.0
            * base_accuracy
            * novel_accuracy
            / (base_accuracy + novel_accuracy).clamp_min(self.conditioner.eps)
        )
        accepted = prediction["accepted"]
        selective_risk = torch.where(
            accepted.any(),
            (~correct[accepted]).float().mean(),
            torch.ones((), dtype=loss.dtype, device=loss.device),
        )

        return {
            f"{stage}_loss": loss,
            f"{stage}_total_loss": loss,
            f"{stage}_joint_acc": correct.float().mean(),
            f"{stage}_base_acc": base_accuracy,
            f"{stage}_novel_acc": novel_accuracy,
            f"{stage}_harmonic_mean": harmonic_mean,
            f"{stage}_coverage": accepted.float().mean(),
            f"{stage}_selective_risk": selective_risk,
            f"{stage}_support_reliability": condition.reliability.mean(),
            f"{stage}_adapter_gate": condition.adapter_gate,
            f"{stage}_novel_temperature": condition.temperature.mean(),
        }


__all__ = [
    "RELIABILITY_FEATURE_NAMES",
    "SupportCondition",
    "SupportReliabilityConditioner",
    "task",
]
