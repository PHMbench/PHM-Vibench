"""Locked baseline implementations for the P09-G060 feature evaluator."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def joint_prediction(
    base_logits: torch.Tensor,
    novel_logits: torch.Tensor,
    *,
    threshold: torch.Tensor | float = 0.0,
    base_class_ids: Sequence[int] = (0, 1),
    novel_class_ids: Sequence[int] = (2, 3),
) -> dict[str, torch.Tensor]:
    if base_logits.ndim != 2 or novel_logits.ndim != 2:
        raise ValueError("base_logits and novel_logits must be rank-2 tensors")
    if base_logits.shape[0] != novel_logits.shape[0]:
        raise ValueError("base and novel logits must have the same observations")
    base_ids = torch.as_tensor(
        base_class_ids, dtype=torch.long, device=base_logits.device
    ).reshape(-1)
    novel_ids = torch.as_tensor(
        novel_class_ids, dtype=torch.long, device=base_logits.device
    ).reshape(-1)
    if base_ids.numel() != base_logits.shape[1]:
        raise ValueError("base_class_ids must match base logit columns")
    if novel_ids.numel() != novel_logits.shape[1]:
        raise ValueError("novel_class_ids must match novel logit columns")
    joint_ids = torch.cat((base_ids, novel_ids))
    if torch.unique(joint_ids).numel() != joint_ids.numel():
        raise ValueError("base and novel class ids must be unique and disjoint")
    if not torch.isfinite(base_logits).all() or not torch.isfinite(novel_logits).all():
        raise ValueError("joint logits contain non-finite values")
    logits = torch.cat((base_logits, novel_logits), dim=1)
    probabilities = torch.softmax(logits, dim=1)
    confidence, prediction_index = probabilities.max(dim=1)
    threshold_value = torch.as_tensor(
        threshold, dtype=confidence.dtype, device=confidence.device
    )
    return {
        "base_logits": base_logits,
        "novel_logits": novel_logits,
        "joint_logits": logits,
        "probabilities": probabilities,
        "confidence": confidence,
        "accepted": confidence >= threshold_value,
        "prediction_index": prediction_index,
        "prediction_label": joint_ids[prediction_index],
        "joint_class_ids": joint_ids,
    }


def _novel_scale(base_weights: torch.Tensor, epsilon: float = 1.0e-8) -> torch.Tensor:
    return torch.linalg.vector_norm(base_weights, dim=1).median().clamp_min(epsilon)


def _class_prototypes(
    support: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_ids: Sequence[int],
    reducer: str,
) -> torch.Tensor:
    if support.ndim != 2 or support.shape[0] == 0:
        raise ValueError("support must be a non-empty rank-2 tensor")
    if labels.ndim != 1 or labels.shape[0] != support.shape[0]:
        raise ValueError("labels must be rank-1 and match support")
    if not torch.is_floating_point(support) or not torch.isfinite(support).all():
        raise ValueError("support must contain finite floating-point features")
    expected = torch.as_tensor(class_ids, dtype=torch.long, device=labels.device).reshape(-1)
    if expected.numel() == 0 or torch.unique(expected).numel() != expected.numel():
        raise ValueError("class_ids must be non-empty and unique")
    observed = torch.unique(labels.to(dtype=torch.long), sorted=True)
    if not torch.equal(torch.sort(expected).values, observed):
        raise ValueError(
            "support labels must exactly match the requested novel classes; "
            f"expected={torch.sort(expected).values.tolist()}, observed={observed.tolist()}"
        )
    values = []
    for class_id in expected:
        selected = support[labels == class_id]
        if reducer == "mean":
            values.append(selected.mean(dim=0))
        elif reducer == "median":
            values.append(selected.median(dim=0).values)
        else:
            raise ValueError(f"unknown prototype reducer: {reducer}")
    return torch.stack(values)


def frozen_base_logits(
    query: torch.Tensor, base_weights: torch.Tensor, base_bias: torch.Tensor
) -> torch.Tensor:
    return query @ base_weights.detach().transpose(0, 1) + base_bias.detach()


def predict_b0(
    query: torch.Tensor,
    support: torch.Tensor,
    labels: torch.Tensor,
    base_weights: torch.Tensor,
    base_bias: torch.Tensor,
    *,
    base_class_ids: Sequence[int] = (0, 1),
    novel_class_ids: Sequence[int] = (2, 3),
) -> dict[str, torch.Tensor]:
    prototypes = F.normalize(
        _class_prototypes(
            support, labels, class_ids=novel_class_ids, reducer="mean"
        ),
        dim=1,
    )
    novel_weights = prototypes * _novel_scale(base_weights)
    return joint_prediction(
        frozen_base_logits(query, base_weights, base_bias),
        query @ novel_weights.transpose(0, 1),
        base_class_ids=base_class_ids,
        novel_class_ids=novel_class_ids,
    )


def predict_b1(
    query: torch.Tensor,
    support: torch.Tensor,
    labels: torch.Tensor,
    source_base_prototypes: torch.Tensor,
    *,
    temperature: float,
    base_class_ids: Sequence[int] = (0, 1),
    novel_class_ids: Sequence[int] = (2, 3),
) -> dict[str, torch.Tensor]:
    if not 0.0 < float(temperature):
        raise ValueError("temperature must be positive")
    if source_base_prototypes.shape[0] != len(base_class_ids):
        raise ValueError("source_base_prototypes must match base_class_ids")
    novel = _class_prototypes(
        support, labels, class_ids=novel_class_ids, reducer="mean"
    )
    prototypes = F.normalize(torch.cat((source_base_prototypes, novel), dim=0), dim=1)
    logits = F.normalize(query, dim=1) @ prototypes.transpose(0, 1) / float(temperature)
    boundary = len(base_class_ids)
    return joint_prediction(
        logits[:, :boundary],
        logits[:, boundary:],
        base_class_ids=base_class_ids,
        novel_class_ids=novel_class_ids,
    )


def predict_b2(
    query: torch.Tensor,
    support: torch.Tensor,
    labels: torch.Tensor,
    base_weights: torch.Tensor,
    base_bias: torch.Tensor,
    *,
    base_class_ids: Sequence[int] = (0, 1),
    novel_class_ids: Sequence[int] = (2, 3),
) -> dict[str, torch.Tensor]:
    prototypes = F.normalize(
        _class_prototypes(
            support, labels, class_ids=novel_class_ids, reducer="median"
        ), dim=1
    )
    return joint_prediction(
        frozen_base_logits(query, base_weights, base_bias),
        query @ (prototypes * _novel_scale(base_weights)).transpose(0, 1),
        base_class_ids=base_class_ids,
        novel_class_ids=novel_class_ids,
    )


class DistanceWeightNet(nn.Module):
    """Source-trained RRPN-style scalar weight from normalized support radius."""

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def prototypes(
        self,
        support: torch.Tensor,
        labels: torch.Tensor,
        novel_class_ids: Sequence[int] = (2, 3),
    ) -> torch.Tensor:
        _class_prototypes(
            support, labels, class_ids=novel_class_ids, reducer="mean"
        )
        values = []
        for class_id in novel_class_ids:
            selected = support[labels == class_id]
            center = selected.median(dim=0).values
            radius = torch.linalg.vector_norm(selected - center, dim=1)
            normalized = radius / radius.median().clamp_min(1.0e-8)
            weights = torch.sigmoid(self.network(normalized[:, None])).reshape(-1)
            values.append(
                (weights[:, None] * selected).sum(dim=0)
                / weights.sum().clamp_min(1.0e-8)
            )
        return torch.stack(values)


def predict_b3(
    module: DistanceWeightNet,
    query: torch.Tensor,
    support: torch.Tensor,
    labels: torch.Tensor,
    base_weights: torch.Tensor,
    base_bias: torch.Tensor,
    *,
    base_class_ids: Sequence[int] = (0, 1),
    novel_class_ids: Sequence[int] = (2, 3),
) -> dict[str, torch.Tensor]:
    prototypes = F.normalize(
        module.prototypes(support, labels, novel_class_ids), dim=1
    )
    return joint_prediction(
        frozen_base_logits(query, base_weights, base_bias),
        query @ (prototypes * _novel_scale(base_weights)).transpose(0, 1),
        base_class_ids=base_class_ids,
        novel_class_ids=novel_class_ids,
    )


class QueryBlindSetAttention(nn.Module):
    def __init__(self, feature_dim: int, heads: int = 4) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            feature_dim, heads, batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)

    def adapt(
        self,
        base_weights: torch.Tensor,
        support: torch.Tensor,
        labels: torch.Tensor,
        novel_class_ids: Sequence[int] = (2, 3),
    ) -> torch.Tensor:
        novel = _class_prototypes(
            support, labels, class_ids=novel_class_ids, reducer="mean"
        )
        tokens = F.normalize(torch.cat((base_weights.detach(), novel), dim=0), dim=1)
        attended, _ = self.attention(
            tokens[None, :, :], tokens[None, :, :], tokens[None, :, :],
            need_weights=False,
        )
        return F.normalize(self.norm(tokens + attended.squeeze(0)), dim=1)


def predict_b4(
    module: QueryBlindSetAttention,
    query: torch.Tensor,
    support: torch.Tensor,
    labels: torch.Tensor,
    base_weights: torch.Tensor,
    *,
    base_class_ids: Sequence[int] = (0, 1),
    novel_class_ids: Sequence[int] = (2, 3),
) -> dict[str, torch.Tensor]:
    prototypes = module.adapt(base_weights, support, labels, novel_class_ids)
    logits = query @ (prototypes * _novel_scale(base_weights)).transpose(0, 1)
    boundary = len(base_class_ids)
    return joint_prediction(
        logits[:, :boundary],
        logits[:, boundary:],
        base_class_ids=base_class_ids,
        novel_class_ids=novel_class_ids,
    )


def predict_b5(
    query: torch.Tensor,
    support: torch.Tensor,
    labels: torch.Tensor,
    base_weights: torch.Tensor,
    base_bias: torch.Tensor,
    *,
    novel_scale: float,
    novel_bias: float,
    base_class_ids: Sequence[int] = (0, 1),
    novel_class_ids: Sequence[int] = (2, 3),
) -> dict[str, torch.Tensor]:
    if not 0.0 < float(novel_scale):
        raise ValueError("novel_scale must be positive")
    base = predict_b0(
        query,
        support,
        labels,
        base_weights,
        base_bias,
        base_class_ids=base_class_ids,
        novel_class_ids=novel_class_ids,
    )
    return joint_prediction(
        base["base_logits"],
        base["novel_logits"] * float(novel_scale) + float(novel_bias),
        base_class_ids=base_class_ids,
        novel_class_ids=novel_class_ids,
    )


class FixedPrompt(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.prompt = nn.Parameter(torch.zeros(feature_dim))


def predict_b6(
    module: FixedPrompt,
    query: torch.Tensor,
    support: torch.Tensor,
    labels: torch.Tensor,
    base_weights: torch.Tensor,
    base_bias: torch.Tensor,
    *,
    base_class_ids: Sequence[int] = (0, 1),
    novel_class_ids: Sequence[int] = (2, 3),
) -> dict[str, torch.Tensor]:
    prototypes = _class_prototypes(
        support, labels, class_ids=novel_class_ids, reducer="mean"
    ) + module.prompt
    adapted_query = query + module.prompt
    novel = adapted_query @ (
        F.normalize(prototypes, dim=1) * _novel_scale(base_weights)
    ).transpose(0, 1)
    return joint_prediction(
        frozen_base_logits(query, base_weights, base_bias),
        novel,
        base_class_ids=base_class_ids,
        novel_class_ids=novel_class_ids,
    )


def ridge_novel_head(
    support: torch.Tensor,
    labels: torch.Tensor,
    *,
    ridge: float,
    novel_class_ids: Sequence[int] = (2, 3),
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < float(ridge):
        raise ValueError("ridge must be positive")
    _class_prototypes(
        support, labels, class_ids=novel_class_ids, reducer="mean"
    )
    design = torch.cat(
        (support, torch.ones(support.shape[0], 1, device=support.device)), dim=1
    )
    targets = torch.stack(
        [labels == int(class_id) for class_id in novel_class_ids], dim=1
    ).to(dtype=support.dtype)
    identity = torch.eye(design.shape[1], dtype=support.dtype, device=support.device)
    identity[-1, -1] = 0.0
    solution = torch.linalg.solve(
        design.transpose(0, 1) @ design + float(ridge) * identity,
        design.transpose(0, 1) @ targets,
    )
    return solution[:-1].transpose(0, 1), solution[-1]


def predict_b7(
    query: torch.Tensor,
    support: torch.Tensor,
    labels: torch.Tensor,
    base_weights: torch.Tensor,
    base_bias: torch.Tensor,
    *,
    ridge: float,
    base_class_ids: Sequence[int] = (0, 1),
    novel_class_ids: Sequence[int] = (2, 3),
) -> dict[str, torch.Tensor]:
    weights, bias = ridge_novel_head(
        support, labels, ridge=ridge, novel_class_ids=novel_class_ids
    )
    return joint_prediction(
        frozen_base_logits(query, base_weights, base_bias),
        query @ weights.transpose(0, 1) + bias,
        base_class_ids=base_class_ids,
        novel_class_ids=novel_class_ids,
    )


class FixedGateAdapter(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        rank: int = 16,
        fixed_gate: float = 0.25,
        relative_bound: float = 0.10,
    ) -> None:
        super().__init__()
        self.down = nn.Linear(feature_dim, rank, bias=False)
        self.up = nn.Linear(rank, feature_dim, bias=False)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        self.fixed_gate = float(fixed_gate)
        self.relative_bound = float(relative_bound)

    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.up(F.gelu(self.down(features)))
        allowed = self.relative_bound * torch.linalg.vector_norm(
            features, dim=1, keepdim=True
        )
        observed = torch.linalg.vector_norm(residual, dim=1, keepdim=True)
        scale = (allowed / observed.clamp_min(1.0e-8)).clamp(max=1.0)
        return features + self.fixed_gate * scale * residual


def predict_b8(
    module: FixedGateAdapter,
    query: torch.Tensor,
    support: torch.Tensor,
    labels: torch.Tensor,
    base_weights: torch.Tensor,
    base_bias: torch.Tensor,
    *,
    base_class_ids: Sequence[int] = (0, 1),
    novel_class_ids: Sequence[int] = (2, 3),
) -> dict[str, torch.Tensor]:
    prototypes = _class_prototypes(
        support, labels, class_ids=novel_class_ids, reducer="mean"
    )
    adapted_prototypes = F.normalize(module.adapt(prototypes), dim=1)
    adapted_query = module.adapt(query)
    novel = adapted_query @ (
        adapted_prototypes * _novel_scale(base_weights)
    ).transpose(0, 1)
    return joint_prediction(
        frozen_base_logits(query, base_weights, base_bias),
        novel,
        base_class_ids=base_class_ids,
        novel_class_ids=novel_class_ids,
    )


def method_signature(
    implementation: Callable[..., Mapping[str, torch.Tensor]],
    *,
    module: nn.Module | None = None,
    settings: Mapping[str, Any] | None = None,
    checkpoint_sha256: str | None = None,
) -> str:
    """Hash the executed implementation without trusting its display arm ID."""

    if not callable(implementation):
        raise TypeError("implementation must be the actual prediction callable")
    digest = hashlib.sha256()
    digest.update(implementation.__module__.encode("utf-8"))
    digest.update(implementation.__qualname__.encode("utf-8"))
    digest.update(inspect.getsource(implementation).encode("utf-8"))
    digest.update(json.dumps(settings or {}, sort_keys=True).encode("utf-8"))
    digest.update((checkpoint_sha256 or "").encode("ascii"))
    if module is not None:
        digest.update(module.__class__.__qualname__.encode("utf-8"))
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prediction_signature(prediction: Mapping[str, torch.Tensor]) -> str:
    """Hash observable behavior on a fixed sentinel episode."""

    required = (
        "base_logits",
        "novel_logits",
        "joint_logits",
        "probabilities",
        "accepted",
        "joint_class_ids",
    )
    if any(name not in prediction for name in required):
        raise ValueError("prediction lacks fields required for a functional signature")
    digest = hashlib.sha256()
    for name in required:
        tensor = prediction[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


ARM_PREDICTORS: Mapping[str, Callable[..., Mapping[str, torch.Tensor]]] = {
    "B0": predict_b0,
    "B1": predict_b1,
    "B2": predict_b2,
    "B3": predict_b3,
    "B4": predict_b4,
    "B5": predict_b5,
    "B6": predict_b6,
    "B7": predict_b7,
    "B8": predict_b8,
    "A7": predict_b0,
}


def resolve_arm_predictor(
    arm_id: str,
) -> Callable[..., Mapping[str, torch.Tensor]]:
    try:
        return ARM_PREDICTORS[arm_id]
    except KeyError as exc:
        raise KeyError(f"arm has no registered feature predictor: {arm_id}") from exc


def trainable_parameters(module: nn.Module | None) -> int:
    return 0 if module is None else sum(
        value.numel() for value in module.parameters() if value.requires_grad
    )


__all__ = [
    "ARM_PREDICTORS",
    "DistanceWeightNet",
    "FixedGateAdapter",
    "FixedPrompt",
    "QueryBlindSetAttention",
    "frozen_base_logits",
    "joint_prediction",
    "method_signature",
    "prediction_signature",
    "predict_b0",
    "predict_b1",
    "predict_b2",
    "predict_b3",
    "predict_b4",
    "predict_b5",
    "predict_b6",
    "predict_b7",
    "predict_b8",
    "ridge_novel_head",
    "resolve_arm_predictor",
    "trainable_parameters",
]
