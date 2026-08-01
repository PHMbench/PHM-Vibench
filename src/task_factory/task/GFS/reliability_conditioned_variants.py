"""Explicit single-factor variants for the P09 reliability conditioner."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal, Sequence

import torch
import torch.nn.functional as F

from .reliability_conditioned import (
    RELIABILITY_FEATURE_NAMES,
    SupportCondition,
    SupportReliabilityConditioner,
)


VariantName = Literal[
    "A0",
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "R1",
    "R2",
    "R3",
    "R4",
]


def _global_reliability(reliability: torch.Tensor, epsilon: float) -> torch.Tensor:
    return reliability.numel() / torch.sum(1.0 / reliability.clamp_min(epsilon))


def _base_priors(
    prototypes: torch.Tensor,
    base_weights: torch.Tensor,
    prior_temperature: float,
) -> torch.Tensor:
    prototype_unit = F.normalize(prototypes, dim=1)
    base_unit = F.normalize(base_weights.detach(), dim=1)
    attention = torch.softmax(
        prototype_unit @ base_unit.transpose(0, 1) / prior_temperature,
        dim=1,
    )
    return F.normalize(attention @ base_unit, dim=1)


def _rebuild(
    conditioner: SupportReliabilityConditioner,
    condition: SupportCondition,
    *,
    robust_prototypes: torch.Tensor | None = None,
    base_priors: torch.Tensor | None = None,
    reliability_features: torch.Tensor | None = None,
    reliability: torch.Tensor | None = None,
    shrink_reliability: torch.Tensor | None = None,
    temperature: torch.Tensor | None = None,
    adapter_gate: torch.Tensor | float | None = None,
) -> SupportCondition:
    robust = condition.robust_prototypes if robust_prototypes is None else robust_prototypes
    priors = condition.base_priors if base_priors is None else base_priors
    features = (
        condition.reliability_features
        if reliability_features is None
        else reliability_features
    )
    reliability_value = condition.reliability if reliability is None else reliability
    shrink = reliability_value if shrink_reliability is None else shrink_reliability
    global_value = _global_reliability(reliability_value, conditioner.eps)
    gate = (
        conditioner.adapter_max_gate * global_value
        if adapter_gate is None
        else torch.as_tensor(
            adapter_gate,
            dtype=robust.dtype,
            device=robust.device,
        ).reshape(())
    )
    shrunk = F.normalize(
        shrink[:, None] * F.normalize(robust, dim=1)
        + (1.0 - shrink[:, None]) * F.normalize(priors, dim=1),
        dim=1,
    )
    adapted = F.normalize(conditioner.apply_adapter(shrunk, gate), dim=1)
    temperature_value = (
        conditioner.temperature_min
        + (conditioner.temperature_max - conditioner.temperature_min)
        * (1.0 - reliability_value)
        if temperature is None
        else temperature
    )
    threshold = conditioner.abstention_min + (
        conditioner.abstention_max - conditioner.abstention_min
    ) * (1.0 - global_value)
    return SupportCondition(
        novel_class_ids=condition.novel_class_ids,
        robust_prototypes=robust.detach().clone(),
        base_priors=priors.detach().clone(),
        adapted_prototypes=adapted.detach().clone(),
        reliability_features=features.detach().clone(),
        reliability=reliability_value.detach().clone(),
        temperature=temperature_value.detach().clone(),
        adapter_gate=gate.detach().clone(),
        abstention_threshold=threshold.detach().clone(),
    )


def condition_variant(
    conditioner: SupportReliabilityConditioner,
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    base_weights: torch.Tensor,
    novel_class_ids: Sequence[int],
    variant: VariantName,
    *,
    fixed_temperature: float = 1.0,
    fixed_gate: float = 0.25,
) -> SupportCondition:
    """Construct one preregistered immutable condition variant."""
    full = conditioner.condition(
        support_features, support_labels, base_weights, novel_class_ids
    )
    if variant in {"A0", "A6"}:
        return full
    if variant == "A1":
        means = torch.stack(
            [
                support_features[support_labels == class_id].mean(dim=0)
                for class_id in full.novel_class_ids
            ]
        )
        priors = _base_priors(means, base_weights, conditioner.prior_temperature)
        return _rebuild(
            conditioner,
            full,
            robust_prototypes=means,
            base_priors=priors,
        )
    if variant == "A2":
        return _rebuild(
            conditioner,
            full,
            shrink_reliability=torch.ones_like(full.reliability),
        )
    if variant == "A3":
        return _rebuild(
            conditioner,
            full,
            temperature=torch.full_like(full.temperature, float(fixed_temperature)),
        )
    if variant == "A4":
        return _rebuild(conditioner, full, adapter_gate=0.0)
    if variant == "A5":
        return _rebuild(conditioner, full, adapter_gate=float(fixed_gate))
    if variant.startswith("R"):
        index = int(variant[1:]) - 1
        if index not in range(len(RELIABILITY_FEATURE_NAMES)):
            raise ValueError(f"unknown reliability leave-one-out variant: {variant}")
        feature_values = full.reliability_features.clone()
        feature_values[:, index] = 1.0
        reliability = torch.exp(torch.log(feature_values).mean(dim=1)).clamp(
            min=conditioner.reliability_floor,
            max=conditioner.reliability_ceiling,
        )
        return _rebuild(
            conditioner,
            full,
            reliability_features=feature_values,
            reliability=reliability,
        )
    raise ValueError(f"unknown conditioner variant: {variant}")


def predict_variant(
    conditioner: SupportReliabilityConditioner,
    query_features: torch.Tensor,
    base_weights: torch.Tensor,
    base_class_ids: Sequence[int],
    condition: SupportCondition,
    variant: VariantName,
    *,
    base_bias: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    prediction = conditioner.predict(
        query_features,
        base_weights,
        base_class_ids,
        condition,
        base_bias=base_bias,
    )
    if variant == "A6":
        prediction = dict(prediction)
        prediction["accepted"] = torch.ones_like(
            prediction["accepted"], dtype=torch.bool
        )
    return prediction


__all__ = ["VariantName", "condition_variant", "predict_variant"]
