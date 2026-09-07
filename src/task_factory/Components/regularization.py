"""Regularization terms used by Task Factory objectives."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import isfinite
from numbers import Real
from typing import Any

import torch


_SUPPORTED_METHODS = frozenset({"l1", "l2"})


def _as_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"{context} must be a mapping, got {type(value).__name__}")


def _resolve_methods(reg_config: Any) -> dict[str, Any]:
    """Resolve the two historical explicit configuration shapes without guessing."""

    if reg_config is None:
        return {}
    outer = _as_mapping(reg_config, context="task.regularization")
    if not outer:
        return {}

    if "flag" in outer or "method" in outer:
        flag = outer.get("flag", False)
        if not isinstance(flag, bool):
            raise TypeError("task.regularization.flag must be boolean")
        methods = _as_mapping(
            outer.get("method", {}),
            context="task.regularization.method",
        )
        if not flag:
            if methods:
                raise ValueError(
                    "task.regularization.flag=false conflicts with non-empty method"
                )
            return {}
        if not methods:
            raise ValueError(
                "task.regularization.flag=true requires at least one method"
            )
        return methods

    if "regularization" in outer:
        raw_methods = outer["regularization"]
        if raw_methods in (None, False):
            return {}
        if raw_methods is True:
            raise ValueError(
                "task.regularization.regularization=true requires a method mapping"
            )
        methods = _as_mapping(
            raw_methods,
            context="task.regularization.regularization",
        )
        if not methods:
            return {}
        return methods

    # A direct mapping such as {"l1": 1e-4} is explicit and unambiguous.
    return outer


def calculate_regularization(
    reg_config: Any,
    params: Iterable[torch.nn.Parameter],
) -> dict[str, torch.Tensor]:
    """Calculate the declared regularization over every trainable parameter."""

    methods = _resolve_methods(reg_config)
    trainable_params = [parameter for parameter in params if parameter.requires_grad]

    if not methods:
        if trainable_params:
            return {"total": trainable_params[0].new_zeros(())}
        return {"total": torch.tensor(0.0, dtype=torch.float32)}

    normalized_methods = {str(name).strip().lower(): weight for name, weight in methods.items()}
    unknown = sorted(set(normalized_methods) - _SUPPORTED_METHODS)
    if unknown:
        raise ValueError(
            f"Unknown regularization method(s): {unknown}. Available methods: "
            f"{', '.join(sorted(_SUPPORTED_METHODS))}. PHMFactory does not "
            "silently skip requested objective terms."
        )

    weights: dict[str, float] = {}
    for method, raw_weight in normalized_methods.items():
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, Real):
            raise TypeError(
                f"regularization weight for {method!r} must be numeric, "
                f"got {raw_weight!r}"
            )
        weight = float(raw_weight)
        if not isfinite(weight) or weight < 0:
            raise ValueError(
                f"regularization weight for {method!r} must be finite and "
                f"non-negative, got {raw_weight!r}"
            )
        weights[method] = weight

    if not trainable_params:
        raise RuntimeError(
            "regularization is enabled but the task has no trainable parameters"
        )

    total = trainable_params[0].new_zeros(())
    losses: dict[str, torch.Tensor] = {}
    for method, weight in weights.items():
        if weight == 0:
            continue
        current = trainable_params[0].new_zeros(())
        if method == "l1":
            for parameter in trainable_params:
                current = current + parameter.abs().sum()
        else:
            for parameter in trainable_params:
                current = current + parameter.square().sum()

        weighted = current * weight
        losses[method] = weighted
        total = total + weighted

    losses["total"] = total
    return losses


if __name__ == "__main__":
    model = torch.nn.Linear(10, 2)
    print(calculate_regularization({"l1": 0.01, "l2": 0.005}, model.parameters()))
