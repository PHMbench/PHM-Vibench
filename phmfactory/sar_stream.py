"""B03: ordered checkpoint-only SAR adaptation with evaluator-isolated target labels."""
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from phmfactory.adaptation_protocol import build_adaptation_view, build_evaluation_view
from phmfactory.source_stream import _check_loader
from src.config_schema import AdaptationProtocolConfig
from src.model_factory.model_factory import load_ckpt
from src.task_factory.Components.sar import SAR


def run_sar_stream(
    model: nn.Module,
    loader: DataLoader,
    protocol: AdaptationProtocolConfig,
    *,
    checkpoint_path: str | Path,
    learning_rate: float,
    margin_e0: float,
    evaluate: Callable[[torch.Tensor, Mapping[str, Any]], None],
    rho: float = 0.05,
) -> SAR:
    """Strict source load -> first prediction/evaluation -> bounded SAR update.

    The caller fixes source-selected hyperparameters and supplies the existing ordered
    target loader and population evaluator. Only x reaches SAR. The first forward is
    evaluated before the current batch update; the second forward exists only for the
    sharpness-aware update and is never exposed as a prediction. This is batch-level
    prequential evaluation, not sample-causal inference when normalization couples a batch.
    """
    if not isinstance(protocol, AdaptationProtocolConfig):
        raise TypeError("protocol must be AdaptationProtocolConfig")
    protocol = AdaptationProtocolConfig.model_validate(protocol.model_dump())
    if protocol.regime not in {"online_tta", "continual_tta"}:
        raise ValueError("B03 requires online_tta or continual_tta")
    for key, value in dict(source_access="checkpoint_only", target_label_access="none",
                           timing="predict_then_update", state_persistence="persistent",
                           label_space="closed_set").items():
        if getattr(protocol, key) != value:
            raise ValueError(f"B03 requires protocol.{key}={value!r}")
    if not callable(evaluate):
        raise TypeError("B03 requires an evaluator callback")
    size = _check_loader(loader)
    load_ckpt(model, checkpoint_path, strict=True)
    adapter = SAR(model, learning_rate=learning_rate, margin_e0=margin_e0, rho=rho)
    for batch in loader:
        view = build_adaptation_view(batch, protocol)
        evaluation = build_evaluation_view(batch)
        if view.get("mask") is not None:
            raise ValueError("B03 does not support masked classification inputs")
        prediction = adapter.predict(view["x"])
        evaluate(prediction, evaluation)
        adapter.adapt()
    if adapter.num_samples != size or adapter.num_batches != len(loader):
        raise RuntimeError(
            f"SAR evaluated {adapter.num_samples} samples/{adapter.num_batches} batches; "
            f"expected {size}/{len(loader)}"
        )
    return adapter


__all__ = ["run_sar_stream"]
