"""B02: a complete ordered target pass with one prequential Tent update per batch."""
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
from src.task_factory.Components.tent import Tent


def run_tent_stream(
    model: nn.Module,
    loader: DataLoader,
    protocol: AdaptationProtocolConfig,
    *,
    checkpoint_path: str | Path,
    learning_rate: float,
    evaluate: Callable[[torch.Tensor, Mapping[str, Any]], None],
) -> Tent:
    """Strict source load -> predict -> evaluate -> one entropy/Adam update.

    The caller supplies source-selected hyperparameters, the B01-compatible target
    loader, device and existing population metric accumulator. Only x enters Tent.
    Returns its completed algorithm state, not a second result format. Publish the
    caller's existing results only after successful return; discard partial model,
    optimizer and metrics on error. No target selection, extra pass, reset or resume
    of an entire experiment is provided by this full-pass entrypoint.
    """
    if not isinstance(protocol, AdaptationProtocolConfig):
        raise TypeError("protocol must be AdaptationProtocolConfig")
    protocol = AdaptationProtocolConfig.model_validate(protocol.model_dump())
    if protocol.regime not in {"online_tta", "continual_tta"}:
        raise ValueError("B02 requires online_tta or continual_tta")
    for key, value in dict(source_access="checkpoint_only", target_label_access="none",
                          timing="predict_then_update", state_persistence="persistent",
                          label_space="closed_set").items():
        if getattr(protocol, key) != value:
            raise ValueError(f"B02 requires protocol.{key}={value!r}")
    if not callable(evaluate):
        raise TypeError("B02 requires an evaluator callback")
    size = _check_loader(loader)
    load_ckpt(model, checkpoint_path, strict=True)
    adapter = Tent(model, learning_rate=learning_rate)
    for batch in loader:
        view = build_adaptation_view(batch, protocol)
        evaluation = build_evaluation_view(batch)
        if view.get("mask") is not None:
            raise ValueError("B02 does not support masked classification inputs")
        prediction = adapter.predict(view["x"])
        evaluate(prediction, evaluation)
        adapter.adapt()
    if adapter.num_samples != size:
        raise RuntimeError(f"Tent evaluated {adapter.num_samples} samples; expected {size}")
    return adapter


__all__ = ["run_tent_stream"]
