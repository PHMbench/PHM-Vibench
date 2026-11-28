"""Ordered Tent/BN-only target passes with persistent or known-domain reset state."""
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
import random

import numpy as np
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
    update: bool = True,
) -> Tent:
    """Strict load -> pre-update prediction -> evaluation -> update or BN control.

    Episodic/domain resets occur before a new, legally visible domain begins. The
    reset reference is the configured source adaptation state (after Tent selects
    its trainable BN affine parameters), including optimizer and all three RNGs.
    Source-only inference remains in run_source_only_stream. The enclosing task owns
    population metrics and result files; no partial success is returned on failure.
    """
    if not isinstance(protocol, AdaptationProtocolConfig):
        raise TypeError("protocol must be AdaptationProtocolConfig")
    protocol = AdaptationProtocolConfig.model_validate(protocol.model_dump())
    if protocol.regime not in {"online_tta", "continual_tta", "episodic_tta"}:
        raise ValueError("B02 requires online_tta, continual_tta or episodic_tta")
    for key, value in dict(source_access="checkpoint_only", target_label_access="none",
                          timing="predict_then_update", label_space="closed_set").items():
        if getattr(protocol, key) != value:
            raise ValueError(f"B02 requires protocol.{key}={value!r}")
    resetting = protocol.state_persistence != "persistent"
    if resetting and protocol.domain_boundary != "known":
        raise ValueError("B02 reset lifecycle requires declared, known domain boundaries")
    if not callable(evaluate) or type(update) is not bool:
        raise TypeError("Tent requires an evaluator callback and a boolean update flag")
    size = _check_loader(loader)
    load_ckpt(model, checkpoint_path, strict=True)
    adapter = Tent(model, learning_rate=learning_rate)
    # DataLoader consumes a base-seed draw when its iterator is created. Capture
    # the adaptation start after that draw so reset matches a fresh episode.
    iterator = iter(loader)
    # No snapshot is allocated for the ordinary persistent path.
    initial = adapter.state_dict() if resetting else None
    python_rng = random.getstate() if resetting else None
    numpy_rng = np.random.get_state() if resetting else None
    previous_domain = None
    count = updates = resets = 0
    for batch in iterator:
        if resetting:
            if "domain_id" in batch:
                domains = torch.as_tensor(batch["domain_id"]).reshape(-1).tolist()
            else:
                domains = [loader.dataset.metadata[int(i)]["Domain_id"]
                           for i in batch["file_id"]]
            if len(set(domains)) != 1:
                raise ValueError("A reset stream batch must not straddle domain boundaries")
            domain = domains[0]
            if previous_domain is not None and domain != previous_domain:
                adapter.load_state_dict(initial)
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
                resets += 1
            previous_domain = domain
        view = build_adaptation_view(batch, protocol)
        if view.get("mask") is not None:
            raise ValueError("Tent does not support masked classification inputs")
        prediction = adapter.predict(view["x"])
        evaluate(prediction, build_evaluation_view(batch))
        if update:
            adapter.adapt()
            updates += 1
        else:
            adapter.discard()
        count += prediction.shape[0]
    if count != size:
        raise RuntimeError(f"Tent evaluated {count} samples; expected {size}")
    # Per-episode optimizer counters remain in state_dict; full-stream counts are
    # separate so reset cannot erase the reported computation/sample budget.
    adapter.stream_num_samples = count
    adapter.stream_num_updates = updates
    adapter.num_resets = resets
    return adapter


__all__ = ["run_tent_stream"]
