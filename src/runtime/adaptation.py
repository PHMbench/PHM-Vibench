"""Protocol-only primitives for label-safe adaptation experiments.

B00 defines observability and ordering.  It deliberately contains no Tent, SAR, CoTTA,
SHOT, optimizer construction, buffer policy, or Trainer implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from src.config_schema import AdaptationProtocolConfig


_TARGET_LABEL_KEYS = frozenset(
    {"y", "label", "labels", "target", "targets", "future_y", "y_future"}
)
_ADAPTATION_KEYS = (
    "x",
    "mask",
    "file_id",
    "sample_id",
    "timestamp",
    "sequence_id",
)
_EVALUATION_KEYS = (
    "y",
    "file_id",
    "sample_id",
    "timestamp",
    "sequence_id",
    "domain_id",
)


class AdaptationAlgorithm(Protocol):
    """Minimal runtime shape used by the protocol tests and future B01 runtime."""

    def predict(self, view: Mapping[str, Any]) -> Any: ...

    def adapt(self, view: Mapping[str, Any]) -> Any: ...


@dataclass(frozen=True)
class ProtocolStep:
    prediction: Any
    evaluation: Mapping[str, Any]
    update_result: Any


def build_adaptation_view(
    batch: Mapping[str, Any],
    protocol: AdaptationProtocolConfig,
    *,
    allowed_physical_metadata: Sequence[str] = (),
) -> dict[str, Any]:
    """Return the only view an unsupervised adaptation transaction may receive."""

    if not isinstance(batch, Mapping):
        raise TypeError("adaptation batch must be a mapping")
    if "x" not in batch:
        raise KeyError("adaptation batch requires 'x'")

    view = {key: batch[key] for key in _ADAPTATION_KEYS if key in batch}
    if protocol.domain_boundary == "known" and "domain_id" in batch:
        view["domain_id"] = batch["domain_id"]

    for key in allowed_physical_metadata:
        if not isinstance(key, str) or not key:
            raise TypeError("allowed physical metadata keys must be non-empty strings")
        if key in _TARGET_LABEL_KEYS:
            raise ValueError(f"target label key {key!r} cannot enter adaptation view")
        if key == "domain_id" and protocol.domain_boundary == "hidden":
            raise ValueError("hidden domain boundaries cannot expose domain_id")
        if key in batch:
            view[key] = batch[key]

    leaked = sorted(_TARGET_LABEL_KEYS.intersection(view))
    if leaked:
        raise AssertionError(f"adaptation view leaked target labels: {leaked}")
    return view


def build_evaluation_view(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Return evaluator-only labels and traceability fields."""

    if not isinstance(batch, Mapping):
        raise TypeError("evaluation batch must be a mapping")
    if "y" not in batch:
        raise KeyError("evaluation batch requires 'y'")
    return {key: batch[key] for key in _EVALUATION_KEYS if key in batch}


def label_event_for_update(
    batch: Mapping[str, Any],
    protocol: AdaptationProtocolConfig,
    *,
    current_step: int,
) -> Any:
    """Release a target label only under an explicitly labelled protocol."""

    if protocol.target_label_access == "none":
        raise PermissionError("this adaptation protocol forbids target-label updates")
    if "y" not in batch:
        raise KeyError("label-bearing update event requires 'y'")
    if protocol.target_label_access == "online_supervised":
        return batch["y"]

    available = batch.get("label_available_step")
    if isinstance(current_step, bool) or not isinstance(current_step, int) or current_step < 0:
        raise ValueError("current_step must be a non-negative integer")
    if isinstance(available, bool) or not isinstance(available, int) or available < 0:
        raise ValueError(
            "delayed-label update requires non-negative integer label_available_step"
        )
    if current_step < available:
        raise RuntimeError(
            f"target label is unavailable at step {current_step}; "
            f"label_available_step={available}"
        )
    return batch["y"]


def execute_protocol_step(
    adapter: AdaptationAlgorithm,
    batch: Mapping[str, Any],
    protocol: AdaptationProtocolConfig,
    *,
    allowed_physical_metadata: Sequence[str] = (),
) -> ProtocolStep:
    """Execute one label-isolated predict/update transaction."""

    adapt_view = build_adaptation_view(
        batch,
        protocol,
        allowed_physical_metadata=allowed_physical_metadata,
    )
    evaluation = build_evaluation_view(batch)

    if protocol.regime == "source_only":
        return ProtocolStep(
            prediction=adapter.predict(adapt_view),
            evaluation=evaluation,
            update_result={"update_applied": False, "reason": "source_only"},
        )

    if protocol.timing == "predict_then_update":
        prediction = adapter.predict(adapt_view)
        update_result = adapter.adapt(adapt_view)
    else:
        update_result = adapter.adapt(adapt_view)
        prediction = adapter.predict(adapt_view)

    return ProtocolStep(
        prediction=prediction,
        evaluation=evaluation,
        update_result=update_result,
    )


__all__ = [
    "AdaptationAlgorithm",
    "ProtocolStep",
    "build_adaptation_view",
    "build_evaluation_view",
    "execute_protocol_step",
    "label_event_for_update",
]
