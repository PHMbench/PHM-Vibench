"""Protocol-only primitives for label-safe adaptation experiments.

B00 defines observability and ordering. It deliberately contains no Tent, SAR, CoTTA,
SHOT, optimizer construction, buffer policy, or Trainer implementation. Only unlabeled
single-stream source/TTA transactions are executable here; SFDA and label-bearing
regimes are schema-only until their own runtimes exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from src.config_schema import AdaptationProtocolConfig


_LABEL_ALIASES = frozenset(
    {
        "y",
        "label",
        "labels",
        "target",
        "targets",
        "fault_label",
        "target_label",
        "class_label",
        "future_y",
        "y_future",
    }
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
_EXECUTABLE_UNLABELED_REGIMES = frozenset(
    {"source_only", "episodic_tta", "online_tta", "continual_tta"}
)


def _normalized_metadata_key(key: str) -> str:
    return key.strip().lower().replace("-", "_").replace(" ", "_")


def _is_target_label_key(key: str) -> bool:
    normalized = _normalized_metadata_key(key)
    return (
        normalized in _LABEL_ALIASES
        or normalized.endswith("_label")
        or normalized.endswith("_labels")
    )


class AdaptationAlgorithm(Protocol):
    """Minimal unlabeled adapter shape used by B00 protocol tests."""

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
    """Return the only view an unlabeled adaptation transaction may receive."""

    if not isinstance(batch, Mapping):
        raise TypeError("adaptation batch must be a mapping")
    if "x" not in batch:
        raise KeyError("adaptation batch requires 'x'")

    view = {key: batch[key] for key in _ADAPTATION_KEYS if key in batch}
    if protocol.domain_boundary == "known" and "domain_id" in batch:
        view["domain_id"] = batch["domain_id"]

    for key in allowed_physical_metadata:
        if not isinstance(key, str) or not key.strip():
            raise TypeError("allowed physical metadata keys must be non-empty strings")
        if _is_target_label_key(key):
            raise ValueError(f"target label key {key!r} cannot enter adaptation view")
        if _normalized_metadata_key(key) == "domain_id" and protocol.domain_boundary == "hidden":
            raise ValueError("hidden domain boundaries cannot expose domain_id")
        if key in batch:
            view[key] = batch[key]

    leaked = sorted(key for key in view if _is_target_label_key(key))
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
    """Release a target label only under an explicitly label-bearing protocol."""

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
    """Execute one unlabeled source/TTA predict-update transaction.

    B00 intentionally refuses SFDA and label-bearing regimes. Their schemas are frozen,
    but executing them requires separate population or label-event lifecycles that belong
    to later bounded changes.
    """

    if protocol.regime not in _EXECUTABLE_UNLABELED_REGIMES:
        raise ValueError(
            f"B00 protocol helper does not execute regime={protocol.regime!r}; "
            "SFDA and label-bearing adaptation require their dedicated runtime."
        )

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
