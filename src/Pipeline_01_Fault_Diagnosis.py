"""Maintained fault-diagnosis Pipeline using the shared classification runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.runtime import (
    ClassificationContext,
    ClassificationHooks,
    run_classification_pipeline,
)


_RENDERER_FIELDS = (
    "n_fft",
    "hop_length",
    "win_length",
    "window",
    "window_periodic",
    "center",
    "pad_mode",
    "normalized",
    "onesided",
    "representation",
    "scaling",
    "resize",
    "normalization",
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): _json_ready(item)
            for key, item in vars(value).items()
        }
    if hasattr(value, "item"):
        return value.item()
    return value


def _renderer_from_config(args_model: Any) -> dict[str, Any]:
    renderer = getattr(args_model, "renderer", None)
    if renderer is None:
        raise ValueError("P01 grouped protocol requires model.renderer configuration.")
    missing = [field for field in _RENDERER_FIELDS if not hasattr(renderer, field)]
    if missing:
        raise ValueError(f"P01 renderer configuration is missing field(s) {missing}.")
    return {field: getattr(renderer, field) for field in _RENDERER_FIELDS}


def build_p01_data_protocol_summary(
    data_factory: Any,
    args_model: Any,
    model: Any = None,
    loader_probe: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    grouped = getattr(data_factory, "grouped_protocol", None)
    split = getattr(data_factory, "split_summary", None)
    if not grouped or not isinstance(split, dict):
        raise ValueError("P01 data protocol summary requires an active grouped split.")

    renderer = _renderer_from_config(args_model)
    if model is not None:
        identity = getattr(model, "renderer_identity", None)
        observed = identity() if callable(identity) else None
        if observed is not None and observed != renderer:
            raise ValueError(
                "Executed model renderer differs from the frozen renderer config."
            )

    return _json_ready(
        {
            "status": "succeeded",
            "scope": "C01_data_protocol_only",
            "endpoint": {
                "name": grouped["endpoint"],
                "admitted_labels": grouped["admitted_labels"],
                "excluded_label_0_reason": grouped[
                    "excluded_label_0_reason"
                ],
                "inferential_unit": grouped["inferential_unit"],
                "verified_run_identity": grouped["verified_run_identity"],
                "observation_hierarchy": grouped["observation_hierarchy"],
                "identity_limit": grouped["identity_limit"],
                "target_label_access_boundary": grouped[
                    "target_label_access_boundary"
                ],
            },
            "split": split,
            "renderer": {
                "identity": renderer,
                "matched_conditions": ["M2", "M3", "M4", "M5"],
                "data_fitting_boundary": (
                    "none: renderer parameters are frozen configuration values"
                ),
            },
            "loader_probe": loader_probe,
            "provenance": provenance,
            "scientific_boundary": (
                "This artifact validates identity, grouping, fitting boundaries, "
                "and paired-view construction only; it contains no performance result."
            ),
        }
    )


def write_p01_data_protocol_summary(
    path: str | Path,
    data_factory: Any,
    args_model: Any,
    model: Any = None,
    loader_probe: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_p01_data_protocol_summary(
        data_factory,
        args_model,
        model=model,
        loader_probe=loader_probe,
        provenance=provenance,
    )
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


class _P01DataProtocolHooks(ClassificationHooks):
    def after_stack_built(self, context: ClassificationContext) -> None:
        if getattr(context.data_factory, "grouped_protocol", None) is None:
            return
        if str(getattr(context.args_model, "name", "")) != "P01Alignment":
            return
        path = write_p01_data_protocol_summary(
            context.path / "data_protocol_summary.json",
            context.data_factory,
            context.args_model,
            model=context.model,
        )
        print(f"[P01 DATA PROTOCOL] {path}")


def pipeline(args: Any) -> list[dict[str, Any]]:
    """Run the standard classification train/test lifecycle."""
    return run_classification_pipeline(args, hooks=_P01DataProtocolHooks())
