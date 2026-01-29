"""NSN (Neural-Symbolic Network)

This module provides a **no-presets** configuration simplification layer for the UXFD merge.

- Runtime backbone stays: `TSPN_UXFD` (see `src/model_factory/X_model/TSPN_UXFD.py`)
- This wrapper adds an optional **flat-ish** config surface and maps it to the existing
  `model.uxfd.*` knobs so existing training code remains stable.

Supported inputs (best-effort, backward compatible):
- Legacy UXFD knobs: `model.uxfd.*` (passed through)
- Optional NSN knobs (inline in the same experiment YAML):
  - `model.decision_configs`: maps to `model.uxfd.fuzzy.*` / `model.uxfd.logic.*`
  - `model.signal_processing_2d`: maps to `model.uxfd.enable_sp2d` / `model.uxfd.sp2d` / `model.uxfd.fusion`
  - `STFT` token in `model.signal_processing_configs.layer*`: enables SP2D (STFT is not an `ALL_SP` key)

Note:
- 1D operator keys must come from `ALL_SP` and feature keys from `ALL_FE` (`src/model_factory/X_model/TSPN.py`).
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

from .TSPN import ALL_FE, ALL_SP
from .TSPN_UXFD import Model as _TSPNUXFD


class Model(_TSPNUXFD):
    """NSN wrapper over `TSPN_UXFD`."""

    def __init__(self, args: Any, metadata: Any = None):
        normalized = _normalize_nsn_args(args)
        super().__init__(normalized, metadata=metadata)


def _normalize_nsn_args(args: Any) -> Any:
    """Map NSN inline knobs into legacy `uxfd.*` knobs (without presets)."""
    normalized = copy.deepcopy(args)

    _map_signal_processing_2d(normalized)
    _map_decision_configs(normalized)
    _strip_stft_tokens_and_validate_1d_ops(normalized)
    _validate_feature_keys(normalized)

    return normalized


def _map_signal_processing_2d(args: Any) -> None:
    sp2d = _get_attr(args, "signal_processing_2d", None)
    if sp2d is None:
        return

    uxfd = _ensure_ns(args, "uxfd")

    # Enable flag
    enable = _get_attr(sp2d, "enable", None)
    if enable is not None and not hasattr(uxfd, "enable_sp2d"):
        uxfd.enable_sp2d = bool(enable)
    elif enable is not None:
        uxfd.enable_sp2d = bool(enable)

    # STFT config
    stft = _get_attr(sp2d, "stft", None)
    if stft is not None and not hasattr(uxfd, "sp2d"):
        uxfd.sp2d = _to_ns(stft)
    elif stft is not None:
        uxfd.sp2d = _to_ns(stft)

    # Fusion config
    fusion = _get_attr(sp2d, "fusion", None)
    if fusion is not None:
        uxfd.fusion = _to_ns(fusion)


def _map_decision_configs(args: Any) -> None:
    decision = _get_attr(args, "decision_configs", None)
    if decision is None:
        return

    decision_ns = _to_ns(decision)
    decision_type = str(getattr(decision_ns, "type", "") or "").strip().lower()

    uxfd = _ensure_ns(args, "uxfd")
    fuzzy = _ensure_ns(uxfd, "fuzzy")
    logic = _ensure_ns(uxfd, "logic")

    # Default: keep legacy values unless explicitly described by decision_configs.
    if decision_type:
        if decision_type in {"linear", "none", "base"}:
            fuzzy.enable = False
            logic.enable = False
        else:
            if "fuzzy" in decision_type:
                fuzzy.enable = True
            if "logic" in decision_type:
                logic.enable = True

    fuzzy_cfg = getattr(decision_ns, "fuzzy", None)
    if fuzzy_cfg is not None:
        _merge_ns_into(_to_ns(fuzzy_cfg), fuzzy)
        if getattr(fuzzy, "enable", None) is None:
            fuzzy.enable = True

    logic_cfg = getattr(decision_ns, "logic", None)
    if logic_cfg is not None:
        _merge_ns_into(_to_ns(logic_cfg), logic)
        if getattr(logic, "enable", None) is None:
            logic.enable = True


def _strip_stft_tokens_and_validate_1d_ops(args: Any) -> None:
    """Allow special token `STFT` in signal_processing_configs lists and map it to enable_sp2d."""
    sp_cfgs = _get_attr(args, "signal_processing_configs", None)
    if sp_cfgs is None:
        return

    # Convert dict-like to namespace so downstream TSPN can read it.
    if isinstance(sp_cfgs, dict):
        setattr(args, "signal_processing_configs", _to_ns(sp_cfgs))
        sp_cfgs = getattr(args, "signal_processing_configs")

    uxfd = _ensure_ns(args, "uxfd")
    found_stft = False

    for layer_key, value in list(getattr(sp_cfgs, "__dict__", {}).items()):
        if not isinstance(value, list):
            continue
        if "STFT" in value:
            found_stft = True
            cleaned = [x for x in value if x != "STFT"]
            setattr(sp_cfgs, layer_key, cleaned)

        # Validate remaining 1D operator keys early with a clearer error.
        invalid = [x for x in getattr(sp_cfgs, layer_key) if isinstance(x, str) and x not in ALL_SP]
        if invalid:
            allowed = ", ".join(sorted(ALL_SP.keys()))
            raise ValueError(
                f"NSN: unsupported 1D operator keys in model.signal_processing_configs.{layer_key}: {invalid}. "
                f"Allowed keys (ALL_SP): {allowed}. Note: 'STFT' is a special NSN token, not an ALL_SP key."
            )

    if found_stft:
        uxfd.enable_sp2d = True


def _validate_feature_keys(args: Any) -> None:
    feats = _get_attr(args, "feature_extractor_configs", None)
    if feats is None:
        return
    if not isinstance(feats, list):
        return
    invalid = [x for x in feats if isinstance(x, str) and x not in ALL_FE]
    if invalid:
        allowed = ", ".join(sorted(ALL_FE.keys()))
        raise ValueError(
            f"NSN: unsupported feature keys in model.feature_extractor_configs: {invalid}. Allowed keys (ALL_FE): {allowed}."
        )


def _get_attr(obj: Any, dotted: str, default: Any) -> Any:
    cur = obj
    for part in dotted.split("."):
        if cur is None or not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur


def _ensure_ns(obj: Any, attr: str) -> SimpleNamespace:
    if hasattr(obj, attr) and isinstance(getattr(obj, attr), SimpleNamespace):
        return getattr(obj, attr)
    if hasattr(obj, attr) and isinstance(getattr(obj, attr), dict):
        ns = _to_ns(getattr(obj, attr))
        setattr(obj, attr, ns)
        return ns
    ns = SimpleNamespace()
    setattr(obj, attr, ns)
    return ns


def _to_ns(value: Any) -> SimpleNamespace:
    if isinstance(value, SimpleNamespace):
        return value
    if isinstance(value, dict):
        ns = SimpleNamespace()
        for k, v in value.items():
            setattr(ns, k, _to_ns(v) if isinstance(v, dict) else v)
        return ns
    # fallback: wrap scalar
    return SimpleNamespace(value=value)


def _merge_ns_into(src: SimpleNamespace, dst: SimpleNamespace) -> None:
    for k, v in src.__dict__.items():
        if isinstance(v, dict):
            setattr(dst, k, _to_ns(v))
        else:
            setattr(dst, k, v)
