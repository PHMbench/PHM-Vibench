"""Lazy public exports for the ISFM model family.

Importing one concrete ISFM model must not import every embedding, backbone,
task head, or model implementation. Some non-selected components have optional
third-party dependencies, so eager package imports make the maintained model
path depend on unrelated research modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_SUBPACKAGE_EXPORTS = {"embedding", "backbone", "task_head"}
_MODEL_EXPORTS = {
    "M_01_ISFM": "M_01_ISFM",
    "M_02_ISFM": "M_02_ISFM",
    "M_02_ISFM_heterogeneous_batch": "M_02_ISFM_heterogeneous_batch",
    "M_03_ISFM": "M_03_ISFM",
}

__all__ = [
    "embedding",
    "backbone",
    "task_head",
    "M_01_ISFM",
    "M_02_ISFM",
    "M_02_ISFM_heterogeneous_batch",
    "M_03_ISFM",
]


def __getattr__(name: str) -> Any:
    """Resolve public ISFM exports only when they are requested."""
    if name in _SUBPACKAGE_EXPORTS:
        value = import_module(f"{__name__}.{name}")
    elif name in _MODEL_EXPORTS:
        module = import_module(f"{__name__}.{_MODEL_EXPORTS[name]}")
        value = module.Model
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
