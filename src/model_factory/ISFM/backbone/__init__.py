"""Lazy exports for ISFM backbone implementations.

A selected backbone should not require dependencies used only by a different
backbone. Public names remain available through module-level ``__getattr__``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_BACKBONE_MODULES = {
    "B_01_basic_transformer": "B_01_basic_transformer",
    "B_03_FITS": "B_03_FITS",
    "B_04_Dlinear": "B_04_Dlinear",
    "B_05_Mamba": "B_05_Mamba",
    "B_06_TimesNet": "B_06_TimesNet",
    "B_07_TSMixer": "B_07_TSMixer",
    "B_08_PatchTST": "B_08_PatchTST",
    "B_09_FNO": "B_09_FNO",
    "B_10_VIBT": "B_10_VIBT",
    "B_11_MomentumEncoder": "B_11_MomentumEncoder",
}

__all__ = list(_BACKBONE_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _BACKBONE_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
