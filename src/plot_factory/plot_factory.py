"""Factory entry point for plot_factory.

Provides a thin, lazy-resolving façade over the registered plotters so callers
do not pay the torch/data-factory import cost unless they actually run a
pipeline.
"""

from __future__ import annotations

from typing import Any, Callable

from .plot_config import configure_matplotlib
from .registry import PLOT_REGISTRY


def configure(style: str = "ieee", font_lang: str = "en", **kwargs: Any) -> None:
    """Convenience wrapper around :func:`plot_config.configure_matplotlib`."""
    configure_matplotlib(style=style, font_lang=font_lang, **kwargs)


# Lazy module path for each registered pipeline (imported only on demand).
_PIPELINE_MODULES = {
    "P_01_pretraining_prediction": ("pretraining_plot", "plot_pipeline"),
}


def get_plotter(name: str) -> Callable[..., Any]:
    """Resolve a plotting pipeline by registry ID.

    For helper IDs already in ``PLOT_REGISTRY`` (``P_00_*``) the registered
    callable is returned directly. For pipeline IDs (``P_01_*``) the relevant
    module is imported lazily and the pipeline callable returned.
    """
    if name in PLOT_REGISTRY.available():
        return PLOT_REGISTRY.get(name)

    if name in _PIPELINE_MODULES:
        module_attr = _PIPELINE_MODULES[name]
        import importlib

        module = importlib.import_module(f"{__name__.rsplit('.', 1)[0]}.{module_attr[0]}")
        return getattr(module, module_attr[1])

    raise KeyError(
        f"Plotter '{name}' is not registered. Available: "
        f"{sorted(set(PLOT_REGISTRY.available()) | set(_PIPELINE_MODULES))}"
    )


def available_plotters() -> list[str]:
    """Return all registered plotter IDs (helpers + lazy pipelines)."""
    return sorted(set(PLOT_REGISTRY.available()) | set(_PIPELINE_MODULES))


__all__ = ["configure", "get_plotter", "available_plotters"]
