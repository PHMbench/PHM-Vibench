"""Plot factory package.

Centralizes plotting / visualization utilities for PHM-Vibench. Mirrors the
repo's factory convention (see ``src/data_factory`` / ``src/model_factory``):
- ``plot_config.py`` — shared matplotlib/seaborn styling helpers.
- ``registry.py``    — ``PLOT_REGISTRY`` of reusable plotters (``P_<id>_<name>``).
- ``plot_factory.py``— factory entry point (``configure``, ``get_plotter``).
- ``pretraining_plot.py`` — pretraining-prediction visualization pipeline.

Note: legacy one-off plot scripts (A3-A8, A10) referenced the pre-factory
``model.`` / ``model_collection.`` / ``trainer.`` modules that no longer exist
in the factory-organized tree. They are archived under
``obsidian/history/dev/plot_legacy/`` rather than carried into ``src/``.
"""

from .plot_config import configure_matplotlib, set_chinese_font
from .plot_factory import configure, get_plotter, available_plotters
from .registry import PLOT_REGISTRY

__all__ = [
    "configure_matplotlib",
    "set_chinese_font",
    "configure",
    "get_plotter",
    "available_plotters",
    "PLOT_REGISTRY",
]
