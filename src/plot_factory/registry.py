"""Plot factory registry.

Holds a ``PLOT_REGISTRY`` of reusable plotting helpers and pipelines. Component
IDs follow the repo convention with a ``P_`` prefix (P for Plot), mirroring
``E_*`` / ``B_*`` / ``H_*`` used by the model factory.
"""

from __future__ import annotations

from ..utils.registry import Registry

PLOT_REGISTRY: Registry = Registry()


def _register_basics() -> None:
    """Register standalone helpers (no heavy deps) eagerly.

    Plotting pipelines that require torch / the data/model factory are resolved
    lazily through :func:`plot_factory.get_plotter` to keep import time light.
    """
    from .plot_config import configure_matplotlib, set_chinese_font

    # Register helpers into the registry without re-decorating the functions.
    PLOT_REGISTRY._items["P_00_configure_matplotlib"] = configure_matplotlib
    PLOT_REGISTRY._items["P_00_set_chinese_font"] = set_chinese_font


_register_basics()


__all__ = ["PLOT_REGISTRY"]
