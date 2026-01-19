from __future__ import annotations


def configure_matplotlib() -> None:
    """Headless-safe matplotlib defaults (no extra dependencies)."""
    import matplotlib

    matplotlib.use("Agg", force=True)

