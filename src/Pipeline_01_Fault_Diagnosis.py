"""Maintained fault-diagnosis Pipeline using the shared classification runtime."""

from __future__ import annotations

from typing import Any

from src.runtime import run_classification_pipeline


def pipeline(args: Any) -> dict[str, Any]:
    """Run the standard classification lifecycle and return direct output paths."""
    return run_classification_pipeline(args)
