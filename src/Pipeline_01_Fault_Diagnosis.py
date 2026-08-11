"""Maintained fault-diagnosis Pipeline using the shared classification runtime."""

from __future__ import annotations

from typing import Any

from src.runtime import run_classification_pipeline
from src.runtime.classification import _write_aggregate_outputs


def pipeline(args: Any) -> list[dict[str, Any]]:
    """Run the standard classification train/test lifecycle."""
    return run_classification_pipeline(args)
