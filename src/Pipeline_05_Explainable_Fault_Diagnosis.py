"""Explainable fault-diagnosis Pipeline over the shared classification runtime."""

from __future__ import annotations

from typing import Any

from src.runtime import run_classification_pipeline
from src.runtime.explainability import ExplainabilityHooks


def pipeline(args: Any) -> list[dict[str, Any]]:
    """Run classification with UXFD-specific evidence hooks."""
    return run_classification_pipeline(args, hooks=ExplainabilityHooks())
