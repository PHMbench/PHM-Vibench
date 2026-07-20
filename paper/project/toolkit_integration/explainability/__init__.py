"""
Unified Explainability Toolkit for Fault Diagnosis

A specialized explainability toolkit for rotating machinery fault diagnosis,
focusing on physical operator networks and time series analysis.

This toolkit provides:
- Intrinsic physical explanations (signal path, operator importance)
- Post-hoc explanations (Integrated Gradients, SHAP)
- Time series specific attribution methods
- Unified interface for all explanation types

Example:
    >>> from explainability import UnifiedExplainer
    >>> explainer = UnifiedExplainer(model, method='signal_path')
    >>> explanation = explainer.explain(signal_data)
    >>> explanation.visualize()
"""

from .core.unified_explainer import UnifiedExplainer
from .core.explanation import Explanation
from .core.base_explainer import BaseExplainer

__version__ = "0.1.0"
__all__ = ["UnifiedExplainer", "Explanation", "BaseExplainer"]