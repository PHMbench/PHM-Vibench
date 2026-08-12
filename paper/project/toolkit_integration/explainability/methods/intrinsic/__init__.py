"""
Intrinsic explanation methods

These methods provide explanations based on the internal structure
and operations of the model itself.
"""

from .signal_path_explainer import SignalPathExplainer

__all__ = ["SignalPathExplainer"]