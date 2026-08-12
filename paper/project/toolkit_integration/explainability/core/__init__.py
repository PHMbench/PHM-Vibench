"""
Core module for Explainable FD Toolkit

This module provides the core interfaces and data structures for the
Explainable Fault Diagnosis Toolkit.
"""

from .signal_data import SignalData
from .explanation import Explanation
from .base_explainer import BaseExplainer
from .interfaces import (
    ExplainabilityMethod,
    ModelPlugin,
    BaseExplainerAdapter,
    BaseModelAdapter
)

__all__ = [
    'SignalData',
    'Explanation',
    'BaseExplainer',
    'ExplainabilityMethod',
    'ModelPlugin',
    'BaseExplainerAdapter',
    'BaseModelAdapter'
]