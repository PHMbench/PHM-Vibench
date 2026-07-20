"""
Core Module

Provides the core functionality for the LLM-enhanced fault diagnosis
toolkit, including explanation generation and diagnostic system.
"""

from .explainer import LLMEnhancedExplainer
from .diagnostic_system import DiagnosticSystem

__version__ = "1.0.0"
__all__ = [
    "LLMEnhancedExplainer",
    "DiagnosticSystem"
]