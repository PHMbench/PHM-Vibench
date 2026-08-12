"""
LLM-Enhanced Explainable Fault Diagnosis Toolkit

A comprehensive toolkit for integrating Large Language Models
with fault diagnosis systems to provide natural language explanations
and interactive diagnostic conversations.
"""

from .core.explainer import LLMEnhancedExplainer
from .core.diagnostic_system import DiagnosticSystem
from .llm_integration.llm_manager import LLMManager
from .knowledge_enhancement.knowledge_base import FaultKnowledgeBase
from .interactive_interface.conversation_agent import ConversationAgent

__version__ = "1.0.0"
__author__ = "LLM-FD Research Team"
__email__ = "research@example.com"

# Main API
__all__ = [
    "LLMEnhancedExplainer",
    "DiagnosticSystem",
    "LLMManager",
    "FaultKnowledgeBase",
    "ConversationAgent"
]

# Convenience functions
def create_toolkit(llm_config=None, knowledge_config=None):
    """
    Create a complete LLM-enhanced fault diagnosis toolkit.

    Args:
        llm_config: Configuration for LLM providers
        knowledge_config: Configuration for knowledge base

    Returns:
        Configured DiagnosticSystem instance
    """
    return DiagnosticSystem(llm_config, knowledge_config)

def quick_explain(signal_data, fault_prediction, user_query=None):
    """
    Quick explanation generation with default settings.

    Args:
        signal_data: Input vibration signal data
        fault_prediction: Model prediction results
        user_query: Optional user query

    Returns:
        Generated explanation
    """
    explainer = LLMEnhancedExplainer()
    return explainer.explain(signal_data, fault_prediction, user_query)

# Version and compatibility
def get_version():
    """Get toolkit version information."""
    return {
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "supported_llm_providers": ["openai", "anthropic", "local"],
        "python_requires": ">=3.8",
        "dependencies": ["torch", "transformers", "openai", "anthropic"]
    }

def check_compatibility():
    """Check system compatibility."""
    import sys
    import torch

    return {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "compatible": sys.version_info >= (3, 8)
    }