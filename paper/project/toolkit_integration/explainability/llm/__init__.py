"""
LLM Enhanced Explainability Module

This module provides Large Language Model (LLM) enhanced explanations
for fault diagnosis, converting technical signal processing explanations
into natural language descriptions and enabling interactive diagnostic conversations.
"""

from .llm_explainer import LLMExplainer
from .signal_encoder import SignalEncoder
from .prompt_manager import PromptManager
from .llm_interface import LLMInterface
from .response_parser import ResponseParser

__version__ = "0.1.0"
__all__ = [
    "LLMExplainer",
    "SignalEncoder",
    "PromptManager",
    "LLMInterface",
    "ResponseParser"
]