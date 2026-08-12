"""
Conversation Module

This module provides interactive conversation capabilities for LLM-enhanced
fault diagnosis, enabling multi-turn dialogues between engineers and the AI system.
"""

from .conversation_engine import ConversationEngine
from .query_processor import QueryProcessor
from .feedback_collector import FeedbackCollector

__version__ = "0.1.0"
__all__ = [
    "ConversationEngine",
    "QueryProcessor",
    "FeedbackCollector"
]