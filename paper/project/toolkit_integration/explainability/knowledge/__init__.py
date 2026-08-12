"""
Knowledge Enhancement Module

This module provides domain knowledge and context processing capabilities
to enhance LLM-generated explanations with structured information about
mechanical fault diagnosis, terminology, and operational context.
"""

from .fault_knowledge_graph import FaultKnowledgeGraph, FaultType, SeverityLevel, FaultPattern
from .terminology_mapper import TerminologyMapper, TermMapping, TermCategory
from .context_processor import ContextProcessor, OperationalContext, HistoricalContext, SystemContext

__version__ = "0.1.0"
__all__ = [
    "FaultKnowledgeGraph",
    "FaultType",
    "SeverityLevel",
    "FaultPattern",
    "TerminologyMapper",
    "TermMapping",
    "TermCategory",
    "ContextProcessor",
    "OperationalContext",
    "HistoricalContext",
    "SystemContext"
]