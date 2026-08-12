"""
Model Adapters Package
====================

This package contains adapters for different transparent models used in the
LLM Explainable Fault Diagnosis Toolkit. Each adapter converts model-specific
outputs into a standardized Intermediate Representation (IR) that can be
consumed by the LLM layer.

Supported Models:
- TSPN (Transparent Signal Processing Network)
- TFON (Time-Frequency Operator Network)
- NNSPN (Neural Symbolic Processing Network)
- OperatorAttention / OperatorAttention_enhanced
- FuzzyLogic_v2
- MoE (Mixture of Experts) variants

Usage Example:
    from adapters import ModelAdapterFactory

    # Create an adapter for Operator Attention model
    adapter = ModelAdapterFactory.create_adapter("OperatorAttention_enhanced")

    # Convert model output to intermediate representation
    ir = adapter.to_intermediate_representation(model_output, context)
"""

from .model_adapter_base import (
    ModelAdapter,
    ModelAdapterFactory,
    ExplanationIR,
    FeatureImportance,
    SignalPath,
    AttentionMap,
    UncertaintyQuantification,
    EquipmentContext
)

from .operator_attention_adapter import OperatorAttentionAdapter
from .fuzzy_logic_adapter import FuzzyLogicAdapter
from .moe_adapter import MoEAdapter

# Convenience functions
def create_adapter(model_name: str, **kwargs) -> ModelAdapter:
    """
    Create an adapter for the specified model.

    Args:
        model_name: Name of the model (e.g., "TSPN", "TFON", "OperatorAttention")
        **kwargs: Additional arguments for the adapter

    Returns:
        Model adapter instance
    """
    return ModelAdapterFactory.create_adapter(model_name, **kwargs)

def list_supported_models() -> list:
    """
    List all supported model types.

    Returns:
        List of model names
    """
    return ModelAdapterFactory.list_supported_models()

# Export all adapters
__all__ = [
    'ModelAdapter',
    'ModelAdapterFactory',
    'ExplanationIR',
    'FeatureImportance',
    'SignalPath',
    'AttentionMap',
    'UncertaintyQuantification',
    'EquipmentContext',
    'OperatorAttentionAdapter',
    'FuzzyLogicAdapter',
    'MoEAdapter',
    'create_adapter',
    'list_supported_models'
]