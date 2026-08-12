"""
Model Adapter Base Class
=======================

This module defines the base class for all model adapters in the LLM Explainable
Fault Diagnosis Toolkit. The adapter pattern allows us to support multiple transparent
models (TSPN, TFON, NNSPN, MoE, etc.) with a unified interface.

Author: LLM Explainable FD Toolkit
Date: 2025-01-15
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class FeatureImportance:
    """Represents feature importance information"""
    feature_name: str
    importance_score: float
    description: str
    evidence: Optional[str] = None


@dataclass
class SignalPath:
    """Represents signal processing pathway"""
    stage_name: str
    input_shape: tuple
    output_shape: tuple
    operation: str
    parameters: Dict[str, Any]
    attention_map: Optional[np.ndarray] = None


@dataclass
class AttentionMap:
    """Represents attention mechanism outputs"""
    attention_type: str  # 'temporal', 'frequency', 'spatial', 'operator'
    weights: np.ndarray
    positions: List[str]
    description: str


@dataclass
class UncertaintyQuantification:
    """Represents uncertainty information"""
    type: str  # 'epistemic', 'aleatoric', 'total'
    confidence_interval: tuple
    entropy: float
    calibration_score: float


@dataclass
class EquipmentContext:
    """Equipment and operational context"""
    equipment_type: str
    operating_conditions: Dict[str, Any]
    maintenance_history: Optional[List[Dict]] = None
    specifications: Optional[Dict[str, Any]] = None


@dataclass
class ExplanationIR:
    """Intermediate Representation for explanation generation"""
    # Core diagnosis
    diagnosis: Dict[str, float]  # fault_type -> confidence
    prediction_confidence: float

    # Feature-level explanations
    key_features: List[FeatureImportance]
    feature_ranking: List[str]  # feature names ordered by importance

    # Signal pathway information
    signal_pathway: List[SignalPath]
    processing_steps: List[str]

    # Attention and attribution
    attention_weights: Optional[AttentionMap]
    attribution_scores: Optional[Dict[str, float]]

    # Uncertainty information
    uncertainty: Optional[UncertaintyQuantification]

    # Context
    context: EquipmentContext

    # Model metadata
    model_name: str
    model_version: str
    explanation_timestamp: str


class ModelAdapter(ABC):
    """
    Base class for model adapters.

    Each model adapter converts model-specific outputs into the standardized
    Intermediate Representation (IR) that can be consumed by the LLM layer.
    """

    def __init__(self, model_name: str, model_version: str = "1.0"):
        """
        Initialize the adapter.

        Args:
            model_name: Name of the model (e.g., "TSPN", "TFON", "NNSPN")
            model_version: Version string of the model
        """
        self.model_name = model_name
        self.model_version = model_version
        self.supported_outputs = self._get_supported_output_types()

    @abstractmethod
    def _get_supported_output_types(self) -> List[str]:
        """Return list of supported output types from this model."""
        pass

    @abstractmethod
    def extract_diagnosis(self, model_output: Any) -> Dict[str, float]:
        """
        Extract fault diagnosis from model output.

        Args:
            model_output: Raw output from the model

        Returns:
            Dictionary mapping fault types to confidence scores
        """
        pass

    @abstractmethod
    def extract_features(self, model_output: Any) -> List[FeatureImportance]:
        """
        Extract feature importance information from model output.

        Args:
            model_output: Raw output from the model

        Returns:
            List of feature importance objects
        """
        pass

    @abstractmethod
    def extract_signal_pathway(self, model_output: Any) -> List[SignalPath]:
        """
        Extract signal processing pathway information.

        Args:
            model_output: Raw output from the model

        Returns:
            List of signal processing steps
        """
        pass

    def extract_attention(self, model_output: Any) -> Optional[AttentionMap]:
        """
        Extract attention weights if available.

        Default implementation returns None. Override if model provides attention.

        Args:
            model_output: Raw output from the model

        Returns:
            Attention map or None if not available
        """
        return None

    def extract_uncertainty(self, model_output: Any) -> Optional[UncertaintyQuantification]:
        """
        Extract uncertainty information if available.

        Default implementation returns None. Override if model provides uncertainty.

        Args:
            model_output: Raw output from the model

        Returns:
            Uncertainty quantification or None if not available
        """
        return None

    def extract_model_metadata(self, model_output: Any) -> Dict[str, Any]:
        """
        Extract additional model metadata.

        Args:
            model_output: Raw output from the model

        Returns:
            Dictionary of metadata
        """
        return {}

    def to_intermediate_representation(
        self,
        model_output: Any,
        context: EquipmentContext,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> ExplanationIR:
        """
        Convert model output to standardized intermediate representation.

        This is the main method that orchestrates all extraction methods.

        Args:
            model_output: Raw output from the model
            context: Equipment and operational context
            additional_info: Additional information for explanation

        Returns:
            Standardized intermediate representation
        """
        from datetime import datetime

        # Extract all components
        diagnosis = self.extract_diagnosis(model_output)
        features = self.extract_features(model_output)
        signal_pathway = self.extract_signal_pathway(model_output)
        attention = self.extract_attention(model_output)
        uncertainty = self.extract_uncertainty(model_output)
        metadata = self.extract_model_metadata(model_output)

        # Create feature ranking
        feature_ranking = [f.feature_name for f in sorted(features, key=lambda x: x.importance_score, reverse=True)]

        # Calculate overall prediction confidence
        prediction_confidence = max(diagnosis.values()) if diagnosis else 0.0

        # Extract processing steps
        processing_steps = [stage.stage_name for stage in signal_pathway]

        # Create IR
        ir = ExplanationIR(
            diagnosis=diagnosis,
            prediction_confidence=prediction_confidence,
            key_features=features,
            feature_ranking=feature_ranking,
            signal_pathway=signal_pathway,
            processing_steps=processing_steps,
            attention_weights=attention,
            attribution_scores={f.feature_name: f.importance_score for f in features},
            uncertainty=uncertainty,
            context=context,
            model_name=self.model_name,
            model_version=self.model_version,
            explanation_timestamp=datetime.now().isoformat()
        )

        # Add additional metadata
        if additional_info:
            ir.__dict__.update(additional_info)

        return ir

    def validate_output(self, ir: ExplanationIR) -> bool:
        """
        Validate the intermediate representation.

        Args:
            ir: Intermediate representation to validate

        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if not ir.diagnosis:
            return False

        if not ir.key_features:
            return False

        if not ir.context:
            return False

        # Check diagnosis probabilities
        total_prob = sum(ir.diagnosis.values())
        if total_prob < 0.95 or total_prob > 1.05:  # Allow small numerical errors
            return False

        # Check feature scores
        for feature in ir.key_features:
            if feature.importance_score < 0 or feature.importance_score > 1:
                return False

        # Check confidence values
        if ir.prediction_confidence < 0 or ir.prediction_confidence > 1:
            return False

        return True

    def format_for_llm(self, ir: ExplanationIR) -> Dict[str, Any]:
        """
        Format intermediate representation for LLM consumption.

        Args:
            ir: Intermediate representation

        Returns:
            Dictionary with formatted information for LLM
        """
        # Format diagnosis
        diagnosis_text = []
        for fault_type, confidence in ir.diagnosis.items():
            diagnosis_text.append(f"{fault_type}: {confidence:.2%}")

        # Format features
        feature_text = []
        for feature in ir.key_features[:5]:  # Top 5 features
            feature_text.append(f"{feature.feature_name} ({feature.importance_score:.3f}): {feature.description}")

        # Format uncertainty if available
        uncertainty_text = None
        if ir.uncertainty:
            uncertainty_text = f"Uncertainty type: {ir.uncertainty.type}, " \
                            f"95% CI: {ir.uncertainty.confidence_interval}, " \
                            f"Entropy: {ir.uncertainty.entropy:.3f}"

        return {
            "diagnosis": diagnosis_text,
            "confidence": ir.prediction_confidence,
            "key_features": feature_text,
            "signal_processing": " → ".join(ir.processing_steps),
            "uncertainty": uncertainty_text,
            "equipment": ir.context.equipment_type,
            "model": f"{ir.model_name} v{ir.model_version}"
        }


class ModelAdapterFactory:
    """Factory for creating model adapters."""

    _adapters = {}

    @classmethod
    def register_adapter(cls, model_name: str, adapter_class):
        """Register a new adapter class."""
        cls._adapters[model_name] = adapter_class

    @classmethod
    def create_adapter(
        cls,
        model_name: str,
        model_version: str = "1.0",
        **kwargs
    ) -> ModelAdapter:
        """
        Create an adapter instance.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            **kwargs: Additional arguments for adapter

        Returns:
            Model adapter instance
        """
        if model_name not in cls._adapters:
            raise ValueError(f"No adapter registered for model: {model_name}")

        adapter_class = cls._adapters[model_name]
        return adapter_class(model_name, model_version, **kwargs)

    @classmethod
    def list_supported_models(cls) -> List[str]:
        """List all supported model names."""
        return list(cls._adapters.keys())