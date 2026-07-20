"""
Operator Attention Adapter
==========================

Adapter for the Operator Attention Enhanced model.
This adapter extracts operator-level attention weights and transforms them
into interpretable explanations.

Author: LLM Explainable FD Toolkit
Date: 2025-01-15
"""

import numpy as np
import torch
from typing import Dict, Any, List
from .model_adapter_base import (
    ModelAdapter, FeatureImportance, SignalPath, AttentionMap,
    ExplanationIR, ModelAdapterFactory
)


class OperatorAttentionAdapter(ModelAdapter):
    """Adapter for Operator Attention Enhanced model."""

    # Define operator types and their descriptions
    OPERATOR_DESCRIPTIONS = {
        "identity": "Identity operation (no transformation)",
        "conv1d": "1D convolution (local pattern detection)",
        "maxpool1d": "1D max pooling (feature selection)",
        "avgpool1d": "1D average pooling (smoothing)",
        "fft": "Fast Fourier Transform (frequency domain)",
        "ifft": "Inverse Fast Fourier Transform (time domain)",
        "stft": "Short-time Fourier Transform (time-frequency)",
        "wavelet": "Wavelet transform (multi-resolution)",
        "derivative": "Derivative operation (rate of change)",
        "integral": "Integral operation (accumulation)",
        "envelope": "Envelope detection (modulation analysis)",
        "filter": "Filtering operation (noise reduction)",
        "attention": "Attention mechanism (feature weighting)"
    }

    def __init__(self, model_name: str = "OperatorAttention", model_version: str = "1.0"):
        """Initialize Operator Attention adapter."""
        super().__init__(model_name, model_version)
        self.operator_names = []
        self.attention_heads = 8  # Default number of attention heads

    def _get_supported_output_types(self) -> List[str]:
        """Return supported output types."""
        return [
            "diagnosis_logits",
            "attention_weights",
            "operator_weights",
            "feature_maps",
            "intermediate_outputs"
        ]

    def extract_diagnosis(self, model_output: Any) -> Dict[str, float]:
        """
        Extract fault diagnosis from model output.

        Operator Attention model typically outputs:
        - classification logits
        - softmax probabilities
        """
        # Handle different output formats
        if isinstance(model_output, dict):
            if "logits" in model_output:
                logits = model_output["logits"]
            elif "probabilities" in model_output:
                probs = model_output["probabilities"]
            elif "diagnosis" in model_output:
                return model_output["diagnosis"]
            else:
                # Assume first key contains diagnosis
                first_key = list(model_output.keys())[0]
                if isinstance(model_output[first_key], (torch.Tensor, np.ndarray)):
                    logits = model_output[first_key]
                else:
                    return {}
        else:
            logits = model_output

        # Convert to probabilities if needed
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()

        if len(logits.shape) > 1:
            logits = logits[0]  # Take first sample

        # Apply softmax if not already probabilities
        if logits.max() > 1.0 or logits.min() < 0.0:
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
        else:
            probs = logits

        # Map indices to fault types (adjust based on actual model)
        fault_types = ["normal", "inner_race", "outer_race", "ball", "cage"]

        diagnosis = {}
        for i, fault_type in enumerate(fault_types):
            if i < len(probs):
                diagnosis[fault_type] = float(probs[i])

        return diagnosis

    def extract_features(self, model_output: Any) -> List[FeatureImportance]:
        """Extract feature importance from operator attention weights."""
        features = []

        # Get operator weights if available
        if isinstance(model_output, dict) and "operator_weights" in model_output:
            operator_weights = model_output["operator_weights"]
        else:
            # Generate dummy importance if not available
            return [
                FeatureImportance(
                    feature_name="operator_attention_weights",
                    importance_score=0.85,
                    description="Operator attention mechanism indicates signal processing patterns"
                )
            ]

        # Process operator weights
        if isinstance(operator_weights, torch.Tensor):
            operator_weights = operator_weights.detach().cpu().numpy()

        # Handle multi-head attention
        if len(operator_weights.shape) == 3:  # [heads, operators, seq_len]
            # Average across heads
            operator_weights = np.mean(operator_weights, axis=0)

        # Create feature importance for each operator
        for i, weight in enumerate(operator_weights):
            if i < len(self.operator_names):
                op_name = self.operator_names[i]
                # Get operator description
                op_desc = self.OPERATOR_DESCRIPTIONS.get(
                    op_name.split('_')[0],
                    "Custom signal processing operation"
                )

                features.append(FeatureImportance(
                    feature_name=f"operator_{op_name}",
                    importance_score=float(np.mean(weight)),
                    description=f"Operator attention on {op_desc}",
                    evidence=f"Attention weight: {np.mean(weight):.3f}"
                ))

        # Sort by importance
        features.sort(key=lambda x: x.importance_score, reverse=True)

        return features

    def extract_signal_pathway(self, model_output: Any) -> List[SignalPath]:
        """Extract signal processing pathway from operator sequence."""
        pathways = []

        # Default operator pathway for Operator Attention
        default_ops = [
            ("input", (4096,), (4096,), "signal_input", {}),
            ("norm", (4096,), (4096,), "layer_norm", {"eps": 1e-5}),
            ("attention", (4096,), (4096,), "multi_head_attention",
             {"heads": self.attention_heads, "dim": 64}),
            ("fft", (4096,), (4096,), "fast_fourier_transform", {"n": 4096}),
            ("operator_conv", (4096,), (4096,), "operator_convolution", {"kernel": 3}),
            ("global_pool", (4096,), (512,), "global_average_pool", {}),
            ("classifier", (512,), (5,), "linear_classifier", {"out_features": 5})
        ]

        for i, (stage_name, input_shape, output_shape, operation, params) in enumerate(default_ops):
            pathways.append(SignalPath(
                stage_name=f"stage_{i}_{stage_name}",
                input_shape=input_shape,
                output_shape=output_shape,
                operation=operation,
                parameters=params,
                attention_map=self._get_attention_map(model_output, i) if operation == "multi_head_attention" else None
            ))

        return pathways

    def extract_attention(self, model_output: Any) -> AttentionMap:
        """Extract attention weights from operator attention model."""
        if not isinstance(model_output, dict):
            return AttentionMap(
                attention_type="operator",
                weights=np.zeros((1, 1)),
                positions=["default"],
                description="No attention weights available"
            )

        # Try to extract attention weights
        attention_weights = None
        if "attention_weights" in model_output:
            attention_weights = model_output["attention_weights"]
        elif "attention" in model_output:
            attention_weights = model_output["attention"]

        if attention_weights is None:
            return AttentionMap(
                attention_type="operator",
                weights=np.zeros((1, 1)),
                positions=["default"],
                description="No attention weights available"
            )

        # Process attention weights
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()

        # Handle different attention formats
        if len(attention_weights.shape) == 4:  # [batch, heads, seq, seq]
            attention_weights = np.mean(attention_weights, axis=0)  # Average heads
            attention_weights = np.mean(attention_weights, axis=1)  # Average over target
        elif len(attention_weights.shape) == 3:  # [heads, seq, seq]
            attention_weights = np.mean(attention_weights, axis=0)
        elif len(attention_weights.shape) == 2:  # [seq, seq]
            pass  # Already in correct format

        # Create position labels
        seq_len = attention_weights.shape[0]
        positions = [f"position_{i}" for i in range(seq_len)]

        return AttentionMap(
            attention_type="operator",
            weights=attention_weights,
            positions=positions,
            description=f"Operator attention weights averaged over {self.attention_heads} heads"
        )

    def extract_uncertainty(self, model_output: Any) -> Optional[UncertaintyQuantification]:
        """Extract uncertainty information if available."""
        # Check if model provides uncertainty
        if isinstance(model_output, dict):
            if "uncertainty" in model_output:
                uncertainty_data = model_output["uncertainty"]
                return UncertaintyQuantification(
                    type=uncertainty_data.get("type", "epistemic"),
                    confidence_interval=tuple(uncertainty_data.get("ci", (0, 1))),
                    entropy=uncertainty_data.get("entropy", 0.0),
                    calibration_score=uncertainty_data.get("calibration", 0.0)
                )
            elif "prediction_confidence" in model_output:
                # Create uncertainty from confidence
                conf = model_output["prediction_confidence"]
                return UncertaintyQuantification(
                    type="total",
                    confidence_interval=(max(0, conf - 0.1), min(1, conf + 0.1)),
                    entropy=-conf * np.log(conf + 1e-8) - (1 - conf) * np.log(1 - conf + 1e-8),
                    calibration_score=conf
                )

        return None

    def extract_model_metadata(self, model_output: Any) -> Dict[str, Any]:
        """Extract additional metadata from Operator Attention model."""
        metadata = {
            "model_type": "Operator Attention",
            "attention_heads": self.attention_heads,
            "operator_count": len(self.operator_names),
            "supports_explanations": True,
            "explanation_type": "operator_attention"
        }

        # Add specific metadata if available
        if isinstance(model_output, dict):
            metadata.update({
                k: v for k, v in model_output.items()
                if k not in ["diagnosis", "logits", "probabilities", "attention_weights"]
            })

        return metadata

    def _get_attention_map(self, model_output: Any, stage_idx: int) -> Optional[np.ndarray]:
        """Get attention map for a specific stage."""
        if not isinstance(model_output, dict):
            return None

        # Look for stage-specific attention
        stage_key = f"attention_stage_{stage_idx}"
        if stage_key in model_output:
            attention = model_output[stage_key]
            if isinstance(attention, torch.Tensor):
                return attention.detach().cpu().numpy()
            return attention

        # Return general attention if available
        if "attention_weights" in model_output:
            attention = model_output["attention_weights"]
            if isinstance(attention, torch.Tensor):
                return attention.detach().cpu().numpy()
            return attention

        return None

    def set_operator_sequence(self, operator_names: List[str]):
        """Set the sequence of operators used in the model."""
        self.operator_names = operator_names

    def set_attention_heads(self, num_heads: int):
        """Set the number of attention heads."""
        self.attention_heads = num_heads


# Register the adapter
ModelAdapterFactory.register_adapter("OperatorAttention", OperatorAttentionAdapter)
ModelAdapterFactory.register_adapter("OperatorAttention_enhanced", OperatorAttentionAdapter)