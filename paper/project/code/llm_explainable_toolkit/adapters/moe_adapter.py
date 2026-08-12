"""
MoE (Mixture of Experts) Adapter
=================================

Adapter for the Mixture of Experts model.
This adapter extracts expert routing information, expert contributions,
and specialization patterns to create interpretable explanations.

Author: LLM Explainable FD Toolkit
Date: 2025-01-15
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
from .model_adapter_base import (
    ModelAdapter, FeatureImportance, SignalPath, ModelAdapterFactory
)


@dataclass
class ExpertInfo:
    """Information about an expert in MoE"""
    expert_id: int
    expert_name: str
    specialization: str  # What type of patterns this expert specializes in
    activation_rate: float  # How often this expert is activated
    contribution_score: float  # Overall contribution to decisions
    confidence: float  # Confidence in expert's predictions


@dataclass
class RoutingDecision:
    """Routing decision information"""
    input_sample_id: str
    selected_experts: List[int]
    routing_weights: List[float]
    routing_entropy: float
    top_expert: int
    confidence: float


class MoEAdapter(ModelAdapter):
    """Adapter for Mixture of Experts model."""

    def __init__(self, model_name: str = "MoE", model_version: str = "1.0"):
        """Initialize MoE adapter."""
        super().__init__(model_name, model_version)
        self.num_experts = 8  # Default number of experts
        self.experts = []

    def _get_supported_output_types(self) -> List[str]:
        """Return supported output types."""
        return [
            "expert_routing",
            "expert_outputs",
            "routing_weights",
            "expert_specializations",
            "feature_expert_correlation"
        ]

    def extract_diagnosis(self, model_output: Any) -> Dict[str, float]:
        """Extract fault diagnosis from MoE model output."""
        # MoE model typically outputs:
        # - final classification logits
        # - expert routing information
        # - individual expert outputs

        if isinstance(model_output, dict):
            if "diagnosis" in model_output:
                return model_output["diagnosis"]
            elif "final_logits" in model_output:
                logits = model_output["final_logits"]
            elif "output" in model_output:
                logits = model_output["output"]
            elif "predictions" in model_output:
                return model_output["predictions"]
            else:
                # Try to find classification output
                logits = self._find_classification_output(model_output)
        else:
            logits = model_output

        # Convert to probabilities if needed
        if logits is None:
            return {}

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

        # Map indices to fault types
        fault_types = ["normal", "inner_race", "outer_race", "ball", "cage"]

        diagnosis = {}
        for i, fault_type in enumerate(fault_types):
            if i < len(probs):
                diagnosis[fault_type] = float(probs[i])

        return diagnosis

    def extract_features(self, model_output: Any) -> List[FeatureImportance]:
        """Extract feature importance from expert routing and contributions."""
        features = []

        # Get expert routing information
        routing_info = self._extract_routing_info(model_output)
        expert_contributions = self._extract_expert_contributions(model_output)

        # Create features based on expert specialization
        for expert in self.experts:
            if expert.contribution_score > 0.1:  # Only include significant contributors
                features.append(FeatureImportance(
                    feature_name=f"expert_{expert.expert_id}_{expert.specialization}",
                    importance_score=expert.contribution_score,
                    description=f"Expert specializing in {expert.specialization}",
                    evidence=f"Activation rate: {expert.activation_rate:.2%}, Confidence: {expert.confidence:.2f}"
                ))

        # Add routing entropy as a feature
        if routing_info:
            avg_entropy = np.mean([r.routing_entropy for r in routing_info])
            features.append(FeatureImportance(
                feature_name="routing_entropy",
                importance_score=avg_entropy,
                description="Uncertainty in expert selection - higher entropy means more distributed routing",
                evidence=f"Average routing entropy across samples: {avg_entropy:.3f}"
            ))

        # Sort by importance
        features.sort(key=lambda x: x.importance_score, reverse=True)

        return features

    def extract_signal_pathway(self, model_output: Any) -> List[SignalPath]:
        """Extract signal processing pathway from MoE model."""
        pathways = []

        # MoE processing stages
        moe_stages = [
            ("input", (4096,), (4096,), "signal_input", {}),
            ("shared_encoder", (4096,), (512,), "shared_encoder", {
                "layers": 4,
                "activation": "relu"
            }),
            ("gating_network", (512,), (self.num_experts,), "gating_network", {
                "num_experts": self.num_experts,
                "top_k": 2  # Number of experts to route to
            }),
            ("expert_networks", (512,), (512,), "mixture_of_experts", {
                "experts": self.num_experts,
                "expert_layers": 2
            }),
            ("expert_combination", (512,), (512,), "weighted_combination", {}),
            ("classifier", (512,), (5,), "final_classifier", {})
        ]

        for i, (stage_name, input_shape, output_shape, operation, params) in enumerate(moe_stages):
            pathways.append(SignalPath(
                stage_name=f"stage_{i}_{stage_name}",
                input_shape=input_shape,
                output_shape=output_shape,
                operation=operation,
                parameters=params
            ))

        return pathways

    def extract_model_metadata(self, model_output: Any) -> Dict[str, Any]:
        """Extract metadata from MoE model."""
        metadata = {
            "model_type": "Mixture of Experts",
            "version": self.model_version,
            "num_experts": self.num_experts,
            "supports_explanations": True,
            "explanation_type": "expert_routing",
            "load_balancing": "gated",
            "expert_specialization": "learned"
        }

        # Add routing statistics if available
        if isinstance(model_output, dict):
            if "routing_stats" in model_output:
                metadata.update(model_output["routing_stats"])
            if "expert_utilization" in model_output:
                metadata["expert_utilization"] = model_output["expert_utilization"]

        return metadata

    def _extract_routing_info(self, model_output: Any) -> List[RoutingDecision]:
        """Extract expert routing decisions."""
        routing_decisions = []

        if isinstance(model_output, dict):
            if "routing_weights" in model_output:
                routing_weights = model_output["routing_weights"]
                if isinstance(routing_weights, torch.Tensor):
                    routing_weights = routing_weights.detach().cpu().numpy()

                # Convert to routing decisions
                for i, weights in enumerate(routing_weights):
                    if len(weights.shape) > 1:
                        weights = weights[0]

                    # Find selected experts (top-k)
                    top_k = np.argsort(weights)[-2:]  # Top 2 experts
                    routing_entropy = -np.sum(weights * np.log(weights + 1e-8))

                    routing_decisions.append(RoutingDecision(
                        input_sample_id=f"sample_{i}",
                        selected_experts=top_k.tolist(),
                        routing_weights=weights[top_k].tolist(),
                        routing_entropy=routing_entropy,
                        top_expert=int(top_k[-1]),
                        confidence=float(np.max(weights))
                    ))

            elif "gating_output" in model_output:
                # Alternative format for gating network output
                gating_output = model_output["gating_output"]
                # Parse similarly...

        return routing_decisions

    def _extract_expert_contributions(self, model_output: Any) -> Dict[int, float]:
        """Extract individual expert contributions."""
        contributions = {}

        if isinstance(model_output, dict):
            if "expert_outputs" in model_output:
                expert_outputs = model_output["expert_outputs"]
                if isinstance(expert_outputs, dict):
                    contributions = {
                        int(k.replace("expert_", "")): self._calculate_expert_contribution(v)
                        for k, v in expert_outputs.items()
                        if "expert_" in k
                    }
                elif isinstance(expert_outputs, (list, tuple)):
                    for i, output in enumerate(expert_outputs):
                        contributions[i] = self._calculate_expert_contribution(output)

            elif "expert_weights" in model_output:
                expert_weights = model_output["expert_weights"]
                if isinstance(expert_weights, torch.Tensor):
                    expert_weights = expert_weights.detach().cpu().numpy()

                # Average weights across samples
                avg_weights = np.mean(expert_weights, axis=0)
                for i, weight in enumerate(avg_weights):
                    contributions[i] = float(weight)

        return contributions

    def _calculate_expert_contribution(self, expert_output: Any) -> float:
        """Calculate contribution score for an expert output."""
        if isinstance(expert_output, (torch.Tensor, np.ndarray)):
            if isinstance(expert_output, torch.Tensor):
                expert_output = expert_output.detach().cpu().numpy()

            # Use magnitude of output as contribution metric
            return float(np.mean(np.abs(expert_output)))
        elif isinstance(expert_output, dict):
            # Extract contribution from dictionary
            if "contribution" in expert_output:
                return float(expert_output["contribution"])
            elif "confidence" in expert_output:
                return float(expert_output["confidence"])
            elif "weight" in expert_output:
                return float(expert_output["weight"])

        return 0.0

    def _find_classification_output(self, obj: Any) -> Optional[np.ndarray]:
        """Find classification output in nested structure."""
        if isinstance(obj, dict):
            # Look for keys that might contain classification output
            class_keys = ["logits", "probs", "classification", "output", "prediction"]
            for key in obj.keys():
                if any(ck in key.lower() for ck in class_keys):
                    return obj[key]
            # Recursively search
            for value in obj.values():
                result = self._find_classification_output(value)
                if result is not None:
                    return result
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                result = self._find_classification_output(item)
                if result is not None:
                    return result
        return None

    def set_experts(self, experts: List[ExpertInfo]):
        """Set the expert information."""
        self.experts = experts
        self.num_experts = len(experts)

    def set_num_experts(self, num_experts: int):
        """Set the number of experts."""
        self.num_experts = num_experts

    def get_expert_specialization_summary(self, model_output: Any) -> Dict[int, str]:
        """Get a summary of expert specializations."""
        specializations = {}

        # Extract from model output if available
        if isinstance(model_output, dict) and "expert_specializations" in model_output:
            spec_data = model_output["expert_specializations"]
            for expert_id, spec in spec_data.items():
                specializations[int(expert_id)] = spec

        # Or use predefined experts
        elif self.experts:
            for expert in self.experts:
                specializations[expert.expert_id] = expert.specialization

        return specializations


# Register the adapter
ModelAdapterFactory.register_adapter("MoE", MoEAdapter)
ModelAdapterFactory.register_adapter("MoE_3experts", MoEAdapter)
ModelAdapterFactory.register_adapter("MoE_5experts", MoEAdapter)
ModelAdapterFactory.register_adapter("MoE_8experts", MoEAdapter)