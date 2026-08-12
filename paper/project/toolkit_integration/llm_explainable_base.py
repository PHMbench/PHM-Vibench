"""
LLM Explainable Base Classes

Provides base classes and mixins for models that support LLM-enhanced explanations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import numpy as np

from .explainable_base import ExplainableMixin


class LLMExplainableMixin(ExplainableMixin, ABC):
    """
    Mixin class that adds LLM-enhanced explainability capabilities to models.

    This mixin extends ExplainableMixin with methods specifically designed for
    integrating with Large Language Models to provide natural language explanations
    and interactive diagnostic conversations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_context = {}
        self.llm_config = {}

    def get_diagnosis_context(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """
        Get comprehensive diagnostic context for LLM explanation.

        Args:
            input_data: Input tensor [batch_size, sequence_length, channels]

        Returns:
            Dictionary containing diagnostic context information
        """
        context = {
            'model_name': type(self).__name__,
            'model_type': self._get_model_type(),
            'input_statistics': self._compute_input_statistics(input_data),
            'expected_frequencies': self._get_expected_frequencies(),
            'device_parameters': self._get_device_parameters(),
            'operating_conditions': self._get_operating_conditions(),
            'timestamp': self._get_current_timestamp()
        }

        return context

    def generate_technical_summary(self,
                                    input_data: torch.Tensor,
                                    prediction_result: torch.Tensor,
                                    explanation: Optional[Any] = None) -> Dict[str, Any]:
        """
        Generate technical summary of diagnosis for LLM processing.

        Args:
            input_data: Input tensor
            prediction_result: Model prediction
            explanation: Existing explanation object

        Returns:
            Dictionary containing technical summary
        """
        summary = {
            'signal_characteristics': self._analyze_signal_characteristics(input_data),
            'prediction_confidence': self._compute_prediction_confidence(prediction_result),
            'key_features': self._extract_key_features(input_data),
            'anomaly_indicators': self._detect_anomaly_indicators(input_data)
        }

        # Include information from existing explanation if available
        if explanation is not None:
            if hasattr(explanation, 'get_data'):
                summary['existing_explanation'] = {
                    'method': explanation.get_meta('method', 'unknown'),
                    'metrics': explanation.get_metrics(),
                    'key_findings': self._extract_key_findings(explanation)
                }

        return summary

    def get_llm_explainability_info(self) -> Dict[str, Any]:
        """
        Get information about LLM explainability capabilities.

        Returns:
            Dictionary containing LLM explainability information
        """
        base_info = super().get_model_explainability_info()

        llm_info = {
            'llm_enabled': True,
            'supported_llm_features': [
                'natural_language_explanation',
                'interactive_diagnosis',
                'technical_summary',
                'fault_context_enhancement',
                'maintenance_suggestions'
            ],
            'llm_models': self._get_supported_llm_models(),
            'domain_knowledge': self._get_domain_knowledge(),
            'conversation_capabilities': self._get_conversation_capabilities()
        }

        # Merge with base information
        base_info.update(llm_info)
        return base_info

    def explain_with_llm(self,
                        input_data: torch.Tensor,
                        user_query: Optional[str] = None,
                        context: Optional[Dict[str, Any]] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Generate LLM-enhanced explanation.

        Args:
            input_data: Input tensor
            user_query: Optional user query for targeted explanation
            context: Additional context information
            **kwargs: Additional parameters

        Returns:
            Dictionary containing LLM-enhanced explanation results
        """
        from explainability.llm.llm_explainer import LLMExplainer

        # Get diagnostic context
        if context is None:
            context = self.get_diagnosis_context(input_data)

        # Generate traditional explanation
        traditional_explanation = None
        try:
            traditional_explanation = self.get_signal_path(input_data)
        except Exception as e:
            print(f"Warning: Could not get signal path: {e}")

        # Generate technical summary
        prediction = self._get_model_predictions(input_data)
        technical_summary = self.generate_technical_summary(input_data, prediction, traditional_explanation)

        # Initialize LLM explainer
        if not hasattr(self, '_llm_explainer') or self._llm_explainer is None:
            self._llm_explainer = LLMExplainer(self, self.llm_config)

        # Generate LLM-enhanced explanation
        llm_result = self._llm_explainer.explain_with_llm(
            traditional_explanation,
            technical_summary,
            user_query,
            context,
            **kwargs
        )

        return {
            'traditional_explanation': traditional_explanation,
            'technical_summary': technical_summary,
            'llm_enhanced_explanation': llm_result,
            'context': context
        }

    def start_diagnostic_conversation(self, input_data: torch.Tensor) -> 'ConversationSession':
        """
        Start an interactive diagnostic conversation.

        Args:
            input_data: Input tensor for diagnosis

        Returns:
            ConversationSession object for managing the dialogue
        """
        from explainability.conversation.conversation_engine import ConversationEngine

        if not hasattr(self, '_conversation_engine') or self._conversation_engine is None:
            self._conversation_engine = ConversationEngine(self, self.llm_config)

        return self._conversation_engine.start_session(input_data)

    def _get_model_type(self) -> str:
        """Get model type for LLM context."""
        model_name = type(self).__name__
        if 'TSPN' in model_name:
            return 'transparent_signal_processing'
        elif 'NNSPN' in model_name:
            return 'neural_signal_processing'
        elif 'TFON' in model_name:
            return 'time_frequency_operator'
        elif 'TKAN' in model_name:
            return 'time_kolmogorov_arnold'
        else:
            return 'unknown'

    def _compute_input_statistics(self, input_data: torch.Tensor) -> Dict[str, float]:
        """Compute comprehensive input statistics."""
        if isinstance(input_data, torch.Tensor):
            signal_np = input_data.detach().cpu().numpy()
        else:
            signal_np = input_data

        return {
            'mean': float(np.mean(signal_np)),
            'std': float(np.std(signal_np)),
            'rms': float(np.sqrt(np.mean(signal_np ** 2))),
            'peak': float(np.max(np.abs(signal_np))),
            'crest_factor': float(np.max(np.abs(signal_np)) / (np.sqrt(np.mean(signal_np ** 2)) + 1e-8)),
            'skewness': float(self._compute_skewness(signal_np)),
            'kurtosis': float(self._compute_kurtosis(signal_np)),
            'energy': float(np.sum(signal_np ** 2))
        }

    def _get_expected_frequencies(self) -> Dict[str, List[float]]:
        """Get expected characteristic frequencies for the system."""
        # This should be overridden by specific models
        return {
            'shaft_frequencies': [],
            'gear_mesh_frequencies': [],
            'bearing_frequencies': [],
            'harmonics': []
        }

    def _get_device_parameters(self) -> Dict[str, Any]:
        """Get device/operating parameters."""
        # This should be overridden by specific models
        return {
            'device_type': 'unknown',
            'rated_speed': 0,
            'load_condition': 'unknown',
            'environmental_conditions': 'unknown'
        }

    def _get_operating_conditions(self) -> Dict[str, Any]:
        """Get current operating conditions."""
        return {
            'timestamp': self._get_current_timestamp(),
            'temperature': 'unknown',
            'vibration_level': 'unknown',
            'noise_level': 'unknown'
        }

    def _get_current_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _analyze_signal_characteristics(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """Analyze signal characteristics for LLM processing."""
        # Basic spectral analysis
        if isinstance(input_data, torch.Tensor):
            signal_np = input_data.detach().cpu().numpy().flatten()
        else:
            signal_np = input_data.flatten()

        # FFT analysis
        fft_vals = np.fft.fft(signal_np)
        fft_freq = np.fft.fftfreq(len(signal_np), 1/1024.0)  # Assuming 1kHz sampling

        # Only positive frequencies
        pos_mask = fft_freq > 0
        pos_freq = fft_freq[pos_mask]
        pos_fft = np.abs(fft_vals[pos_mask])

        # Find dominant frequency
        if len(pos_fft) > 0:
            dominant_freq_idx = np.argmax(pos_fft)
            dominant_freq = pos_freq[dominant_freq_idx]
            dominant_power = pos_fft[dominant_freq_idx]
        else:
            dominant_freq = 0.0
            dominant_power = 0.0

        # Spectral centroid
        spectral_centroid = np.sum(pos_freq * pos_fft) / (np.sum(pos_fft) + 1e-8)

        return {
            'dominant_frequency': float(dominant_freq),
            'dominant_power': float(dominant_power),
            'spectral_centroid': float(spectral_centroid),
            'frequency_content': 'mixed',  # Will be refined by specific models
            'signal_type': 'unknown'  # Will be refined by specific models
        }

    def _compute_prediction_confidence(self, prediction: torch.Tensor) -> Dict[str, float]:
        """Compute confidence metrics for model prediction."""
        if isinstance(prediction, torch.Tensor):
            pred_probs = torch.softmax(prediction, dim=-1)
            confidence, predicted_class = torch.max(pred_probs, dim=-1)

            return {
                'confidence': float(confidence),
                'predicted_class': int(predicted_class),
                'entropy': float(-torch.sum(pred_probs * torch.log(pred_probs + 1e-8))),
                'class_probabilities': pred_probs.tolist()
            }
        else:
            return {'confidence': 0.0, 'predicted_class': -1, 'entropy': 0.0}

    def _extract_key_features(self, input_data: torch.Tensor) -> List[Dict[str, Any]]:
        """Extract key features for LLM explanation."""
        # This should be implemented by specific models
        return []

    def _detect_anomaly_indicators(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """Detect potential anomalies in the input data."""
        stats = self._compute_input_statistics(input_data)

        indicators = {}

        # High vibration level
        if stats['rms'] > 10.0:  # Adjust threshold based on application
            indicators['high_vibration'] = {
                'detected': True,
                'value': stats['rms'],
                'threshold': 10.0,
                'severity': 'high' if stats['rms'] > 20.0 else 'medium'
            }

        # High crest factor (indicates impacts)
        if stats['crest_factor'] > 5.0:
            indicators['impulsive_vibration'] = {
                'detected': True,
                'value': stats['crest_factor'],
                'threshold': 5.0,
                'severity': 'high' if stats['crest_factor'] > 10.0 else 'medium'
            }

        # High skewness (indicates non-symmetrical vibration)
        if abs(stats['skewness']) > 2.0:
            indicators['asymmetric_vibration'] = {
                'detected': True,
                'value': stats['skewness'],
                'threshold': 2.0,
                'direction': 'positive' if stats['skewness'] > 0 else 'negative'
            }

        return indicators

    def _extract_key_findings(self, explanation) -> Dict[str, Any]:
        """Extract key findings from existing explanation."""
        findings = {}

        if hasattr(explanation, 'get_data'):
            data = explanation.get_data()

            # Extract from signal path
            if 'signal_path' in data:
                path = data['signal_path']
                findings['signal_processing_stages'] = len(path)

                # Look for high-energy stages
                high_energy_stages = []
                for i, stage in enumerate(path):
                    if 'output_stats' in stage and 'input_stats' in stage:
                        output_energy = stage['output_stats'].get('energy', 0)
                        input_energy = stage['input_stats'].get('energy', 0)
                        if input_energy > 0:
                            energy_ratio = output_energy / input_energy
                            if energy_ratio > 1.5 or energy_ratio < 0.5:
                                high_energy_stages.append({
                                    'stage': i,
                                    'layer_name': stage.get('layer_name', f'stage_{i}'),
                                    'energy_ratio': energy_ratio
                                })

                findings['high_energy_transformations'] = high_energy_stages

            # Extract from physical analysis
            if 'physical_analysis' in data:
                physical = data['physical_analysis']

                # Energy flow analysis
                if 'energy_flow' in physical:
                    energy_flow = physical['energy_flow']
                    if energy_flow:
                        max_energy_change = max(
                            flow.get('energy_change_ratio', 0) for flow in energy_flow
                        )
                        findings['max_energy_transformation'] = {
                            'change_ratio': max_energy_change,
                            'significance': 'high' if abs(max_energy_change) > 0.5 else 'medium'
                        }

        return findings

    def _get_supported_llm_models(self) -> List[str]:
        """Get list of supported LLM models."""
        return [
            'gpt-4',
            'gpt-3.5-turbo',
            'claude-3-sonnet',
            'claude-3-haiku',
            'local-model'
        ]

    def _get_domain_knowledge(self) -> Dict[str, Any]:
        """Get domain knowledge available for LLM enhancement."""
        return {
            'fault_types': [
                'inner_race',
                'outer_race',
                'ball_defect',
                'cage_damage',
                'misalignment',
                'imbalance',
                'looseness',
                'bearing_wear'
            ],
            'characteristic_frequencies': [
                'shaft_frequency',
                'ball_pass_frequency_outer',
                'ball_pass_frequency_inner',
                'ball_spin_frequency',
                'fundamental_train_frequency'
            ],
            'diagnostic_principles': [
                'vibration_spectrum_analysis',
                'time_domain_analysis',
                'phase_analysis',
                'envelope_analysis'
            ]
        }

    def _get_conversation_capabilities(self) -> Dict[str, Any]:
        """Get conversation capabilities."""
        return {
            'supported_queries': [
                'fault_identification',
                'cause_analysis',
                'maintenance_recommendations',
                'severity_assessment',
                'historical_comparison',
                'what_if_scenarios'
            ],
            'conversation_features': [
                'multi_turn_dialogue',
                'context_awareness',
                'follow_up_questions',
                'technical_explanation',
                'practical_advice'
            ]
        }

    def _compute_skewness(self, signal: np.ndarray) -> float:
        """Compute skewness of signal."""
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return 0.0
        return np.mean(((signal - mean) / std) ** 3)

    def _compute_kurtosis(self, signal: np.ndarray) -> float:
        """Compute kurtosis of signal."""
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return 0.0
        return np.mean(((signal - mean) / std) ** 4) - 3