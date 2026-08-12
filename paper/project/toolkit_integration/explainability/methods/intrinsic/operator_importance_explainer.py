"""
Operator Importance Explainer

Provides explanations based on the importance of different operators
and processing modules in the model. This explainer focuses on
identifying which signal processing operators contribute most to
the model's decisions.
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
import numpy as np
from ...core.base_explainer import BaseExplainer
from ...core.explanation import Explanation


class OperatorImportanceExplainer(BaseExplainer):
    """
    Operator Importance Explainer for analyzing the contribution
    of different signal processing operators and modules.

    This explainer analyzes which operators, modules, or processing
    steps are most important for the model's decision-making process.
    It's particularly useful for understanding operator networks and
    identifying critical signal processing components.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(model, config)

        # Configuration options
        self.importance_metric = self.config.get('importance_metric', 'gradient_based')  # gradient_based, ablation, activation_based
        self.ablation_method = self.config.get('ablation_method', 'zero')  # zero, noise, mean
        self.num_samples = self.config.get('num_samples', 10)
        self.include_attention_weights = self.config.get('include_attention_weights', True)
        self.include_feature_importance = self.config.get('include_feature_importance', True)

        # Operator type mappings for physical interpretation
        self.operator_interpretations = {
            'FFT': 'Frequency Domain Analysis',
            'HT': 'Hilbert Transform - Envelope Detection',
            'WF': 'Wavelet Filter - Time-Frequency Analysis',
            'I': 'Identity - Pass Through',
            'LNO': 'Laplacian Neural Operator - Continuous Processing',
            'Linear': 'Linear Transformation',
            'Conv': 'Convolutional Filtering',
            'ReLU': 'Non-linear Activation'
        }

    def explain(self,
                input_data: torch.Tensor,
                target_class: Optional[int] = None,
                **kwargs) -> Explanation:
        """
        Generate operator importance explanation.

        Args:
            input_data: Input tensor [batch_size, sequence_length, channels]
            target_class: Target class for explanation
            **kwargs: Additional arguments

        Returns:
            Explanation object containing operator importance information
        """
        self._validate_input(input_data)

        # Get target class
        target = self._get_target_class(input_data, target_class)

        # Compute importance scores using the specified method
        if self.importance_metric == 'gradient_based':
            importance_scores = self._compute_gradient_based_importance(input_data, target)
        elif self.importance_metric == 'ablation':
            importance_scores = self._compute_ablation_importance(input_data, target)
        elif self.importance_metric == 'activation_based':
            importance_scores = self._compute_activation_based_importance(input_data)
        else:
            raise ValueError(f"Unknown importance metric: {self.importance_metric}")

        # Get additional importance information
        attention_importance = {}
        if self.include_attention_weights and hasattr(self.model, 'get_attention_maps'):
            try:
                attention_maps = self.model.get_attention_maps(input_data)
                attention_importance = self._analyze_attention_importance(attention_maps)
            except Exception as e:
                print(f"Warning: Could not compute attention importance: {e}")

        feature_importance = {}
        if self.include_feature_importance and hasattr(self.model, 'get_signal_path'):
            try:
                signal_path = self.model.get_signal_path(input_data)
                feature_importance = self._extract_feature_importance(signal_path)
            except Exception as e:
                print(f"Warning: Could not extract feature importance: {e}")

        # Create explanation data
        explanation_data = {
            'importance_scores': importance_scores,
            'attention_importance': attention_importance,
            'feature_importance': feature_importance,
            'operator_interpretations': self._add_physical_interpretations(importance_scores),
            'original_signal': input_data,
            'target_class': target,
            'importance_method': self.importance_metric
        }

        # Create metadata
        metadata = {
            'method': 'operator_importance',
            'model_name': type(self.model).__name__,
            'input_shape': list(input_data.shape),
            'target_class': target,
            'importance_metric': self.importance_metric,
            'num_operators': len(importance_scores) if importance_scores else 0
        }

        return Explanation(explanation_data, metadata)

    def _compute_gradient_based_importance(self,
                                         input_data: torch.Tensor,
                                         target_class: int) -> Dict[str, Dict[str, float]]:
        """
        Compute operator importance using gradient-based methods.

        This method computes the gradient of the output with respect to
        intermediate layer outputs to determine which operators are most
        important for the prediction.
        """
        importance_scores = {}

        # Get model's intermediate outputs
        try:
            intermediate_outputs = self._get_intermediate_outputs_with_gradients(input_data, target_class)

            for layer_name, output_info in intermediate_outputs.items():
                if 'gradient' in output_info:
                    gradient = output_info['gradient']
                    activation = output_info['activation']

                    # Compute importance as gradient magnitude weighted by activation
                    importance = torch.mean(torch.abs(gradient * activation), dim=tuple(range(1, len(gradient.shape))))

                    if isinstance(importance, torch.Tensor):
                        importance = importance.item()

                    importance_scores[layer_name] = {
                        'gradient_importance': importance,
                        'gradient_mean': float(torch.mean(torch.abs(gradient))),
                        'activation_mean': float(torch.mean(torch.abs(activation)))
                    }

        except Exception as e:
            print(f"Error computing gradient-based importance: {e}")
            # Fallback: try to get signal path and compute basic importance
            if hasattr(self.model, 'get_signal_path'):
                try:
                    signal_path = self.model.get_signal_path(input_data)
                    importance_scores = self._compute_path_based_importance(signal_path)
                except Exception:
                    pass

        return importance_scores

    def _compute_ablation_importance(self,
                                   input_data: torch.Tensor,
                                   target_class: int) -> Dict[str, Dict[str, float]]:
        """
        Compute operator importance using ablation studies.

        This method systematically ablates (disables) different operators
        and measures the impact on model performance.
        """
        importance_scores = {}
        original_output = self._get_model_predictions(input_data)

        # Get baseline score for target class
        if len(original_output.shape) > 1:
            baseline_score = original_output[0, target_class].item()
        else:
            baseline_score = original_output[target_class].item()

        # Try to ablate different layers/modules
        if hasattr(self.model, 'signal_processing_layers'):
            for i, layer in enumerate(self.model.signal_processing_layers):
                layer_name = f'signal_processing_layer_{i}'

                # Create a copy of the model for ablation (or temporarily disable layer)
                try:
                    # Temporarily replace layer with identity
                    original_layer = layer
                    identity_layer = torch.nn.Identity()

                    # Replace layer temporarily
                    self.model.signal_processing_layers[i] = identity_layer

                    # Get prediction with ablated layer
                    with torch.no_grad():
                        ablated_output = self.model(input_data)

                    # Restore original layer
                    self.model.signal_processing_layers[i] = original_layer

                    # Compute importance based on score change
                    if len(ablated_output.shape) > 1:
                        ablated_score = ablated_output[0, target_class].item()
                    else:
                        ablated_score = ablated_output[target_class].item()

                    importance = baseline_score - ablated_score
                    importance_scores[layer_name] = {
                        'ablation_importance': importance,
                        'baseline_score': baseline_score,
                        'ablated_score': ablated_score,
                        'relative_change': importance / (abs(baseline_score) + 1e-8)
                    }

                except Exception as e:
                    print(f"Could not ablate layer {layer_name}: {e}")

        return importance_scores

    def _compute_activation_based_importance(self,
                                          input_data: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """
        Compute operator importance based on activation magnitudes.

        This method uses the magnitude of activations as a proxy for
        operator importance, assuming that larger activations indicate
        more important processing.
        """
        importance_scores = {}

        try:
            # Get intermediate activations
            if hasattr(self.model, 'get_signal_path'):
                signal_path = self.model.get_signal_path(input_data)

                for step in signal_path:
                    layer_name = step.get('layer_name', '')
                    if 'output_stats' in step:
                        output_stats = step['output_stats']

                        # Use different activation statistics as importance measures
                        rms_importance = output_stats.get('rms', 0)
                        max_importance = output_stats.get('max', 0)
                        energy_importance = output_stats.get('energy', 0)

                        importance_scores[layer_name] = {
                            'activation_rms': rms_importance,
                            'activation_max': max_importance,
                            'activation_energy': energy_importance,
                            'combined_importance': (rms_importance + max_importance) / 2
                        }

        except Exception as e:
            print(f"Error computing activation-based importance: {e}")

        return importance_scores

    def _get_intermediate_outputs_with_gradients(self,
                                              input_data: torch.Tensor,
                                              target_class: int) -> Dict[str, Dict[str, Any]]:
        """Get intermediate outputs and their gradients."""
        intermediate_outputs = {}
        hooks = []

        def create_hook(name):
            def hook(module, input, output):
                if output.requires_grad:
                    intermediate_outputs[name] = {
                        'activation': output.clone().detach(),
                        'shape': output.shape
                    }
            return hook

        # Register hooks for all layers
        for name, module in self.model.named_modules():
            if name and any(p.requires_grad for p in module.parameters()):
                hook = module.register_forward_hook(create_hook(name))
                hooks.append(hook)

        try:
            # Forward pass
            input_data.requires_grad_(True)
            output = self.model(input_data)

            # Backward pass for target class
            if len(output.shape) > 1:
                target_score = output[0, target_class]
            else:
                target_score = output[target_class]

            target_score.backward(retain_graph=True)

            # Get gradients for intermediate outputs
            for name, output_info in intermediate_outputs.items():
                if hasattr(self.model, name):
                    layer = getattr(self.model)
                    # This is a simplified approach - in practice, you might need more sophisticated gradient tracking
                    pass

        finally:
            # Clean up
            input_data.requires_grad_(False)
            for hook in hooks:
                hook.remove()

        return intermediate_outputs

    def _compute_path_based_importance(self,
                                     signal_path: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Compute importance from signal path information."""
        importance_scores = {}

        for step in signal_path:
            layer_name = step.get('layer_name', '')
            layer_type = step.get('layer_type', '')

            # Use energy change as importance measure
            if 'input_stats' in step and 'output_stats' in step:
                input_energy = step['input_stats'].get('energy', 0)
                output_energy = step['output_stats'].get('energy', 0)
                energy_change = abs(output_energy - input_energy) / (input_energy + 1e-8)

                importance_scores[layer_name] = {
                    'energy_change_importance': energy_change,
                    'input_energy': input_energy,
                    'output_energy': output_energy,
                    'layer_type': layer_type
                }

        return importance_scores

    def _analyze_attention_importance(self,
                                    attention_maps: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """Analyze attention weights to determine importance."""
        attention_importance = {}

        for name, attention_weights in attention_maps.items():
            if isinstance(attention_weights, torch.Tensor):
                # Compute attention statistics
                attention_stats = {
                    'mean_attention': float(torch.mean(attention_weights)),
                    'max_attention': float(torch.max(attention_weights)),
                    'attention_variance': float(torch.var(attention_weights)),
                    'attention_entropy': self._compute_attention_entropy(attention_weights)
                }

                # Find most attended units
                if len(attention_weights.shape) >= 2:
                    max_indices = torch.argmax(torch.mean(attention_weights, dim=0))
                    attention_stats['most_attended_units'] = max_indices.tolist() if hasattr(max_indices, 'tolist') else max_indices

                attention_importance[name] = attention_stats

        return attention_importance

    def _extract_feature_importance(self,
                                  signal_path: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Extract feature importance from signal path."""
        feature_importance = {}

        # Look for feature extractor layer
        for step in signal_path:
            if step.get('layer_name') == 'feature_extractor':
                if 'feature_importance' in step:
                    feature_importance = step['feature_importance']
                elif 'feature_outputs' in step:
                    # Compute importance from feature outputs
                    feature_outputs = step['feature_outputs']
                    for feature_name, feature_info in feature_outputs.items():
                        if 'output' in feature_info:
                            feature_tensor = feature_info['output']
                            importance = torch.mean(torch.abs(feature_tensor)).item()
                            feature_importance[feature_name] = {
                                'importance': importance,
                                'feature_type': feature_info.get('module_type', 'Unknown')
                            }

        return feature_importance

    def _add_physical_interpretations(self,
                                     importance_scores: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """Add physical interpretations to operator importance."""
        interpretations = {}

        for operator_name, scores in importance_scores.items():
            # Try to extract operator type from the name
            operator_type = None
            for op_key, op_interpretation in self.operator_interpretations.items():
                if op_key in operator_name.upper():
                    operator_type = op_key
                    break

            if operator_type:
                interpretations[operator_name] = self.operator_interpretations.get(operator_type, 'Unknown Operator')
            else:
                # Generic interpretation based on layer type
                if 'signal_processing' in operator_name.lower():
                    interpretations[operator_name] = 'Signal Processing Layer'
                elif 'feature' in operator_name.lower():
                    interpretations[operator_name] = 'Feature Extraction Layer'
                elif 'classifier' in operator_name.lower():
                    interpretations[operator_name] = 'Classification Layer'
                else:
                    interpretations[operator_name] = 'Processing Layer'

        return interpretations

    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention weights."""
        # Normalize to probability distribution
        attention_probs = F.softmax(attention_weights.flatten(), dim=0)

        # Compute entropy
        entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-8))

        return entropy.item()

    def get_operator_ranking(self, explanation: Explanation) -> List[Tuple[str, float]]:
        """
        Get a ranking of operators by importance.

        Args:
            explanation: Explanation object containing importance scores

        Returns:
            List of (operator_name, importance_score) tuples sorted by importance
        """
        importance_scores = explanation.get_data('importance_scores')
        if not importance_scores:
            return []

        # Use the first available importance metric for ranking
        operator_scores = []
        for operator_name, scores in importance_scores.items():
            if isinstance(scores, dict):
                # Use the first score value (prefer specific metrics)
                score = next(iter(scores.values()))
            else:
                score = scores

            operator_scores.append((operator_name, float(score)))

        # Sort by importance (descending)
        operator_scores.sort(key=lambda x: x[1], reverse=True)

        return operator_scores

    def __repr__(self) -> str:
        """String representation of the operator importance explainer."""
        return f"OperatorImportanceExplainer(metric='{self.importance_metric}', model={type(self.model).__name__})"