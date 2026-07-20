"""
Operator Weight Explainer

Intrinsic explanation method that analyzes operator weights and parameters
in transparent signal processing models to provide insights into how different
operators contribute to the final decision.
"""

from typing import Dict, Any, Optional, Sequence, List, Union, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ...core import SignalData, Explanation, ExplainabilityMethod, BaseExplainerAdapter


class OperatorWeightExplainer(BaseExplainerAdapter, ExplainabilityMethod):
    """
    Operator Weight Explainer for intrinsic explanations.

    This method analyzes the weights and parameters of different operators
    in transparent signal processing models, providing insights into which
    operators and components are most influential in the decision-making process.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Operator Weight Explainer.

        Args:
            config: Configuration dictionary with the following options:
                - include_weight_magnitude: bool (default True)
                - include_weight_gradients: bool (default False)
                - include_activation_patterns: bool (default True)
                - weight_analysis_method: str ('magnitude', 'variance', 'spectral', default 'magnitude')
                - layer_importance_threshold: float (default 0.05)
                - top_k_operators: int (default 10)
                - normalize_weights: bool (default True)
        """
        super().__init__(config)
        self._method_name = "OperatorWeight"
        self._method_type = "intrinsic"

        # Configuration options
        self.include_weight_magnitude = self.config.get('include_weight_magnitude', True)
        self.include_weight_gradients = self.config.get('include_weight_gradients', False)
        self.include_activation_patterns = self.config.get('include_activation_patterns', True)
        self.weight_analysis_method = self.config.get('weight_analysis_method', 'magnitude')
        self.layer_importance_threshold = self.config.get('layer_importance_threshold', 0.05)
        self.top_k_operators = self.config.get('top_k_operators', 10)
        self.normalize_weights = self.config.get('normalize_weights', True)

    def explain(self,
                signal: SignalData,
                prediction: Any,
                model: Optional[torch.nn.Module] = None,
                **kwargs) -> Explanation:
        """
        Generate operator weight-based explanation.

        Args:
            signal: SignalData object containing the signal to explain
            prediction: Model prediction or target class
            model: Optional model for weight analysis
            **kwargs: Additional arguments including:
                - target_layers: List of layer names to analyze
                - compute_gradients: bool (default False)
                - activation_data: Optional precomputed activation data

        Returns:
            Explanation object containing operator weight analysis results
        """
        self._validate_signal(signal)

        # Get model if provided
        model = model or kwargs.get('model')
        if model is None:
            raise ValueError("Model must be provided for operator weight analysis")

        # Extract operator weights
        operator_weights = self._extract_operator_weights(model, **kwargs)

        # Analyze weight importance
        weight_importance = self._analyze_weight_importance(operator_weights)

        # Get activation patterns if requested
        activation_patterns = None
        if self.include_activation_patterns:
            activation_patterns = self._compute_activation_patterns(signal, model, **kwargs)

        # Generate explanations
        explanation_data = {
            'operator_weights': operator_weights,
            'weight_importance': weight_importance,
            'activation_patterns': activation_patterns,
            'original_signal': signal.raw_signal,
            'method_specific': {
                'total_parameters': self._count_parameters(model),
                'important_layers': self._identify_important_layers(weight_importance),
                'weight_statistics': self._compute_weight_statistics(operator_weights),
                'layer_rankings': self._rank_layers_by_importance(weight_importance),
                'operator_type_analysis': self._analyze_operator_types(operator_weights)
            }
        }

        explanation_meta = {
            'method': self.get_method_name(),
            'method_type': self.get_method_type(),
            'signal_info': {
                'shape': signal.get_shape(),
                'duration': signal.get_duration(),
                'sampling_rate': signal.sampling_rate,
                'channels': signal.get_num_channels()
            },
            'prediction': prediction,
            'config': self.get_config()
        }

        return Explanation(explanation_data, explanation_meta)

    def visualize(self,
                  explanation: Explanation,
                  mode: str = 'auto',
                  **kwargs) -> plt.Figure:
        """
        Create visualization for operator weight explanation.

        Args:
            explanation: Explanation object to visualize
            mode: Visualization mode ('auto', 'weights', 'importance', 'activations', 'comparison')
            **kwargs: Additional visualization parameters

        Returns:
            Matplotlib figure object
        """
        self._validate_explanation(explanation)

        if mode == 'auto':
            return self._visualize_weight_overview(explanation)
        elif mode == 'weights':
            return self._visualize_operator_weights(explanation)
        elif mode == 'importance':
            return self._visualize_weight_importance(explanation)
        elif mode == 'activations':
            return self._visualize_activation_patterns(explanation)
        elif mode == 'comparison':
            return self._visualize_layer_comparison(explanation)
        else:
            raise ValueError(f"Unknown visualization mode: {mode}")

    def evaluate(self,
                 explanations: Sequence[Explanation],
                 ground_truth: Optional[Sequence[Any]] = None,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate operator weight explanations.

        Args:
            explanations: Sequence of explanation objects to evaluate
            ground_truth: Optional ground truth for evaluation
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}

        if not explanations:
            return metrics

        # Weight analysis metrics
        weight_entropies = []
        parameter_counts = []
        important_layer_ratios = []

        for exp in explanations:
            weight_importance = exp.get_data('weight_importance', {})
            method_specific = exp.get_data('method_specific', {})

            # Compute entropy of weight importance distribution
            if weight_importance:
                importance_values = [info.get('importance_score', 0) for info in weight_importance.values()]
                if importance_values and sum(importance_values) > 0:
                    probs = np.array(importance_values) / np.sum(importance_values)
                    entropy = -np.sum(probs * np.log2(probs + 1e-10))
                    weight_entropies.append(entropy)

            parameter_counts.append(method_specific.get('total_parameters', 0))

            # Ratio of important layers
            total_layers = len(weight_importance)
            important_layers = len(method_specific.get('important_layers', []))
            if total_layers > 0:
                important_layer_ratios.append(important_layers / total_layers)

        # Compute metrics
        metrics['avg_weight_entropy'] = float(np.mean(weight_entropies)) if weight_entropies else 0.0
        metrics['avg_parameter_count'] = float(np.mean(parameter_counts)) if parameter_counts else 0.0
        metrics['avg_important_layer_ratio'] = float(np.mean(important_layer_ratios)) if important_layer_ratios else 0.0
        metrics['explanation_sparsity'] = float(1.0 - metrics['avg_weight_entropy'] / 10.0) if metrics['avg_weight_entropy'] > 0 else 0.0

        return metrics

    def _extract_operator_weights(self,
                                model: torch.nn.Module,
                                **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Extract weights from model operators.

        Args:
            model: PyTorch model
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping layer names to weight information
        """
        operator_weights = {}
        target_layers = kwargs.get('target_layers', [])

        for name, param in model.named_parameters():
            # Filter by target layers if specified
            if target_layers and not any(target in name for target in target_layers):
                continue

            weight_data = param.data

            if weight_data is not None:
                # Extract weight statistics
                weight_info = {
                    'shape': list(weight_data.shape),
                    'num_parameters': weight_data.numel(),
                    'weight_values': weight_data.detach().cpu().numpy(),
                    'mean': float(weight_data.mean()),
                    'std': float(weight_data.std()),
                    'min': float(weight_data.min()),
                    'max': float(weight_data.max()),
                    'norm': float(weight_data.norm()),
                    'parameter_type': 'weight' if 'weight' in name else 'bias',
                    'operator_type': self._infer_operator_type_from_name(name)
                }

                # Additional weight analysis based on method
                if self.weight_analysis_method == 'magnitude':
                    weight_info['magnitude_importance'] = float(torch.abs(weight_data).mean())
                elif self.weight_analysis_method == 'variance':
                    weight_info['variance_importance'] = float(weight_data.var())
                elif self.weight_analysis_method == 'spectral':
                    if weight_data.ndim >= 2:
                        # Compute spectral norm for 2D+ tensors
                        try:
                            weight_info['spectral_importance'] = float(torch.norm(weight_data, p='fro'))
                        except Exception:
                            weight_info['spectral_importance'] = float(weight_data.norm())
                    else:
                        weight_info['spectral_importance'] = float(weight_data.norm())

                # Gradient information if requested
                if self.include_weight_gradients and param.grad is not None:
                    grad_data = param.grad.data
                    weight_info['gradient_stats'] = {
                        'mean': float(grad_data.mean()),
                        'std': float(grad_data.std()),
                        'norm': float(grad_data.norm())
                    }

                operator_weights[name] = weight_info

        return operator_weights

    def _analyze_weight_importance(self, operator_weights: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze importance of operator weights.

        Args:
            operator_weights: Dictionary of weight information

        Returns:
            Dictionary of importance analysis
        """
        weight_importance = {}

        for layer_name, weight_info in operator_weights.items():
            importance_score = 0.0

            # Calculate importance based on analysis method
            if self.weight_analysis_method == 'magnitude':
                importance_score = weight_info.get('magnitude_importance', 0)
            elif self.weight_analysis_method == 'variance':
                importance_score = weight_info.get('variance_importance', 0)
            elif self.weight_analysis_method == 'spectral':
                importance_score = weight_info.get('spectral_importance', 0)
            else:
                # Default: combine multiple metrics
                magnitude = weight_info.get('magnitude_importance', weight_info.get('mean', 0))
                variance = weight_info.get('variance_importance', weight_info.get('std', 0))
                importance_score = magnitude + variance

            # Adjust for parameter count (favor layers with fewer parameters but high impact)
            num_params = weight_info.get('num_parameters', 1)
            efficiency_factor = np.log(num_params + 1)  # Log scaling
            adjusted_score = importance_score / efficiency_factor

            # Boost scores for certain operator types
            operator_type = weight_info.get('operator_type', '')
            if operator_type in ['Convolution', 'Attention', 'Linear']:
                adjusted_score *= 1.1
            elif operator_type in ['FFT', 'Wavelet', 'Hilbert']:
                adjusted_score *= 1.2  # Boost signal processing operators

            weight_importance[layer_name] = {
                'importance_score': importance_score,
                'adjusted_importance': adjusted_score,
                'num_parameters': num_params,
                'efficiency_factor': efficiency_factor,
                'operator_type': operator_type,
                'parameter_type': weight_info.get('parameter_type', 'unknown')
            }

        # Normalize importance scores
        if weight_importance and self.normalize_weights:
            max_importance = max([info['adjusted_importance'] for info in weight_importance.values()])
            if max_importance > 0:
                for layer_name in weight_importance:
                    weight_importance[layer_name]['normalized_importance'] = (
                        weight_importance[layer_name]['adjusted_importance'] / max_importance
                    )

        return weight_importance

    def _compute_activation_patterns(self,
                                   signal: SignalData,
                                   model: torch.nn.Module,
                                   **kwargs) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Compute activation patterns for the given signal.

        Args:
            signal: Input signal
            model: Model to analyze
            **kwargs: Additional parameters

        Returns:
            Dictionary of activation patterns per layer
        """
        if not self.include_activation_patterns:
            return None

        activation_patterns = {}
        target_layers = kwargs.get('target_layers', [])

        # Convert signal to tensor
        if isinstance(signal.raw_signal, np.ndarray):
            input_tensor = torch.FloatTensor(signal.raw_signal)
        else:
            input_tensor = signal.raw_signal

        # Add batch dimension if needed
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        elif len(input_tensor.shape) == 2:
            input_tensor = input_tensor.unsqueeze(0)

        model.eval()
        with torch.no_grad():
            # Try to get intermediate activations
            try:
                if hasattr(model, 'get_intermediate_activations'):
                    activations = model.get_intermediate_activations(input_tensor, target_layers)
                    for layer_name, activation in activations.items():
                        activation_data = activation.detach().cpu().numpy()
                        activation_patterns[layer_name] = {
                            'activation_stats': {
                                'mean': float(np.mean(activation_data)),
                                'std': float(np.std(activation_data)),
                                'min': float(np.min(activation_data)),
                                'max': float(np.max(activation_data)),
                                'sparsity': float(np.mean(activation_data == 0)),
                                'activation_ratio': float(np.mean(np.abs(activation_data) > 0.01))
                            },
                            'activation_shape': list(activation.shape)
                        }
                else:
                    # Fallback: manual forward pass with hooks
                    activation_patterns = self._extract_activations_with_hooks(model, input_tensor, target_layers)

            except Exception as e:
                # If activation extraction fails, return empty patterns
                activation_patterns = {}

        return activation_patterns

    def _extract_activations_with_hooks(self,
                                      model: torch.nn.Module,
                                      input_tensor: torch.Tensor,
                                      target_layers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Extract activations using forward hooks.

        Args:
            model: PyTorch model
            input_tensor: Input tensor
            target_layers: Target layer names

        Returns:
            Dictionary of activation patterns
        """
        activations = {}
        hooks = []

        def create_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    activation_data = output.detach().cpu().numpy()
                    activations[name] = {
                        'activation_stats': {
                            'mean': float(np.mean(activation_data)),
                            'std': float(np.std(activation_data)),
                            'min': float(np.min(activation_data)),
                            'max': float(np.max(activation_data)),
                            'sparsity': float(np.mean(activation_data == 0)),
                            'activation_ratio': float(np.mean(np.abs(activation_data) > 0.01))
                        },
                        'activation_shape': list(output.shape)
                    }
            return hook_fn

        # Register hooks for target layers
        for name, module in model.named_modules():
            if name == '':
                continue
            if not target_layers or any(target in name for target in target_layers):
                hooks.append(module.register_forward_hook(create_hook(name)))

        try:
            # Forward pass
            _ = model(input_tensor)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        return activations

    def _count_parameters(self, model: torch.nn.Module) -> int:
        """Count total parameters in the model."""
        return sum(p.numel() for p in model.parameters())

    def _identify_important_layers(self, weight_importance: Dict[str, Dict[str, Any]]) -> List[str]:
        """Identify important layers based on weight importance."""
        important_layers = []

        for layer_name, importance_info in weight_importance.items():
            normalized_score = importance_info.get('normalized_importance',
                                                importance_info.get('adjusted_importance', 0))
            if normalized_score > self.layer_importance_threshold:
                important_layers.append(layer_name)

        return important_layers

    def _compute_weight_statistics(self, operator_weights: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Compute overall weight statistics."""
        all_weights = []
        weight_norms = []

        for weight_info in operator_weights.values():
            weight_values = weight_info.get('weight_values', np.array([]))
            if len(weight_values) > 0:
                all_weights.extend(weight_values.flatten())
            weight_norms.append(weight_info.get('norm', 0))

        stats = {}
        if all_weights:
            all_weights = np.array(all_weights)
            stats['overall_mean'] = float(np.mean(all_weights))
            stats['overall_std'] = float(np.std(all_weights))
            stats['overall_min'] = float(np.min(all_weights))
            stats['overall_max'] = float(np.max(all_weights))
            stats['weight_sparsity'] = float(np.mean(np.abs(all_weights) < 0.01))

        if weight_norms:
            stats['avg_layer_norm'] = float(np.mean(weight_norms))
            stats['max_layer_norm'] = float(np.max(weight_norms))

        return stats

    def _rank_layers_by_importance(self, weight_importance: Dict[str, Dict[str, Any]]) -> List[Tuple[str, float]]:
        """Rank layers by importance score."""
        layer_scores = [
            (layer_name, info.get('normalized_importance', info.get('adjusted_importance', 0)))
            for layer_name, info in weight_importance.items()
        ]
        return sorted(layer_scores, key=lambda x: x[1], reverse=True)[:self.top_k_operators]

    def _analyze_operator_types(self, operator_weights: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Analyze weight statistics by operator type."""
        type_stats = {}

        for layer_name, weight_info in operator_weights.items():
            operator_type = weight_info.get('operator_type', 'Unknown')
            num_params = weight_info.get('num_parameters', 0)
            importance = weight_info.get('importance_score', 0)

            if operator_type not in type_stats:
                type_stats[operator_type] = {
                    'layer_count': 0,
                    'total_parameters': 0,
                    'avg_importance': 0,
                    'layers': []
                }

            type_stats[operator_type]['layer_count'] += 1
            type_stats[operator_type]['total_parameters'] += num_params
            type_stats[operator_type]['layers'].append(layer_name)

        # Compute average importance per type
        for operator_type in type_stats:
            layers = type_stats[operator_type]['layers']
            importances = [
                operator_weights[layer].get('importance_score', 0)
                for layer in layers
            ]
            type_stats[operator_type]['avg_importance'] = float(np.mean(importances)) if importances else 0.0

        return type_stats

    def _infer_operator_type_from_name(self, layer_name: str) -> str:
        """Infer operator type from layer name."""
        layer_name_lower = layer_name.lower()

        if 'conv' in layer_name_lower:
            return 'Convolution'
        elif 'linear' in layer_name_lower or 'fc' in layer_name_lower:
            return 'Linear'
        elif 'fft' in layer_name_lower:
            return 'FFT'
        elif 'wavelet' in layer_name_lower or 'wf' in layer_name_lower:
            return 'Wavelet'
        elif 'hilbert' in layer_name_lower or 'ht' in layer_name_lower:
            return 'Hilbert'
        elif 'attention' in layer_name_lower or 'attn' in layer_name_lower:
            return 'Attention'
        elif 'pool' in layer_name_lower:
            return 'Pooling'
        elif 'activation' in layer_name_lower or 'relu' in layer_name_lower:
            return 'Activation'
        elif 'norm' in layer_name_lower or 'batchnorm' in layer_name_lower:
            return 'Normalization'
        elif 'lno' in layer_name_lower:
            return 'LNO'
        elif 'embedding' in layer_name_lower:
            return 'Embedding'
        else:
            return 'Unknown'

    # Visualization methods
    def _visualize_weight_overview(self, explanation: Explanation) -> plt.Figure:
        """Create overview visualization of operator weights."""
        weight_importance = explanation.get_data('weight_importance', {})
        method_specific = explanation.get_data('method_specific', {})

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Top important layers
        layer_rankings = method_specific.get('layer_rankings', [])
        if layer_rankings:
            layer_names = [name for name, _ in layer_rankings[:10]]
            scores = [score for _, score in layer_rankings[:10]]

            axes[0, 0].barh(range(len(layer_names)), scores)
            axes[0, 0].set_yticks(range(len(layer_names)))
            axes[0, 0].set_yticklabels(layer_names)
            axes[0, 0].set_title('Top 10 Important Layers')
            axes[0, 0].set_xlabel('Importance Score')
        else:
            axes[0, 0].text(0.5, 0.5, 'No layer rankings available', ha='center', va='center', transform=axes[0, 0].transAxes)

        # Operator type distribution
        operator_types = method_specific.get('operator_type_analysis', {})
        if operator_types:
            type_names = list(operator_types.keys())
            type_counts = [info['layer_count'] for info in operator_types.values()]

            axes[0, 1].pie(type_counts, labels=type_names, autopct='%1.1f%%')
            axes[0, 1].set_title('Operator Type Distribution')
        else:
            axes[0, 1].text(0.5, 0.5, 'No operator type data available', ha='center', va='center', transform=axes[0, 1].transAxes)

        # Weight statistics
        weight_stats = method_specific.get('weight_statistics', {})
        if weight_stats:
            stats_text = "Weight Statistics:\n"
            for key, value in weight_stats.items():
                stats_text += f"{key}: {value:.4f}\n"
            axes[1, 0].text(0.1, 0.9, stats_text, transform=axes[1, 0].transAxes, verticalalignment='top')
            axes[1, 0].set_title('Overall Weight Statistics')
            axes[1, 0].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, 'No weight statistics available', ha='center', va='center', transform=axes[1, 0].transAxes)

        # Summary information
        total_params = method_specific.get('total_parameters', 0)
        important_layers_count = len(method_specific.get('important_layers', []))
        summary_text = f"Total Parameters: {total_params:,}\n"
        summary_text += f"Important Layers: {important_layers_count}\n"
        summary_text += f"Total Layers Analyzed: {len(weight_importance)}"

        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, verticalalignment='center', fontsize=12)
        axes[1, 1].set_title('Analysis Summary')
        axes[1, 1].axis('off')

        plt.tight_layout()
        return fig

    def _visualize_operator_weights(self, explanation: Explanation) -> plt.Figure:
        """Visualize operator weight distributions."""
        operator_weights = explanation.get_data('operator_weights', {})

        if not operator_weights:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'No weight data available', ha='center', va='center', transform=ax.transAxes)
            return fig

        # Select top layers by parameter count for visualization
        sorted_layers = sorted(operator_weights.items(),
                             key=lambda x: x[1]['num_parameters'],
                             reverse=True)[:6]

        n_layers = len(sorted_layers)
        if n_layers == 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'No layers with weights found', ha='center', va='center', transform=ax.transAxes)
            return fig

        # Calculate grid dimensions
        cols = min(3, n_layers)
        rows = (n_layers + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_layers == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for i, (layer_name, weight_info) in enumerate(sorted_layers):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]

            weight_values = weight_info.get('weight_values', np.array([]))
            if len(weight_values) > 0:
                # Plot histogram of weight values
                ax.hist(weight_values.flatten(), bins=50, alpha=0.7, density=True)
                ax.set_title(f"{layer_name}\n({weight_info['num_parameters']:,} params)")
                ax.set_xlabel('Weight Value')
                ax.set_ylabel('Density')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No weight data', ha='center', va='center', transform=ax.transAxes)

        # Hide unused subplots
        for i in range(n_layers, rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.set_visible(False)

        plt.tight_layout()
        return fig

    def _visualize_weight_importance(self, explanation: Explanation) -> plt.Figure:
        """Visualize weight importance scores."""
        weight_importance = explanation.get_data('weight_importance', {})

        if not weight_importance:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'No importance data available', ha='center', va='center', transform=ax.transAxes)
            return fig

        # Prepare data for visualization
        layer_names = list(weight_importance.keys())
        importance_scores = [info.get('normalized_importance', info.get('adjusted_importance', 0))
                           for info in weight_importance.values()]
        num_parameters = [info.get('num_parameters', 0) for info in weight_importance.values()]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Importance scores bar chart
        bars = ax1.barh(range(len(layer_names)), importance_scores)
        ax1.set_yticks(range(len(layer_names)))
        ax1.set_yticklabels(layer_names)
        ax1.set_title('Layer Weight Importance Scores')
        ax1.set_xlabel('Normalized Importance Score')

        # Add value labels on bars
        for bar, score in zip(bars, importance_scores):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{score:.3f}', ha='left', va='center')

        # Parameter counts
        ax2.bar(range(len(layer_names)), np.log10(np.array(num_parameters) + 1))
        ax2.set_xticks(range(len(layer_names)))
        ax2.set_xticklabels(layer_names, rotation=45, ha='right')
        ax2.set_title('Layer Parameter Counts (log scale)')
        ax2.set_ylabel('log10(Number of Parameters)')

        plt.tight_layout()
        return fig

    def _visualize_activation_patterns(self, explanation: Explanation) -> plt.Figure:
        """Visualize activation patterns."""
        activation_patterns = explanation.get_data('activation_patterns', {})

        if not activation_patterns:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'No activation data available', ha='center', va='center', transform=ax.transAxes)
            return fig

        # Extract activation statistics
        layer_names = list(activation_patterns.keys())
        activation_ratios = []
        sparsity_values = []

        for layer_name in layer_names:
            stats = activation_patterns[layer_name].get('activation_stats', {})
            activation_ratios.append(stats.get('activation_ratio', 0))
            sparsity_values.append(stats.get('sparsity', 0))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Activation ratios
        bars1 = ax1.bar(range(len(layer_names)), activation_ratios)
        ax1.set_xticks(range(len(layer_names)))
        ax1.set_xticklabels(layer_names, rotation=45, ha='right')
        ax1.set_title('Activation Ratios by Layer')
        ax1.set_ylabel('Activation Ratio')

        # Sparsity values
        bars2 = ax2.bar(range(len(layer_names)), sparsity_values)
        ax2.set_xticks(range(len(layer_names)))
        ax2.set_xticklabels(layer_names, rotation=45, ha='right')
        ax2.set_title('Sparsity by Layer')
        ax2.set_ylabel('Sparsity (fraction of zeros)')

        plt.tight_layout()
        return fig

    def _visualize_layer_comparison(self, explanation: Explanation) -> plt.Figure:
        """Create comparison visualization of different aspects."""
        weight_importance = explanation.get_data('weight_importance', {})
        activation_patterns = explanation.get_data('activation_patterns', {})

        # Find common layers
        weight_layers = set(weight_importance.keys())
        activation_layers = set(activation_patterns.keys())
        common_layers = list(weight_layers.intersection(activation_layers))

        if not common_layers:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'No common layers for comparison', ha='center', va='center', transform=ax.transAxes)
            return fig

        # Extract comparison metrics
        importance_scores = []
        activation_ratios = []
        layer_names = []

        for layer_name in common_layers[:10]:  # Limit to top 10
            importance_scores.append(weight_importance[layer_name].get('normalized_importance', 0))
            activation_stats = activation_patterns[layer_name].get('activation_stats', {})
            activation_ratios.append(activation_stats.get('activation_ratio', 0))
            layer_names.append(layer_name)

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(layer_names))
        width = 0.35

        bars1 = ax.bar(x - width/2, importance_scores, width, label='Weight Importance', alpha=0.7)
        bars2 = ax.bar(x + width/2, activation_ratios, width, label='Activation Ratio', alpha=0.7)

        ax.set_xlabel('Layers')
        ax.set_ylabel('Score')
        ax.set_title('Weight Importance vs Activation Ratio')
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig