"""
Operator Attention Analysis and Visualization Tools

This module provides specialized tools for analyzing and visualizing Operator Attention
mechanisms, including attention weight analysis, operator importance visualization,
and comparison with traditional self-attention mechanisms.

Key Features:
- Operator attention weight visualization
- Temporal analysis of operator selection patterns
- Comparison with self-attention mechanisms
- Complexity analysis and performance metrics
- Integration with Explainable_FD_Toolkit framework
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Sequence, List, Tuple, Union
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
import time
import warnings

from ...core import SignalData, Explanation, BaseExplainerAdapter, ExplainabilityMethod


@dataclass
class OperatorAttentionMetrics:
    """Data class for operator attention performance metrics."""
    complexity_analysis: Dict[str, float]
    performance_metrics: Dict[str, float]
    attention_statistics: Dict[str, float]
    operator_importance: Dict[str, float]
    temporal_patterns: Optional[np.ndarray] = None
    comparison_with_self_attention: Optional[Dict[str, float]] = None


class OperatorAttentionExplainer(BaseExplainerAdapter, ExplainabilityMethod):
    """
    Specialized explainer for Operator Attention mechanisms.

    This explainer provides in-depth analysis of operator attention weights,
    temporal patterns, and comparisons with traditional attention mechanisms.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Operator Attention Explainer.

        Args:
            config: Configuration dictionary with the following options:
                - include_temporal_analysis: bool (default True)
                - include_complexity_analysis: bool (default True)
                - include_self_attention_comparison: bool (default True)
                - operator_names: List[str] (default ['FFT', 'HT', 'WF', 'I'])
                - attention_aggregation: str ('mean', 'max', 'weighted_mean', default 'mean')
                - visualize_temporal_patterns: bool (default True)
                - complexity_analysis_method: str ('theoretical', 'empirical', 'both', default 'both')
        """
        super().__init__(config)
        self._method_name = "OperatorAttention"
        self._method_type = "intrinsic"

        # Configuration options
        self.include_temporal_analysis = self.config.get('include_temporal_analysis', True)
        self.include_complexity_analysis = self.config.get('include_complexity_analysis', True)
        self.include_self_attention_comparison = self.config.get('include_self_attention_comparison', True)
        self.operator_names = self.config.get('operator_names', ['FFT', 'HT', 'WF', 'I'])
        self.attention_aggregation = self.config.get('attention_aggregation', 'mean')
        self.visualize_temporal_patterns = self.config.get('visualize_temporal_patterns', True)
        self.complexity_analysis_method = self.config.get('complexity_analysis_method', 'both')

    def explain(self,
                signal: SignalData,
                prediction: Any,
                model: Optional[torch.nn.Module] = None,
                **kwargs) -> Explanation:
        """
        Generate operator attention explanation.

        Args:
            signal: SignalData object containing the signal to explain
            prediction: Model prediction or target class
            model: Optional model with operator attention mechanisms
            **kwargs: Additional arguments including:
                - attention_weights: Pre-computed attention weights
                - operator_outputs: Pre-computed operator outputs
                - baseline_model: Model without operator attention for comparison
                - compute_complexity: bool (default True)

        Returns:
            Explanation object containing operator attention analysis results
        """
        self._validate_signal(signal)

        # Get model if provided
        model = model or kwargs.get('model')
        if model is None:
            raise ValueError("Model must be provided for operator attention analysis")

        # Extract operator attention data
        attention_data = self._extract_operator_attention_data(model, signal, **kwargs)

        # Analyze attention patterns
        attention_analysis = self._analyze_attention_patterns(attention_data, **kwargs)

        # Temporal analysis if requested
        temporal_analysis = None
        if self.include_temporal_analysis:
            temporal_analysis = self._perform_temporal_analysis(attention_data, signal)

        # Complexity analysis if requested
        complexity_analysis = None
        if self.include_complexity_analysis:
            complexity_analysis = self._perform_complexity_analysis(model, attention_data)

        # Self-attention comparison if requested
        self_attention_comparison = None
        if self.include_self_attention_comparison:
            baseline_model = kwargs.get('baseline_model')
            if baseline_model is not None:
                self_attention_comparison = self._compare_with_self_attention(
                    model, baseline_model, signal, **kwargs
                )

        # Generate explanation data
        explanation_data = {
            'attention_weights': attention_data.get('attention_weights'),
            'operator_importance': attention_analysis.get('operator_importance'),
            'attention_entropy': attention_analysis.get('attention_entropy'),
            'attention_sparsity': attention_analysis.get('attention_sparsity'),
            'temporal_analysis': temporal_analysis,
            'complexity_analysis': complexity_analysis,
            'self_attention_comparison': self_attention_comparison,
            'original_signal': signal.raw_signal,
            'method_specific': {
                'operator_names': self.operator_names,
                'attention_aggregation': self.attention_aggregation,
                'total_attention_weight': float(np.sum(attention_data.get('attention_weights', np.array([])))),
                'dominant_operator': self._find_dominant_operator(attention_analysis),
                'attention_consistency': self._compute_attention_consistency(attention_data),
                'operator_activation_patterns': self._analyze_operator_activation_patterns(attention_data)
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
            'config': self.get_config(),
            'model_type': self._infer_model_type(model)
        }

        return Explanation(explanation_data, explanation_meta)

    def visualize(self,
                  explanation: Explanation,
                  mode: str = 'auto',
                  **kwargs) -> plt.Figure:
        """
        Create visualization for operator attention explanation.

        Args:
            explanation: Explanation object to visualize
            mode: Visualization mode ('auto', 'attention_weights', 'temporal_patterns',
                                    'complexity_comparison', 'operator_importance')
            **kwargs: Additional visualization parameters

        Returns:
            Matplotlib figure object
        """
        self._validate_explanation(explanation)

        if mode == 'auto':
            return self._visualize_comprehensive_overview(explanation)
        elif mode == 'attention_weights':
            return self._visualize_attention_weights(explanation)
        elif mode == 'temporal_patterns':
            return self._visualize_temporal_patterns(explanation)
        elif mode == 'complexity_comparison':
            return self._visualize_complexity_comparison(explanation)
        elif mode == 'operator_importance':
            return self._visualize_operator_importance(explanation)
        elif mode == 'attention_entropy':
            return self._visualize_attention_entropy(explanation)
        else:
            raise ValueError(f"Unknown visualization mode: {mode}")

    def evaluate(self,
                 explanations: Sequence[Explanation],
                 ground_truth: Optional[Sequence[Any]] = None,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate operator attention explanations.

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

        # Attention-based metrics
        attention_entropies = []
        attention_sparsities = []
        operator_consistencies = []
        dominant_operator_ratios = []

        for exp in explanations:
            attention_weights = exp.get_data('attention_weights', np.array([]))
            method_specific = exp.get_data('method_specific', {})

            # Compute attention entropy if not provided
            if len(attention_weights) > 0:
                if 'attention_entropy' in exp.get_data('attention_entropy', {}):
                    entropy = exp.get_data('attention_entropy', {}).get('entropy', 0)
                else:
                    entropy = self._compute_entropy(attention_weights)
                attention_entropies.append(entropy)

                # Compute sparsity
                sparsity = self._compute_sparsity(attention_weights)
                attention_sparsities.append(sparsity)

            # Operator consistency
            consistency = method_specific.get('attention_consistency', 0.0)
            operator_consistencies.append(consistency)

            # Dominant operator analysis
            dominant_operator = method_specific.get('dominant_operator', {})
            if dominant_operator:
                dominant_ratio = dominant_operator.get('dominance_ratio', 0.0)
                dominant_operator_ratios.append(dominant_ratio)

        # Compute aggregated metrics
        metrics['avg_attention_entropy'] = float(np.mean(attention_entropies)) if attention_entropies else 0.0
        metrics['avg_attention_sparsity'] = float(np.mean(attention_sparsities)) if attention_sparsities else 0.0
        metrics['avg_operator_consistency'] = float(np.mean(operator_consistencies)) if operator_consistencies else 0.0
        metrics['avg_dominant_operator_ratio'] = float(np.mean(dominant_operator_ratios)) if dominant_operator_ratios else 0.0

        # Interpretability metrics
        metrics['attention_interpretability'] = self._compute_interpretability_score(metrics)
        metrics['operator_selection_confidence'] = float(1.0 - metrics['avg_attention_entropy'] / np.log2(len(self.operator_names))) if len(self.operator_names) > 0 else 0.0

        return metrics

    def _extract_operator_attention_data(self,
                                        model: torch.nn.Module,
                                        signal: SignalData,
                                        **kwargs) -> Dict[str, Any]:
        """Extract operator attention data from model."""
        attention_data = {}

        # Check if attention weights are provided
        if 'attention_weights' in kwargs:
            attention_data['attention_weights'] = kwargs['attention_weights']
        elif hasattr(model, 'get_operator_attention_weights'):
            # Extract from model
            signal_tensor = self._signal_to_tensor(signal)
            attention_weights = model.get_operator_attention_weights(signal_tensor)
            attention_data['attention_weights'] = attention_weights
        else:
            # Try to extract from TSPN with operator attention
            attention_data = self._extract_from_tspn_model(model, signal)

        return attention_data

    def _extract_from_tspn_model(self,
                                model: torch.nn.Module,
                                signal: SignalData) -> Dict[str, Any]:
        """Extract attention data from TSPN model with operator attention."""
        attention_data = {}

        try:
            signal_tensor = self._signal_to_tensor(signal)

            # Check if model has operator attention layers
            if hasattr(model, 'signal_processing_layers'):
                attention_weights_list = []

                for layer in model.signal_processing_layers:
                    if hasattr(layer, 'operator_attention') and hasattr(layer.operator_attention, 'last_attention_weights'):
                        attention_weights_list.append(layer.operator_attention.last_attention_weights)

                if attention_weights_list:
                    # Aggregate attention weights across layers
                    attention_data['attention_weights'] = self._aggregate_attention_weights(attention_weights_list)

            # Additional extraction methods can be added here

        except Exception as e:
            print(f"Warning: Could not extract attention weights from model: {e}")
            attention_data['attention_weights'] = np.array([])

        return attention_data

    def _aggregate_attention_weights(self, attention_weights_list: List[np.ndarray]) -> np.ndarray:
        """Aggregate attention weights from multiple layers."""
        if not attention_weights_list:
            return np.array([])

        stacked_weights = np.stack(attention_weights_list, axis=1)  # (batch, layers, operators)

        if self.attention_aggregation == 'mean':
            return np.mean(stacked_weights, axis=1)
        elif self.attention_aggregation == 'max':
            return np.max(stacked_weights, axis=1)
        elif self.attention_aggregation == 'weighted_mean':
            # Simple weighted mean where later layers have higher weights
            weights = np.linspace(0.5, 1.5, stacked_weights.shape[1])
            weighted_weights = stacked_weights * weights[None, :, None]
            return np.mean(weighted_weights, axis=1)
        else:
            return np.mean(stacked_weights, axis=1)

    def _analyze_attention_patterns(self,
                                   attention_data: Dict[str, Any],
                                   **kwargs) -> Dict[str, Any]:
        """Analyze operator attention patterns."""
        attention_weights = attention_data.get('attention_weights', np.array([]))

        if len(attention_weights) == 0:
            return {}

        analysis = {}

        # Operator importance (average attention weight per operator)
        operator_importance = {}
        for i, op_name in enumerate(self.operator_names):
            if i < attention_weights.shape[-1]:
                operator_importance[op_name] = float(np.mean(attention_weights[:, i]))
        analysis['operator_importance'] = operator_importance

        # Attention entropy
        analysis['attention_entropy'] = {
            'entropy': float(np.mean([self._compute_entropy(weights) for weights in attention_weights])),
            'entropy_std': float(np.std([self._compute_entropy(weights) for weights in attention_weights]))
        }

        # Attention sparsity
        analysis['attention_sparsity'] = {
            'sparsity': float(np.mean([self._compute_sparsity(weights) for weights in attention_weights])),
            'sparsity_std': float(np.std([self._compute_sparsity(weights) for weights in attention_weights]))
        }

        # Most/least attended operators
        avg_weights = np.mean(attention_weights, axis=0)
        most_attended_idx = np.argmax(avg_weights)
        least_attended_idx = np.argmin(avg_weights)

        analysis['most_attended_operator'] = {
            'name': self.operator_names[most_attended_idx] if most_attended_idx < len(self.operator_names) else 'Unknown',
            'weight': float(avg_weights[most_attended_idx])
        }

        analysis['least_attended_operator'] = {
            'name': self.operator_names[least_attended_idx] if least_attended_idx < len(self.operator_names) else 'Unknown',
            'weight': float(avg_weights[least_attended_idx])
        }

        return analysis

    def _perform_temporal_analysis(self,
                                   attention_data: Dict[str, Any],
                                   signal: SignalData) -> Dict[str, Any]:
        """Perform temporal analysis of operator attention."""
        attention_weights = attention_data.get('attention_weights', np.array([]))

        if len(attention_weights) == 0:
            return {}

        temporal_analysis = {}

        # Temporal stability (how attention weights change over time/batch)
        temporal_analysis['attention_stability'] = float(np.std(attention_weights, axis=0).mean())

        # Temporal patterns (if we have temporal sequence)
        if attention_weights.shape[0] > 1:
            # Compute attention change over time
            attention_diff = np.diff(attention_weights, axis=0)
            temporal_analysis['temporal_variance'] = float(np.var(attention_diff))

            # Identify attention switching points
            switching_points = np.where(np.max(np.abs(attention_diff), axis=1) > 0.1)[0]
            temporal_analysis['attention_switches'] = len(switching_points)
            temporal_analysis['switching_frequency'] = len(switching_points) / len(attention_weights)

        return temporal_analysis

    def _perform_complexity_analysis(self,
                                     model: torch.nn.Module,
                                     attention_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform complexity analysis of operator attention."""
        complexity_analysis = {}

        # Theoretical complexity
        if self.complexity_analysis_method in ['theoretical', 'both']:
            complexity_analysis['theoretical'] = self._compute_theoretical_complexity(model, attention_data)

        # Empirical complexity
        if self.complexity_analysis_method in ['empirical', 'both']:
            complexity_analysis['empirical'] = self._compute_empirical_complexity(model)

        return complexity_analysis

    def _compute_theoretical_complexity(self,
                                        model: torch.nn.Module,
                                        attention_data: Dict[str, Any]) -> Dict[str, float]:
        """Compute theoretical complexity of operator attention."""
        attention_weights = attention_data.get('attention_weights', np.array([]))

        # Basic complexity metrics
        K = len(self.operator_names)  # Number of operators
        L = 1024  # Typical sequence length (can be made configurable)
        C = 2     # Number of channels (can be made configurable)

        # Operator Attention complexity: O(K * L * C)
        op_attention_complexity = K * L * C

        # Standard Self-Attention complexity: O(L^2 * C)
        self_attention_complexity = L * L * C

        # Memory complexity
        op_attention_memory = K * L * C
        self_attention_memory = L * L

        return {
            'operator_attention_flops': op_attention_complexity,
            'self_attention_flops': self_attention_complexity,
            'complexity_ratio': op_attention_complexity / self_attention_complexity,
            'operator_attention_memory': op_attention_memory,
            'self_attention_memory': self_attention_memory,
            'memory_ratio': op_attention_memory / self_attention_memory
        }

    def _compute_empirical_complexity(self, model: torch.nn.Module) -> Dict[str, float]:
        """Compute empirical complexity through timing."""
        complexity_metrics = {}

        # Create dummy input
        dummy_input = torch.randn(1, 1024, 2)

        # Measure inference time
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            for _ in range(10):  # Average over 10 runs
                _ = model(dummy_input)
            end_time = time.time()

        avg_inference_time = (end_time - start_time) / 10

        # Count parameters
        num_parameters = sum(p.numel() for p in model.parameters())

        complexity_metrics['avg_inference_time'] = avg_inference_time
        complexity_metrics['num_parameters'] = num_parameters
        complexity_metrics['flops_estimate'] = num_parameters * 2  # Rough estimate

        return complexity_metrics

    def _compare_with_self_attention(self,
                                     operator_attention_model: torch.nn.Module,
                                     baseline_model: torch.nn.Module,
                                     signal: SignalData,
                                     **kwargs) -> Dict[str, Any]:
        """Compare operator attention with baseline self-attention model."""
        comparison = {}

        signal_tensor = self._signal_to_tensor(signal)

        # Measure performance metrics
        models = {
            'operator_attention': operator_attention_model,
            'self_attention': baseline_model
        }

        performance_metrics = {}

        for model_name, model in models.items():
            model.eval()
            with torch.no_grad():
                start_time = time.time()
                output = model(signal_tensor)
                end_time = time.time()

            performance_metrics[model_name] = {
                'inference_time': end_time - start_time,
                'output_shape': list(output.shape) if hasattr(output, 'shape') else None,
                'num_parameters': sum(p.numel() for p in model.parameters())
            }

        comparison['performance_metrics'] = performance_metrics
        comparison['speedup_ratio'] = (performance_metrics['self_attention']['inference_time'] /
                                     performance_metrics['operator_attention']['inference_time'])

        return comparison

    def _signal_to_tensor(self, signal: SignalData) -> torch.Tensor:
        """Convert SignalData to PyTorch tensor."""
        if isinstance(signal.raw_signal, np.ndarray):
            tensor = torch.FloatTensor(signal.raw_signal)
        else:
            tensor = signal.raw_signal

        # Ensure proper shape (B, L, C)
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0)

        return tensor

    def _compute_entropy(self, attention_weights: np.ndarray) -> float:
        """Compute entropy of attention weights."""
        # Add small epsilon to avoid log(0)
        weights = attention_weights + 1e-10
        weights = weights / np.sum(weights)  # Normalize
        return -np.sum(weights * np.log2(weights))

    def _compute_sparsity(self, attention_weights: np.ndarray) -> float:
        """Compute sparsity of attention weights."""
        threshold = 1.0 / len(attention_weights)  # Uniform distribution threshold
        return float(np.mean(attention_weights < threshold))

    def _find_dominant_operator(self, attention_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Find the dominant operator."""
        operator_importance = attention_analysis.get('operator_importance', {})

        if not operator_importance:
            return {}

        dominant_op = max(operator_importance.items(), key=lambda x: x[1])
        total_importance = sum(operator_importance.values())

        return {
            'name': dominant_op[0],
            'importance': dominant_op[1],
            'dominance_ratio': dominant_op[1] / total_importance if total_importance > 0 else 0.0
        }

    def _compute_attention_consistency(self, attention_data: Dict[str, Any]) -> float:
        """Compute consistency of attention weights."""
        attention_weights = attention_data.get('attention_weights', np.array([]))

        if len(attention_weights) <= 1:
            return 1.0

        # Compute pairwise correlations
        correlations = []
        for i in range(len(attention_weights) - 1):
            corr = np.corrcoef(attention_weights[i], attention_weights[i + 1])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

        return float(np.mean(correlations)) if correlations else 0.0

    def _analyze_operator_activation_patterns(self, attention_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze operator activation patterns."""
        attention_weights = attention_data.get('attention_weights', np.array([]))

        if len(attention_weights) == 0:
            return {}

        patterns = {}

        # Activation frequency per operator
        threshold = 1.0 / len(self.operator_names)  # Above uniform threshold
        activation_frequency = (attention_weights > threshold).mean(axis=0)

        for i, op_name in enumerate(self.operator_names):
            if i < len(activation_frequency):
                patterns[op_name] = {
                    'activation_frequency': float(activation_frequency[i]),
                    'avg_weight': float(np.mean(attention_weights[:, i])),
                    'weight_std': float(np.std(attention_weights[:, i]))
                }

        return patterns

    def _compute_interpretability_score(self, metrics: Dict[str, float]) -> float:
        """Compute overall interpretability score."""
        # Combine multiple metrics into a single interpretability score
        entropy_component = 1.0 - metrics['avg_attention_entropy'] / 4.0  # Normalize
        sparsity_component = metrics['avg_attention_sparsity']
        consistency_component = metrics['avg_operator_consistency']

        # Weighted combination
        interpretability_score = (0.4 * entropy_component +
                                0.3 * sparsity_component +
                                0.3 * consistency_component)

        return float(np.clip(interpretability_score, 0.0, 1.0))

    def _infer_model_type(self, model: torch.nn.Module) -> str:
        """Infer model type from model structure."""
        model_name = model.__class__.__name__.lower()

        if 'operator_attention' in model_name or hasattr(model, 'operator_attention'):
            return 'OperatorAttention'
        elif 'tspn' in model_name:
            return 'TSPN'
        elif 'attention' in model_name:
            return 'SelfAttention'
        else:
            return 'Unknown'

    # Visualization methods
    def _visualize_comprehensive_overview(self, explanation: Explanation) -> plt.Figure:
        """Create comprehensive overview of operator attention analysis."""
        fig = plt.figure(figsize=(20, 12))

        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # 1. Operator importance (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_operator_importance(ax1, explanation)

        # 2. Attention weights distribution (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_attention_weights_distribution(ax2, explanation)

        # 3. Attention entropy over time (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_attention_entropy(ax3, explanation)

        # 4. Complexity comparison (top far right)
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_complexity_comparison(ax4, explanation)

        # 5. Temporal patterns (middle left)
        ax5 = fig.add_subplot(gs[1, :2])
        self._plot_temporal_patterns(ax5, explanation)

        # 6. Operator activation patterns (middle right)
        ax6 = fig.add_subplot(gs[1, 2:])
        self._plot_operator_activation_patterns(ax6, explanation)

        # 7. Summary statistics (bottom left)
        ax7 = fig.add_subplot(gs[2, :2])
        self._plot_summary_statistics(ax7, explanation)

        # 8. Interpretability metrics (bottom right)
        ax8 = fig.add_subplot(gs[2, 2:])
        self._plot_interpretability_metrics(ax8, explanation)

        fig.suptitle('Operator Attention Comprehensive Analysis', fontsize=16, fontweight='bold')
        return fig

    def _plot_operator_importance(self, ax: plt.Axes, explanation: Explanation):
        """Plot operator importance scores."""
        attention_analysis = explanation.get_data('method_specific', {})
        importance_data = explanation.get_data('operator_importance', {})

        if importance_data:
            operators = list(importance_data.keys())
            scores = list(importance_data.values())

            bars = ax.bar(operators, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            ax.set_title('Operator Importance Scores')
            ax.set_ylabel('Average Attention Weight')
            ax.set_ylim(0, 1)

            # Add value labels
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No operator importance data', ha='center', va='center', transform=ax.transAxes)

    def _plot_attention_weights_distribution(self, ax: plt.Axes, explanation: Explanation):
        """Plot distribution of attention weights."""
        attention_weights = explanation.get_data('attention_weights', np.array([]))

        if len(attention_weights) > 0:
            # Create heatmap of attention weights
            if len(attention_weights.shape) == 2:
                im = ax.imshow(attention_weights.T, aspect='auto', cmap='viridis', interpolation='nearest')
                ax.set_xlabel('Time/Batch')
                ax.set_ylabel('Operators')
                ax.set_title('Attention Weights Heatmap')

                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # Set operator labels
                ax.set_yticks(range(len(self.operator_names)))
                ax.set_yticklabels(self.operator_names)
            else:
                ax.text(0.5, 0.5, f'Attention weights shape: {attention_weights.shape}',
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No attention weights data', ha='center', va='center', transform=ax.transAxes)

    def _plot_attention_entropy(self, ax: plt.Axes, explanation: Explanation):
        """Plot attention entropy statistics."""
        entropy_data = explanation.get_data('attention_entropy', {})

        if entropy_data:
            entropy = entropy_data.get('entropy', 0)
            entropy_std = entropy_data.get('entropy_std', 0)

            # Create bar plot with error bars
            ax.bar(['Attention Entropy'], [entropy], yerr=[entropy_std], capsize=5, color='orange')
            ax.set_ylabel('Entropy (bits)')
            ax.set_title('Attention Entropy')
            ax.set_ylim(0, max(2.0, entropy + entropy_std + 0.5))

            # Add reference line for maximum entropy
            max_entropy = np.log2(len(self.operator_names))
            ax.axhline(y=max_entropy, color='red', linestyle='--', label=f'Max Entropy ({max_entropy:.2f})')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No entropy data available', ha='center', va='center', transform=ax.transAxes)

    def _plot_complexity_comparison(self, ax: plt.Axes, explanation: Explanation):
        """Plot complexity comparison with self-attention."""
        complexity_data = explanation.get_data('complexity_analysis', {})

        if complexity_data and 'theoretical' in complexity_data:
            theoretical = complexity_data['theoretical']

            metrics = ['FLOPs', 'Memory']
            op_attention_values = [
                theoretical.get('operator_attention_flops', 1),
                theoretical.get('operator_attention_memory', 1)
            ]
            self_attention_values = [
                theoretical.get('self_attention_flops', 1),
                theoretical.get('self_attention_memory', 1)
            ]

            x = np.arange(len(metrics))
            width = 0.35

            bars1 = ax.bar(x - width/2, op_attention_values, width, label='Operator Attention', alpha=0.7)
            bars2 = ax.bar(x + width/2, self_attention_values, width, label='Self-Attention', alpha=0.7)

            ax.set_ylabel('Complexity (relative units)')
            ax.set_title('Complexity Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()

            # Add ratio annotations
            ratio = theoretical.get('complexity_ratio', 1)
            ax.text(0.5, 0.95, f'Complexity Ratio: {ratio:.3f}',
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'No complexity data available', ha='center', va='center', transform=ax.transAxes)

    def _plot_temporal_patterns(self, ax: plt.Axes, explanation: Explanation):
        """Plot temporal patterns of operator attention."""
        attention_weights = explanation.get_data('attention_weights', np.array([]))

        if len(attention_weights) > 0 and len(attention_weights.shape) == 2:
            # Plot attention weights over time for each operator
            time_steps = range(attention_weights.shape[0])

            for i, op_name in enumerate(self.operator_names):
                if i < attention_weights.shape[1]:
                    ax.plot(time_steps, attention_weights[:, i], label=op_name, marker='o', markersize=3)

            ax.set_xlabel('Time/Batch')
            ax.set_ylabel('Attention Weight')
            ax.set_title('Temporal Patterns of Operator Attention')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No temporal attention data', ha='center', va='center', transform=ax.transAxes)

    def _plot_operator_activation_patterns(self, ax: plt.Axes, explanation: Explanation):
        """Plot operator activation patterns."""
        method_specific = explanation.get_data('method_specific', {})
        patterns = method_specific.get('operator_activation_patterns', {})

        if patterns:
            operators = list(patterns.keys())
            frequencies = [patterns[op]['activation_frequency'] for op in operators]
            avg_weights = [patterns[op]['avg_weight'] for op in operators]

            x = np.arange(len(operators))
            width = 0.35

            bars1 = ax.bar(x - width/2, frequencies, width, label='Activation Frequency', alpha=0.7)
            bars2 = ax.bar(x + width/2, avg_weights, width, label='Average Weight', alpha=0.7)

            ax.set_xlabel('Operators')
            ax.set_ylabel('Value')
            ax.set_title('Operator Activation Patterns')
            ax.set_xticks(x)
            ax.set_xticklabels(operators)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No activation pattern data', ha='center', va='center', transform=ax.transAxes)

    def _plot_summary_statistics(self, ax: plt.Axes, explanation: Explanation):
        """Plot summary statistics."""
        attention_weights = explanation.get_data('attention_weights', np.array([]))
        method_specific = explanation.get_data('method_specific', {})

        stats_text = "Operator Attention Summary Statistics:\n\n"

        if len(attention_weights) > 0:
            stats_text += f"Total Samples: {len(attention_weights)}\n"
            stats_text += f"Number of Operators: {len(self.operator_names)}\n"
            stats_text += f"Average Attention Weight: {np.mean(attention_weights):.4f}\n"
            stats_text += f"Attention Weight Std: {np.std(attention_weights):.4f}\n"
            stats_text += f"Min Attention Weight: {np.min(attention_weights):.4f}\n"
            stats_text += f"Max Attention Weight: {np.max(attention_weights):.4f}\n"

        # Add dominant operator info
        dominant_op = method_specific.get('dominant_operator', {})
        if dominant_op:
            stats_text += f"\nDominant Operator: {dominant_op.get('name', 'Unknown')}\n"
            stats_text += f"Dominance Ratio: {dominant_op.get('dominance_ratio', 0):.3f}\n"

        # Add consistency info
        consistency = method_specific.get('attention_consistency', 0)
        stats_text += f"Attention Consistency: {consistency:.3f}\n"

        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, verticalalignment='top',
               fontsize=10, family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.1))
        ax.set_title('Summary Statistics')
        ax.axis('off')

    def _plot_interpretability_metrics(self, ax: plt.Axes, explanation: Explanation):
        """Plot interpretability metrics."""
        attention_weights = explanation.get_data('attention_weights', np.array([]))
        entropy_data = explanation.get_data('attention_entropy', {})
        sparsity_data = explanation.get_data('attention_sparsity', {})

        metrics = {}

        if len(attention_weights) > 0:
            # Compute various interpretability metrics
            avg_entropy = entropy_data.get('entropy', 0)
            max_entropy = np.log2(len(self.operator_names))
            entropy_score = 1.0 - (avg_entropy / max_entropy) if max_entropy > 0 else 0

            sparsity_score = sparsity_data.get('sparsity', 0)

            # Concentration score (how concentrated attention is)
            max_weights = np.max(attention_weights, axis=1)
            concentration_score = np.mean(max_weights)

            metrics.update({
                'Entropy\n(lower is better)': entropy_score,
                'Sparsity\n(higher is better)': sparsity_score,
                'Concentration\n(higher is better)': concentration_score
            })

        if metrics:
            names = list(metrics.keys())
            scores = list(metrics.values())

            colors = ['red' if 'lower' in name else 'green' for name in names]
            bars = ax.barh(names, scores, color=colors, alpha=0.7)

            ax.set_xlim(0, 1)
            ax.set_xlabel('Score')
            ax.set_title('Interpretability Metrics')

            # Add value labels
            for bar, score in zip(bars, scores):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                       f'{score:.3f}', ha='left', va='center')

            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No interpretability metrics available',
                   ha='center', va='center', transform=ax.transAxes)

    # Additional visualization methods for specific modes
    def _visualize_attention_weights(self, explanation: Explanation) -> plt.Figure:
        """Visualize attention weights in detail."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        attention_weights = explanation.get_data('attention_weights', np.array([]))

        if len(attention_weights) > 0:
            # Heatmap
            if len(attention_weights.shape) == 2:
                im = axes[0, 0].imshow(attention_weights.T, aspect='auto', cmap='viridis')
                axes[0, 0].set_title('Attention Weights Heatmap')
                axes[0, 0].set_xlabel('Time/Batch')
                axes[0, 0].set_ylabel('Operators')
                plt.colorbar(im, ax=axes[0, 0])

            # Distribution
            axes[0, 1].hist(attention_weights.flatten(), bins=50, alpha=0.7, density=True)
            axes[0, 1].set_title('Attention Weights Distribution')
            axes[0, 1].set_xlabel('Weight Value')
            axes[0, 1].set_ylabel('Density')

            # Box plot per operator
            if len(attention_weights.shape) == 2:
                axes[1, 0].boxplot([attention_weights[:, i] for i in range(min(len(self.operator_names), attention_weights.shape[1]))],
                                  labels=self.operator_names[:attention_weights.shape[1]])
                axes[1, 0].set_title('Attention Weights per Operator')
                axes[1, 0].set_ylabel('Weight Value')

            # Statistics table
            stats_text = "Operator Statistics:\n"
            for i, op_name in enumerate(self.operator_names):
                if i < attention_weights.shape[1]:
                    weights = attention_weights[:, i]
                    stats_text += f"{op_name}: μ={np.mean(weights):.3f}, σ={np.std(weights):.3f}\n"

            axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                           verticalalignment='top', family='monospace')
            axes[1, 1].set_title('Operator Statistics')
            axes[1, 1].axis('off')

        plt.tight_layout()
        return fig

    def _visualize_temporal_patterns(self, explanation: Explanation) -> plt.Figure:
        """Visualize temporal patterns in detail."""
        attention_weights = explanation.get_data('attention_weights', np.array([]))

        if len(attention_weights) == 0 or len(attention_weights.shape) != 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No temporal attention data available',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        time_steps = range(attention_weights.shape[0])

        # Line plot for each operator
        for i, op_name in enumerate(self.operator_names):
            if i < attention_weights.shape[1]:
                axes[0, 0].plot(time_steps, attention_weights[:, i], label=op_name, marker='o', markersize=2)
        axes[0, 0].set_title('Operator Attention Over Time')
        axes[0, 0].set_xlabel('Time/Batch')
        axes[0, 0].set_ylabel('Attention Weight')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Stacked area chart
        if attention_weights.shape[1] == len(self.operator_names):
            axes[0, 1].stackplot(time_steps, attention_weights.T, labels=self.operator_names, alpha=0.7)
            axes[0, 1].set_title('Stacked Attention Weights')
            axes[0, 1].set_xlabel('Time/Batch')
            axes[0, 1].set_ylabel('Cumulative Weight')
            axes[0, 1].legend(loc='upper left')

        # Attention changes (difference plot)
        if len(attention_weights) > 1:
            attention_diff = np.diff(attention_weights, axis=0)
            for i, op_name in enumerate(self.operator_names):
                if i < attention_weights.shape[1]:
                    axes[1, 0].plot(time_steps[:-1], attention_diff[:, i], label=op_name, alpha=0.7)
            axes[1, 0].set_title('Attention Weight Changes')
            axes[1, 0].set_xlabel('Time/Batch')
            axes[1, 0].set_ylabel('Weight Change')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Variance over time
        attention_variance = np.var(attention_weights, axis=1)
        axes[1, 1].plot(time_steps, attention_variance, 'b-', linewidth=2)
        axes[1, 1].set_title('Attention Variance Over Time')
        axes[1, 1].set_xlabel('Time/Batch')
        axes[1, 1].set_ylabel('Variance')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _visualize_complexity_comparison(self, explanation: Explanation) -> plt.Figure:
        """Visualize complexity comparison in detail."""
        complexity_data = explanation.get_data('complexity_analysis', {})

        if not complexity_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No complexity data available',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Theoretical complexity
        if 'theoretical' in complexity_data:
            theoretical = complexity_data['theoretical']

            metrics = ['FLOPs', 'Memory']
            op_values = [theoretical.get('operator_attention_flops', 1),
                        theoretical.get('operator_attention_memory', 1)]
            self_values = [theoretical.get('self_attention_flops', 1),
                          theoretical.get('self_attention_memory', 1)]

            x = np.arange(len(metrics))
            width = 0.35

            axes[0, 0].bar(x - width/2, op_values, width, label='Operator Attention', alpha=0.7)
            axes[0, 0].bar(x + width/2, self_values, width, label='Self-Attention', alpha=0.7)
            axes[0, 0].set_title('Theoretical Complexity Comparison')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(metrics)
            axes[0, 0].legend()
            axes[0, 0].set_ylabel('Complexity (relative units)')

        # Empirical complexity
        if 'empirical' in complexity_data:
            empirical = complexity_data['empirical']

            metrics = ['Inference Time', 'Parameters']
            values = [empirical.get('avg_inference_time', 0),
                     empirical.get('num_parameters', 0)]

            axes[0, 1].bar(metrics, values, color=['orange', 'green'], alpha=0.7)
            axes[0, 1].set_title('Empirical Complexity')
            axes[0, 1].set_ylabel('Value')

            # Add value labels on bars
            for bar, value in zip(axes[0, 1].patches, values):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(values)*0.01,
                               f'{value:.2e}', ha='center', va='bottom')

        # Complexity ratio chart
        if 'theoretical' in complexity_data:
            theoretical = complexity_data['theoretical']

            ratios = {
                'FLOPs Ratio': theoretical.get('complexity_ratio', 1),
                'Memory Ratio': theoretical.get('memory_ratio', 1)
            }

            for i, (metric, ratio) in enumerate(ratios.items()):
                color = 'green' if ratio < 1 else 'red'
                axes[1, 0].bar(metric, ratio, color=color, alpha=0.7)

            axes[1, 0].axhline(y=1, color='black', linestyle='--', label='Equal Complexity')
            axes[1, 0].set_title('Complexity Ratios (< 1 is better)')
            axes[1, 0].set_ylabel('Ratio (OA / SA)')
            axes[1, 0].legend()

        # Summary text
        summary_text = "Complexity Analysis Summary:\n\n"

        if 'theoretical' in complexity_data:
            theoretical = complexity_data['theoretical']
            summary_text += f"Theoretical FLOPs Reduction: {(1 - theoretical.get('complexity_ratio', 0)):.1%}\n"
            summary_text += f"Theoretical Memory Reduction: {(1 - theoretical.get('memory_ratio', 0)):.1%}\n"

        if 'self_attention_comparison' in complexity_data:
            comparison = complexity_data['self_attention_comparison']
            speedup = comparison.get('speedup_ratio', 1)
            summary_text += f"Empirical Speedup: {speedup:.2f}x\n"

        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                       verticalalignment='top', fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        axes[1, 1].set_title('Complexity Summary')
        axes[1, 1].axis('off')

        plt.tight_layout()
        return fig

    def _visualize_operator_importance(self, explanation: Explanation) -> plt.Figure:
        """Visualize operator importance in detail."""
        operator_importance = explanation.get_data('operator_importance', {})

        if not operator_importance:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No operator importance data available',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        operators = list(operator_importance.keys())
        scores = list(operator_importance.values())

        # Bar chart
        bars = axes[0, 0].bar(operators, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[0, 0].set_title('Operator Importance Scores')
        axes[0, 0].set_ylabel('Average Attention Weight')
        axes[0, 0].set_ylim(0, 1)

        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')

        # Pie chart
        axes[0, 1].pie(scores, labels=operators, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Operator Importance Distribution')

        # Normalized scores
        total_score = sum(scores)
        normalized_scores = [score / total_score for score in scores]

        bars2 = axes[1, 0].bar(operators, normalized_scores, color='skyblue', alpha=0.7)
        axes[1, 0].set_title('Normalized Importance Scores')
        axes[1, 0].set_ylabel('Normalized Score')
        axes[1, 0].set_ylim(0, 1)

        # Rank ordering
        sorted_operators = sorted(zip(operators, scores), key=lambda x: x[1], reverse=True)
        sorted_names = [op for op, score in sorted_operators]
        sorted_scores = [score for op, score in sorted_operators]

        bars3 = axes[1, 1].barh(range(len(sorted_names)), sorted_scores, color='lightcoral', alpha=0.7)
        axes[1, 1].set_yticks(range(len(sorted_names)))
        axes[1, 1].set_yticklabels(sorted_names)
        axes[1, 1].set_title('Ranked Operator Importance')
        axes[1, 1].set_xlabel('Importance Score')

        # Add rank labels
        for i, (bar, score) in enumerate(zip(bars3, sorted_scores)):
            axes[1, 1].text(score + max(sorted_scores)*0.01, bar.get_y() + bar.get_height()/2.,
                           f'#{i+1}', ha='left', va='center', fontweight='bold')

        plt.tight_layout()
        return fig

    def _visualize_attention_entropy(self, explanation: Explanation) -> plt.Figure:
        """Visualize attention entropy in detail."""
        entropy_data = explanation.get_data('attention_entropy', {})
        attention_weights = explanation.get_data('attention_weights', np.array([]))

        if not entropy_data and len(attention_weights) == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No entropy data available',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Compute entropy distribution if we have raw weights
        if len(attention_weights) > 0:
            entropies = [self._compute_entropy(weights) for weights in attention_weights]

            # Histogram of entropies
            axes[0, 0].hist(entropies, bins=20, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 0].set_title('Distribution of Attention Entropy')
            axes[0, 0].set_xlabel('Entropy (bits)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].axvline(np.mean(entropies), color='red', linestyle='--', label=f'Mean: {np.mean(entropies):.3f}')
            axes[0, 0].legend()

            # Entropy over time
            axes[0, 1].plot(range(len(entropies)), entropies, 'g-', marker='o', markersize=3)
            axes[0, 1].set_title('Attention Entropy Over Time')
            axes[0, 1].set_xlabel('Time/Batch')
            axes[0, 1].set_ylabel('Entropy (bits)')
            axes[0, 1].grid(True, alpha=0.3)

            # Entropy vs max weight correlation
            max_weights = np.max(attention_weights, axis=1)
            axes[1, 0].scatter(max_weights, entropies, alpha=0.6)
            axes[1, 0].set_title('Entropy vs Max Attention Weight')
            axes[1, 0].set_xlabel('Max Attention Weight')
            axes[1, 0].set_ylabel('Entropy (bits)')

            # Compute correlation
            if len(max_weights) > 1:
                correlation = np.corrcoef(max_weights, entropies)[0, 1]
                axes[1, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                               transform=axes[1, 0].transAxes, bbox=dict(boxstyle='round',
                               facecolor='wheat', alpha=0.5))

        # Summary statistics
        if entropy_data:
            stats_text = "Entropy Statistics:\n\n"
            stats_text += f"Mean Entropy: {entropy_data.get('entropy', 0):.3f} bits\n"
            stats_text += f"Std Entropy: {entropy_data.get('entropy_std', 0):.3f} bits\n"

            # Add theoretical maximum entropy
            max_entropy = np.log2(len(self.operator_names))
            current_entropy = entropy_data.get('entropy', 0)
            efficiency = current_entropy / max_entropy if max_entropy > 0 else 0

            stats_text += f"Max Possible Entropy: {max_entropy:.3f} bits\n"
            stats_text += f"Entropy Efficiency: {efficiency:.1%}\n"

            # Interpretation
            if efficiency < 0.3:
                interpretation = "Highly concentrated attention"
            elif efficiency < 0.7:
                interpretation = "Balanced attention distribution"
            else:
                interpretation = "Distributed attention"

            stats_text += f"\nInterpretation: {interpretation}"

            axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                           verticalalignment='top', fontsize=11,
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            axes[1, 1].set_title('Entropy Summary')
            axes[1, 1].axis('off')

        plt.tight_layout()
        return fig