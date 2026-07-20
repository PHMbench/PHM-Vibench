"""
Path Analysis Explainer

Intrinsic explanation method that tracks signal transformations through
physical operator networks, providing path-level explanations for fault diagnosis.
"""

from typing import Dict, Any, Optional, Sequence, List, Union
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ...core import SignalData, Explanation, ExplainabilityMethod, BaseExplainerAdapter


class PathAnalysisExplainer(BaseExplainerAdapter, ExplainabilityMethod):
    """
    Path Analysis Explainer for intrinsic explanations.

    This method tracks how the input signal is transformed through each layer of
    transparent signal processing networks, providing physical interpretations of
    each transformation step. It's particularly suitable for TSPN, NNSPN, and
    other models with interpretable signal processing paths.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Path Analysis Explainer.

        Args:
            config: Configuration dictionary with the following options:
                - include_frequency_analysis: bool (default True)
                - include_energy_analysis: bool (default True)
                - include_statistical_analysis: bool (default True)
                - sampling_rate: float (default 1024.0 Hz)
                - physical_interpretations: dict (default {})
                - max_path_depth: int (default 10)
                - importance_threshold: float (default 0.1)
        """
        super().__init__(config)
        self._method_name = "PathAnalysis"
        self._method_type = "intrinsic"

        # Configuration options
        self.include_frequency_analysis = self.config.get('include_frequency_analysis', True)
        self.include_energy_analysis = self.config.get('include_energy_analysis', True)
        self.include_statistical_analysis = self.config.get('include_statistical_analysis', True)
        self.sampling_rate = self.config.get('sampling_rate', 1024.0)
        self.physical_interpretations = self.config.get('physical_interpretations', {})
        self.max_path_depth = self.config.get('max_path_depth', 10)
        self.importance_threshold = self.config.get('importance_threshold', 0.1)

    def explain(self,
                signal: SignalData,
                prediction: Any,
                model: Optional[torch.nn.Module] = None,
                **kwargs) -> Explanation:
        """
        Generate path-based explanation for the given signal.

        Args:
            signal: SignalData object containing the signal to explain
            prediction: Model prediction or target class
            model: Optional model for path extraction (if not provided, tries to extract from kwargs)
            **kwargs: Additional arguments including:
                - layer_names: List of layer names to track
                - track_gradients: bool (default False)

        Returns:
            Explanation object containing path analysis results
        """
        self._validate_signal(signal)

        # Get model if provided
        model = model or kwargs.get('model')
        if model is None:
            raise ValueError("Model must be provided for path analysis")

        # Extract signal path
        signal_path = self._extract_signal_path(signal, model, **kwargs)

        # Analyze path importance
        importance_scores = self._analyze_path_importance(signal_path)

        # Generate explanations
        explanation_data = {
            'path': signal_path,
            'importance_scores': importance_scores,
            'original_signal': signal.raw_signal,
            'method_specific': {
                'path_length': len(signal_path),
                'critical_layers': self._identify_critical_layers(importance_scores),
                'dominant_frequencies': self._extract_dominant_frequencies(signal_path) if self.include_frequency_analysis else None,
                'energy_distribution': self._analyze_energy_distribution(signal_path) if self.include_energy_analysis else None,
                'statistical_summary': self._compute_statistical_summary(signal_path) if self.include_statistical_analysis else None
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
        Create visualization for path-based explanation.

        Args:
            explanation: Explanation object to visualize
            mode: Visualization mode ('auto', 'path', 'importance', 'energy', 'frequency')
            **kwargs: Additional visualization parameters

        Returns:
            Matplotlib figure object
        """
        self._validate_explanation(explanation)

        if mode == 'auto':
            return self._visualize_path_overview(explanation)
        elif mode == 'path':
            return self._visualize_signal_path(explanation)
        elif mode == 'importance':
            return self._visualize_importance_scores(explanation)
        elif mode == 'energy':
            return self._visualize_energy_distribution(explanation)
        elif mode == 'frequency':
            return self._visualize_frequency_analysis(explanation)
        else:
            raise ValueError(f"Unknown visualization mode: {mode}")

    def evaluate(self,
                 explanations: Sequence[Explanation],
                 ground_truth: Optional[Sequence[Any]] = None,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate path-based explanations.

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

        # Path consistency metrics
        path_lengths = []
        importance_entropies = []
        critical_layer_counts = []

        for exp in explanations:
            path_data = exp.get_data('path', [])
            importance_scores = exp.get_data('importance_scores', {})

            path_lengths.append(len(path_data))
            critical_layer_counts.append(len(exp.get_data('method_specific', {}).get('critical_layers', [])))

            if importance_scores:
                # Compute entropy of importance distribution
                scores = np.array([v.get('combined_score', v) if isinstance(v, dict) else v
                                  for v in importance_scores.values()])
                if len(scores) > 0 and np.sum(scores) > 0:
                    probs = scores / np.sum(scores)
                    entropy = -np.sum(probs * np.log2(probs + 1e-10))
                    importance_entropies.append(entropy)

        # Compute metrics
        metrics['avg_path_length'] = float(np.mean(path_lengths)) if path_lengths else 0.0
        metrics['path_length_std'] = float(np.std(path_lengths)) if path_lengths else 0.0
        metrics['avg_importance_entropy'] = float(np.mean(importance_entropies)) if importance_entropies else 0.0
        metrics['avg_critical_layers'] = float(np.mean(critical_layer_counts)) if critical_layer_counts else 0.0
        metrics['explanation_coverage'] = float(np.mean([pl > 0 for pl in path_lengths])) if path_lengths else 0.0

        return metrics

    def _extract_signal_path(self,
                           signal: SignalData,
                           model: torch.nn.Module,
                           **kwargs) -> List[Dict[str, Any]]:
        """
        Extract signal transformation path through the model.

        Args:
            signal: Input signal
            model: Model to extract path from
            **kwargs: Additional parameters

        Returns:
            List of path information for each layer
        """
        signal_path = []
        layer_names = kwargs.get('layer_names', [])
        track_gradients = kwargs.get('track_gradients', False)

        # Convert signal to tensor if needed
        if isinstance(signal.raw_signal, np.ndarray):
            input_tensor = torch.FloatTensor(signal.raw_signal)
        else:
            input_tensor = signal.raw_signal

        # Add batch dimension if needed
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        elif len(input_tensor.shape) == 2:
            input_tensor = input_tensor.unsqueeze(0)  # [1, C, T]

        # Set gradient tracking if requested
        if track_gradients:
            input_tensor.requires_grad_(True)

        model.eval()
        with torch.set_grad_enabled(track_gradients):
            try:
                # Try to get intermediate features if model supports it
                if hasattr(model, 'get_intermediate_features'):
                    intermediate_features = model.get_intermediate_features(input_tensor, layer_names)
                    for layer_name, feature in intermediate_features.items():
                        signal_path.append({
                            'layer_name': layer_name,
                            'output_signal': feature.detach() if hasattr(feature, 'detach') else feature,
                            'signal_shape': list(feature.shape) if hasattr(feature, 'shape') else None,
                            'operator_type': self._infer_operator_type(layer_name),
                            'physical_meaning': self.physical_interpretations.get(layer_name, 'Unknown')
                        })
                else:
                    # Fallback: try to traverse model layers
                    signal_path = self._traverse_model_layers(model, input_tensor)

            except Exception as e:
                # If path extraction fails, create minimal path
                signal_path = [{
                    'layer_name': 'input',
                    'output_signal': input_tensor,
                    'signal_shape': list(input_tensor.shape),
                    'operator_type': 'input',
                    'physical_meaning': 'Raw input signal'
                }, {
                    'layer_name': 'output',
                    'output_signal': model(input_tensor),
                    'signal_shape': None,
                    'operator_type': 'output',
                    'physical_meaning': 'Model output'
                }]

        return signal_path

    def _traverse_model_layers(self,
                             model: torch.nn.Module,
                             input_tensor: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Traverse model layers and extract intermediate outputs.

        Args:
            model: PyTorch model
            input_tensor: Input tensor

        Returns:
            List of layer information
        """
        signal_path = []
        current_output = input_tensor

        # Add input layer
        signal_path.append({
            'layer_name': 'input',
            'output_signal': current_output,
            'signal_shape': list(current_output.shape),
            'operator_type': 'input',
            'physical_meaning': 'Raw input signal'
        })

        # Try to traverse through named modules
        for name, module in model.named_modules():
            if name == '':  # Skip the root module
                continue

            try:
                # Apply the module
                if isinstance(module, torch.nn.Module):
                    current_output = module(current_output)
                    signal_path.append({
                        'layer_name': name,
                        'output_signal': current_output.detach() if hasattr(current_output, 'detach') else current_output,
                        'signal_shape': list(current_output.shape) if hasattr(current_output, 'shape') else None,
                        'operator_type': self._infer_operator_type(name),
                        'physical_meaning': self.physical_interpretations.get(name, 'Unknown')
                    })

                    # Limit path depth to prevent memory issues
                    if len(signal_path) >= self.max_path_depth:
                        break

            except Exception:
                # Skip layers that can't be applied directly
                continue

        return signal_path

    def _analyze_path_importance(self, signal_path: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze importance of each layer in the signal path.

        Args:
            signal_path: List of layer information

        Returns:
            Dictionary of importance scores
        """
        importance_scores = {}

        for i, layer_info in enumerate(signal_path):
            layer_name = layer_info['layer_name']

            # Compute importance based on signal characteristics
            importance = 0.0
            output_signal = layer_info.get('output_signal')

            if output_signal is not None and hasattr(output_signal, 'shape'):
                # Importance based on signal magnitude
                if hasattr(output_signal, 'norm'):
                    magnitude = output_signal.norm().item()
                elif hasattr(output_signal, 'abs') and hasattr(output_signal.abs(), 'mean'):
                    magnitude = output_signal.abs().mean().item()
                else:
                    magnitude = float(np.mean(np.abs(output_signal)))

                # Normalized by position in path (later layers get higher weight)
                position_weight = (i + 1) / len(signal_path)
                importance = magnitude * position_weight

                # Additional importance based on operator type
                operator_type = layer_info.get('operator_type', '')
                if operator_type in ['FFT', 'Wavelet', 'Hilbert']:
                    importance *= 1.2  # Boost importance of signal processing operators
                elif operator_type in ['Attention', 'Convolution']:
                    importance *= 1.1  # Slightly boost feature extraction operators

            importance_scores[layer_name] = {
                'raw_importance': importance,
                'position_weight': (i + 1) / len(signal_path) if signal_path else 0.0,
                'combined_score': importance,
                'operator_type': layer_info.get('operator_type', '')
            }

        # Normalize importance scores
        if importance_scores:
            max_score = max([info['combined_score'] for info in importance_scores.values()])
            if max_score > 0:
                for layer_name in importance_scores:
                    importance_scores[layer_name]['normalized_score'] = importance_scores[layer_name]['combined_score'] / max_score

        return importance_scores

    def _identify_critical_layers(self, importance_scores: Dict[str, Any]) -> List[str]:
        """
        Identify critical layers based on importance scores.

        Args:
            importance_scores: Dictionary of importance scores

        Returns:
            List of critical layer names
        """
        critical_layers = []

        for layer_name, score_info in importance_scores.items():
            normalized_score = score_info.get('normalized_score', score_info.get('combined_score', 0))
            if normalized_score > self.importance_threshold:
                critical_layers.append(layer_name)

        return critical_layers

    def _extract_dominant_frequencies(self, signal_path: List[Dict[str, Any]]) -> Optional[Dict[str, List[float]]]:
        """
        Extract dominant frequencies from signal path.

        Args:
            signal_path: List of layer information

        Returns:
            Dictionary mapping layer names to dominant frequencies
        """
        if not self.include_frequency_analysis:
            return None

        dominant_freqs = {}

        for layer_info in signal_path:
            layer_name = layer_info['layer_name']
            output_signal = layer_info.get('output_signal')

            if output_signal is not None and hasattr(output_signal, 'detach'):
                signal_data = output_signal.detach().cpu().numpy().flatten()

                # Simple FFT to find dominant frequencies
                if len(signal_data) > 0:
                    try:
                        fft = np.fft.fft(signal_data)
                        freqs = np.fft.fftfreq(len(signal_data), 1/self.sampling_rate)
                        magnitude = np.abs(fft)

                        # Get top 5 dominant frequencies
                        top_indices = np.argsort(magnitude)[-5:][::-1]
                        dominant_freqs[layer_name] = [float(freqs[i]) for i in top_indices if freqs[i] >= 0]
                    except Exception:
                        dominant_freqs[layer_name] = []

        return dominant_freqs

    def _analyze_energy_distribution(self, signal_path: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        """
        Analyze energy distribution across the signal path.

        Args:
            signal_path: List of layer information

        Returns:
            Dictionary mapping layer names to energy values
        """
        if not self.include_energy_analysis:
            return None

        energy_distribution = {}

        for layer_info in signal_path:
            layer_name = layer_info['layer_name']
            output_signal = layer_info.get('output_signal')

            if output_signal is not None:
                if hasattr(output_signal, 'norm'):
                    energy = output_signal.norm().item() ** 2
                elif hasattr(output_signal, 'detach'):
                    signal_data = output_signal.detach().cpu().numpy()
                    energy = float(np.sum(signal_data ** 2))
                else:
                    energy = float(np.sum(np.array(output_signal) ** 2))

                energy_distribution[layer_name] = energy

        return energy_distribution

    def _compute_statistical_summary(self, signal_path: List[Dict[str, Any]]) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Compute statistical summary of signals across the path.

        Args:
            signal_path: List of layer information

        Returns:
            Dictionary of statistical summaries
        """
        if not self.include_statistical_analysis:
            return None

        stats = {}

        for layer_info in signal_path:
            layer_name = layer_info['layer_name']
            output_signal = layer_info.get('output_signal')

            if output_signal is not None:
                if hasattr(output_signal, 'detach'):
                    signal_data = output_signal.detach().cpu().numpy().flatten()
                else:
                    signal_data = np.array(output_signal).flatten()

                if len(signal_data) > 0:
                    stats[layer_name] = {
                        'mean': float(np.mean(signal_data)),
                        'std': float(np.std(signal_data)),
                        'min': float(np.min(signal_data)),
                        'max': float(np.max(signal_data)),
                        'rms': float(np.sqrt(np.mean(signal_data ** 2))),
                        'peak_to_peak': float(np.ptp(signal_data))
                    }

        return stats

    def _infer_operator_type(self, layer_name: str) -> str:
        """
        Infer operator type from layer name.

        Args:
            layer_name: Name of the layer

        Returns:
            Inferred operator type
        """
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
        elif 'norm' in layer_name_lower:
            return 'Normalization'
        elif 'lno' in layer_name_lower:
            return 'LNO'
        else:
            return 'Unknown'

    # Visualization methods
    def _visualize_path_overview(self, explanation: Explanation) -> plt.Figure:
        """Create overview visualization of signal path."""
        path_data = explanation.get_data('path', [])
        importance_scores = explanation.get_data('importance_scores', {})

        n_layers = len(path_data)
        if n_layers == 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'No path data available', ha='center', va='center', transform=ax.transAxes)
            return fig

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Layer importance bar chart
        if importance_scores:
            layer_names = list(importance_scores.keys())
            scores = [info.get('combined_score', 0) for info in importance_scores.values()]

            bars = axes[0].bar(range(len(layer_names)), scores)
            axes[0].set_title('Layer Importance Scores')
            axes[0].set_xlabel('Layer')
            axes[0].set_ylabel('Importance Score')
            axes[0].set_xticks(range(len(layer_names)))
            axes[0].set_xticklabels(layer_names, rotation=45, ha='right')

            # Add value labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{score:.3f}', ha='center', va='bottom')
        else:
            axes[0].text(0.5, 0.5, 'No importance scores available', ha='center', va='center', transform=axes[0].transAxes)

        # Signal path flow
        axes[1].text(0.5, 0.5, f'Path contains {n_layers} layers', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Signal Path Overview')
        axes[1].axis('off')

        # Method-specific information
        method_specific = explanation.get_data('method_specific', {})
        info_text = f"Critical layers: {len(method_specific.get('critical_layers', []))}\n"
        info_text += f"Path length: {method_specific.get('path_length', 0)}"
        axes[2].text(0.1, 0.5, info_text, transform=axes[2].transAxes, verticalalignment='center')
        axes[2].set_title('Analysis Summary')
        axes[2].axis('off')

        plt.tight_layout()
        return fig

    def _visualize_signal_path(self, explanation: Explanation) -> plt.Figure:
        """Visualize signal transformations through the path."""
        return explanation.visualize(mode='path')  # Reuse existing visualization

    def _visualize_importance_scores(self, explanation: Explanation) -> plt.Figure:
        """Visualize importance scores."""
        return explanation.visualize(mode='importance')  # Reuse existing visualization

    def _visualize_energy_distribution(self, explanation: Explanation) -> plt.Figure:
        """Visualize energy distribution across layers."""
        energy_data = explanation.get_data('method_specific', {}).get('energy_distribution', {})

        if not energy_data:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'No energy data available', ha='center', va='center', transform=ax.transAxes)
            return fig

        fig, ax = plt.subplots(figsize=(10, 6))
        layer_names = list(energy_data.keys())
        energies = list(energy_data.values())

        bars = ax.bar(layer_names, energies)
        ax.set_title('Energy Distribution Across Layers')
        ax.set_ylabel('Energy')
        ax.set_xlabel('Layer')
        plt.xticks(rotation=45, ha='right')

        # Add value labels on bars
        for bar, energy in zip(bars, energies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{energy:.1e}', ha='center', va='bottom')

        plt.tight_layout()
        return fig

    def _visualize_frequency_analysis(self, explanation: Explanation) -> plt.Figure:
        """Visualize frequency analysis results."""
        freq_data = explanation.get_data('method_specific', {}).get('dominant_frequencies', {})

        if not freq_data:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'No frequency data available', ha='center', va='center', transform=ax.transAxes)
            return fig

        fig, ax = plt.subplots(figsize=(12, 6))

        for layer_name, freqs in freq_data.items():
            if freqs:  # Only plot if we have frequency data
                ax.scatter([layer_name] * len(freqs), freqs, label=layer_name, alpha=0.7, s=50)

        ax.set_title('Dominant Frequencies Across Layers')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Layer')
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig