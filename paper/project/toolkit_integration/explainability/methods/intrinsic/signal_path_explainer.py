"""
Signal Path Explainer

Provides intrinsic explanations by tracking signal transformations through
physical operator networks. This explainer focuses on the physical meaning
of each transformation step.
"""

from typing import Dict, Any, Optional, List
import torch
import numpy as np
from ...core.base_explainer import BaseExplainer
from ...core.explanation import Explanation


class SignalPathExplainer(BaseExplainer):
    """
    Signal Path Explainer for physical operator networks.

    This explainer tracks how the input signal is transformed through each
    layer of the model, providing physical interpretations of each step.
    It's particularly useful for models like TSPN that use physical signal
    processing operators.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(model, config)

        # Configuration options
        self.include_frequency_analysis = self.config.get('include_frequency_analysis', True)
        self.include_energy_analysis = self.config.get('include_energy_analysis', True)
        self.sampling_rate = self.config.get('sampling_rate', 1024.0)  # Hz
        self.physical_interpretations = self.config.get('physical_interpretations', {})

    def explain(self,
                input_data: torch.Tensor,
                target_class: Optional[int] = None,
                **kwargs) -> Explanation:
        """
        Generate signal path explanation.

        Args:
            input_data: Input tensor [batch_size, sequence_length, channels]
            target_class: Target class (not used for signal path but kept for consistency)
            **kwargs: Additional arguments

        Returns:
            Explanation object containing signal path information
        """
        self._validate_input(input_data)

        # Get signal path from model
        signal_path = self._get_signal_path(input_data)

        # Analyze physical meaning of each transformation
        physical_analysis = self._analyze_physical_transformations(signal_path)

        # Create explanation data
        explanation_data = {
            'signal_path': signal_path,
            'physical_analysis': physical_analysis,
            'original_signal': input_data,
            'transformation_summary': self._create_transformation_summary(signal_path)
        }

        # Create metadata
        metadata = {
            'method': 'signal_path',
            'model_name': type(self.model).__name__,
            'input_shape': list(input_data.shape),
            'num_transformations': len(signal_path),
            'analysis_type': 'intrinsic_physical'
        }

        return Explanation(explanation_data, metadata)

    def _get_signal_path(self, input_data: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Get signal transformation path from the model.

        This method tries to get the signal path from explainable models.
        If the model doesn't support it directly, it uses intermediate outputs.
        """
        # First, try to get signal path directly from model
        if hasattr(self.model, 'get_signal_path'):
            try:
                return self.model.get_signal_path(input_data)
            except Exception as e:
                print(f"Warning: Could not get signal path directly: {e}")

        # Fallback: use intermediate outputs and layer analysis
        return self._extract_signal_path_from_layers(input_data)

    def _extract_signal_path_from_layers(self, input_data: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Extract signal path by analyzing layer-by-layer forward pass.
        """
        signal_path = []
        current_signal = input_data.clone().detach()

        # Get all modules with parameters
        modules_with_params = []
        for name, module in self.model.named_modules():
            if name and any(p.requires_grad for p in module.parameters()):
                modules_with_params.append((name, module))

        # Process each module
        for i, (layer_name, layer) in enumerate(modules_with_params):
            # Store input signal stats
            input_stats = self._compute_signal_stats(current_signal)
            input_freq_analysis = self._analyze_frequency_content(current_signal) if self.include_frequency_analysis else {}

            # Forward pass through this layer
            with torch.no_grad():
                layer_output = layer(current_signal)

            # Store output signal stats
            output_stats = self._compute_signal_stats(layer_output)
            output_freq_analysis = self._analyze_frequency_content(layer_output) if self.include_frequency_analysis else {}

            # Determine operator type based on layer characteristics
            operator_type = self._identify_operator_type(layer, current_signal, layer_output)

            # Create layer information
            layer_info = {
                'layer_index': i,
                'layer_name': layer_name,
                'layer_type': type(layer).__name__,
                'operator_type': operator_type,
                'input_stats': input_stats,
                'output_stats': output_stats,
                'frequency_analysis': {
                    'input': input_freq_analysis,
                    'output': output_freq_analysis
                } if self.include_frequency_analysis else {},
                'parameters': {name: param.shape for name, param in layer.named_parameters()},
                'input_signal': current_signal.clone(),
                'output_signal': layer_output.clone()
            }

            # Add physical interpretation if available
            if operator_type in self.physical_interpretations:
                layer_info['physical_meaning'] = self.physical_interpretations[operator_type]

            signal_path.append(layer_info)
            current_signal = layer_output

        return signal_path

    def _identify_operator_type(self, layer, input_signal: torch.Tensor, output_signal: torch.Tensor) -> str:
        """
        Identify the physical operator type based on layer characteristics and signal changes.
        """
        layer_type = type(layer).__name__.lower()

        # Direct identification based on layer type
        if 'conv' in layer_type or 'linear' in layer_type:
            # Analyze frequency response to distinguish between different operators
            input_freq = self._analyze_frequency_content(input_signal)
            output_freq = self._analyze_frequency_content(output_signal)

            # Check if it's acting as a filter
            if self.include_frequency_analysis and input_freq and output_freq:
                input_power = input_freq.get('total_power', 0)
                output_power = output_freq.get('total_power', 0)
                freq_change = abs(output_power - input_power) / (input_power + 1e-8)

                if freq_change > 0.5:
                    return 'frequency_filter'
                elif output_freq.get('dominant_frequency', 0) != input_freq.get('dominant_frequency', 0):
                    return 'frequency_transform'
                else:
                    return 'amplitude_transform'
            else:
                return 'linear_transform'

        elif 'pool' in layer_type or 'downsample' in layer_type:
            return 'downsampling'

        elif 'batchnorm' in layer_type or 'layernorm' in layer_type:
            return 'normalization'

        elif 'activation' in layer_type or 'relu' in layer_type or 'sigmoid' in layer_type or 'tanh' in layer_type:
            return 'nonlinear_activation'

        elif 'lstm' in layer_type or 'rnn' in layer_type or 'gru' in layer_type:
            return 'temporal_memory'

        elif 'attention' in layer_type:
            return 'attention_mechanism'

        else:
            return 'unknown_operator'

    def _analyze_physical_transformations(self, signal_path: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the physical meaning of the signal transformations.
        """
        analysis = {
            'energy_flow': [],
            'frequency_evolution': [],
            'dominant_transformations': [],
            'physical_consistency': {}
        }

        if not signal_path:
            return analysis

        # Analyze energy flow
        if self.include_energy_analysis:
            for layer_info in signal_path:
                input_energy = layer_info['input_stats'].get('energy', 0)
                output_energy = layer_info['output_stats'].get('energy', 0)
                energy_change = (output_energy - input_energy) / (input_energy + 1e-8)

                analysis['energy_flow'].append({
                    'layer_name': layer_info['layer_name'],
                    'operator_type': layer_info['operator_type'],
                    'energy_change_ratio': energy_change,
                    'energy_preserved': output_energy / (input_energy + 1e-8)
                })

        # Analyze frequency evolution
        if self.include_frequency_analysis:
            for layer_info in signal_path:
                freq_analysis = layer_info.get('frequency_analysis', {})
                input_freq = freq_analysis.get('input', {})
                output_freq = freq_analysis.get('output', {})

                if input_freq and output_freq:
                    freq_shift = output_freq.get('dominant_frequency', 0) - input_freq.get('dominant_frequency', 0)
                    centroid_change = output_freq.get('spectral_centroid', 0) - input_freq.get('spectral_centroid', 0)

                    analysis['frequency_evolution'].append({
                        'layer_name': layer_info['layer_name'],
                        'operator_type': layer_info['operator_type'],
                        'frequency_shift': freq_shift,
                        'centroid_change': centroid_change,
                        'power_change': output_freq.get('total_power', 0) - input_freq.get('total_power', 0)
                    })

        # Identify dominant transformations
        operator_counts = {}
        for layer_info in signal_path:
            op_type = layer_info['operator_type']
            operator_counts[op_type] = operator_counts.get(op_type, 0) + 1

        # Sort by frequency and get top transformations
        sorted_ops = sorted(operator_counts.items(), key=lambda x: x[1], reverse=True)
        analysis['dominant_transformations'] = sorted_ops[:3]  # Top 3 transformations

        return analysis

    def _create_transformation_summary(self, signal_path: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a summary of all transformations.
        """
        if not signal_path:
            return {}

        # Compute overall transformation statistics
        initial_stats = signal_path[0]['input_stats']
        final_stats = signal_path[-1]['output_stats']

        overall_transformations = {
            'total_layers': len(signal_path),
            'initial_signal_stats': initial_stats,
            'final_signal_stats': final_stats,
            'overall_energy_change': (final_stats.get('energy', 0) - initial_stats.get('energy', 0)) / (initial_stats.get('energy', 0) + 1e-8),
            'overall_rms_change': (final_stats.get('rms', 0) - initial_stats.get('rms', 0)) / (initial_stats.get('rms', 0) + 1e-8),
        }

        # Add frequency summary if available
        if self.include_frequency_analysis and 'frequency_analysis' in signal_path[-1]:
            final_freq = signal_path[-1]['frequency_analysis'].get('output', {})
            initial_freq = signal_path[0]['frequency_analysis'].get('input', {})

            if final_freq and initial_freq:
                overall_transformations['frequency_summary'] = {
                    'initial_dominant_freq': initial_freq.get('dominant_frequency', 0),
                    'final_dominant_freq': final_freq.get('dominant_frequency', 0),
                    'frequency_shift': final_freq.get('dominant_frequency', 0) - initial_freq.get('dominant_frequency', 0),
                    'centroid_shift': final_freq.get('spectral_centroid', 0) - initial_freq.get('spectral_centroid', 0)
                }

        return overall_transformations

    def _compute_signal_stats(self, signal: torch.Tensor) -> Dict[str, float]:
        """Compute basic signal statistics."""
        if isinstance(signal, torch.Tensor):
            signal_np = signal.detach().cpu().numpy()
        else:
            signal_np = np.array(signal)

        return {
            'mean': float(np.mean(signal_np)),
            'std': float(np.std(signal_np)),
            'rms': float(np.sqrt(np.mean(signal_np ** 2))),
            'max': float(np.max(signal_np)),
            'min': float(np.min(signal_np)),
            'energy': float(np.sum(signal_np ** 2))
        }

    def _analyze_frequency_content(self, signal: torch.Tensor) -> Dict[str, float]:
        """Analyze frequency content of the signal."""
        if isinstance(signal, torch.Tensor):
            signal_np = signal.detach().cpu().numpy().flatten()
        else:
            signal_np = np.array(signal).flatten()

        # Compute FFT
        fft_vals = np.fft.fft(signal_np)
        fft_freq = np.fft.fftfreq(len(signal_np), 1/self.sampling_rate)

        # Only keep positive frequencies
        pos_mask = fft_freq > 0
        pos_freq = fft_freq[pos_mask]
        pos_fft = np.abs(fft_vals[pos_mask])

        if len(pos_fft) > 0:
            dominant_freq_idx = np.argmax(pos_fft)
            dominant_freq = pos_freq[dominant_freq_idx]
            dominant_power = pos_fft[dominant_freq_idx]
            spectral_centroid = np.sum(pos_freq * pos_fft) / (np.sum(pos_fft) + 1e-8)
            total_power = np.sum(pos_fft)
        else:
            dominant_freq = 0.0
            dominant_power = 0.0
            spectral_centroid = 0.0
            total_power = 0.0

        return {
            'dominant_frequency': dominant_freq,
            'dominant_power': float(dominant_power),
            'spectral_centroid': float(spectral_centroid),
            'total_power': float(total_power)
        }