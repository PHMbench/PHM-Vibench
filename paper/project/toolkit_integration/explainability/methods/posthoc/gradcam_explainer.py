"""
Grad-CAM Explainer for Signal Processing Networks

Post-hoc explanation method that implements Gradient-weighted Class Activation Mapping
for fault diagnosis models. This method generates heatmaps showing which regions of the
input signal are most important for the model's prediction.
"""

from typing import Dict, Any, Optional, Sequence, List, Union, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ...core import SignalData, Explanation, ExplainabilityMethod, BaseExplainerAdapter


class GradCAMExplainer(BaseExplainerAdapter, ExplainabilityMethod):
    """
    Grad-CAM Explainer for post-hoc explanations.

    This method implements Gradient-weighted Class Activation Mapping for
    signal processing networks, generating attribution maps that highlight
    important regions in the input signal for the model's prediction.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Grad-CAM Explainer.

        Args:
            config: Configuration dictionary with the following options:
                - target_layers: List of layer names to use for Grad-CAM (default: auto-detect)
                - use_abs_gradients: bool (default True)
                - normalize_attributions: bool (default True)
                - interpolation_method: str ('linear', 'nearest', 'bilinear', default 'linear')
                - attribution_smoothing: bool (default True)
                - smoothing_kernel: int (default 5)
        """
        super().__init__(config)
        self._method_name = "GradCAM"
        self._method_type = "posthoc"

        # Configuration options
        self.target_layers = self.config.get('target_layers', [])
        self.use_abs_gradients = self.config.get('use_abs_gradients', True)
        self.normalize_attributions = self.config.get('normalize_attributions', True)
        self.interpolation_method = self.config.get('interpolation_method', 'linear')
        self.attribution_smoothing = self.config.get('attribution_smoothing', True)
        self.smoothing_kernel = self.config.get('smoothing_kernel', 5)

        # Internal state
        self._model = None
        self._hooks = []
        self._activations = {}
        self._gradients = {}

    def explain(self,
                signal: SignalData,
                prediction: Any,
                model: Optional[torch.nn.Module] = None,
                target_class: Optional[int] = None,
                **kwargs) -> Explanation:
        """
        Generate Grad-CAM explanation for the given signal.

        Args:
            signal: SignalData object containing the signal to explain
            prediction: Model prediction or target class
            model: Model to explain (must be provided)
            target_class: Target class for explanation (uses prediction if None)
            **kwargs: Additional arguments including:
                - layer_names: List of specific layer names to use
                - return_raw_heatmap: bool (default False)

        Returns:
            Explanation object containing Grad-CAM attribution results
        """
        self._validate_signal(signal)

        # Get model
        model = model or kwargs.get('model')
        if model is None:
            raise ValueError("Model must be provided for Grad-CAM explanation")

        self._model = model
        target_layers = kwargs.get('layer_names', self.target_layers)

        # Convert signal to tensor
        input_tensor = self._prepare_input(signal)
        if len(input_tensor.shape) == 2:
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

        # Get target class
        if target_class is None:
            target_class = self._get_prediction_class(model, input_tensor, prediction)

        # Generate Grad-CAM attribution
        attribution_map = self._generate_gradcam(input_tensor, target_class, target_layers)

        # Process attribution map
        processed_attribution = self._process_attribution(attribution_map, signal)

        # Generate explanation
        explanation_data = {
            'attributions': processed_attribution,
            'original_signal': signal.raw_signal,
            'target_class': target_class,
            'method_specific': {
                'attribution_statistics': self._compute_attribution_stats(processed_attribution),
                'important_regions': self._identify_important_regions(processed_attribution),
                'attribution_coverage': self._compute_attribution_coverage(processed_attribution),
                'peak_attribution_indices': self._find_peak_attributions(processed_attribution),
                'layer_contributions': self._analyze_layer_contributions()
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
            'target_class': target_class,
            'config': self.get_config()
        }

        # Clean up hooks
        self._cleanup_hooks()

        return Explanation(explanation_data, explanation_meta)

    def visualize(self,
                  explanation: Explanation,
                  mode: str = 'auto',
                  **kwargs) -> plt.Figure:
        """
        Create visualization for Grad-CAM explanation.

        Args:
            explanation: Explanation object to visualize
            mode: Visualization mode ('auto', 'heatmap', 'overlay', 'importance')
            **kwargs: Additional visualization parameters

        Returns:
            Matplotlib figure object
        """
        self._validate_explanation(explanation)

        if mode == 'auto':
            return self._visualize_gradcam_overview(explanation)
        elif mode == 'heatmap':
            return self._visualize_attribution_heatmap(explanation)
        elif mode == 'overlay':
            return self._visualize_signal_overlay(explanation)
        elif mode == 'importance':
            return self._visualize_importance_profile(explanation)
        else:
            raise ValueError(f"Unknown visualization mode: {mode}")

    def evaluate(self,
                 explanations: Sequence[Explanation],
                 ground_truth: Optional[Sequence[Any]] = None,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate Grad-CAM explanations.

        Args:
            explanations: Sequence of explanation objects to evaluate
            ground_truth: Optional ground truth explanations
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}

        if not explanations:
            return metrics

        # Attribution-based metrics
        attribution_entropies = []
        coverage_ratios = []
        peak_counts = []
        attribution_magnitudes = []

        for exp in explanations:
            attributions = exp.get_data('attributions')
            method_specific = exp.get_data('method_specific', {})

            if attributions is not None:
                # Compute attribution entropy
                attr_array = np.array(attributions).flatten()
                if np.sum(np.abs(attr_array)) > 0:
                    # Normalize for entropy computation
                    normalized_attr = np.abs(attr_array) / np.sum(np.abs(attr_array))
                    entropy = -np.sum(normalized_attr * np.log2(normalized_attr + 1e-10))
                    attribution_entropies.append(entropy)

                # Attribution magnitude statistics
                attribution_magnitudes.append(float(np.mean(np.abs(attr_array))))

            # Coverage and peak analysis
            coverage_ratios.append(method_specific.get('attribution_coverage', 0.0))
            peak_indices = method_specific.get('peak_attribution_indices', [])
            peak_counts.append(len(peak_indices))

        # Compute metrics
        metrics['avg_attribution_entropy'] = float(np.mean(attribution_entropies)) if attribution_entropies else 0.0
        metrics['avg_attribution_magnitude'] = float(np.mean(attribution_magnitudes)) if attribution_magnitudes else 0.0
        metrics['avg_coverage_ratio'] = float(np.mean(coverage_ratios)) if coverage_ratios else 0.0
        metrics['avg_peak_count'] = float(np.mean(peak_counts)) if peak_counts else 0.0
        metrics['explanation_sparsity'] = float(1.0 - metrics['avg_attribution_entropy'] / 10.0) if metrics['avg_attribution_entropy'] > 0 else 0.0

        return metrics

    def _prepare_input(self, signal: SignalData) -> torch.Tensor:
        """Prepare signal data for model input."""
        if isinstance(signal.raw_signal, np.ndarray):
            input_tensor = torch.FloatTensor(signal.raw_signal)
        else:
            input_tensor = signal.raw_signal

        # Ensure correct shape [batch, channels, sequence_length]
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        elif len(input_tensor.shape) == 2:
            # Could be [channels, sequence] or [batch, sequence]
            if input_tensor.shape[0] == signal.get_num_channels():
                input_tensor = input_tensor.unsqueeze(0)
            else:
                input_tensor = input_tensor.unsqueeze(1)

        return input_tensor

    def _get_prediction_class(self, model: torch.nn.Module, input_tensor: torch.Tensor, prediction: Any) -> int:
        """Get the target class for explanation."""
        if isinstance(prediction, (int, np.integer)):
            return int(prediction)
        elif isinstance(prediction, torch.Tensor):
            return int(torch.argmax(prediction).item())
        else:
            # Get model prediction
            with torch.no_grad():
                model_output = model(input_tensor)
                if hasattr(model_output, 'logits'):
                    model_output = model_output.logits
                return int(torch.argmax(model_output, dim=-1).item())

    def _generate_gradcam(self,
                         input_tensor: torch.Tensor,
                         target_class: int,
                         target_layers: Optional[List[str]]) -> np.ndarray:
        """Generate Grad-CAM attribution map."""
        # Register hooks for target layers
        self._register_hooks(target_layers)

        # Forward pass
        model_output = self._model(input_tensor)
        if hasattr(model_output, 'logits'):
            model_output = model_output.logits

        # Backward pass for target class
        self._model.zero_grad()
        target_score = model_output[:, target_class] if model_output.dim() > 1 else model_output[target_class]
        target_score.backward(retain_graph=True)

        # Generate Grad-CAM
        gradcam_map = self._compute_gradcam_map()

        return gradcam_map

    def _register_hooks(self, target_layers: Optional[List[str]]) -> None:
        """Register forward and backward hooks for target layers."""
        self._cleanup_hooks()  # Clean up any existing hooks

        def forward_hook(name):
            def hook_fn(module, input, output):
                self._activations[name] = output.detach()
            return hook_fn

        def backward_hook(name):
            def hook_fn(module, grad_input, grad_output):
                self._gradients[name] = grad_output[0].detach()
            return hook_fn

        # Auto-detect target layers if not specified
        if not target_layers:
            target_layers = self._auto_detect_target_layers()

        # Register hooks for each target layer
        for name, module in self._model.named_modules():
            if name == '' or name not in target_layers:
                continue

            # Register forward hook
            forward_hook_handle = module.register_forward_hook(forward_hook(name))
            self._hooks.append(forward_hook_handle)

            # Register backward hook
            backward_hook_handle = module.register_backward_hook(backward_hook(name))
            self._hooks.append(backward_hook_handle)

    def _auto_detect_target_layers(self) -> List[str]:
        """Automatically detect suitable target layers for Grad-CAM."""
        target_layers = []

        for name, module in self._model.named_modules():
            if name == '':
                continue

            # Look for convolutional or linear layers
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
                # Prefer layers that are not the final layer
                if not any(keyword in name.lower() for keyword in ['classifier', 'fc_final', 'output']):
                    target_layers.append(name)

        # Return the last few suitable layers
        return target_layers[-3:] if len(target_layers) > 3 else target_layers

    def _compute_gradcam_map(self) -> np.ndarray:
        """Compute Grad-CAM map from activations and gradients."""
        if not self._activations or not self._gradients:
            raise ValueError("No activations or gradients captured. Check hook registration.")

        gradcam_maps = []

        for layer_name in self._activations:
            if layer_name not in self._gradients:
                continue

            activations = self._activations[layer_name]
            gradients = self._gradients[layer_name]

            # Compute weights as mean of gradients
            if gradients.dim() == 4:  # 2D conv: [batch, channels, height, width]
                weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
                gradcam = torch.sum(weights * activations, dim=1).squeeze(1)
            elif gradients.dim() == 3:  # 1D conv: [batch, channels, sequence]
                weights = torch.mean(gradients, dim=2, keepdim=True)
                gradcam = torch.sum(weights * activations, dim=1).squeeze(1)
            else:  # Linear: [batch, features]
                weights = torch.mean(gradients, dim=1, keepdim=True)
                gradcam = torch.sum(weights * activations, dim=1).squeeze(1)

            # Apply ReLU to focus on positive influences
            gradcam = F.relu(gradcam)

            # Take absolute value if configured
            if self.use_abs_gradients:
                gradcam = torch.abs(gradcam)

            gradcam_maps.append(gradcam)

        if not gradcam_maps:
            raise ValueError("No valid Grad-CAM maps computed")

        # Combine maps (average for now)
        combined_map = torch.mean(torch.stack(gradcam_maps), dim=0)

        return combined_map.detach().cpu().numpy()

    def _process_attribution(self, attribution_map: np.ndarray, signal: SignalData) -> np.ndarray:
        """Process and normalize attribution map."""
        if attribution_map.ndim == 1:
            attribution_map = attribution_map.reshape(1, -1)

        # Interpolate to match signal length if needed
        if attribution_map.shape[-1] != signal.get_length():
            attribution_map = self._interpolate_attribution(attribution_map, signal.get_length())

        # Normalize if configured
        if self.normalize_attributions:
            attribution_map = self._normalize_attribution(attribution_map)

        # Apply smoothing if configured
        if self.attribution_smoothing:
            attribution_map = self._smooth_attribution(attribution_map)

        return attribution_map

    def _interpolate_attribution(self, attribution_map: np.ndarray, target_length: int) -> np.ndarray:
        """Interpolate attribution map to target length."""
        if attribution_map.shape[-1] == target_length:
            return attribution_map

        # Convert to tensor for interpolation
        attr_tensor = torch.FloatTensor(attribution_map).unsqueeze(0)  # Add batch dimension

        # Reshape for 1D interpolation
        if attr_tensor.dim() == 3:
            attr_tensor = attr_tensor.permute(0, 2, 1)  # [batch, sequence, channels]

        # Interpolate
        if self.interpolation_method == 'linear':
            interpolated = F.interpolate(
                attr_tensor.unsqueeze(-1),  # Add last dimension for 1D
                size=(target_length, 1),
                mode='linear',
                align_corners=False
            ).squeeze(-1)
        elif self.interpolation_method == 'nearest':
            interpolated = F.interpolate(
                attr_tensor.unsqueeze(-1),
                size=(target_length, 1),
                mode='nearest'
            ).squeeze(-1)
        else:
            # Default to linear
            interpolated = F.interpolate(
                attr_tensor.unsqueeze(-1),
                size=(target_length, 1),
                mode='linear',
                align_corners=False
            ).squeeze(-1)

        # Convert back to original shape
        if interpolated.dim() == 3:
            interpolated = interpolated.permute(0, 2, 1)  # [batch, channels, sequence]

        return interpolated.squeeze(0).numpy()  # Remove batch dimension

    def _normalize_attribution(self, attribution_map: np.ndarray) -> np.ndarray:
        """Normalize attribution map to [0, 1] range."""
        # Flatten for statistics
        flat_attr = attribution_map.flatten()

        if np.max(np.abs(flat_attr)) > 0:
            # Min-max normalization to [0, 1]
            attr_min, attr_max = np.min(flat_attr), np.max(flat_attr)
            if attr_max > attr_min:
                normalized = (attribution_map - attr_min) / (attr_max - attr_min)
            else:
                normalized = np.zeros_like(attribution_map)
        else:
            normalized = attribution_map

        return normalized

    def _smooth_attribution(self, attribution_map: np.ndarray) -> np.ndarray:
        """Apply smoothing to attribution map."""
        # Simple moving average smoothing
        kernel_size = min(self.smoothing_kernel, attribution_map.shape[-1] // 4)
        if kernel_size < 3:
            return attribution_map

        smoothed = np.zeros_like(attribution_map)
        for i in range(attribution_map.shape[0]):  # For each channel
            for j in range(attribution_map.shape[1]):  # For each time point
                start_idx = max(0, j - kernel_size // 2)
                end_idx = min(attribution_map.shape[1], j + kernel_size // 2 + 1)
                smoothed[i, j] = np.mean(attribution_map[i, start_idx:end_idx])

        return smoothed

    def _compute_attribution_stats(self, attribution: np.ndarray) -> Dict[str, float]:
        """Compute statistics of attribution map."""
        flat_attr = attribution.flatten()

        return {
            'mean': float(np.mean(flat_attr)),
            'std': float(np.std(flat_attr)),
            'min': float(np.min(flat_attr)),
            'max': float(np.max(flat_attr)),
            'median': float(np.median(flat_attr)),
            'positive_ratio': float(np.mean(flat_attr > 0)),
            'zero_ratio': float(np.mean(flat_attr == 0)),
            'magnitude_mean': float(np.mean(np.abs(flat_attr)))
        }

    def _identify_important_regions(self, attribution: np.ndarray, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Identify important regions in the attribution map."""
        if attribution.ndim == 1:
            attribution = attribution.reshape(1, -1)

        important_regions = []
        threshold_value = np.max(attribution) * threshold

        for channel_idx in range(attribution.shape[0]):
            channel_attr = attribution[channel_idx]
            important_indices = np.where(channel_attr > threshold_value)[0]

            if len(important_indices) > 0:
                # Group consecutive indices into regions
                regions = []
                start_idx = important_indices[0]
                prev_idx = important_indices[0]

                for idx in important_indices[1:]:
                    if idx != prev_idx + 1:
                        regions.append((start_idx, prev_idx))
                        start_idx = idx
                    prev_idx = idx
                regions.append((start_idx, prev_idx))

                important_regions.append({
                    'channel': channel_idx,
                    'regions': regions,
                    'peak_value': float(np.max(channel_attr)),
                    'coverage': float(len(important_indices) / len(channel_attr))
                })

        return important_regions

    def _compute_attribution_coverage(self, attribution: np.ndarray, threshold: float = 0.1) -> float:
        """Compute coverage ratio of attribution above threshold."""
        if attribution.ndim == 1:
            attribution = attribution.reshape(1, -1)

        threshold_value = np.max(attribution) * threshold
        important_pixels = np.sum(np.abs(attribution) > threshold_value)
        total_pixels = attribution.size

        return float(important_pixels / total_pixels) if total_pixels > 0 else 0.0

    def _find_peak_attributions(self, attribution: np.ndarray, n_peaks: int = 5) -> List[Dict[str, Any]]:
        """Find peak attribution locations."""
        if attribution.ndim == 1:
            attribution = attribution.reshape(1, -1)

        peaks = []

        for channel_idx in range(attribution.shape[0]):
            channel_attr = attribution[channel_idx]

            # Find peaks using simple threshold-based approach
            threshold = np.mean(channel_attr) + 2 * np.std(channel_attr)
            peak_indices = np.where(channel_attr > threshold)[0]

            # Sort by value and take top n_peaks
            if len(peak_indices) > 0:
                peak_values = channel_attr[peak_indices]
                sorted_indices = np.argsort(peak_values)[-n_peaks:][::-1]

                for idx in sorted_indices:
                    peaks.append({
                        'channel': channel_idx,
                        'index': int(peak_indices[idx]),
                        'value': float(peak_values[idx]),
                        'time': float(peak_indices[idx] / 1024.0)  # Assuming 1024 Hz sampling
                    })

        # Sort all peaks by value and return top n_peaks
        peaks.sort(key=lambda x: x['value'], reverse=True)
        return peaks[:n_peaks]

    def _analyze_layer_contributions(self) -> Dict[str, float]:
        """Analyze contribution of each layer to the final Grad-CAM map."""
        contributions = {}

        for layer_name in self._activations:
            if layer_name in self._gradients:
                activation = self._activations[layer_name]
                gradient = self._gradients[layer_name]

                # Simple contribution metric: mean of absolute gradients
                if hasattr(gradient, 'abs'):
                    contribution = float(torch.mean(torch.abs(gradient)))
                else:
                    contribution = float(np.mean(np.abs(gradient)))

                contributions[layer_name] = contribution

        return contributions

    def _cleanup_hooks(self) -> None:
        """Clean up registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._activations = {}
        self._gradients = {}

    # Visualization methods
    def _visualize_gradcam_overview(self, explanation: Explanation) -> plt.Figure:
        """Create overview visualization of Grad-CAM results."""
        attributions = explanation.get_data('attributions')
        method_specific = explanation.get_data('method_specific', {})

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Attribution heatmap
        if attributions is not None:
            if attributions.ndim == 1:
                attributions = attributions.reshape(1, -1)

            im = axes[0, 0].imshow(attributions, aspect='auto', cmap='hot', interpolation='bilinear')
            axes[0, 0].set_title('Grad-CAM Attribution Heatmap')
            axes[0, 0].set_xlabel('Time Steps')
            axes[0, 0].set_ylabel('Channels')
            plt.colorbar(im, ax=axes[0, 0])
        else:
            axes[0, 0].text(0.5, 0.5, 'No attribution data available', ha='center', va='center', transform=axes[0, 0].transAxes)

        # Attribution statistics
        attr_stats = method_specific.get('attribution_statistics', {})
        if attr_stats:
            stats_text = "Attribution Statistics:\n"
            for key, value in attr_stats.items():
                stats_text += f"{key}: {value:.4f}\n"
            axes[0, 1].text(0.1, 0.9, stats_text, transform=axes[0, 1].transAxes, verticalalignment='top')
            axes[0, 1].set_title('Attribution Statistics')
            axes[0, 1].axis('off')
        else:
            axes[0, 1].text(0.5, 0.5, 'No statistics available', ha='center', va='center', transform=axes[0, 1].transAxes)

        # Important regions
        important_regions = method_specific.get('important_regions', [])
        if important_regions:
            regions_text = f"Important Regions: {len(important_regions)}\n"
            for i, region in enumerate(important_regions[:3]):  # Show top 3
                regions_text += f"Channel {region['channel']}: {region['coverage']:.2%} coverage\n"
            axes[1, 0].text(0.1, 0.9, regions_text, transform=axes[1, 0].transAxes, verticalalignment='top')
            axes[1, 0].set_title('Important Regions Summary')
            axes[1, 0].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, 'No important regions identified', ha='center', va='center', transform=axes[1, 0].transAxes)

        # Peak attributions
        peak_attributions = method_specific.get('peak_attribution_indices', [])
        if peak_attributions:
            axes[1, 1].scatter([p['time'] for p in peak_attributions], [p['value'] for p in peak_attributions])
            axes[1, 1].set_title('Peak Attribution Locations')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Attribution Value')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No peak attributions found', ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.tight_layout()
        return fig

    def _visualize_attribution_heatmap(self, explanation: Explanation) -> plt.Figure:
        """Visualize attribution as heatmap."""
        attributions = explanation.get_data('attributions')

        if attributions is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'No attribution data available', ha='center', va='center', transform=ax.transAxes)
            return fig

        if attributions.ndim == 1:
            attributions = attributions.reshape(1, -1)

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(attributions, aspect='auto', cmap='hot', interpolation='bilinear')
        ax.set_title('Grad-CAM Attribution Heatmap')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Channels')
        plt.colorbar(im, ax=ax, label='Attribution Intensity')

        plt.tight_layout()
        return fig

    def _visualize_signal_overlay(self, explanation: Explanation) -> plt.Figure:
        """Visualize attribution overlaid on original signal."""
        attributions = explanation.get_data('attributions')
        original_signal = explanation.get_data('original_signal')

        if attributions is None or original_signal is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'Missing attribution or signal data', ha='center', va='center', transform=ax.transAxes)
            return fig

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot original signal
        if isinstance(original_signal, torch.Tensor):
            signal_data = original_signal.detach().cpu().numpy().flatten()
        else:
            signal_data = np.array(original_signal).flatten()

        time_axis = np.arange(len(signal_data)) / 1024.0  # Assuming 1024 Hz
        ax.plot(time_axis, signal_data, 'b-', alpha=0.7, label='Original Signal')

        # Overlay attribution as background
        if attributions.ndim == 1:
            attribution_data = attributions
        else:
            attribution_data = attributions[0]  # Use first channel

        # Interpolate attribution to match signal length
        if len(attribution_data) != len(signal_data):
            from scipy import interpolate
            f = interpolate.interp1d(np.linspace(0, 1, len(attribution_data)), attribution_data, kind='linear')
            attribution_data = f(np.linspace(0, 1, len(signal_data)))

        # Normalize attribution for visualization
        if np.max(np.abs(attribution_data)) > 0:
            attribution_normalized = (attribution_data - np.min(attribution_data)) / (np.max(attribution_data) - np.min(attribution_data))
            attribution_normalized = attribution_normalized * (np.max(signal_data) - np.min(signal_data)) * 0.3 + np.min(signal_data)
        else:
            attribution_normalized = attribution_data

        ax.fill_between(time_axis, attribution_normalized, alpha=0.3, color='red', label='Grad-CAM Attribution')
        ax.set_title('Signal with Grad-CAM Attribution Overlay')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _visualize_importance_profile(self, explanation: Explanation) -> plt.Figure:
        """Visualize importance profile over time."""
        attributions = explanation.get_data('attributions')

        if attributions is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'No attribution data available', ha='center', va='center', transform=ax.transAxes)
            return fig

        if attributions.ndim == 1:
            attributions = attributions.reshape(1, -1)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot attribution profiles for each channel
        for channel_idx in range(attributions.shape[0]):
            channel_attr = attributions[channel_idx]
            time_axis = np.arange(len(channel_attr)) / 1024.0
            ax.plot(time_axis, channel_attr, label=f'Channel {channel_idx}', alpha=0.7)

        ax.set_title('Attribution Importance Profile')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Attribution Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig