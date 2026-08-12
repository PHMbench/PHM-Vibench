"""
SHAP Explainer for Signal Processing Networks

Post-hoc explanation method that provides a lightweight wrapper for SHAP (SHapley Additive exPlanations)
adapted for fault diagnosis models. This method computes feature attributions using game-theoretic
approaches to explain individual predictions.
"""

from typing import Dict, Any, Optional, Sequence, List, Union, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

from ...core import SignalData, Explanation, ExplainabilityMethod, BaseExplainerAdapter


class SHAPExplainer(BaseExplainerAdapter, ExplainabilityMethod):
    """
    SHAP Explainer for post-hoc explanations.

    This method provides a lightweight implementation of SHAP values for signal processing
    networks, computing feature attributions that fairly distribute the prediction among
    different features of the input signal.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SHAP Explainer.

        Args:
            config: Configuration dictionary with the following options:
                - explanation_method: str ('kernel', 'gradient', 'deep', default 'gradient')
                - background_samples: int (default 10)
                - n_segments: int (default 50) - number of segments to group signal features
                - use_segments: bool (default True) - whether to segment signals for efficiency
                - normalize_shap_values: bool (default True)
                - aggregate_channels: bool (default True) - aggregate SHAP values across channels
        """
        super().__init__(config)
        self._method_name = "SHAP"
        self._method_type = "posthoc"

        # Configuration options
        self.explanation_method = self.config.get('explanation_method', 'gradient')
        self.background_samples = self.config.get('background_samples', 10)
        self.n_segments = self.config.get('n_segments', 50)
        self.use_segments = self.config.get('use_segments', True)
        self.normalize_shap_values = self.config.get('normalize_shap_values', True)
        self.aggregate_channels = self.config.get('aggregate_channels', True)

        # Internal state
        self._model = None
        self._background_data = None
        self._segment_boundaries = None

    def explain(self,
                signal: SignalData,
                prediction: Any,
                model: Optional[torch.nn.Module] = None,
                target_class: Optional[int] = None,
                **kwargs) -> Explanation:
        """
        Generate SHAP explanation for the given signal.

        Args:
            signal: SignalData object containing the signal to explain
            prediction: Model prediction or target class
            model: Model to explain (must be provided)
            target_class: Target class for explanation (uses prediction if None)
            **kwargs: Additional arguments including:
                - background_signals: List of background SignalData objects
                - return_raw_shap: bool (default False)

        Returns:
            Explanation object containing SHAP attribution results
        """
        self._validate_signal(signal)

        # Get model
        model = model or kwargs.get('model')
        if model is None:
            raise ValueError("Model must be provided for SHAP explanation")

        self._model = model

        # Prepare background data
        background_signals = kwargs.get('background_signals', None)
        self._prepare_background_data(background_signals, signal)

        # Get target class
        if target_class is None:
            input_tensor = self._prepare_input(signal)
            target_class = self._get_prediction_class(model, input_tensor, prediction)

        # Generate SHAP values
        shap_values = self._compute_shap_values(signal, target_class)

        # Process SHAP values
        processed_shap = self._process_shap_values(shap_values, signal)

        # Generate explanation
        explanation_data = {
            'attributions': processed_shap,
            'original_signal': signal.raw_signal,
            'target_class': target_class,
            'method_specific': {
                'shap_statistics': self._compute_shap_statistics(processed_shap),
                'feature_importance': self._compute_feature_importance(processed_shap),
                'segment_contributions': self._compute_segment_contributions(processed_shap) if self.use_segments else None,
                'shap_interactions': self._compute_shap_interactions(processed_shap) if self.explanation_method == 'kernel' else None,
                'explanation_method': self.explanation_method
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

        return Explanation(explanation_data, explanation_meta)

    def visualize(self,
                  explanation: Explanation,
                  mode: str = 'auto',
                  **kwargs) -> plt.Figure:
        """
        Create visualization for SHAP explanation.

        Args:
            explanation: Explanation object to visualize
            mode: Visualization mode ('auto', 'values', 'features', 'segments', 'waterfall')
            **kwargs: Additional visualization parameters

        Returns:
            Matplotlib figure object
        """
        self._validate_explanation(explanation)

        if mode == 'auto':
            return self._visualize_shap_overview(explanation)
        elif mode == 'values':
            return self._visualize_shap_values(explanation)
        elif mode == 'features':
            return self._visualize_feature_importance(explanation)
        elif mode == 'segments':
            return self._visualize_segment_contributions(explanation)
        elif mode == 'waterfall':
            return self._visualize_shap_waterfall(explanation)
        else:
            raise ValueError(f"Unknown visualization mode: {mode}")

    def evaluate(self,
                 explanations: Sequence[Explanation],
                 ground_truth: Optional[Sequence[Any]] = None,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate SHAP explanations.

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

        # SHAP-specific metrics
        shap_entropies = []
        feature_importance_spreads = []
        explanation_magnitudes = []
        attribution_consistencies = []

        for exp in explanations:
            shap_values = exp.get_data('attributions')
            method_specific = exp.get_data('method_specific', {})

            if shap_values is not None:
                # Compute SHAP entropy
                shap_flat = np.array(shap_values).flatten()
                if np.sum(np.abs(shap_flat)) > 0:
                    shap_abs = np.abs(shap_flat)
                    shap_abs = shap_abs / np.sum(shap_abs)  # Normalize
                    entropy = -np.sum(shap_abs * np.log2(shap_abs + 1e-10))
                    shap_entropies.append(entropy)

                # Magnitude statistics
                explanation_magnitudes.append(float(np.mean(np.abs(shap_flat))))

            # Feature importance analysis
            feature_importance = method_specific.get('feature_importance', {})
            if feature_importance:
                importance_values = list(feature_importance.values())
                feature_importance_spreads.append(float(np.std(importance_values)) if len(importance_values) > 1 else 0.0)

        # Compute metrics
        metrics['avg_shap_entropy'] = float(np.mean(shap_entropies)) if shap_entropies else 0.0
        metrics['avg_explanation_magnitude'] = float(np.mean(explanation_magnitudes)) if explanation_magnitudes else 0.0
        metrics['avg_importance_spread'] = float(np.mean(feature_importance_spreads)) if feature_importance_spreads else 0.0
        metrics['explanation_diversity'] = metrics['avg_shap_entropy']  # Reuse entropy as diversity measure
        metrics['feature_selectivity'] = 1.0 - min(metrics['avg_shap_entropy'] / 10.0, 1.0)  # Inverse of entropy

        return metrics

    def _prepare_background_data(self, background_signals: Optional[List[SignalData]], reference_signal: SignalData) -> None:
        """Prepare background data for SHAP computation."""
        if background_signals is not None:
            # Use provided background signals
            background_tensors = [self._prepare_input(signal) for signal in background_signals]
            self._background_data = torch.stack(background_tensors) if background_tensors else None
        else:
            # Create synthetic background using noise or zeros
            if self.explanation_method == 'gradient':
                # For gradient SHAP, use zeros as background
                signal_tensor = self._prepare_input(reference_signal)
                self._background_data = torch.zeros_like(signal_tensor).unsqueeze(0).repeat(self.background_samples, 1, 1)
            else:
                # For other methods, create noisy versions
                background_tensors = []
                signal_tensor = self._prepare_input(reference_signal)
                for _ in range(self.background_samples):
                    noise = torch.randn_like(signal_tensor) * 0.1
                    background_tensors.append(signal_tensor + noise)
                self._background_data = torch.stack(background_tensors)

        # Prepare segment boundaries if using segments
        if self.use_segments:
            signal_length = reference_signal.get_length()
            segment_length = signal_length // self.n_segments
            self._segment_boundaries = [(i * segment_length, (i + 1) * segment_length)
                                      for i in range(self.n_segments)]

    def _prepare_input(self, signal: SignalData) -> torch.Tensor:
        """Prepare signal data for model input."""
        if isinstance(signal.raw_signal, np.ndarray):
            input_tensor = torch.FloatTensor(signal.raw_signal)
        else:
            input_tensor = signal.raw_signal

        # Ensure correct shape [channels, sequence_length]
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0)
        elif len(input_tensor.shape) == 3:
            input_tensor = input_tensor.squeeze(0)  # Remove batch dimension

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
                model_output = model(input_tensor.unsqueeze(0))  # Add batch dimension
                if hasattr(model_output, 'logits'):
                    model_output = model_output.logits
                return int(torch.argmax(model_output, dim=-1).item())

    def _compute_shap_values(self, signal: SignalData, target_class: int) -> np.ndarray:
        """Compute SHAP values for the signal."""
        if self.explanation_method == 'gradient':
            return self._compute_gradient_shap(signal, target_class)
        elif self.explanation_method == 'kernel':
            return self._compute_kernel_shap(signal, target_class)
        elif self.explanation_method == 'deep':
            return self._compute_deep_shap(signal, target_class)
        else:
            raise ValueError(f"Unsupported SHAP method: {self.explanation_method}")

    def _compute_gradient_shap(self, signal: SignalData, target_class: int) -> np.ndarray:
        """Compute SHAP values using gradient-based approach."""
        input_tensor = self._prepare_input(signal)
        input_tensor.requires_grad_(True)

        shap_values = []

        for background_sample in self._background_data:
            # Compute gradients for interpolation between background and input
            alpha = torch.linspace(0, 1, 50)  # 50 interpolation steps
            total_grad = 0.0

            for a in alpha:
                interpolated = background_sample + a * (input_tensor - background_sample)
                interpolated.requires_grad_(True)

                # Forward pass
                model_output = self._model(interpolated.unsqueeze(0))
                if hasattr(model_output, 'logits'):
                    model_output = model_output.logits

                target_score = model_output[0, target_class] if model_output.dim() > 1 else model_output[target_class]

                # Backward pass
                self._model.zero_grad()
                target_score.backward(retain_graph=True)

                if interpolated.grad is not None:
                    total_grad += interpolated.grad.detach()

            # Average gradients and multiply by input difference
            avg_grad = total_grad / len(alpha)
            sample_shap = avg_grad * (input_tensor - background_sample)
            shap_values.append(sample_shap)

        # Average over all background samples
        shap_values = torch.stack(shap_values).mean(dim=0)

        return shap_values.detach().cpu().numpy()

    def _compute_kernel_shap(self, signal: SignalData, target_class: int) -> np.ndarray:
        """Compute SHAP values using kernel SHAP approach (simplified)."""
        input_tensor = self._prepare_input(signal)

        if self.use_segments:
            return self._compute_kernel_shap_segments(input_tensor, target_class)
        else:
            return self._compute_kernel_shap_full(input_tensor, target_class)

    def _compute_kernel_shap_segments(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """Compute kernel SHAP using segmented approach."""
        n_features = len(self._segment_boundaries)
        shap_values = torch.zeros_like(input_tensor)

        # Compute model output for original input
        with torch.no_grad():
            original_output = self._model(input_tensor.unsqueeze(0))
            if hasattr(original_output, 'logits'):
                original_output = original_output.logits
            original_score = original_output[0, target_class] if original_output.dim() > 1 else original_output[target_class]

        # Compute model output for background
        with torch.no_grad():
            background_output = self._model(self._background_data)
            if hasattr(background_output, 'logits'):
                background_output = background_output.logits
            background_score = background_output[:, target_class].mean() if background_output.dim() > 1 else background_output[target_class].mean()

        # Compute SHAP values for each segment
        for i, (start_idx, end_idx) in enumerate(self._segment_boundaries):
            # Create perturbed input with this segment replaced by background
            perturbed_input = input_tensor.clone()
            for background_sample in self._background_data:
                perturbed_input[:, start_idx:end_idx] = background_sample[:, start_idx:end_idx]
                break  # Use first background sample

            # Compute model output for perturbed input
            with torch.no_grad():
                perturbed_output = self._model(perturbed_input.unsqueeze(0))
                if hasattr(perturbed_output, 'logits'):
                    perturbed_output = perturbed_output.logits
                perturbed_score = perturbed_output[0, target_class] if perturbed_output.dim() > 1 else perturbed_output[target_class]

            # Compute SHAP value for this segment
            segment_shap = (original_score - perturbed_score) * (n_features / (n_features - 1))
            shap_values[:, start_idx:end_idx] = segment_shap / (end_idx - start_idx)

        return shap_values.detach().cpu().numpy()

    def _compute_kernel_shap_full(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """Compute kernel SHAP for full features (computationally expensive)."""
        # This is a simplified version - full kernel SHAP would be too expensive
        warnings.warn("Full kernel SHAP is computationally expensive. Consider using segments or gradient SHAP.")

        # Fall back to gradient-based approach
        return self._compute_gradient_shap(
            SignalData(
                raw_signal=input_tensor.detach().cpu().numpy(),
                sampling_rate=1024.0  # Default sampling rate
            ),
            target_class
        )

    def _compute_deep_shap(self, signal: SignalData, target_class: int) -> np.ndarray:
        """Compute SHAP values using Deep SHAP approach (simplified)."""
        # Simplified Deep SHAP using gradient-based approach with layer-wise propagation
        input_tensor = self._prepare_input(signal)
        input_tensor.requires_grad_(True)

        # Forward pass
        model_output = self._model(input_tensor.unsqueeze(0))
        if hasattr(model_output, 'logits'):
            model_output = model_output.logits

        target_score = model_output[0, target_class] if model_output.dim() > 1 else model_output[target_class]

        # Backward pass
        self._model.zero_grad()
        target_score.backward(retain_graph=True)

        if input_tensor.grad is not None:
            shap_values = input_tensor.grad * input_tensor
        else:
            shap_values = torch.zeros_like(input_tensor)

        return shap_values.detach().cpu().numpy()

    def _process_shap_values(self, shap_values: np.ndarray, signal: SignalData) -> np.ndarray:
        """Process SHAP values for final output."""
        if self.aggregate_channels and shap_values.ndim > 1:
            # Aggregate across channels
            shap_values = np.mean(shap_values, axis=0, keepdims=True)

        # Normalize if configured
        if self.normalize_shap_values:
            max_abs_shap = np.max(np.abs(shap_values))
            if max_abs_shap > 0:
                shap_values = shap_values / max_abs_shap

        # Ensure 2D shape [channels, time]
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        return shap_values

    def _compute_shap_statistics(self, shap_values: np.ndarray) -> Dict[str, float]:
        """Compute statistics of SHAP values."""
        flat_shap = shap_values.flatten()

        return {
            'mean': float(np.mean(flat_shap)),
            'std': float(np.std(flat_shap)),
            'min': float(np.min(flat_shap)),
            'max': float(np.max(flat_shap)),
            'median': float(np.median(flat_shap)),
            'positive_ratio': float(np.mean(flat_shap > 0)),
            'negative_ratio': float(np.mean(flat_shap < 0)),
            'zero_ratio': float(np.mean(np.abs(flat_shap) < 1e-6)),
            'magnitude_mean': float(np.mean(np.abs(flat_shap))),
            'total_magnitude': float(np.sum(np.abs(flat_shap)))
        }

    def _compute_feature_importance(self, shap_values: np.ndarray) -> Dict[str, float]:
        """Compute feature importance from SHAP values."""
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        feature_importance = {}

        if self.use_segments and self._segment_boundaries:
            # Compute importance per segment
            for i, (start_idx, end_idx) in enumerate(self._segment_boundaries):
                segment_shap = np.abs(shap_values[:, start_idx:end_idx])
                importance = float(np.mean(segment_shap))
                feature_importance[f'segment_{i}'] = importance
        else:
            # Compute importance for top features
            feature_count = min(50, shap_values.shape[-1])
            feature_indices = np.linspace(0, shap_values.shape[-1]-1, feature_count, dtype=int)

            for i, idx in enumerate(feature_indices):
                feature_shap = np.abs(shap_values[:, idx])
                importance = float(np.mean(feature_shap))
                feature_importance[f'feature_{idx}'] = importance

        return feature_importance

    def _compute_segment_contributions(self, shap_values: np.ndarray) -> Optional[Dict[str, Any]]:
        """Compute detailed segment contributions."""
        if not self.use_segments or not self._segment_boundaries:
            return None

        contributions = {}

        for i, (start_idx, end_idx) in enumerate(self._segment_boundaries):
            segment_shap = shap_values[:, start_idx:end_idx]

            contributions[f'segment_{i}'] = {
                'start_idx': int(start_idx),
                'end_idx': int(end_idx),
                'mean_shap': float(np.mean(segment_shap)),
                'std_shap': float(np.std(segment_shap)),
                'total_shap': float(np.sum(np.abs(segment_shap))),
                'positive_shap': float(np.sum(segment_shap[segment_shap > 0])),
                'negative_shap': float(np.sum(segment_shap[segment_shap < 0])),
                'length': int(end_idx - start_idx)
            }

        return contributions

    def _compute_shap_interactions(self, shap_values: np.ndarray) -> Optional[Dict[str, float]]:
        """Compute SHAP interaction effects (simplified)."""
        if self.explanation_method != 'kernel' or shap_values.ndim == 1:
            return None

        # Simplified interaction computation for nearby features
        interactions = {}

        if self.use_segments and self._segment_boundaries:
            # Compute interactions between adjacent segments
            for i in range(len(self._segment_boundaries) - 1):
                segment_a = f'segment_{i}'
                segment_b = f'segment_{i+1}'

                # Simple interaction measure: correlation of SHAP values
                start_a, end_a = self._segment_boundaries[i]
                start_b, end_b = self._segment_boundaries[i + 1]

                shap_a = shap_values[:, start_a:end_a].flatten()
                shap_b = shap_values[:, start_b:end_b].flatten()

                if len(shap_a) > 0 and len(shap_b) > 0:
                    correlation = float(np.corrcoef(shap_a, shap_b)[0, 1])
                    interactions[f'{segment_a}_x_{segment_b}'] = correlation if not np.isnan(correlation) else 0.0

        return interactions

    # Visualization methods
    def _visualize_shap_overview(self, explanation: Explanation) -> plt.Figure:
        """Create overview visualization of SHAP results."""
        shap_values = explanation.get_data('attributions')
        method_specific = explanation.get_data('method_specific', {})

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # SHAP values heatmap
        if shap_values is not None:
            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(1, -1)

            im = axes[0, 0].imshow(shap_values, aspect='auto', cmap='RdBu_r', interpolation='bilinear')
            axes[0, 0].set_title('SHAP Values Heatmap')
            axes[0, 0].set_xlabel('Time Steps')
            axes[0, 0].set_ylabel('Channels')
            plt.colorbar(im, ax=axes[0, 0])
        else:
            axes[0, 0].text(0.5, 0.5, 'No SHAP data available', ha='center', va='center', transform=axes[0, 0].transAxes)

        # SHAP statistics
        shap_stats = method_specific.get('shap_statistics', {})
        if shap_stats:
            stats_text = "SHAP Statistics:\n"
            for key, value in shap_stats.items():
                stats_text += f"{key}: {value:.4f}\n"
            axes[0, 1].text(0.1, 0.9, stats_text, transform=axes[0, 1].transAxes, verticalalignment='top')
            axes[0, 1].set_title('SHAP Statistics')
            axes[0, 1].axis('off')
        else:
            axes[0, 1].text(0.5, 0.5, 'No statistics available', ha='center', va='center', transform=axes[0, 1].transAxes)

        # Feature importance
        feature_importance = method_specific.get('feature_importance', {})
        if feature_importance:
            # Show top 10 important features
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            feature_names = [name for name, _ in sorted_features]
            importance_values = [value for _, value in sorted_features]

            axes[1, 0].barh(range(len(feature_names)), importance_values)
            axes[1, 0].set_yticks(range(len(feature_names)))
            axes[1, 0].set_yticklabels(feature_names)
            axes[1, 0].set_title('Top 10 Feature Importance')
            axes[1, 0].set_xlabel('Importance Score')
        else:
            axes[1, 0].text(0.5, 0.5, 'No feature importance data', ha='center', va='center', transform=axes[1, 0].transAxes)

        # Method information
        method_info = method_specific.get('explanation_method', 'Unknown')
        info_text = f"SHAP Method: {method_info}\n"
        info_text += f"Using Segments: {self.use_segments}\n"
        if self.use_segments:
            info_text += f"Number of Segments: {self.n_segments}\n"

        axes[1, 1].text(0.1, 0.5, info_text, transform=axes[1, 1].transAxes, verticalalignment='center', fontsize=12)
        axes[1, 1].set_title('Configuration Information')
        axes[1, 1].axis('off')

        plt.tight_layout()
        return fig

    def _visualize_shap_values(self, explanation: Explanation) -> plt.Figure:
        """Visualize SHAP values over time."""
        shap_values = explanation.get_data('attributions')

        if shap_values is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'No SHAP data available', ha='center', va='center', transform=ax.transAxes)
            return fig

        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot SHAP values for each channel
        for channel_idx in range(shap_values.shape[0]):
            channel_shap = shap_values[channel_idx]
            time_axis = np.arange(len(channel_shap)) / 1024.0  # Assuming 1024 Hz
            ax.plot(time_axis, channel_shap, label=f'Channel {channel_idx}', alpha=0.7)

        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_title('SHAP Values Over Time')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('SHAP Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _visualize_feature_importance(self, explanation: Explanation) -> plt.Figure:
        """Visualize feature importance rankings."""
        feature_importance = explanation.get_data('method_specific', {}).get('feature_importance', {})

        if not feature_importance:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'No feature importance data available', ha='center', va='center', transform=ax.transAxes)
            return fig

        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        # Limit to top 20 for visualization
        top_features = sorted_features[:20]
        feature_names = [name for name, _ in top_features]
        importance_values = [value for _, value in top_features]

        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(range(len(feature_names)), importance_values)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_title('Feature Importance Rankings')
        ax.set_xlabel('Importance Score')

        # Add value labels on bars
        for bar, value in zip(bars, importance_values):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{value:.3f}', ha='left', va='center')

        plt.tight_layout()
        return fig

    def _visualize_segment_contributions(self, explanation: Explanation) -> plt.Figure:
        """Visualize segment-wise contributions."""
        segment_contributions = explanation.get_data('method_specific', {}).get('segment_contributions', {})

        if not segment_contributions:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'No segment contribution data available', ha='center', va='center', transform=ax.transAxes)
            return fig

        # Extract segment information
        segment_names = list(segment_contributions.keys())
        positive_contributions = [segment_contributions[name]['positive_shap'] for name in segment_names]
        negative_contributions = [segment_contributions[name]['negative_shap'] for name in segment_names]

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(segment_names))
        width = 0.35

        bars1 = ax.bar(x - width/2, positive_contributions, width, label='Positive SHAP', alpha=0.7, color='green')
        bars2 = ax.bar(x + width/2, negative_contributions, width, label='Negative SHAP', alpha=0.7, color='red')

        ax.set_xlabel('Segments')
        ax.set_ylabel('SHAP Contribution')
        ax.set_title('Segment-wise SHAP Contributions')
        ax.set_xticks(x)
        ax.set_xticklabels(segment_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _visualize_shap_waterfall(self, explanation: Explanation) -> plt.Figure:
        """Create waterfall plot of SHAP contributions."""
        shap_values = explanation.get_data('attributions')
        method_specific = explanation.get_data('method_specific', {})

        if shap_values is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'No SHAP data available', ha='center', va='center', transform=ax.transAxes)
            return fig

        # Flatten SHAP values for waterfall plot
        flat_shap = shap_values.flatten()

        # Get top positive and negative contributions
        top_n = min(10, len(flat_shap))

        # Get indices for top positive and negative values
        pos_indices = np.argsort(flat_shap)[-top_n//2:]
        neg_indices = np.argsort(flat_shap)[:top_n//2]

        all_indices = np.concatenate([neg_indices, pos_indices])
        sorted_indices = all_indices[np.argsort([flat_shap[i] for i in all_indices])]

        contributions = flat_shap[sorted_indices]
        base_value = np.mean(flat_shap)  # Simplified base value
        cumulative = base_value + np.cumsum(contributions)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create waterfall plot
        x_pos = np.arange(len(sorted_indices))

        # Plot bars
        colors = ['red' if c < 0 else 'green' for c in contributions]
        bars = ax.bar(x_pos, contributions, color=colors, alpha=0.7)

        # Plot cumulative line
        ax.plot(x_pos, cumulative, 'k-o', linewidth=2, markersize=4, label='Cumulative')

        ax.axhline(y=base_value, color='blue', linestyle='--', alpha=0.5, label=f'Base Value: {base_value:.3f}')
        ax.set_title('SHAP Waterfall Plot')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('SHAP Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig