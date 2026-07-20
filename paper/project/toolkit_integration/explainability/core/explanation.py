"""
Unified Explanation Object

Provides a standardized format for explanation results across all methods.
"""

from typing import Dict, Any, Optional, List, Union
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


class Explanation:
    """
    Unified explanation object that standardizes the format of explanation results.

    This class provides a consistent interface for accessing explanation data,
    regardless of the explanation method used. It includes methods for visualization,
    serialization, and metric computation.
    """

    def __init__(self,
                 data: Dict[str, Any],
                 meta: Optional[Dict[str, Any]] = None):
        """
        Initialize explanation object.

        Args:
            data: Dictionary containing explanation data (attributions, paths, etc.)
            meta: Dictionary containing metadata (method info, model info, etc.)
        """
        self.data = data
        self.meta = meta or {}
        self._plots = {}  # Cache for generated plots

    def get_data(self, key: str, default: Any = None) -> Any:
        """Get data by key with default fallback."""
        return self.data.get(key, default)

    def get_meta(self, key: str, default: Any = None) -> Any:
        """Get metadata by key with default fallback."""
        return self.meta.get(key, default)

    def get_attribution(self) -> Optional[np.ndarray]:
        """Get main attribution values if available."""
        for key in ['attributions', 'importance_scores', 'saliency', 'path']:
            if key in self.data:
                attr = self.data[key]
                if isinstance(attr, torch.Tensor):
                    return attr.detach().cpu().numpy()
                elif isinstance(attr, np.ndarray):
                    return attr
                elif isinstance(attr, list):
                    return np.array(attr)
        return None

    def get_method_name(self) -> str:
        """Get the explanation method name."""
        return self.get_meta('method', 'unknown')

    def get_model_name(self) -> str:
        """Get the model name if available."""
        return self.get_meta('model_name', 'unknown')

    def visualize(self, mode: str = 'auto') -> plt.Figure:
        """
        Generate visualization of the explanation.

        Args:
            mode: Visualization mode ('auto', 'attribution', 'path', 'importance')

        Returns:
            Matplotlib figure object
        """
        if mode in self._plots:
            return self._plots[mode]

        fig = None

        if mode == 'auto':
            # Auto-select best visualization based on available data
            if 'path' in self.data:
                fig = self._visualize_signal_path()
            elif 'importance_scores' in self.data:
                fig = self._visualize_importance_scores()
            else:
                fig = self._visualize_attribution()
        elif mode == 'path' and 'path' in self.data:
            fig = self._visualize_signal_path()
        elif mode == 'importance' and 'importance_scores' in self.data:
            fig = self._visualize_importance_scores()
        else:
            fig = self._visualize_attribution()

        if fig:
            self._plots[mode] = fig

        return fig

    def _visualize_attribution(self) -> plt.Figure:
        """Visualize attribution values."""
        attribution = self.get_attribution()
        if attribution is None:
            raise ValueError("No attribution data available for visualization")

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Raw signal (if available)
        if 'original_signal' in self.data:
            signal = self.data['original_signal']
            if isinstance(signal, torch.Tensor):
                signal = signal.detach().cpu().numpy()
            axes[0].plot(signal.flatten())
            axes[0].set_title('Original Signal')
            axes[0].set_xlabel('Time')
            axes[0].set_ylabel('Amplitude')
            axes[0].grid(True, alpha=0.3)

        # Attribution values
        axes[1].plot(attribution.flatten())
        axes[1].set_title(f'Attribution ({self.get_method_name()})')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Attribution Score')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _visualize_signal_path(self) -> plt.Figure:
        """Visualize signal transformation path."""
        if 'path' not in self.data:
            raise ValueError("No path data available for visualization")

        path = self.data['path']
        if not isinstance(path, list):
            raise ValueError("Path data must be a list")

        n_layers = len(path)
        fig, axes = plt.subplots(n_layers + 1, 1, figsize=(12, 2 * (n_layers + 1)))

        # Original signal
        if 'original_signal' in self.data:
            signal = self.data['original_signal']
            if isinstance(signal, torch.Tensor):
                signal = signal.detach().cpu().numpy()
            axes[0].plot(signal.flatten())
            axes[0].set_title('Original Signal')
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, 'Original Signal\n(Not Available)',
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Original Signal')

        # Signal path
        for i, layer_info in enumerate(path):
            if 'output_signal' in layer_info:
                signal = layer_info['output_signal']
                if isinstance(signal, torch.Tensor):
                    signal = signal.detach().cpu().numpy()
                axes[i + 1].plot(signal.flatten())
                title = layer_info.get('layer_name', f'Layer {i}')
                if 'operator_type' in layer_info:
                    title += f' ({layer_info["operator_type"]})'
                axes[i + 1].set_title(title)
                axes[i + 1].grid(True, alpha=0.3)
            else:
                title = layer_info.get('layer_name', f'Layer {i}')
                axes[i + 1].text(0.5, 0.5, f'{title}\n(Not Available)',
                                ha='center', va='center', transform=axes[i + 1].transAxes)
                axes[i + 1].set_title(title)

        plt.tight_layout()
        return fig

    def _visualize_importance_scores(self) -> plt.Figure:
        """Visualize importance scores."""
        if 'importance_scores' not in self.data:
            raise ValueError("No importance scores available for visualization")

        scores = self.data['importance_scores']
        if not isinstance(scores, dict):
            raise ValueError("Importance scores must be a dictionary")

        # Extract names and scores
        names = list(scores.keys())
        values = []

        for name in names:
            score_data = scores[name]
            if isinstance(score_data, dict):
                # Use combined_score if available, otherwise use first value
                values.append(score_data.get('combined_score', list(score_data.values())[0]))
            else:
                values.append(score_data)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names, values)
        ax.set_title(f'Importance Scores ({self.get_method_name()})')
        ax.set_ylabel('Importance Score')
        ax.set_xlabel('Operators/Layers')
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        return fig

    def get_metrics(self) -> Dict[str, float]:
        """
        Compute basic explanation metrics.

        Returns:
            Dictionary of computed metrics
        """
        metrics = {}

        attribution = self.get_attribution()
        if attribution is not None:
            # Basic attribution statistics
            metrics['attribution_mean'] = float(np.mean(attribution))
            metrics['attribution_std'] = float(np.std(attribution))
            metrics['attribution_max'] = float(np.max(np.abs(attribution)))
            metrics['attribution_sparsity'] = float(np.mean(np.abs(attribution) < 0.01))

        return metrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation to dictionary for serialization."""
        # Convert tensors to numpy for serialization
        serializable_data = {}
        for key, value in self.data.items():
            if isinstance(value, torch.Tensor):
                serializable_data[key] = value.detach().cpu().numpy().tolist()
            elif isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            else:
                serializable_data[key] = value

        return {
            'data': serializable_data,
            'meta': self.meta,
            'metrics': self.get_metrics()
        }

    def to_json(self, filepath: Union[str, Path]) -> None:
        """Save explanation as JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_visualization(self, filepath: Union[str, Path], mode: str = 'auto') -> None:
        """Save visualization to file."""
        fig = self.visualize(mode=mode)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')

    def __repr__(self) -> str:
        """String representation of the explanation."""
        method = self.get_method_name()
        model = self.get_model_name()
        return f"Explanation(method='{method}', model='{model}', data_keys={list(self.data.keys())})"