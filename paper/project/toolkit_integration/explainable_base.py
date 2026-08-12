"""
Explainable Model Base Classes

Provides base classes and mixins for making models explainable.
This module defines the standard interface that explainable models should implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import numpy as np


class ExplainableMixin(ABC):
    """
    Mixin class that provides explainability interface for models.

    Models that inherit from this mixin should implement the explainability
    methods to provide standardized access to their internal representations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._explainability_hooks = {}
        self._intermediate_outputs = {}

    @abstractmethod
    def get_signal_path(self, input_data: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Get the signal transformation path through the model.

        Returns a list of dictionaries, where each dictionary contains information
        about a layer's input, output, and transformation type.

        Args:
            input_data: Input tensor [batch_size, sequence_length, channels]

        Returns:
            List of layer information dictionaries
        """
        pass

    @abstractmethod
    def get_operator_graph(self) -> Dict[str, Any]:
        """
        Get the operator graph structure of the model.

        Returns information about how different operators/processing blocks
        are connected in the model.

        Returns:
            Dictionary containing graph structure information
        """
        pass

    @abstractmethod
    def get_attention_maps(self, input_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get attention weights/maps from the model.

        Args:
            input_data: Input tensor

        Returns:
            Dictionary containing attention tensors
        """
        pass

    def register_forward_hook(self, layer_name: str, hook_fn):
        """
        Register a forward hook for a specific layer.

        Args:
            layer_name: Name of the layer to hook
            hook_fn: Hook function to register
        """
        if hasattr(self, layer_name):
            layer = getattr(self, layer_name)
            hook_handle = layer.register_forward_hook(hook_fn)
            self._explainability_hooks[layer_name] = hook_handle
            return hook_handle
        else:
            raise AttributeError(f"Layer '{layer_name}' not found in model")

    def clear_explainability_hooks(self):
        """Remove all registered explainability hooks."""
        for hook_handle in self._explainability_hooks.values():
            hook_handle.remove()
        self._explainability_hooks.clear()

    def get_intermediate_outputs(self, input_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate outputs from all layers with parameters.

        Args:
            input_data: Input tensor

        Returns:
            Dictionary mapping layer names to their output tensors
        """
        intermediate_outputs = {}
        hooks = []

        def create_hook(name):
            def hook(module, input, output):
                intermediate_outputs[name] = output.clone().detach()
            return hook

        # Register hooks for all layers with parameters
        for name, module in self.named_modules():
            if name and any(p.requires_grad for p in module.parameters()):
                hook = module.register_forward_hook(create_hook(name))
                hooks.append(hook)

        try:
            # Forward pass to collect intermediate outputs
            with torch.no_grad():
                _ = self.forward(input_data)
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()

        return intermediate_outputs

    def _compute_signal_stats(self, signal: torch.Tensor) -> Dict[str, float]:
        """
        Compute basic statistics for a signal tensor.

        Args:
            signal: Signal tensor

        Returns:
            Dictionary of signal statistics
        """
        if isinstance(signal, torch.Tensor):
            signal_np = signal.detach().cpu().numpy()
        else:
            signal_np = signal

        return {
            'mean': float(np.mean(signal_np)),
            'std': float(np.std(signal_np)),
            'rms': float(np.sqrt(np.mean(signal_np ** 2))),
            'max': float(np.max(signal_np)),
            'min': float(np.min(signal_np)),
            'energy': float(np.sum(signal_np ** 2))
        }

    def _analyze_frequency_content(self, signal: torch.Tensor, sampling_rate: float = 1024.0) -> Dict[str, Any]:
        """
        Analyze frequency content of a signal.

        Args:
            signal: Signal tensor
            sampling_rate: Sampling rate in Hz

        Returns:
            Dictionary containing frequency analysis results
        """
        if isinstance(signal, torch.Tensor):
            signal_np = signal.detach().cpu().numpy().flatten()
        else:
            signal_np = signal.flatten()

        # Compute FFT
        fft_vals = np.fft.fft(signal_np)
        fft_freq = np.fft.fftfreq(len(signal_np), 1/sampling_rate)

        # Only keep positive frequencies
        pos_mask = fft_freq > 0
        pos_freq = fft_freq[pos_mask]
        pos_fft = np.abs(fft_vals[pos_mask])

        # Find dominant frequencies
        if len(pos_fft) > 0:
            dominant_freq_idx = np.argmax(pos_fft)
            dominant_freq = pos_freq[dominant_freq_idx]
            dominant_power = pos_fft[dominant_freq_idx]
        else:
            dominant_freq = 0.0
            dominant_power = 0.0

        # Compute spectral centroid
        spectral_centroid = np.sum(pos_freq * pos_fft) / (np.sum(pos_fft) + 1e-8)

        return {
            'dominant_frequency': dominant_freq,
            'dominant_power': float(dominant_power),
            'spectral_centroid': float(spectral_centroid),
            'total_power': float(np.sum(pos_fft))
        }


class ExplainableModel(nn.Module, ExplainableMixin):
    """
    Base class for explainable models.

    This class combines nn.Module with ExplainableMixin to provide a base
    class for models that want to be explainable.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self._setup_explainability()

    def _setup_explainability(self):
        """
        Setup explainability features for the model.
        Override this method in subclasses to add model-specific explainability setup.
        """
        pass

    def explain(self,
                input_data: torch.Tensor,
                method: str = 'signal_path',
                **kwargs) -> Dict[str, Any]:
        """
        Generate explanation for the given input.

        Args:
            input_data: Input tensor
            method: Explanation method to use
            **kwargs: Additional arguments

        Returns:
            Explanation results dictionary
        """
        if method == 'signal_path':
            return {'signal_path': self.get_signal_path(input_data)}
        elif method == 'operator_graph':
            return {'operator_graph': self.get_operator_graph()}
        elif method == 'attention':
            return {'attention_maps': self.get_attention_maps(input_data)}
        elif method == 'intermediate':
            return {'intermediate_outputs': self.get_intermediate_outputs(input_data)}
        else:
            raise ValueError(f"Unknown explanation method: {method}")

    def get_explainability_info(self) -> Dict[str, Any]:
        """
        Get information about the explainability capabilities of this model.

        Returns:
            Dictionary containing explainability information
        """
        methods = []

        # Check which methods are implemented
        try:
            # Try with dummy input to check if method is implemented
            dummy_input = torch.randn(1, 1000, 1)  # Adjust size as needed
            try:
                self.get_signal_path(dummy_input)
                methods.append('signal_path')
            except NotImplementedError:
                pass
            except Exception:
                # Method exists but might have issues with dummy input
                methods.append('signal_path')
        except Exception:
            pass

        try:
            self.get_operator_graph()
            methods.append('operator_graph')
        except NotImplementedError:
            pass
        except Exception:
            methods.append('operator_graph')

        try:
            dummy_input = torch.randn(1, 1000, 1)
            self.get_attention_maps(dummy_input)
            methods.append('attention')
        except NotImplementedError:
            pass
        except Exception:
            methods.append('attention')

        return {
            'model_type': type(self).__name__,
            'supported_methods': methods,
            'config': self.config
        }