"""
Unified Explainer

Provides a unified interface for all explanation methods in the toolkit.
This is the main entry point for users to generate explanations.
"""

from typing import Dict, Any, Optional, Union, List
import torch
from .explanation import Explanation
from .base_explainer import BaseExplainer

# Import specific explainers
from ..methods.intrinsic.signal_path_explainer import SignalPathExplainer
from ..methods.posthoc.captum_wrapper import CaptumWrapper


class UnifiedExplainer:
    """
    Unified interface for all explanation methods.

    This class provides a simple, consistent interface for generating explanations
    using different methods. It automatically selects the appropriate explainer
    and returns standardized explanation objects.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 config: Optional[Dict[str, Any]] = None,
                 method: str = 'auto'):
        """
        Initialize the unified explainer.

        Args:
            model: The model to explain
            config: Configuration dictionary
            method: Explanation method to use:
                   - 'auto': automatically select best method
                   - 'signal_path': intrinsic signal path explanation
                   - 'integrated_gradients': Captum Integrated Gradients
                   - 'captum': Captum wrapper (specify method in config)
        """
        self.model = model
        self.config = config or {}
        self.method = method

        # Initialize explainer based on method
        self.explainer = self._initialize_explainer()

    def _initialize_explainer(self) -> BaseExplainer:
        """Initialize the specific explainer based on method."""
        if self.method == 'auto':
            # Auto-select method based on model capabilities
            return self._auto_select_explainer()
        elif self.method == 'signal_path':
            return SignalPathExplainer(self.model, self.config)
        elif self.method in ['integrated_gradients', 'deeplift', 'saliency', 'captum']:
            # Use Captum wrapper
            if self.method == 'captum':
                captum_method = self.config.get('captum_method', 'integrated_gradients')
                captum_config = self.config.copy()
                captum_config['method'] = captum_method
            else:
                captum_config = self.config.copy()
                captum_config['method'] = self.method

            return CaptumWrapper(self.model, captum_config)
        else:
            raise ValueError(f"Unknown explanation method: {self.method}")

    def _auto_select_explainer(self) -> BaseExplainer:
        """
        Automatically select the best explainer based on model capabilities.

        Selection logic:
        1. If model supports signal path -> use signal_path
        2. Otherwise -> use integrated_gradients
        """
        # Check if model supports intrinsic explanations
        if hasattr(self.model, 'get_signal_path'):
            try:
                # Test with dummy input
                dummy_input = torch.randn(1, 1000, 1)
                self.model.get_signal_path(dummy_input)
                return SignalPathExplainer(self.model, self.config)
            except Exception:
                pass

        # Fall back to post-hoc explanation
        return CaptumWrapper(self.model, {'method': 'integrated_gradients'})

    def explain(self,
                input_data: torch.Tensor,
                target_class: Optional[int] = None,
                method: Optional[str] = None,
                **kwargs) -> Explanation:
        """
        Generate explanation for the given input.

        Args:
            input_data: Input tensor [batch_size, sequence_length, channels]
            target_class: Target class for explanation
            method: Override the default explanation method
            **kwargs: Additional arguments passed to the specific explainer

        Returns:
            Explanation object containing the explanation results
        """
        # Use temporary explainer if method is overridden
        if method is not None and method != self.method:
            temp_config = self.config.copy()
            temp_explainer = UnifiedExplainer(self.model, temp_config, method)
            return temp_explainer.explain(input_data, target_class, **kwargs)

        # Use default explainer
        return self.explainer.explain(input_data, target_class, **kwargs)

    def explain_batch(self,
                      input_data: torch.Tensor,
                      target_classes: Optional[List[int]] = None,
                      **kwargs) -> List[Explanation]:
        """
        Generate explanations for a batch of inputs.

        Args:
            input_data: Input tensor [batch_size, sequence_length, channels]
            target_classes: List of target classes for each sample
            **kwargs: Additional arguments

        Returns:
            List of Explanation objects
        """
        if hasattr(self.explainer, 'explain_batch'):
            return self.explainer.explain_batch(input_data, target_classes, **kwargs)

        # Fallback: explain each sample individually
        explanations = []
        batch_size = input_data.shape[0]

        for i in range(batch_size):
            sample_input = input_data[i:i+1]  # Keep batch dimension
            sample_target = target_classes[i] if target_classes else None

            explanation = self.explainer.explain(sample_input, sample_target, **kwargs)
            explanations.append(explanation)

        return explanations

    def compare_methods(self,
                        input_data: torch.Tensor,
                        target_class: Optional[int] = None,
                        methods: Optional[List[str]] = None,
                        **kwargs) -> Dict[str, Explanation]:
        """
        Compare multiple explanation methods on the same input.

        Args:
            input_data: Input tensor
            target_class: Target class for explanation
            methods: List of methods to compare
            **kwargs: Additional arguments

        Returns:
            Dictionary mapping method names to Explanation objects
        """
        if methods is None:
            # Default methods to compare
            methods = ['signal_path', 'integrated_gradients']

            # Only include signal_path if model supports it
            if not hasattr(self.model, 'get_signal_path'):
                methods = ['integrated_gradients']

        explanations = {}

        for method in methods:
            try:
                explanation = self.explain(input_data, target_class, method=method, **kwargs)
                explanations[method] = explanation
            except Exception as e:
                print(f"Warning: Could not generate explanation with {method}: {e}")
                explanations[method] = None

        return explanations

    def get_available_methods(self) -> Dict[str, str]:
        """
        Get information about available explanation methods.

        Returns:
            Dictionary mapping method names to descriptions
        """
        methods = {
            'signal_path': 'Intrinsic explanation tracking signal transformations through model layers',
            'integrated_gradients': 'Post-hoc explanation using Integrated Gradients (Captum)',
            'deeplift': 'Post-hoc explanation using DeepLift (Captum)',
            'saliency': 'Post-hoc explanation using gradient-based saliency (Captum)',
            'auto': 'Automatically select the best method based on model capabilities'
        }

        # Check model-specific capabilities
        model_info = {}
        if hasattr(self.model, 'get_signal_path'):
            model_info['supports_signal_path'] = True
        else:
            model_info['supports_signal_path'] = False

        # Add model info to descriptions
        if model_info.get('supports_signal_path', False):
            methods['signal_path'] += ' (Available for this model)'
        else:
            methods['signal_path'] += ' (Not available for this model)'

        return methods

    def get_model_explainability_info(self) -> Dict[str, Any]:
        """
        Get information about the explainability capabilities of the model.

        Returns:
            Dictionary containing model explainability information
        """
        info = {
            'model_type': type(self.model).__name__,
            'current_method': self.method,
            'supported_methods': []
        }

        # Check which methods are supported
        if hasattr(self.model, 'get_signal_path'):
            info['supported_methods'].append('signal_path')

        # All models support post-hoc methods (they work on any model)
        info['supported_methods'].extend(['integrated_gradients', 'deeplift', 'saliency'])

        # Try to get more detailed info if available
        if hasattr(self.model, 'get_explainability_info'):
            info.update(self.model.get_explainability_info())

        return info

    def __repr__(self) -> str:
        """String representation of the unified explainer."""
        return f"UnifiedExplainer(model={type(self.model).__name__}, method='{self.method}')"

    @staticmethod
    def create_explainer(model: torch.nn.Module,
                        method: str = 'auto',
                        **config_kwargs) -> 'UnifiedExplainer':
        """
        Factory method to create an explainer with simplified interface.

        Args:
            model: The model to explain
            method: Explanation method
            **config_kwargs: Configuration options

        Returns:
            UnifiedExplainer instance
        """
        return UnifiedExplainer(model, config_kwargs, method)


# Convenience function for quick usage
def explain_model(model: torch.nn.Module,
                  input_data: torch.Tensor,
                  method: str = 'auto',
                  target_class: Optional[int] = None,
                  **kwargs) -> Explanation:
    """
    Quick function to generate model explanation.

    Args:
        model: The model to explain
        input_data: Input tensor
        method: Explanation method
        target_class: Target class for explanation
        **kwargs: Additional configuration

    Returns:
        Explanation object
    """
    explainer = UnifiedExplainer(model, kwargs, method)
    return explainer.explain(input_data, target_class)