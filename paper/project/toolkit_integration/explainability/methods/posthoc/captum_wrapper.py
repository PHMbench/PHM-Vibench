"""
Captum Wrapper for Post-hoc Explanations

Provides a unified interface to Captum's attribution methods, specifically
focusing on Integrated Gradients for fault diagnosis applications.
"""

from typing import Dict, Any, Optional, Union, List
import torch
import torch.nn as nn
import numpy as np

try:
    from captum.attr import IntegratedGradients, DeepLift, Saliency
    from captum.attr import visualization as viz
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("Warning: Captum not available. Install with: pip install captum")

from ...core.base_explainer import BaseExplainer
from ...core.explanation import Explanation


class CaptumWrapper(BaseExplainer):
    """
    Wrapper for Captum attribution methods.

    This class provides a unified interface to various Captum attribution methods,
    with special focus on Integrated Gradients for time series fault diagnosis.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(model, config)

        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum is required for CaptumWrapper. Install with: pip install captum")

        # Configuration
        self.method = self.config.get('method', 'integrated_gradients')  # integrated_gradients, deeplift, saliency
        self.n_steps = self.config.get('n_steps', 50)
        self.internal_batch_size = self.config.get('internal_batch_size', 10)
        self.baseline = self.config.get('baseline', None)
        self.return_convergence_delta = self.config.get('return_convergence_delta', False)

        # Initialize the specific attribution method
        self._initialize_attribution_method()

    def _initialize_attribution_method(self):
        """Initialize the specific Captum attribution method."""
        if self.method == 'integrated_gradients':
            self.attribution_method = IntegratedGradients(
                self.model,
                n_steps=self.n_steps,
                internal_batch_size=self.internal_batch_size
            )
        elif self.method == 'deeplift':
            self.attribution_method = DeepLift(self.model)
        elif self.method == 'saliency':
            self.attribution_method = Saliency(self.model)
        else:
            raise ValueError(f"Unknown Captum method: {self.method}")

    def explain(self,
                input_data: torch.Tensor,
                target_class: Optional[int] = None,
                **kwargs) -> Explanation:
        """
        Generate explanation using the specified Captum method.

        Args:
            input_data: Input tensor [batch_size, sequence_length, channels]
            target_class: Target class for explanation
            **kwargs: Additional arguments

        Returns:
            Explanation object containing attribution results
        """
        self._validate_input(input_data)

        # Get target class
        target = self._get_target_class(input_data, target_class)

        # Handle batch dimension
        if len(input_data.shape) == 3:
            # [batch, seq, channels] -> [batch, channels, seq] for Captum
            input_data_captum = input_data.transpose(1, 2)
        else:
            input_data_captum = input_data

        # Create baseline if not provided
        baseline = self._create_baseline(input_data_captum)

        # Compute attributions
        with torch.no_grad():
            attributions = self.attribution_method.attribute(
                input_data_captum,
                target=target,
                baselines=baseline,
                return_convergence_delta=self.return_convergence_delta
            )

        # Reshape attributions back to original format
        if len(input_data.shape) == 3 and len(attributions.shape) == 3:
            attributions = attributions.transpose(1, 2)

        # Create explanation data
        explanation_data = {
            'attributions': attributions,
            'original_signal': input_data,
            'target_class': target,
            'method': self.method,
            'baseline': baseline
        }

        # Add method-specific information
        if self.method == 'integrated_gradients':
            explanation_data['n_steps'] = self.n_steps
        elif self.return_convergence_delta and isinstance(attributions, tuple):
            explanation_data['attributions'] = attributions[0]
            explanation_data['convergence_delta'] = attributions[1]

        # Create metadata
        metadata = {
            'method': f'captum_{self.method}',
            'model_name': type(self.model).__name__,
            'input_shape': list(input_data.shape),
            'target_class': target,
            'config': self.config
        }

        return Explanation(explanation_data, metadata)

    def _create_baseline(self, input_data: torch.Tensor) -> Union[torch.Tensor, None]:
        """
        Create baseline for attribution computation.

        For time series, common baselines include:
        - Zero baseline (default)
        - Noise baseline
        - Mean signal
        - Random baseline
        """
        if self.baseline is not None:
            if isinstance(self.baseline, str):
                return self._create_baseline_from_type(input_data, self.baseline)
            else:
                # Assume it's a tensor
                return self.baseline

        # Default: zero baseline
        return torch.zeros_like(input_data)

    def _create_baseline_from_type(self, input_data: torch.Tensor, baseline_type: str) -> torch.Tensor:
        """Create baseline from specified type."""
        if baseline_type == 'zero':
            return torch.zeros_like(input_data)
        elif baseline_type == 'noise':
            return torch.randn_like(input_data) * 0.1
        elif baseline_type == 'mean':
            return torch.full_like(input_data, torch.mean(input_data))
        elif baseline_type == 'random':
            # Random signal from normal distribution
            return torch.randn_like(input_data) * torch.std(input_data)
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")

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
        explanations = []

        if target_classes is None:
            # Get model predictions for all samples
            with torch.no_grad():
                predictions = self.model(input_data)
                target_classes = torch.argmax(predictions, dim=-1).tolist()

        # Generate explanations for each sample
        for i in range(input_data.shape[0]):
            sample_input = input_data[i:i+1]  # Keep batch dimension
            sample_target = target_classes[i] if isinstance(target_classes, list) else target_classes

            explanation = self.explain(sample_input, sample_target, **kwargs)
            explanations.append(explanation)

        return explanations

    def compare_methods(self,
                        input_data: torch.Tensor,
                        target_class: Optional[int] = None,
                        methods: Optional[List[str]] = None) -> Dict[str, Explanation]:
        """
        Compare multiple attribution methods on the same input.

        Args:
            input_data: Input tensor
            target_class: Target class for explanation
            methods: List of methods to compare ['integrated_gradients', 'deeplift', 'saliency']

        Returns:
            Dictionary mapping method names to Explanation objects
        """
        if methods is None:
            methods = ['integrated_gradients', 'deeplift', 'saliency']

        explanations = {}
        original_method = self.method

        for method in methods:
            try:
                # Temporarily change method
                self.method = method
                self._initialize_attribution_method()

                # Generate explanation
                explanation = self.explain(input_data, target_class)
                explanations[method] = explanation

            except Exception as e:
                print(f"Warning: Could not compute {method} explanation: {e}")
                explanations[method] = None

        # Restore original method
        self.method = original_method
        self._initialize_attribution_method()

        return explanations

    def get_attribution_statistics(self, explanation: Explanation) -> Dict[str, float]:
        """
        Compute statistics for attribution values.

        Args:
            explanation: Explanation object containing attributions

        Returns:
            Dictionary of attribution statistics
        """
        attributions = explanation.get_data('attributions')
        if attributions is None:
            return {}

        if isinstance(attributions, torch.Tensor):
            attributions_np = attributions.detach().cpu().numpy()
        else:
            attributions_np = np.array(attributions)

        return {
            'mean_attribution': float(np.mean(attributions_np)),
            'std_attribution': float(np.std(attributions_np)),
            'max_attribution': float(np.max(attributions_np)),
            'min_attribution': float(np.min(attributions_np)),
            'positive_ratio': float(np.mean(attributions_np > 0)),
            'sparsity': float(np.mean(np.abs(attributions_np) < 0.01)),
            'total_attribution': float(np.sum(np.abs(attributions_np)))
        }

    def __repr__(self) -> str:
        """String representation of the Captum wrapper."""
        return f"CaptumWrapper(method='{self.method}', model={type(self.model).__name__})"