"""
Base Explainer Abstract Class

Provides the common interface that all explainers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import torch
from .explanation import Explanation


class BaseExplainer(ABC):
    """
    Abstract base class for all explainers in the toolkit.

    All explainers must inherit from this class and implement the explain method.
    This ensures consistent interface across different explanation methods.
    """

    def __init__(self, model: torch.nn.Module, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the explainer.

        Args:
            model: The model to explain
            config: Configuration dictionary for the explainer
        """
        self.model = model
        self.config = config or {}
        self.model.eval()  # Set model to evaluation mode

    @abstractmethod
    def explain(self,
                input_data: torch.Tensor,
                target_class: Optional[int] = None,
                **kwargs) -> Explanation:
        """
        Generate explanation for the given input.

        Args:
            input_data: Input tensor to explain [batch_size, sequence_length, channels]
            target_class: Target class for explanation (for multi-class classification)
            **kwargs: Additional method-specific arguments

        Returns:
            Explanation object containing the explanation results
        """
        pass

    def _validate_input(self, input_data: torch.Tensor) -> None:
        """Validate input tensor format."""
        if not isinstance(input_data, torch.Tensor):
            raise TypeError("Input data must be a torch.Tensor")

        if len(input_data.shape) < 2:
            raise ValueError("Input data must have at least 2 dimensions [batch, sequence]")

    def _get_model_predictions(self, input_data: torch.Tensor) -> torch.Tensor:
        """Get model predictions for the input data."""
        with torch.no_grad():
            return self.model(input_data)

    def _get_target_class(self,
                         input_data: torch.Tensor,
                         target_class: Optional[int] = None) -> int:
        """
        Get the target class for explanation.

        If target_class is None, uses the model's prediction.
        """
        if target_class is not None:
            return target_class

        predictions = self._get_model_predictions(input_data)
        if len(predictions.shape) == 1:  # Single sample
            return torch.argmax(predictions).item()
        else:  # Batch
            return torch.argmax(predictions[0]).item()

    def __repr__(self) -> str:
        """String representation of the explainer."""
        return f"{self.__class__.__name__}(model={type(self.model).__name__})"