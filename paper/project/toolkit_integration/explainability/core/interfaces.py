"""
Core Interface Protocols for Explainable FD Toolkit

This module defines the standardized interface protocols that all explainability methods
and model plugins should implement for consistency and interoperability.
"""

from typing import Protocol, Dict, Any, Optional, Sequence, List, Union, runtime_checkable
import numpy as np
import torch
from matplotlib.figure import Figure

from .signal_data import SignalData
from .explanation import Explanation


@runtime_checkable
class ExplainabilityMethod(Protocol):
    """
    统一的可解释性方法接口

    所有具体方法（本征/事后）都应实现这三类能力：
    - explain：生成解释
    - visualize：可视化单个解释
    - evaluate：对一批解释做指标评估
    """

    def explain(self,
                signal: SignalData,
                prediction: Any,
                **kwargs) -> Explanation:
        """
        Generate explanation for a given signal and prediction.

        Args:
            signal: SignalData object containing the signal to explain
            prediction: Model prediction or target for explanation
            **kwargs: Additional method-specific arguments

        Returns:
            Explanation object containing the explanation results
        """
        ...

    def visualize(self,
                  explanation: Explanation,
                  mode: str = 'auto',
                  **kwargs) -> Figure:
        """
        Create visualization for a single explanation.

        Args:
            explanation: Explanation object to visualize
            mode: Visualization mode ('auto', 'attribution', 'path', 'importance', etc.)
            **kwargs: Additional visualization parameters

        Returns:
            Matplotlib figure object
        """
        ...

    def evaluate(self,
                 explanations: Sequence[Explanation],
                 ground_truth: Optional[Sequence[Any]] = None,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate explanation quality across multiple explanations.

        Args:
            explanations: Sequence of explanation objects to evaluate
            ground_truth: Optional ground truth explanations for evaluation
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary of evaluation metrics (coverage, stability, consistency, etc.)
        """
        ...

    def get_method_name(self) -> str:
        """
        Get the name of this explanation method.

        Returns:
            Method name string
        """
        ...

    def get_method_type(self) -> str:
        """
        Get the type of explanation method.

        Returns:
            One of: 'intrinsic', 'posthoc', 'hybrid'
        """
        ...

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of this method.

        Returns:
            Configuration dictionary
        """
        ...

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Update the configuration of this method.

        Args:
            config: New configuration parameters
        """
        ...


@runtime_checkable
class ModelPlugin(Protocol):
    """
    模型插件接口：使任意模型都能接入工具集

    This protocol defines the interface that model wrappers should implement
    to be compatible with the Explainable FD Toolkit.
    """

    def fit(self, data: Sequence[SignalData], labels: Sequence[Any], **kwargs) -> None:
        """
        Train the model on the given data.

        Args:
            data: Sequence of SignalData objects for training
            labels: Corresponding labels/targets
            **kwargs: Additional training parameters
        """
        ...

    def predict(self, signal: SignalData, **kwargs) -> Any:
        """
        Make prediction for a single signal.

        Args:
            signal: SignalData object to predict
            **kwargs: Additional prediction parameters

        Returns:
            Model prediction
        """
        ...

    def predict_batch(self, signals: Sequence[SignalData], **kwargs) -> List[Any]:
        """
        Make predictions for a batch of signals.

        Args:
            signals: Sequence of SignalData objects to predict
            **kwargs: Additional prediction parameters

        Returns:
            List of predictions
        """
        ...

    def get_explanation(self,
                       signal: SignalData,
                       method: ExplainabilityMethod,
                       **kwargs) -> Explanation:
        """
        Generate explanation for a signal using the specified method.

        Args:
            signal: SignalData object to explain
            method: Explanation method to use
            **kwargs: Additional explanation parameters

        Returns:
            Explanation object
        """
        ...

    def get_intermediate_features(self,
                                 signal: SignalData,
                                 layer_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Extract intermediate features from the model.

        Args:
            signal: Input signal
            layer_names: Specific layer names to extract (None for all available)

        Returns:
            Dictionary mapping layer names to feature arrays
        """
        ...

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary containing model metadata
        """
        ...

    def save_model(self, filepath: str) -> None:
        """
        Save the model to disk.

        Args:
            filepath: Path to save the model
        """
        ...

    def load_model(self, filepath: str) -> None:
        """
        Load the model from disk.

        Args:
            filepath: Path to load the model from
        """
        ...

    def get_supported_methods(self) -> List[str]:
        """
        Get list of explanation methods supported by this model.

        Returns:
            List of method names
        """
        ...


class BaseExplainerAdapter:
    """
    Base adapter class for implementing ExplainabilityMethod protocol.

    This class provides common functionality and default implementations
    for explanation methods.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the explainer adapter.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._method_name = self.__class__.__name__
        self._method_type = 'posthoc'  # Default method type

    def get_method_name(self) -> str:
        """Get the name of this explanation method."""
        return self._method_name

    def get_method_type(self) -> str:
        """Get the type of explanation method."""
        return self._method_type

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of this method."""
        return self.config.copy()

    def set_config(self, config: Dict[str, Any]) -> None:
        """Update the configuration of this method."""
        self.config.update(config)

    def _validate_signal(self, signal: SignalData) -> None:
        """Validate input signal data."""
        if not isinstance(signal, SignalData):
            raise TypeError("Input must be a SignalData object")

    def _validate_explanation(self, explanation: Explanation) -> None:
        """Validate explanation object."""
        if not isinstance(explanation, Explanation):
            raise TypeError("Explanation must be an Explanation object")

    def __repr__(self) -> str:
        """String representation of the explainer."""
        return f"{self.__class__.__name__}(name='{self.get_method_name()}', type='{self.get_method_type()}')"


class BaseModelAdapter:
    """
    Base adapter class for implementing ModelPlugin protocol.

    This class provides common functionality and default implementations
    for model plugins.
    """

    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model adapter.

        Args:
            model: The underlying model to wrap
            config: Configuration dictionary
        """
        self.model = model
        self.config = config or {}
        self._model_name = type(model).__name__

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            'model_name': self._model_name,
            'model_type': type(self.model).__name__,
            'config': self.config,
            'supported_methods': self.get_supported_methods()
        }

    def get_supported_methods(self) -> List[str]:
        """Get list of explanation methods supported by this model."""
        return ['grad_cam', 'lime', 'shap']  # Default supported methods

    def _validate_signal(self, signal: SignalData) -> None:
        """Validate input signal data."""
        if not isinstance(signal, SignalData):
            raise TypeError("Input must be a SignalData object")

    def _prepare_input(self, signal: SignalData) -> torch.Tensor:
        """
        Prepare signal data for model input.

        Args:
            signal: SignalData object

        Returns:
            Tensor in the format expected by the model
        """
        # Default implementation - should be overridden by subclasses
        return torch.FloatTensor(signal.raw_signal)

    def __repr__(self) -> str:
        """String representation of the model adapter."""
        return f"{self.__class__.__name__}(model='{self._model_name}', methods={self.get_supported_methods()})"