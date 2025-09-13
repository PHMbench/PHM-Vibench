"""
Improved Factory Pattern Implementation for PHM-Vibench

This module provides a unified, extensible factory system with proper error handling,
validation, and scientific reproducibility features.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
import torch.nn as nn
import pytorch_lightning as pl
from dataclasses import dataclass

# Type variables for generic factory
T = TypeVar('T')
ModelType = TypeVar('ModelType', bound=nn.Module)
TaskType = TypeVar('TaskType', bound=pl.LightningModule)

logger = logging.getLogger(__name__)


@dataclass
class ComponentInfo:
    """Information about a registered component."""
    name: str
    component_type: str
    class_ref: Type
    description: str
    parameters: Dict[str, Any]
    paper_reference: Optional[str] = None
    implementation_notes: Optional[str] = None


class FactoryRegistry:
    """
    Unified registry for all factory components with validation and introspection.

    This registry provides:
    - Type-safe component registration
    - Automatic parameter validation
    - Component discovery and listing
    - Detailed error messages
    """

    def __init__(self, name: str):
        self.name = name
        self._components: Dict[str, ComponentInfo] = {}
        self._aliases: Dict[str, str] = {}

    def register(
        self,
        name: str,
        component_type: str,
        description: str = "",
        paper_reference: Optional[str] = None,
        implementation_notes: Optional[str] = None,
        aliases: Optional[List[str]] = None
    ) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register a component with the factory.

        Parameters
        ----------
        name : str
            Unique component name
        component_type : str
            Component type/category
        description : str
            Human-readable description
        paper_reference : Optional[str]
            Reference to original paper
        implementation_notes : Optional[str]
            Implementation-specific notes
        aliases : Optional[List[str]]
            Alternative names for the component

        Returns
        -------
        Callable
            Decorator function
        """
        def decorator(cls: Type[T]) -> Type[T]:
            # Extract parameter information from class
            parameters = self._extract_parameters(cls)

            # Create component info
            component_info = ComponentInfo(
                name=name,
                component_type=component_type,
                class_ref=cls,
                description=description,
                parameters=parameters,
                paper_reference=paper_reference,
                implementation_notes=implementation_notes
            )

            # Register component
            full_name = f"{component_type}.{name}"
            if full_name in self._components:
                logger.warning(f"Overriding existing component: {full_name}")

            self._components[full_name] = component_info

            # Register aliases
            if aliases:
                for alias in aliases:
                    alias_key = f"{component_type}.{alias}"
                    self._aliases[alias_key] = full_name

            logger.debug(f"Registered {self.name} component: {full_name}")
            return cls

        return decorator

    def get(self, key: str) -> ComponentInfo:
        """
        Get component information by key.

        Parameters
        ----------
        key : str
            Component key in format "type.name"

        Returns
        -------
        ComponentInfo
            Component information

        Raises
        ------
        KeyError
            If component not found
        """
        # Check aliases first
        if key in self._aliases:
            key = self._aliases[key]

        if key not in self._components:
            available = list(self._components.keys())
            raise KeyError(
                f"Component '{key}' not found in {self.name} registry. "
                f"Available components: {available}"
            )

        return self._components[key]

    def list_components(self, component_type: Optional[str] = None) -> List[ComponentInfo]:
        """
        List all registered components, optionally filtered by type.

        Parameters
        ----------
        component_type : Optional[str]
            Filter by component type

        Returns
        -------
        List[ComponentInfo]
            List of component information
        """
        components = list(self._components.values())

        if component_type:
            components = [c for c in components if c.component_type == component_type]

        return sorted(components, key=lambda x: (x.component_type, x.name))

    def _extract_parameters(self, cls: Type) -> Dict[str, Any]:
        """Extract parameter information from class constructor."""
        try:
            sig = inspect.signature(cls.__init__)
            parameters = {}

            for param_name, param in sig.parameters.items():
                if param_name in ('self', 'args', 'kwargs'):
                    continue

                param_info = {
                    'type': param.annotation if param.annotation != inspect.Parameter.empty else 'Any',
                    'default': param.default if param.default != inspect.Parameter.empty else None,
                    'required': param.default == inspect.Parameter.empty
                }
                parameters[param_name] = param_info

            return parameters
        except Exception as e:
            logger.warning(f"Could not extract parameters from {cls}: {e}")
            return {}


class BaseFactory(ABC):
    """
    Abstract base class for all factories.

    Provides common functionality for component instantiation with
    proper error handling and validation.
    """

    def __init__(self, registry: FactoryRegistry):
        self.registry = registry
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    @abstractmethod
    def create(self, config: Any, **kwargs) -> Any:
        """Create component instance from configuration."""
        pass

    def _validate_config(self, config: Any, required_fields: List[str]) -> None:
        """Validate that configuration has required fields."""
        for field in required_fields:
            if not hasattr(config, field):
                raise ValueError(f"Configuration missing required field: {field}")

    def _create_instance(
        self,
        component_info: ComponentInfo,
        config: Any,
        **kwargs
    ) -> Any:
        """Create instance with proper error handling."""
        try:
            # Log creation attempt
            self.logger.debug(f"Creating {component_info.name} instance")

            # Create instance
            instance = component_info.class_ref(config, **kwargs)

            # Log success
            self.logger.info(f"Successfully created {component_info.name}")

            return instance

        except Exception as e:
            error_msg = (
                f"Failed to create {component_info.name}: {str(e)}\n"
                f"Component type: {component_info.component_type}\n"
                f"Description: {component_info.description}\n"
                f"Required parameters: {component_info.parameters}"
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e


class ModelFactory(BaseFactory):
    """
    Factory for creating neural network models.

    Supports automatic parameter inference from metadata and
    proper model initialization with reproducible weights.
    """

    def create(self, config: Any, metadata: Optional[Any] = None) -> nn.Module:
        """
        Create model instance from configuration.

        Parameters
        ----------
        config : ModelConfig
            Model configuration
        metadata : Optional[Any]
            Dataset metadata for automatic parameter inference

        Returns
        -------
        nn.Module
            Instantiated model
        """
        # Validate configuration
        self._validate_config(config, ['name', 'type'])

        # Get component info
        key = f"{config.type}.{config.name}"
        component_info = self.registry.get(key)

        # Enhance config with metadata if available
        enhanced_config = self._enhance_config_with_metadata(config, metadata)

        # Create model instance
        model = self._create_instance(component_info, enhanced_config, metadata=metadata)

        # Initialize weights if specified
        if hasattr(enhanced_config, 'weight_init'):
            self._initialize_weights(model, enhanced_config)

        return model

    def _enhance_config_with_metadata(self, config: Any, metadata: Optional[Any]) -> Any:
        """Enhance configuration with metadata-derived parameters."""
        if metadata is None:
            return config

        # Create a copy to avoid modifying original
        enhanced_config = type(config)(**config.__dict__)

        # Auto-infer num_classes if not specified
        if hasattr(enhanced_config, 'num_classes') and enhanced_config.num_classes is None:
            if hasattr(metadata, 'num_classes'):
                enhanced_config.num_classes = metadata.num_classes
                self.logger.info(f"Auto-inferred num_classes: {metadata.num_classes}")

        return enhanced_config

    def _initialize_weights(self, model: nn.Module, config: Any) -> None:
        """Initialize model weights according to configuration."""
        init_method = getattr(config, 'weight_init', 'xavier_uniform')

        def init_fn(m):
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if init_method == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif init_method == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                elif init_method == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif init_method == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

                if m.bias is not None:
                    bias_init = getattr(config, 'bias_init', 'zeros')
                    if bias_init == 'zeros':
                        nn.init.zeros_(m.bias)
                    elif bias_init == 'uniform':
                        nn.init.uniform_(m.bias, -0.1, 0.1)

        model.apply(init_fn)
        self.logger.info(f"Initialized weights using {init_method}")


class TaskFactory(BaseFactory):
    """
    Factory for creating task modules (Lightning modules).

    Handles complex task configurations and ensures proper
    integration between models, data, and training parameters.
    """

    def create(
        self,
        config: Any,
        model: nn.Module,
        metadata: Optional[Any] = None,
        **kwargs
    ) -> pl.LightningModule:
        """
        Create task instance from configuration.

        Parameters
        ----------
        config : TaskConfig
            Task configuration
        model : nn.Module
            Model instance to wrap
        metadata : Optional[Any]
            Dataset metadata
        **kwargs
            Additional arguments for task creation

        Returns
        -------
        pl.LightningModule
            Instantiated task module
        """
        # Validate configuration
        self._validate_config(config, ['name', 'type'])

        # Get component info
        key = f"{config.type}.{config.name}"
        component_info = self.registry.get(key)

        # Create task instance
        task = self._create_instance(
            component_info,
            config,
            model=model,
            metadata=metadata,
            **kwargs
        )

        return task


# Global registries
MODEL_REGISTRY = FactoryRegistry("Model")
TASK_REGISTRY = FactoryRegistry("Task")
DATA_REGISTRY = FactoryRegistry("Data")

# Global factories
model_factory = ModelFactory(MODEL_REGISTRY)
task_factory = TaskFactory(TASK_REGISTRY)


# Convenience decorators
def register_model(
    name: str,
    model_type: str,
    description: str = "",
    paper_reference: Optional[str] = None,
    aliases: Optional[List[str]] = None
):
    """Decorator to register a model with the global model registry."""
    return MODEL_REGISTRY.register(
        name=name,
        component_type=model_type,
        description=description,
        paper_reference=paper_reference,
        aliases=aliases
    )


def register_task(
    name: str,
    task_type: str,
    description: str = "",
    paper_reference: Optional[str] = None,
    aliases: Optional[List[str]] = None
):
    """Decorator to register a task with the global task registry."""
    return TASK_REGISTRY.register(
        name=name,
        component_type=task_type,
        description=description,
        paper_reference=paper_reference,
        aliases=aliases
    )


if __name__ == "__main__":
    # Example usage and testing

    # Mock model class for testing
    class MockResNet1D(nn.Module):
        def __init__(self, config, metadata=None):
            super().__init__()
            self.config = config
            self.linear = nn.Linear(config.input_dim, config.num_classes)

        def forward(self, x):
            return self.linear(x.mean(dim=1))

    # Register the mock model
    @register_model(
        name="ResNet1D",
        model_type="CNN",
        description="1D ResNet for time-series classification",
        paper_reference="He et al. Deep Residual Learning for Image Recognition, CVPR 2016"
    )
    class RegisteredResNet1D(MockResNet1D):
        pass

    # Test configuration
    from types import SimpleNamespace

    config = SimpleNamespace(
        name="ResNet1D",
        type="CNN",
        input_dim=3,
        num_classes=10,
        weight_init="xavier_uniform"
    )

    # Test model creation
    try:
        model = model_factory.create(config)
        print("✅ Model factory test passed!")
        print(f"Created model: {type(model).__name__}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

        # Test component listing
        components = MODEL_REGISTRY.list_components("CNN")
        print(f"Available CNN models: {[c.name for c in components]}")

    except Exception as e:
        print(f"❌ Model factory test failed: {e}")
        raise