"""
Configuration Management for Explainability Methods

This module provides standardized configuration templates and management
for all explanation methods in the Explainable FD Toolkit.
"""

from typing import Dict, Any, Optional, Union, List
import yaml
from pathlib import Path
import json


# Default configuration templates for each method
DEFAULT_CONFIGS = {
    'PathAnalysis': {
        'include_frequency_analysis': True,
        'include_energy_analysis': True,
        'include_statistical_analysis': True,
        'sampling_rate': 1024.0,
        'physical_interpretations': {},
        'max_path_depth': 10,
        'importance_threshold': 0.1,
        'visualization_modes': ['auto', 'path', 'importance', 'energy', 'frequency']
    },

    'OperatorWeight': {
        'include_weight_magnitude': True,
        'include_weight_gradients': False,
        'include_activation_patterns': True,
        'weight_analysis_method': 'magnitude',  # 'magnitude', 'variance', 'spectral'
        'layer_importance_threshold': 0.05,
        'top_k_operators': 10,
        'normalize_weights': True,
        'visualization_modes': ['auto', 'weights', 'importance', 'activations', 'comparison']
    },

    'GradCAM': {
        'target_layers': [],  # Auto-detect if empty
        'use_abs_gradients': True,
        'normalize_attributions': True,
        'interpolation_method': 'linear',  # 'linear', 'nearest', 'bilinear'
        'attribution_smoothing': True,
        'smoothing_kernel': 5,
        'visualization_modes': ['auto', 'heatmap', 'overlay', 'importance']
    },

    'SHAP': {
        'explanation_method': 'gradient',  # 'gradient', 'kernel', 'deep'
        'background_samples': 10,
        'n_segments': 50,
        'use_segments': True,
        'normalize_shap_values': True,
        'aggregate_channels': True,
        'visualization_modes': ['auto', 'values', 'features', 'segments', 'waterfall']
    }
}

# Method types and categories
METHOD_CATEGORIES = {
    'intrinsic': ['PathAnalysis', 'OperatorWeight'],
    'posthoc': ['GradCAM', 'SHAP']
}

# Method class mapping
METHOD_CLASSES = {
    'PathAnalysis': 'PathAnalysisExplainer',
    'OperatorWeight': 'OperatorWeightExplainer',
    'GradCAM': 'GradCAMExplainer',
    'SHAP': 'SHAPExplainer'
}


class MethodConfigManager:
    """
    Configuration manager for explainability methods.

    This class provides a unified interface for managing configurations
    of different explanation methods, including loading, saving, and
    validation of configurations.
    """

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else None
        self._configs = {}
        self._load_default_configs()

    def _load_default_configs(self) -> None:
        """Load default configurations."""
        for method_name, config in DEFAULT_CONFIGS.items():
            self._configs[method_name] = config.copy()

    def get_config(self, method_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific method.

        Args:
            method_name: Name of the explanation method

        Returns:
            Configuration dictionary
        """
        if method_name not in self._configs:
            raise ValueError(f"Unknown method: {method_name}. Available methods: {list(self._configs.keys())}")

        return self._configs[method_name].copy()

    def set_config(self, method_name: str, config: Dict[str, Any]) -> None:
        """
        Set configuration for a specific method.

        Args:
            method_name: Name of the explanation method
            config: Configuration dictionary
        """
        if method_name not in DEFAULT_CONFIGS:
            raise ValueError(f"Unknown method: {method_name}")

        # Validate configuration
        self._validate_config(method_name, config)

        self._configs[method_name] = config.copy()

    def update_config(self, method_name: str, updates: Dict[str, Any]) -> None:
        """
        Update configuration for a specific method.

        Args:
            method_name: Name of the explanation method
            updates: Configuration updates to apply
        """
        current_config = self.get_config(method_name)
        current_config.update(updates)
        self.set_config(method_name, current_config)

    def get_default_config(self, method_name: str) -> Dict[str, Any]:
        """
        Get default configuration for a specific method.

        Args:
            method_name: Name of the explanation method

        Returns:
            Default configuration dictionary
        """
        if method_name not in DEFAULT_CONFIGS:
            raise ValueError(f"Unknown method: {method_name}")

        return DEFAULT_CONFIGS[method_name].copy()

    def reset_config(self, method_name: str) -> None:
        """
        Reset configuration to default values.

        Args:
            method_name: Name of the explanation method
        """
        self.set_config(method_name, self.get_default_config(method_name))

    def load_config_from_file(self, method_name: str, filepath: Union[str, Path]) -> None:
        """
        Load configuration from a file.

        Args:
            method_name: Name of the explanation method
            filepath: Path to configuration file
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        if filepath.suffix.lower() in ['.yml', '.yaml']:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {filepath.suffix}")

        self.set_config(method_name, config)

    def save_config_to_file(self, method_name: str, filepath: Union[str, Path]) -> None:
        """
        Save configuration to a file.

        Args:
            method_name: Name of the explanation method
            filepath: Path to save configuration file
        """
        config = self.get_config(method_name)
        filepath = Path(filepath)

        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if filepath.suffix.lower() in ['.yml', '.yaml']:
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {filepath.suffix}")

    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all method configurations.

        Returns:
            Dictionary mapping method names to configurations
        """
        return {name: config.copy() for name, config in self._configs.items()}

    def get_method_category(self, method_name: str) -> str:
        """
        Get the category of a method.

        Args:
            method_name: Name of the explanation method

        Returns:
            Method category ('intrinsic' or 'posthoc')
        """
        for category, methods in METHOD_CATEGORIES.items():
            if method_name in methods:
                return category
        raise ValueError(f"Unknown method: {method_name}")

    def get_methods_by_category(self, category: str) -> List[str]:
        """
        Get all methods in a specific category.

        Args:
            category: Method category ('intrinsic' or 'posthoc')

        Returns:
            List of method names
        """
        if category not in METHOD_CATEGORIES:
            raise ValueError(f"Unknown category: {category}. Available categories: {list(METHOD_CATEGORIES.keys())}")

        return METHOD_CATEGORIES[category].copy()

    def get_available_methods(self) -> List[str]:
        """
        Get list of all available methods.

        Returns:
            List of method names
        """
        return list(self._configs.keys())

    def validate_method_name(self, method_name: str) -> bool:
        """
        Validate if a method name is supported.

        Args:
            method_name: Name of the explanation method

        Returns:
            True if method is supported, False otherwise
        """
        return method_name in self._configs

    def _validate_config(self, method_name: str, config: Dict[str, Any]) -> None:
        """
        Validate configuration for a method.

        Args:
            method_name: Name of the explanation method
            config: Configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        default_config = DEFAULT_CONFIGS.get(method_name, {})

        # Check for required keys
        for key, default_value in default_config.items():
            if key not in config:
                print(f"Warning: Missing key '{key}' in {method_name} configuration, using default value")
                config[key] = default_value
            else:
                # Type validation
                if not isinstance(config[key], type(default_value)):
                    try:
                        # Attempt type conversion
                        config[key] = type(default_value)(config[key])
                    except (ValueError, TypeError):
                        raise ValueError(f"Invalid type for key '{key}' in {method_name} configuration. "
                                       f"Expected {type(default_value).__name__}, got {type(config[key]).__name__}")

    def create_experiment_config(self,
                                method_names: List[str],
                                experiment_name: str,
                                **common_params) -> Dict[str, Any]:
        """
        Create configuration for multiple methods in an experiment.

        Args:
            method_names: List of method names to include
            experiment_name: Name of the experiment
            **common_params: Common parameters to apply to all methods

        Returns:
            Experiment configuration dictionary
        """
        experiment_config = {
            'experiment_name': experiment_name,
            'methods': {},
            'common_parameters': common_params
        }

        for method_name in method_names:
            if not self.validate_method_name(method_name):
                raise ValueError(f"Unknown method: {method_name}")

            method_config = self.get_config(method_name)
            method_config.update(common_params)
            experiment_config['methods'][method_name] = method_config

        return experiment_config

    def save_experiment_config(self,
                              experiment_config: Dict[str, Any],
                              filepath: Union[str, Path]) -> None:
        """
        Save experiment configuration to file.

        Args:
            experiment_config: Experiment configuration dictionary
            filepath: Path to save configuration file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if filepath.suffix.lower() in ['.yml', '.yaml']:
            with open(filepath, 'w') as f:
                yaml.dump(experiment_config, f, default_flow_style=False, indent=2)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'w') as f:
                json.dump(experiment_config, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {filepath.suffix}")

    @classmethod
    def get_method_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available methods.

        Returns:
            Dictionary mapping method names to method information
        """
        method_info = {}

        for method_name, config in DEFAULT_CONFIGS.items():
            category = None
            for cat, methods in METHOD_CATEGORIES.items():
                if method_name in methods:
                    category = cat
                    break

            method_info[method_name] = {
                'class_name': METHOD_CLASSES.get(method_name, 'Unknown'),
                'category': category,
                'description': cls._get_method_description(method_name),
                'config_keys': list(config.keys()),
                'visualization_modes': config.get('visualization_modes', ['auto'])
            }

        return method_info

    @staticmethod
    def _get_method_description(method_name: str) -> str:
        """Get description for a method."""
        descriptions = {
            'PathAnalysis': 'Tracks signal transformations through model layers to provide path-level explanations.',
            'OperatorWeight': 'Analyzes operator weights and parameters to identify influential components.',
            'GradCAM': 'Generates attribution maps using gradient-weighted class activation mapping.',
            'SHAP': 'Computes feature attributions using game-theoretic SHAP values.'
        }
        return descriptions.get(method_name, 'Unknown method')


# Global configuration manager instance
config_manager = MethodConfigManager()


# Convenience functions
def get_method_config(method_name: str) -> Dict[str, Any]:
    """Get configuration for a method."""
    return config_manager.get_config(method_name)


def set_method_config(method_name: str, config: Dict[str, Any]) -> None:
    """Set configuration for a method."""
    config_manager.set_config(method_name, config)


def create_method(method_name: str, config: Optional[Dict[str, Any]] = None):
    """
    Create an instance of an explanation method.

    Args:
        method_name: Name of the method to create
        config: Optional configuration (uses default if None)

    Returns:
        Method instance
    """
    if config is None:
        config = get_method_config(method_name)

    # Import the method class
    class_name = METHOD_CLASSES.get(method_name)
    if class_name is None:
        raise ValueError(f"Unknown method: {method_name}")

    # Import from the appropriate module
    if method_name in METHOD_CATEGORIES['intrinsic']:
        from ..methods.intrinsic import PathAnalysisExplainer, OperatorWeightExplainer
        if method_name == 'PathAnalysis':
            return PathAnalysisExplainer(config)
        elif method_name == 'OperatorWeight':
            return OperatorWeightExplainer(config)
    elif method_name in METHOD_CATEGORIES['posthoc']:
        from ..methods.posthoc import GradCAMExplainer, SHAPExplainer
        if method_name == 'GradCAM':
            return GradCAMExplainer(config)
        elif method_name == 'SHAP':
            return SHAPExplainer(config)

    raise ValueError(f"Could not create method: {method_name}")


def list_available_methods() -> Dict[str, List[str]]:
    """List all available methods by category."""
    return METHOD_CATEGORIES.copy()


def get_method_class_name(method_name: str) -> str:
    """Get the class name for a method."""
    return METHOD_CLASSES.get(method_name, 'Unknown')