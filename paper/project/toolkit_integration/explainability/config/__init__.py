"""
Configuration module for Explainable FD Toolkit

This module provides configuration management for all explanation methods
and visualization settings.
"""

from .method_configs import (
    MethodConfigManager,
    config_manager,
    get_method_config,
    set_method_config,
    create_method,
    list_available_methods,
    get_method_class_name,
    DEFAULT_CONFIGS,
    METHOD_CATEGORIES,
    METHOD_CLASSES
)

__all__ = [
    'MethodConfigManager',
    'config_manager',
    'get_method_config',
    'set_method_config',
    'create_method',
    'list_available_methods',
    'get_method_class_name',
    'DEFAULT_CONFIGS',
    'METHOD_CATEGORIES',
    'METHOD_CLASSES'
]