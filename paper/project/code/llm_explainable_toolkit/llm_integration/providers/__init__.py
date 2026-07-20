"""
LLM Providers Package

This package contains various LLM provider implementations for the fault diagnosis system.
"""

from .base_provider import BaseLLMProvider
from .deepseek_provider import DeepseekProvider
from .glm_provider import GLMProvider

__all__ = [
    'BaseLLMProvider',
    'DeepseekProvider',
    'GLMProvider'
]

# Provider registry for easy access
PROVIDER_REGISTRY = {
    'deepseek': DeepseekProvider,
    'glm': GLMProvider
}

def get_provider(provider_name: str, **kwargs):
    """
    Get provider instance by name.

    Args:
        provider_name: Name of the provider (e.g., 'deepseek', 'glm')
        **kwargs: Provider initialization arguments

    Returns:
        Provider instance

    Raises:
        ValueError: If provider not found
    """
    if provider_name not in PROVIDER_REGISTRY:
        raise ValueError(f"Unknown provider: {provider_name}. Available providers: {list(PROVIDER_REGISTRY.keys())}")

    provider_class = PROVIDER_REGISTRY[provider_name]
    return provider_class(**kwargs)

def list_providers():
    """
    List all available providers.

    Returns:
        List of provider names
    """
    return list(PROVIDER_REGISTRY.keys())