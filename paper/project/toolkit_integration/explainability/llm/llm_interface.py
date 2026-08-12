"""
LLM Interface - Unified API for Large Language Model Integration

Provides a unified interface for accessing different LLM providers (OpenAI, Anthropic,
local models) with consistent request/response handling and error management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Callable
import json
import time
import requests
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response from LLM."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider implementation."""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", base_url: Optional[str] = None):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model name (gpt-3.5-turbo, gpt-4, etc.)
            base_url: Optional custom base URL
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize OpenAI client."""
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            logger.warning("OpenAI library not installed. Install with: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")

    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', 2000),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 1.0),
                frequency_penalty=kwargs.get('frequency_penalty', 0.0),
                presence_penalty=kwargs.get('presence_penalty', 0.0)
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI API request failed: {e}")

    def is_available(self) -> bool:
        """Check if OpenAI service is available."""
        if not self.client:
            return False

        try:
            # Simple test request
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return True
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information."""
        return {
            "provider": "openai",
            "model": self.model,
            "api_base": self.base_url,
            "supported_models": [
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-4-turbo-preview",
                "gpt-4o"
            ]
        }


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation."""

    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            model: Model name (claude-3-sonnet, claude-3-haiku, etc.)
        """
        self.api_key = api_key
        self.model = model
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            logger.warning("Anthropic library not installed. Install with: pip install anthropic")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")

    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Anthropic API."""
        if not self.client:
            raise RuntimeError("Anthropic client not initialized")

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', 2000),
                temperature=kwargs.get('temperature', 0.7),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise RuntimeError(f"Anthropic API request failed: {e}")

    def is_available(self) -> bool:
        """Check if Anthropic service is available."""
        if not self.client:
            return False

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}]
            )
            return True
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get Anthropic model information."""
        return {
            "provider": "anthropic",
            "model": self.model,
            "supported_models": [
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
                "claude-3-opus-20240229"
            ]
        }


class LocalProvider(LLMProvider):
    """Local model provider implementation."""

    def __init__(self, model_path: str, api_url: Optional[str] = None):
        """
        Initialize local provider.

        Args:
            model_path: Path to local model
            api_url: URL for local model API (if using model server)
        """
        self.model_path = Path(model_path)
        self.api_url = api_url or "http://localhost:8000/v1"
        self.model_name = "local-model"

    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using local model."""
        if not self.model_path.exists() and not self.api_url:
            raise RuntimeError("Neither valid model path nor API URL provided")

        # Try API first if available
        if self.api_url:
            try:
                return self._generate_via_api(prompt, **kwargs)
            except Exception as e:
                logger.warning(f"API request failed: {e}")

        # Fallback to local inference if libraries available
        try:
            return self._generate_local(prompt, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Local model inference failed: {e}")

    def _generate_via_api(self, prompt: str, **kwargs) -> str:
        """Generate via API endpoint."""
        headers = {
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get('max_tokens', 2000),
            "temperature": kwargs.get('temperature', 0.7)
        }

        response = requests.post(
            f"{self.api_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=kwargs.get('timeout', 60)
        )

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise RuntimeError(f"API request failed with status {response.status_code}")

    def _generate_local(self, prompt: str, **kwargs) -> str:
        """Generate using local inference."""
        # This would need model-specific implementation
        # Placeholder for now
        raise NotImplementedError("Local inference not yet implemented")

    def is_available(self) -> bool:
        """Check if local model is available."""
        if self.api_url:
            try:
                response = requests.get(f"{self.api_url}/models", timeout=5)
                return response.status_code == 200
            except Exception:
                return False
        else:
            return self.model_path.exists()

    def get_model_info(self) -> Dict[str, Any]:
        """Get local model information."""
        return {
            "provider": "local",
            "model": self.model_name,
            "model_path": str(self.model_path),
            "api_url": self.api_url
        }


class LLMInterface:
    """
    Unified interface for accessing different LLM providers.

    This class provides a consistent API for generating responses from various
    LLM providers, with automatic fallback and error handling.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM interface with configuration.

        Args:
            config: Configuration dictionary containing provider settings
        """
        self.config = config
        self.providers = {}
        self.primary_provider = None
        self.fallback_providers = []

        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize configured providers."""
        provider_configs = self.config.get('providers', {})

        for provider_name, provider_config in provider_configs.items():
            try:
                provider = self._create_provider(provider_config)
                if provider.is_available():
                    self.providers[provider_name] = provider

                    # Set primary provider (first available)
                    if self.primary_provider is None:
                        self.primary_provider = provider_name
                    else:
                        self.fallback_providers.append(provider_name)
                else:
                    logger.warning(f"Provider {provider_name} is not available")

            except Exception as e:
                logger.error(f"Failed to initialize provider {provider_name}: {e}")

        if not self.providers:
            logger.warning("No LLM providers are available")

    def _create_provider(self, config: Dict[str, Any]) -> LLMProvider:
        """Create provider instance from configuration."""
        provider_type = config.get('type', '').lower()

        if provider_type == 'openai':
            return OpenAIProvider(
                api_key=config['api_key'],
                model=config.get('model', 'gpt-3.5-turbo'),
                base_url=config.get('base_url')
            )
        elif provider_type == 'anthropic':
            return AnthropicProvider(
                api_key=config['api_key'],
                model=config.get('model', 'claude-3-sonnet-20240229')
            )
        elif provider_type == 'local':
            return LocalProvider(
                model_path=config['model_path'],
                api_url=config.get('api_url')
            )
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")

    def generate_response(self,
                         prompt: str,
                         provider_name: Optional[str] = None,
                         use_fallback: bool = True,
                         **kwargs) -> str:
        """
        Generate response from LLM.

        Args:
            prompt: Input prompt
            provider_name: Specific provider to use (optional)
            use_fallback: Whether to use fallback providers on failure
            **kwargs: Additional generation parameters

        Returns:
            Generated response text

        Raises:
            RuntimeError: If no providers are available or all requests fail
        """
        if not self.providers:
            raise RuntimeError("No LLM providers available")

        # Determine providers to try
        if provider_name and provider_name in self.providers:
            providers_to_try = [provider_name]
        elif self.primary_provider:
            providers_to_try = [self.primary_provider] + self.fallback_providers
        else:
            providers_to_try = list(self.providers.keys())

        last_error = None

        for provider_name in providers_to_try:
            try:
                logger.info(f"Attempting to generate response using {provider_name}")
                provider = self.providers[provider_name]
                response = provider.generate_response(prompt, **kwargs)
                logger.info(f"Successfully generated response using {provider_name}")
                return response

            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
                last_error = e

                if not use_fallback:
                    break

                # Add delay between retries
                time.sleep(1)

        # All providers failed
        error_msg = f"All providers failed to generate response. Last error: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    async def generate_response_async(self,
                                    prompt: str,
                                    provider_name: Optional[str] = None,
                                    **kwargs) -> str:
        """
        Generate response asynchronously.

        Args:
            prompt: Input prompt
            provider_name: Specific provider to use (optional)
            **kwargs: Additional generation parameters

        Returns:
            Generated response text
        """
        # For now, just use the sync method
        # In a real implementation, this would use async API calls
        return self.generate_response(prompt, provider_name, **kwargs)

    def is_available(self) -> bool:
        """Check if any providers are available."""
        return len(self.providers) > 0

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return list(self.providers.keys())

    def get_primary_provider(self) -> Optional[str]:
        """Get the primary provider name."""
        return self.primary_provider

    def get_provider_info(self, provider_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a specific provider or all providers."""
        if provider_name:
            if provider_name in self.providers:
                return self.providers[provider_name].get_model_info()
            else:
                return {"error": f"Provider {provider_name} not available"}
        else:
            info = {
                "primary_provider": self.primary_provider,
                "available_providers": list(self.providers.keys()),
                "fallback_providers": self.fallback_providers,
                "provider_details": {}
            }

            for name, provider in self.providers.items():
                info["provider_details"][name] = provider.get_model_info()

            return info

    def test_provider(self, provider_name: str) -> bool:
        """Test if a specific provider is working."""
        if provider_name not in self.providers:
            return False

        try:
            provider = self.providers[provider_name]
            test_response = provider.generate_response("Hello", max_tokens=10)
            return True
        except Exception:
            return False

    def switch_primary_provider(self, provider_name: str) -> bool:
        """
        Switch the primary provider.

        Args:
            provider_name: Name of the provider to make primary

        Returns:
            True if successful, False otherwise
        """
        if provider_name not in self.providers:
            logger.error(f"Provider {provider_name} not available")
            return False

        old_primary = self.primary_provider
        self.primary_provider = provider_name

        # Update fallback providers list
        self.fallback_providers = [p for p in self.fallback_providers if p != provider_name]
        if old_primary:
            self.fallback_providers.insert(0, old_primary)

        logger.info(f"Switched primary provider from {old_primary} to {provider_name}")
        return True

    def add_provider(self, name: str, config: Dict[str, Any]) -> bool:
        """
        Add a new provider.

        Args:
            name: Provider name
            config: Provider configuration

        Returns:
            True if successful, False otherwise
        """
        try:
            provider = self._create_provider(config)
            if provider.is_available():
                self.providers[name] = provider

                if self.primary_provider is None:
                    self.primary_provider = name
                else:
                    self.fallback_providers.append(name)

                logger.info(f"Successfully added provider: {name}")
                return True
            else:
                logger.warning(f"Provider {name} is not available")
                return False

        except Exception as e:
            logger.error(f"Failed to add provider {name}: {e}")
            return False

    def remove_provider(self, name: str) -> bool:
        """
        Remove a provider.

        Args:
            name: Provider name to remove

        Returns:
            True if successful, False otherwise
        """
        if name not in self.providers:
            return False

        # Handle primary provider removal
        if self.primary_provider == name:
            if self.fallback_providers:
                self.primary_provider = self.fallback_providers.pop(0)
            else:
                self.primary_provider = None
        elif name in self.fallback_providers:
            self.fallback_providers.remove(name)

        del self.providers[name]
        logger.info(f"Removed provider: {name}")
        return True


# Convenience function for creating LLM interface
def create_llm_interface(config_path: Optional[str] = None,
                        provider: Optional[str] = None,
                        api_key: Optional[str] = None,
                        model: Optional[str] = None) -> LLMInterface:
    """
    Create LLM interface with common configurations.

    Args:
        config_path: Path to configuration file
        provider: Provider type (openai, anthropic, local)
        api_key: API key for provider
        model: Model name

    Returns:
        Configured LLM interface
    """
    config = {}

    # Load from config file if provided
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)

    # Override with direct parameters
    if provider and api_key:
        provider_config = {
            'type': provider,
            'api_key': api_key
        }
        if model:
            provider_config['model'] = model

        config['providers'] = {provider: provider_config}

    return LLMInterface(config)