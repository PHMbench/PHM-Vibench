"""
LLM Manager

This module provides a unified manager for multiple LLM providers,
including domestic models (Deepseek, GLM) and template-based fallback.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .providers import get_provider, list_providers
from .enhanced_template_llm import EnhancedTemplateLLM

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    # Deepseek configuration
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # GLM configuration
    glm_api_key: Optional[str] = None
    glm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    glm_model: str = "glm-4"

    # General settings
    primary_provider: str = "template"
    fallback_provider: str = "template"
    timeout: int = 30
    retry_attempts: int = 3
    cache_enabled: bool = True

    @classmethod
    def from_env(cls):
        """Create configuration from environment variables."""
        return cls(
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            glm_api_key=os.getenv("GLM_API_KEY"),
            primary_provider=os.getenv("LLM_PRIMARY_PROVIDER", "template"),
            fallback_provider=os.getenv("LLM_FALLBACK_PROVIDER", "template")
        )


class LLMManager:
    """
    Unified manager for LLM providers with automatic fallback and load balancing.

    Supports domestic LLM providers (Deepseek, GLM) with automatic fallback
    to template-based system when API calls fail.
    """

    def __init__(self, config: LLMConfig = None):
        """
        Initialize LLM manager.

        Args:
            config: LLM configuration object
        """
        self.config = config or LLMConfig.from_env()
        self.providers = {}
        self.template_llm = EnhancedTemplateLLM()
        self._initialize_providers()
        self._log_available_providers()

    def _initialize_providers(self):
        """Initialize all available providers."""
        # Initialize Deepseek if API key is available
        if self.config.deepseek_api_key:
            try:
                self.providers['deepseek'] = get_provider(
                    'deepseek',
                    api_key=self.config.deepseek_api_key,
                    base_url=self.config.deepseek_base_url,
                    model=self.config.deepseek_model
                )
                logger.info("Deepseek provider initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Deepseek provider: {str(e)}")
        else:
            logger.warning("Deepseek API key not provided, using template fallback")

        # Initialize GLM if API key is available
        if self.config.glm_api_key:
            try:
                self.providers['glm'] = get_provider(
                    'glm',
                    api_key=self.config.glm_api_key,
                    base_url=self.config.glm_base_url,
                    model=self.config.glm_model
                )
                logger.info("GLM provider initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize GLM provider: {str(e)}")
        else:
            logger.warning("GLM API key not provided, using template fallback")

        # Always add template fallback
        self.providers['template'] = self.template_llm

    def _log_available_providers(self):
        """Log information about available providers."""
        available = list(self.providers.keys())
        primary = self.config.primary_provider if self.config.primary_provider in available else 'template'

        logger.info(f"Available LLM providers: {available}")
        logger.info(f"Primary provider: {primary}")
        logger.info(f"Fallback provider: {self.config.fallback_provider}")

    async def generate_explanation(self, fault_data: Dict[str, Any], style: str = "standard",
                                 provider: Optional[str] = None) -> str:
        """
        Generate fault diagnosis explanation using specified or primary provider.

        Args:
            fault_data: Fault diagnosis results from unified baseline
            style: Explanation style (standard, simple, detailed, formal, technical, concise)
            provider: Specific provider to use (optional)

        Returns:
            Natural language explanation string

        Raises:
            Exception: If all providers fail
        """
        # Determine providers to try
        if provider and provider in self.providers:
            providers_to_try = [provider, self.config.fallback_provider]
        else:
            providers_to_try = [self.config.primary_provider, self.config.fallback_provider]

        # Ensure fallback is available
        if self.config.fallback_provider not in providers_to_try:
            providers_to_try.append('template')

        last_error = None

        for provider_name in providers_to_try:
            if provider_name not in self.providers:
                logger.warning(f"Provider {provider_name} not available, skipping")
                continue

            try:
                logger.info(f"Attempting to generate explanation using {provider_name}")

                provider = self.providers[provider_name]

                if provider_name == 'template':
                    # Template LLM is synchronous
                    explanation = provider.generate_explanation(fault_data, style)
                else:
                    # API providers are async
                    explanation = await provider.generate_explanation(fault_data, style)

                logger.info(f"Successfully generated explanation using {provider_name}")
                return explanation

            except Exception as e:
                last_error = e
                logger.error(f"Provider {provider_name} failed: {str(e)}")

                if provider_name != providers_to_try[-1]:
                    logger.info(f"Falling back to next provider...")
                continue

        # All providers failed
        raise Exception(f"All LLM providers failed. Last error: {str(last_error)}")

    async def batch_generate_explanations(self, fault_data_list: List[Dict[str, Any]],
                                        style: str = "standard",
                                        provider: Optional[str] = None) -> List[str]:
        """
        Generate explanations for multiple fault diagnosis results.

        Args:
            fault_data_list: List of fault diagnosis results
            style: Explanation style
            provider: Specific provider to use

        Returns:
            List of explanations
        """
        tasks = []
        for fault_data in fault_data_list:
            task = self.generate_explanation(fault_data, style, provider)
            tasks.append(task)

        # Execute all tasks concurrently
        explanations = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        results = []
        for i, explanation in enumerate(explanations):
            if isinstance(explanation, Exception):
                logger.error(f"Failed to generate explanation for item {i}: {str(explanation)}")
                results.append(f"生成解释失败: {str(explanation)}")
            else:
                results.append(explanation)

        return results

    def get_available_providers(self) -> List[str]:
        """
        Get list of available providers.

        Returns:
            List of provider names
        """
        return list(self.providers.keys())

    def get_provider_info(self, provider_name: str = None) -> Dict[str, Any]:
        """
        Get information about a specific provider or all providers.

        Args:
            provider_name: Specific provider name (optional)

        Returns:
            Provider information dictionary
        """
        if provider_name:
            if provider_name in self.providers:
                if provider_name == 'template':
                    return {"provider": "template", "type": "template", "status": "available"}
                else:
                    return self.providers[provider_name].get_provider_info()
            else:
                return {"error": f"Provider {provider_name} not found"}
        else:
            # Return info for all providers
            info = {}
            for name, provider in self.providers.items():
                if name == 'template':
                    info[name] = {"provider": "template", "type": "template", "status": "available"}
                else:
                    info[name] = provider.get_provider_info()
            return info

    async def test_all_providers(self) -> Dict[str, bool]:
        """
        Test connection to all available providers.

        Returns:
            Dictionary mapping provider names to test results
        """
        results = {}
        for name, provider in self.providers.items():
            try:
                if name == 'template':
                    # Template is always available
                    results[name] = True
                else:
                    # Test API provider connection
                    results[name] = provider.test_connection()
            except Exception as e:
                logger.error(f"Error testing provider {name}: {str(e)}")
                results[name] = False

        return results

    async def close_all_providers(self):
        """Close all provider connections."""
        for name, provider in self.providers.items():
            try:
                if hasattr(provider, 'close'):
                    await provider.close()
                logger.info(f"Closed provider {name}")
            except Exception as e:
                logger.error(f"Error closing provider {name}: {str(e)}")

    def update_config(self, new_config: LLMConfig):
        """
        Update manager configuration.

        Args:
            new_config: New configuration object
        """
        self.config = new_config
        self._initialize_providers()
        logger.info("Configuration updated and providers reinitialized")

    def get_domestic_model_advantages(self) -> Dict[str, List[str]]:
        """
        Get advantages of domestic LLM models.

        Returns:
            Dictionary of model advantages
        """
        return {
            "deepseek": [
                "成本效益高，适合大规模部署",
                "中文理解能力强，适合国内应用",
                "API 兼容 OpenAI 格式，集成简单",
                "响应速度快，服务稳定"
            ],
            "glm": [
                "智谱 AI 专业支持，服务稳定可靠",
                "GLM-4 在技术文档理解方面表现优异",
                "支持多种行业场景，包括工业诊断",
                "持续更新优化，功能不断完善"
            ],
            "general": [
                "数据安全性高，符合国内法规要求",
                "网络延迟低，国内访问速度快",
                "技术支持响应快，沟通无障碍",
                "支持定制化开发和私有化部署"
            ]
        }