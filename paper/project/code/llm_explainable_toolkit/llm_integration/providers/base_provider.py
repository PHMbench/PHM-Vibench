"""
Base LLM Provider Abstract Class

This module defines the abstract base class for all LLM providers,
ensuring consistent interface across different implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    This class defines the common interface that all LLM providers must implement,
    ensuring consistency across different LLM APIs and services.
    """

    def __init__(self, api_key: str = None, model: str = None, **kwargs):
        """
        Initialize the base LLM provider.

        Args:
            api_key: API key for authentication
            model: Model name/identifier
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.model = model
        self.config = kwargs
        self.last_call_time = None
        self.call_count = 0

    @abstractmethod
    async def generate_explanation(self, fault_data: Dict[str, Any], style: str = "standard") -> str:
        """
        Generate natural language explanation for fault diagnosis data.

        Args:
            fault_data: Dictionary containing fault diagnosis results
            style: Explanation style (standard, simple, detailed, formal, technical, concise)

        Returns:
            Natural language explanation string
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test connection to the LLM provider.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get provider information and status.

        Returns:
            Dictionary containing provider details
        """
        return {
            "provider": self.__class__.__name__,
            "model": self.model,
            "call_count": self.call_count,
            "last_call": self.last_call_time
        }

    def _validate_fault_data(self, fault_data: Dict[str, Any]) -> bool:
        """
        Validate fault data format.

        Args:
            fault_data: Fault diagnosis data to validate

        Returns:
            True if data format is valid
        """
        required_fields = ['fault_type', 'confidence']
        return all(field in fault_data for field in required_fields)

    def _format_fault_data(self, fault_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format fault data for LLM input.

        Args:
            fault_data: Raw fault data

        Returns:
            Formatted fault data
        """
        formatted = {
            'fault_type': fault_data.get('fault_type', 'unknown'),
            'confidence': fault_data.get('confidence', 0.0),
            'model_name': fault_data.get('model_name', 'unknown'),
            'dataset': fault_data.get('dataset', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }

        # Add optional fields if present
        for field in ['equipment_info', 'key_features', 'raw_signal', 'attention_weights']:
            if field in fault_data:
                formatted[field] = fault_data[field]

        return formatted

    def _build_system_prompt(self, style: str) -> str:
        """
        Build system prompt based on style.

        Args:
            style: Explanation style

        Returns:
            System prompt string
        """
        base_prompt = """你是一个专业的工业设备故障诊断专家，擅长分析振动信号并提供准确的故障解释。
你需要根据提供的故障诊断数据，生成清晰、专业的自然语言解释。"""

        style_prompts = {
            "standard": "请提供标准的技术解释，包含故障原因和可能的影响。",
            "simple": "请用简单易懂的语言解释故障情况，适合非技术人员理解。",
            "detailed": "请提供详细的技术分析，包含具体的信号特征和诊断依据。",
            "formal": "请提供正式的故障诊断报告，适合工程文档使用。",
            "technical": "请提供高度技术性的解释，包含专业术语和分析方法。",
            "concise": "请提供简洁的故障总结，突出关键信息。"
        }

        return base_prompt + "\n\n" + style_prompts.get(style, style_prompts["standard"])

    def _update_call_stats(self):
        """Update call statistics."""
        self.call_count += 1
        self.last_call_time = datetime.now()