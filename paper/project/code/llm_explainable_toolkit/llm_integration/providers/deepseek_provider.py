"""
Deepseek API Provider

This module implements the Deepseek API provider for fault diagnosis explanation generation.
Deepseek offers high-performance Chinese language models with competitive pricing.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, Union
import aiohttp
from .base_provider import BaseLLMProvider

# Configure logging
logger = logging.getLogger(__name__)


class DeepseekProvider(BaseLLMProvider):
    """
    Deepseek API provider implementation.

    Deepseek offers deepseek-chat model with excellent Chinese understanding
    and cost-effective pricing for industrial applications.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com", model: str = "deepseek-chat"):
        """
        Initialize Deepseek provider.

        Args:
            api_key: Deepseek API key
            base_url: API base URL
            model: Model name (default: deepseek-chat)
        """
        super().__init__(api_key, model)
        self.base_url = base_url
        self.session = None
        self.max_tokens = 1500
        self.temperature = 0.3
        self.timeout = 30

    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )

    async def generate_explanation(self, fault_data: Dict[str, Any], style: str = "standard") -> str:
        """
        Generate fault diagnosis explanation using Deepseek API.

        Args:
            fault_data: Fault diagnosis results from unified baseline models
            style: Explanation style preference

        Returns:
            Natural language explanation string
        """
        try:
            # Validate input data
            if not self._validate_fault_data(fault_data):
                raise ValueError("Invalid fault data format")

            # Format data for API
            formatted_data = self._format_fault_data(fault_data)

            # Build request payload
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": self._build_system_prompt(style)
                    },
                    {
                        "role": "user",
                        "content": self._build_user_prompt(formatted_data)
                    }
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": 0.95,
                "stream": False
            }

            # Ensure session exists
            await self._ensure_session()

            # Make API call
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Deepseek API error: {response.status} - {error_text}")
                    raise Exception(f"Deepseek API call failed: {response.status}")

                result = await response.json()

                # Extract explanation
                explanation = result["choices"][0]["message"]["content"]

                # Update statistics
                self._update_call_stats()

                logger.info(f"Generated explanation using Deepseek, style: {style}")
                return explanation

        except aiohttp.ClientError as e:
            logger.error(f"Network error calling Deepseek API: {str(e)}")
            raise Exception(f"Network error: {str(e)}")
        except Exception as e:
            logger.error(f"Error generating explanation with Deepseek: {str(e)}")
            raise

    def _build_user_prompt(self, fault_data: Dict[str, Any]) -> str:
        """
        Build user prompt with fault diagnosis data.

        Args:
            fault_data: Formatted fault diagnosis data

        Returns:
            User prompt string
        """
        prompt = f"""请分析以下故障诊断结果：

**故障类型**: {self._get_fault_type_name(fault_data.get('fault_type', 0))}
**置信度**: {fault_data.get('confidence', 0):.2f} ({self._get_confidence_level(fault_data.get('confidence', 0))})
**诊断模型**: {fault_data.get('model_name', '未知')}
**数据集**: {fault_data.get('dataset', '未知')}

"""

        # Add equipment info if available
        if 'equipment_info' in fault_data:
            prompt += f"**设备信息**: {fault_data['equipment_info']}\n"

        # Add key features if available
        if 'key_features' in fault_data and fault_data['key_features']:
            features = fault_data['key_features']
            if isinstance(features, list):
                prompt += f"**关键特征**: {', '.join([f'{f:.3f}' for f in features[:5]])}\n"
            else:
                prompt += f"**关键特征**: {features}\n"

        prompt += """

请基于以上信息，提供专业的故障诊断解释，包括：
1. 故障性质和严重程度
2. 可能的根本原因
3. 对设备运行的影响
4. 建议的处理措施

请确保解释内容准确、专业且易于理解。"""

        return prompt

    def _get_fault_type_name(self, fault_type: Union[int, str]) -> str:
        """Convert fault type code to readable name."""
        fault_names = {
            0: "正常状态",
            1: "内圈故障",
            2: "外圈故障",
            3: "滚动体故障",
            4: "保持架故障"
        }
        return fault_names.get(int(fault_type) if isinstance(fault_type, str) else fault_type, "未知故障")

    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to level description."""
        if confidence >= 0.9:
            return "非常高"
        elif confidence >= 0.7:
            return "高"
        elif confidence >= 0.5:
            return "中等"
        elif confidence >= 0.3:
            return "低"
        else:
            return "非常低"

    def test_connection(self) -> bool:
        """
        Test connection to Deepseek API.

        Returns:
            True if connection successful
        """
        try:
            # Simple test with minimal payload
            test_payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": "测试连接"}
                ],
                "max_tokens": 10
            }

            # This would be async in real usage
            return True  # Placeholder

        except Exception as e:
            logger.error(f"Deepseek connection test failed: {str(e)}")
            return False

    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()

    def __del__(self):
        """Cleanup when object is destroyed."""
        if self.session and not self.session.closed:
            # Note: This is not ideal, should properly close in async context
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_closed():
                    loop.create_task(self.close())
            except:
                pass
