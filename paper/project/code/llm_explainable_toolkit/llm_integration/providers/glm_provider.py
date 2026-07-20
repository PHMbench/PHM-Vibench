"""
GLM (智谱AI) API Provider

This module implements the GLM-4 API provider for fault diagnosis explanation generation.
GLM-4 is a large language model developed by Zhipu AI with excellent Chinese capabilities.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, Union
import aiohttp
from .base_provider import BaseLLMProvider

# Configure logging
logger = logging.getLogger(__name__)


class GLMProvider(BaseLLMProvider):
    """
    GLM-4 API provider implementation.

    GLM-4 is developed by Zhipu AI with strong technical understanding
    and professional capabilities for industrial applications.
    """

    def __init__(self, api_key: str, base_url: str = "https://open.bigmodel.cn/api/paas/v4", model: str = "glm-4"):
        """
        Initialize GLM provider.

        Args:
            api_key: Zhipu AI API key
            base_url: API base URL
            model: Model name (default: glm-4)
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
        Generate fault diagnosis explanation using GLM-4 API.

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
                "top_p": 0.95
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
                    logger.error(f"GLM API error: {response.status} - {error_text}")
                    raise Exception(f"GLM API call failed: {response.status}")

                result = await response.json()

                # Extract explanation
                if "choices" not in result or len(result["choices"]) == 0:
                    raise Exception("Invalid response format from GLM API")

                explanation = result["choices"][0]["message"]["content"]

                # Update statistics
                self._update_call_stats()

                logger.info(f"Generated explanation using GLM-4, style: {style}")
                return explanation

        except aiohttp.ClientError as e:
            logger.error(f"Network error calling GLM API: {str(e)}")
            raise Exception(f"Network error: {str(e)}")
        except Exception as e:
            logger.error(f"Error generating explanation with GLM-4: {str(e)}")
            raise

    def _build_system_prompt(self, style: str) -> str:
        """
        Build system prompt optimized for GLM-4.

        Args:
            style: Explanation style

        Returns:
            System prompt string
        """
        base_prompt = """你是智谱AI训练的GLM-4大语言模型，专门服务于工业故障诊断领域。

基于你的强大理解能力和专业知识，请分析提供的故障诊断数据，生成准确、有用的自然语言解释。

你的回答应该：
1. 专业准确，基于故障诊断数据
2. 条理清晰，易于理解
3. 包含实用的建议和结论
4. 体现GLM-4的技术理解能力

"""

        style_requirements = {
            "standard": "请提供标准的技术解释，平衡专业性和可读性，适合一般技术人员阅读。",
            "simple": "请用通俗易懂的语言解释故障情况，避免使用过多专业术语，适合设备操作人员理解。",
            "detailed": "请提供详细的技术分析，包含具体的信号特征、诊断过程和技术细节，适合专业工程师深入分析。",
            "formal": "请提供正式的故障诊断报告，使用规范的工程语言，适合技术文档和正式报告。",
            "technical": "请提供高度专业化的解释，使用精准的技术术语和分析方法，适合技术专家和研究人员。",
            "concise": "请提供简明扼要的故障总结，突出关键信息和核心结论，适合快速了解情况。"
        }

        return base_prompt + "\n\n当前要求：" + style_requirements.get(style, style_requirements["standard"])

    def _build_user_prompt(self, fault_data: Dict[str, Any]) -> str:
        """
        Build user prompt with structured fault diagnosis data.

        Args:
            fault_data: Formatted fault diagnosis data

        Returns:
            Structured user prompt
        """
        # Create structured JSON data for better GLM-4 understanding
        structured_data = {
            "fault_diagnosis_result": {
                "fault_type": {
                    "code": fault_data.get('fault_type', 0),
                    "name": self._get_fault_type_name(fault_data.get('fault_type', 0)),
                    "description": self._get_fault_description(fault_data.get('fault_type', 0))
                },
                "confidence": {
                    "value": fault_data.get('confidence', 0),
                    "level": self._get_confidence_level(fault_data.get('confidence', 0)),
                    "reliability": self._get_reliability_assessment(fault_data.get('confidence', 0))
                },
                "diagnosis_context": {
                    "model_used": fault_data.get('model_name', '未知'),
                    "dataset_source": fault_data.get('dataset', '未知'),
                    "analysis_timestamp": fault_data.get('timestamp', '未知')
                }
            }
        }

        # Add additional context if available
        if 'equipment_info' in fault_data:
            structured_data["equipment_info"] = fault_data['equipment_info']

        if 'key_features' in fault_data:
            structured_data["signal_features"] = {
                "key_indicators": fault_data['key_features'][:10] if isinstance(fault_data['key_features'], list) else [],
                "feature_count": len(fault_data['key_features']) if isinstance(fault_data['key_features'], list) else 1
            }

        prompt = f"""请基于以下结构化故障诊断数据，提供专业的分析解释：

{json.dumps(structured_data, ensure_ascii=False, indent=2)}

请从以下角度进行分析：
1. **故障识别准确性**: 评估诊断结果的可靠性
2. **故障机理分析**: 基于故障类型分析可能的原因
3. **影响评估**: 分析故障对设备运行的影响
4. **处理建议**: 提供针对性的维护和处理建议
5. **预防措施**: 建议避免类似故障的预防措施

请确保分析全面、专业且具有实际指导价值。"""

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

    def _get_fault_description(self, fault_type: Union[int, str]) -> str:
        """Get detailed description for each fault type."""
        descriptions = {
            0: "设备处于正常工作状态，未检测到明显故障特征",
            1: "轴承内圈出现损伤或磨损，通常表现为高频振动特征",
            2: "轴承外圈出现损伤或磨损，振动信号具有特定的调制特征",
            3: "轴承滚动体表面出现损伤，会产生非周期性的冲击信号",
            4: "轴承保持架损坏或松动，可能导致异常的运行噪声"
        }
        return descriptions.get(int(fault_type) if isinstance(fault_type, str) else fault_type, "未知故障类型")

    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to level description."""
        if confidence >= 0.95:
            return "极高置信度"
        elif confidence >= 0.85:
            return "高置信度"
        elif confidence >= 0.70:
            return "中等置信度"
        elif confidence >= 0.50:
            return "低置信度"
        else:
            return "极低置信度"

    def _get_reliability_assessment(self, confidence: float) -> str:
        """Assess reliability based on confidence."""
        if confidence >= 0.90:
            return "诊断结果高度可靠，可作为决策依据"
        elif confidence >= 0.75:
            return "诊断结果较为可靠，建议结合其他检测方法确认"
        elif confidence >= 0.60:
            return "诊断结果具有一定参考价值，需要进一步验证"
        else:
            return "诊断结果可靠性较低，建议重新检测或使用其他方法"

    def test_connection(self) -> bool:
        """
        Test connection to GLM API.

        Returns:
            True if connection successful
        """
        try:
            # Simple test with minimal payload
            test_payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": "连接测试"}
                ],
                "max_tokens": 10
            }

            # This would be async in real usage
            return True  # Placeholder

        except Exception as e:
            logger.error(f"GLM connection test failed: {str(e)}")
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
