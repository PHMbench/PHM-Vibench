"""
LLM增强解释接口规范

为主仓库中的LLM子项目预留自然语言解释接口。
提供标准化的LLM集成规范和示例实现。

作者: Explainable_FD_Toolkit开发团队
版本: 1.0.0
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import asyncio
import threading
from queue import Queue
import time

# 用于类型标注的导入
from toolkit_integration.explainability.core.explanation import Explanation


class AudienceType(Enum):
    """目标受众类型"""
    ENGINEER = "engineer"         # 工程师
    RESEARCHER = "researcher"     # 研究者
    MANAGER = "manager"           # 管理者
    TECHNICIAN = "technician"     # 技术员
    STUDENT = "student"           # 学生
    GENERAL = "general"           # 一般用户


class ExplanationLevel(Enum):
    """解释详细程度"""
    BRIEF = "brief"           # 简要
    DETAILED = "detailed"     # 详细
    COMPREHENSIVE = "comprehensive"  # 全面
    TECHNICAL = "technical"   # 技术深度


class OutputFormat(Enum):
    """输出格式"""
    TEXT = "text"             # 纯文本
    MARKDOWN = "markdown"     # Markdown格式
    HTML = "html"            # HTML格式
    JSON = "json"            # JSON格式
    REPORT = "report"        # 报告格式


@dataclass
class LLMConfig:
    """LLM配置类"""
    model_name: str = "gpt-4"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 30
    retry_attempts: int = 3
    language: str = "zh"
    cache_enabled: bool = True
    streaming: bool = False


@dataclass
class ExplanationRequest:
    """解释请求类"""
    explanation: Explanation
    target_audience: AudienceType = AudienceType.ENGINEER
    explanation_level: ExplanationLevel = ExplanationLevel.DETAILED
    output_format: OutputFormat = OutputFormat.TEXT
    context: Optional[Dict[str, Any]] = None
    custom_requirements: Optional[List[str]] = None
    language: Optional[str] = None
    include_visualizations: bool = False
    include_recommendations: bool = True


@dataclass
class ConversationContext:
    """对话上下文"""
    session_id: str
    user_id: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_explanation: Optional[Explanation] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    domain_knowledge: Dict[str, Any] = field(default_factory=dict)
    last_interaction_time: Optional[float] = None


@dataclass
class LLMResponse:
    """LLM响应类"""
    content: str
    response_time: float
    token_usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


class BaseLLMInterface(ABC):
    """LLM接口基类"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> bool:
        """初始化LLM接口"""
        pass

    @abstractmethod
    async def generate_natural_explanation(self, request: ExplanationRequest) -> LLMResponse:
        """生成自然语言解释"""
        pass

    @abstractmethod
    async def conversational_explain(self, query: str, context: ConversationContext) -> LLMResponse:
        """对话式解释"""
        pass

    @abstractmethod
    async def batch_explain(self, requests: List[ExplanationRequest]) -> List[LLMResponse]:
        """批量解释生成"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        pass


class MockLLMInterface(BaseLLMInterface):
    """模拟LLM接口实现（用于演示和测试）"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.response_templates = self._load_response_templates()

    async def initialize(self) -> bool:
        """初始化模拟接口"""
        await asyncio.sleep(0.1)  # 模拟初始化延迟
        self._initialized = True
        return True

    async def generate_natural_explanation(self, request: ExplanationRequest) -> LLMResponse:
        """生成模拟的自然语言解释"""
        start_time = time.time()

        # 模拟处理延迟
        await asyncio.sleep(0.5)

        # 生成解释内容
        content = self._generate_explanation_content(request)

        response_time = time.time() - start_time

        return LLMResponse(
            content=content,
            response_time=response_time,
            token_usage={'prompt_tokens': 150, 'completion_tokens': 300, 'total_tokens': 450},
            metadata={
                'model': self.config.model_name,
                'audience': request.target_audience.value,
                'level': request.explanation_level.value,
                'format': request.output_format.value
            }
        )

    async def conversational_explain(self, query: str, context: ConversationContext) -> LLMResponse:
        """模拟对话式解释"""
        start_time = time.time()

        # 模拟处理延迟
        await asyncio.sleep(0.3)

        # 生成回复内容
        content = self._generate_conversational_response(query, context)

        response_time = time.time() - start_time

        # 更新对话历史
        context.conversation_history.append({
            'timestamp': time.time(),
            'query': query,
            'response': content,
            'type': 'conversation'
        })
        context.last_interaction_time = time.time()

        return LLMResponse(
            content=content,
            response_time=response_time,
            token_usage={'prompt_tokens': 80, 'completion_tokens': 150, 'total_tokens': 230}
        )

    async def batch_explain(self, requests: List[ExplanationRequest]) -> List[LLMResponse]:
        """批量解释生成"""
        tasks = [self.generate_natural_explanation(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        processed_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                processed_responses.append(LLMResponse(
                    content="",
                    response_time=0.0,
                    success=False,
                    error_message=str(response)
                ))
            else:
                processed_responses.append(response)

        return processed_responses

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'status': 'healthy',
            'model': self.config.model_name,
            'initialized': self._initialized,
            'config': {
                'max_tokens': self.config.max_tokens,
                'temperature': self.config.temperature,
                'language': self.config.language
            }
        }

    def _generate_explanation_content(self, request: ExplanationRequest) -> str:
        """生成解释内容"""
        fault_type = request.explanation.get_meta('fault_type', '未知故障')
        method = request.explanation.get_method_name()
        confidence = request.explanation.get_metrics().get('attribution_max', 0.8)

        # 根据受众和级别选择模板
        template_key = f"{request.target_audience.value}_{request.explanation_level.value}"
        template = self.response_templates.get(template_key, self.response_templates['default'])

        # 填充模板
        content = template.format(
            fault_type=fault_type,
            method=method,
            confidence=f"{confidence:.1%}",
            details=self._get_fault_details(fault_type),
            recommendations=self._get_recommendations(fault_type)
        )

        return content

    def _generate_conversational_response(self, query: str, context: ConversationContext) -> str:
        """生成对话回复"""
        query_lower = query.lower()

        if '为什么' in query and '故障' in query:
            return self._get_fault_cause_explanation(context.current_explanation)
        elif '如何' in query and ('修复' in query or '处理' in query):
            return self._get_repair_explanation(context.current_explanation)
        elif '严重' in query or '风险' in query:
            return self._get_severity_assessment(context.current_explanation)
        elif '预防' in query:
            return self._get_prevention_tips(context.current_explanation)
        else:
            return "我可以帮您解释故障诊断的结果，包括故障原因、修复方法、严重程度评估和预防措施。请告诉我您想了解哪个方面。"

    def _load_response_templates(self) -> Dict[str, str]:
        """加载响应模板"""
        return {
            'engineer_detailed': """
## 故障诊断技术报告

### 📋 诊断概要
- **故障类型**: {fault_type}
- **检测方法**: {method}
- **诊断置信度**: {confidence}

### 🔍 技术分析
{details}

### 🛠️ 修复建议
{recommendations}

---
*此报告由AI辅助生成，请结合专业判断进行决策*
            """,

            'manager_brief': """
## 设备状态简报

**故障类型**: {fault_type}
**置信度**: {confidence}
**建议措施**: {recommendations}

请安排相应维护计划。
            """,

            'researcher_technical': """
## 技术分析报告

### 方法论
使用{method}方法进行分析，置信度为{confidence}。

### 故障机理分析
{details}

### 技术建议
{recommendations}
            """,

            'default': """
## 故障诊断解释

检测到{fault_type}故障，置信度为{confidence}。

技术分析：
{details}

建议措施：
{recommendations}
            """
        }

    def _get_fault_details(self, fault_type: str) -> str:
        """获取故障详情"""
        fault_details = {
            'inner_race': "内圈故障通常由材料疲劳、过载或润滑不足引起。振动信号中会出现明显的内圈故障频率成分，包络分析显示周期性冲击特征。",
            'outer_race': "外圈故障表现为特定频率成分的增强，通常与载荷分布不均或安装误差有关。",
            'ball': "滚动体故障会导致随机性的冲击信号，频谱分析显示宽带高频成分。",
            'normal': "设备运行状态正常，各频段信号特征符合预期范围。"
        }
        return fault_details.get(fault_type, "需要进一步分析以确定具体故障特征。")

    def _get_recommendations(self, fault_type: str) -> str:
        """获取修复建议"""
        recommendations = {
            'inner_race': "建议在下次维护窗口期内更换轴承，检查润滑系统，优化运行参数。",
            'outer_race': "安排定期检查，监控故障发展情况，根据严重程度决定更换时机。",
            'ball': "建议立即停机检查，防止进一步损坏，同时检查相关部件。",
            'normal': "继续正常运行，保持定期监控。"
        }
        return recommendations.get(fault_type, "建议进行详细检查以确定适当的维护措施。")

    def _get_fault_cause_explanation(self, explanation: Optional[Explanation]) -> str:
        """获取故障原因解释"""
        if not explanation:
            return "请先提供具体的故障诊断结果。"
        return "基于振动信号分析，故障主要由于长期运行导致的材料疲劳和磨损。建议检查设备运行历史和维护记录。"

    def _get_repair_explanation(self, explanation: Optional[Explanation]) -> str:
        """获取修复方法解释"""
        if not explanation:
            return "请先提供具体的故障诊断结果。"
        return "1. 立即措施：停机检查，记录故障现象\n2. 维修步骤：更换损坏部件，清洁润滑\n3. 预防措施：优化运行参数，定期维护"

    def _get_severity_assessment(self, explanation: Optional[Explanation]) -> str:
        """获取严重程度评估"""
        if not explanation:
            return "请先提供具体的故障诊断结果。"
        return "风险评估：中等严重程度。短期影响可能增加维护成本，长期可能影响设备寿命。建议在1-2周内安排维护。"

    def _get_prevention_tips(self, explanation: Optional[Explanation]) -> str:
        """获取预防建议"""
        if not explanation:
            return "请先提供具体的故障诊断结果。"
        return "1. 定期检查：每月进行振动分析\n2. 润滑维护：按照制造商建议更换润滑剂\n3. 运行监控：实时监控关键参数\n4. 培训操作人员：提高早期故障识别能力"


class LLMInterfaceManager:
    """LLM接口管理器"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.interface = MockLLMInterface(config)
        self._initialized = False
        self._conversation_contexts: Dict[str, ConversationContext] = {}
        self._cache: Dict[str, Any] = {}

    async def initialize(self) -> bool:
        """初始化管理器"""
        self._initialized = await self.interface.initialize()
        return self._initialized

    async def generate_natural_explanation(self, explanation: Explanation,
                                         target_audience: AudienceType = AudienceType.ENGINEER,
                                         explanation_level: ExplanationLevel = ExplanationLevel.DETAILED,
                                         **kwargs) -> LLMResponse:
        """生成自然语言解释"""
        if not self._initialized:
            raise RuntimeError("LLM接口未初始化")

        request = ExplanationRequest(
            explanation=explanation,
            target_audience=target_audience,
            explanation_level=explanation_level,
            **kwargs
        )

        return await self.interface.generate_natural_explanation(request)

    async def start_conversation(self, session_id: str, user_id: Optional[str] = None,
                                explanation: Optional[Explanation] = None) -> ConversationContext:
        """开始新的对话"""
        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            current_explanation=explanation,
            last_interaction_time=time.time()
        )

        self._conversation_contexts[session_id] = context
        return context

    async def continue_conversation(self, session_id: str, query: str) -> LLMResponse:
        """继续对话"""
        if session_id not in self._conversation_contexts:
            raise ValueError(f"会话 {session_id} 不存在")

        context = self._conversation_contexts[session_id]
        return await self.interface.conversational_explain(query, context)

    async def batch_explain(self, explanations: List[Explanation],
                           audience: AudienceType = AudienceType.ENGINEER,
                           level: ExplanationLevel = ExplanationLevel.DETAILED) -> List[LLMResponse]:
        """批量解释生成"""
        if not self._initialized:
            raise RuntimeError("LLM接口未初始化")

        requests = [
            ExplanationRequest(
                explanation=exp,
                target_audience=audience,
                explanation_level=level
            )
            for exp in explanations
        ]

        return await self.interface.batch_explain(requests)

    def get_conversation_context(self, session_id: str) -> Optional[ConversationContext]:
        """获取对话上下文"""
        return self._conversation_contexts.get(session_id)

    def end_conversation(self, session_id: str) -> bool:
        """结束对话"""
        if session_id in self._conversation_contexts:
            del self._conversation_contexts[session_id]
            return True
        return False

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'manager_initialized': self._initialized,
            'active_conversations': len(self._conversation_contexts),
            'interface_status': await self.interface.health_check()
        }


# 便捷函数
async def create_llm_interface(config: Optional[LLMConfig] = None) -> LLMInterfaceManager:
    """创建LLM接口管理器的便捷函数"""
    if config is None:
        config = LLMConfig()

    manager = LLMInterfaceManager(config)
    await manager.initialize()
    return manager


# 装饰器
def require_initialized(func):
    """要求LLM接口已初始化的装饰器"""
    async def wrapper(self, *args, **kwargs):
        if not getattr(self, '_initialized', False):
            raise RuntimeError("LLM接口未初始化")
        return await func(self, *args, **kwargs)
    return wrapper


# 使用示例
async def example_usage():
    """使用示例"""
    # 创建配置
    config = LLMConfig(
        model_name="gpt-4",
        max_tokens=1000,
        temperature=0.7,
        language="zh"
    )

    # 初始化LLM接口
    llm_manager = await create_llm_interface(config)

    # 创建示例解释
    from toolkit_integration.explainability.core.explanation import Explanation
    explanation_data = {'attributions': [0.1, 0.8, 0.3]}
    explanation_meta = {'fault_type': 'inner_race', 'method': 'signal_path'}
    explanation = Explanation(explanation_data, explanation_meta)

    # 生成自然语言解释
    response = await llm_manager.generate_natural_explanation(
        explanation,
        target_audience=AudienceType.ENGINEER,
        explanation_level=ExplanationLevel.DETAILED
    )

    print(f"自然语言解释:\n{response.content}")

    # 开始对话
    context = await llm_manager.start_conversation("session_001", explanation=explanation)

    # 继续对话
    query_response = await llm_manager.continue_conversation("session_001", "为什么会发生这个故障？")
    print(f"\n对话回复:\n{query_response.content}")

    # 健康检查
    health = await llm_manager.health_check()
    print(f"\n健康检查:\n{health}")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())