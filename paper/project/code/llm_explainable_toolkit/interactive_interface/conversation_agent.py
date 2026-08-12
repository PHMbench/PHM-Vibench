"""
Conversation Agent - Interactive Diagnostic Interface

Provides an intelligent conversational interface for fault diagnosis,
enabling natural language interactions with the diagnostic system.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import re

from ..core.explainer import LLMEnhancedExplainer


class ConversationAgent:
    """
    Intelligent conversation agent for diagnostic interactions.

    This agent manages conversations with users, providing natural
    language responses about fault diagnosis and explanations.
    """

    def __init__(self, explainer: LLMEnhancedExplainer):
        """
        Initialize the conversation agent.

        Args:
            explainer: The LLM-enhanced explainer instance
        """
        self.explainer = explainer

        # Conversation patterns and templates
        self._initialize_conversation_patterns()

        # State tracking
        self.conversation_state = {
            "current_phase": "greeting",
            "topics_discussed": [],
            "user_questions_asked": [],
            "clarifications_needed": []
        }

    def generate_greeting(self, session: Dict[str, Any]) -> str:
        """
        Generate initial greeting for conversation.

        Args:
            session: Session information

        Returns:
            Greeting message
        """
        initial_diagnosis = session["initial_diagnosis"]
        fault_type = initial_diagnosis["model_prediction"]["fault_type"]
        confidence = initial_diagnosis["model_prediction"]["confidence"]
        device_info = session.get("device_info", {})

        device_type = device_info.get("device_type", "设备")
        operating_speed = device_info.get("operating_speed", "未知")

        greeting = f"""您好！我是智能故障诊断助手。

根据您的{device_type}振动数据分析，我检测到可能存在 **{fault_type}** 故障，诊断置信度为 {confidence:.1%}。

{f'设备转速：{operating_speed} RPM' if operating_speed != '未知' else ''}

## 我可以为您提供以下帮助：

**🔍 技术分析**
• 详细的故障机理和信号特征分析
• 故障原因的深入探讨
• 相关故障模式的风险评估

**🛠️ 维修指导**
• 具体的维修步骤和注意事项
• 备件准备和工具需求
• 维修时间窗口建议

**📊 系统分析**
• 信号处理路径解释
• 诊断结果的可信度分析
• 相关技术参数的详细说明

**💡 对话式诊断**
• 回答您的具体技术问题
• 提供个性化的诊断建议
• 协助制定维护策略

请问您希望了解哪个方面？或者有其他相关的技术问题需要讨论吗？"""

        # Update conversation state
        self.conversation_state["current_phase"] = "initial_discussion"

        return greeting

    def process_message(self,
                       user_message: str,
                       session: Dict[str, Any]) -> str:
        """
        Process user message and generate response.

        Args:
            user_message: User's message
            session: Session information

        Returns:
            Generated response
        """
        # Analyze user intent
        intent = self._analyze_user_intent(user_message)

        # Update conversation state
        self._update_conversation_state(intent, user_message)

        # Generate contextual response
        context = self._prepare_context(session, intent)

        # Generate response using explainer
        response = self.explainer.explain_conversation(
            session.get("session_id", "unknown"),
            user_message,
            context
        )

        # If explainer fails or returns empty response, use rule-based response
        if not response:
            response = self._generate_rule_based_response(intent, user_message, session)

        return response

    def generate_conclusion(self, session: Dict[str, Any]) -> str:
        """
        Generate conversation conclusion.

        Args:
            session: Session information

        Returns:
            Conclusion message
        """
        duration = (datetime.now() - session["start_time"]).total_seconds()
        num_messages = len(session["conversation_history"])
        initial_diagnosis = session["initial_diagnosis"]
        fault_type = initial_diagnosis["model_prediction"]["fault_type"]

        # Summarize key topics discussed
        topics_summary = self._summarize_discussed_topics()

        conclusion = f"""感谢您的咨询！我们的诊断对话已经完成。

## 对话总结
• 对话时长：{duration:.0f} 秒
• 交流轮次：{num_messages} 次
• 主要问题：**{fault_type}** 故障诊断

{topics_summary}

## 核心建议

### 立即行动
1. **确认诊断结果**：建议结合其他检测方法验证故障
2. **安全措施**：如为高风险故障，请立即采取安全措施
3. **资源准备**：根据诊断结果准备必要的维修资源

### 后续计划
1. **详细检查**：安排专业人员对设备进行详细检查
2. **维修规划**：制定详细的维修计划和进度安排
3. **预防措施**：建立预防性维护策略，避免类似故障复发

### 监测建议
1. **增加监测频率**：短期内增加振动监测频率
2. **趋势分析**：跟踪故障特征的变化趋势
3. **数据记录**：建立完整的设备健康档案

## 工具使用说明

### 系统功能
- **智能诊断**：基于振动信号的自动故障识别
- **自然语言解释**：将技术分析转换为易懂的描述
- **对话式咨询**：支持多轮技术问答和深入讨论
- **个性化建议**：根据设备状态提供定制化方案

### 下次使用
当设备出现新的问题时，您可以：
1. 上传新的振动数据
2. 描述具体的技术问题
3. 获得针对性的诊断和建议

## 技术支持

如果您在使用过程中遇到问题，或需要更深入的技术支持：
- 查看系统使用手册和教程
- 联系技术支持团队
- 参考相关的技术文献和标准

---

**安全提醒**：在实际应用中，请始终将安全放在首位，遵循相关的操作规程和安全标准。

祝您工作顺利，设备运行稳定！"""

        # Reset conversation state
        self._reset_conversation_state()

        return conclusion

    def get_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "agent_type": "ConversationAgent",
            "capabilities": [
                "intent_recognition",
                "context_management",
                "multi_turn_dialogue",
                "technical_explanation",
                "rule_based_fallback"
            ],
            "conversation_state": self.conversation_state,
            "supported_intents": list(self.intent_patterns.keys())
        }

    def _initialize_conversation_patterns(self):
        """Initialize conversation patterns and templates."""
        self.intent_patterns = {
            "cause_analysis": {
                "keywords": ["原因", "为什么", "why", "cause", "机理", "原理"],
                "response_type": "technical_explanation"
            },
            "maintenance_guidance": {
                "keywords": ["维修", "维护", "处理", "修复", "repair", "maintenance", "fix"],
                "response_type": "practical_guidance"
            },
            "severity_assessment": {
                "keywords": ["严重", "程度", "风险", "危险", "severity", "level", "risk"],
                "response_type": "risk_analysis"
            },
            "technical_explanation": {
                "keywords": ["解释", "说明", "技术", "原理", "explain", "technical", "how"],
                "response_type": "detailed_explanation"
            },
            "prevention_strategy": {
                "keywords": ["预防", "避免", "防止", "prevention", "avoid", "prevent"],
                "response_type": "prevention_guidance"
            },
            "monitoring_advice": {
                "keywords": ["监测", "监控", "观察", "monitor", "watch", "track"],
                "response_type": "monitoring_guidance"
            },
            "comparison_question": {
                "keywords": ["比较", "对比", "区别", "compare", "difference", "versus"],
                "response_type": "comparative_analysis"
            },
            "general_inquiry": {
                "keywords": ["什么", "如何", "怎么样", "what", "how"],
                "response_type": "general_response"
            }
        }

        # Response templates for different intents
        self.response_templates = {
            "cause_analysis": "关于 **{fault_type}** 的原因分析：{technical_explanation}",
            "maintenance_guidance": "针对 **{fault_type}** 的维修建议：{maintenance_steps}",
            "severity_assessment": "**{fault_type}** 的严重程度评估：{risk_analysis}",
            "technical_explanation": "**{fault_type}** 的技术解释：{technical_details}",
            "prevention_strategy": "**{fault_type}** 的预防措施：{prevention_steps}",
            "monitoring_advice": "**{fault_type}** 的监测建议：{monitoring_plan}",
            "comparison_question": "{comparison_analysis}",
            "general_inquiry": "{general_response}"
        }

    def _analyze_user_intent(self, user_message: str) -> str:
        """Analyze user intent from message."""
        message_lower = user_message.lower()

        # Check for specific intent keywords
        for intent, pattern in self.intent_patterns.items():
            for keyword in pattern["keywords"]:
                if keyword in message_lower:
                    return intent

        return "general_inquiry"

    def _update_conversation_state(self, intent: str, user_message: str):
        """Update conversation state based on intent."""
        self.conversation_state["user_questions_asked"].append({
            "intent": intent,
            "message": user_message,
            "timestamp": datetime.now().isoformat()
        })

        # Update topics discussed
        if intent not in self.conversation_state["topics_discussed"]:
            self.conversation_state["topics_discussed"].append(intent)

        # Update phase
        if len(self.conversation_state["user_questions_asked"]) > 1:
            self.conversation_state["current_phase"] = "deep_discussion"

    def _prepare_context(self, session: Dict[str, Any], intent: str) -> Dict[str, Any]:
        """Prepare context for response generation."""
        initial_diagnosis = session["initial_diagnosis"]
        device_info = session.get("device_info", {})

        return {
            "session_info": {
                "session_id": session.get("session_id"),
                "duration": (datetime.now() - session["start_time"]).total_seconds(),
                "num_messages": len(session["conversation_history"])
            },
            "diagnostic_context": {
                "fault_type": initial_diagnosis["model_prediction"]["fault_type"],
                "confidence": initial_diagnosis["model_prediction"]["confidence"],
                "explanation": initial_diagnosis["explanation"]
            },
            "device_context": device_info,
            "conversation_context": {
                "current_intent": intent,
                "topics_discussed": self.conversation_state["topics_discussed"],
                "conversation_phase": self.conversation_state["current_phase"]
            }
        }

    def _generate_rule_based_response(self,
                                     intent: str,
                                     user_message: str,
                                     session: Dict[str, Any]) -> str:
        """Generate rule-based response when LLM is unavailable."""
        fault_type = session["initial_diagnosis"]["model_prediction"]["fault_type"]
        confidence = session["initial_diagnosis"]["model_prediction"]["confidence"]

        # Get response template
        template = self.response_templates.get(intent, self.response_templates["general_inquiry"])

        # Generate content based on intent
        if intent == "cause_analysis":
            content = f"这种故障通常由以下原因引起：1) 正常磨损和材料疲劳 2) 润滑不良或污染 3) 过载运行或冲击载荷 4) 安装不当或对中不良。基于当前的置信度 {confidence:.1%}，建议首先检查设备的运行历史和维护记录。"

        elif intent == "maintenance_guidance":
            content = f"""建议的维修步骤：
1. **安全措施**：确保设备已停止运行，采取必要的安全防护
2. **详细检查**：检查{fault_type}相关的具体部件和状态
3. **评估损坏**：确定损坏程度和是否需要更换部件
4. **维修方案**：制定具体的维修计划和时间安排
5. **验证测试**：维修后进行功能测试和振动分析"""

        elif intent == "severity_assessment":
            severity_text = "高" if confidence > 0.8 else "中等" if confidence > 0.6 else "低"
            content = f"根据诊断置信度 {confidence:.1%}，评估故障严重程度为{severity_text}。建议的应对措施：{'立即停机检查' if confidence > 0.8 else '安排计划性检查' if confidence > 0.6 else '加强监测观察'}。"

        elif intent == "technical_explanation":
            content = f"{fault_type}是一种常见的机械故障。其主要特征包括：1) 特定的频率成分在频谱中出现 2) 振动幅值明显增加 3) 可能伴随温度或噪声异常。这种故障如果不及时处理，可能导致设备性能下降甚至完全失效。"

        elif intent == "prevention_strategy":
            content = f"预防{fault_type}的措施包括：1) 定期进行润滑和保养 2) 控制设备的运行载荷 3) 定期检查设备状态 4) 建立振动监测系统进行早期预警 5) 制定预防性维护计划。"

        elif intent == "monitoring_advice":
            content = f"建议的监测策略：1) 增加振动监测频率至每周一次 2) 重点监测{fault_type}的特征频率 3) 记录振动趋势变化 4) 设置适当的报警阈值 5) 定期分析监测数据并调整监测策略。"

        elif intent == "comparison_question":
            content = f"与其他故障类型相比，{fault_type}具有独特的特征：1) 特定的频率模式 2) 独特的演化规律 3) 特定的维修要求。准确的故障识别需要综合考虑多种因素，建议进行详细的专业分析。"

        else:
            content = f"关于您的设备问题（{fault_type}），我可以提供故障机理分析、维修建议、严重程度评估和预防措施等方面的帮助。请告诉我您希望了解哪个具体方面。"

        return template.format(
            fault_type=fault_type,
            technical_explanation=content,
            maintenance_steps=content,
            risk_analysis=content,
            technical_details=content,
            prevention_steps=content,
            monitoring_plan=content,
            comparison_analysis=content,
            general_response=content
        )

    def _summarize_discussed_topics(self) -> str:
        """Summarize topics discussed in conversation."""
        if not self.conversation_state["topics_discussed"]:
            return ""

        topic_descriptions = {
            "cause_analysis": "故障原因分析",
            "maintenance_guidance": "维修指导建议",
            "severity_assessment": "严重程度评估",
            "technical_explanation": "技术原理解释",
            "prevention_strategy": "预防策略制定",
            "monitoring_advice": "监测方案建议"
        }

        discussed_topics = [
            topic_descriptions.get(topic, topic)
            for topic in self.conversation_state["topics_discussed"]
            if topic in topic_descriptions
        ]

        if discussed_topics:
            return f"• 主要讨论了：{', '.join(discussed_topics)}"
        else:
            return ""

    def _reset_conversation_state(self):
        """Reset conversation state for new conversation."""
        self.conversation_state = {
            "current_phase": "greeting",
            "topics_discussed": [],
            "user_questions_asked": [],
            "clarifications_needed": []
        }