"""
Prompt Manager for LLM-Enhanced Explainability

Provides domain-specific prompt templates and management for generating
contextual explanations in mechanical fault diagnosis.
"""

from typing import Dict, Any, List, Optional, Union
import json
from pathlib import Path
from datetime import datetime


class PromptManager:
    """
    Manages prompt templates and builds domain-specific prompts for LLM explanations.

    This class provides structured templates for different types of explanations
    and queries in mechanical fault diagnosis, ensuring consistent and comprehensive
    prompt generation.
    """

    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize PromptManager with domain-specific templates.

        Args:
            template_dir: Directory containing custom prompt templates
        """
        self.template_dir = Path(template_dir) if template_dir else None
        self.templates = self._load_default_templates()
        self.domain_context = self._get_domain_context()

        if self.template_dir and self.template_dir.exists():
            self._load_custom_templates()

    def build_prompt(self,
                    encoded_explanation: str,
                    encoded_technical_summary: str,
                    user_query: Optional[str] = None,
                    encoded_context: Optional[str] = None,
                    prompt_type: str = "comprehensive",
                    language: str = "zh") -> str:
        """
        Build a comprehensive prompt for LLM explanation.

        Args:
            encoded_explanation: Encoded technical explanation
            encoded_technical_summary: Encoded technical summary
            user_query: Optional user query
            encoded_context: Encoded context information
            prompt_type: Type of prompt ('comprehensive', 'focused', 'conversational')
            language: Language for the response ('zh' or 'en')

        Returns:
            Formatted prompt string
        """
        template = self.templates.get(prompt_type, self.templates["comprehensive"])

        # Get language-specific templates
        if language == "zh":
            role_instruction = self.domain_context["role_instruction_zh"]
            context_info = self.domain_context["context_info_zh"]
        else:
            role_instruction = self.domain_context["role_instruction_en"]
            context_info = self.domain_context["context_info_en"]

        # Build prompt components
        prompt_components = {
            "role_instruction": role_instruction,
            "domain_context": context_info,
            "technical_explanation": encoded_explanation,
            "technical_summary": encoded_technical_summary,
            "additional_context": encoded_context or "无额外上下文信息。",
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_query": user_query or ""
        }

        # Format the prompt
        try:
            prompt = template.format(**prompt_components)
        except KeyError as e:
            # Fallback to simple template if formatting fails
            prompt = self._build_fallback_prompt(prompt_components, language)

        return prompt

    def build_conversation_prompt(self,
                                conversation_history: List[Dict[str, str]],
                                new_query: str,
                                context: Dict[str, Any],
                                language: str = "zh") -> str:
        """
        Build prompt for conversational interaction.

        Args:
            conversation_history: List of previous conversation turns
            new_query: New user query
            context: Current diagnostic context
            language: Language for response

        Returns:
            Formatted conversation prompt
        """
        template = self.templates["conversational"]

        # Format conversation history
        history_text = self._format_conversation_history(conversation_history, language)

        # Get context summary
        context_summary = self._summarize_context(context, language)

        prompt_components = {
            "role_instruction": self.domain_context[f"role_instruction_{language}"],
            "conversation_history": history_text,
            "new_query": new_query,
            "context_summary": context_summary,
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            prompt = template.format(**prompt_components)
        except KeyError:
            prompt = self._build_conversation_fallback(prompt_components, language)

        return prompt

    def build_focused_prompt(self,
                           focus_area: str,
                           explanation_data: Dict[str, Any],
                           language: str = "zh") -> str:
        """
        Build focused prompt for specific explanation aspect.

        Args:
            focus_area: Area to focus on ('fault_mechanism', 'maintenance', 'severity')
            explanation_data: Explanation data
            language: Language for response

        Returns:
            Focused prompt string
        """
        focus_templates = {
            "fault_mechanism": self.templates["fault_mechanism_focus"],
            "maintenance": self.templates["maintenance_focus"],
            "severity": self.templates["severity_focus"]
        }

        template = focus_templates.get(focus_area, self.templates["comprehensive"])

        # Get focus-specific context
        focus_context = self._get_focus_context(focus_area, language)

        prompt_components = {
            "role_instruction": self.domain_context[f"role_instruction_{language}"],
            "focus_context": focus_context,
            "explanation_data": self._format_explanation_data(explanation_data, language),
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            prompt = template.format(**prompt_components)
        except KeyError:
            prompt = self._build_fallback_prompt(prompt_components, language)

        return prompt

    def _load_default_templates(self) -> Dict[str, str]:
        """Load default prompt templates."""
        return {
            "comprehensive": self._get_comprehensive_template(),
            "focused": self._get_focused_template(),
            "conversational": self._get_conversational_template(),
            "fault_mechanism_focus": self._get_fault_mechanism_template(),
            "maintenance_focus": self._get_maintenance_template(),
            "severity_focus": self._get_severity_template()
        }

    def _get_comprehensive_template(self) -> str:
        """Get comprehensive explanation template."""
        return """你是一位专业的机械故障诊断专家，具有丰富的旋转机械故障分析经验。请基于以下技术诊断信息，提供一个全面、准确且易于理解的故障解释。

## 角色设定
{role_instruction}

## 领域知识背景
{domain_context}

## 技术诊断信息
### 1. 信号处理路径分析
{technical_explanation}

### 2. 技术摘要
{technical_summary}

### 3. 系统上下文
{additional_context}

### 4. 当前时间
{current_time}

## 用户查询
{user_query}

## 任务要求
请基于上述技术信息，提供以下内容的详细解释：

1. **故障识别与定位**：明确指出识别出的故障类型、位置和严重程度
2. **机理分析**：深入解释故障产生的物理机理和信号特征
3. **证据链分析**：说明技术数据如何支持诊断结论
4. **工程建议**：提供具体的维修、维护建议
5. **风险评估**：评估故障风险和紧急程度

## 输出格式
请按照以下结构输出：

### 🔍 故障诊断结论
[明确指出故障类型、位置、严重程度]

### ⚙️ 故障机理分析
[详细解释故障产生的物理原理和信号特征关系]

### 📊 技术证据支持
[分析关键信号特征、频谱成分、能量分布等技术证据]

### 🛠️ 工程建议
[提供具体维修步骤、时间窗口、资源需求]

### ⚠️ 风险评估
[评估故障风险等级、紧急程度、潜在后果]

### 💡 预防措施
[长期预防建议和监测策略]

请确保解释专业、准确，同时便于工程师理解和操作。"""

    def _get_conversational_template(self) -> str:
        """Get conversational interaction template."""
        return """你是一位专业的机械故障诊断顾问，正在与工程师进行诊断对话。

## 角色设定
{role_instruction}

## 对话历史
{conversation_history}

## 当前查询
用户问题：{new_query}

## 诊断上下文
{context_summary}

## 当前时间
{current_time}

## 任务要求
请基于对话历史和当前诊断上下文，针对用户的问题提供专业、准确的回答：

1. **直接回应**：明确回答用户的问题
2. **技术解释**：必要时提供技术背景和原理
3. **实用建议**：给出可操作的建议
4. **后续提问**：如需要更多信息，提出相关问题

请保持对话的连贯性和专业性。"""

    def _get_fault_mechanism_template(self) -> str:
        """Get fault mechanism focused template."""
        return """你专注于机械故障机理分析的专家。请深入分析故障产生的物理原理。

## 分析重点
{focus_context}

## 诊断数据
{explanation_data}

## 任务要求
请详细解释：
1. **故障机理**：从物理学角度解释故障产生过程
2. **信号特征**：说明故障如何体现在振动信号中
3. **传播路径**：分析故障如何通过机械系统传播
4. **演化过程**：预测故障可能的发展趋势

请提供深入的技术分析和物理原理解释。"""

    def _get_maintenance_template(self) -> str:
        """Get maintenance focused template."""
        return """你是设备维护规划专家。请基于诊断结果制定维护策略。

## 维护重点
{focus_context}

## 诊断依据
{explanation_data}

## 任务要求
请提供：
1. **维护优先级**：基于故障严重程度确定处理顺序
2. **维修方案**：具体的维修步骤和方法
3. **资源需求**：所需人员、工具、备件和时间
4. **风险控制**：维修过程中的安全注意事项
5. **验证方法**：维修后的效果验证方案

请确保建议切实可行且安全有效。"""

    def _get_severity_template(self) -> str:
        """Get severity assessment focused template."""
        return """你是故障风险评估专家。请评估故障严重程度和风险等级。

## 评估重点
{focus_context}

## 诊断数据
{explanation_data}

## 任务要求
请评估：
1. **故障等级**：根据技术指标确定严重程度
2. **风险分析**：评估对设备和生产的影响
3. **紧急程度**：确定处理的时效性要求
4. **影响范围**：分析可能扩散到的其他部件
5. **监控建议**：提出后续监控的关键指标

请提供量化的风险评估和应对建议。"""

    def _get_domain_context(self) -> Dict[str, str]:
        """Get domain-specific context information."""
        return {
            "role_instruction_zh": "你具有20年旋转机械故障诊断经验，精通各类轴承、齿轮、轴系故障的特征识别和分析方法。",
            "role_instruction_en": "You have 20 years of experience in rotating machinery fault diagnosis, expert in identifying and analyzing various bearing, gear, and shaft system faults.",
            "context_info_zh": """专业领域：
- 旋转机械故障诊断（轴承、齿轮、轴系、联轴器）
- 振动信号分析（时域、频域、时频域）
- 故障机理研究（疲劳、磨损、腐蚀、断裂等）
- 预测性维护和状态监测

技术专长：
- FFT频谱分析、包络分析、时频分析
- 故障特征频率计算和识别
- 信号路径追踪和能量传递分析
- 多传感器数据融合诊断""",
            "context_info_en": """Expertise Areas:
- Rotating machinery fault diagnosis (bearings, gears, shafts, couplings)
- Vibration signal analysis (time domain, frequency domain, time-frequency domain)
- Fault mechanism research (fatigue, wear, corrosion, fracture)
- Predictive maintenance and condition monitoring

Technical Skills:
- FFT spectrum analysis, envelope analysis, time-frequency analysis
- Fault characteristic frequency calculation and identification
- Signal path tracking and energy transfer analysis
- Multi-sensor data fusion diagnosis"""
        }

    def _format_conversation_history(self, history: List[Dict[str, str]], language: str) -> str:
        """Format conversation history for prompt."""
        if not history:
            return "无对话历史" if language == "zh" else "No conversation history"

        history_lines = []
        for i, turn in enumerate(history[-5:], 1):  # Keep last 5 turns
            role = "用户" if turn["role"] == "user" else "专家"
            content = turn["content"]
            history_lines.append(f"{i}. {role}: {content}")

        return "\n".join(history_lines)

    def _summarize_context(self, context: Dict[str, Any], language: str) -> str:
        """Summarize diagnostic context."""
        if not context:
            return "无可用上下文" if language == "zh" else "No available context"

        summary_parts = []

        if "model_name" in context:
            model_info = f"诊断模型: {context['model_name']}" if language == "zh" else f"Diagnostic model: {context['model_name']}"
            summary_parts.append(model_info)

        if "input_statistics" in context:
            stats = context["input_statistics"]
            if language == "zh":
                stats_info = f"信号统计: 均值={stats.get('mean', 0):.3f}, RMS={stats.get('rms', 0):.3f}"
            else:
                stats_info = f"Signal statistics: mean={stats.get('mean', 0):.3f}, RMS={stats.get('rms', 0):.3f}"
            summary_parts.append(stats_info)

        return "\n".join(summary_parts)

    def _get_focus_context(self, focus_area: str, language: str) -> str:
        """Get focus-specific context."""
        focus_contexts = {
            "fault_mechanism": {
                "zh": "专注于分析故障产生的物理机理和信号特征的因果关系",
                "en": "Focus on analyzing the physical mechanisms of fault generation and causal relationships with signal characteristics"
            },
            "maintenance": {
                "zh": "专注于制定实用的维护维修策略和实施方案",
                "en": "Focus on developing practical maintenance and repair strategies and implementation plans"
            },
            "severity": {
                "zh": "专注于评估故障严重程度和风险等级",
                "en": "Focus on assessing fault severity and risk levels"
            }
        }

        return focus_contexts.get(focus_area, focus_contexts["fault_mechanism"]).get(
            language, focus_contexts["fault_mechanism"]["en"]
        )

    def _format_explanation_data(self, data: Dict[str, Any], language: str) -> str:
        """Format explanation data for prompt."""
        if not data:
            return "无解释数据" if language == "zh" else "No explanation data"

        formatted_parts = []

        for key, value in data.items():
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value, ensure_ascii=False, indent=2)
            else:
                value_str = str(value)

            if language == "zh":
                formatted_parts.append(f"{key}: {value_str}")
            else:
                formatted_parts.append(f"{key}: {value_str}")

        return "\n".join(formatted_parts)

    def _build_fallback_prompt(self, components: Dict[str, str], language: str) -> str:
        """Build fallback prompt if template formatting fails."""
        if language == "zh":
            return f"""请基于以下信息提供故障诊断解释：

技术解释：{components.get('technical_explanation', '')}
技术摘要：{components.get('technical_summary', '')}
用户查询：{components.get('user_query', '')}

请提供专业的故障诊断分析和建议。"""
        else:
            return f"""Please provide fault diagnosis explanation based on the following information:

Technical explanation: {components.get('technical_explanation', '')}
Technical summary: {components.get('technical_summary', '')}
User query: {components.get('user_query', '')}

Please provide professional fault diagnosis analysis and recommendations."""

    def _build_conversation_fallback(self, components: Dict[str, str], language: str) -> str:
        """Build fallback conversation prompt."""
        if language == "zh":
            return f"""对话历史：{components.get('conversation_history', '')}
用户问题：{components.get('new_query', '')}
上下文：{components.get('context_summary', '')}

请针对用户问题提供专业回答。"""
        else:
            return f"""Conversation history: {components.get('conversation_history', '')}
User question: {components.get('new_query', '')}
Context: {components.get('context_summary', '')}

Please provide a professional answer to the user's question."""

    def _load_custom_templates(self) -> None:
        """Load custom templates from directory."""
        if not self.template_dir or not self.template_dir.exists():
            return

        for template_file in self.template_dir.glob("*.txt"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_name = template_file.stem
                    self.templates[template_name] = f.read()
            except Exception as e:
                print(f"Warning: Could not load template {template_file}: {e}")

    def add_template(self, name: str, template: str) -> None:
        """Add or update a template."""
        self.templates[name] = template

    def get_template(self, name: str) -> Optional[str]:
        """Get a specific template."""
        return self.templates.get(name)

    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self.templates.keys())