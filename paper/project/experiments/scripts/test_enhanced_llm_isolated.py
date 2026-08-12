#!/usr/bin/env python3
"""
Isolated test for enhanced template LLM
Includes all necessary classes to avoid import issues
"""

import sys
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import uuid


@dataclass
class FaultInfo:
    """Fault information structure."""
    fault_type: str
    confidence: float
    severity: str = "medium"
    description: str = ""


@dataclass
class SignalAnalysis:
    """Signal analysis information."""
    signal_length: int
    sampling_rate: float
    statistics: Dict[str, float]
    frequency_analysis: Dict[str, Any]
    key_findings: List[str]


@dataclass
class TechnicalExplanation:
    """Technical explanation information."""
    signal_path: Dict[str, Any]
    important_features: List[Dict[str, Any]]
    attention_weights: Dict[str, Any]
    layer_contributions: Dict[str, Any]
    processing_steps: List[Dict[str, Any]]


@dataclass
class DeviceContext:
    """Device context information."""
    device_type: str
    operating_conditions: Dict[str, Any]
    maintenance_history: str = ""
    specifications: str = ""


@dataclass
class LLMIntermediateRepresentation:
    """Complete intermediate representation for LLM."""
    explanation_id: str
    timestamp: str
    fault_info: FaultInfo
    signal_analysis: SignalAnalysis
    technical_explanation: TechnicalExplanation
    device_context: DeviceContext
    metadata: Dict[str, Any]


class IsolatedEnhancedLLM:
    """
    Isolated enhanced template-based LLM with multiple output styles.
    """

    def __init__(self, style: str = "standard", language: str = "zh"):
        """
        Initialize the enhanced template LLM.

        Args:
            style: Default explanation style
            language: Output language ("zh" for Chinese, "en" for English)
        """
        self.style = style
        self.language = language
        self.conversation_history = []
        self._initialize_enhanced_templates()
        self._initialize_fault_knowledge()

    def set_style(self, style: str):
        """Set the explanation style."""
        self.style = style

    def set_language(self, language: str):
        """Set the output language."""
        self.language = language

    def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate response using enhanced template-based approach.

        Args:
            prompt: Input prompt
            context: Context information (should contain intermediate representation)

        Returns:
            Generated natural language response
        """
        if not context or "intermediate_representation" not in context:
            return self._generate_error_response("缺少必要的上下文信息")

        ir = context["intermediate_representation"]

        # Add to conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "context": context
        })

        # Determine response type and style
        response_type = self._determine_response_type(prompt, ir)
        effective_style = self._determine_effective_style(prompt)

        # Generate response using appropriate template and style
        response = self._generate_styled_response(response_type, ir, effective_style)

        # Add style-specific formatting
        response = self._apply_style_formatting(response, effective_style)

        return response

    def _determine_response_type(self, prompt: str, ir) -> str:
        """Determine the type of response needed based on prompt analysis."""
        prompt_lower = prompt.lower()

        # Check for specific response types
        if any(keyword in prompt_lower for keyword in ["原因", "为什么", "why", "cause", "机理"]):
            return "cause_analysis"
        elif any(keyword in prompt_lower for keyword in ["维修", "维护", "修复", "repair", "fix", "处理"]):
            return "maintenance_guidance"
        elif any(keyword in prompt_lower for keyword in ["严重", "风险", "危险", "severity", "risk", "程度"]):
            return "severity_assessment"
        elif any(keyword in prompt_lower for keyword in ["技术", "详细", "原理", "technical", "detail", "具体"]):
            return "technical_details"
        elif any(keyword in prompt_lower for keyword in ["预防", "避免", "防止", "prevention", "avoid"]):
            return "prevention_strategy"
        elif any(keyword in prompt_lower for keyword in ["监测", "监控", "观察", "monitor", "watch", "检查"]):
            return "monitoring_advice"
        elif any(keyword in prompt_lower for keyword in ["总结", "概要", "summary", "overview"]):
            return "summary_report"
        elif any(keyword in prompt_lower for keyword in ["建议", "recommend", "advice"]):
            return "recommendations"
        else:
            return "general_explanation"

    def _determine_effective_style(self, prompt: str) -> str:
        """Determine the effective style based on prompt and current style."""
        prompt_lower = prompt.lower()

        # Check for style overrides in prompt
        if any(keyword in prompt_lower for keyword in ["简单", "易懂", "通俗", "simple"]):
            return "simple"
        elif any(keyword in prompt_lower for keyword in ["详细", "全面", "深入", "detailed", "comprehensive"]):
            return "detailed"
        elif any(keyword in prompt_lower for keyword in ["技术", "专业", "technical", "professional"]):
            return "technical"
        elif any(keyword in prompt_lower for keyword in ["简洁", "简短", "brief", "concise"]):
            return "concise"
        elif any(keyword in prompt_lower for keyword in ["报告", "正式", "report", "formal"]):
            return "formal"
        else:
            return self.style

    def _generate_styled_response(self, response_type: str, ir, style: str) -> str:
        """Generate response with specific style."""
        try:
            # Get base template
            templates = self.enhanced_templates.get(response_type, {})
            template = templates.get(style, templates.get("standard", ""))

            if not template:
                return f"无法找到适合的模板（类型：{response_type}，风格：{style}）"

            # Prepare template variables
            template_vars = self._prepare_template_variables(ir, response_type)

            # Format template
            response = template.format(**template_vars)

            return response

        except Exception as e:
            return f"生成响应时出错：{str(e)}"

    def _prepare_template_variables(self, ir, response_type: str) -> Dict[str, Any]:
        """Prepare template variables based on IR and response type."""
        fault_type = ir.fault_info.fault_type
        confidence = ir.fault_info.confidence
        device_type = ir.device_context.device_type

        # Basic variables
        variables = {
            "fault_type": fault_type,
            "confidence": f"{confidence:.1%}",
            "confidence_value": f"{confidence:.3f}",
            "device_type": device_type,
            "severity": ir.fault_info.severity,
            "description": ir.fault_info.description,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Signal analysis variables
        if ir.signal_analysis:
            rms = ir.signal_analysis.statistics.get('rms', 0)
            peak_factor = ir.signal_analysis.statistics.get('peak_factor', 0)
            variables.update({
                "signal_length": ir.signal_analysis.signal_length,
                "sampling_rate": ir.signal_analysis.sampling_rate,
                "rms": f"{rms:.2f}",
                "peak_factor": f"{peak_factor:.2f}",
                "dominant_frequency": f"{ir.signal_analysis.frequency_analysis.get('dominant_frequency', 0):.1f}",
                "spectral_centroid": f"{ir.signal_analysis.frequency_analysis.get('spectral_centroid', 0):.1f}"
            })

            # Key findings
            key_findings = ir.signal_analysis.key_findings[:3]
            variables["key_findings"] = "；".join(key_findings) if key_findings else "信号分析显示异常特征"
            variables["key_findings_list"] = "\n".join([f"• {finding}" for finding in key_findings])

        # Technical explanation variables
        if ir.technical_explanation:
            important_features = ir.technical_explanation.important_features[:3]
            variables["important_features"] = self._format_features(important_features)
            variables["processing_steps"] = self._format_processing_steps(ir.technical_explanation.processing_steps)
            variables["layer_contributions"] = self._format_layer_contributions(ir.technical_explanation.layer_contributions)

        # Fault-specific knowledge
        fault_knowledge = self.fault_knowledge.get(fault_type, {})
        variables.update({
            "fault_description": fault_knowledge.get("description", "设备故障"),
            "common_causes": self._format_list(fault_knowledge.get("causes", [])),
            "maintenance_steps": self._format_numbered_list(fault_knowledge.get("maintenance_steps", [])),
            "prevention_measures": self._format_list(fault_knowledge.get("prevention_measures", [])),
            "monitoring_points": self._format_list(fault_knowledge.get("monitoring_points", []))
        })

        # Response type specific variables
        if response_type == "severity_assessment":
            variables.update(self._get_severity_variables(ir))
        elif response_type == "maintenance_guidance":
            variables.update(self._get_maintenance_variables(ir))
        elif response_type == "recommendations":
            variables["recommendations"] = self._generate_recommendations(ir)

        return variables

    def _format_features(self, features: List[Dict[str, Any]]) -> str:
        """Format important features for display."""
        if not features:
            return "无显著特征"
        return "\n".join([f"• {feat['feature']}: {feat.get('value', 0):.2f} (重要性: {feat.get('significance', 0):.2f})"
                         for feat in features])

    def _format_processing_steps(self, steps: List[Dict[str, Any]]) -> str:
        """Format processing steps for display."""
        if not steps:
            return "无处理步骤信息"
        return "\n".join([f"{i+1}. {step.get('description', '')} ({step.get('layer', '')})"
                         for i, step in enumerate(steps)])

    def _format_layer_contributions(self, contributions: Dict[str, float]) -> str:
        """Format layer contributions for display."""
        if not contributions:
            return "无层贡献信息"
        sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        return "\n".join([f"• {layer}: {contrib:.2f}" for layer, contrib in sorted_contrib[:5]])

    def _format_list(self, items: List[str]) -> str:
        """Format list as bullet points."""
        if not items:
            return "无相关信息"
        return "\n".join([f"• {item}" for item in items])

    def _format_numbered_list(self, items: List[str]) -> str:
        """Format list as numbered items."""
        if not items:
            return "无相关信息"
        return "\n".join([f"{i+1}. {item}" for i, item in enumerate(items)])

    def _get_severity_variables(self, ir) -> Dict[str, Any]:
        """Get severity assessment variables."""
        confidence = ir.fault_info.confidence
        rms = ir.signal_analysis.statistics.get('rms', 0) if ir.signal_analysis else 0

        if confidence > 0.9 or rms > 15:
            severity_level = "严重"
            risk_level = "高风险"
            action_required = "立即停机检查"
        elif confidence > 0.7 or rms > 8:
            severity_level = "中等"
            risk_level = "中等风险"
            action_required = "计划性维修"
        else:
            severity_level = "轻微"
            risk_level = "低风险"
            action_required = "加强监测"

        return {
            "severity_level": severity_level,
            "risk_level": risk_level,
            "action_required": action_required
        }

    def _get_maintenance_variables(self, ir) -> Dict[str, Any]:
        """Get maintenance-specific variables."""
        confidence = ir.fault_info.confidence

        if confidence > 0.8:
            urgency = "紧急"
            time_frame = "24小时内"
            priority = "最高优先级"
        elif confidence > 0.6:
            urgency = "高优先级"
            time_frame = "一周内"
            priority = "高优先级"
        else:
            urgency = "计划性"
            time_frame = "下次维护窗口"
            priority = "中等优先级"

        return {
            "urgency": urgency,
            "time_frame": time_frame,
            "priority": priority
        }

    def _generate_recommendations(self, ir) -> str:
        """Generate specific recommendations based on IR."""
        recommendations = []

        fault_type = ir.fault_info.fault_type
        confidence = ir.fault_info.confidence

        # Basic recommendations based on confidence
        if confidence > 0.8:
            recommendations.append("立即安排专业技术人员进行现场检查")
            recommendations.append("考虑减少设备运行负载直至问题解决")

        # Fault-specific recommendations
        fault_recs = self.fault_knowledge.get(fault_type, {}).get("recommendations", [])
        recommendations.extend(fault_recs)

        # Signal-based recommendations
        if ir.signal_analysis:
            rms = ir.signal_analysis.statistics.get('rms', 0)
            if rms > 10:
                recommendations.append("监测振动能量水平，考虑临时减载运行")

        # Format recommendations
        if recommendations:
            return "\n".join([f"• {rec}" for rec in recommendations[:5]])
        else:
            return "• 继续正常监测设备运行状态"

    def _apply_style_formatting(self, response: str, style: str) -> str:
        """Apply style-specific formatting to the response."""
        if style == "formal":
            # Add formal header and footer
            header = "故障诊断分析报告\n" + "=" * 30 + "\n\n"
            footer = f"\n\n报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            return header + response + footer

        elif style == "simple":
            # Simplify technical terms
            response = response.replace("RMS值", "振动强度")
            response = response.replace("频谱分析", "频率分析")
            response = response.replace("特征频率", "主要频率")
            return response

        elif style == "concise":
            # Remove redundant phrases and make more concise
            response = re.sub(r'根据.*?分析，', '', response)
            response = re.sub(r'从.*?来看，', '', response)
            return response.strip()

        return response

    def _generate_error_response(self, error_message: str) -> str:
        """Generate error response."""
        return f"抱歉，处理您的请求时出现错误：{error_message}"

    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation."""
        if not self.conversation_history:
            return "暂无对话记录"

        summary = f"对话记录（共 {len(self.conversation_history)} 条）\n"
        summary += "=" * 40 + "\n"

        for i, turn in enumerate(self.conversation_history, 1):
            summary += f"\n{i}. 时间: {turn['timestamp'][:19]}\n"
            summary += f"   问题: {turn['prompt'][:50]}...\n"

        return summary

    def clear_conversation_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def _initialize_enhanced_templates(self):
        """Initialize enhanced templates for different response types and styles."""
        self.enhanced_templates = {
            "general_explanation": {
                "standard": """
故障诊断结果分析：

根据对{device_type}的信号分析，检测到{fault_type}，置信度为{confidence}。

主要发现：
{key_findings_list}

技术特征：
{important_features}

建议及时关注设备状态，必要时采取相应维护措施。
""",
                "simple": """
设备检测报告：

您的{device_type}出现了{fault_type}的迹象。
检测可信度：{confidence}

主要问题：{key_findings}

建议：请关注设备运行状态，如有异常请及时处理。
""",
                "detailed": """
{device_type}故障诊断详细报告

==========================================
故障基本信息
==========================================
故障类型：{fault_type}
检测置信度：{confidence}（{confidence_value}）
故障严重程度：{severity}
检测时间：{timestamp}

==========================================
信号分析结果
==========================================
信号长度：{signal_length} 采样点
采样频率：{sampling_rate} Hz
振动强度(RMS)：{rms}
峰值因子：{peak_factor}
主频分量：{dominant_frequency} Hz
频谱重心：{spectral_centroid} Hz

==========================================
关键发现
==========================================
{key_findings_list}

==========================================
技术特征分析
==========================================
{important_features}

==========================================
结论与建议
==========================================
基于上述分析，设备确认为{fault_type}，建议根据故障严重程度制定相应的维护策略。
""",
                "formal": """
关于{device_type}的故障诊断分析报告

1. 故障概述
   经系统分析，检测到设备存在{fault_type}情况，检测置信度为{confidence}。

2. 分析依据
   {key_findings}

3. 技术参数
   - 振动强度：{rms}
   - 主频特征：{dominant_frequency} Hz
   - 检测置信度：{confidence}

4. 初步结论
   基于信号特征分析，建议对设备进行进一步检查以确认故障状态。
""",
                "concise": """
检测结果：{fault_type}（置信度{confidence}）

关键发现：{key_findings}

建议：关注设备状态，必要时安排检查。
"""
            },
            "maintenance_guidance": {
                "standard": """
{fault_type}维修指导：

根据检测结果（置信度{confidence}），建议按以下步骤进行维修：

{maintenance_steps}

紧急程度：{urgency}
建议处理时间：{time_frame}

注意事项：
- 维修前请确保设备完全停止并断电
- 遵循安全操作规程
- 维修后进行功能测试验证
""",
                "simple": """
{fault_type}维修建议：

紧急程度：{urgency}

处理步骤：
{maintenance_steps}

请在{time_frame}内安排处理。
""",
                "detailed": """
{fault_type}详细维修指导方案

==========================================
维修优先级评估
==========================================
故障置信度：{confidence}
紧急程度：{urgency}
建议处理时间：{time_frame}
优先级：{priority}

==========================================
详细维修步骤
==========================================
{maintenance_steps}

==========================================
维修要点
==========================================
1. 安全第一：确保设备完全停止，采取安全防护措施
2. 精准定位：根据故障特征确定具体问题部位
3. 彻底检查：不仅处理已发现问题，还要排查潜在风险
4. 质量控制：使用合格备件，按技术规范操作
5. 测试验证：维修后进行全面功能测试

==========================================
后续监测建议
==========================================
{monitoring_points}
"""
            },
            "severity_assessment": {
                "standard": """
{fault_type}严重程度评估：

根据检测结果，评估如下：

严重程度：{severity_level}
风险等级：{risk_level}
检测置信度：{confidence}

技术指标：
- 振动强度：{rms}
- 主频特征：{dominant_frequency} Hz

建议措施：{action_required}

{recommendations}
""",
                "simple": """
故障严重程度：{severity_level}

风险等级：{risk_level}
建议：{action_required}

请根据实际情况及时处理。
"""
            },
            "recommendations": {
                "standard": """
针对{fault_type}的建议：

{recommendations}

优先级：{priority}
建议时间：{time_frame}

请根据实际情况安排执行。
""",
                "simple": """
{fault_type}处理建议：

{recommendations}

建议尽快处理。
"""
            }
        }

    def _initialize_fault_knowledge(self):
        """Initialize fault-specific knowledge base."""
        self.fault_knowledge = {
            "内圈故障": {
                "description": "轴承内圈表面出现疲劳、剥落或损伤",
                "causes": [
                    "材料疲劳和循环应力",
                    "润滑不良或污染物侵入",
                    "安装不当或过载运行",
                    "制造缺陷或材料质量问题"
                ],
                "maintenance_steps": [
                    "停止设备运行，确保安全",
                    "拆卸轴承进行检查",
                    "评估内圈损伤程度",
                    "清洁轴承及相关部件",
                    "更换损坏的内圈或整套轴承",
                    "检查润滑系统",
                    "重新安装并进行测试"
                ],
                "prevention_measures": [
                    "定期检查润滑状态",
                    "避免过载运行",
                    "确保正确的安装和配合",
                    "使用高质量的润滑剂",
                    "定期监测设备振动"
                ],
                "monitoring_points": [
                    "监测振动RMS值变化",
                    "关注特征频率幅值",
                    "检查温度异常",
                    "监听异常噪声"
                ],
                "recommendations": [
                    "立即检查轴承润滑状态",
                    "评估设备运行负载是否合理",
                    "制定轴承更换计划",
                    "加强振动监测频率"
                ]
            },
            "外圈故障": {
                "description": "轴承外圈表面出现疲劳裂纹或剥落",
                "causes": [
                    "座孔变形或配合不当",
                    "外部振动或冲击载荷",
                    "润滑失效或污染",
                    "材料疲劳或制造缺陷"
                ],
                "maintenance_steps": [
                    "停止设备运行",
                    "检查轴承座和配合",
                    "拆卸轴承进行详细检查",
                    "测量座孔尺寸和形位公差",
                    "更换损坏部件",
                    "重新装配并调整配合",
                    "进行功能测试"
                ],
                "prevention_measures": [
                    "确保正确的座孔加工",
                    "控制外部振动源",
                    "定期检查配合状态",
                    "保持良好的润滑",
                    "避免过大的预载荷"
                ],
                "monitoring_points": [
                    "监测座孔温度",
                    "检查配合松动",
                    "关注外部振动",
                    "监听运行噪声"
                ],
                "recommendations": [
                    "检查轴承座状况",
                    "评估外部振动影响",
                    "调整轴承配合",
                    "考虑使用减振措施"
                ]
            },
            "不对中": {
                "description": "旋转轴系中心线不在同一直线上",
                "causes": [
                    "安装误差或基础沉降",
                    "热变形引起的位置变化",
                    "负载变化导致的变形",
                    "联轴器损坏或磨损"
                ],
                "maintenance_steps": [
                    "测量当前对中状态",
                    "检查基础和支撑结构",
                    "调整设备位置和对中",
                    "检查联轴器状态",
                    "重新进行对中调整",
                    "验证调整效果",
                    "制定定期检查计划"
                ],
                "prevention_measures": [
                    "精确的初始安装对中",
                    "定期检查对中状态",
                    "控制热变形影响",
                    "使用柔性联轴器",
                    "监测基础沉降"
                ],
                "monitoring_points": [
                    "监测轴向和径向振动",
                    "检查轴承温度分布",
                    "测量轴位置变化",
                    "监听异常噪声"
                ],
                "recommendations": [
                    "进行激光对中检查",
                    "检查联轴器状态",
                    "评估基础稳定性",
                    "制定定期对中检查计划"
                ]
            },
            "正常状态": {
                "description": "设备运行正常，未检测到明显故障特征",
                "causes": [],
                "maintenance_steps": [
                    "继续正常监测",
                    "按计划进行预防性维护",
                    "记录运行参数",
                    "保持良好的运行环境"
                ],
                "prevention_measures": [
                    "定期检查和保养",
                    "监测运行参数趋势",
                    "及时处理早期异常",
                    "保持良好的操作习惯"
                ],
                "monitoring_points": [
                    "继续常规振动监测",
                    "关注参数变化趋势",
                    "记录运行状态",
                    "定期设备检查"
                ],
                "recommendations": [
                    "保持当前维护计划",
                    "继续监测设备状态",
                    "记录基准运行参数",
                    "计划下次维护时间"
                ]
            }
        }


def create_sample_ir():
    """Create a sample intermediate representation for testing."""
    class MockIR:
        def __init__(self):
            self.explanation_id = "demo_001"
            self.timestamp = datetime.now().isoformat()

            class FaultInfo:
                def __init__(self):
                    self.fault_type = "内圈故障"
                    self.confidence = 0.89
                    self.severity = "高"
                    self.description = "轴承内圈疲劳损伤"

            class SignalAnalysis:
                def __init__(self):
                    self.signal_length = 4096
                    self.sampling_rate = 1024.0
                    self.statistics = {"rms": 12.5, "peak_factor": 4.2}
                    self.frequency_analysis = {"dominant_frequency": 157.3, "spectral_centroid": 200.1}
                    self.key_findings = [
                        "振动能量显著增高，RMS值达12.5",
                        "检测到157.3Hz特征频率分量",
                        "时域信号显示周期性冲击特征"
                    ]

            class TechnicalExplanation:
                def __init__(self):
                    self.important_features = [
                        {"feature": "RMS值", "value": 12.5, "significance": 0.92},
                        {"feature": "峰值因子", "value": 4.2, "significance": 0.87},
                        {"feature": "主频幅值", "value": 157.3, "significance": 0.95}
                    ]
                    self.processing_steps = [
                        {"layer": "Input", "description": "原始振动信号输入"},
                        {"layer": "SignalProcessing", "description": "多路径信号处理"},
                        {"layer": "FeatureExtraction", "description": "多域特征提取"},
                        {"layer": "Attention", "description": "注意力机制加权"},
                        {"layer": "Classification", "description": "故障分类输出"}
                    ]
                    self.layer_contributions = {
                        "layer1_fft": 0.35,
                        "layer1_wf": 0.28,
                        "layer1_ht": 0.22,
                        "feature_extractor": 0.78,
                        "attention": 0.65,
                        "classifier": 1.0
                    }

            class DeviceContext:
                def __init__(self):
                    self.device_type = "滚动轴承6205"
                    self.operating_conditions = {"speed": 1800.0, "load": "中等载荷"}
                    self.maintenance_history = "上次维护：3个月前，更换润滑脂"
                    self.specifications = "内径25mm, 外径52mm, 宽度15mm"

            self.fault_info = FaultInfo()
            self.signal_analysis = SignalAnalysis()
            self.technical_explanation = TechnicalExplanation()
            self.device_context = DeviceContext()

    return MockIR()


def test_enhanced_llm_isolated():
    """Test the isolated enhanced template LLM."""
    print("🚀 隔离增强版模板LLM测试")
    print("=" * 50)

    # Create enhanced LLM
    llm = IsolatedEnhancedLLM(style="standard")

    # Create sample IR
    ir = create_sample_ir()

    # Test different styles
    test_cases = [
        {"prompt": "请解释这个故障", "style": "standard"},
        {"prompt": "请详细分析", "style": "detailed"},
        {"prompt": "用简单的话说", "style": "simple"},
        {"prompt": "正式报告", "style": "formal"},
        {"prompt": "维修建议", "style": "standard"},
        {"prompt": "严重程度", "style": "standard"}
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 测试 {i}: {case['prompt']} (风格: {case['style']})")
        print("-" * 60)

        llm.set_style(case['style'])
        context = {"intermediate_representation": ir}
        response = llm.generate(case['prompt'], context)

        # Limit output length for display
        display_response = response[:500] + "..." if len(response) > 500 else response
        print(display_response)

    print(f"\n📊 对话历史:")
    print(llm.get_conversation_summary())

    return True


def main():
    """Main function."""
    try:
        success = test_enhanced_llm_isolated()
        if success:
            print("\n🎉 增强版模板LLM测试成功！")
            print("\n主要特性:")
            print("- ✅ 多种输出风格（标准、详细、简单、正式、简洁）")
            print("- ✅ 智能意图识别")
            print("- ✅ 丰富的故障知识库")
            print("- ✅ 对话历史功能")
        return success
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)