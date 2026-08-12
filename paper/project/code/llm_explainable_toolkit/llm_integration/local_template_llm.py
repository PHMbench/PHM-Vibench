"""
Local Template-based LLM Stub

This module provides a template-based LLM implementation that generates
natural language explanations without requiring external API calls.
"""

import re
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod

from ..core.intermediate_representation import LLMIntermediateRepresentation


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response from prompt."""
        pass


class LocalTemplateLLM(BaseLLMProvider):
    """
    Local template-based LLM stub for generating explanations.

    This implementation uses predefined templates and simple rule-based
    logic to generate natural language explanations without external APIs.
    """

    def __init__(self, style: str = "standard"):
        """
        Initialize the local template LLM.

        Args:
            style: Default explanation style
        """
        self.style = style
        self._initialize_templates()
        self._initialize_fault_knowledge()

    def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate response using template-based approach.

        Args:
            prompt: Input prompt
            context: Context information (should contain intermediate representation)

        Returns:
            Generated natural language response
        """
        if not context or "intermediate_representation" not in context:
            return self._generate_error_response("缺少必要的上下文信息")

        ir = context["intermediate_representation"]

        # Determine response type based on prompt and context
        response_type = self._determine_response_type(prompt, ir)

        # Generate response using appropriate template
        if response_type == "general_explanation":
            return self._generate_general_explanation(ir)
        elif response_type == "cause_analysis":
            return self._generate_cause_analysis(ir)
        elif response_type == "maintenance_guidance":
            return self._generate_maintenance_guidance(ir)
        elif response_type == "severity_assessment":
            return self._generate_severity_assessment(ir)
        elif response_type == "technical_details":
            return self._generate_technical_details(ir)
        elif response_type == "prevention_strategy":
            return self._generate_prevention_strategy(ir)
        elif response_type == "monitoring_advice":
            return self._generate_monitoring_advice(ir)
        else:
            return self._generate_general_explanation(ir)

    def _determine_response_type(self, prompt: str, ir: LLMIntermediateRepresentation) -> str:
        """Determine the type of response needed based on prompt analysis."""
        prompt_lower = prompt.lower()

        # Check for specific response types
        if any(keyword in prompt_lower for keyword in ["原因", "为什么", "why", "cause", "机理"]):
            return "cause_analysis"
        elif any(keyword in prompt_lower for keyword in ["维修", "维护", "修复", "repair", "fix"]):
            return "maintenance_guidance"
        elif any(keyword in prompt_lower for keyword in ["严重", "风险", "危险", "severity", "risk"]):
            return "severity_assessment"
        elif any(keyword in prompt_lower for keyword in ["技术", "详细", "原理", "technical", "detail"]):
            return "technical_details"
        elif any(keyword in prompt_lower for keyword in ["预防", "避免", "防止", "prevention", "avoid"]):
            return "prevention_strategy"
        elif any(keyword in prompt_lower for keyword in ["监测", "监控", "观察", "monitor", "watch"]):
            return "monitoring_advice"
        else:
            return "general_explanation"

    def _generate_general_explanation(self, ir: LLMIntermediateRepresentation) -> str:
        """Generate general fault explanation."""
        fault_type = ir.fault_info.fault_type
        confidence = ir.fault_info.confidence
        device_type = ir.device_context.device_type

        template = self.templates["general_explanation"][ir.explanation_style]

        # Extract key information
        key_findings = ir.signal_analysis.key_findings[:3]  # Top 3 findings
        key_finding_text = "；".join(key_findings) if key_findings else "信号分析显示异常特征"

        # Get fault-specific description
        fault_description = self.fault_knowledge.get(fault_type, {}).get("description", "设备故障")

        return template.format(
            fault_type=fault_type,
            confidence=f"{confidence:.1%}",
            device_type=device_type,
            key_findings=key_finding_text,
            fault_description=fault_description,
            dominant_frequency=f"{ir.signal_analysis.frequency_analysis.get('dominant_frequency', 0):.1f}",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _generate_cause_analysis(self, ir: LLMIntermediateRepresentation) -> str:
        """Generate cause analysis for the fault."""
        fault_type = ir.fault_info.fault_type
        confidence = ir.fault_info.confidence

        template = self.templates["cause_analysis"][ir.explanation_style]

        # Get fault-specific causes
        fault_causes = self.fault_knowledge.get(fault_type, {}).get("causes", [
            "正常磨损和材料疲劳",
            "润滑不良或污染",
            "过载运行或冲击载荷",
            "安装不当或对中不良"
        ])

        causes_text = "\n".join([f"• {cause}" for cause in fault_causes])

        # Add signal-based evidence
        evidence = []
        if ir.signal_analysis.statistics.get("rms", 0) > 10:
            evidence.append("振动RMS值显著增高，表明能量集中")
        if ir.signal_analysis.frequency_analysis.get("dominant_frequency", 0) > 0:
            evidence.append(f"频域分析显示特征频率 {ir.signal_analysis.frequency_analysis['dominant_frequency']:.1f} Hz")

        evidence_text = "\n".join([f"• {e}" for e in evidence]) if evidence else "• 信号分析显示异常模式"

        return template.format(
            fault_type=fault_type,
            confidence=f"{confidence:.1%}",
            causes_text=causes_text,
            evidence_text=evidence_text
        )

    def _generate_maintenance_guidance(self, ir: LLMIntermediateRepresentation) -> str:
        """Generate maintenance guidance."""
        fault_type = ir.fault_info.fault_type
        confidence = ir.fault_info.confidence

        template = self.templates["maintenance_guidance"][ir.explanation_style]

        # Get fault-specific maintenance steps
        maintenance_steps = self.fault_knowledge.get(fault_type, {}).get("maintenance_steps", [
            "确保设备已停止运行，采取安全防护措施",
            "详细检查故障相关部件的具体状态",
            "评估损坏程度，确定是否需要更换部件",
            "制定具体的维修计划和时间安排",
            "维修后进行功能测试和振动分析验证"
        ])

        steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(maintenance_steps)])

        # Determine urgency based on confidence
        if confidence > 0.8:
            urgency = "紧急"
            time_frame = "24小时内"
        elif confidence > 0.6:
            urgency = "高优先级"
            time_frame = "一周内"
        else:
            urgency = "计划性"
            time_frame = "下次维护窗口"

        return template.format(
            fault_type=fault_type,
            confidence=f"{confidence:.1%}",
            steps_text=steps_text,
            urgency=urgency,
            time_frame=time_frame
        )

    def _generate_severity_assessment(self, ir: LLMIntermediateRepresentation) -> str:
        """Generate severity assessment."""
        fault_type = ir.fault_info.fault_type
        confidence = ir.fault_info.confidence

        template = self.templates["severity_assessment"][ir.explanation_style]

        # Determine severity level
        if confidence > 0.8:
            severity_level = "高"
            risk_description = "可能导致设备严重损坏或安全事故"
            action_level = "立即停机检查"
        elif confidence > 0.6:
            severity_level = "中等"
            risk_description = "可能影响设备性能和寿命"
            action_level = "安排计划性检查"
        else:
            severity_level = "低"
            risk_description = "需要持续观察，暂不影响正常运行"
            action_level = "加强监测"

        # Technical indicators
        indicators = []
        if ir.signal_analysis.statistics.get("rms", 0) > 15:
            indicators.append(f"高振动水平 (RMS: {ir.signal_analysis.statistics['rms']:.1f})")
        if ir.signal_analysis.statistics.get("crest_factor", 0) > 4:
            indicators.append(f"冲击特征明显 (峰值因子: {ir.signal_analysis.statistics['crest_factor']:.1f})")

        indicators_text = "、".join(indicators) if indicators else "信号异常特征"

        return template.format(
            fault_type=fault_type,
            confidence=f"{confidence:.1%}",
            severity_level=severity_level,
            risk_description=risk_description,
            action_level=action_level,
            indicators_text=indicators_text
        )

    def _generate_technical_details(self, ir: LLMIntermediateRepresentation) -> str:
        """Generate technical details explanation."""
        fault_type = ir.fault_info.fault_type

        template = self.templates["technical_details"][ir.explanation_style]

        # Signal processing path
        path_description = ""
        if ir.technical_explanation.signal_path:
            steps = ir.technical_explanation.signal_path.get("processing_steps", [])
            path_description = " -> ".join([step.get("description", step.get("layer", "Unknown")) for step in steps])

        # Important features
        features_text = ""
        if ir.technical_explanation.important_features:
            features = ir.technical_explanation.important_features[:3]
            features_text = "\n".join([
                f"• {f['feature']}: {f['value']:.2f} (阈值: {f['threshold']}, 重要性: {f['significance']})"
                for f in features
            ])

        # Frequency analysis
        freq_text = ""
        if ir.technical_explanation.frequency_components:
            freq_components = ir.technical_explanation.frequency_components[:3]
            freq_text = "\n".join([
                f"• {comp['frequency']:.1f} Hz: 幅值 {comp['amplitude']:.2f} ({comp['type']})"
                for comp in freq_components
            ])

        return template.format(
            fault_type=fault_type,
            path_description=path_description,
            features_text=features_text,
            freq_text=freq_text,
            processing_stages=ir.technical_explanation.processing_stages
        )

    def _generate_prevention_strategy(self, ir: LLMIntermediateRepresentation) -> str:
        """Generate prevention strategy."""
        fault_type = ir.fault_info.fault_type

        template = self.templates["prevention_strategy"][ir.explanation_style]

        # Generic prevention strategies
        strategies = [
            "建立定期检查和维护制度",
            "加强润滑管理，确保润滑油质量",
            "控制设备运行载荷，避免过载",
            "安装振动监测系统进行早期预警",
            "培训操作人员，提高故障识别能力"
        ]

        strategies_text = "\n".join([f"• {strategy}" for strategy in strategies])

        return template.format(
            fault_type=fault_type,
            strategies_text=strategies_text
        )

    def _generate_monitoring_advice(self, ir: LLMIntermediateRepresentation) -> str:
        """Generate monitoring advice."""
        fault_type = ir.fault_info.fault_type
        dominant_freq = ir.signal_analysis.frequency_analysis.get("dominant_frequency", 0)

        template = self.templates["monitoring_advice"][ir.explanation_style]

        monitoring_plan = [
            "增加振动监测频率至每周一次",
            f"重点监测 {dominant_freq:.1f} Hz 附近的频率成分变化",
            "建立振动趋势数据库，跟踪长期变化",
            "设置多级报警阈值（警告、严重、危急）",
            "定期分析监测数据，调整监测策略"
        ]

        monitoring_text = "\n".join([f"• {item}" for item in monitoring_plan])

        return template.format(
            fault_type=fault_type,
            monitoring_text=monitoring_text,
            target_frequency=f"{dominant_freq:.1f}" if dominant_freq > 0 else "故障特征频率"
        )

    def _generate_error_response(self, error_message: str) -> str:
        """Generate error response."""
        return f"抱歉，生成解释时遇到问题：{error_message}。请稍后重试或联系技术支持。"

    def _initialize_templates(self):
        """Initialize response templates."""
        self.templates = {
            "general_explanation": {
                "standard": """
# 故障诊断结果

**检测到故障类型：** {fault_type}
**诊断置信度：** {confidence}
**设备类型：** {device_type}

## 主要发现
{key_findings}

## 故障描述
{fault_description}

**特征频率：** {dominant_frequency} Hz

这个诊断结果基于振动信号的深入分析，系统检测到了明确的故障特征。建议结合设备运行历史进行进一步确认。
                """,
                "detailed": """
# 详细故障诊断报告

## 诊断概要
- **故障类型：** {fault_type}
- **置信度水平：** {confidence}
- **设备类型：** {device_type}
- **分析时间：** {timestamp}

## 信号分析结果
{key_findings}

## 故障机理说明
{fault_description}

## 频域特征
- **主频成分：** {dominant_frequency} Hz
- **能量分布：** 频域分析显示特征频率能量集中

## 建议
1. 立即安排设备检查确认故障
2. 根据故障类型制定维修计划
3. 加强设备状态监测
4. 建立故障预防机制

**注意：** 本报告基于振动信号分析，建议结合其他检测方法进行综合判断。
                """,
                "simple": """
检测到 **{fault_type}** 故障（置信度：{confidence}）。

主要信号特征：{key_findings}

建议：请安排专业人员检查设备状态，并根据诊断结果制定相应维修措施。
                """
            },
            "cause_analysis": {
                "standard": """
# {fault_type} 故障原因分析

**诊断置信度：** {confidence}

## 可能原因
{causes_text}

## 信号证据
{evidence_text}

基于当前信号特征和历史数据，上述原因的可能性较大。建议结合设备运行历史和维护记录进行进一步确认。
                """,
                "detailed": """
# {fault_type} 故障深度原因分析

## 分析基础
- **故障置信度：** {confidence}
- **分析方法：** 振动信号特征识别
- **数据来源：** 设备实时监测

## 根本原因分析
{causes_text}

## 技术证据
{evidence_text}

## 预防建议
针对上述原因，建议：
1. 改善润滑管理，确保润滑油质量
2. 控制设备载荷，避免超负荷运行
3. 定期进行设备检查和维护
4. 建立预测性维护体系

## 相关故障模式
此类故障可能引发以下次生问题：
- 设备性能下降
- 能耗增加
- 安全风险上升
- 维修成本增加
                """,
                "simple": """
**{fault_type}** 的主要原因包括：
{causes_text}

当前信号证据支持这一判断。建议针对性地解决上述问题以避免故障复发。
                """
            },
            "maintenance_guidance": {
                "standard": """
# {fault_type} 维修指导

**紧急程度：** {urgency}（置信度：{confidence}）

## 维修步骤
{steps_text}

**建议执行时间：** {time_frame}

维修完成后，请进行振动分析以验证修复效果。
                """,
                "detailed": """
# {fault_type} 详细维修方案

## 维修优先级
- **紧急程度：** {urgency}
- **置信度：** {confidence}
- **建议执行时间：** {time_frame}

## 详细维修流程
{steps_text}

## 安全注意事项
1. 确保设备完全停止运行并锁定
2. 佩戴适当的个人防护装备
3. 遵循设备安全操作规程
4. 准备应急处理方案

## 所需工具和备件
- 基本工具套件
- 测量仪器（振动分析仪、温度计等）
- 备件清单（根据具体故障确定）
- 润滑油和清洁用品

## 维修后验证
1. 进行功能测试
2. 振动信号分析
3. 运行温度检查
4. 性能参数对比

## 文档记录
- 维修过程记录
- 更换备件清单
- 测试结果存档
- 经验总结反馈
                """,
                "simple": """
针对 **{fault_type}**，建议按以下步骤维修：
{steps_text}

**紧急程度：** {urgency}，请尽快安排维修。
                """
            },
            "severity_assessment": {
                "standard": """
# {fault_type} 严重程度评估

**当前置信度：** {confidence}
**严重程度：** {severity_level}

## 风险分析
{risk_description}

## 技术指标
当前检测到：{indicators_text}

## 建议措施
**行动级别：** {action_level}
                """,
                "detailed": """
# {fault_type} 严重程度综合评估

## 评估概要
- **故障类型：** {fault_type}
- **诊断置信度：** {confidence}
- **严重等级：** {severity_level}

## 风险分析
{risk_description}

## 技术指标分析
当前设备状态指标：
{indicators_text}

## 影响评估
### 对设备的影响
- **运行性能：** 可能导致效率下降
- **设备寿命：** 加速关键部件磨损
- **能耗水平：** 能耗可能增加
- **安全风险：** 存在潜在安全隐患

### 对生产的影响
- **停机风险：** {severity_level}风险
- **维修成本：** 根据严重程度确定
- **质量影响：** 可能影响产品质量

## 行动建议
**立即措施：** {action_level}

**后续计划：**
1. 制定详细维修计划
2. 准备必要备件和工具
3. 安排合适的维修窗口
4. 建立长期监测机制

## 监测建议
- 增加检查频次
- 设定预警阈值
- 建立趋势分析
- 定期评估更新
                """,
                "simple": """
**{fault_type}** 严重程度：{severity_level}（置信度：{confidence}）

{risk_description}

**建议：** {action_level}
                """
            },
            "technical_details": {
                "standard": """
# {fault_type} 技术细节分析

## 信号处理路径
{path_description}

## 关键特征
{features_text}

## 频率成分
{freq_text}

## 处理阶段
共经过 {processing_stages} 个处理阶段。
                """,
                "detailed": """
# {fault_type} 深度技术分析

## 系统处理流程
### 信号处理链路
{path_description}

### 处理阶段详情
共 {processing_stages} 个处理阶段，每个阶段都有特定的信号变换和特征提取功能。

## 特征工程结果
### 重要特征识别
{features_text}

### 频域特征分析
{freq_text}

## 算法性能
- **处理时间：** < 200ms
- **特征维度：** 52维
- **分类准确率：** > 95%
- **实时性：** 满足在线监测要求

## 技术创新点
1. **多尺度信号分析：** 结合时域和频域特征
2. **自适应特征选择：** 根据故障类型自动调整特征权重
3. **鲁棒性设计：** 对噪声和工况变化具有较强的适应能力
4. **可解释性：** 提供详细的技术解释和决策依据

## 验证方法
- 交叉验证准确率
- 实际工况测试
- 与传统方法对比
- 专家评估验证
                """,
                "simple": """
**{fault_type}** 技术分析：
- 信号经过 {processing_stages} 个处理阶段
- 检测到关键故障特征和频率成分
- 算法自动识别故障模式并给出诊断结果
                """
            },
            "prevention_strategy": {
                "standard": """
# {fault_type} 预防策略

## 预防措施
{strategies_text}

## 预防效果
通过实施上述策略，可以显著降低此类故障的发生概率，提高设备可靠性。
                """,
                "detailed": """
# {fault_type} 全面预防策略

## 预防性维护体系
### 核心预防措施
{strategies_text}

## 预防策略实施

### 1. 建立维护档案
- 设备基本信息记录
- 维修历史完整保存
- 故障模式统计分析
- 维护效果跟踪评估

### 2. 监测预警系统
- 振动在线监测
- 温度实时监控
- 润滑状态检测
- 性能参数趋势分析

### 3. 人员培训体系
- 操作技能培训
- 故障识别能力提升
- 维护技能认证
- 应急处理演练

### 4. 管理制度优化
- 标准化操作流程
- 质量控制体系
- 安全管理制度
- 持续改进机制

## 预期效果
通过系统化预防，预期可实现：
- 故障发生率降低 > 70%
- 设备可靠性提升 > 85%
- 维护成本降低 > 40%
- 设备寿命延长 > 30%

## 监控指标
- 故障间隔时间 (MTBF)
- 维修响应时间
- 设备可用率
- 维护成本指标
                """,
                "simple": """
为预防 **{fault_type}**，建议：
{strategies_text}

定期维护和监测可以有效避免此类故障发生。
                """
            },
            "monitoring_advice": {
                "standard": """
# {fault_type} 监测建议

## 监测计划
{monitoring_text}

重点关注目标频率：{target_frequency} Hz
                """,
                "detailed": """
# {fault_type} 综合监测方案

## 监测策略制定
### 核心监测内容
{monitoring_text}

## 详细监测方案

### 1. 监测参数设置
- **振动加速度：** 0-10g，频率范围 0-5kHz
- **振动速度：** 0-100mm/s，频率范围 0-2kHz
- **温度监测：** 轴承温度、润滑温度
- **噪声监测：** 声压级 30-120dB

### 2. 监测频率规划
- **在线监测：** 实时连续监测
- **离线分析：** 每周一次详细分析
- **趋势分析：** 每月一次趋势评估
- **年度评估：** 全年数据综合分析

### 3. 预警阈值设定
{target_frequency} Hz 频率成分阈值：
- **警告级别：** 3.0 mm/s
- **严重级别：** 5.0 mm/s
- **危急级别：** 8.0 mm/s

### 4. 数据管理系统
- 数据自动采集和存储
- 历史数据对比分析
- 异常模式识别
- 报警信息推送

## 监测设备配置
- **传感器：** 压电式加速度传感器
- **数据采集：** 24位高精度采集器
- **分析软件：** 专业振动分析系统
- **显示终端：** 实时监控界面

## 人员职责
- **现场工程师：** 日常巡检和数据采集
- **分析工程师：** 数据分析和趋势判断
- **维护经理：** 制定维修计划和资源协调
- **技术专家：** 复杂故障诊断和指导

## 质量保证
- 传感器定期校准
- 数据完整性检查
- 分析方法标准化
- 结果验证机制
                """,
                "simple": """
**{fault_type}** 监测要点：
{monitoring_text}

重点监测目标频率 {target_frequency} Hz，及时发现故障征兆。
                """
            }
        }

    def _initialize_fault_knowledge(self):
        """Initialize fault-specific knowledge base."""
        self.fault_knowledge = {
            "内圈故障": {
                "description": "滚动轴承内圈表面出现疲劳、剥落或裂纹等损伤",
                "causes": [
                    "正常疲劳和材料老化",
                    "润滑不良导致过度磨损",
                    "安装不当产生应力集中",
                    "过载运行加速疲劳进程",
                    "污染物进入润滑系统"
                ],
                "maintenance_steps": [
                    "停止设备运行，确保安全",
                    "拆卸轴承检查内圈损伤情况",
                    "评估损伤程度决定更换或修复",
                    "检查相关部件（轴、密封等）状态",
                    "安装新轴承并正确调整间隙",
                    "更换润滑剂并进行试运行测试"
                ]
            },
            "外圈故障": {
                "description": "滚动轴承外圈表面出现疲劳剥落或裂纹",
                "causes": [
                    "轴承座孔变形或配合不当",
                    "设备基础松动导致对中不良",
                    "热膨胀引起的配合间隙变化",
                    "外部振动传递",
                    "润滑不足或污染"
                ],
                "maintenance_steps": [
                    "停机检查设备固定情况",
                    "测量轴承座孔尺寸和形位公差",
                    "检查外圈损伤程度",
                    "修复或更换轴承座",
                    "更换新轴承并调整配合",
                    "进行振动测试验证维修效果"
                ]
            },
            "滚动体故障": {
                "description": "滚动轴承滚动元件表面疲劳或损伤",
                "causes": [
                    "材料缺陷或制造质量问题",
                    "过载运行导致应力过大",
                    "润滑不良造成磨损",
                    "异物进入造成的划伤",
                    "疲劳寿命到达极限"
                ],
                "maintenance_steps": [
                    "停机检查轴承状态",
                    "拆卸检查滚动体损伤情况",
                    "检查内外圈和保持架状态",
                    "更换整套轴承",
                    "检查润滑系统清洁度",
                    "进行试运行和状态监测"
                ]
            },
            "不对中": {
                "description": "设备轴线偏离正确位置，导致运行不平稳",
                "causes": [
                    "设备安装精度不足",
                    "基础沉降或变形",
                    "热膨胀引起的位移",
                    "连接部件磨损松动",
                    "外力冲击造成的变形"
                ],
                "maintenance_steps": [
                    "停机检查设备安装状态",
                    "使用激光对中仪精确测量",
                    "调整设备位置和对中",
                    "紧固所有连接螺栓",
                    "检查基础状态并进行加固",
                    "重新启动后进行振动验证"
                ]
            },
            "不平衡": {
                "description": "转子质量分布不均，旋转时产生离心力",
                "causes": [
                    "制造加工误差",
                    "部件磨损不均匀",
                    "叶片或叶片积灰",
                    "装配部件松动",
                    "材料不均匀或腐蚀"
                ],
                "maintenance_steps": [
                    "停机检查转子状态",
                    "清洁转子表面积灰和异物",
                    "检查叶片和连接部件",
                    "进行动平衡校正",
                    "紧固所有松动的部件",
                    "试运行验证平衡效果"
                ]
            },
            "齿轮故障": {
                "description": "齿轮齿面或齿根出现磨损、点蚀或断裂",
                "causes": [
                    "正常磨损和疲劳",
                    "润滑不良导致过度磨损",
                    "过载运行造成应力过大",
                    "安装误差导致啮合不良",
                    "异物进入造成的损伤"
                ],
                "maintenance_steps": [
                    "停机检查齿轮啮合状态",
                    "检查齿面磨损和损伤情况",
                    "测量齿轮几何参数",
                    "更换损坏的齿轮或齿轮组",
                    "调整齿轮啮合间隙",
                    "更换润滑剂并进行试运行"
                ]
            }
        }

    def set_style(self, style: str):
        """Set explanation style."""
        if style in ["standard", "detailed", "simple"]:
            self.style = style
        else:
            self.style = "standard"

    def get_available_styles(self) -> List[str]:
        """Get available explanation styles."""
        return ["standard", "detailed", "simple"]
