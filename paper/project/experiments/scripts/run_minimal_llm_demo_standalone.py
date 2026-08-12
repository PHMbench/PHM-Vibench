#!/usr/bin/env python3
"""
Standalone Minimal LLM Demo Script

This script demonstrates the LLM explanation generation pipeline
without requiring external dependencies or model files.
"""

import sys
import os
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import json
import uuid
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class FaultInfo:
    """Fault information from model prediction."""
    fault_type: str
    confidence: float
    probability_distribution: List[float] = field(default_factory=list)
    predicted_class: int = -1
    prediction_method: str = "Unknown"
    model_name: str = "Unknown"


@dataclass
class SignalAnalysis:
    """Signal analysis results."""
    statistics: Dict[str, float] = field(default_factory=dict)
    frequency_analysis: Dict[str, float] = field(default_factory=dict)
    signal_length: int = 0
    sampling_rate: int = 1024
    key_findings: List[str] = field(default_factory=list)


@dataclass
class TechnicalExplanation:
    """Technical explanation components."""
    signal_path: Optional[Dict[str, Any]] = None
    processing_stages: int = 0
    energy_analysis: Dict[str, Any] = field(default_factory=dict)
    important_features: List[Dict[str, Any]] = field(default_factory=list)
    frequency_components: List[Dict[str, Any]] = field(default_factory=list)
    path_signature: Optional[Dict[str, Any]] = None


@dataclass
class DeviceContext:
    """Device and operational context."""
    device_type: str = "Unknown"
    operating_speed: Optional[float] = None
    load_condition: str = "Unknown"
    environment: str = "Unknown"
    maintenance_history: str = "Unknown"
    installation_date: Optional[str] = None


@dataclass
class LLMIntermediateRepresentation:
    """Intermediate representation for LLM-based explanation generation."""
    explanation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    fault_info: FaultInfo = field(default_factory=lambda: FaultInfo("Unknown", 0.0))
    signal_analysis: SignalAnalysis = field(default_factory=SignalAnalysis)
    technical_explanation: TechnicalExplanation = field(default_factory=TechnicalExplanation)
    device_context: DeviceContext = field(default_factory=DeviceContext)
    user_query: Optional[str] = None
    explanation_style: str = "standard"

    signal_data_info: Dict[str, Any] = field(default_factory=dict)
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: Optional[float] = None

    def get_summary(self) -> str:
        """Get a concise summary of the representation."""
        return (f"故障类型：{self.fault_info.fault_type} (置信度: {self.fault_info.confidence:.1%}), "
                f"设备类型：{self.device_context.device_type}, "
                f"关键发现：{len(self.signal_analysis.key_findings)} 项")


class LocalTemplateLLM:
    """
    Local template-based LLM stub for generating explanations.
    """

    def __init__(self, style: str = "standard"):
        """Initialize the local template LLM."""
        self.style = style
        self._initialize_templates()
        self._initialize_fault_knowledge()

    def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response using template-based approach."""
        if not context or "intermediate_representation" not in context:
            return self._generate_error_response("缺少必要的上下文信息")

        ir = context["intermediate_representation"]

        # Determine response type based on prompt analysis
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
        """Determine the type of response needed."""
        prompt_lower = prompt.lower()

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

        key_findings = ir.signal_analysis.key_findings[:3]
        key_finding_text = "；".join(key_findings) if key_findings else "信号分析显示异常特征"

        fault_description = self.fault_knowledge.get(fault_type, {}).get("description", "设备故障")

        template = """
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
        """.format(
            fault_type=fault_type,
            confidence=f"{confidence:.1%}",
            device_type=device_type,
            key_findings=key_finding_text,
            fault_description=fault_description,
            dominant_frequency=f"{ir.signal_analysis.frequency_analysis.get('dominant_frequency', 0):.1f}"
        )

        return template.strip()

    def _generate_cause_analysis(self, ir: LLMIntermediateRepresentation) -> str:
        """Generate cause analysis for the fault."""
        fault_type = ir.fault_info.fault_type
        confidence = ir.fault_info.confidence

        fault_causes = self.fault_knowledge.get(fault_type, {}).get("causes", [
            "正常磨损和材料疲劳",
            "润滑不良或污染",
            "过载运行或冲击载荷",
            "安装不当或对中不良"
        ])

        causes_text = "\n".join([f"• {cause}" for cause in fault_causes])

        template = """
# {fault_type} 故障原因分析

**诊断置信度：** {confidence}

## 可能原因
{causes_text}

基于当前信号特征和历史数据，上述原因的可能性较大。建议结合设备运行历史和维护记录进行进一步确认。
        """.format(
            fault_type=fault_type,
            confidence=f"{confidence:.1%}",
            causes_text=causes_text
        )

        return template.strip()

    def _generate_maintenance_guidance(self, ir: LLMIntermediateRepresentation) -> str:
        """Generate maintenance guidance."""
        fault_type = ir.fault_info.fault_type
        confidence = ir.fault_info.confidence

        if confidence > 0.8:
            urgency = "紧急"
            time_frame = "24小时内"
        elif confidence > 0.6:
            urgency = "高优先级"
            time_frame = "一周内"
        else:
            urgency = "计划性"
            time_frame = "下次维护窗口"

        maintenance_steps = self.fault_knowledge.get(fault_type, {}).get("maintenance_steps", [
            "确保设备已停止运行，采取安全防护措施",
            "详细检查故障相关部件的具体状态",
            "评估损坏程度，确定是否需要更换部件",
            "制定具体的维修计划和时间安排",
            "维修后进行功能测试和振动分析验证"
        ])

        steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(maintenance_steps)])

        template = """
# {fault_type} 维修指导

**紧急程度：** {urgency}（置信度：{confidence}）

## 维修步骤
{steps_text}

**建议执行时间：** {time_frame}

维修完成后，请进行振动分析以验证修复效果。
        """.format(
            fault_type=fault_type,
            confidence=f"{confidence:.1%}",
            steps_text=steps_text,
            urgency=urgency,
            time_frame=time_frame
        )

        return template.strip()

    def _generate_severity_assessment(self, ir: LLMIntermediateRepresentation) -> str:
        """Generate severity assessment."""
        fault_type = ir.fault_info.fault_type
        confidence = ir.fault_info.confidence

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

        template = """
# {fault_type} 严重程度评估

**当前置信度：** {confidence}
**严重程度：** {severity_level}

## 风险分析
{risk_description}

## 建议措施
**行动级别：** {action_level}
        """.format(
            fault_type=fault_type,
            confidence=f"{confidence:.1%}",
            severity_level=severity_level,
            risk_description=risk_description,
            action_level=action_level
        )

        return template.strip()

    def _generate_technical_details(self, ir: LLMIntermediateRepresentation) -> str:
        """Generate technical details explanation."""
        fault_type = ir.fault_info.fault_type

        features_text = ""
        if ir.technical_explanation.important_features:
            features = ir.technical_explanation.important_features[:3]
            features_text = "\n".join([
                f"• {f['feature']}: {f['value']:.2f} (阈值: {f['threshold']}, 重要性: {f['significance']})"
                for f in features
            ])

        template = """
# {fault_type} 技术细节分析

## 关键特征
{features_text}

## 处理阶段
共经过 {processing_stages} 个处理阶段。
        """.format(
            fault_type=fault_type,
            features_text=features_text or "未检测到显著特征",
            processing_stages=ir.technical_explanation.processing_stages
        )

        return template.strip()

    def _generate_prevention_strategy(self, ir: LLMIntermediateRepresentation) -> str:
        """Generate prevention strategy."""
        fault_type = ir.fault_info.fault_type

        strategies = [
            "建立定期检查和维护制度",
            "加强润滑管理，确保润滑油质量",
            "控制设备运行载荷，避免过载",
            "安装振动监测系统进行早期预警",
            "培训操作人员，提高故障识别能力"
        ]

        strategies_text = "\n".join([f"• {strategy}" for strategy in strategies])

        template = """
# {fault_type} 预防策略

## 预防措施
{strategies_text}

通过实施上述策略，可以显著降低此类故障的发生概率，提高设备可靠性。
        """.format(
            fault_type=fault_type,
            strategies_text=strategies_text
        )

        return template.strip()

    def _generate_monitoring_advice(self, ir: LLMIntermediateRepresentation) -> str:
        """Generate monitoring advice."""
        fault_type = ir.fault_info.fault_type
        dominant_freq = ir.signal_analysis.frequency_analysis.get("dominant_frequency", 0)

        monitoring_plan = [
            "增加振动监测频率至每周一次",
            f"重点监测 {dominant_freq:.1f} Hz 附近的频率成分变化",
            "建立振动趋势数据库，跟踪长期变化",
            "设置多级报警阈值（警告、严重、危急）",
            "定期分析监测数据，调整监测策略"
        ]

        monitoring_text = "\n".join([f"• {item}" for item in monitoring_plan])

        template = """
# {fault_type} 监测建议

## 监测计划
{monitoring_text}

重点关注目标频率：{target_frequency} Hz
        """.format(
            fault_type=fault_type,
            monitoring_text=monitoring_text,
            target_frequency=f"{dominant_freq:.1f}" if dominant_freq > 0 else "故障特征频率"
        )

        return template.strip()

    def _generate_error_response(self, error_message: str) -> str:
        """Generate error response."""
        return f"抱歉，生成解释时遇到问题：{error_message}。请稍后重试或联系技术支持。"

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
            }
        }

    def _initialize_templates(self):
        """Initialize response templates."""
        pass  # Templates are embedded in methods for this standalone version


class MockDataAdapter:
    """Adapter for creating mock data for testing and demonstration."""

    @staticmethod
    def create_comprehensive_example(fault_type: str = "内圈故障",
                                   confidence: float = 0.87,
                                   device_type: str = "滚动轴承") -> LLMIntermediateRepresentation:
        """Create a comprehensive mock example for demonstration."""
        ir = LLMIntermediateRepresentation()

        # Set fault info
        ir.fault_info.fault_type = fault_type
        ir.fault_info.confidence = confidence
        ir.fault_info.probability_distribution = [0.03, 0.05, confidence, 0.04, 0.01]
        ir.fault_info.predicted_class = 2
        ir.fault_info.prediction_method = "TSPN"
        ir.fault_info.model_name = "Transparent Signal Processing Network v1.0"

        # Set signal analysis
        ir.signal_analysis.statistics = {
            "mean": 2.5,
            "std": 15.8,
            "rms": 16.0,
            "peak": 45.2,
            "crest_factor": 2.8,
            "skewness": 0.3,
            "kurtosis": 2.1
        }

        ir.signal_analysis.frequency_analysis = {
            "dominant_frequency": 157.5,
            "dominant_power": 12.8,
            "spectral_centroid": 89.3
        }

        ir.signal_analysis.signal_length = 4096
        ir.signal_analysis.sampling_rate = 1024

        ir.signal_analysis.key_findings = [
            f"检测到 {fault_type} 特征频率成分",
            "振动RMS值显著增高",
            "频域分析显示明显谐波",
            "时域波形存在周期性冲击"
        ]

        # Set technical explanation
        ir.technical_explanation.processing_stages = 4
        ir.technical_explanation.important_features = [
            {"feature": "RMS值", "value": 16.0, "threshold": 5.0, "significance": "高"},
            {"feature": "峰值因子", "value": 2.8, "threshold": 3.0, "significance": "中等"},
            {"feature": "频谱峰值", "value": 157.5, "threshold": 100.0, "significance": "高"}
        ]

        ir.technical_explanation.frequency_components = [
            {"frequency": 157.5, "amplitude": 12.8, "type": "故障频率"},
            {"frequency": 315.0, "amplitude": 6.2, "type": "谐波"},
            {"frequency": 472.5, "amplitude": 2.1, "type": "谐波"}
        ]

        # Set device context
        ir.device_context.device_type = device_type
        ir.device_context.operating_speed = 1800.0
        ir.device_context.load_condition = "正常载荷"
        ir.device_context.environment = "室内"
        ir.device_context.maintenance_history = "定期维护"

        return ir


class MinimalLLMDemo:
    """Minimal demonstration of LLM-based fault diagnosis explanation."""

    def __init__(self):
        """Initialize the demo."""
        self.llm = LocalTemplateLLM(style="standard")
        self.demo_cases = self._create_demo_cases()

    def _create_demo_cases(self) -> list:
        """Create demonstration cases."""
        return [
            {
                "name": "轴承内圈故障",
                "description": "高置信度检测到的典型轴承内圈故障",
                "ir": MockDataAdapter.create_comprehensive_example(
                    fault_type="内圈故障",
                    confidence=0.92,
                    device_type="滚动轴承"
                ),
                "queries": [
                    "请解释这个故障的原因",
                    "应该如何维修这个故障？",
                    "故障的严重程度如何？",
                    "请提供详细的技术分析"
                ]
            },
            {
                "name": "设备不对中故障",
                "description": "中等置信度的设备不对中故障",
                "ir": MockDataAdapter.create_comprehensive_example(
                    fault_type="不对中",
                    confidence=0.78,
                    device_type="电机驱动系统"
                ),
                "queries": [
                    "这是什么类型的故障？",
                    "有什么预防措施？",
                    "如何监测这种故障？"
                ]
            }
        ]

    def run_pipeline_demo(self) -> None:
        """Demonstrate the complete pipeline."""
        print("\n" + "="*80)
        print("完整处理流程演示")
        print("="*80)

        # Step 1: Create mock signal data
        print(f"\n📡 步骤 1: 生成模拟信号数据")
        signal_data = np.random.randn(4096) * 5
        # Add some periodic components to simulate fault
        t = np.linspace(0, 4, 4096)
        signal_data += 2 * np.sin(2 * np.pi * 50 * t)  # 50 Hz component
        signal_data += 1.5 * np.sin(2 * np.pi * 150 * t)  # 150 Hz harmonic
        print(f"   • 信号长度：{len(signal_data)}")
        print(f"   • 采样率：1024 Hz")
        print(f"   • 信号范围：[{signal_data.min():.2f}, {signal_data.max():.2f}]")

        # Step 2: Create mock model prediction
        print(f"\n🧠 步骤 2: 模型预测结果")
        model_prediction = {
            "fault_type": "内圈故障",
            "confidence": 0.87,
            "probabilities": [0.03, 0.05, 0.87, 0.04, 0.01],
            "predicted_class": 2,
            "method": "TSPN",
            "model_name": "Transparent Signal Processing Network"
        }
        print(f"   • 故障类型：{model_prediction['fault_type']}")
        print(f"   • 置信度：{model_prediction['confidence']:.1%}")
        print(f"   • 预测方法：{model_prediction['method']}")

        # Step 3: Create intermediate representation
        print(f"\n🔄 步骤 3: 创建中间表示")
        ir = MockDataAdapter.create_comprehensive_example(
            fault_type=model_prediction['fault_type'],
            confidence=model_prediction['confidence'],
            device_type="滚动轴承"
        )
        print(f"   • 表示ID：{ir.explanation_id}")
        print(f"   • 生成时间：{ir.timestamp}")
        print(f"   • 关键发现：{len(ir.signal_analysis.key_findings)} 项")

        # Step 4: Generate LLM explanation
        print(f"\n💬 步骤 4: 生成自然语言解释")
        context = {"intermediate_representation": ir}
        query = "请解释这个诊断结果"
        response = self.llm.generate(query, context)
        print(response)

        # Step 5: Generate different types of explanations
        print(f"\n🎯 步骤 5: 多类型解释演示")
        queries = [
            "这个故障是什么原因造成的？",
            "应该如何维修这个故障？",
            "故障的严重程度如何评估？",
            "有什么预防措施吗？"
        ]

        for i, query in enumerate(queries, 1):
            print(f"\n--- 查询 {i}: {query} ---")
            response = self.llm.generate(query, context)
            print(response)

        # Step 6: Summary
        print(f"\n📊 步骤 6: 处理流程总结")
        print(f"   ✅ 信号数据生成")
        print(f"   ✅ 模型预测模拟")
        print(f"   ✅ 中间表示构建")
        print(f"   ✅ 自然语言生成")
        print(f"   ✅ 多类型解释输出")

        print(f"\n🎉 演示完成！数据流验证成功。")

    def run_single_case_demo(self, case_index: int = 0) -> None:
        """Run demo for a single case."""
        if case_index >= len(self.demo_cases):
            print(f"错误：案例索引 {case_index} 超出范围")
            return

        case = self.demo_cases[case_index]
        ir = case["ir"]

        print("\n" + "="*80)
        print(f"案例演示：{case['name']}")
        print(f"描述：{case['description']}")
        print("="*80)

        # Show diagnosis summary
        print(f"\n📋 诊断摘要:")
        print(f"   故障类型：{ir.fault_info.fault_type}")
        print(f"   置信度：{ir.fault_info.confidence:.1%}")
        print(f"   设备类型：{ir.device_context.device_type}")
        print(f"   信号长度：{ir.signal_analysis.signal_length} 点")

        # Show key signal findings
        if ir.signal_analysis.key_findings:
            print(f"\n🔍 关键发现:")
            for finding in ir.signal_analysis.key_findings[:3]:
                print(f"   • {finding}")

        # Show technical features
        if ir.technical_explanation.important_features:
            print(f"\n⚙️ 技术特征:")
            for feature in ir.technical_explanation.important_features[:3]:
                print(f"   • {feature['feature']}: {feature['value']:.2f} "
                      f"(重要性: {feature['significance']})")

        # Generate explanations for different queries
        print(f"\n💬 智能对话演示:")
        print("-"*60)

        for i, query in enumerate(case["queries"], 1):
            print(f"\n用户问题 {i}: {query}")

            # Generate response
            context = {"intermediate_representation": ir}
            response = self.llm.generate(query, context)

            print(f"助手回答:")
            print(response)


def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description="LLM Fault Diagnosis Explanation Demo")
    parser.add_argument("--mode", choices=["pipeline", "single"],
                       default="pipeline", help="Demo mode to run")
    parser.add_argument("--case", type=int, default=0, help="Case index for single mode")

    args = parser.parse_args()

    print("🚀 LLM故障诊断解释演示系统 (独立版本)")
    print("=" * 50)

    demo = MinimalLLMDemo()

    try:
        if args.mode == "pipeline":
            demo.run_pipeline_demo()
        elif args.mode == "single":
            demo.run_single_case_demo(args.case)

        print(f"\n✅ 演示完成！")

    except KeyboardInterrupt:
        print(f"\n\n👋 演示被中断，再见！")
    except Exception as e:
        print(f"\n❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()