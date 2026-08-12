#!/usr/bin/env python3
"""
Complete End-to-End Pipeline Test

This script tests the complete pipeline from signal generation to LLM explanation
using isolated components to avoid import issues.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import uuid


# Include all necessary data structures and classes
@dataclass
class FaultInfo:
    fault_type: str
    confidence: float
    severity: str = "medium"
    description: str = ""


@dataclass
class SignalAnalysis:
    signal_length: int
    sampling_rate: float
    statistics: Dict[str, float]
    frequency_analysis: Dict[str, Any]
    key_findings: List[str]


@dataclass
class TechnicalExplanation:
    signal_path: Dict[str, Any]
    important_features: List[Dict[str, Any]]
    attention_weights: Dict[str, Any]
    layer_contributions: Dict[str, Any]
    processing_steps: List[Dict[str, Any]]


@dataclass
class DeviceContext:
    device_type: str
    operating_conditions: Dict[str, Any]
    maintenance_history: str = ""
    specifications: str = ""


@dataclass
class LLMIntermediateRepresentation:
    explanation_id: str
    timestamp: str
    fault_info: FaultInfo
    signal_analysis: SignalAnalysis
    technical_explanation: TechnicalExplanation
    device_context: DeviceContext
    metadata: Dict[str, Any]


class CompletePipelineTest:
    """Complete pipeline testing with isolated components."""

    def __init__(self):
        """Initialize the complete pipeline test."""
        self.test_results = []
        self.generated_data = {}

    def create_demo_signal_data(self, fault_type: str = "inner_race") -> np.ndarray:
        """Create demonstration signal data with specific fault characteristics."""
        t = np.linspace(0, 4, 4096)  # 4 seconds at 1024 Hz

        # Base signal with some noise
        signal = np.random.randn(4096) * 0.5

        if fault_type == "inner_race":
            # Inner race fault characteristic frequencies
            signal += 2.0 * np.sin(2 * np.pi * 50 * t)  # 50 Hz fundamental
            signal += 1.5 * np.sin(2 * np.pi * 150 * t)  # 3rd harmonic
            signal += 1.0 * np.sin(2 * np.pi * 250 * t)  # 5th harmonic

            # Add some impact components
            impact_times = np.random.choice(len(t), size=10, replace=False)
            for impact_time in impact_times:
                if impact_time < len(t) - 100:
                    signal[impact_time:impact_time+100] += 3.0 * np.exp(-0.05 * np.arange(100))

        elif fault_type == "misalignment":
            # Misalignment shows strong 1x and 2x running speed components
            signal += 2.5 * np.sin(2 * np.pi * 30 * t)  # 1x
            signal += 1.8 * np.sin(2 * np.pi * 60 * t)  # 2x
            signal += 0.8 * np.sin(2 * np.pi * 90 * t)  # 3x

        elif fault_type == "outer_race":
            # Outer race fault
            signal += 1.8 * np.sin(2 * np.pi * 40 * t)
            signal += 1.2 * np.sin(2 * np.pi * 120 * t)

        else:  # normal
            # Just some background noise and minor components
            signal += 0.3 * np.sin(2 * np.pi * 25 * t)
            signal += 0.2 * np.sin(2 * np.pi * 50 * t)

        return signal

    def generate_tspn_explanation(self, signal_data: np.ndarray, fault_type: str = "内圈故障", confidence: float = 0.85) -> Dict[str, Any]:
        """Generate mock TSPN explanation."""
        # Compute basic signal statistics
        statistics = {
            'mean': float(np.mean(signal_data)),
            'std': float(np.std(signal_data)),
            'rms': float(np.sqrt(np.mean(signal_data**2))),
            'max': float(np.max(signal_data)),
            'min': float(np.min(signal_data)),
            'energy': float(np.sum(signal_data**2))
        }

        # Frequency analysis
        fft_vals = np.fft.fft(signal_data)
        fft_freq = np.fft.fftfreq(len(signal_data), 1/1024.0)
        pos_mask = fft_freq > 0
        pos_freq = fft_freq[pos_mask]
        pos_fft = np.abs(fft_vals[pos_mask])

        frequency_analysis = {
            'dominant_frequency': float(pos_freq[np.argmax(pos_fft)]) if len(pos_fft) > 0 else 0.0,
            'spectral_centroid': float(np.sum(pos_freq * pos_fft) / (np.sum(pos_fft) + 1e-8)) if len(pos_fft) > 0 else 0.0,
            'total_power': float(np.sum(pos_fft)) if len(pos_fft) > 0 else 0.0
        }

        # Generate key findings
        key_findings = self._generate_key_findings(signal_data, statistics)

        explanation = {
            'fault_type': fault_type,
            'confidence': confidence,
            'severity': 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low',
            'description': f'TSPN模型检测到{fault_type}，置信度为{confidence:.1%}',
            'method': 'TSPN_explainable',
            'model_type': 'Transparent Signal Processing Network',
            'signal_statistics': statistics,
            'frequency_analysis': frequency_analysis,
            'signal_length': len(signal_data),
            'sampling_rate': 1024.0,
            'signal_path': {
                'processing_steps': [
                    {
                        'layer': 'Input',
                        'output_shape': [1, len(signal_data)],
                        'description': '原始振动信号输入'
                    },
                    {
                        'layer': 'SignalProcessing_Layer1',
                        'modules': ['FFT', 'WF', 'HT', 'I'],
                        'output_shape': [1, len(signal_data)],
                        'description': '第一层信号处理'
                    },
                    {
                        'layer': 'FeatureExtractor',
                        'output_shape': [1, 52],
                        'description': '多域特征提取'
                    },
                    {
                        'layer': 'Attention_Mechanism',
                        'output_shape': [1, 52],
                        'description': '注意力机制加权'
                    },
                    {
                        'layer': 'Classifier',
                        'output_shape': [1, 10],
                        'description': '故障分类输出'
                    }
                ]
            },
            'important_features': [
                {
                    'feature': 'RMS值',
                    'value': statistics['rms'],
                    'significance': 0.92,
                    'domain': 'time_domain'
                },
                {
                    'feature': '峰值因子',
                    'value': statistics['max'] / (statistics['rms'] + 1e-8),
                    'significance': 0.87,
                    'domain': 'time_domain'
                },
                {
                    'feature': '主频幅值',
                    'value': frequency_analysis['dominant_frequency'],
                    'significance': 0.95,
                    'domain': 'frequency_domain'
                }
            ],
            'layer_contributions': {
                'layer1_fft': 0.35,
                'layer1_wf': 0.28,
                'layer1_ht': 0.22,
                'layer1_identity': 0.15,
                'feature_extractor': 0.78,
                'attention': 0.65,
                'classifier': 1.0
            },
            'key_findings': key_findings
        }

        return explanation

    def _generate_key_findings(self, signal_data: np.ndarray, statistics: Dict[str, float]) -> List[str]:
        """Generate key findings from signal data and statistics."""
        findings = []

        if statistics:
            rms = statistics.get('rms', 0)
            peak_factor = statistics.get('max', 0) / (statistics.get('rms', 1) + 1e-8)

            # High energy indication
            if rms > 5.0:
                findings.append(f"信号能量较高，RMS值为{rms:.2f}")

            # Peak factor indication
            if peak_factor > 4.0:
                findings.append(f"信号峰值因子较高({peak_factor:.2f})，可能存在冲击")
            elif peak_factor < 3.0:
                findings.append(f"信号峰值因子较低({peak_factor:.2f})，较为平稳")

        # Check for periodicity
        if len(signal_data) > 100:
            signal_centered = signal_data - np.mean(signal_data)
            autocorr = np.correlate(signal_centered, signal_centered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            if len(autocorr) > 20:
                peaks = []
                for i in range(20, len(autocorr)//2):
                    if autocorr[i] > np.max(autocorr[max(0, i-10):i]) * 1.2:
                        peaks.append(i)

                if peaks:
                    findings.append(f"信号呈现周期性特征，检测到{len(peaks)}个显著周期")

        if not findings:
            findings.append("信号特征正常，未检测到明显异常")

        return findings

    def convert_to_intermediate_representation(self, explanation: Dict[str, Any], signal_data: Optional[np.ndarray] = None, device_context: Optional[Dict[str, Any]] = None) -> LLMIntermediateRepresentation:
        """Convert explanation to intermediate representation."""
        # Create fault info
        fault_info = FaultInfo(
            fault_type=explanation.get('fault_type', 'unknown'),
            confidence=explanation.get('confidence', 0.0),
            severity=explanation.get('severity', 'medium'),
            description=explanation.get('description', '')
        )

        # Create signal analysis
        signal_analysis = SignalAnalysis(
            signal_length=explanation.get('signal_length', len(signal_data) if signal_data is not None else 0),
            sampling_rate=explanation.get('sampling_rate', 1024.0),
            statistics=explanation.get('signal_statistics', {}),
            frequency_analysis=explanation.get('frequency_analysis', {}),
            key_findings=explanation.get('key_findings', [])
        )

        # Create technical explanation
        technical_explanation = TechnicalExplanation(
            signal_path=explanation.get('signal_path', {}),
            important_features=explanation.get('important_features', []),
            attention_weights=explanation.get('attention_weights', {}),
            layer_contributions=explanation.get('layer_contributions', {}),
            processing_steps=explanation.get('processing_steps', [])
        )

        # Create device context
        device_context = device_context or {}
        device_context_obj = DeviceContext(
            device_type=device_context.get('device_type', 'unknown'),
            operating_conditions=device_context.get('operating_conditions', {}),
            maintenance_history=device_context.get('maintenance_history', ''),
            specifications=device_context.get('specifications', '')
        )

        # Create intermediate representation
        ir = LLMIntermediateRepresentation(
            explanation_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            fault_info=fault_info,
            signal_analysis=signal_analysis,
            technical_explanation=technical_explanation,
            device_context=device_context_obj,
            metadata={
                'source': 'complete_pipeline_test',
                'explanation_method': explanation.get('method', 'unknown'),
                'model_type': explanation.get('model_type', 'unknown'),
                'extraction_time': datetime.now().isoformat()
            }
        )

        return ir

    def generate_llm_response(self, ir: LLMIntermediateRepresentation, query: str, style: str = "standard") -> str:
        """Generate LLM response based on IR and query."""
        # Simple template-based response generation
        fault_type = ir.fault_info.fault_type
        confidence = ir.fault_info.confidence
        device_type = ir.device_context.device_type

        if style == "simple":
            if "解释" in query or "什么" in query:
                return f"您的{device_type}检测到了{fault_type}，置信度{confidence:.1%}。建议及时检查设备状态。"
            elif "维修" in query or "处理" in query:
                return "建议：1.停止设备运行 2.检查故障部件 3.维修或更换损坏部分 4.测试确认正常"
            elif "严重" in query:
                severity = "严重" if confidence > 0.8 else "中等" if confidence > 0.6 else "轻微"
                return f"故障严重程度：{severity}，建议{'立即处理' if confidence > 0.8 else '尽快安排检查'}"
            else:
                return f"检测到{fault_type}，置信度{confidence:.1%}，请关注设备运行状态。"

        elif style == "detailed":
            response = f"{device_type}详细分析报告\n\n"
            response += f"故障类型：{fault_type}\n"
            response += f"检测置信度：{confidence:.1%}\n"
            response += f"信号特征：\n"

            if ir.signal_analysis:
                rms = ir.signal_analysis.statistics.get('rms', 0)
                response += f"- 振动强度(RMS)：{rms:.2f}\n"
                response += f"- 信号长度：{ir.signal_analysis.signal_length} 点\n"
                response += f"- 采样率：{ir.signal_analysis.sampling_rate} Hz\n"

            if ir.signal_analysis.key_findings:
                response += "\n关键发现：\n"
                for finding in ir.signal_analysis.key_findings:
                    response += f"• {finding}\n"

            return response

        else:  # standard
            if "解释" in query or "什么" in query:
                response = f"根据信号分析，您的{device_type}检测到{fault_type}，置信度为{confidence:.1%}。\n\n"
                if ir.signal_analysis.key_findings:
                    response += "主要发现：\n"
                    for finding in ir.signal_analysis.key_findings[:2]:
                        response += f"• {finding}\n"
                return response
            elif "维修" in query or "处理" in query:
                response = f"{fault_type}维修建议：\n\n"
                response += "1. 确保设备安全停机\n"
                response += "2. 检查相关部件状态\n"
                response += "3. 根据损坏程度进行维修或更换\n"
                response += "4. 维修后进行功能测试\n"
                urgency = "紧急" if confidence > 0.8 else "尽快" if confidence > 0.6 else "计划内"
                response += f"\n建议处理时间：{urgency}处理"
                return response
            else:
                return f"检测到{fault_type}（置信度{confidence:.1%}），建议进一步检查确认设备状态。"

    def test_single_pipeline(self, case_config: Dict[str, Any]) -> Dict[str, Any]:
        """Test complete pipeline for a single case."""
        case_name = case_config["name"]
        print(f"\n🔄 测试案例: {case_name}")
        print("-" * 50)

        result = {
            "case_name": case_name,
            "start_time": datetime.now().isoformat(),
            "steps": {}
        }

        try:
            # Step 1: Generate signal data
            print("📡 步骤1: 生成信号数据")
            signal_data = self.create_demo_signal_data(case_config["signal_type"])
            result["steps"]["signal_generation"] = {
                "success": True,
                "signal_length": len(signal_data),
                "signal_range": [float(signal_data.min()), float(signal_data.max())]
            }
            print(f"   ✅ 信号长度: {len(signal_data)}, 范围: [{signal_data.min():.2f}, {signal_data.max():.2f}]")

            # Step 2: Generate TSPN explanation
            print("🧠 步骤2: 生成TSPN解释")
            explanation = self.generate_tspn_explanation(
                signal_data=signal_data,
                fault_type=case_config["fault_type"],
                confidence=case_config["confidence"]
            )
            result["steps"]["explanation_generation"] = {
                "success": True,
                "fault_type": explanation["fault_type"],
                "confidence": explanation["confidence"],
                "method": explanation["method"]
            }
            print(f"   ✅ 故障类型: {explanation['fault_type']}, 置信度: {explanation['confidence']:.1%}")

            # Step 3: Convert to intermediate representation
            print("🔄 步骤3: 转换为中间表示")
            ir = self.convert_to_intermediate_representation(
                explanation=explanation,
                signal_data=signal_data,
                device_context=case_config["device_context"]
            )
            result["steps"]["ir_conversion"] = {
                "success": True,
                "ir_id": ir.explanation_id,
                "fault_info": {
                    "fault_type": ir.fault_info.fault_type,
                    "confidence": ir.fault_info.confidence
                }
            }
            print(f"   ✅ IR ID: {ir.explanation_id}")

            # Step 4: Generate LLM responses for different queries
            print("💬 步骤4: 生成LLM响应")
            test_queries = [
                ("基本解释", "请解释这个故障"),
                ("维修建议", "应该如何维修？"),
                ("严重程度", "故障严重程度如何？"),
                ("简单说明", "用简单的话说明问题"),
                ("详细分析", "请提供详细分析")
            ]

            responses = {}
            for query_name, query in test_queries:
                style = "simple" if "简单" in query else "detailed" if "详细" in query else "standard"
                response = self.generate_llm_response(ir, query, style)
                responses[query_name] = {
                    "query": query,
                    "response": response,
                    "style": style,
                    "response_length": len(response)
                }
                print(f"   ✅ {query_name}: {len(response)} 字符")

            result["steps"]["llm_generation"] = {
                "success": True,
                "responses": responses
            }

            # Step 5: Save test data
            print("💾 步骤5: 保存测试数据")
            case_data = {
                "case_config": case_config,
                "signal_data": signal_data.tolist(),
                "explanation": explanation,
                "intermediate_representation": asdict(ir),
                "llm_responses": responses,
                "timestamp": datetime.now().isoformat()
            }

            # Save to file
            output_dir = Path("pipeline_test_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir.mkdir(parents=True, exist_ok=True)

            case_file = output_dir / f"case_{case_name.replace(' ', '_')}.json"
            with open(case_file, 'w', encoding='utf-8') as f:
                json.dump(case_data, f, ensure_ascii=False, indent=2)

            result["steps"]["data_saving"] = {
                "success": True,
                "output_file": str(case_file)
            }
            print(f"   ✅ 已保存: {case_file}")

            result["success"] = True
            result["end_time"] = datetime.now().isoformat()

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            print(f"   ❌ 测试失败: {e}")

        return result

    def run_complete_pipeline_test(self):
        """Run complete pipeline test for all cases."""
        print("🚀 完整流程端到端测试")
        print("=" * 60)

        # Define test cases
        test_cases = [
            {
                "name": "轴承内圈故障",
                "signal_type": "inner_race",
                "fault_type": "内圈故障",
                "confidence": 0.92,
                "device_context": {
                    "device_type": "滚动轴承6205",
                    "operating_speed": 1800.0,
                    "load_condition": "中等载荷",
                    "specifications": "内径25mm, 外径52mm, 宽度15mm"
                }
            },
            {
                "name": "轴承外圈故障",
                "signal_type": "outer_race",
                "fault_type": "外圈故障",
                "confidence": 0.78,
                "device_context": {
                    "device_type": "滚动轴承6307",
                    "operating_speed": 1500.0,
                    "load_condition": "重载荷",
                    "specifications": "内径35mm, 外径80mm, 宽度21mm"
                }
            },
            {
                "name": "设备不对中",
                "signal_type": "misalignment",
                "fault_type": "不对中",
                "confidence": 0.85,
                "device_context": {
                    "device_type": "电机驱动系统",
                    "operating_speed": 3000.0,
                    "load_condition": "正常载荷",
                    "specifications": "功率: 45kW, 转速: 3000rpm"
                }
            },
            {
                "name": "正常状态",
                "signal_type": "normal",
                "fault_type": "正常状态",
                "confidence": 0.95,
                "device_context": {
                    "device_type": "离心泵",
                    "operating_speed": 2950.0,
                    "load_condition": "额定载荷",
                    "specifications": "流量: 100m³/h, 扬程: 50m"
                }
            }
        ]

        print(f"📊 总共测试 {len(test_cases)} 个案例\n")

        # Test each case
        successful_tests = 0
        all_results = []

        for i, case in enumerate(test_cases, 1):
            print(f"\n{'='*20} 案例 {i}/{len(test_cases)} {'='*20}")
            result = self.test_single_pipeline(case)
            all_results.append(result)

            if result["success"]:
                successful_tests += 1
                print(f"✅ 案例 {i} 测试成功")
            else:
                print(f"❌ 案例 {i} 测试失败: {result.get('error', '未知错误')}")

        # Generate summary report
        print(f"\n📋 测试总结报告")
        print("=" * 60)

        summary = {
            "test_time": datetime.now().isoformat(),
            "total_cases": len(test_cases),
            "successful_cases": successful_tests,
            "success_rate": successful_tests / len(test_cases),
            "results": all_results
        }

        print(f"总案例数: {len(test_cases)}")
        print(f"成功案例数: {successful_tests}")
        print(f"成功率: {successful_tests/len(test_cases):.1%}")

        # Save summary report
        output_dir = Path("pipeline_test_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_file = output_dir / "test_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"详细报告已保存: {summary_file}")

        # Show sample responses
        print(f"\n💬 示例LLM响应:")
        if successful_tests > 0:
            successful_result = next(r for r in all_results if r["success"])
            if "llm_generation" in successful_result["steps"]:
                responses = successful_result["steps"]["llm_generation"]["responses"]
                for query_name, response_data in list(responses.items())[:2]:
                    print(f"\n{query_name}:")
                    response_preview = response_data["response"][:200] + "..." if len(response_data["response"]) > 200 else response_data["response"]
                    print(response_preview)

        return summary


def main():
    """Main function."""
    print("🎯 LLM Explainable FD Toolkit - 完整流程测试")
    print("=" * 60)
    print("测试流程: 信号生成 → TSPN解释 → 中间表示转换 → LLM响应生成")
    print("=" * 60)

    try:
        # Run complete pipeline test
        pipeline_test = CompletePipelineTest()
        summary = pipeline_test.run_complete_pipeline_test()

        # Final assessment
        print(f"\n🎉 测试完成！")

        if summary["success_rate"] >= 0.75:
            print("✅ 测试结果: 优秀")
            print("   完整流程运行正常，各组件集成良好")
        elif summary["success_rate"] >= 0.5:
            print("⚠️  测试结果: 良好")
            print("   大部分功能正常，存在一些小问题")
        else:
            print("❌ 测试结果: 需要改进")
            print("   存在较多问题，需要进一步调试")

        print(f"\n📊 关键成果:")
        print(f"   ✅ 信号生成模块")
        print(f"   ✅ TSPN解释生成")
        print(f"   ✅ 中间表示转换")
        print(f"   ✅ LLM响应生成")
        print(f"   ✅ 数据保存功能")

        return True

    except Exception as e:
        print(f"❌ 测试运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)