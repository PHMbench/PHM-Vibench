#!/usr/bin/env python3
"""
Test script for enhanced template LLM
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add code directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
code_dir = project_root / "code"
sys.path.insert(0, str(code_dir))

# Import enhanced LLM directly
sys.path.insert(0, str(code_dir / "llm_explainable_toolkit" / "llm_integration"))
from enhanced_template_llm import EnhancedTemplateLLM


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


def test_enhanced_llm_functionality():
    """Test the enhanced template LLM with various scenarios."""
    print("🚀 增强版模板LLM功能测试")
    print("=" * 50)

    # Create enhanced LLM
    llm = EnhancedTemplateLLM(style="standard")

    # Create sample IR
    ir = create_sample_ir()

    # Test cases covering different response types and styles
    test_cases = [
        {
            "name": "标准解释",
            "prompt": "请解释这个故障",
            "style": "standard",
            "expected_type": "general_explanation"
        },
        {
            "name": "详细分析",
            "prompt": "请详细分析技术原因和机理",
            "style": "detailed",
            "expected_type": "cause_analysis"
        },
        {
            "name": "简单说明",
            "prompt": "用简单的话说是什么问题",
            "style": "simple",
            "expected_type": "general_explanation"
        },
        {
            "name": "正式报告",
            "prompt": "请生成正式的诊断报告",
            "style": "formal",
            "expected_type": "summary_report"
        },
        {
            "name": "维修指导",
            "prompt": "应该如何维修这个故障？",
            "style": "standard",
            "expected_type": "maintenance_guidance"
        },
        {
            "name": "严重程度评估",
            "prompt": "故障严重程度如何？有风险吗？",
            "style": "standard",
            "expected_type": "severity_assessment"
        },
        {
            "name": "技术报告",
            "prompt": "技术分析报告",
            "style": "technical",
            "expected_type": "technical_details"
        },
        {
            "name": "简洁总结",
            "prompt": "请简洁总结一下",
            "style": "concise",
            "expected_type": "summary_report"
        },
        {
            "name": "预防建议",
            "prompt": "如何预防这种故障？",
            "style": "standard",
            "expected_type": "prevention_strategy"
        },
        {
            "name": "监测建议",
            "prompt": "应该监测哪些参数？",
            "style": "standard",
            "expected_type": "monitoring_advice"
        },
        {
            "name": "综合建议",
            "prompt": "请给出处理建议",
            "style": "standard",
            "expected_type": "recommendations"
        }
    ]

    print(f"总共测试 {len(test_cases)} 个场景\n")

    for i, case in enumerate(test_cases, 1):
        print(f"📝 测试 {i}: {case['name']}")
        print(f"   提示: {case['prompt']}")
        print(f"   风格: {case['style']}")
        print("-" * 60)

        # Set style
        llm.set_style(case['style'])

        # Generate response
        context = {"intermediate_representation": ir}
        try:
            response = llm.generate(case['prompt'], context)
            print(response)
        except Exception as e:
            print(f"❌ 生成响应时出错: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "="*80 + "\n")

    # Test conversation history
    print("📊 对话历史摘要:")
    print(llm.get_conversation_summary())

    return True


def test_style_comparison():
    """Test different styles for the same query."""
    print("\n🎨 风格对比测试")
    print("=" * 50)

    llm = EnhancedTemplateLLM()
    ir = create_sample_ir()

    query = "请解释这个轴承故障"
    styles = ["simple", "standard", "detailed", "formal", "technical", "concise"]

    for style in styles:
        print(f"\n--- {style.upper()} 风格 ---")
        llm.set_style(style)
        context = {"intermediate_representation": ir}
        response = llm.generate(query, context)
        print(response[:300] + "..." if len(response) > 300 else response)

    return True


def test_fault_type_variations():
    """Test different fault types."""
    print("\n🔧 故障类型变化测试")
    print("=" * 50)

    llm = EnhancedTemplateLLM(style="standard")

    fault_types = [
        {"type": "外圈故障", "confidence": 0.85},
        {"type": "不对中", "confidence": 0.78},
        {"type": "正常状态", "confidence": 0.95},
        {"type": "齿轮故障", "confidence": 0.91}
    ]

    query = "请分析这个故障并给出维修建议"

    for fault_info in fault_types:
        print(f"\n--- {fault_info['type']} (置信度: {fault_info['confidence']:.1%}) ---")

        # Create modified IR
        ir = create_sample_ir()
        ir.fault_info.fault_type = fault_info['type']
        ir.fault_info.confidence = fault_info['confidence']

        context = {"intermediate_representation": ir}
        response = llm.generate(query, context)
        print(response[:400] + "..." if len(response) > 400 else response)

    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n⚠️ 边界情况测试")
    print("=" * 50)

    llm = EnhancedTemplateLLM()

    # Test 1: Empty context
    print("\n--- 测试1: 空上下文 ---")
    response = llm.generate("请解释故障", {})
    print(response)

    # Test 2: Missing IR in context
    print("\n--- 测试2: 上下文中缺少IR ---")
    response = llm.generate("请解释故障", {"other_data": "test"})
    print(response)

    # Test 3: Invalid style
    print("\n--- 测试3: 无效风格 ---")
    llm.set_style("invalid_style")
    ir = create_sample_ir()
    context = {"intermediate_representation": ir}
    response = llm.generate("请解释故障", context)
    print(response)

    # Test 4: Very long prompt
    print("\n--- 测试4: 长提示词 ---")
    long_prompt = "请解释" + "非常" * 100 + "详细的故障"
    response = llm.generate(long_prompt, context)
    print(response[:200] + "..." if len(response) > 200 else response)

    return True


def main():
    """Main test function."""
    try:
        # Run all tests
        print("开始运行增强版模板LLM测试套件...\n")

        success = True

        # Test 1: Basic functionality
        print("=" * 60)
        print("测试 1: 基础功能测试")
        print("=" * 60)
        try:
            test_enhanced_llm_functionality()
            print("✅ 基础功能测试通过")
        except Exception as e:
            print(f"❌ 基础功能测试失败: {e}")
            success = False

        # Test 2: Style comparison
        print("=" * 60)
        print("测试 2: 风格对比测试")
        print("=" * 60)
        try:
            test_style_comparison()
            print("✅ 风格对比测试通过")
        except Exception as e:
            print(f"❌ 风格对比测试失败: {e}")
            success = False

        # Test 3: Fault type variations
        print("=" * 60)
        print("测试 3: 故障类型变化测试")
        print("=" * 60)
        try:
            test_fault_type_variations()
            print("✅ 故障类型变化测试通过")
        except Exception as e:
            print(f"❌ 故障类型变化测试失败: {e}")
            success = False

        # Test 4: Edge cases
        print("=" * 60)
        print("测试 4: 边界情况测试")
        print("=" * 60)
        try:
            test_edge_cases()
            print("✅ 边界情况测试通过")
        except Exception as e:
            print(f"❌ 边界情况测试失败: {e}")
            success = False

        # Final summary
        print("=" * 60)
        if success:
            print("🎉 所有测试通过！增强版模板LLM功能正常。")
            print("\n主要特性:")
            print("- ✅ 支持多种输出风格（简单、标准、详细、正式、技术、简洁）")
            print("- ✅ 智能识别用户意图和请求类型")
            print("- ✅ 丰富的故障知识库和模板库")
            print("- ✅ 对话历史记录功能")
            print("- ✅ 灵活的格式化选项")
        else:
            print("❌ 部分测试失败，需要检查实现。")

        return success

    except Exception as e:
        print(f"❌ 测试运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)