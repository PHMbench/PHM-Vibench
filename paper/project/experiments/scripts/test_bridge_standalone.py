#!/usr/bin/env python3
"""
Standalone test for toolkit bridge
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add the code directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
code_dir = project_root / "code"
sys.path.insert(0, str(code_dir))

# Import just the bridge directly without going through __init__.py
sys.path.insert(0, str(code_dir / "llm_explainable_toolkit" / "core"))
from toolkit_bridge import (
    ExplainableToolkitBridge,
    create_demo_signal_data
)


def test_bridge_functionality():
    """Test the bridge functionality."""
    print("🔗 测试Toolkit桥接功能...")

    # Initialize bridge
    bridge = ExplainableToolkitBridge()

    # Test 1: Generate mock TSPN explanation
    print("\n📊 测试1: 生成模拟TSPN解释")
    signal_data = create_demo_signal_data("inner_race")
    explanation = bridge.generate_mock_tspn_explanation(
        signal_data=signal_data,
        fault_type="内圈故障",
        confidence=0.92
    )

    print(f"   ✅ 生成解释完成")
    print(f"   故障类型: {explanation['fault_type']}")
    print(f"   置信度: {explanation['confidence']:.1%}")
    print(f"   信号长度: {explanation['signal_length']}")
    print(f"   重要特征数: {len(explanation['important_features'])}")

    # Test 2: Convert to intermediate representation
    print("\n🔄 测试2: 转换为中间表示")
    ir = bridge.convert_toolkit_explanation_to_ir(
        explanation=explanation,
        signal_data=signal_data,
        device_context={
            "device_type": "滚动轴承",
            "operating_speed": 1800.0,
            "load_condition": "正常载荷"
        }
    )

    print(f"   ✅ 转换完成")
    print(f"   表示ID: {ir.explanation_id}")
    print(f"   故障信息: {ir.fault_info.fault_type} (置信度: {ir.fault_info.confidence:.1%})")
    print(f"   设备类型: {ir.device_context.device_type}")
    print(f"   关键发现数: {len(ir.signal_analysis.key_findings)}")

    # Test 3: Save explanation batch
    print("\n💾 测试3: 批量保存解释")
    explanations = []

    # Generate multiple explanations
    signal_types = ["inner_race", "outer_race", "misalignment", "normal"]
    for signal_type in signal_types:
        signal_data = create_demo_signal_data(signal_type)
        explanation = bridge.generate_mock_tspn_explanation(
            signal_data=signal_data,
            fault_type=bridge._map_signal_to_fault(signal_type) if hasattr(bridge, '_map_signal_to_fault') else signal_type,
            confidence=np.random.uniform(0.75, 0.95)
        )
        explanations.append(explanation)

    # Save batch
    saved_files = bridge.save_explanation_batch(
        explanations,
        output_dir="test_output",
        format="json"
    )

    print(f"   ✅ 保存完成")
    print(f"   生成解释数: {len(explanations)}")
    print(f"   保存文件数: {len(saved_files)}")

    # Test 4: Load and verify saved files
    print("\n📂 测试4: 加载和验证保存的文件")
    for i, saved_file in enumerate(saved_files[:2]):  # Test first 2 files
        loaded_explanation = bridge.load_explanation_from_file(saved_file)
        original_explanation = explanations[i]

        # Basic verification
        assert loaded_explanation['fault_type'] == original_explanation['fault_type']
        assert abs(loaded_explanation['confidence'] - original_explanation['confidence']) < 1e-6
        print(f"   ✅ 文件 {saved_file.name} 验证通过")

    print(f"\n🎉 所有测试通过！")
    return True


def generate_sample_explanations():
    """Generate sample explanations for different fault types."""
    print("\n🚀 生成样本解释数据...")

    bridge = ExplainableToolkitBridge()
    explanations = []

    # Define fault scenarios
    fault_scenarios = [
        {
            "signal_type": "inner_race",
            "fault_type": "内圈故障",
            "confidence": 0.91,
            "device_context": {
                "device_type": "滚动轴承6205",
                "operating_speed": 1800.0,
                "load_condition": "中等载荷",
                "specifications": "内径25mm, 外径52mm, 宽度15mm"
            }
        },
        {
            "signal_type": "outer_race",
            "fault_type": "外圈故障",
            "confidence": 0.88,
            "device_context": {
                "device_type": "滚动轴承6307",
                "operating_speed": 1500.0,
                "load_condition": "重载荷",
                "specifications": "内径35mm, 外径80mm, 宽度21mm"
            }
        },
        {
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

    for scenario in fault_scenarios:
        # Generate signal data
        signal_data = create_demo_signal_data(scenario["signal_type"])

        # Generate explanation
        explanation = bridge.generate_mock_tspn_explanation(
            signal_data=signal_data,
            fault_type=scenario["fault_type"],
            confidence=scenario["confidence"]
        )

        # Add device context
        explanation["device_context"] = scenario["device_context"]

        # Add metadata
        explanation["metadata"] = {
            "signal_type": scenario["signal_type"],
            "generation_time": datetime.now().isoformat(),
            "description": f"模拟{scenario['fault_type']}场景"
        }

        explanations.append(explanation)

        print(f"   ✅ {scenario['fault_type']}: 置信度 {scenario['confidence']:.1%}")

    # Save explanations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("sample_explanations") / timestamp
    saved_files = bridge.save_explanation_batch(explanations, output_dir)

    print(f"\n💾 样本解释已保存到: {output_dir}")
    print(f"   文件数量: {len(saved_files)}")

    # Create a summary file
    summary = {
        "generation_time": datetime.now().isoformat(),
        "total_explanations": len(explanations),
        "fault_types": list(set(exp['fault_type'] for exp in explanations)),
        "files": [f.name for f in saved_files],
        "description": "TSPN模型解释样本数据，用于LLM工具包测试"
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"📋 摘要文件: {summary_file}")

    return explanations


def test_integration_pipeline():
    """Test the complete integration pipeline."""
    print("\n🔄 测试完整集成流程...")

    bridge = ExplainableToolkitBridge()

    # Step 1: Generate signal
    signal_data = create_demo_signal_data("inner_race")
    print(f"   步骤1: 生成信号数据 (长度: {len(signal_data)})")

    # Step 2: Generate explanation
    explanation = bridge.generate_mock_tspn_explanation(
        signal_data=signal_data,
        fault_type="内圈故障",
        confidence=0.89
    )
    print(f"   步骤2: 生成TSPN解释")

    # Step 3: Convert to IR
    ir = bridge.convert_toolkit_explanation_to_ir(
        explanation=explanation,
        signal_data=signal_data,
        device_context={
            "device_type": "滚动轴承",
            "operating_speed": 1800.0,
            "load_condition": "正常载荷"
        }
    )
    print(f"   步骤3: 转换为中间表示")

    # Step 4: Test IR methods
    ir_dict = ir.to_dict()
    print(f"   步骤4: 序列化测试 (字段数: {len(ir_dict)})")

    # Step 5: Verify key components
    assert ir.fault_info.fault_type == "内圈故障"
    assert ir.fault_info.confidence == 0.89
    assert len(ir.signal_analysis.key_findings) > 0
    assert len(ir.technical_explanation.important_features) > 0
    print(f"   步骤5: 组件验证通过")

    print(f"   ✅ 集成流程测试完成")
    return True


def main():
    """Main function."""
    print("🚀 Toolkit Bridge 独立测试")
    print("=" * 50)

    try:
        # Test bridge functionality
        if test_bridge_functionality():
            print("✅ 桥接功能测试通过")

        # Generate sample explanations
        explanations = generate_sample_explanations()
        print("✅ 样本生成完成")

        # Test integration pipeline
        if test_integration_pipeline():
            print("✅ 集成流程测试通过")

        print(f"\n🎉 所有测试成功完成！")
        print(f"   生成样本解释: {len(explanations)} 个")
        print(f"   输出目录: sample_explanations/")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)