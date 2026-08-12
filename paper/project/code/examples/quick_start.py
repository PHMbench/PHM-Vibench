#!/usr/bin/env python3
"""
Quick Start Example

Demonstrates basic usage of the LLM-Enhanced Fault Diagnosis Toolkit.
"""

import sys
import os
import numpy as np
import torch

# Add the toolkit to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from llm_explainable_toolkit import DiagnosticSystem, create_toolkit


def create_sample_signal():
    """Create a sample vibration signal with bearing fault characteristics."""
    # Signal parameters
    fs = 1024  # Sampling frequency (Hz)
    duration = 4  # seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Simulate bearing fault
    shaft_freq = 30  # Hz
    bpfi = 3.05 * shaft_freq  # Ball pass frequency inner race
    bpfo = 2.05 * shaft_freq  # Ball pass frequency outer race

    # Generate signal with fault characteristics
    signal = (
        0.5 * np.sin(2 * np.pi * shaft_freq * t) +  # Shaft vibration
        0.3 * np.sin(2 * np.pi * bpfi * t) +        # Inner race fault
        0.2 * np.sin(2 * np.pi * bpfo * t) +        # Outer race contribution
        0.1 * np.sin(2 * np.pi * 2 * bpfi * t) +    # Harmonics
        0.05 * np.random.randn(len(t))               # Noise
    )

    return signal


def example_1_basic_diagnosis():
    """Example 1: Basic fault diagnosis with explanation."""
    print("=" * 60)
    print("示例1: 基础故障诊断与解释")
    print("=" * 60)

    # Create sample signal
    signal_data = create_sample_signal()
    print(f"✓ 生成样本信号: {signal_data.shape}")

    # Create diagnostic system
    # Note: In real usage, you would configure LLM API keys
    llm_config = {
        'providers': {
            'mock': {
                'type': 'local',
                'model_path': 'mock_model'
            }
        }
    }
    system = create_toolkit(llm_config=llm_config)
    print("✓ 初始化诊断系统")

    # Perform diagnosis
    print("\n🔍 执行故障诊断...")
    result = system.diagnose(
        signal_data,
        style="standard"
    )

    # Display results
    fault_type = result["model_prediction"]["fault_type"]
    confidence = result["model_prediction"]["confidence"]
    print(f"📊 诊断结果: {fault_type}")
    print(f"📈 置信度: {confidence:.1%}")

    # Display explanation
    if result["explanation"]["natural_language_explanation"]:
        print(f"\n💬 自然语言解释:")
        print(result["explanation"]["natural_language_explanation"])

    # Display recommendations
    print(f"\n🛠️ 维修建议:")
    for rec in result["explanation"]["recommendations"][:2]:
        print(f"  • {rec['action']} (优先级: {rec['priority']})")


def example_2_conversation():
    """Example 2: Interactive diagnostic conversation."""
    print("\n" + "=" * 60)
    print("示例2: 交互式诊断对话")
    print("=" * 60)

    # Create diagnostic system
    system = create_toolkit()

    # Create sample signal
    signal_data = create_sample_signal()

    # Device information
    device_info = {
        "device_type": "电机",
        "operating_speed": 1800,
        "criticality_level": "high",
        "operating_hours": 15000
    }

    print("🚀 开始诊断对话...")
    session_result = system.start_conversation(
        signal_data,
        device_info=device_info
    )

    session_id = session_result["session_id"]
    print(f"✓ 对话会话ID: {session_id}")
    print(f"\n{session_result['greeting']}")

    # Simulate conversation
    test_questions = [
        "这个故障的主要原因是什么？",
        "应该如何维修这个故障？",
        "故障严重程度如何评估？"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n👤 用户 ({i}): {question}")
        response = system.continue_conversation(session_id, question)
        print(f"🤖 助手: {response}")

    # End conversation
    conclusion = system.end_conversation(session_id)
    print(f"\n📋 对话总结:")
    print(f"  时长: {conclusion['duration_seconds']:.0f} 秒")
    print(f"  消息数: {conclusion['num_messages']}")
    print(f"  {conclusion['conclusion']}")


def example_3_batch_processing():
    """Example 3: Batch diagnosis of multiple signals."""
    print("\n" + "=" * 60)
    print("示例3: 批量故障诊断")
    print("=" * 60)

    # Create diagnostic system
    system = create_toolkit()

    # Generate multiple sample signals
    print("📊 生成批量信号数据...")
    signals = []
    for i in range(3):
        signal = create_sample_signal()
        # Add some variation
        signal += 0.1 * np.random.randn(len(signal))
        signals.append(signal)

    print(f"✓ 生成 {len(signals)} 个信号样本")

    # Batch processing configuration
    batch_config = {
        "style": "detailed",
        "context": {
            "batch_processing": True,
            "analysis_type": "comparative"
        }
    }

    # Perform batch diagnosis
    print("\n🔍 执行批量诊断...")
    results = system.batch_diagnose(signals, batch_config)

    # Display summary
    print(f"📊 批量诊断结果:")
    for i, result in enumerate(results):
        if "error" in result:
            print(f"  样本 {i}: 诊断失败 - {result['error']}")
        else:
            fault_type = result["model_prediction"]["fault_type"]
            confidence = result["model_prediction"]["confidence"]
            print(f"  样本 {i}: {fault_type} (置信度: {confidence:.1%})")

    # Display diagnostic history
    history = system.get_diagnostic_history(limit=5)
    print(f"\n📚 诊断历史: 最近 {len(history)} 次记录")


def example_4_custom_explanation():
    """Example 4: Custom explanation with user query."""
    print("\n" + "=" * 60)
    print("示例4: 自定义解释和用户查询")
    print("=" * 60)

    # Create diagnostic system
    system = create_toolkit()

    # Create sample signal
    signal_data = create_sample_signal()

    # User-specific queries
    custom_queries = [
        "请详细解释这个内圈故障的频谱特征",
        "对于这种故障，应该准备哪些维修工具和备件？",
        "如何通过日常维护来预防这种故障的发生？"
    ]

    for i, query in enumerate(custom_queries, 1):
        print(f"\n🔍 查询 {i}: {query}")
        result = system.diagnose(
            signal_data,
            user_query=query,
            style="expert"
        )

        if result["explanation"]["natural_language_explanation"]:
            print(f"💬 回答: {result['explanation']['natural_language_explanation'][:200]}...")


def example_5_export_and_analysis():
    """Example 5: Data export and analysis."""
    print("\n" + "=" * 60)
    print("示例5: 数据导出和分析")
    print("=" * 60)

    # Create diagnostic system
    system = create_toolkit()

    # Perform multiple diagnoses
    signals = [create_sample_signal() for _ in range(2)]
    for signal in signals:
        system.diagnose(signal, style="standard")

    # Export system data
    export_path = "./system_data_export.json"
    system.export_data(export_path)
    print(f"📤 数据已导出到: {export_path}")

    # Get system information
    system_info = system.get_system_info()
    print(f"\n📊 系统信息:")
    print(f"  工具包版本: {system_info['toolkit_version']}")
    print(f"  活跃会话: {system_info['active_sessions']}")
    print(f"  诊断历史: {system_info['diagnostic_history_size']} 条记录")

    # Get diagnostic statistics
    history = system.get_diagnostic_history(limit=10)
    fault_types = [item.get("model_prediction", {}).get("fault_type", "unknown")
                  for item in history]

    if fault_types:
        from collections import Counter
        fault_counts = Counter(fault_types)
        print(f"\n📈 故障类型统计:")
        for fault_type, count in fault_counts.items():
            print(f"  {fault_type}: {count} 次")


def main():
    """Run all examples."""
    print("🚀 LLM增强故障诊断工具包 - 快速开始演示")
    print("=" * 60)

    try:
        # Run examples
        example_1_basic_diagnosis()
        example_2_conversation()
        example_3_batch_processing()
        example_4_custom_explanation()
        example_5_export_and_analysis()

        print("\n" + "=" * 60)
        print("✅ 所有示例执行完成！")
        print("\n📚 更多功能:")
        print("• 查看 code/examples/ 目录获取更多示例")
        print("• 阅读 README.md 了解详细配置")
        print("• 运行 experiments/ 中的实验脚本")
        print("• 使用 Jupyter notebooks 进行交互式分析")

    except Exception as e:
        print(f"\n❌ 示例执行失败: {e}")
        print("\n💡 解决建议:")
        print("1. 确保已安装所有依赖: pip install -r requirements.txt")
        print("2. 检查Python版本 >= 3.8")
        print("3. 配置LLM API密钥（用于实际LLM功能）")
        print("4. 查看错误日志了解详细信息")


if __name__ == "__main__":
    main()