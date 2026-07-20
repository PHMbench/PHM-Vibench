#!/usr/bin/env python3
"""
Interactive LLM Demo with Real TSPN Integration

This script provides an interactive interface for demonstrating the LLM
explanation capabilities with real TSPN model integration.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add paths
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root / "code"))
sys.path.insert(0, str(project_root / "code" / "llm_explainable_toolkit" / "core"))
sys.path.insert(0, str(project_root / "code" / "llm_explainable_toolkit" / "llm_integration"))

try:
    from toolkit_bridge import ExplainableToolkitBridge, create_demo_signal_data
except ImportError:
    # Fallback - use isolated version
    print("⚠️  使用隔离版本进行演示")
    from test_bridge_isolated import IsolatedToolkitBridge as ExplainableToolkitBridge, create_demo_signal_data

try:
    from enhanced_template_llm import IsolatedEnhancedLLM
except ImportError:
    print("❌ 无法导入增强LLM")
    sys.exit(1)


class InteractiveLLMDemo:
    """
    Interactive demonstration of LLM-based fault diagnosis explanation.
    """

    def __init__(self):
        """Initialize the demo."""
        self.bridge = ExplainableToolkitBridge()
        self.llm = IsolatedEnhancedLLM(style="standard")
        self.current_ir = None
        self.demo_cases = self._create_demo_cases()
        self.current_case_index = 0

    def _create_demo_cases(self) -> List[Dict[str, Any]]:
        """Create demonstration cases."""
        return [
            {
                "name": "轴承内圈故障",
                "description": "高置信度检测到的典型轴承内圈故障",
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
                "description": "中等置信度的轴承外圈故障",
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
                "description": "设备对中不良故障",
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
                "description": "设备正常运行状态",
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

    def load_demo_case(self, case_index: int) -> bool:
        """Load a demonstration case."""
        if case_index < 0 or case_index >= len(self.demo_cases):
            print(f"❌ 案例索引 {case_index} 超出范围")
            return False

        case = self.demo_cases[case_index]
        self.current_case_index = case_index

        print(f"\n🔄 加载案例: {case['name']}")
        print(f"   描述: {case['description']}")

        try:
            # Generate signal data
            signal_data = create_demo_signal_data(case["signal_type"])

            # Generate explanation
            explanation = self.bridge.generate_mock_tspn_explanation(
                signal_data=signal_data,
                fault_type=case["fault_type"],
                confidence=case["confidence"]
            )

            # Convert to intermediate representation
            self.current_ir = self.bridge.convert_to_intermediate_representation(
                explanation=explanation,
                signal_data=signal_data,
                device_context=case["device_context"]
            )

            print(f"   ✅ 案例加载成功")
            print(f"   故障类型: {self.current_ir.fault_info.fault_type}")
            print(f"   置信度: {self.current_ir.fault_info.confidence:.1%}")
            print(f"   设备类型: {self.current_ir.device_context.device_type}")

            return True

        except Exception as e:
            print(f"   ❌ 案例加载失败: {e}")
            return False

    def show_current_case_info(self):
        """Show information about the current case."""
        if not self.current_ir:
            print("❌ 当前没有加载的案例")
            return

        ir = self.current_ir

        print(f"\n📋 当前案例信息:")
        print(f"   案例名称: {self.demo_cases[self.current_case_index]['name']}")
        print(f"   故障类型: {ir.fault_info.fault_type}")
        print(f"   置信度: {ir.fault_info.confidence:.1%}")
        print(f"   严重程度: {ir.fault_info.severity}")
        print(f"   设备类型: {ir.device_context.device_type}")

        if ir.signal_analysis:
            print(f"   信号长度: {ir.signal_analysis.signal_length} 点")
            print(f"   采样率: {ir.signal_analysis.sampling_rate} Hz")
            print(f"   振动强度: {ir.signal_analysis.statistics.get('rms', 0):.2f}")

        if ir.signal_analysis.key_findings:
            print(f"   关键发现:")
            for i, finding in enumerate(ir.signal_analysis.key_findings[:3], 1):
                print(f"     {i}. {finding}")

    def process_user_query(self, query: str) -> str:
        """Process user query and generate response."""
        if not self.current_ir:
            return "请先加载一个案例数据"

        try:
            # Auto-detect and set style based on query
            query_lower = query.lower()
            if any(keyword in query_lower for keyword in ["简单", "通俗"]):
                self.llm.set_style("simple")
            elif any(keyword in query_lower for keyword in ["详细", "全面", "深入"]):
                self.llm.set_style("detailed")
            elif any(keyword in query_lower for keyword in ["正式", "报告"]):
                self.llm.set_style("formal")
            elif any(keyword in query_lower for keyword in ["简洁", "简短"]):
                self.llm.set_style("concise")
            else:
                self.llm.set_style("standard")

            # Generate response
            context = {"intermediate_representation": self.current_ir}
            response = self.llm.generate(query, context)

            return response

        except Exception as e:
            return f"处理查询时出错: {str(e)}"

    def show_style_options(self):
        """Show available explanation styles."""
        print(f"\n🎨 可用的解释风格:")
        styles = ["standard", "simple", "detailed", "formal", "concise"]
        for style in styles:
            description = {
                "standard": "标准风格 - 平衡详细度和易读性",
                "simple": "简单风格 - 通俗易懂，适合非技术人员",
                "detailed": "详细风格 - 全面深入的技术分析",
                "formal": "正式风格 - 专业的诊断报告格式",
                "concise": "简洁风格 - 简明扼要的要点总结"
            }
            print(f"   • {style}: {description[style]}")

    def show_query_examples(self):
        """Show example queries."""
        print(f"\n💡 示例问题:")
        examples = [
            "请解释这个故障的原因",
            "应该如何维修？",
            "故障严重程度如何？",
            "用简单的话说明问题",
            "请提供详细的技术分析",
            "生成正式的诊断报告",
            "如何预防这种故障？",
            "需要监测哪些参数？",
            "请总结一下情况"
        ]

        for i, example in enumerate(examples, 1):
            print(f"   {i}. {example}")

    def run_interactive_mode(self):
        """Run interactive conversation mode."""
        print(f"\n🤖 进入交互模式")
        print(f"   输入 'help' 查看帮助")
        print(f"   输入 'quit' 退出交互模式")
        print(f"   输入 'case <编号>' 切换案例（1-{len(self.demo_cases)}）")
        print(f"   输入 'info' 查看当前案例信息")
        print(f"   输入 'styles' 查看可用风格")
        print(f"   输入 'examples' 查看示例问题")

        while True:
            try:
                user_input = input(f"\n❓ 请输入您的问题: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == 'quit':
                    print("👋 退出交互模式")
                    break

                elif user_input.lower() == 'help':
                    print(f"\n📖 帮助信息:")
                    print(f"   • 直接输入问题进行对话")
                    print(f"   • 输入 'quit' 退出")
                    print(f"   • 输入 'case <编号>' 切换案例")
                    print(f"   • 输入 'info' 查看当前案例")
                    print(f"   • 输入 'styles' 查看可用风格")
                    print(f"   • 输入 'examples' 查看示例问题")

                elif user_input.lower().startswith('case '):
                    try:
                        case_num = int(user_input.split()[1]) - 1
                        if self.load_demo_case(case_num):
                            print(f"✅ 已切换到案例 {case_num + 1}")
                        else:
                            print(f"❌ 切换案例失败")
                    except (ValueError, IndexError):
                        print(f"❌ 无效的案例编号，请使用 1-{len(self.demo_cases)}")

                elif user_input.lower() == 'info':
                    self.show_current_case_info()

                elif user_input.lower() == 'styles':
                    self.show_style_options()

                elif user_input.lower() == 'examples':
                    self.show_query_examples()

                else:
                    # Process as a query
                    print(f"\n🤖 思考中...")
                    response = self.process_user_query(user_input)
                    print(f"\n💬 回答:")
                    print(response)

            except KeyboardInterrupt:
                print(f"\n\n👋 交互被中断，再见！")
                break
            except Exception as e:
                print(f"\n❌ 处理输入时出错: {e}")

    def run_batch_demo(self):
        """Run batch demonstration with all cases."""
        print(f"\n🚀 批量演示模式")
        print("=" * 60)

        for i, case in enumerate(self.demo_cases):
            print(f"\n📊 案例 {i+1}: {case['name']}")
            print("-" * 40)

            # Load case
            if not self.load_demo_case(i):
                continue

            # Test different query types
            test_queries = [
                ("基本解释", "请解释这个故障"),
                ("简单说明", "用简单的话说明问题"),
                ("维修指导", "应该如何维修？"),
                ("严重程度", "故障严重程度如何？"),
                ("详细分析", "请提供详细的技术分析"),
                ("正式报告", "生成正式的诊断报告")
            ]

            for query_name, query in test_queries:
                print(f"\n🔍 {query_name}:")
                try:
                    response = self.process_user_query(query)
                    # Limit response length for batch demo
                    display_response = response[:300] + "..." if len(response) > 300 else response
                    print(display_response)
                except Exception as e:
                    print(f"❌ 生成响应失败: {e}")

        print(f"\n✅ 批量演示完成")

    def save_conversation_log(self, filename: Optional[str] = None):
        """Save conversation history to file."""
        if not hasattr(self.llm, 'conversation_history'):
            print("❌ 没有对话历史可保存")
            return

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_log_{timestamp}.json"

        try:
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "current_case": self.current_case_index,
                "case_info": self.demo_cases[self.current_case_index] if self.current_case_index < len(self.demo_cases) else None,
                "conversation_history": self.llm.conversation_history
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)

            print(f"💾 对话记录已保存到: {filename}")

        except Exception as e:
            print(f"❌ 保存对话记录失败: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Interactive LLM Demo")
    parser.add_argument("--mode", choices=["interactive", "batch"], default="interactive",
                       help="Demo mode to run")
    parser.add_argument("--case", type=int, default=1,
                       help="Starting case number (1-4)")
    parser.add_argument("--save-log", action="store_true",
                       help="Save conversation log")

    args = parser.parse_args()

    print("🚀 LLM故障诊断解释交互演示系统")
    print("=" * 50)

    # Initialize demo
    demo = InteractiveLLMDemo()

    try:
        # Load initial case
        if not demo.load_demo_case(args.case - 1):
            print("❌ 无法加载初始案例")
            return False

        # Run demo
        if args.mode == "interactive":
            demo.run_interactive_mode()
        else:
            demo.run_batch_demo()

        # Save conversation log if requested
        if args.save_log:
            demo.save_conversation_log()

        print(f"\n✅ 演示完成！")
        return True

    except KeyboardInterrupt:
        print(f"\n\n👋 演示被中断，再见！")
        return True
    except Exception as e:
        print(f"\n❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)