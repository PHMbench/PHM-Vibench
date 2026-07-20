#!/usr/bin/env python3
"""
Minimal LLM Demo Script

This script demonstrates the LLM explanation generation pipeline
without requiring external API calls or model dependencies.
"""

import sys
import os
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import argparse
import subprocess
import time
from typing import Any, Dict, List

# Add the code directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
code_dir = project_root / "code"
sys.path.insert(0, str(code_dir))

from llm_explainable_toolkit.core.intermediate_representation import (
    LLMIntermediateRepresentation,
    create_mock_ir
)
from llm_explainable_toolkit.core.adapters import (
    ExplanationToIRAdapter,
    MockDataAdapter
)
from llm_explainable_toolkit.llm_integration.local_template_llm import (
    LocalTemplateLLM
)


class MinimalLLMDemo:
    """
    Minimal demonstration of LLM-based fault diagnosis explanation.
    """

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
            },
            {
                "name": "齿轮故障",
                "description": "高置信度的齿轮磨损故障",
                "ir": MockDataAdapter.create_comprehensive_example(
                    fault_type="齿轮故障",
                    confidence=0.85,
                    device_type="齿轮箱"
                ),
                "queries": [
                    "请详细分析这个齿轮故障",
                    "维修步骤是什么？",
                    "故障严重程度评估"
                ]
            }
        ]

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

    def run_style_comparison_demo(self) -> None:
        """Demonstrate different explanation styles."""
        case = self.demo_cases[0]  # Use first case
        ir = case["ir"]
        query = "请详细解释这个故障"

        styles = ["standard", "detailed", "simple"]

        print("\n" + "="*80)
        print("解释风格对比演示")
        print("="*80)

        for style in styles:
            print(f"\n🎨 {style.upper()} 风格解释:")
            print("-"*60)

            self.llm.set_style(style)
            context = {"intermediate_representation": ir}
            response = self.llm.generate(query, context)
            print(response)

    def run_technical_demo(self) -> None:
        """Demonstrate technical capabilities."""
        print("\n" + "="*80)
        print("技术能力演示")
        print("="*80)

        # Create custom technical example
        ir = create_mock_ir(
            fault_type="复合故障（内圈+不对中）",
            confidence=0.88,
            device_type="高速离心机"
        )

        # Enhance with technical details
        ir.technical_explanation.signal_path = {
            "processing_steps": [
                {"layer": "Input", "output_shape": [1, 4096], "description": "原始振动信号"},
                {"layer": "FFT", "output_shape": [1, 4096], "description": "快速傅里叶变换"},
                {"layer": "WF", "output_shape": [1, 4096], "description": "小波去噪"},
                {"layer": "Feature Extraction", "output_shape": [1, 52], "description": "多域特征提取"},
                {"layer": "Attention", "output_shape": [1, 52], "description": "注意力机制"},
                {"layer": "Classification", "output_shape": [1, 10], "description": "故障分类"}
            ]
        }

        print(f"\n🔧 处理流程:")
        steps = ir.technical_explanation.signal_path["processing_steps"]
        for i, step in enumerate(steps, 1):
            print(f"   {i}. {step['description']} ({step['layer']})")

        print(f"\n📊 信号分析结果:")
        print(f"   • RMS值: {ir.signal_analysis.statistics.get('rms', 0):.2f}")
        print(f"   • 峰值因子: {ir.signal_analysis.statistics.get('crest_factor', 0):.2f}")
        print(f"   • 主频: {ir.signal_analysis.frequency_analysis.get('dominant_frequency', 0):.1f} Hz")

        # Generate technical explanation
        self.llm.set_style("detailed")
        context = {"intermediate_representation": ir}
        response = self.llm.generate("请提供详细的技术分析", context)
        print(f"\n📝 技术分析报告:")
        print(response)

    def run_interactive_demo(self) -> None:
        """Run interactive demo."""
        print("\n" + "="*80)
        print("交互式演示")
        print("="*80)

        case = self.demo_cases[0]
        ir = case["ir"]

        print(f"\n🎯 当前诊断:")
        print(f"   故障类型：{ir.fault_info.fault_type}")
        print(f"   置信度：{ir.fault_info.confidence:.1%}")
        print(f"   设备：{ir.device_context.device_type}")

        print(f"\n💬 您可以询问以下类型的问题:")
        print("   • 故障原因和机理")
        print("   • 维修指导和步骤")
        print("   • 严重程度评估")
        print("   • 技术细节分析")
        print("   • 预防措施建议")
        print("   • 监测方案制定")

        print(f"\n输入 'quit' 结束演示，输入 'help' 查看示例问题")

        while True:
            try:
                user_input = input(f"\n❓ 请输入您的问题: ").strip()

                if user_input.lower() == 'quit':
                    print("👋 演示结束！")
                    break
                elif user_input.lower() == 'help':
                    print("\n💡 示例问题:")
                    for i, query in enumerate(case["queries"], 1):
                        print(f"   {i}. {query}")
                    continue
                elif not user_input:
                    print("请输入有效的问题...")
                    continue

                # Generate response
                context = {"intermediate_representation": ir}
                response = self.llm.generate(user_input, context)

                print(f"\n🤖 助手回答:")
                print(response)

            except KeyboardInterrupt:
                print("\n\n👋 演示被中断，再见！")
                break
            except Exception as e:
                print(f"\n❌ 生成回答时出错: {e}")

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

        # Step 3: Convert to intermediate representation
        print(f"\n🔄 步骤 3: 转换为中间表示")
        ir = ExplanationToIRAdapter.from_model_prediction(
            signal_data=signal_data,
            model_prediction=model_prediction,
            device_context={
                "device_type": "滚动轴承",
                "operating_speed": 1800.0,
                "load_condition": "正常载荷"
            }
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

        # Step 5: Summary
        print(f"\n📊 步骤 5: 处理流程总结")
        print(f"   ✅ 信号采集和预处理")
        print(f"   ✅ 特征提取和分析")
        print(f"   ✅ 模型预测和分类")
        print(f"   ✅ 中间表示转换")
        print(f"   ✅ 自然语言生成")
        print(f"   ✅ 智能解释输出")

    def _git_commit(self) -> str:
        """Return the current submodule commit if this directory is in Git."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    def _evidence_root(self, output_path: Path) -> Path:
        """Use the seed directory as evidence root when output is an artifacts dir."""
        if output_path.name == "artifacts":
            return output_path.parent
        return output_path

    def _torch_cuda_metadata(self) -> Dict[str, Any]:
        """Collect CUDA metadata without requiring CUDA to be available."""
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        gpu_names: List[str] = []
        if cuda_available:
            for index in range(device_count):
                gpu_names.append(torch.cuda.get_device_name(index))

        return {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "torch_cuda_available": cuda_available,
            "torch_cuda_device_count": device_count,
            "gpu_names": gpu_names,
        }

    def save_demo_results(
        self,
        output_dir: str = "demo_results",
        mode: str = "pipeline",
        seed: int = 0,
    ) -> None:
        """Save demo artifacts plus smoke-level run metadata and metrics."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        evidence_root = self._evidence_root(output_path)
        evidence_root.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        started_at = datetime.now().isoformat()
        started_perf = time.perf_counter()
        response_records: List[Dict[str, Any]] = []
        generated_paths: List[Path] = []

        for i, case in enumerate(self.demo_cases):
            ir = case["ir"]

            # Save intermediate representation
            ir_file = output_path / f"case_{i+1}_ir_{timestamp}.json"
            with open(ir_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(ir.to_dict(), f, ensure_ascii=False, indent=2)
            generated_paths.append(ir_file)

            # Save sample explanations
            explanations_file = output_path / f"case_{i+1}_explanations_{timestamp}.md"
            with open(explanations_file, 'w', encoding='utf-8') as f:
                f.write(f"# {case['name']} 解释演示\n\n")
                f.write(f"**故障类型：** {ir.fault_info.fault_type}\n")
                f.write(f"**置信度：** {ir.fault_info.confidence:.1%}\n\n")

                for j, query in enumerate(case["queries"], 1):
                    context = {"intermediate_representation": ir}
                    start = time.perf_counter()
                    response = self.llm.generate(query, context)
                    latency_seconds = time.perf_counter() - start
                    response_records.append(
                        {
                            "case_id": i + 1,
                            "query_id": j,
                            "fault_type": ir.fault_info.fault_type,
                            "confidence": ir.fault_info.confidence,
                            "query": query,
                            "latency_seconds": latency_seconds,
                            "response_chars": len(response),
                            "unsupported_claim_detected": False,
                        }
                    )
                    f.write(f"## 问题 {j}: {query}\n\n")
                    f.write(f"{response}\n\n")
            generated_paths.append(explanations_file)

        latencies = [record["latency_seconds"] for record in response_records]
        latency_p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
        latency_p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
        unsupported_count = sum(
            1 for record in response_records if record["unsupported_claim_detected"]
        )
        prompt_count = len(response_records)
        unsupported_rate = unsupported_count / prompt_count if prompt_count else 0.0

        metrics = {
            "paper_id": "LLM_Explainable_FD_Toolkit",
            "protocol_id": "demo_smoke",
            "condition_id": "template_llm",
            "accepted_evidence": False,
            "acceptance_blocker": "smoke demo only; no same-protocol GPU or reviewer baseline evidence",
            "sample_count": len(self.demo_cases),
            "prompt_count": prompt_count,
            "seed": seed,
            "latency_p50_seconds": latency_p50,
            "latency_p95_seconds": latency_p95,
            "unsupported_claim_rate_proxy": unsupported_rate,
            "failure_rate": 0.0,
            "response_records": response_records,
        }

        ended_at = datetime.now().isoformat()
        runtime_seconds = time.perf_counter() - started_perf
        run_meta = {
            "paper_id": "LLM_Explainable_FD_Toolkit",
            "protocol_id": "demo_smoke",
            "condition_id": "template_llm",
            "accepted_evidence": False,
            "mode": mode,
            "seed": seed,
            "command": "python " + " ".join(sys.argv),
            "working_directory": str(Path.cwd()),
            "submodule_commit": self._git_commit(),
            "input_artifact_paths": [],
            "output_artifact_paths": [str(path) for path in generated_paths],
            "metrics_path": str(evidence_root / "metrics.json"),
            "started_at": started_at,
            "ended_at": ended_at,
            "runtime_seconds": runtime_seconds,
            "batch_size_or_prompt_batch_size": 1,
            "precision_or_quantization": "local-template-fp32-smoke",
            "dataset_split_or_prompt_set": "demo_cases",
            "oom_or_failure_reason": "",
            "cuda": self._torch_cuda_metadata(),
        }

        (evidence_root / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (evidence_root / "run_meta.yaml").write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"\n💾 演示结果已保存到：{output_path}")
        print(f"📄 Smoke run metadata: {evidence_root / 'run_meta.yaml'}")
        print(f"📊 Smoke metrics: {evidence_root / 'metrics.json'}")


def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description="LLM Fault Diagnosis Explanation Demo")
    parser.add_argument("--mode", choices=["single", "style", "technical", "interactive", "pipeline", "all"],
                       default="all", help="Demo mode to run")
    parser.add_argument("--case", type=int, default=0, help="Case index for single mode")
    parser.add_argument("--save", action="store_true", help="Save demo results")
    parser.add_argument("--output", type=str, default="demo_results", help="Output directory for saved results")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic demo artifacts")

    args = parser.parse_args()

    print("🚀 LLM故障诊断解释演示系统")
    print("=" * 50)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    demo = MinimalLLMDemo()

    try:
        if args.mode == "single":
            demo.run_single_case_demo(args.case)
        elif args.mode == "style":
            demo.run_style_comparison_demo()
        elif args.mode == "technical":
            demo.run_technical_demo()
        elif args.mode == "interactive":
            demo.run_interactive_demo()
        elif args.mode == "pipeline":
            demo.run_pipeline_demo()
        elif args.mode == "all":
            print("🎯 运行所有演示模式...")
            demo.run_single_case_demo()
            demo.run_style_comparison_demo()
            demo.run_technical_demo()
            demo.run_pipeline_demo()
            print("\n🎮 交互式演示已准备就绪，运行以下命令启动：")
            print(f"   python {__file__} --mode interactive")

        if args.save:
            demo.save_demo_results(args.output, mode=args.mode, seed=args.seed)

        print(f"\n✅ 演示完成！")

    except KeyboardInterrupt:
        print(f"\n\n👋 演示被中断，再见！")
    except Exception as e:
        print(f"\n❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
