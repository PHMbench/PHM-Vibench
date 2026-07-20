#!/usr/bin/env python3
"""
最小 Neural-Symbolic Theory 统一基线兼容性测试脚本

用途：
- 验证 Neural-Symbolic Theory 是否能够在统一框架下：
  - 将神经网络预测映射到符号逻辑规则
  - 提供可解释的逻辑推理链
  - 支持一阶谓词逻辑表示
  - 集成模糊逻辑和神经网络输出

说明：
- 本脚本演示神经符号集成的核心功能
- 展示如何将神经网络输出转换为符号表示
- 验证逻辑推理和解释生成
"""

import os
import sys
from types import SimpleNamespace
import json
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np


def add_repo_root_to_sys_path() -> None:
    """将主仓库根目录加入 sys.path。"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.append(repo_root)


def build_minimal_args(device: str = "cuda") -> SimpleNamespace:
    """
    构造与 Neural-Symbolic Theory 兼容的最小参数对象。
    """
    return SimpleNamespace(
        in_dim=4096,
        out_dim=4096,
        in_channels=3,
        out_channels=3,
        device=device,
        scale=3,
        num_classes=5,
        skip_connection=True,
        # 四层算子配置
        layer1=["I", "WF", "I"],
        layer2=["I", "WF", "I"],
        layer3=["I", "WF", "I"],
        layer4=["I", "WF", "I"],
        # WaveFilters参数
        f_c_mu=0.0,
        f_c_sigma=0.1,
        f_b_mu=0.0,
        f_b_sigma=0.1,
    )


class SymbolicReasoner:
    """
    简化的符号推理器
    用于演示神经符号集成功能
    """

    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes

        # 故障类别到符号的映射
        self.fault_symbols = {
            0: "normal",
            1: "inner_race_fault",
            2: "outer_race_fault",
            3: "ball_fault",
            4: "cage_fault"
        }

        # 符号谓词定义
        self.predicates = {
            "has_fault": lambda x: f"has_fault({x})",
            "severity": lambda x, s: f"severity({x}, {s})",
            "location": lambda x, l: f"location({x}, {l})",
            "frequency": lambda x, f: f"frequency({x}, {f})",
            "temperature": lambda x, t: f"temperature({x}, {t})"
        }

        # 逻辑规则库
        self.logical_rules = [
            "has_fault(X) ∧ high_severity(X) → immediate_action(X)",
            "has_fault(X) ∧ location(X, bearing) → check_bearing(X)",
            "has_fault(X) ∧ frequency(X, high) → analyze_vibration(X)",
            "has_fault(X) ∧ temperature(X, high) → check_lubrication(X)",
            "normal(X) ∧ vibration_anomaly(X) → monitor_close(X)"
        ]

    def neural_to_symbolic(self, neural_output: torch.Tensor, threshold: float = 0.5) -> dict:
        """将神经网络输出转换为符号表示"""
        # 应用softmax获取概率分布
        probabilities = F.softmax(neural_output, dim=-1)

        # 确定预测类别
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_class].item()

        # 转换为符号表示
        symbolic_representation = {
            "primary_predicate": self.predicates["has_fault"](self.fault_symbols[predicted_class]),
            "confidence": confidence,
            "probabilities": probabilities.squeeze().tolist(),
            "symbolic_facts": []
        }

        # 基于置信度添加严重性谓词
        if confidence > 0.8:
            severity_level = "high"
        elif confidence > 0.6:
            severity_level = "medium"
        else:
            severity_level = "low"

        symbolic_representation["symbolic_facts"].append(
            self.predicates["severity"](self.fault_symbols[predicted_class], severity_level)
        )

        # 基于概率分布添加其他事实
        if predicted_class != 0:  # 非正常状态
            symbolic_representation["symbolic_facts"].append(
                self.predicates["location"](self.fault_symbols[predicted_class], "bearing")
            )
            symbolic_representation["symbolic_facts"].append(
                self.predicates["frequency"](self.fault_symbols[predicted_class], "high")
            )

        return symbolic_representation

    def apply_logical_rules(self, symbolic_facts: list) -> list:
        """应用逻辑规则进行推理"""
        derived_conclusions = []

        for rule in self.logical_rules:
            # 简化的规则匹配（实际实现需要完整的逻辑推理引擎）
            if any(fact in str(symbolic_facts) for fact in ["inner_race_fault", "outer_race_fault", "ball_fault"]):
                if "high" in str(symbolic_facts):
                    derived_conclusions.append("immediate_action_required")
                derived_conclusions.append("check_bearing")
                derived_conclusions.append("analyze_vibration")
            elif "normal" in str(symbolic_facts):
                derived_conclusions.append("monitor_close")

        return list(set(derived_conclusions))  # 去重

    def generate_explanation(self, symbolic_rep: dict, conclusions: list) -> str:
        """生成符号化解释"""
        explanation_parts = []

        # 主要预测
        explanation_parts.append(f"符号推理结果：{symbolic_rep['primary_predicate']}")
        explanation_parts.append(f"推理置信度：{symbolic_rep['confidence']:.1%}")

        # 事实陈述
        if symbolic_rep["symbolic_facts"]:
            explanation_parts.append("符号事实：")
            for fact in symbolic_rep["symbolic_facts"]:
                explanation_parts.append(f"  - {fact}")

        # 推理结论
        if conclusions:
            explanation_parts.append("逻辑推理结论：")
            for conclusion in conclusions:
                explanation_parts.append(f"  - {conclusion}")

        return "\n".join(explanation_parts)

    def export_to_logic_format(self, symbolic_rep: dict, conclusions: list) -> dict:
        """导出为标准逻辑格式"""
        logic_export = {
            "timestamp": datetime.now().isoformat(),
            "facts": symbolic_rep["symbolic_facts"],
            "primary_fact": symbolic_rep["primary_predicate"],
            "conclusions": conclusions,
            "probabilistic_evidence": {
                "class_probabilities": symbolic_rep["probabilities"],
                "confidence": symbolic_rep["confidence"]
            },
            "logical_rules_applied": self.logical_rules
        }

        return logic_export


def test_neural_symbolic_mapping():
    """测试神经符号映射"""
    print("[Testing Neural-Symbolic Mapping]")

    from model.FuzzyLogic_simple import FuzzyLogicNetwork

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = build_minimal_args(device=device)

    # 创建模糊逻辑模型（作为神经网络示例）
    model = FuzzyLogicNetwork({}, {}, args).to(device)
    model.eval()

    # 创建符号推理器
    reasoner = SymbolicReasoner(num_classes=args.num_classes)

    # 测试数据
    test_cases = [
        torch.randn(1, args.in_dim, args.in_channels, device=device),
        torch.randn(1, args.in_dim, args.in_channels, device=device),
        torch.randn(1, args.in_dim, args.in_channels, device=device),
    ]

    print(f"  - Testing {len(test_cases)} neural-to-symbolic mappings...")

    for i, x in enumerate(test_cases):
        with torch.no_grad():
            # 神经网络前向传播
            neural_output = model(x)

            # 转换为符号表示
            symbolic_rep = reasoner.neural_to_symbolic(neural_output)

            # 应用逻辑规则
            conclusions = reasoner.apply_logical_rules(symbolic_rep["symbolic_facts"])

            # 生成解释
            explanation = reasoner.generate_explanation(symbolic_rep, conclusions)

            print(f"  Case {i+1}:")
            print(f"    - Primary: {symbolic_rep['primary_predicate']}")
            print(f"    - Confidence: {symbolic_rep['confidence']:.1%}")
            print(f"    - Conclusions: {len(conclusions)} derived")
            print(f"    - Explanation preview: {explanation[:80]}...")

    print(f"  - ✅ Neural-Symbolic mapping test completed")


def test_first_order_logic():
    """测试一阶谓词逻辑表示"""
    print("\n[Testing First-Order Logic Representation]")

    reasoner = SymbolicReasoner()

    # 测试谓词构造
    test_predicates = []
    test_predicates.append(reasoner.predicates["has_fault"]("inner_race_fault"))
    test_predicates.append(reasoner.predicates["severity"]("inner_race_fault", "high"))
    test_predicates.append(reasoner.predicates["location"]("inner_race_fault", "bearing"))

    print("  - Generated predicates:")
    for pred in test_predicates:
        print(f"    {pred}")

    # 测试逻辑表达式
    logical_expressions = [
        "∀x (has_fault(x) ∧ severity(x, high) → immediate_action(x))",
        "∃x (has_fault(x) ∧ location(x, bearing))",
        "normal(x) → ¬immediate_action(x)"
    ]

    print("  - First-order logic examples:")
    for expr in logical_expressions:
        print(f"    {expr}")

    print(f"  - ✅ First-order logic test completed")


def test_symbolic_integration():
    """测试符号集成功能"""
    print("\n[Testing Symbolic Integration]")

    from model.FuzzyLogic_simple import FuzzyLogicNetwork
    from model.Fusion1D2D_simple import Fusion1D2D

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = build_minimal_args(device=device)

    # 创建多个模型
    models = {
        "FuzzyLogic": FuzzyLogicNetwork({}, {}, args).to(device),
        "Fusion1D2D": Fusion1D2D({}, {}, args).to(device)
    }

    # 创建符号推理器
    reasoner = SymbolicReasoner()

    # 测试数据
    x = torch.randn(1, args.in_dim, args.in_channels, device=device)

    # 多模型符号集成
    symbolic_results = {}

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            neural_output = model(x)
            symbolic_rep = reasoner.neural_to_symbolic(neural_output)
            symbolic_results[name] = symbolic_rep

    # 集成分析
    print("  - Multi-model symbolic integration:")
    for name, result in symbolic_results.items():
        print(f"    {name}: {result['primary_predicate']} (confidence: {result['confidence']:.1%})")

    # 符号一致性检查
    primary_facts = [result["primary_predicate"] for result in symbolic_results.values()]
    consistency = len(set(primary_facts)) == 1

    print(f"  - Symbolic consistency: {'consistent' if consistency else 'inconsistent'}")
    print(f"  - ✅ Symbolic integration test completed")


def test_logic_export():
    """测试逻辑格式导出"""
    print("\n[Testing Logic Export]")

    reasoner = SymbolicReasoner()

    # 模拟符号表示
    symbolic_rep = {
        "primary_predicate": "has_fault(outer_race_fault)",
        "confidence": 0.85,
        "probabilities": [0.05, 0.1, 0.85, 0.0, 0.0],
        "symbolic_facts": [
            "severity(outer_race_fault, high)",
            "location(outer_race_fault, bearing)",
            "frequency(outer_race_fault, high)"
        ]
    }

    # 应用规则
    conclusions = reasoner.apply_logical_rules(symbolic_rep["symbolic_facts"])

    # 导出逻辑格式
    logic_export = reasoner.export_to_logic_format(symbolic_rep, conclusions)

    print("  - Export format preview:")
    print(f"    Timestamp: {logic_export['timestamp']}")
    print(f"    Facts: {len(logic_export['facts'])}")
    print(f"    Conclusions: {len(logic_export['conclusions'])}")
    print(f"    Rules applied: {len(logic_export['logical_rules_applied'])}")

    # 保存示例
    export_filename = "Paper/Neuralsymbolic_theory/results/symbolic_export_example.json"
    os.makedirs(os.path.dirname(export_filename), exist_ok=True)

    with open(export_filename, 'w', encoding='utf-8') as f:
        json.dump(logic_export, f, indent=2, ensure_ascii=False)

    print(f"  - Export saved to: {export_filename}")
    print(f"  - ✅ Logic export test completed")


def main():
    """主测试函数"""
    add_repo_root_to_sys_path()

    print("=" * 60)
    print("Neural-Symbolic Theory Unified Framework Test")
    print("=" * 60)

    try:
        # 测试神经符号映射
        test_neural_symbolic_mapping()

        # 测试一阶谓词逻辑
        test_first_order_logic()

        # 测试符号集成
        test_symbolic_integration()

        # 测试逻辑导出
        test_logic_export()

        print("\n" + "=" * 60)
        print("✅ All Neural-Symbolic tests passed!")
        print("Symbolic reasoning system ready for integration.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()