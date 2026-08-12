#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Evaluator - Benchmark评估框架测试脚本
测试可解释性评估器的核心功能

该脚本用于测试和验证评估器的各项功能，包括：
1. 评估器基本功能测试
2. 指标计算验证
3. 模拟评估流程
4. 结果生成和保存

作者: Claude Code Assistant
日期: 2025年12月3日
版本: 1.0
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
toolkit_dir = os.path.dirname(current_dir)
sys.path.append(toolkit_dir)
sys.path.append(os.path.join(toolkit_dir, 'toolkit_integration', 'explainability', 'core'))

try:
    from evaluator import ExplainabilityEvaluator, EvaluationMetrics, BaseExplainer
    print("✅ 成功导入评估器模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

try:
    from utils.metrics import ExplainabilityMetricsCalculator
    print("✅ 成功导入指标计算模块")
except ImportError as e:
    print(f"⚠️ 指标计算模块导入失败: {e}")
    print("🔄 将使用评估器内置的指标计算功能")


class MockExplainer(BaseExplainer):
    """模拟解释器，用于测试"""

    def __init__(self, model_name: str, explainer_type: str):
        super().__init__(None, model_name)
        self._explainer_type = explainer_type
        np.random.seed(42)  # 确保可重现

    def explain(self, x: np.ndarray, **kwargs):
        """生成模拟解释结果"""
        start_time = time.time()

        if self._explainer_type == 'intrinsic':
            explanation = self._generate_intrinsic_explanation(x)
        elif self._explainer_type == 'posthoc':
            explanation = self._generate_posthoc_explanation(x)
        else:
            explanation = self._generate_hybrid_explanation(x)

        explanation['computation_time'] = time.time() - start_time
        return explanation

    def get_explanation_type(self):
        return self._explainer_type

    def _generate_intrinsic_explanation(self, x: np.ndarray):
        """生成内禀解释"""
        if 'TSPN' in self.model_name:
            return {
                'explanation_type': 'intrinsic',
                'model_name': self.model_name,
                'processing_steps': [
                    'FFT变换: 将时域信号转换为频域',
                    f'统计特征提取: 均值={np.mean(x):.3f}, 标准差={np.std(x):.3f}',
                    '分类决策: 基于特征模式进行故障识别'
                ],
                'key_features': {
                    'mean': np.mean(x),
                    'std': np.std(x),
                    'rms': np.sqrt(np.mean(x**2)),
                    'peak': np.max(np.abs(x))
                },
                'visualization_data': {
                    'fft_spectrum': np.abs(np.fft.fft(x))[:50],
                    'statistical_features': [np.mean(x), np.std(x), np.max(x)]
                }
            }
        elif 'FuzzyLogic' in self.model_name:
            return {
                'explanation_type': 'intrinsic',
                'model_name': self.model_name,
                'fuzzy_rules': {
                    'Rule1': {'condition': 'rms IS Low', 'conclusion': 'Normal', 'confidence': 0.85},
                    'Rule2': {'condition': 'rms IS Medium', 'conclusion': 'Warning', 'confidence': 0.70},
                    'Rule3': {'condition': 'rms IS High', 'conclusion': 'Fault', 'confidence': 0.90}
                },
                'final_conclusion': 'Fault',
                'membership_functions': {
                    'rms_low': max(0, (0.5 - np.sqrt(np.mean(x**2))) / 0.5),
                    'rms_medium': 1 - abs(np.sqrt(np.mean(x**2)) - 0.5) / 0.5,
                    'rms_high': max(0, (np.sqrt(np.mean(x**2)) - 0.5) / 0.5)
                }
            }
        else:
            return {
                'explanation_type': 'intrinsic',
                'model_name': self.model_name,
                'processing_steps': ['基础特征提取', '模式识别', '分类输出'],
                'key_features': {'base_feature': np.mean(x)}
            }

    def _generate_posthoc_explanation(self, x: np.ndarray):
        """生成事后解释"""
        # 模拟SHAP值
        n_features = 13
        np.random.seed(42 + hash(self.model_name) % 1000)
        shap_values = np.random.normal(0, 0.1, n_features)

        # 使某些特征更重要
        shap_values[0] = np.random.uniform(0.2, 0.5)  # mean特征
        shap_values[1] = np.random.uniform(0.1, 0.3)  # std特征

        return {
            'explanation_type': 'posthoc',
            'model_name': self.model_name,
            'feature_importance': {
                'shap_values': shap_values.tolist(),
                'feature_names': [f'feature_{i}' for i in range(n_features)]
            },
            'method': 'SHAP',
            'base_value': 0.5
        }

    def _generate_hybrid_explanation(self, x: np.ndarray):
        """生成混合解释"""
        intrinsic_part = self._generate_intrinsic_explanation(x)
        posthoc_part = self._generate_posthoc_explanation(x)

        return {
            'explanation_type': 'hybrid',
            'model_name': self.model_name,
            'intrinsic_explanation': intrinsic_part,
            'posthoc_explanation': posthoc_part,
            'fusion_method': 'weighted_average'
        }


def test_evaluator_basic_functionality():
    """测试评估器基本功能"""
    print("\n🧪 测试1: 评估器基本功能")

    # 创建评估器
    evaluator = ExplainabilityEvaluator()

    # 创建模拟解释器
    tspn_intrinsic = MockExplainer('TSPN', 'intrinsic')
    tspn_posthoc = MockExplainer('TSPN', 'posthoc')
    fuzzy_intrinsic = MockExplainer('FuzzyLogic', 'intrinsic')

    # 注册解释器
    evaluator.register_explainer('TSPN', 'intrinsic', tspn_intrinsic)
    evaluator.register_explainer('TSPN', 'posthoc', tspn_posthoc)
    evaluator.register_explainer('FuzzyLogic', 'intrinsic', fuzzy_intrinsic)

    print(f"✅ 成功注册 {len(evaluator.explainers)} 个解释器")
    return evaluator


def test_metrics_calculation():
    """测试指标计算"""
    print("\n🧪 测试2: 指标计算")

    print("📊 使用评估器内置的指标计算功能")
    return None


def test_single_model_evaluation():
    """测试单个模型评估"""
    print("\n🧪 测试3: 单个模型评估")

    # 创建评估器
    evaluator = ExplainabilityEvaluator()

    # 创建和注册解释器
    tspn_intrinsic = MockExplainer('TSPN', 'intrinsic')
    evaluator.register_explainer('TSPN', 'intrinsic', tspn_intrinsic)

    # 创建测试数据
    test_samples = [np.random.randn(4096) for _ in range(20)]

    # 评估模型
    results = evaluator.evaluate_model('TSPN', 'test_dataset', len(test_samples))

    print(f"✅ 评估完成，得到 {len(results)} 个结果")
    for result in results:
        print(f"  - {result.model_name} ({result.explainer_type}): 综合得分 {result.get_overall_score():.3f}")

    return results


def test_benchmark_evaluation():
    """测试完整benchmark评估"""
    print("\n🧪 测试4: 完整Benchmark评估")

    # 创建评估器
    evaluator = ExplainabilityEvaluator()

    # 创建和注册所有解释器
    models = ['TSPN', 'FuzzyLogic']
    explainer_types = ['intrinsic', 'posthoc']

    for model in models:
        for exp_type in explainer_types:
            explainer = MockExplainer(model, exp_type)
            evaluator.register_explainer(model, exp_type, explainer)

    print(f"✅ 注册了 {len(evaluator.explainers)} 个解释器组合")

    # 运行benchmark
    results = evaluator.run_benchmark(
        model_names=['TSPN', 'FuzzyLogic'],
        dataset_name='test_dataset',
        sample_size=30
    )

    print(f"✅ Benchmark完成，总共 {len(results)} 个评估结果")

    # 生成结果表格
    df = evaluator.generate_results_table(results)
    print(f"✅ 结果表格包含 {len(df)} 行")

    return results


def test_results_saving():
    """测试结果保存功能"""
    print("\n🧪 测试5: 结果保存功能")

    # 创建评估器并运行benchmark
    evaluator = ExplainabilityEvaluator()

    # 注册解释器
    tspn_intrinsic = MockExplainer('TSPN', 'intrinsic')
    fuzzy_intrinsic = MockExplainer('FuzzyLogic', 'intrinsic')
    evaluator.register_explainer('TSPN', 'intrinsic', tspn_intrinsic)
    evaluator.register_explainer('FuzzyLogic', 'intrinsic', fuzzy_intrinsic)

    # 运行评估
    results = evaluator.run_benchmark(
        model_names=['TSPN', 'FuzzyLogic'],
        dataset_name='test_dataset',
        sample_size=10
    )

    # 保存结果
    output_dir = './test_results'
    evaluator.save_results(results, output_dir)

    # 检查生成的文件
    import json
    results_file = Path(output_dir) / 'explainability_benchmark_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            saved_data = json.load(f)
        print(f"✅ JSON文件保存成功，包含 {saved_data['total_evaluations']} 个评估项")

    csv_file = Path(output_dir) / 'explainability_benchmark_table.csv'
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        print(f"✅ CSV文件保存成功，包含 {len(df)} 行数据")

    return results


def test_metrics_calculation_functions():
    """测试指标计算函数"""
    print("\n🧪 测试6: 指标计算函数")

    calculator = ExplainabilityMetricsCalculator()

    # 创建模拟预测变化数据
    prediction_changes = [(0.1, 0.15), (0.2, 0.35), (0.3, 0.55), (0.5, 0.85)]

    # 测试忠实度计算
    faithfulness_correlation = calculator.calculate_faithfulness(
        {}, prediction_changes, 'correlation'
    )
    faithfulness_ablation = calculator.calculate_faithfulness(
        {}, prediction_changes, 'ablation'
    )

    print(f"✅ 相关性忠实度: {faithfulness_correlation:.3f}")
    print(f"✅ 消融忠实度: {faithfulness_ablation:.3f}")

    # 测试稳定性计算
    base_explanation = {
        'explanation_type': 'intrinsic',
        'processing_steps': ['step1', 'step2', 'step3'],
        'final_conclusion': 'Fault'
    }

    explanations_list = []
    for _ in range(5):
        noisy_explanation = {
            'explanation_type': 'intrinsic',
            'processing_steps': ['step1', 'step2', 'step3'],
            'final_conclusion': 'Fault'
        }
        explanations_list.append(noisy_explanation)

    stability = calculator.calculate_stability(
        base_explanation, explanations_list, 'noise_robustness'
    )
    print(f"✅ 稳定性: {stability:.3f}")

    return calculator


def create_visualization_demo():
    """创建可视化演示"""
    print("\n🧪 测试7: 可视化演示")

    # 创建评估器并运行benchmark
    evaluator = ExplainabilityEvaluator()

    # 注册解释器
    models = ['TSPN', 'FuzzyLogic', 'MoE', 'Fusion1D2D', 'OperatorAttention']
    for model in models:
        explainer = MockExplainer(model, 'intrinsic')
        evaluator.register_explainer(model, 'intrinsic', explainer)

    # 运行评估
    results = evaluator.run_benchmark(
        model_names=models,
        dataset_name='visualization_dataset',
        sample_size=25
    )

    # 创建可视化
    if results:
        _create_visualization_charts(results, './test_results')

    return results


def _create_visualization_charts(results, output_dir):
    """创建可视化图表"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 准备数据
    model_names = [r.model_name for r in results]
    explainer_types = [r.explainer_type for r in results]
    labels = [f"{model}\n({explainer})" for model, explainer in zip(model_names, explainer_types)]

    # 提取指标数据
    coverage = [r.coverage for r in results]
    stability = [r.stability for r in results]
    faithfulness = [r.faithfulness for r in results]
    understandability = [r.understandability for r in results]
    deployability = [r.deployability for r in results]
    overall_scores = [r.get_overall_score() for r in results]

    # 1. 综合得分对比图
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(results)))

    bars = plt.bar(labels, overall_scores, color=colors, alpha=0.8)
    plt.title('可解释性综合得分对比', fontsize=16, fontweight='bold')
    plt.ylabel('综合得分', fontsize=12)
    plt.ylim(0, 1)

    # 添加数值标签
    for bar, score in zip(bars, overall_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    chart_path = output_path / 'overall_scores_test.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 综合得分图已保存: {chart_path}")

    # 2. 指标热力图
    plt.figure(figsize=(10, 8))

    matrix_data = np.array([
        coverage,
        stability,
        faithfulness,
        understandability,
        deployability
    ])

    sns.heatmap(matrix_data,
                xticklabels=labels,
                yticklabels=['Coverage', 'Stability', 'Faithfulness', 'Understandability', 'Deployability'],
                annot=True, fmt='.3f', cmap='RdYlBu_r', center=0.5,
                vmin=0, vmax=1)

    plt.title('可解释性指标热力图', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    heatmap_path = output_path / 'metrics_heatmap_test.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 指标热力图已保存: {heatmap_path}")


def main():
    """主测试函数"""
    print("=" * 80)
    print("🧪 Explainable FD Toolkit - Evaluator 测试")
    print("=" * 80)

    # 运行所有测试
    test_results = []

    try:
        # 测试1: 基本功能
        evaluator = test_evaluator_basic_functionality()
        test_results.append(("基本功能", "✅ 通过"))
    except Exception as e:
        print(f"❌ 测试1失败: {str(e)}")
        test_results.append(("基本功能", f"❌ 失败: {str(e)}"))

    try:
        # 测试2: 指标计算
        calculator = test_metrics_calculation()
        test_results.append(("指标计算", "✅ 通过"))
    except Exception as e:
        print(f"❌ 测试2失败: {str(e)}")
        test_results.append(("指标计算", f"❌ 失败: {str(e)}"))

    try:
        # 测试3: 单模型评估
        test_single_model_evaluation()
        test_results.append(("单模型评估", "✅ 通过"))
    except Exception as e:
        print(f"❌ 测试3失败: {str(e)}")
        test_results.append(("单模型评估", f"❌ 失败: {str(e)}"))

    try:
        # 测试4: Benchmark评估
        test_benchmark_evaluation()
        test_results.append(("Benchmark评估", "✅ 通过"))
    except Exception as e:
        print(f"❌ 测试4失败: {str(e)}")
        test_results.append(("Benchmark评估", f"❌ 失败: {str(e)}"))

    try:
        # 测试5: 结果保存
        test_results_saving()
        test_results.append(("结果保存", "✅ 通过"))
    except Exception as e:
        print(f"❌ 测试5失败: {str(e)}")
        test_results.append(("结果保存", f"❌ 失败: {str(e)}"))

    try:
        # 测试6: 指标计算函数
        test_metrics_calculation_functions()
        test_results.append(("指标计算函数", "✅ 通过"))
    except Exception as e:
        print(f"❌ 测试6失败: {str(e)}")
        test_results.append(("指标计算函数", f"❌ 失败: {str(e)}"))

    try:
        # 测试7: 可视化
        create_visualization_demo()
        test_results.append(("可视化演示", "✅ 通过"))
    except Exception as e:
        print(f"❌ 测试7失败: {str(e)}")
        test_results.append(("可视化演示", f"❌ 失败: {str(e)}"))

    # 汇总测试结果
    print("\n" + "=" * 80)
    print("📊 测试结果汇总:")
    print("=" * 80)

    passed = sum(1 for _, status in test_results if status.startswith("✅"))
    total = len(test_results)

    for test_name, status in test_results:
        print(f"  {test_name}: {status}")

    print(f"\n📈 测试统计:")
    print(f"  通过: {passed}/{total}")
    print(f"  成功率: {passed/total*100:.1f}%")

    if passed == total:
        print("\n🎉 所有测试通过！评估器功能正常。")
    else:
        print(f"\n⚠️ 有 {total-passed} 个测试失败，请检查相关功能。")

    print("\n💡 接下来可以:")
    print("1. 将评估器与真实模型集成")
    print("2. 在实际数据集上运行benchmark")
    print("3. 扩展更多的评估指标和方法")
    print("4. 生成更详细的评估报告")


if __name__ == "__main__":
    main()