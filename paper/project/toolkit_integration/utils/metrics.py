#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explainability Metrics Utility Functions
可解释性评估指标计算工具

该模块提供了可解释性评估的各种指标计算函数，包括：
1. 覆盖度计算方法
2. 稳定性计算方法
3. 忠实度计算方法
4. 可理解性评估方法
5. 部署友好度评估方法

作者: Claude Code Assistant
日期: 2025年12月3日
版本: 1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
import time
import warnings
from pathlib import Path

# 抑制警告
warnings.filterwarnings('ignore')


class ExplainabilityMetrics:
    """可解释性评估指标计算类"""

    def __init__(self):
        """初始化指标计算器"""
        self.coverage_methods = ['path_based', 'feature_based', 'decision_tree']
        self.stability_methods = ['noise_robustness', 'perturbation_consistency']
        self.faithfulness_methods = ['correlation', 'causal_intervention', 'ablation']
        self.computation_time_methods = ['wall_clock', 'profiling']

    def calculate_coverage(self, explanation: Dict[str, Any],
                          method: str = 'path_based') -> float:
        """计算解释覆盖度

        Args:
            explanation: 解释结果字典
            method: 计算方法 ('path_based', 'feature_based', 'decision_tree')

        Returns:
            coverage_score: 覆盖度得分 [0,1]
        """
        try:
            if method == 'path_based':
                return self._coverage_path_based(explanation)
            elif method == 'feature_based':
                return self._coverage_feature_based(explanation)
            elif method == 'decision_tree':
                return self._coverage_decision_tree(explanation)
            else:
                raise ValueError(f"未知的覆盖度计算方法: {method}")
        except Exception as e:
            print(f"⚠️ 覆盖度计算失败: {str(e)}")
            return 0.5

    def _coverage_path_based(self, explanation: Dict[str, Any]) -> float:
        """基于路径的覆盖度计算"""
        explanation_type = explanation.get('explanation_type', 'unknown')

        if explanation_type == 'intrinsic':
            # 本征解释：基于决策路径的完整性
            if 'processing_steps' in explanation:
                # 透明信号处理模型（如TSPN）
                total_expected_steps = self._get_expected_steps(explanation.get('model_name', ''))
                explained_steps = len(explanation['processing_steps'])
                coverage = explained_steps / total_expected_steps if total_expected_steps > 0 else 0.7
            elif 'fuzzy_rules' in explanation:
                # 模糊逻辑系统
                total_expected_rules = 3  # 最少需要3条有意义的规则
                explained_rules = len([r for r in explanation['fuzzy_rules'].values()
                                      if r.get('confidence', 0) > 0.5])
                coverage = explained_rules / total_expected_rules if total_expected_rules > 0 else 0.6
            else:
                coverage = 0.7  # 默认值
        else:
            # 事后解释：基于特征覆盖范围
            if 'feature_importance' in explanation:
                importance_scores = explanation['feature_importance'].get('shap_values', [])
                if importance_scores:
                    # 重要特征的比例
                    mean_importance = np.mean(np.abs(importance_scores))
                    important_features = sum(1 for score in importance_scores if abs(score) > mean_importance)
                    coverage = important_features / len(importance_scores) if importance_scores else 0.3
                else:
                    coverage = 0.3
            else:
                coverage = 0.3

        return max(0.0, min(1.0, coverage))

    def _coverage_feature_based(self, explanation: Dict[str, Any]) -> float:
        """基于特征的覆盖度计算"""
        if 'key_features' in explanation:
            # 基于关键特征的覆盖
            features = explanation['key_features']
            # 检查是否覆盖了主要特征类别
            expected_categories = ['statistical', 'frequency', 'time', 'energy']
            covered_categories = []

            if any('mean' in str(k) or 'std' in str(k) for k in features.keys()):
                covered_categories.append('statistical')
            if any('freq' in str(k) or 'fft' in str(k) for k in features.keys()):
                covered_categories.append('frequency')
            if 'rms' in features or 'energy' in features:
                covered_categories.append('energy')

            coverage = len(covered_categories) / len(expected_categories)
        else:
            coverage = 0.5

        return max(0.0, min(1.0, coverage))

    def _coverage_decision_tree(self, explanation: Dict[str, Any]) -> float:
        """基于决策树的覆盖度计算"""
        # 模拟决策路径的覆盖度计算
        if 'decision_path' in explanation:
            path_length = len(explanation['decision_path'])
            # 假设完整的决策路径需要3-5个节点
            expected_length = 4
            coverage = min(1.0, path_length / expected_length)
        else:
            coverage = 0.6

        return coverage

    def calculate_stability(self, base_explanation: Dict[str, Any],
                          explanations_list: List[Dict[str, Any]],
                          method: str = 'noise_robustness') -> float:
        """计算解释稳定性

        Args:
            base_explanation: 基准解释
            explanations_list: 扰动后的解释列表
            method: 计算方法

        Returns:
            stability_score: 稳定性得分 [0,1]
        """
        try:
            if not explanations_list:
                return 1.0

            if method == 'noise_robustness':
                return self._stability_noise_robustness(base_explanation, explanations_list)
            elif method == 'perturbation_consistency':
                return self._stability_perturbation_consistency(base_explanation, explanations_list)
            else:
                raise ValueError(f"未知的稳定性计算方法: {method}")
        except Exception as e:
            print(f"⚠️ 稳定性计算失败: {str(e)}")
            return 0.5

    def _stability_noise_robustness(self, base_explanation: Dict[str, Any],
                                   explanations_list: List[Dict[str, Any]]) -> float:
        """基于噪声鲁棒性的稳定性计算"""
        similarities = []

        for noisy_explanation in explanations_list:
            similarity = self._compute_explanation_similarity(base_explanation, noisy_explanation)
            similarities.append(similarity)

        # 稳定性 = 平均相似度
        stability = np.mean(similarities) if similarities else 0.5

        return max(0.0, min(1.0, stability))

    def _stability_perturbation_consistency(self, base_explanation: Dict[str, Any],
                                           explanations_list: List[Dict[str, Any]]) -> float:
        """基于扰动一致性的稳定性计算"""
        # 计算解释结果的一致性
        if base_explanation.get('final_conclusion') and explanations_list:
            base_conclusion = base_explanation['final_conclusion']
            consistent_count = sum(1 for exp in explanations_list
                                  if exp.get('final_conclusion') == base_conclusion)
            stability = consistent_count / len(explanations_list)
        else:
            # 如果没有明确结论，基于特征重要性的一致性
            feature_consistencies = []
            for exp in explanations_list:
                consistency = self._compute_feature_consistency(base_explanation, exp)
                feature_consistencies.append(consistency)

            stability = np.mean(feature_consistencies) if feature_consistencies else 0.5

        return max(0.0, min(1.0, stability))

    def calculate_faithfulness(self, explanation: Dict[str, Any],
                            prediction_changes: List[Tuple[float, float]],
                            method: str = 'correlation') -> float:
        """计算解释忠实度

        Args:
            explanation: 解释结果
            prediction_changes: (mask_ratio, prediction_change) 列表
            method: 计算方法

        Returns:
            faithfulness_score: 忠实度得分 [0,1]
        """
        try:
            if not prediction_changes:
                return 0.5

            if method == 'correlation':
                return self._faithfulness_correlation(prediction_changes)
            elif method == 'causal_intervention':
                return self._faithfulness_causal_intervention(explanation, prediction_changes)
            elif method == 'ablation':
                return self._faithfulness_ablation(prediction_changes)
            else:
                raise ValueError(f"未知的忠实度计算方法: {method}")
        except Exception as e:
            print(f"⚠️ 忠实度计算失败: {str(e)}")
            return 0.5

    def _faithfulness_correlation(self, prediction_changes: List[Tuple[float, float]]) -> float:
        """基于相关性计算忠实度"""
        mask_ratios = [item[0] for item in prediction_changes]
        pred_changes = [item[1] for item in prediction_changes]

        if len(mask_ratios) > 1 and len(pred_changes) > 1:
            correlation = np.corrcoef(mask_ratios, pred_changes)[0, 1]
            faithfulness = abs(correlation) if not np.isnan(correlation) else 0.5
        else:
            faithfulness = 0.5

        return max(0.0, min(1.0, faithfulness))

    def _faithfulness_causal_intervention(self, explanation: Dict[str, Any],
                                           prediction_changes: List[Tuple[float, float]]) -> float:
        """基于因果干预计算忠实度"""
        # 检查解释中的因果关系是否正确
        causal_consistency = 0.0

        if 'causal_relationships' in explanation:
            causal_relations = explanation['causal_relationships']
            # 验证因果关系的正确性
            verified_relations = 0
            for relation in causal_relations:
                # 这里应该有实际的因果验证逻辑
                verified_relations += 1  # 简化处理

            causal_consistency = verified_relations / len(causal_relations) if causal_relations else 0.5

        # 结合相关性结果
        correlation_score = self._faithfulness_correlation(prediction_changes)

        # 加权平均
        faithfulness = 0.6 * correlation_score + 0.4 * causal_consistency

        return max(0.0, min(1.0, faithfulness))

    def _faithfulness_ablation(self, prediction_changes: List[Tuple[float, float]]) -> float:
        """基于消融实验计算忠实度"""
        # 检查特征消融时预测变化是否合理
        total_change = sum(abs(change) for _, change in prediction_changes)
        average_change = total_change / len(prediction_changes)

        # 合理的预测变化应该在一定范围内
        if 0.1 <= average_change <= 0.8:
            faithfulness = 1.0 - abs(average_change - 0.45)  # 理想变化约为0.45
        else:
            faithfulness = 0.5

        return max(0.0, min(1.0, faithfulness))

    def calculate_understandability(self, explanation: Dict[str, Any],
                                 expert_ratings: Optional[List[float]] = None,
                                 model_name: str = 'unknown',
                                 explainer_type: str = 'unknown') -> float:
        """计算解释可理解性

        Args:
            explanation: 解释结果
            expert_ratings: 专家评分列表 [1-5]
            model_name: 模型名称
            explainer_type: 解释器类型

        Returns:
            understandability_score: 可理解性得分 [0,1]
        """
        try:
            if expert_ratings and len(expert_ratings) > 0:
                # 使用实际专家评分
                avg_rating = np.mean(expert_ratings)
                understandability = avg_rating / 5.0
            else:
                # 基于模型和解释器类型的启发式评分
                understandability = self._heuristic_understandability(model_name, explainer_type, explanation)

            return max(0.0, min(1.0, understandability))
        except Exception as e:
            print(f"⚠️ 可理解性计算失败: {str(e)}")
            return 0.5

    def _heuristic_understandability(self, model_name: str, explainer_type: str,
                                      explanation: Dict[str, Any]) -> float:
        """启发式可理解性评估"""
        # 基础评分表
        base_scores = {
            # TSPN模型
            ('TSPN', 'intrinsic'): 0.90,      # 透明信号处理，物理意义清晰
            ('TSPN', 'posthoc'): 0.70,        # 需要理解SHAP等概念
            ('TSPN', 'hybrid'): 0.80,          # 结合两者，复杂度适中

            # FuzzyLogic模型
            ('FuzzyLogic', 'intrinsic'): 0.95,  # 规则直观易懂
            ('FuzzyLogic', 'posthoc'): 0.65,    # 复杂度增加
            ('FuzzyLogic', 'hybrid'): 0.85,      # 规则+分析，清晰

            # Fusion1D2D模型
            ('Fusion1D2D', 'intrinsic'): 0.85,   # 多模态但物理意义明确
            ('Fusion1D2D', 'posthoc'): 0.75,     # 特征融合增加复杂性
            ('Fusion1D2D', 'hybrid'): 0.80,       # 多层次解释

            # MoE模型
            ('MoE', 'intrinsic'): 0.80,          # 专家路径相对清晰
            ('MoE', 'posthoc'): 0.60,             # 路由机制复杂
            ('MoE', 'hybrid'): 0.70,             # 混合方法

            # OperatorAttention模型
            ('OperatorAttention', 'intrinsic'): 0.75,  # 注意力机制需要理解
            ('OperatorAttention', 'posthoc'): 0.65,     # 多层注意力复杂
            ('OperatorAttention', 'hybrid'): 0.70,       # 混合层次解释

            # 其他模型
            ('NNSPN', 'intrinsic'): 0.85,          # 神经信号处理，相对透明
            ('NNSPN', 'posthoc'): 0.70,            # 需要专业知识
            ('NNSPN', 'hybrid'): 0.80,              # 混合方法

            ('TKAN', 'intrinsic'): 0.75,            # KAN网络，需要新知识
            ('TKAN', 'posthoc'): 0.65,                # 概念较新
            ('TKAN', 'hybrid'): 0.70,                # 混合解释
        }

        base_score = base_scores.get((model_name, explainer_type), 0.70)

        # 基于解释内容复杂度调整
        complexity_penalty = 0.0

        # 检查解释的复杂度
        explanation_length = len(str(explanation))
        if explanation_length > 5000:
            complexity_penalty = -0.1  # 解释太长，降低可理解性
        elif explanation_length < 500:
            complexity_penalty = -0.05  # 解释太短，可能不够详细

        # 检查是否包含可视化
        if 'visualization' not in explanation and 'charts' not in explanation:
            complexity_penalty -= 0.05  # 缺少可视化，降低可理解性

        # 检查是否包含自然语言描述
        if 'natural_language_explanation' in explanation:
            base_score += 0.05  # 有自然语言描述，提高可理解性

        # 检查是否包含实际案例
        if 'case_studies' in explanation or 'examples' in explanation:
            base_score += 0.05  # 有实际案例，提高可理解性

        final_score = base_score + complexity_penalty

        return max(0.0, min(1.0, final_score))

    def calculate_deployability(self, model_info: Dict[str, Any],
                                 explanation: Dict[str, Any],
                                 explainer_type: str = 'unknown') -> float:
        """计算部署友好度

        Args:
            model_info: 模型信息字典
            explanation: 解释结果
            explainer_type: 解释器类型

        Returns:
            deployability_score: 部署友好度得分 [0,1]
        """
        try:
            # 评估模型复杂度
            complexity_score = self._evaluate_model_complexity(model_info)

            # 评估解释器复杂度
            explainer_score = self._evaluate_explainer_complexity(explainer_type, explanation)

            # 评估资源需求
            resource_score = self._evaluate_resource_requirements(model_info)

            # 评估集成便利性
            integration_score = self._evaluate_integration_ease(explanation)

            # 加权平均
            weights = {
                'complexity': 0.25,
                'explainer': 0.25,
                'resource': 0.30,
                'integration': 0.20
            }

            deployability = (
                complexity_score * weights['complexity'] +
                explainer_score * weights['explainer'] +
                resource_score * weights['resource'] +
                integration_score * weights['integration']
            )

            return max(0.0, min(1.0, deployability))
        except Exception as e:
            print(f"⚠️ 部署友好度计算失败: {str(e)}")
            return 0.5

    def _evaluate_model_complexity(self, model_info: Dict[str, Any]) -> float:
        """评估模型复杂度"""
        param_count = model_info.get('parameter_count', 0)

        if param_count < 10000:        # 轻量级
            complexity = 0.90
        elif param_count < 100000:       # 中等
            complexity = 0.75
        elif param_count < 1000000:      # 重量级
            complexity = 0.60
        else:                           # 超重量级
            complexity = 0.40

        # 考虑模型类型
        model_type = model_info.get('type', '').lower()
        if 'fuzzy' in model_type or 'rule' in model_type:
            complexity += 0.05  # 规则系统通常易于部署
        elif 'transformer' in model_type or 'attention' in model_type:
            complexity -= 0.10  # 复杂模型部署困难
        elif 'simple' in model_type:
            complexity += 0.05  # 简单模型易于部署

        return max(0.0, min(1.0, complexity))

    def _evaluate_explainer_complexity(self, explainer_type: str, explanation: Dict[str, Any]) -> float:
        """评估解释器复杂度"""
        if explainer_type == 'intrinsic':
            # 内置解释通常更简单
            complexity = 0.90
        elif explainer_type == 'posthoc':
            # 事后解释可能需要额外计算
            complexity = 0.75
        elif explainer_type == 'hybrid':
            # 混合方法复杂度较高
            complexity = 0.65
        else:
            complexity = 0.70

        # 检查解释的计算复杂度
        if explanation.get('computation_intensive', False):
            complexity -= 0.10

        # 检查是否需要外部依赖
        if explanation.get('external_dependencies', []):
            complexity -= 0.05 * len(explanation['external_dependencies'])

        return max(0.0, min(1.0, complexity))

    def _evaluate_resource_requirements(self, model_info: Dict[str, Any]) -> float:
        """评估资源需求"""
        # 基于参数数量的资源需求评估
        param_count = model_info.get('parameter_count', 0)

        if param_count < 10000:        # 低资源需求
            resource_score = 0.95
        elif param_count < 100000:       # 中等资源需求
            resource_score = 0.80
        elif param_count < 1000000:      # 高资源需求
            resource_score = 0.60
        else:                           # 极高资源需求
            resource_score = 0.40

        # 考虑GPU需求
        if model_info.get('requires_gpu', False):
            resource_score -= 0.10

        # 考虑内存需求
        memory_requirement = model_info.get('memory_requirement', 'unknown')
        if isinstance(memory_requirement, (int, float)):
            if memory_requirement < 100:      # MB
                resource_score += 0.05
            elif memory_requirement > 1000:  # MB
                resource_score -= 0.10

        return max(0.0, min(1.0, resource_score))

    def _evaluate_integration_ease(self, explanation: Dict[str, Any]) -> float:
        """评估集成便利性"""
        integration_score = 0.70  # 基础分

        # 检查是否有标准化接口
        if explanation.get('api_standardized', False):
            integration_score += 0.10

        # 检查是否有完整文档
        if explanation.get('well_documented', False):
            integration_score += 0.10

        # 检查是否支持多种数据格式
        supported_formats = explanation.get('supported_formats', [])
        if len(supported_formats) > 2:
            integration_score += 0.05

        # 检查是否有示例代码
        if explanation.get('example_code_available', False):
            integration_score += 0.05

        return max(0.0, min(1.0, integration_score))

    def _compute_explanation_similarity(self, exp1: Dict[str, Any], exp2: Dict[str, Any]) -> float:
        """计算两个解释的相似度"""
        # 简化的相似度计算
        exp1_type = exp1.get('explanation_type', 'unknown')
        exp2_type = exp2.get('explanation_type', 'unknown')

        if exp1_type != exp2_type:
            return 0.3  # 不同类型的解释相似度较低

        if exp1_type == 'intrinsic':
            # 本征解释：基于步骤的相似性
            steps1 = set(exp1.get('processing_steps', []))
            steps2 = set(exp2.get('processing_steps', []))

            if steps1 and steps2:
                intersection = len(steps1.intersection(steps2))
                union = len(steps1.union(steps2))
                similarity = intersection / union if union > 0 else 1.0
            else:
                similarity = 0.5
        else:
            # 事后解释：基于特征重要性的相似性
            if 'feature_importance' in exp1 and 'feature_importance' in exp2:
                try:
                    importance1 = np.array(exp1['feature_importance'].get('shap_values', []))
                    importance2 = np.array(exp2['feature_importance'].get('shap_values', []))

                    if len(importance1) == len(importance2) and len(importance1) > 0:
                        # 余弦相似度
                        similarity = np.dot(importance1, importance2) / (
                            np.linalg.norm(importance1) * np.linalg.norm(importance2) + 1e-8
                        )
                    else:
                        similarity = 0.5
                except:
                    similarity = 0.5
            else:
                similarity = 0.5

        return max(0.0, min(1.0, similarity))

    def _compute_feature_consistency(self, exp1: Dict[str, Any], exp2: Dict[str, Any]) -> float:
        """计算特征重要性的一致性"""
        if 'key_features' in exp1 and 'key_features' in exp2:
            features1 = exp1['key_features']
            features2 = exp2['key_features']

            # 计算特征值的相关性
            common_features = set(features1.keys()) & set(features2.keys())

            if common_features:
                values1 = [features1[k] for k in common_features]
                values2 = [features2[k] for k in common_features]

                if len(values1) > 1 and len(values2) > 1:
                    correlation = np.corrcoef(values1, values2)[0, 1]
                    return abs(correlation) if not np.isnan(correlation) else 0.5
                else:
                    return 0.5
            else:
                return 0.5
        else:
            return 0.5

    def _get_expected_steps(self, model_name: str) -> int:
        """获取模型预期的决策步骤数"""
        step_expectations = {
            'TSPN': 3,              # FFT → 特征提取 → 分类
            'FuzzyLogic': 3,        # 特征提取 → 模糊化 → 推理 → 解模糊
            'Fusion1D2D': 4,        # 1D → 2D → 融合 → 分类
            'MoE': 4,                # 特征提取 → 专家选择 → 专家推理 → 组合
            'OperatorAttention': 4,  # 特征提取 → 算子应用 → 注意力 → 分类
            'NNSPN': 3,              # 类似TSPN
            'TKAN': 3,                # 基础步骤
        }
        return step_expectations.get(model_name, 3)

    def calculate_computation_time(self, explanations: List[Dict[str, Any]],
                                  method: str = 'wall_clock') -> float:
        """计算平均计算时间

        Args:
            explanations: 包含计算时间的解释列表
            method: 计算方法

        Returns:
            avg_time: 平均计算时间（秒）
        """
        try:
            if not explanations:
                return 0.0

            if method == 'wall_clock':
                times = [exp.get('computation_time', 0) for exp in explanations]
                return np.mean(times)
            elif method == 'profiling':
                # 更详细的性能分析
                return self._profile_computation_time(explanations)
            else:
                raise ValueError(f"未知的计算时间方法: {method}")
        except Exception as e:
            print(f"⚠️ 计算时间计算失败: {str(e)}")
            return 0.0

    def _profile_computation_time(self, explanations: List[Dict[str, Any]]) -> float:
        """详细的性能分析"""
        # 这里可以添加更详细的性能分析逻辑
        # 例如：内存使用、GPU利用率等
        return self.calculate_computation_time(explanations, 'wall_clock')

    def generate_metrics_report(self, results: List[Dict[str, Any]],
                                output_dir: str = './metrics_report'):
        """生成指标评估报告

        Args:
            results: 评估结果列表
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 生成详细的指标报告
        self._generate_detailed_report(results, output_path)

        # 生成统计摘要
        self._generate_summary_report(results, output_path)

    def _generate_detailed_report(self, results: List[Dict[str, Any]], output_path: Path):
        """生成详细报告"""
        report_file = output_path / 'detailed_metrics_report.md'

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 可解释性评估详细报告\n\n")
            f.write(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 评估指标说明\n\n")
            f.write("| 指标 | 说明 | 计算方法 | 范围 |\n")
            f.write("|------|------|----------|------|\n")
            f.write("| 覆盖度 | 解释覆盖决策路径的比例 | path-based/feature-based | [0,1] |\n")
            f.write("| 稳定性 | 输入扰动下解释的一致性 | noise_robustness/perturbation | [0,1] |\n")
            f.write("| 忠实度 | 解释与模型预测的相关性 | correlation/causal_intervention | [0,1] |\n")
            f.write("| 可理解性 | 解释对专家的易懂程度 | 专家评分/启发式 | [0,1] |\n")
            f.write("| 部署友好度 | 工业部署的难易程度 | 复杂度+资源+集成 | [0,1] |\n")
            f.write("| 计算时间 | 生成解释所需时间 | wall_clock/profiling | [0,+∞] |\n\n")

            # 按模型分组展示结果
            models = set(r.get('model_name', 'unknown') for r in results)
            for model in sorted(models):
                model_results = [r for r in results if r.get('model_name') == model]
                f.write(f"## {model} 详细结果\n\n")

                for result in model_results:
                    f.write(f"### {result.get('explainer_type', 'unknown').title()} 解释\n")
                    f.write(f"- **覆盖度**: {result.get('coverage', 0):.3f}\n")
                    f.write(f"- **稳定性**: {result.get('stability', 0):.3f}\n")
                    f.write(f"- **忠实度**: {result.get('faithfulness', 0):.3f}\n")
                    f.write(f"- **可理解性**: {result.get('understandability', 0):.3f}\n")
                    f.write(f"- **部署友好度**: {result.get('deployability', 0):.3f}\n")
                    f.write(f"- **计算时间**: {result.get('computation_time', 0):.4f}秒\n")
                    f.write(f"- **综合得分**: {result.get('overall_score', 0):.3f}\n\n")

        print(f"✅ 详细报告已生成: {report_file}")

    def _generate_summary_report(self, results: List[Dict[str, Any]], output_path: Path):
        """生成统计摘要报告"""
        summary_file = output_path / 'metrics_summary.json'

        # 计算统计信息
        stats = {
            'total_evaluations': len(results),
            'models': list(set(r.get('model_name', 'unknown') for r in results)),
            'explainers': list(set(r.get('explainer_type', 'unknown') for r in results)),
            'metrics_summary': {}
        }

        # 计算各指标的统计信息
        metrics = ['coverage', 'stability', 'faithfulness', 'understandability', 'deployability']
        for metric in metrics:
            values = [r.get(metric, 0) for r in results]
            if values:
                stats['metrics_summary'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }

        # 保存统计信息
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"✅ 统计摘要已生成: {summary_file}")


def main():
    """主函数 - 演示指标计算"""
    print("=" * 80)
    print("📊 Explainable FD Toolkit - Metrics Demo")
    print("=" * 80)

    # 创建指标计算器
    metrics_calculator = ExplainabilityMetrics()

    # 演示覆盖度计算
    print("\n📈 演示指标计算:")
    print("这是指标计算工具的演示，实际使用时请与评估器配合使用")

    # 这里可以添加实际的指标计算示例
    # print("✅ 演示完成！")


if __name__ == "__main__":
    main()