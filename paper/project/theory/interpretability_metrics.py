"""
Interpretability Metrics for Explainable Fault Diagnosis
可解释性评估指标

本模块提供了多种可解释性评估指标，用于验证理论命题。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from sklearn.metrics import mutual_info_score
import math


class FidelityMetrics:
    """保真度指标

    衡量解释与模型预测的一致性。
    """

    @staticmethod
    def explanation_fidelity(model: nn.Module,
                            explain_func: Callable,
                            test_data: torch.Tensor,
                            target_class: Optional[int] = None) -> float:
        """
        计算解释的保真度

        Args:
            model: 待解释的模型
            explain_func: 解释生成函数
            test_data: 测试数据
            target_class: 目标类别（可选）

        Returns:
            fidelity_score: 保真度分数 (0-1)
        """
        model.eval()
        total_fidelity = 0.0
        num_samples = len(test_data)

        for x in test_data:
            # 原始预测
            with torch.no_grad():
                original_pred = model(x.unsqueeze(0))
                original_class = torch.argmax(original_pred, dim=1).item()

            # 生成解释并修改输入
            explanation = explain_func(model, x, target_class or original_class)

            # 基于解释修改输入（简化版）
            # 实际应用中需要根据解释类型进行更复杂的修改
            modified_x = x.clone()
            if 'important_features' in explanation:
                important_indices = explanation['important_features']
                # 移除重要特征测试保真度
                modified_x[important_indices] = 0

            # 修改后的预测
            with torch.no_grad():
                modified_pred = model(modified_x.unsqueeze(0))
                modified_class = torch.argmax(modified_pred, dim=1).item()

            # 保真度：预测是否改变
            if target_class is not None:
                # 对于目标类别的保真度
                fidelity = 1 - abs(modified_pred[0, target_class] - original_pred[0, target_class])
            else:
                # 整体预测保真度
                fidelity = 1.0 if original_class == modified_class else 0.0

            total_fidelity += fidelity

        return total_fidelity / num_samples

    @staticmethod
    def local_fidelity(model: nn.Module,
                      explanation: Dict,
                      x: torch.Tensor,
                      perturbation_size: float = 0.1) -> float:
        """
        计算局部保真度

        Args:
            model: 模型
            explanation: 解释字典
            x: 输入样本
            perturbation_size: 扰动大小

        Returns:
            local_fidelity: 局部保真度分数
        """
        model.eval()

        # 原始预测
        with torch.no_grad():
            original_output = model(x.unsqueeze(0))

        # 扰动输入
        perturbed_x = x + torch.randn_like(x) * perturbation_size

        # 扰动后的预测
        with torch.no_grad():
            perturbed_output = model(perturbed_x.unsqueeze(0))

        # 计算输出变化
        output_change = torch.norm(perturbed_output - original_output, p=2)

        # 计算解释预测的变化
        if 'feature_importance' in explanation:
            importance = torch.tensor(explanation['feature_importance'])
            # 加权变化
            weighted_change = output_change * torch.mean(importance)

            # 归一化保真度
            fidelity = torch.exp(-weighted_change).item()
        else:
            fidelity = torch.exp(-output_change).item()

        return fidelity


class ComprehensibilityMetrics:
    """可理解性指标

    衡量解释的复杂度和可理解性。
    """

    @staticmethod
    def rule_complexity(rules: List[str]) -> float:
        """
        计算规则集的复杂度

        Args:
            rules: 规则列表

        Returns:
            complexity: 复杂度分数（越低越简单）
        """
        total_complexity = 0.0

        for rule in rules:
            # 规则长度
            length_penalty = len(rule.split())

            # 逻辑操作符数量
            logical_ops = sum(rule.count(op) for op in ['AND', 'OR', 'NOT', 'IF', 'THEN'])

            # 条件数量
            conditions = rule.count('IF')

            # 综合复杂度
            rule_complexity = (length_penalty + 2 * logical_ops + 3 * conditions)
            total_complexity += rule_complexity

        # 归一化
        if len(rules) > 0:
            avg_complexity = total_complexity / len(rules)
            # 转换为可理解性分数（0-1，越高越简单）
            comprehensibility = 1.0 / (1.0 + math.log(avg_complexity + 1))
        else:
            comprehensibility = 0.0

        return comprehensibility

    @staticmethod
    def explanation_breadth(explanation: Dict) -> float:
        """
        计算解释的广度（覆盖的特征数量）

        Args:
            explanation: 解释字典

        Returns:
            breadth: 解释广度分数
        """
        if 'feature_importance' in explanation:
            # 非零特征的比例
            importance = np.array(explanation['feature_importance'])
            non_zero_features = np.sum(np.abs(importance) > 1e-3)
            total_features = len(importance)

            breadth = non_zero_features / total_features
        elif 'rules' in explanation:
            # 规则中涉及的特征数量
            all_features = set()
            for rule in explanation['rules']:
                features = [int(s.split('_')[1]) - 1
                           for s in rule.split() if s.startswith('feature')]
                all_features.update(features)

            breadth = len(all_features) / 10  # 假设总共10个特征
        else:
            breadth = 0.0

        return breadth

    @staticmethod
    def explanation_depth(explanation: Dict) -> int:
        """
        计算解释的深度（推理链长度）

        Args:
            explanation: 解释字典

        Returns:
            depth: 解释深度
        """
        if 'reasoning_chain' in explanation:
            return len(explanation['reasoning_chain'])
        elif 'rules' in explanation:
            # 最长规则的推理步数
            max_depth = 0
            for rule in explanation['rules']:
                depth = rule.count('AND') + rule.count('OR') + 1
                max_depth = max(max_depth, depth)
            return max_depth
        else:
            return 1


class TrustworthinessMetrics:
    """可信度指标

    衡量解释的稳定性和一致性。
    """

    @staticmethod
    def explanation_stability(model: nn.Module,
                            explain_func: Callable,
                            x: torch.Tensor,
                            num_perturbations: int = 10,
                            noise_level: float = 0.01) -> float:
        """
        计算解释的稳定性

        Args:
            model: 模型
            explain_func: 解释函数
            x: 输入样本
            num_perturbations: 扰动次数
            noise_level: 噪声水平

        Returns:
            stability: 稳定性分数
        """
        model.eval()

        # 原始解释
        original_explanation = explain_func(model, x)
        if 'feature_importance' not in original_explanation:
            return 0.0

        original_importance = np.array(original_explanation['feature_importance'])

        # 扰动解释
        perturbed_importances = []
        for _ in range(num_perturbations):
            # 添加噪声
            perturbed_x = x + torch.randn_like(x) * noise_level

            # 生成解释
            perturbed_explanation = explain_func(model, perturbed_x)
            if 'feature_importance' in perturbed_explanation:
                perturbed_importances.append(np.array(perturbed_explanation['feature_importance']))

        # 计算相似度
        if len(perturbed_importances) > 0:
            perturbed_importances = np.array(perturbed_importances)

            # 使用余弦相似度
            similarities = []
            for imp in perturbed_importances:
                similarity = np.dot(original_importance, imp) / (
                    np.linalg.norm(original_importance) * np.linalg.norm(imp) + 1e-8
                )
                similarities.append(similarity)

            stability = np.mean(similarities)
        else:
            stability = 0.0

        return stability

    @staticmethod
    def consistency_across_classes(model: nn.Module,
                                  explain_func: Callable,
                                  test_data: torch.Tensor,
                                  num_classes: int) -> float:
        """
        计算跨类别解释的一致性

        Args:
            model: 模型
            explain_func: 解释函数
            test_data: 测试数据
            num_classes: 类别数

        Returns:
            consistency: 一致性分数
        """
        model.eval()

        class_explanations = {i: [] for i in range(num_classes)}

        # 收集各类别的解释
        for x in test_data:
            with torch.no_grad():
                pred = model(x.unsqueeze(0))
                pred_class = torch.argmax(pred, dim=1).item()

            explanation = explain_func(model, x, pred_class)
            if 'feature_importance' in explanation:
                class_explanations[pred_class].append(explanation['feature_importance'])

        # 计算各类别内部的一致性
        consistencies = []
        for class_idx, explanations in class_explanations.items():
            if len(explanations) > 1:
                explanations = np.array(explanations)

                # 计算所有解释对的平均相似度
                similarities = []
                for i in range(len(explanations)):
                    for j in range(i + 1, len(explanations)):
                        sim = np.dot(explanations[i], explanations[j]) / (
                            np.linalg.norm(explanations[i]) * np.linalg.norm(explanations[j]) + 1e-8
                        )
                        similarities.append(sim)

                if similarities:
                    consistencies.append(np.mean(similarities))

        # 返回平均一致性
        return np.mean(consistencies) if consistencies else 0.0


class ComprehensiveInterpretabilityEvaluator:
    """综合可解释性评估器

    整合所有可解释性指标，提供综合评估。
    """

    def __init__(self,
                 fidelity_weight: float = 0.4,
                 comprehensibility_weight: float = 0.3,
                 trustworthiness_weight: float = 0.3):
        """
        Args:
            fidelity_weight: 保真度权重
            comprehensibility_weight: 可理解性权重
            trustworthiness_weight: 可信度权重
        """
        self.fidelity_weight = fidelity_weight
        self.comprehensibility_weight = comprehensibility_weight
        self.trustworthiness_weight = trustworthiness_weight

    def evaluate(self,
                 model: nn.Module,
                 explain_func: Callable,
                 test_data: torch.Tensor,
                 num_classes: int,
                 explanation_samples: Optional[Dict[int, List[Dict]]] = None) -> Dict[str, float]:
        """
        综合评估模型的可解释性

        Args:
            model: 待评估模型
            explain_func: 解释函数
            test_data: 测试数据
            num_classes: 类别数
            explanation_samples: 预生成的解释样本（可选）

        Returns:
            metrics: 各项指标分数
        """
        metrics = {}

        # 1. 保真度评估
        if explanation_samples is None:
            # 生成解释样本
            sample_size = min(50, len(test_data))
            sample_data = test_data[:sample_size]

            fidelity = FidelityMetrics.explanation_fidelity(
                model, explain_func, sample_data
            )
        else:
            # 使用预生成的样本
            fidelity = self._evaluate_fidelity_from_samples(
                model, explanation_samples
            )
        metrics['fidelity'] = fidelity

        # 2. 可理解性评估
        if explanation_samples:
            comprehensibility = self._evaluate_comprehensibility_from_samples(
                explanation_samples
            )
        else:
            # 生成样本进行评估
            sample_explanations = self._generate_sample_explanations(
                model, explain_func, test_data[:10]
            )
            comprehensibility = self._evaluate_comprehensibility_from_samples(
                sample_explanations
            )
        metrics['comprehensibility'] = comprehensibility

        # 3. 可信度评估
        sample_x = test_data[0]
        stability = TrustworthinessMetrics.explanation_stability(
            model, explain_func, sample_x
        )
        consistency = TrustworthinessMetrics.consistency_across_classes(
            model, explain_func, test_data[:30], num_classes
        )
        trustworthiness = (stability + consistency) / 2
        metrics['trustworthiness'] = trustworthiness

        # 4. 综合分数
        metrics['overall'] = (
            self.fidelity_weight * fidelity +
            self.comprehensibility_weight * comprehensibility +
            self.trustworthiness_weight * trustworthiness
        )

        return metrics

    def _generate_sample_explanations(self,
                                    model: nn.Module,
                                    explain_func: Callable,
                                    data: torch.Tensor) -> Dict[int, List[Dict]]:
        """生成样本解释"""
        explanations = {i: [] for i in range(10)}  # 假设10个类别

        model.eval()
        for x in data:
            with torch.no_grad():
                pred = model(x.unsqueeze(0))
                pred_class = torch.argmax(pred, dim=1).item()

            explanation = explain_func(model, x, pred_class)
            explanations[pred_class].append(explanation)

        return explanations

    def _evaluate_fidelity_from_samples(self,
                                      model: nn.Module,
                                      explanations: Dict[int, List[Dict]]) -> float:
        """从样本评估保真度"""
        total_fidelity = 0.0
        total_samples = 0

        for class_explanations in explanations.values():
            for explanation in class_explanations:
                if 'fidelity_score' in explanation:
                    total_fidelity += explanation['fidelity_score']
                    total_samples += 1

        return total_fidelity / total_samples if total_samples > 0 else 0.0

    def _evaluate_comprehensibility_from_samples(self,
                                                explanations: Dict[int, List[Dict]]) -> float:
        """从样本评估可理解性"""
        all_rules = []
        all_depths = []

        for class_explanations in explanations.values():
            for explanation in class_explanations:
                if 'rules' in explanation:
                    all_rules.extend(explanation['rules'])
                if 'depth' in explanation:
                    all_depths.append(explanation['depth'])

        # 计算规则复杂度
        rule_complexity = ComprehensibilityMetrics.rule_complexity(all_rules)

        # 计算平均深度
        avg_depth = np.mean(all_depths) if all_depths else 1.0
        depth_score = 1.0 / (1.0 + math.log(avg_depth + 1))

        # 综合可理解性
        comprehensibility = (rule_complexity + depth_score) / 2

        return comprehensibility


# 便捷函数
def evaluate_model_interpretability(model: nn.Module,
                                  explain_func: Callable,
                                  test_data: torch.Tensor,
                                  num_classes: int,
                                  weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    便捷函数：评估模型可解释性

    Args:
        model: 模型
        explain_func: 解释函数
        test_data: 测试数据
        num_classes: 类别数
        weights: 权重配置

    Returns:
        metrics: 评估结果
    """
    if weights is None:
        weights = {
            'fidelity': 0.4,
            'comprehensibility': 0.3,
            'trustworthiness': 0.3
        }

    evaluator = ComprehensiveInterpretabilityEvaluator(
        fidelity_weight=weights['fidelity'],
        comprehensibility_weight=weights['comprehensibility'],
        trustworthiness_weight=weights['trustworthiness']
    )

    return evaluator.evaluate(model, explain_func, test_data, num_classes)


# 测试函数
def test_metrics():
    """测试指标计算"""
    # 模拟数据
    test_data = torch.randn(100, 50)  # 100个样本，50维特征
    num_classes = 5

    # 模拟解释函数
    def dummy_explain_func(model, x, target_class=None):
        return {
            'feature_importance': np.random.rand(50),
            'fidelity_score': np.random.rand(),
            'depth': np.random.randint(1, 5),
            'rules': [f"IF feature_{i} > 0.5 THEN class == {np.random.randint(5)}"
                     for i in range(np.random.randint(1, 4))]
        }

    # 模拟模型
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(50, num_classes)

        def forward(self, x):
            return self.fc(x)

    model = DummyModel()

    # 评估
    metrics = evaluate_model_interpretability(
        model, dummy_explain_func, test_data, num_classes
    )

    print("可解释性评估结果:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    test_metrics()