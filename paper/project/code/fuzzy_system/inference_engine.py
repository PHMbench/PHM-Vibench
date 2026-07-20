"""
模糊推理引擎模块

实现模糊推理的核心算法，包括Mamdani推理、Sugeno推理和不同的解模糊化方法。

核心组件：
- FuzzyInferenceEngine: 模糊推理引擎抽象基类
- MamdaniInferenceEngine: Mamdani推理引擎实现
- DefuzzificationMethod: 解模糊化方法枚举和实现
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
import torch
from enum import Enum
from .membership_functions import FuzzyVariable
from .rule_base import RuleBase, FuzzyRule


class DefuzzificationMethod(Enum):
    """解模糊化方法枚举"""
    CENTROID = "centroid"          # 重心法
    BISECTOR = "bisector"         # 面积平分法
    MOM = "mom"                   # 平均最大隶属度法
    SOM = "som"                   # 最大隶属度最小值法
    LOM = "lom"                   # 最大隶属度最大值法
    WEIGHTED_AVERAGE = "weighted_avg"  # 加权平均法


class FuzzyInferenceEngine(ABC):
    """模糊推理引擎抽象基类"""

    def __init__(self, rule_base: RuleBase, fuzzy_variables: Dict[str, FuzzyVariable]):
        """
        初始化推理引擎

        Args:
            rule_base: 规则库
            fuzzy_variables: 模糊变量字典
        """
        self.rule_base = rule_base
        self.fuzzy_variables = fuzzy_variables
        self.last_inference_result = None

    @abstractmethod
    def fuzzy_inference(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        执行模糊推理

        Args:
            features: 输入特征字典

        Returns:
            推理结果字典 {conclusion: confidence}
        """
        pass

    def fuzzify_features(self, features: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        将数值特征模糊化

        Args:
            features: 数值特征字典

        Returns:
            模糊化后的特征 {feature_name: {fuzzy_set: membership_value}}
        """
        fuzzy_features = {}

        for feature_name, feature_value in features.items():
            if feature_name in self.fuzzy_variables:
                fuzzy_var = self.fuzzy_variables[feature_name]
                memberships = fuzzy_var.get_all_memberships(feature_value)
                fuzzy_features[feature_name] = memberships
            else:
                # 如果特征没有对应的模糊变量，跳过或给予默认值
                fuzzy_features[feature_name] = {}

        return fuzzy_features

    def get_explanation(self, features: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        获取推理过程的解释

        Args:
            features: 输入特征

        Returns:
            解释信息列表
        """
        fuzzy_features = self.fuzzify_features(features)
        active_rules = self.rule_base.get_active_rules(fuzzy_features)

        explanations = []
        for rule, activation in active_rules:
            if activation > 0.1:  # 只显示激活度较高的规则
                explanation = {
                    "rule_id": rule.rule_id,
                    "rule_description": rule.description,
                    "activation_strength": activation,
                    "conclusion": rule.conclusion,
                    "premises": [
                        {
                            "feature": premise.feature_name,
                            "fuzzy_set": premise.fuzzy_set,
                            "membership": fuzzy_features.get(premise.feature_name, {}).get(premise.fuzzy_set, 0.0)
                        } for premise in rule.premises
                    ]
                }
                explanations.append(explanation)

        return explanations


class MamdaniInferenceEngine(FuzzyInferenceEngine):
    """Mamdani模糊推理引擎"""

    def __init__(self, rule_base: RuleBase, fuzzy_variables: Dict[str, FuzzyVariable],
                 defuzzification_method: DefuzzificationMethod = DefuzzificationMethod.CENTROID,
                 universe_range: Tuple[float, float] = (0.0, 1.0),
                 resolution: int = 100):
        """
        初始化Mamdani推理引擎

        Args:
            rule_base: 规则库
            fuzzy_variables: 模糊变量字典
            defuzzification_method: 解模糊化方法
            universe_range: 论域范围
            resolution: 解离散化分辨率
        """
        super().__init__(rule_base, fuzzy_variables)
        self.defuzzification_method = defuzzification_method
        self.universe_range = universe_range
        self.resolution = resolution
        self.universe = np.linspace(universe_range[0], universe_range[1], resolution)

    def fuzzy_inference(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        执行Mamdani模糊推理

        Args:
            features: 输入特征字典

        Returns:
            推理结果字典 {conclusion: confidence}
        """
        # 第一步：模糊化输入特征
        fuzzy_features = self.fuzzify_features(features)

        # 第二步：规则评估和聚合
        conclusion_activations = self.rule_base.evaluate_all_rules(fuzzy_features)

        # 第三步：解模糊化（这里直接返回激活度，简化处理）
        result = {}
        for conclusion, activation in conclusion_activations.items():
            result[conclusion] = float(activation)

        self.last_inference_result = result
        return result

    def aggregate_output_fuzzy_sets(self, fuzzy_features: Dict[str, Dict[str, float]]) -> Dict[str, np.ndarray]:
        """
        聚合输出模糊集合（用于更复杂的解模糊化）

        Args:
            fuzzy_features: 模糊化特征

        Returns:
            聚合后的输出模糊集合 {conclusion: membership_array}
        """
        output_sets = {}
        conclusions = self.rule_base.get_all_conclusions()

        for conclusion in conclusions:
            # 初始化输出隶属度函数
            aggregated_membership = np.zeros(self.resolution)

            # 获取支持该结论的所有规则
            rules = self.rule_base.get_rules_for_conclusion(conclusion)

            for rule in rules:
                activation = rule.calculate_firing_strength(fuzzy_features)

                if activation > 0:
                    # 创建输出隶属度函数（这里简化为单一值）
                    # 在实际应用中，可能需要定义输出的模糊集合
                    output_membership = np.ones(self.resolution) * activation

                    # 聚合（取最大值）
                    aggregated_membership = np.maximum(aggregated_membership, output_membership)

            output_sets[conclusion] = aggregated_membership

        return output_sets

    def defuzzify(self, conclusion: str, membership_array: np.ndarray) -> float:
        """
        解模糊化

        Args:
            conclusion: 结论名称
            membership_array: 隶属度数组

        Returns:
            解模糊化后的精确值
        """
        if self.defuzzification_method == DefuzzificationMethod.CENTROID:
            return self._centroid_defuzzification(membership_array)
        elif self.defuzzification_method == DefuzzificationMethod.BISECTOR:
            return self._bisector_defuzzification(membership_array)
        elif self.defuzzification_method == DefuzzificationMethod.MOM:
            return self._mom_defuzzification(membership_array)
        elif self.defuzzification_method == DefuzzificationMethod.SOM:
            return self._som_defuzzification(membership_array)
        elif self.defuzzification_method == DefuzzificationMethod.LOM:
            return self._lom_defuzzification(membership_array)
        elif self.defuzzification_method == DefuzzificationMethod.WEIGHTED_AVERAGE:
            return self._weighted_average_defuzzification(conclusion, membership_array)
        else:
            raise ValueError(f"Unsupported defuzzification method: {self.defuzzification_method}")

    def _centroid_defuzzification(self, membership_array: np.ndarray) -> float:
        """重心法解模糊化"""
        if np.sum(membership_array) == 0:
            return 0.0

        weighted_sum = np.sum(self.universe * membership_array)
        total_membership = np.sum(membership_array)

        return float(weighted_sum / total_membership)

    def _bisector_defuzzification(self, membership_array: np.ndarray) -> float:
        """面积平分法解模糊化"""
        if np.sum(membership_array) == 0:
            return 0.0

        total_area = np.sum(membership_array)
        half_area = total_area / 2

        cumulative_area = 0.0
        for i, membership in enumerate(membership_array):
            cumulative_area += membership
            if cumulative_area >= half_area:
                return float(self.universe[i])

        return float(self.universe[-1])

    def _mom_defuzzification(self, membership_array: np.ndarray) -> float:
        """平均最大隶属度法解模糊化"""
        max_membership = np.max(membership_array)
        if max_membership == 0:
            return 0.0

        max_indices = np.where(membership_array == max_membership)[0]
        return float(np.mean(self.universe[max_indices]))

    def _som_defuzzification(self, membership_array: np.ndarray) -> float:
        """最大隶属度最小值法解模糊化"""
        max_membership = np.max(membership_array)
        if max_membership == 0:
            return 0.0

        max_indices = np.where(membership_array == max_membership)[0]
        return float(self.universe[max_indices[0]])

    def _lom_defuzzification(self, membership_array: np.ndarray) -> float:
        """最大隶属度最大值法解模糊化"""
        max_membership = np.max(membership_array)
        if max_membership == 0:
            return 0.0

        max_indices = np.where(membership_array == max_membership)[0]
        return float(self.universe[max_indices[-1]])

    def _weighted_average_defuzzification(self, conclusion: str, membership_array: np.ndarray) -> float:
        """加权平均法解模糊化"""
        # 这里可以基于规则权重或其他先验知识进行加权
        # 简化实现，使用重心法
        return self._centroid_defuzzification(membership_array)


class SugenoInferenceEngine(FuzzyInferenceEngine):
    """Sugeno（Takagi-Sugeno-Kang）模糊推理引擎"""

    def __init__(self, rule_base: RuleBase, fuzzy_variables: Dict[str, FuzzyVariable],
                 consequent_functions: Dict[str, Any]):
        """
        初始化Sugeno推理引擎

        Args:
            rule_base: 规则库
            fuzzy_variables: 模糊变量字典
            consequent_functions: 后件函数字典 {conclusion: function}
        """
        super().__init__(rule_base, fuzzy_variables)
        self.consequent_functions = consequent_functions

    def fuzzy_inference(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        执行Sugeno模糊推理

        Args:
            features: 输入特征字典

        Returns:
            推理结果字典 {conclusion: output_value}
        """
        fuzzy_features = self.fuzzify_features(features)

        # 计算每个规则的激活强度和后件输出
        weighted_outputs = {}
        total_weight = {}

        for rule in self.rule_base.rules.values():
            activation = rule.calculate_firing_strength(fuzzy_features)

            if activation > 0:
                conclusion = rule.conclusion

                # 计算后件函数输出
                if conclusion in self.consequent_functions:
                    consequent_func = self.consequent_functions[conclusion]
                    if callable(consequent_func):
                        consequent_output = consequent_func(features)
                    else:
                        # 如果是常数
                        consequent_output = float(consequent_func)
                else:
                    consequent_output = 0.0

                # 累积加权和
                if conclusion not in weighted_outputs:
                    weighted_outputs[conclusion] = 0.0
                    total_weight[conclusion] = 0.0

                weighted_outputs[conclusion] += activation * consequent_output
                total_weight[conclusion] += activation

        # 计算最终输出（加权平均）
        result = {}
        for conclusion in weighted_outputs:
            if total_weight[conclusion] > 0:
                result[conclusion] = weighted_outputs[conclusion] / total_weight[conclusion]
            else:
                result[conclusion] = 0.0

        self.last_inference_result = result
        return result


def create_default_fuzzy_variables() -> Dict[str, FuzzyVariable]:
    """
    创建默认的模糊变量（用于故障诊断）

    Returns:
        模糊变量字典
    """
    from .membership_functions import create_triangular_sets

    # 基于统计特征的模糊变量
    fuzzy_variables = {}

    # RMS特征
    fuzzy_variables["RMS"] = create_triangular_sets("RMS", (0.0, 5.0), 3)

    # 峭度特征
    fuzzy_variables["Kurtosis"] = create_triangular_sets("Kurtosis", (1.0, 10.0), 3)

    # 峰值因子特征
    fuzzy_variables["CrestFactor"] = create_triangular_sets("CrestFactor", (2.0, 8.0), 3)

    # 偏度特征
    fuzzy_variables["Skewness"] = create_triangular_sets("Skewness", (-2.0, 2.0), 3)

    # 形状因子特征
    fuzzy_variables["ShapeFactor"] = create_triangular_sets("ShapeFactor", (1.0, 2.0), 3)

    return fuzzy_variables


def create_inference_engine(rule_base: Optional[RuleBase] = None,
                          fuzzy_variables: Optional[Dict[str, FuzzyVariable]] = None,
                          engine_type: str = "mamdani") -> FuzzyInferenceEngine:
    """
    创建推理引擎的便捷函数

    Args:
        rule_base: 规则库（可选）
        fuzzy_variables: 模糊变量（可选）
        engine_type: 引擎类型（"mamdani" 或 "sugeno"）

    Returns:
        配置好的推理引擎
    """
    if rule_base is None:
        from .rule_base import create_fault_diagnosis_rules
        rule_base = create_fault_diagnosis_rules()

    if fuzzy_variables is None:
        fuzzy_variables = create_default_fuzzy_variables()

    if engine_type.lower() == "mamdani":
        return MamdaniInferenceEngine(rule_base, fuzzy_variables)
    elif engine_type.lower() == "sugeno":
        # Sugeno需要后件函数，这里提供一个简单的默认实现
        consequent_functions = {
            "IF": lambda f: f.get("RMS", 0) * 0.8 + f.get("Kurtosis", 0) * 0.2,
            "OF": lambda f: f.get("RMS", 0) * 0.6 + f.get("ShapeFactor", 0) * 0.4,
            "BF": lambda f: f.get("CrestFactor", 0) * 0.7 + f.get("RMS", 0) * 0.3,
            "HE": 0.1  # 健康状态的低输出
        }
        return SugenoInferenceEngine(rule_base, fuzzy_variables, consequent_functions)
    else:
        raise ValueError(f"Unsupported engine type: {engine_type}")


# 模糊推理系统的便捷接口类
class FuzzyInferenceSystem:
    """模糊推理系统高级接口"""

    def __init__(self, rule_base: Optional[RuleBase] = None,
                 fuzzy_variables: Optional[Dict[str, FuzzyVariable]] = None,
                 engine_type: str = "mamdani"):
        """
        初始化模糊推理系统

        Args:
            rule_base: 规则库
            fuzzy_variables: 模糊变量
            engine_type: 推理引擎类型
        """
        self.engine = create_inference_engine(rule_base, fuzzy_variables, engine_type)

    def diagnose(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        执行故障诊断

        Args:
            features: 输入特征

        Returns:
            诊断结果和解释
        """
        # 执行推理
        result = self.engine.fuzzy_inference(features)

        # 获取解释
        explanation = self.engine.get_explanation(features)

        # 确定最可能的诊断结果
        if result:
            best_diagnosis = max(result.items(), key=lambda x: x[1])
        else:
            best_diagnosis = ("Unknown", 0.0)

        return {
            "diagnosis_result": result,
            "primary_diagnosis": best_diagnosis[0],
            "confidence": best_diagnosis[1],
            "explanation": explanation,
            "fuzzy_features": self.engine.fuzzify_features(features)
        }

    def add_rule(self, rule: FuzzyRule) -> None:
        """添加新规则"""
        self.engine.rule_base.add_rule(rule)

    def get_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            "rule_base_stats": self.engine.rule_base.get_statistics(),
            "fuzzy_variables": list(self.engine.fuzzy_variables.keys()),
            "inference_engine_type": type(self.engine).__name__
        }