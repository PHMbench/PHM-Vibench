"""
1阶谓词逻辑框架模块

实现基于1阶谓词逻辑的模糊谓词系统，为故障诊断提供形式化的逻辑推理基础。

核心概念：
- FirstOrderPredicate: 1阶谓词的抽象基类
- FuzzyPredicate: 模糊谓词，将连续特征映射到[0,1]的真值
- LogicalConnective: 逻辑连接词（AND, OR, NOT等）
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union, Optional
import numpy as np
import torch
from enum import Enum


class LogicalConnective(Enum):
    """逻辑连接词枚举"""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    IFF = "iff"


class FirstOrderPredicate(ABC):
    """1阶谓词抽象基类"""

    def __init__(self, name: str, arity: int = 1):
        """
        初始化1阶谓词

        Args:
            name: 谓词名称
            arity: 谓词的元数（参数个数）
        """
        self.name = name
        self.arity = arity

    @abstractmethod
    def evaluate(self, *args: Any) -> float:
        """
        评估谓词的真值

        Args:
            *args: 谓词的参数

        Returns:
            谓词的真值（[0,1]范围，对于模糊谓词）
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """获取谓词的自然语言描述"""
        pass

    def __call__(self, *args: Any) -> float:
        return self.evaluate(*args)


class FuzzyPredicate(FirstOrderPredicate):
    """模糊谓词类"""

    def __init__(self, name: str, feature_extractor, fuzzy_set, description: str = ""):
        """
        初始化模糊谓词

        Args:
            name: 谓词名称
            feature_extractor: 特征提取函数或特征名
            fuzzy_set: 模糊集合，用于计算隶属度
            description: 自然语言描述
        """
        super().__init__(name, arity=1)
        self.feature_extractor = feature_extractor
        self.fuzzy_set = fuzzy_set
        self.description = description or f"{name} holds with fuzzy membership"

    def evaluate(self, features: Union[Dict[str, float], np.ndarray, torch.Tensor]) -> float:
        """
        评估模糊谓词

        Args:
            features: 特征值字典或数组

        Returns:
            模糊真值（[0,1]）
        """
        if isinstance(features, dict):
            if isinstance(self.feature_extractor, str):
                # 如果feature_extractor是特征名称字符串
                feature_value = features.get(self.feature_extractor, 0.0)
            else:
                # 如果feature_extractor是函数
                feature_value = self.feature_extractor(features)
        else:
            # 如果features是数值或数组
            feature_value = features

        return float(self.fuzzy_set(feature_value))

    def get_description(self) -> str:
        return self.description


class ComparisonPredicate(FirstOrderPredicate):
    """比较谓词（大于、小于、等于等）"""

    def __init__(self, name: str, feature_name: str, threshold: float,
                 comparison_type: str = ">", fuzzy_margin: float = 0.0):
        """
        初始化比较谓词

        Args:
            name: 谓词名称
            feature_name: 特征名称
            threshold: 阈值
            comparison_type: 比较类型（">", "<", "=", ">=", "<="）
            fuzzy_margin: 模糊边界宽度（0表示精确比较）
        """
        super().__init__(name, arity=1)
        self.feature_name = feature_name
        self.threshold = threshold
        self.comparison_type = comparison_type
        self.fuzzy_margin = max(fuzzy_margin, 0.0)

    def evaluate(self, features: Union[Dict[str, float], np.ndarray, torch.Tensor]) -> float:
        """
        评估比较谓词

        Args:
            features: 特征字典或数组

        Returns:
            比较的真值（[0,1]）
        """
        if isinstance(features, dict):
            feature_value = features.get(self.feature_name, 0.0)
        else:
            feature_value = float(features)

        diff = feature_value - self.threshold

        if self.fuzzy_margin == 0.0:
            # 精确比较
            if self.comparison_type == ">":
                return 1.0 if diff > 0 else 0.0
            elif self.comparison_type == "<":
                return 1.0 if diff < 0 else 0.0
            elif self.comparison_type == ">=":
                return 1.0 if diff >= 0 else 0.0
            elif self.comparison_type == "<=":
                return 1.0 if diff <= 0 else 0.0
            elif self.comparison_type == "=":
                return 1.0 if abs(diff) < 1e-6 else 0.0
        else:
            # 模糊比较
            if self.comparison_type == ">":
                return min(1.0, max(0.0, diff / self.fuzzy_margin + 0.5))
            elif self.comparison_type == "<":
                return min(1.0, max(0.0, -diff / self.fuzzy_margin + 0.5))
            elif self.comparison_type == ">=":
                return min(1.0, max(0.0, diff / self.fuzzy_margin + 0.5))
            elif self.comparison_type == "<=":
                return min(1.0, max(0.0, -diff / self.fuzzy_margin + 0.5))
            elif self.comparison_type == "=":
                return min(1.0, max(0.0, 1.0 - abs(diff) / self.fuzzy_margin))

        return 0.0

    def get_description(self) -> str:
        return f"{self.feature_name} {self.comparison_type} {self.threshold}"


class CompoundPredicate(FirstOrderPredicate):
    """复合谓词（通过逻辑连接词组合多个谓词）"""

    def __init__(self, name: str, predicates: List[FirstOrderPredicate],
                 connective: LogicalConnective):
        """
        初始化复合谓词

        Args:
            name: 谓词名称
            predicates: 子谓词列表
            connective: 逻辑连接词
        """
        super().__init__(name, arity=max(p.arity for p in predicates))
        self.predicates = predicates
        self.connective = connective

    def evaluate(self, *args: Any) -> float:
        """
        评估复合谓词

        Args:
            *args: 谓词参数

        Returns:
            复合谓词的真值
        """
        # 计算所有子谓词的真值
        truth_values = [pred.evaluate(*args) for pred in self.predicates]

        # 根据逻辑连接词计算最终真值
        if self.connective == LogicalConnective.AND:
            return min(truth_values)
        elif self.connective == LogicalConnective.OR:
            return max(truth_values)
        elif self.connective == LogicalConnective.NOT:
            if len(truth_values) != 1:
                raise ValueError("NOT connective requires exactly one predicate")
            return 1.0 - truth_values[0]
        elif self.connective == LogicalConnective.IMPLIES:
            if len(truth_values) != 2:
                raise ValueError("IMPLIES connective requires exactly two predicates")
            # 实质蕴涵: P → Q = ¬P ∨ Q
            return max(1.0 - truth_values[0], truth_values[1])
        elif self.connective == LogicalConnective.IFF:
            if len(truth_values) != 2:
                raise ValueError("IFF connective requires exactly two predicates")
            # 等价: P ↔ Q = (P ∧ Q) ∨ (¬P ∧ ¬Q)
            return max(min(truth_values[0], truth_values[1]),
                      min(1.0 - truth_values[0], 1.0 - truth_values[1]))
        else:
            raise ValueError(f"Unsupported connective: {self.connective}")

    def get_description(self) -> str:
        if len(self.predicates) == 1:
            return f"{self.connective.value} ({self.predicates[0].get_description()})"

        descriptions = [pred.get_description() for pred in self.predicates]
        if len(self.predicates) == 2:
            return f"({descriptions[0]}) {self.connective.value} ({descriptions[1]})"
        else:
            return f" {self.connective.value} ".join(f"({desc})" for desc in descriptions)


class QuantifiedPredicate(FirstOrderPredicate):
    """量化谓词（全称量词∀或存在量词∃）"""

    def __init__(self, name: str, predicate: FirstOrderPredicate,
                 quantifier: str = "forall", aggregation_func: str = "min"):
        """
        初始化量化谓词

        Args:
            name: 谓词名称
            predicate: 要量化的谓词
            quantifier: 量词类型（"forall" 或 "exists"）
            aggregation_func: 聚合函数（"min", "max", "mean", "prod"）
        """
        super().__init__(name, predicate.arity)
        self.predicate = predicate
        self.quantifier = quantifier.lower()
        self.aggregation_func = aggregation_func.lower()

    def evaluate(self, *args: Any) -> float:
        """
        评估量化谓词

        Args:
            *args: 谓词参数，最后一个参数应为可迭代对象

        Returns:
            量化谓词的真值
        """
        if len(args) < self.predicate.arity + 1:
            raise ValueError("Insufficient arguments for quantified predicate")

        # 分离固定参数和可迭代参数
        fixed_args = args[:-1]
        iterable_arg = args[-1]

        # 计算所有实例的真值
        truth_values = []
        for item in iterable_arg:
            full_args = fixed_args + (item,)
            truth_values.append(self.predicate.evaluate(*full_args))

        # 根据聚合函数计算最终真值
        if self.aggregation_func == "min":
            result = min(truth_values)
        elif self.aggregation_func == "max":
            result = max(truth_values)
        elif self.aggregation_func == "mean":
            result = np.mean(truth_values)
        elif self.aggregation_func == "prod":
            result = np.prod(truth_values)
        else:
            raise ValueError(f"Unsupported aggregation function: {self.aggregation_func}")

        # 对于存在量词，如果至少有一个为真则结果为真
        if self.quantifier == "exists":
            return max(truth_values)
        # 对于全称量词，所有都必须为真
        elif self.quantifier == "forall":
            return min(truth_values)
        else:
            raise ValueError(f"Unsupported quantifier: {self.quantifier}")

    def get_description(self) -> str:
        return f"{self.quantifier} x: {self.predicate.get_description()}"


def create_feature_predicates(feature_names: List[str], fuzzy_variables: Dict[str, Any]) -> Dict[str, FuzzyPredicate]:
    """
    为特征创建模糊谓词

    Args:
        feature_names: 特征名称列表
        fuzzy_variables: 模糊变量字典

    Returns:
        特征谓词字典
    """
    predicates = {}

    for feature_name in feature_names:
        if feature_name in fuzzy_variables:
            fuzzy_var = fuzzy_variables[feature_name]
            for set_name, fuzzy_set in fuzzy_var.fuzzy_sets.items():
                predicate_name = f"{feature_name}_{set_name}"
                predicate = FuzzyPredicate(
                    name=predicate_name,
                    feature_extractor=feature_name,
                    fuzzy_set=fuzzy_set,
                    description=f"Feature {feature_name} is {set_name}"
                )
                predicates[predicate_name] = predicate

    return predicates


def create_domain_specific_predicates() -> Dict[str, FuzzyPredicate]:
    """
    创建故障诊断领域特定的谓词

    Returns:
        领域特定谓词字典
    """
    # 这里可以根据故障诊断的领域知识创建特定的谓词
    # 例如：高振动、强冲击、频率特征明显等

    predicates = {}

    # 示例：振动强度谓词
    # 注意：这些谓词需要与具体的模糊变量结合使用
    predicates["HighVibration"] = None  # 需要在具体使用时绑定模糊集合
    predicates["StrongImpact"] = None
    predicates["FrequencyDominance"] = None

    return predicates