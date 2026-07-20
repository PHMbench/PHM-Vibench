"""
模糊规则库模块

实现模糊规则的定义、管理和组织，支持基于1阶谓词逻辑的规则表示。

核心组件：
- Predicate: 谓词条件（规则前提）
- FuzzyRule: 模糊规则定义
- RuleBase: 规则库管理器
"""

from typing import Dict, List, Union, Tuple, Any, Optional
import numpy as np
import torch
from dataclasses import dataclass
from .predicates import FirstOrderPredicate, LogicalConnective


@dataclass
class Predicate:
    """规则前提中的谓词条件"""

    feature_name: str
    fuzzy_set: str
    weight: float = 1.0

    def __str__(self) -> str:
        return f"{self.feature_name} is {self.fuzzy_set}"


class FuzzyRule:
    """模糊规则类"""

    def __init__(self, rule_id: str, premises: List[Predicate],
                 conclusion: str, weight: float = 1.0,
                 description: str = "", connective: LogicalConnective = LogicalConnective.AND):
        """
        初始化模糊规则

        Args:
            rule_id: 规则唯一标识
            premises: 前提条件列表（谓词列表）
            conclusion: 结论（故障类型）
            weight: 规则权重
            description: 规则描述
            connective: 前提之间的逻辑连接词
        """
        self.rule_id = rule_id
        self.premises = premises
        self.conclusion = conclusion
        self.weight = weight
        self.description = description
        self.connective = connective
        self.activation_degree = 0.0  # 规则激活度
        self.firing_strength = 0.0    # 规则激发强度

    def evaluate_premises(self, features: Dict[str, Dict[str, float]]) -> float:
        """
        评估规则前提的满足程度

        Args:
            features: 特征隶属度字典 {feature_name: {fuzzy_set: membership_value}}

        Returns:
            前提的满足程度 [0,1]
        """
        premise_values = []

        for premise in self.premises:
            feature_name = premise.feature_name
            fuzzy_set = premise.fuzzy_set

            # 获取特征在该模糊集合下的隶属度
            if (feature_name in features and
                fuzzy_set in features[feature_name]):
                membership = features[feature_name][fuzzy_set]
                weighted_membership = membership * premise.weight
                premise_values.append(weighted_membership)
            else:
                # 如果特征或模糊集合不存在，则前提不满足
                premise_values.append(0.0)

        # 根据逻辑连接词计算整体前提满足度
        if not premise_values:
            return 0.0

        if self.connective == LogicalConnective.AND:
            return min(premise_values)
        elif self.connective == LogicalConnective.OR:
            return max(premise_values)
        elif self.connective == LogicalConnective.NOT:
            if len(premise_values) != 1:
                raise ValueError("NOT connective requires exactly one premise")
            return 1.0 - premise_values[0]
        else:
            raise ValueError(f"Unsupported connective in rule premises: {self.connective}")

    def calculate_firing_strength(self, features: Dict[str, Dict[str, float]]) -> float:
        """
        计算规则激发强度

        Args:
            features: 特征隶属度字典

        Returns:
            规则激发强度
        """
        premise_satisfaction = self.evaluate_premises(features)
        self.firing_strength = premise_satisfaction * self.weight
        return self.firing_strength

    def get_contribution(self, features: Dict[str, Dict[str, float]]) -> Tuple[str, float]:
        """
        获取规则对结论的贡献

        Args:
            features: 特征隶属度字典

        Returns:
            (结论, 贡献度)
        """
        contribution = self.calculate_firing_strength(features)
        return self.conclusion, contribution

    def __str__(self) -> str:
        premise_str = f" {self.connective.value} ".join(str(p) for p in self.premises)
        return f"Rule {self.rule_id}: IF {premise_str} THEN {self.conclusion} (weight={self.weight})"

    def to_dict(self) -> Dict[str, Any]:
        """将规则转换为字典格式"""
        return {
            "rule_id": self.rule_id,
            "premises": [
                {
                    "feature_name": p.feature_name,
                    "fuzzy_set": p.fuzzy_set,
                    "weight": p.weight
                } for p in self.premises
            ],
            "conclusion": self.conclusion,
            "weight": self.weight,
            "description": self.description,
            "connective": self.connective.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FuzzyRule':
        """从字典创建规则"""
        premises = [
            Predicate(
                feature_name=p["feature_name"],
                fuzzy_set=p["fuzzy_set"],
                weight=p.get("weight", 1.0)
            ) for p in data["premises"]
        ]

        return cls(
            rule_id=data["rule_id"],
            premises=premises,
            conclusion=data["conclusion"],
            weight=data.get("weight", 1.0),
            description=data.get("description", ""),
            connective=LogicalConnective(data.get("connective", "and"))
        )


class RuleBase:
    """规则库管理器"""

    def __init__(self, name: str = "DefaultRuleBase"):
        """
        初始化规则库

        Args:
            name: 规则库名称
        """
        self.name = name
        self.rules: Dict[str, FuzzyRule] = {}
        self.conclusions: Dict[str, List[str]] = {}  # conclusion -> [rule_ids]

    def add_rule(self, rule: FuzzyRule) -> None:
        """
        添加规则到规则库

        Args:
            rule: 要添加的规则
        """
        if rule.rule_id in self.rules:
            raise ValueError(f"Rule with ID '{rule.rule_id}' already exists")

        self.rules[rule.rule_id] = rule

        # 更新结论索引
        if rule.conclusion not in self.conclusions:
            self.conclusions[rule.conclusion] = []
        self.conclusions[rule.conclusion].append(rule.rule_id)

    def remove_rule(self, rule_id: str) -> None:
        """
        从规则库移除规则

        Args:
            rule_id: 要移除的规则ID
        """
        if rule_id not in self.rules:
            raise ValueError(f"Rule with ID '{rule_id}' not found")

        rule = self.rules[rule_id]

        # 从结论索引中移除
        if rule.conclusion in self.conclusions:
            self.conclusions[rule.conclusion].remove(rule_id)
            if not self.conclusions[rule.conclusion]:
                del self.conclusions[rule.conclusion]

        del self.rules[rule_id]

    def get_rule(self, rule_id: str) -> Optional[FuzzyRule]:
        """获取指定ID的规则"""
        return self.rules.get(rule_id)

    def get_rules_for_conclusion(self, conclusion: str) -> List[FuzzyRule]:
        """获取支持特定结论的所有规则"""
        rule_ids = self.conclusions.get(conclusion, [])
        return [self.rules[rule_id] for rule_id in rule_ids if rule_id in self.rules]

    def get_all_conclusions(self) -> List[str]:
        """获取所有可能的结论"""
        return list(self.conclusions.keys())

    def evaluate_all_rules(self, features: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        评估所有规则的激活程度

        Args:
            features: 特征隶属度字典

        Returns:
            结论到激活程度的映射 {conclusion: activation_strength}
        """
        conclusion_activations = {}

        for rule in self.rules.values():
            conclusion, activation = rule.get_contribution(features)

            if conclusion not in conclusion_activations:
                conclusion_activations[conclusion] = 0.0

            conclusion_activations[conclusion] = max(
                conclusion_activations[conclusion], activation
            )

        return conclusion_activations

    def get_active_rules(self, features: Dict[str, Dict[str, float]],
                        threshold: float = 0.0) -> List[Tuple[FuzzyRule, float]]:
        """
        获取被激活的规则

        Args:
            features: 特征隶属度字典
            threshold: 激活阈值

        Returns:
            被激活的规则列表 [(rule, activation_strength), ...]
        """
        active_rules = []

        for rule in self.rules.values():
            activation = rule.calculate_firing_strength(features)
            if activation > threshold:
                active_rules.append((rule, activation))

        # 按激活强度排序
        active_rules.sort(key=lambda x: x[1], reverse=True)
        return active_rules

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取规则库统计信息

        Returns:
            统计信息字典
        """
        if not self.rules:
            return {
                "total_rules": 0,
                "conclusions": [],
                "avg_premises_per_rule": 0,
                "avg_weight": 0,
                "rule_distribution": {}
            }

        premise_counts = [len(rule.premises) for rule in self.rules.values()]
        weights = [rule.weight for rule in self.rules.values()]

        conclusion_counts = {}
        for rule in self.rules.values():
            conclusion_counts[rule.conclusion] = conclusion_counts.get(rule.conclusion, 0) + 1

        return {
            "total_rules": len(self.rules),
            "conclusions": list(self.conclusions.keys()),
            "avg_premises_per_rule": np.mean(premise_counts),
            "avg_weight": np.mean(weights),
            "rule_distribution": conclusion_counts,
            "connective_distribution": self._get_connective_distribution()
        }

    def _get_connective_distribution(self) -> Dict[str, int]:
        """获取逻辑连接词分布统计"""
        connective_counts = {}
        for rule in self.rules.values():
            connective = rule.connective.value
            connective_counts[connective] = connective_counts.get(connective, 0) + 1
        return connective_counts

    def save_to_file(self, filename: str) -> None:
        """将规则库保存到文件"""
        import json

        rules_data = [rule.to_dict() for rule in self.rules.values()]

        data = {
            "name": self.name,
            "rules": rules_data,
            "metadata": {
                "total_rules": len(self.rules),
                "conclusions": list(self.conclusions.keys())
            }
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filename: str) -> 'RuleBase':
        """从文件加载规则库"""
        import json

        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        rule_base = cls(name=data.get("name", "LoadedRuleBase"))

        for rule_data in data["rules"]:
            rule = FuzzyRule.from_dict(rule_data)
            rule_base.add_rule(rule)

        return rule_base

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules.values())

    def __str__(self) -> str:
        return f"RuleBase '{self.name}' with {len(self.rules)} rules covering {len(self.conclusions)} conclusions"


def create_fault_diagnosis_rules() -> RuleBase:
    """
    创建故障诊断的基础规则库

    Returns:
        配置好的规则库
    """
    rule_base = RuleBase("FaultDiagnosisRules")

    # 内圈故障规则
    rule1 = FuzzyRule(
        rule_id="IF_001",
        premises=[
            Predicate("RMS", "high"),
            Predicate("Kurtosis", "high"),
            Predicate("CrestFactor", "medium")
        ],
        conclusion="IF",  # Inner Race Fault
        weight=0.9,
        description="高均方根、高峰度、中等峰值因子表明内圈故障",
        connective=LogicalConnective.AND
    )

    # 外圈故障规则
    rule2 = FuzzyRule(
        rule_id="OF_001",
        premises=[
            Predicate("RMS", "medium"),
            Predicate("Skewness", "low"),
            Predicate("ShapeFactor", "high")
        ],
        conclusion="OF",  # Outer Race Fault
        weight=0.8,
        description="中等均方根、低偏度、高形状因子表明外圈故障",
        connective=LogicalConnective.AND
    )

    # 滚动体故障规则
    rule3 = FuzzyRule(
        rule_id="BF_001",
        premises=[
            Predicate("CrestFactor", "high"),
            Predicate("Kurtosis", "medium"),
            Predicate("RMS", "medium")
        ],
        conclusion="BF",  # Ball Fault
        weight=0.7,
        description="高峰值因子、中等峭度、中等均方根表明滚动体故障",
        connective=LogicalConnective.AND
    )

    # 健康状态规则
    rule4 = FuzzyRule(
        rule_id="HE_001",
        premises=[
            Predicate("RMS", "low"),
            Predicate("Kurtosis", "low"),
            Predicate("CrestFactor", "low")
        ],
        conclusion="HE",  # Healthy
        weight=0.95,
        description="低均方根、低峭度、低峰值因子表明健康状态",
        connective=LogicalConnective.AND
    )

    rule_base.add_rule(rule1)
    rule_base.add_rule(rule2)
    rule_base.add_rule(rule3)
    rule_base.add_rule(rule4)

    return rule_base