"""
Fuzzy-XFD: 模糊逻辑可解释故障诊断系统

该模块提供了基于1阶谓词逻辑的模糊推理系统实现，
包括隶属度函数、规则库定义和推理引擎等核心组件。

主要组件:
- membership_functions: 隶属度函数定义
- rule_base: 规则库管理
- inference_engine: 模糊推理引擎
- predicates: 1阶谓词逻辑框架
"""

from .membership_functions import (
    TriangularMembershipFunction,
    GaussianMembershipFunction,
    TrapezoidalMembershipFunction,
    FuzzyVariable,
    FuzzySet
)

from .rule_base import (
    FuzzyRule,
    RuleBase,
    Predicate,
    create_fault_diagnosis_rules
)

from .inference_engine import (
    FuzzyInferenceEngine,
    MamdaniInferenceEngine,
    DefuzzificationMethod,
    create_inference_engine,
    create_default_fuzzy_variables,
    FuzzyInferenceSystem
)

from .predicates import (
    FirstOrderPredicate,
    FuzzyPredicate,
    LogicalConnective
)

__version__ = "0.1.0"
__all__ = [
    # Membership functions
    "TriangularMembershipFunction",
    "GaussianMembershipFunction",
    "TrapezoidalMembershipFunction",
    "FuzzyVariable",
    "FuzzySet",

    # Rule base
    "FuzzyRule",
    "RuleBase",
    "Predicate",
    "create_fault_diagnosis_rules",

    # Inference engine
    "FuzzyInferenceEngine",
    "MamdaniInferenceEngine",
    "DefuzzificationMethod",
    "create_inference_engine",
    "create_default_fuzzy_variables",
    "FuzzyInferenceSystem",

    # Predicates
    "FirstOrderPredicate",
    "FuzzyPredicate",
    "LogicalConnective"
]