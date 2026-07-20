"""
Evaluation Package
==================

This package provides tools for evaluating the quality of LLM-generated explanations
in fault diagnosis scenarios.

Modules:
- quality_evaluator: Main evaluation framework with 5 quality dimensions
- metrics: Individual evaluation metrics

Usage Example:
    from evaluation import ExplanationQualityEvaluator

    evaluator = ExplanationQualityEvaluator()
    result = evaluator.evaluate(explanation, ground_truth, ir)
    print(f"Overall score: {result.overall_score:.2f}")
"""

from .quality_evaluator import (
    ExplanationQualityEvaluator,
    EvaluationResult,
    QualityMetric,
    UnderstandabilityMetric,
    TechnicalAccuracyMetric,
    UsefulnessMetric,
    CompletenessMetric,
    TrustworthinessMetric
)

__all__ = [
    'ExplanationQualityEvaluator',
    'EvaluationResult',
    'QualityMetric',
    'UnderstandabilityMetric',
    'TechnicalAccuracyMetric',
    'UsefulnessMetric',
    'CompletenessMetric',
    'TrustworthinessMetric'
]