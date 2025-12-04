"""
Evaluation utilities for PHM-Vibench.

This module provides evaluation utilities for measuring model performance,
zero-shot transfer learning, and representation quality assessment.

Available evaluators:
- ZeroShotEvaluator: Zero-shot evaluation with linear probe on frozen backbones
- RepresentationQualityAnalyzer: Analysis of learned representation quality
"""

from .ZeroShotEvaluator import ZeroShotEvaluator, RepresentationQualityAnalyzer, LinearProbeClassifier

__all__ = [
    'ZeroShotEvaluator',
    'RepresentationQualityAnalyzer', 
    'LinearProbeClassifier'
]