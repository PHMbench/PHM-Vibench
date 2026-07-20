"""
Explanation Quality Evaluator
=============================

This module implements the evaluation framework for LLM-generated explanations
in fault diagnosis. It provides quantitative metrics for five quality dimensions:
understandability, technical accuracy, usefulness, completeness, and trustworthiness.

Author: LLM Explainable FD Toolkit
Date: 2025-01-15
"""

import numpy as np
import torch
import nltk
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from ..adapters.model_adapter_base import ExplanationIR

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


@dataclass
class EvaluationResult:
    """Result of explanation quality evaluation"""
    overall_score: float  # 0-1
    detailed_scores: Dict[str, float]  # Scores for each dimension
    recommendations: List[str]  # Improvement suggestions
    metrics: Dict[str, Any]  # Raw metrics


class QualityMetric(ABC):
    """Base class for quality evaluation metrics."""

    @abstractmethod
    def score(self, explanation: str, ground_truth: Optional[str] = None,
              ir: Optional[ExplanationIR] = None, **kwargs) -> float:
        """Calculate quality score (0-1)."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get metric name."""
        pass


class UnderstandabilityMetric(QualityMetric):
    """Evaluates how easily explanations can be understood."""

    def __init__(self):
        self.readability_scores = []
        self.length_penalty_weight = 0.1

    def score(self, explanation: str, ground_truth: Optional[str] = None,
              ir: Optional[ExplanationIR] = None, **kwargs) -> float:
        """Calculate understandability score."""
        # Components of understandability
        readability = self._calculate_readability(explanation)
        simplicity = self._calculate_simplicity(explanation)
        clarity = self._calculate_clarity(explanation)
        conciseness = self._calculate_conciseness(explanation)

        # Weighted combination
        scores = np.array([readability, simplicity, clarity, conciseness])
        weights = np.array([0.3, 0.25, 0.25, 0.2])

        return float(np.dot(scores, weights))

    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score."""
        # Flesch Reading Ease approximation
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return 0.0

        # Count words and syllables (simplified)
        words = text.split()
        avg_sentence_length = len(words) / len(sentences)

        # Average words per sentence should be around 15-20 for good readability
        if avg_sentence_length < 10:
            readability = 1.0
        elif avg_sentence_length > 25:
            readability = 0.5
        else:
            readability = 1.0 - (avg_sentence_length - 10) / 30

        return max(0.0, readability)

    def _calculate_simplicity(self, text: str) -> float:
        """Calculate simplicity based on jargon and complexity."""
        # Count technical terms
        jargon_terms = [
            'frequency', 'amplitude', 'phase', 'harmonic', 'resonance',
            'modulation', 'demodulation', 'convolution', 'transform',
            'eigenvalue', 'eigenvector', 'gradient', 'backpropagation'
        ]

        words = text.lower().split()
        jargon_count = sum(1 for word in words if any(term in word for term in jargon_terms))

        # Penalize too much jargon
        if len(words) == 0:
            return 1.0

        jargon_ratio = jargon_count / len(words)
        if jargon_ratio < 0.05:
            return 1.0
        elif jargon_ratio < 0.1:
            return 0.8
        elif jargon_ratio < 0.2:
            return 0.6
        else:
            return 0.4

    def _calculate_clarity(self, text: str) -> float:
        """Calculate clarity based on structure."""
        # Check for clear structure
        has_numbers = any(c.isdigit() for c in text)
        has_percentages = '%' in text
        has_units = any(unit in text for unit in ['Hz', 'kHz', 'mm', '°C', 'g'])

        clarity_score = 0.5  # Base score
        if has_numbers:
            clarity_score += 0.2
        if has_percentages:
            clarity_score += 0.2
        if has_units:
            clarity_score += 0.1

        return min(1.0, clarity_score)

    def _calculate_conciseness(self, text: str) -> float:
        """Calculate conciseness."""
        words = text.split()
        if len(words) == 0:
            return 1.0

        # Ideal length: 50-150 words
        if len(words) < 30:
            return 0.8  # Too short might be incomplete
        elif len(words) < 150:
            return 1.0
        elif len(words) < 300:
            return 0.7
        else:
            return 0.5

    def get_name(self) -> str:
        return "Understandability"


class TechnicalAccuracyMetric(QualityMetric):
    """Evaluates technical accuracy of explanations."""

    def score(self, explanation: str, ground_truth: Optional[str] = None,
              ir: Optional[ExplanationIR] = None, **kwargs) -> float:
        """Calculate technical accuracy score."""
        if not ground_truth or not ir:
            return 0.5  # Cannot assess without ground truth

        # Check factual consistency
        factual_score = self._check_factual_consistency(explanation, ir)

        # Check numerical accuracy
        numerical_score = self._check_numerical_accuracy(explanation, ground_truth)

        # Check terminology consistency
        terminology_score = self._check_terminology_consistency(explanation, ir)

        # Weighted average
        return (factual_score * 0.4 + numerical_score * 0.3 + terminology_score * 0.3)

    def _check_factual_consistency(self, explanation: str, ir: ExplanationIR) -> float:
        """Check if explanation is consistent with IR."""
        # Extract key facts from IR
        key_facts = {
            'model': ir.model_name,
            'confidence': ir.prediction_confidence,
            'top_features': ir.feature_ranking[:3],
            'processing_steps': ir.processing_steps
        }

        consistency_score = 1.0

        # Check model mention
        if key_facts['model'] not in explanation:
            consistency_score -= 0.2

        # Check confidence range
        conf_lower = key_facts['confidence'] - 0.1
        conf_upper = key_facts['confidence'] + 0.1
        if 'confidence' in explanation.lower():
            # Extract confidence number (simplified)
            import re
            conf_matches = re.findall(r'\d+\.?\d*', explanation)
            for match in conf_matches:
                conf_val = float(match)
                if not (conf_lower <= conf_val <= conf_upper):
                    consistency_score -= 0.2

        return max(0.0, consistency_score)

    def _check_numerical_accuracy(self, explanation: str, ground_truth: str) -> float:
        """Check numerical accuracy."""
        import re

        # Extract numbers from explanation and ground truth
        expl_numbers = re.findall(r'\d+\.?\d*', explanation)
        gt_numbers = re.findall(r'\d+\.?\d*', ground_truth)

        if not expl_numbers or not gt_numbers:
            return 1.0

        # Simple comparison of number distributions
        accuracy_score = 0.0
        for expl_num in expl_numbers:
            expl_val = float(expl_num)
            for gt_num in gt_numbers:
                gt_val = float(gt_num)
                # Check if numbers are close (10% tolerance)
                if abs(expl_val - gt_val) / (abs(gt_val) + 1e-8) < 0.1:
                    accuracy_score += 1.0

        return min(1.0, accuracy_score / max(len(expl_numbers), 1))

    def _check_terminology_consistency(self, explanation: str, ir: ExplanationIR) -> float:
        """Check terminology consistency."""
        # Extract feature names from explanation
        mentioned_features = []
        for feature in ir.feature_ranking:
            if feature in explanation:
                mentioned_features.append(feature)

        # Calculate coverage
        coverage = len(mentioned_features) / min(len(ir.feature_ranking), 3)

        return coverage

    def get_name(self) -> str:
        return "Technical Accuracy"


class UsefulnessMetric(QualityMetric):
    """Evaluates usefulness for decision-making."""

    def score(self, explanation: str, ground_truth: Optional[str] = None,
              ir: Optional[ExplanationIR] = None, **kwargs) -> float:
        """Calculate usefulness score."""
        # Check for actionable information
        actionability = self._check_actionability(explanation)

        # Check for practical recommendations
        practicality = self._check_practicality(explanation)

        # Check for decision support value
        decision_support = self._check_decision_support(explanation, ir)

        return (actionability * 0.4 + practicality * 0.3 + decision_support * 0.3)

    def _check_actionability(self, text: str) -> float:
        """Check if explanation provides actionable insights."""
        action_words = [
            'should', 'recommend', 'suggest', 'advise', 'need to',
            'must', 'check', 'inspect', 'replace', 'repair', 'monitor'
        ]

        words = text.lower().split()
        action_count = sum(1 for word in words if word in action_words)

        # Score based on presence of action words
        if action_count > 3:
            return 1.0
        elif action_count > 1:
            return 0.7
        elif action_count > 0:
            return 0.5
        else:
            return 0.2

    def _check_practicality(self, text: str) -> float:
        """Check if explanation is practically useful."""
        practical_indicators = [
            'hours', 'days', 'weeks', 'cost', 'budget', 'schedule',
            'priority', 'urgency', 'severity', 'risk'
        ]

        practical_count = sum(1 for indicator in practical_indicators if indicator in text.lower())

        # Normalize score
        return min(1.0, practical_count / 3)

    def _check_decision_support(self, text: str, ir: Optional[ExplanationIR] = None) -> float:
        """Check decision support value."""
        if not ir:
            return 0.5

        # Check if explanation addresses uncertainty
        if ir.uncertainty and 'uncertainty' in text.lower():
            return 1.0

        # Check if explanation provides confidence
        if 'confidence' in text.lower():
            return 0.8

        # Check if explanation compares alternatives
        if 'alternatives' in text.lower() or 'compare' in text.lower():
            return 0.7

        return 0.5

    def get_name(self) -> str:
        return "Usefulness"


class CompletenessMetric(QualityMetric):
    """Evaluates completeness of explanation."""

    def __init__(self):
        self.checklist = [
            'diagnosis_present',
            'confidence_mentioned',
            'key_features_explained',
            'evidence_provided',
            'recommendations_given'
        ]

    def score(self, explanation: str, ground_truth: Optional[str] = None,
              ir: Optional[ExplanationIR] = None, **kwargs) -> float:
        """Calculate completeness score."""
        score = 0.0

        # Check checklist items
        if self._has_diagnosis(explanation):
            score += 0.2
        if self._has_confidence(explanation):
            score += 0.2
        if self._explains_features(explanation):
            score += 0.2
        if self._has_evidence(explanation):
            score += 0.2
        if self._has_recommendations(explanation):
            score += 0.2

        return score

    def _has_diagnosis(self, text: str) -> bool:
        """Check if diagnosis is present."""
        diagnosis_words = ['fault', 'normal', 'healthy', 'abnormal', 'detected', 'diagnosis']
        return any(word in text.lower() for word in diagnosis_words)

    def _has_confidence(self, text: str) -> bool:
        """Check if confidence is mentioned."""
        return 'confidence' in text.lower() or 'certain' in text.lower()

    def _explains_features(self, text: str) -> bool:
        """Check if key features are explained."""
        feature_words = ['feature', 'characteristic', 'indicator', 'pattern']
        return any(word in text.lower() for word in feature_words)

    def _has_evidence(self, text: str) -> bool:
        """Check if evidence is provided."""
        evidence_words = ['evidence', 'because', 'due to', 'since', 'based on', 'according to']
        return any(word in text.lower() for word in evidence_words)

    def _has_recommendations(self, text: str) -> bool:
        """Check if recommendations are given."""
        recommend_words = ['recommend', 'suggest', 'should', 'advise', 'action']
        return any(word in text.lower() for word in recommend_words)

    def get_name(self) -> str:
        return "Completeness"


class TrustworthinessMetric(QualityMetric):
    """Evaluates trustworthiness of explanation."""

    def score(self, explanation: str, ground_truth: Optional[str] = None,
              ir: Optional[ExplanationIR] = None, **kwargs) -> float:
        """Calculate trustworthiness score."""
        # Check for uncertainty communication
        uncertainty_comm = self._check_uncertainty_communication(explanation)

        # Check for overconfidence
        overconfidence = self._check_overconfidence(explanation)

        # Check for hedging and limitations
        hedging = self._check_hedging(explanation)

        # Check for specificity
        specificity = self._check_specificity(explanation)

        base_score = (uncertainty_comm * 0.3 + (1 - overconfidence) * 0.3 + hedging * 0.2 + specificity * 0.2)

        return max(0.0, min(1.0, base_score))

    def _check_uncertainty_communication(self, text: str) -> float:
        """Check if uncertainty is properly communicated."""
        uncertainty_phrases = [
            'uncertain', 'unsure', 'might', 'could be', 'possibly',
            'approximate', 'estimate', 'around', 'about', 'likely'
        ]

        count = sum(1 for phrase in uncertainty_phrases if phrase in text.lower())

        # Some uncertainty communication is good, too much is bad
        if count == 0:
            return 0.5  # No uncertainty communication
        elif count <= 3:
            return 1.0
        else:
            return max(0.0, 1.0 - (count - 3) * 0.1)

    def _check_overconfidence(self, text: str) -> float:
        """Check for overconfidence (penalty score)."""
        overconfident_words = ['definitely', 'certainly', 'always', 'never', 'perfectly', 'exactly']

        count = sum(1 for word in overconfident_words if word in text.lower())

        # Return penalty score (0 = no overconfidence)
        return min(1.0, count * 0.3)

    def _check_hedging(self, text: str) -> float:
        """Check for appropriate hedging."""
        hedge_words = ['may', 'might', 'could', 'perhaps', 'typically', 'generally']

        count = sum(1 for word in hedge_words if word in text.lower())

        # Some hedging is good
        if count <= 2:
            return 1.0
        elif count <= 4:
            return 0.8
        else:
            return 0.6

    def _check_specificity(self, text: str) -> float:
        """Check if explanation is appropriately specific."""
        # Check for specific numbers, percentages, or measurements
        has_specifics = (
            any(c.isdigit() for c in text) and
            ('%' in text or 'hz' in text.lower() or 'khz' in text.lower())
        )

        return 1.0 if has_specifics else 0.7

    def get_name(self) -> str:
        return "Trustworthiness"


class ExplanationQualityEvaluator:
    """Main evaluator for explanation quality."""

    def __init__(self):
        """Initialize the evaluator with default metrics."""
        self.metrics = {
            'understandability': UnderstandabilityMetric(),
            'technical_accuracy': TechnicalAccuracyMetric(),
            'usefulness': UsefulnessMetric(),
            'completeness': CompletenessMetric(),
            'trustworthiness': TrustworthinessMetric()
        }

        # Minimum scores for passing evaluation
        self.min_scores = {
            'understandability': 0.7,
            'technical_accuracy': 0.8,
            'usefulness': 0.6,
            'completeness': 0.7,
            'trustworthiness': 0.7
        }

    def evaluate(self, explanation: str, ground_truth: Optional[str] = None,
                 ir: Optional[ExplanationIR] = None, **kwargs) -> EvaluationResult:
        """
        Evaluate explanation quality across all dimensions.

        Args:
            explanation: The generated explanation text
            ground_truth: Ground truth explanation (optional)
            ir: Intermediate representation (optional)
            **kwargs: Additional parameters

        Returns:
            EvaluationResult with scores and recommendations
        """
        detailed_scores = {}
        recommendations = []

        # Evaluate each metric
        for name, metric in self.metrics.items():
            score = metric.score(explanation, ground_truth, ir, **kwargs)
            detailed_scores[name] = score

            # Check if below minimum
            if score < self.min_scores[name]:
                recommendations.append(f"Improve {name}: Score {score:.2f} below minimum {self.min_scores[name]:.2f}")

        # Calculate overall score
        overall_score = np.mean(list(detailed_scores.values()))

        # Add general recommendations
        if overall_score < 0.7:
            recommendations.append("Overall quality needs improvement")

        # Collect metrics for analysis
        metrics_data = {
            'word_count': len(explanation.split()),
            'sentence_count': len(nltk.sent_tokenize(explanation)),
            'has_ground_truth': ground_truth is not None,
            'has_ir': ir is not None
        }

        return EvaluationResult(
            overall_score=overall_score,
            detailed_scores=detailed_scores,
            recommendations=recommendations,
            metrics=metrics_data
        )

    def batch_evaluate(self, explanations: List[str],
                       ground_truths: Optional[List[str]] = None,
                       irs: Optional[List[ExplanationIR]] = None,
                       **kwargs) -> List[EvaluationResult]:
        """
        Evaluate multiple explanations.

        Args:
            explanations: List of explanation texts
            ground_truths: List of ground truths (optional)
            irs: List of intermediate representations (optional)
            **kwargs: Additional parameters

        Returns:
            List of evaluation results
        """
        results = []

        for i, explanation in enumerate(explanations):
            gt = ground_truths[i] if ground_truths and i < len(ground_truths) else None
            ir_obj = irs[i] if irs and i < len(irs) else None

            result = self.evaluate(explanation, gt, ir_obj, **kwargs)
            results.append(result)

        return results

    def add_metric(self, name: str, metric: QualityMetric):
        """Add a custom evaluation metric."""
        self.metrics[name] = metric

    def remove_metric(self, name: str):
        """Remove an evaluation metric."""
        if name in self.metrics:
            del self.metrics[name]

    def get_metric_names(self) -> List[str]:
        """Get list of metric names."""
        return list(self.metrics.keys())

    def set_min_score(self, metric_name: str, min_score: float):
        """Set minimum score for a metric."""
        self.min_scores[metric_name] = min_score