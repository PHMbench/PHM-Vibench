"""
Feedback Collector for Diagnostic Conversations

Collects, processes, and analyzes user feedback to improve the
diagnostic conversation system and LLM responses.
"""

from typing import Dict, Any, List, Optional, Callable
import json
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback."""
    RESPONSE_QUALITY = "response_quality"
    DIAGNOSIS_ACCURACY = "diagnosis_accuracy"
    CONVERSATION_FLOW = "conversation_flow"
    USER_SATISFACTION = "user_satisfaction"
    TECHNICAL_CORRECTNESS = "technical_correctness"
    USABILITY = "usability"


class FeedbackRating(Enum):
    """Rating levels for feedback."""
    EXCELLENT = 5
    VERY_GOOD = 4
    GOOD = 3
    FAIR = 2
    POOR = 1


@dataclass
class FeedbackItem:
    """Individual feedback item."""
    feedback_id: str
    session_id: str
    timestamp: datetime
    feedback_type: FeedbackType
    rating: FeedbackRating
    comment: str
    context: Dict[str, Any]
    specific_issues: List[str]
    suggestions: List[str]
    user_role: str
    device_info: Dict[str, Any]


@dataclass
class ConversationMetrics:
    """Metrics for conversation performance."""
    session_id: str
    duration: float
    num_turns: int
    user_engagement_score: float
    response_quality_score: float
    resolution_achieved: bool
    user_satisfaction: float
    technical_accuracy: float
    conversation_flow_score: float


class FeedbackCollector:
    """
    Collects and analyzes user feedback for continuous improvement.

    This class manages feedback collection, processes feedback data,
    and provides analytics for system improvement.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize feedback collector.

        Args:
            storage_path: Path for storing feedback data
        """
        self.storage_path = storage_path
        self.feedback_items = []
        self.conversation_metrics = []
        self.feedback_processors = self._initialize_feedback_processors()
        self.analytics_engine = self._initialize_analytics_engine()

    def collect_session_feedback(self,
                                session_id: str,
                                ratings: Dict[str, int],
                                comments: Dict[str, str],
                                context: Optional[Dict[str, Any]] = None) -> str:
        """
        Collect feedback for a conversation session.

        Args:
            session_id: Session identifier
            ratings: Dictionary of feedback ratings
            comments: Dictionary of user comments
            context: Additional context information

        Returns:
            Feedback ID
        """
        feedback_id = str(uuid.uuid4())
        timestamp = datetime.now()

        # Create feedback items for different types
        feedback_items = []

        for feedback_type_str, rating in ratings.items():
            try:
                feedback_type = FeedbackType(feedback_type_str)
                rating_enum = FeedbackRating(rating)
                comment = comments.get(feedback_type_str, "")

                feedback_item = FeedbackItem(
                    feedback_id=f"{feedback_id}_{feedback_type_str}",
                    session_id=session_id,
                    timestamp=timestamp,
                    feedback_type=feedback_type,
                    rating=rating_enum,
                    comment=comment,
                    context=context or {},
                    specific_issues=self._extract_issues(comment),
                    suggestions=self._extract_suggestions(comment),
                    user_role="engineer",
                    device_info=context.get("device_info", {}) if context else {}
                )

                feedback_items.append(feedback_item)

            except ValueError as e:
                logger.warning(f"Invalid feedback type or rating: {e}")

        # Store feedback items
        self.feedback_items.extend(feedback_items)

        # Save to storage if configured
        if self.storage_path:
            self._save_feedback_to_storage(feedback_items)

        logger.info(f"Collected {len(feedback_items)} feedback items for session {session_id}")
        return feedback_id

    def collect_response_feedback(self,
                                 session_id: str,
                                 turn_id: str,
                                 rating: int,
                                 comment: str,
                                 response_quality_aspects: Dict[str, int]) -> None:
        """
        Collect feedback for specific response.

        Args:
            session_id: Session identifier
            turn_id: Conversation turn ID
            rating: Overall rating (1-5)
            comment: User comment
            response_quality_aspects: Ratings for specific aspects
        """
        feedback_item = FeedbackItem(
            feedback_id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=datetime.now(),
            feedback_type=FeedbackType.RESPONSE_QUALITY,
            rating=FeedbackRating(rating),
            comment=comment,
            context={
                "turn_id": turn_id,
                "quality_aspects": response_quality_aspects
            },
            specific_issues=self._extract_issues(comment),
            suggestions=self._extract_suggestions(comment),
            user_role="engineer",
            device_info={}
        )

        self.feedback_items.append(feedback_item)

        if self.storage_path:
            self._save_feedback_to_storage([feedback_item])

    def collect_conversation_metrics(self,
                                    session_id: str,
                                    duration: float,
                                    num_turns: int,
                                    final_outcome: str,
                                    context: Optional[Dict[str, Any]] = None) -> None:
        """
        Collect conversation performance metrics.

        Args:
            session_id: Session identifier
            duration: Conversation duration in seconds
            num_turns: Number of conversation turns
            final_outcome: Final outcome of conversation
            context: Additional context
        """
        # Calculate metrics
        metrics = ConversationMetrics(
            session_id=session_id,
            duration=duration,
            num_turns=num_turns,
            user_engagement_score=self._calculate_engagement_score(num_turns, duration),
            response_quality_score=self._calculate_response_quality_score(session_id),
            resolution_achieved=final_outcome in ["resolved", "completed"],
            user_satisfaction=self._calculate_satisfaction_score(session_id),
            technical_accuracy=self._calculate_technical_accuracy_score(session_id),
            conversation_flow_score=self._calculate_flow_score(session_id, duration, num_turns)
        )

        self.conversation_metrics.append(metrics)

        if self.storage_path:
            self._save_metrics_to_storage([metrics])

    def get_feedback_summary(self,
                           feedback_type: Optional[FeedbackType] = None,
                           date_range: Optional[tuple] = None) -> Dict[str, Any]:
        """
        Get summary of feedback data.

        Args:
            feedback_type: Filter by feedback type
            date_range: Filter by date range (start, end)

        Returns:
            Feedback summary statistics
        """
        filtered_feedback = self._filter_feedback(feedback_type, date_range)

        if not filtered_feedback:
            return {"message": "No feedback data available"}

        # Calculate statistics
        ratings = [item.rating.value for item in filtered_feedback]
        avg_rating = sum(ratings) / len(ratings)

        rating_distribution = {}
        for rating in FeedbackRating:
            count = sum(1 for item in filtered_feedback if item.rating == rating)
            rating_distribution[rating.name] = count

        # Common issues and suggestions
        all_issues = []
        all_suggestions = []
        for item in filtered_feedback:
            all_issues.extend(item.specific_issues)
            all_suggestions.extend(item.suggestions)

        common_issues = self._get_most_common(all_issues)
        common_suggestions = self._get_most_common(all_suggestions)

        return {
            "total_feedback": len(filtered_feedback),
            "average_rating": round(avg_rating, 2),
            "rating_distribution": rating_distribution,
            "common_issues": common_issues,
            "common_suggestions": common_suggestions,
            "feedback_types": list(set(item.feedback_type for item in filtered_feedback))
        }

    def get_conversation_analytics(self) -> Dict[str, Any]:
        """
        Get conversation performance analytics.

        Returns:
            Conversation analytics data
        """
        if not self.conversation_metrics:
            return {"message": "No conversation data available"}

        # Calculate aggregate metrics
        avg_duration = sum(m.duration for m in self.conversation_metrics) / len(self.conversation_metrics)
        avg_turns = sum(m.num_turns for m in self.conversation_metrics) / len(self.conversation_metrics)
        avg_satisfaction = sum(m.user_satisfaction for m in self.conversation_metrics) / len(self.conversation_metrics)

        resolution_rate = sum(1 for m in self.conversation_metrics if m.resolution_achieved) / len(self.conversation_metrics)

        # Performance trends
        recent_metrics = self.conversation_metrics[-10:]  # Last 10 conversations
        if len(recent_metrics) >= 2:
            satisfaction_trend = recent_metrics[-1].user_satisfaction - recent_metrics[0].user_satisfaction
        else:
            satisfaction_trend = 0

        return {
            "total_conversations": len(self.conversation_metrics),
            "average_duration": round(avg_duration, 1),
            "average_turns": round(avg_turns, 1),
            "average_satisfaction": round(avg_satisfaction, 2),
            "resolution_rate": round(resolution_rate, 2),
            "satisfaction_trend": round(satisfaction_trend, 2),
            "performance_breakdown": self._get_performance_breakdown()
        }

    def generate_improvement_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate improvement recommendations based on feedback.

        Returns:
            List of improvement recommendations
        """
        recommendations = []

        # Analyze feedback for patterns
        feedback_summary = self.get_feedback_summary()
        conversation_analytics = self.get_conversation_analytics()

        # Low rating areas
        if "average_rating" in feedback_summary and feedback_summary["average_rating"] < 3.5:
            recommendations.append({
                "category": "response_quality",
                "priority": "high",
                "issue": "Overall response quality below expectations",
                "recommendation": "Improve LLM prompt engineering and response generation",
                "action_items": [
                    "Review and enhance prompt templates",
                    "Improve technical accuracy checks",
                    "Add more domain-specific knowledge"
                ]
            })

        # Low resolution rate
        if "resolution_rate" in conversation_analytics and conversation_analytics["resolution_rate"] < 0.7:
            recommendations.append({
                "category": "conversation_effectiveness",
                "priority": "medium",
                "issue": "Low conversation resolution rate",
                "recommendation": "Improve conversation flow and problem-solving capabilities",
                "action_items": [
                    "Enhance query understanding",
                    "Improve context management",
                    "Add follow-up question suggestions"
                ]
            })

        # Common issues
        if "common_issues" in feedback_summary and feedback_summary["common_issues"]:
            top_issue = feedback_summary["common_issues"][0]
            recommendations.append({
                "category": "user_experience",
                "priority": "medium",
                "issue": f"Frequently reported issue: {top_issue}",
                "recommendation": "Address common user concerns",
                "action_items": [
                    "Investigate root causes of common issues",
                    "Implement targeted improvements",
                    "Add explanatory content for problematic areas"
                ]
            })

        return recommendations

    def export_feedback_data(self, filename: str, format: str = "json") -> None:
        """
        Export feedback data for external analysis.

        Args:
            filename: Output filename
            format: Export format (json, csv)
        """
        if format == "json":
            export_data = {
                "feedback_items": [
                    {
                        "feedback_id": item.feedback_id,
                        "session_id": item.session_id,
                        "timestamp": item.timestamp.isoformat(),
                        "feedback_type": item.feedback_type.value,
                        "rating": item.rating.value,
                        "comment": item.comment,
                        "specific_issues": item.specific_issues,
                        "suggestions": item.suggestions
                    }
                    for item in self.feedback_items
                ],
                "conversation_metrics": [
                    {
                        "session_id": metrics.session_id,
                        "duration": metrics.duration,
                        "num_turns": metrics.num_turns,
                        "user_satisfaction": metrics.user_satisfaction,
                        "resolution_achieved": metrics.resolution_achieved
                    }
                    for metrics in self.conversation_metrics
                ]
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported feedback data to {filename}")

    def _extract_issues(self, comment: str) -> List[str]:
        """Extract specific issues from user comment."""
        issues = []
        comment_lower = comment.lower()

        issue_indicators = {
            "不准确": ["不准确", "错误", "错误信息"],
            "不清楚": ["不清楚", "模糊", "不明白", "难理解"],
            "不完整": ["不完整", "缺少", "遗漏"],
            "太长": ["太长", "冗长", "啰嗦"],
            "不相关": ["不相关", "跑题", "无关"],
            "太技术": ["太技术", "太专业", "难懂"]
        }

        for issue, indicators in issue_indicators.items():
            if any(indicator in comment_lower for indicator in indicators):
                issues.append(issue)

        return issues

    def _extract_suggestions(self, comment: str) -> List[str]:
        """Extract suggestions from user comment."""
        suggestions = []
        comment_lower = comment.lower()

        suggestion_indicators = {
            "更多细节": ["详细", "更多细节", "深入"],
            "更简单": ["简单", "易懂", "通俗"],
            "更实用": ["实用", "可操作", "具体"],
            "更快": ["快速", "简洁", "直接"],
            "更多例子": ["例子", "案例", "实例"]
        }

        for suggestion, indicators in suggestion_indicators.items():
            if any(indicator in comment_lower for indicator in indicators):
                suggestions.append(suggestion)

        return suggestions

    def _filter_feedback(self,
                        feedback_type: Optional[FeedbackType] = None,
                        date_range: Optional[tuple] = None) -> List[FeedbackItem]:
        """Filter feedback based on criteria."""
        filtered = self.feedback_items

        if feedback_type:
            filtered = [item for item in filtered if item.feedback_type == feedback_type]

        if date_range:
            start_date, end_date = date_range
            filtered = [
                item for item in filtered
                if start_date <= item.timestamp <= end_date
            ]

        return filtered

    def _get_most_common(self, items: List[str], top_n: int = 5) -> List[str]:
        """Get most common items from list."""
        from collections import Counter

        if not items:
            return []

        counter = Counter(items)
        return [item for item, count in counter.most_common(top_n)]

    def _calculate_engagement_score(self, num_turns: int, duration: float) -> float:
        """Calculate user engagement score."""
        if duration == 0:
            return 0.0

        # Engagement based on turn frequency and conversation depth
        turn_frequency = num_turns / (duration / 60)  # Turns per minute
        depth_score = min(num_turns / 10, 1.0)  # Normalize to 0-1

        engagement_score = (min(turn_frequency / 2, 1.0) + depth_score) / 2
        return round(engagement_score, 2)

    def _calculate_response_quality_score(self, session_id: str) -> float:
        """Calculate average response quality score for session."""
        session_feedback = [item for item in self.feedback_items if item.session_id == session_id]
        quality_feedback = [item for item in session_feedback if item.feedback_type == FeedbackType.RESPONSE_QUALITY]

        if not quality_feedback:
            return 3.0  # Default neutral score

        ratings = [item.rating.value for item in quality_feedback]
        return round(sum(ratings) / len(ratings), 2)

    def _calculate_satisfaction_score(self, session_id: str) -> float:
        """Calculate user satisfaction score for session."""
        session_feedback = [item for item in self.feedback_items if item.session_id == session_id]
        satisfaction_feedback = [item for item in session_feedback if item.feedback_type == FeedbackType.USER_SATISFACTION]

        if not satisfaction_feedback:
            return 3.0  # Default neutral score

        ratings = [item.rating.value for item in satisfaction_feedback]
        return round(sum(ratings) / len(ratings), 2)

    def _calculate_technical_accuracy_score(self, session_id: str) -> float:
        """Calculate technical accuracy score for session."""
        session_feedback = [item for item in self.feedback_items if item.session_id == session_id]
        accuracy_feedback = [item for item in session_feedback if item.feedback_type == FeedbackType.TECHNICAL_CORRECTNESS]

        if not accuracy_feedback:
            return 3.0  # Default neutral score

        ratings = [item.rating.value for item in accuracy_feedback]
        return round(sum(ratings) / len(ratings), 2)

    def _calculate_flow_score(self, session_id: str, duration: float, num_turns: int) -> float:
        """Calculate conversation flow score."""
        # Simple heuristic based on conversation efficiency
        if duration == 0:
            return 0.0

        efficiency = num_turns / (duration / 60)  # Turns per minute
        flow_score = min(efficiency / 3, 1.0)  # Normalize to 0-1

        return round(flow_score, 2)

    def _get_performance_breakdown(self) -> Dict[str, float]:
        """Get detailed performance breakdown."""
        if not self.conversation_metrics:
            return {}

        return {
            "avg_engagement": round(sum(m.user_engagement_score for m in self.conversation_metrics) / len(self.conversation_metrics), 2),
            "avg_response_quality": round(sum(m.response_quality_score for m in self.conversation_metrics) / len(self.conversation_metrics), 2),
            "avg_technical_accuracy": round(sum(m.technical_accuracy for m in self.conversation_metrics) / len(self.conversation_metrics), 2),
            "avg_conversation_flow": round(sum(m.conversation_flow_score for m in self.conversation_metrics) / len(self.conversation_metrics), 2)
        }

    def _save_feedback_to_storage(self, feedback_items: List[FeedbackItem]) -> None:
        """Save feedback items to storage."""
        if not self.storage_path:
            return

        # Append to existing file or create new one
        try:
            with open(self.storage_path, 'a', encoding='utf-8') as f:
                for item in feedback_items:
                    feedback_data = {
                        "feedback_id": item.feedback_id,
                        "session_id": item.session_id,
                        "timestamp": item.timestamp.isoformat(),
                        "feedback_type": item.feedback_type.value,
                        "rating": item.rating.value,
                        "comment": item.comment,
                        "context": item.context,
                        "specific_issues": item.specific_issues,
                        "suggestions": item.suggestions
                    }
                    f.write(json.dumps(feedback_data, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to save feedback to storage: {e}")

    def _save_metrics_to_storage(self, metrics_list: List[ConversationMetrics]) -> None:
        """Save conversation metrics to storage."""
        if not self.storage_path:
            return

        metrics_path = self.storage_path.replace('.json', '_metrics.json')

        try:
            with open(metrics_path, 'a', encoding='utf-8') as f:
                for metrics in metrics_list:
                    metrics_data = {
                        "session_id": metrics.session_id,
                        "duration": metrics.duration,
                        "num_turns": metrics.num_turns,
                        "user_engagement_score": metrics.user_engagement_score,
                        "response_quality_score": metrics.response_quality_score,
                        "resolution_achieved": metrics.resolution_achieved,
                        "user_satisfaction": metrics.user_satisfaction,
                        "technical_accuracy": metrics.technical_accuracy,
                        "conversation_flow_score": metrics.conversation_flow_score,
                        "timestamp": datetime.now().isoformat()
                    }
                    f.write(json.dumps(metrics_data, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to save metrics to storage: {e}")

    def _initialize_feedback_processors(self) -> Dict[str, Callable]:
        """Initialize feedback processing functions."""
        return {
            "quality_analysis": self._analyze_response_quality,
            "sentiment_analysis": self._analyze_sentiment,
            "issue_categorization": self._categorize_issues
        }

    def _initialize_analytics_engine(self) -> Any:
        """Initialize analytics engine."""
        # Mock implementation
        class MockAnalyticsEngine:
            def process_feedback(self, feedback_data):
                return {"processed": True, "insights": []}

        return MockAnalyticsEngine()

    def _analyze_response_quality(self, feedback_item: FeedbackItem) -> Dict[str, Any]:
        """Analyze response quality from feedback."""
        return {
            "clarity": "good" if "不清楚" not in feedback_item.comment else "poor",
            "accuracy": "good" if "不准确" not in feedback_item.comment else "poor",
            "completeness": "good" if "不完整" not in feedback_item.comment else "poor"
        }

    def _analyze_sentiment(self, feedback_item: FeedbackItem) -> str:
        """Analyze sentiment from feedback comment."""
        comment = feedback_item.comment.lower()

        positive_words = ["好", "棒", "满意", "excellent", "good", "satisfied"]
        negative_words = ["差", "坏", "不满意", "poor", "bad", "unsatisfied"]

        if any(word in comment for word in positive_words):
            return "positive"
        elif any(word in comment for word in negative_words):
            return "negative"
        else:
            return "neutral"

    def _categorize_issues(self, feedback_item: FeedbackItem) -> List[str]:
        """Categorize issues from feedback."""
        categories = []
        issues = feedback_item.specific_issues

        if "不准确" in issues:
            categories.append("accuracy")
        if "不清楚" in issues:
            categories.append("clarity")
        if "不完整" in issues:
            categories.append("completeness")
        if "太技术" in issues:
            categories.append("complexity")

        return categories