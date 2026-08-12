"""
Query Processor for Diagnostic Conversations

Processes user queries in diagnostic conversations, extracting intent,
entities, and determining query types for appropriate response generation.
"""

from typing import Dict, Any, List, Optional, Tuple
import re
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Intents behind user queries."""
    GREETING = "greeting"
    FAULT_DIAGNOSIS = "fault_diagnosis"
    CAUSE_ANALYSIS = "cause_analysis"
    SEVERITY_ASSESSMENT = "severity_assessment"
    MAINTENANCE_GUIDANCE = "maintenance_guidance"
    TECHNICAL_EXPLANATION = "technical_explanation"
    MONITORING_ADVICE = "monitoring_advice"
    PREVENTION_STRATEGY = "prevention_strategy"
    GENERAL_QUESTION = "general_question"
    CONCLUSION = "conclusion"
    UNKNOWN = "unknown"


class QueryType(Enum):
    """Types of user queries."""
    IDENTIFICATION = "identification"
    EXPLANATION = "explanation"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    CONFIRMATION = "confirmation"
    CLARIFICATION = "clarification"
    FOLLOW_UP = "follow_up"


@dataclass
class ProcessedQuery:
    """Processed query with extracted information."""
    original_query: str
    normalized_query: str
    query_type: QueryType
    intent: QueryIntent
    entities: List[str]
    keywords: List[str]
    sentiment: str
    urgency: str
    confidence: float
    context_dependencies: List[str]
    suggested_actions: List[str]


class QueryProcessor:
    """
    Processes user queries to extract semantic information.

    This class analyzes user input to determine intent, extract entities,
    classify query types, and provide structured information for response generation.
    """

    def __init__(self):
        """Initialize query processor."""
        self.intent_patterns = self._initialize_intent_patterns()
        self.entity_patterns = self._initialize_entity_patterns()
        self.keyword_patterns = self._initialize_keyword_patterns()
        self.urgency_indicators = self._initialize_urgency_indicators()
        self.context_dependencies = self._initialize_context_dependencies()

    def process_query(self,
                     query: str,
                     conversation_context: Optional[Dict[str, Any]] = None) -> ProcessedQuery:
        """
        Process user query and extract semantic information.

        Args:
            query: User's input query
            conversation_context: Current conversation context

        Returns:
            Processed query with extracted information
        """
        # Normalize query
        normalized = self._normalize_query(query)

        # Determine intent
        intent = self._classify_intent(normalized)

        # Determine query type
        query_type = self._classify_query_type(normalized, intent)

        # Extract entities
        entities = self._extract_entities(normalized)

        # Extract keywords
        keywords = self._extract_keywords(normalized)

        # Analyze sentiment
        sentiment = self._analyze_sentiment(normalized)

        # Determine urgency
        urgency = self._determine_urgency(normalized, intent)

        # Calculate confidence
        confidence = self._calculate_confidence(normalized, intent, query_type)

        # Identify context dependencies
        context_deps = self._identify_context_dependencies(intent, entities)

        # Suggest actions
        suggested_actions = self._suggest_actions(intent, query_type, urgency)

        return ProcessedQuery(
            original_query=query,
            normalized_query=normalized,
            query_type=query_type,
            intent=intent,
            entities=entities,
            keywords=keywords,
            sentiment=sentiment,
            urgency=urgency,
            confidence=confidence,
            context_dependencies=context_deps,
            suggested_actions=suggested_actions
        )

    def extract_technical_terms(self, query: str) -> List[str]:
        """
        Extract technical terms from query.

        Args:
            query: Input query

        Returns:
            List of technical terms
        """
        technical_terms = []

        # Technical term patterns
        tech_patterns = [
            r'\b(?:轴承|齿轮|轴|联轴器|电机|泵|风机)\b',
            r'\b(?:振动|频率|幅值|加速度|速度|位移)\b',
            r'\b(?:FFT|频谱|时域|频域|包络|相位)\b',
            r'\b(?:内圈|外圈|滚动体|保持架)\b',
            r'\b(?:不对中|不平衡|松动|磨损|疲劳)\b'
        ]

        for pattern in tech_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            technical_terms.extend(matches)

        return list(set(technical_terms))  # Remove duplicates

    def detect_question_type(self, query: str) -> str:
        """
        Detect the type of question.

        Args:
            query: Input query

        Returns:
            Question type (what, how, why, when, where, yes/no)
        """
        query_lower = query.lower()

        if any(word in query_lower for word in ["什么", "是", "是什么", "what"]):
            return "what"
        elif any(word in query_lower for word in ["怎么", "如何", "how"]):
            return "how"
        elif any(word in query_lower for word in ["为什么", "为何", "why"]):
            return "why"
        elif any(word in query_lower for word in ["何时", "什么时候", "when"]):
            return "when"
        elif any(word in query_lower for word in ["哪里", "何处", "where"]):
            return "where"
        elif any(word in query_lower for word in ["是否", "会不会", "能不能", "can", "will", "is", "are"]):
            return "yes_no"
        else:
            return "general"

    def analyze_complexity(self, query: str) -> Dict[str, Any]:
        """
        Analyze query complexity.

        Args:
            query: Input query

        Returns:
            Complexity analysis
        """
        # Simple complexity metrics
        length_score = min(len(query) / 100, 1.0)  # Normalize to 0-1
        sentence_count = len(re.split(r'[.。!?？]', query))
        sentence_score = min(sentence_count / 5, 1.0)

        # Technical term count
        tech_terms = self.extract_technical_terms(query)
        tech_score = min(len(tech_terms) / 5, 1.0)

        # Question indicators
        question_score = 1.0 if re.search(r'[?？]', query) else 0.3

        # Overall complexity
        complexity_score = (length_score + sentence_score + tech_score + question_score) / 4

        return {
            "overall_score": complexity_score,
            "length_score": length_score,
            "sentence_score": sentence_score,
            "technical_score": tech_score,
            "question_score": question_score,
            "technical_terms": tech_terms,
            "complexity_level": self._classify_complexity_level(complexity_score)
        }

    def _normalize_query(self, query: str) -> str:
        """Normalize query text."""
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', query.strip())

        # Convert to lowercase for processing
        # Keep original case for the normalized field
        return normalized

    def _classify_intent(self, query: str) -> QueryIntent:
        """Classify the intent behind the query."""
        query_lower = query.lower()

        # Check for greeting
        if any(word in query_lower for word in ["你好", "您好", "hello", "hi"]):
            return QueryIntent.GREETING

        # Check for conclusion
        if any(word in query_lower for word in ["结束", "完成", "再见", "bye", "结束"]):
            return QueryIntent.CONCLUSION

        # Check for fault diagnosis
        if any(word in query_lower for word in ["故障", "问题", "诊断", "检测", "fault", "problem"]):
            return QueryIntent.FAULT_DIAGNOSIS

        # Check for cause analysis
        if any(word in query_lower for word in ["原因", "为什么", "为何", "cause", "why"]):
            return QueryIntent.CAUSE_ANALYSIS

        # Check for severity assessment
        if any(word in query_lower for word in ["严重", "程度", "级别", "severity", "level"]):
            return QueryIntent.SEVERITY_ASSESSMENT

        # Check for maintenance guidance
        if any(word in query_lower for word in ["维修", "维护", "保养", "处理", "maintenance", "repair"]):
            return QueryIntent.MAINTENANCE_GUIDANCE

        # Check for technical explanation
        if any(word in query_lower for word in ["解释", "说明", "原理", "技术", "explain", "technical"]):
            return QueryIntent.TECHNICAL_EXPLANATION

        # Check for monitoring advice
        if any(word in query_lower for word in ["监测", "监控", "观察", "monitor", "watch"]):
            return QueryIntent.MONITORING_ADVICE

        # Check for prevention strategy
        if any(word in query_lower for word in ["预防", "避免", "防止", "prevention", "avoid"]):
            return QueryIntent.PREVENTION_STRATEGY

        # Default to general question
        return QueryIntent.GENERAL_QUESTION

    def _classify_query_type(self, query: str, intent: QueryIntent) -> QueryType:
        """Classify the type of query."""
        query_lower = query.lower()

        # Check for comparison queries
        if any(word in query_lower for word in ["比较", "对比", "区别", "compare", "difference"]):
            return QueryType.COMPARISON

        # Check for confirmation queries
        if any(word in query_lower for word in ["是吗", "对吗", "确认", "confirm", "correct"]):
            return QueryType.CONFIRMATION

        # Check for clarification queries
        if any(word in query_lower for word in ["什么意思", "详细", "解释", "clarify", "detail"]):
            return QueryType.CLARIFICATION

        # Classify based on intent
        if intent in [QueryIntent.FAULT_DIAGNOSIS, QueryIntent.SEVERITY_ASSESSMENT]:
            return QueryType.IDENTIFICATION
        elif intent in [QueryIntent.TECHNICAL_EXPLANATION, QueryIntent.CAUSE_ANALYSIS]:
            return QueryType.EXPLANATION
        elif intent in [QueryIntent.MAINTENANCE_GUIDANCE, QueryIntent.MONITORING_ADVICE]:
            return QueryType.RECOMMENDATION
        else:
            return QueryType.GENERAL_QUESTION

    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from query."""
        entities = []

        # Component entities
        component_pattern = r'\b(?:轴承|齿轮|轴|联轴器|电机|泵|风机|压缩机|轴承座)\b'
        component_matches = re.findall(component_pattern, query, re.IGNORECASE)
        entities.extend([f"COMPONENT:{match}" for match in component_matches])

        # Fault type entities
        fault_pattern = r'\b(?:内圈|外圈|滚动体|保持架|不对中|不平衡|松动|磨损|疲劳|裂纹)\b'
        fault_matches = re.findall(fault_pattern, query, re.IGNORECASE)
        entities.extend([f"FAULT:{match}" for match in fault_matches])

        # Measurement entities
        measurement_pattern = r'\b(?:mm/s|g|RPM|Hz|mm|μm|dB)\b'
        measurement_matches = re.findall(measurement_pattern, query, re.IGNORECASE)
        entities.extend([f"MEASUREMENT:{match}" for match in measurement_matches])

        # Time entities
        time_pattern = r'\b(?:小时|天|周|月|立即|马上|urgent|immediate)\b'
        time_matches = re.findall(time_pattern, query, re.IGNORECASE)
        entities.extend([f"TIME:{match}" for match in time_matches])

        return entities

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query."""
        keywords = []

        # Extract important keywords
        important_words = [
            "故障", "诊断", "维修", "维护", "监测", "振动", "频率", "分析", "检查",
            "fault", "diagnosis", "maintenance", "monitor", "vibration", "frequency", "analysis", "check"
        ]

        for word in important_words:
            if word.lower() in query.lower():
                keywords.append(word)

        # Extract technical terms
        tech_terms = self.extract_technical_terms(query)
        keywords.extend(tech_terms)

        return list(set(keywords))  # Remove duplicates

    def _analyze_sentiment(self, query: str) -> str:
        """Analyze sentiment of query."""
        query_lower = query.lower()

        positive_words = ["好", "正常", "稳定", "improved", "good", "normal", "stable"]
        negative_words = ["坏", "异常", "严重", "危险", "bad", "abnormal", "severe", "dangerous"]
        urgent_words = ["紧急", "立即", "马上", "urgent", "immediate", "asap"]

        if any(word in query_lower for word in urgent_words):
            return "urgent"
        elif any(word in query_lower for word in negative_words):
            return "negative"
        elif any(word in query_lower for word in positive_words):
            return "positive"
        else:
            return "neutral"

    def _determine_urgency(self, query: str, intent: QueryIntent) -> str:
        """Determine urgency level of query."""
        query_lower = query.lower()

        urgent_indicators = ["紧急", "立即", "马上", "urgent", "immediate", "asap", "critical"]
        high_indicators = ["严重", "重要", "严重", "severe", "important", "high"]

        if any(word in query_lower for word in urgent_indicators):
            return "urgent"
        elif any(word in query_lower for word in high_indicators):
            return "high"
        elif intent in [QueryIntent.SEVERITY_ASSESSMENT, QueryIntent.MAINTENANCE_GUIDANCE]:
            return "medium"
        else:
            return "low"

    def _calculate_confidence(self,
                            query: str,
                            intent: QueryIntent,
                            query_type: QueryType) -> float:
        """Calculate confidence score for query classification."""
        base_confidence = 0.5

        # Increase confidence for clear intent matches
        intent_matches = {
            QueryIntent.FAULT_DIAGNOSIS: ["故障", "问题", "fault", "problem"],
            QueryIntent.CAUSE_ANALYSIS: ["原因", "为什么", "cause", "why"],
            QueryIntent.MAINTENANCE_GUIDANCE: ["维修", "维护", "maintenance", "repair"]
        }

        if intent in intent_matches:
            matches = sum(1 for pattern in intent_matches[intent] if pattern in query.lower())
            base_confidence += min(matches * 0.1, 0.3)

        # Increase confidence for clear query structure
        if re.search(r'[?？]', query):
            base_confidence += 0.1

        if len(query.split()) > 3:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _identify_context_dependencies(self, intent: QueryIntent, entities: List[str]) -> List[str]:
        """Identify what context information is needed."""
        dependencies = []

        if intent == QueryIntent.FAULT_DIAGNOSIS:
            dependencies.extend(["initial_diagnosis", "device_info", "measurement_data"])
        elif intent == QueryIntent.CAUSE_ANALYSIS:
            dependencies.extend(["fault_type", "operating_conditions", "maintenance_history"])
        elif intent == QueryIntent.SEVERITY_ASSESSMENT:
            dependencies.extend(["measurement_values", "standards", "thresholds"])
        elif intent == QueryIntent.MAINTENANCE_GUIDANCE:
            dependencies.extend(["fault_type", "severity", "device_info", "resources"])

        # Add entity-specific dependencies
        for entity in entities:
            if entity.startswith("FAULT:"):
                dependencies.append("fault_details")
            elif entity.startswith("MEASUREMENT:"):
                dependencies.append("measurement_context")
            elif entity.startswith("COMPONENT:"):
                dependencies.append("component_info")

        return list(set(dependencies))

    def _suggest_actions(self, intent: QueryIntent, query_type: QueryType, urgency: str) -> List[str]:
        """Suggest actions based on query analysis."""
        actions = []

        if urgency == "urgent":
            actions.append("prioritize_response")
            actions.append("provide_immediate_guidance")

        if intent == QueryIntent.FAULT_DIAGNOSIS:
            actions.extend(["provide_detailed_analysis", "request_additional_info"])
        elif intent == QueryIntent.MAINTENANCE_GUIDANCE:
            actions.extend(["provide_step_by_step_instructions", "check_resources"])
        elif intent == QueryIntent.CAUSE_ANALYSIS:
            actions.extend(["analyze_root_causes", "suggest_investigations"])

        if query_type == QueryType.CLARIFICATION:
            actions.append("provide_detailed_explanation")

        return actions

    def _classify_complexity_level(self, score: float) -> str:
        """Classify complexity level based on score."""
        if score < 0.3:
            return "simple"
        elif score < 0.6:
            return "moderate"
        elif score < 0.8:
            return "complex"
        else:
            return "very_complex"

    def _initialize_intent_patterns(self) -> Dict[QueryIntent, List[str]]:
        """Initialize intent recognition patterns."""
        return {
            QueryIntent.GREETING: ["你好", "您好", "hello", "hi", "早上好", "下午好"],
            QueryIntent.FAULT_DIAGNOSIS: ["故障", "问题", "诊断", "检测", "fault", "problem", "issue"],
            QueryIntent.CAUSE_ANALYSIS: ["原因", "为什么", "为何", "cause", "why", "reason"],
            QueryIntent.SEVERITY_ASSESSMENT: ["严重", "程度", "级别", "severity", "level", "grade"],
            QueryIntent.MAINTENANCE_GUIDANCE: ["维修", "维护", "保养", "处理", "maintenance", "repair", "fix"],
            QueryIntent.TECHNICAL_EXPLANATION: ["解释", "说明", "原理", "技术", "explain", "technical", "how"],
            QueryIntent.MONITORING_ADVICE: ["监测", "监控", "观察", "monitor", "watch", "check"],
            QueryIntent.PREVENTION_STRATEGY: ["预防", "避免", "防止", "prevention", "avoid", "prevent"],
            QueryIntent.CONCLUSION: ["结束", "完成", "再见", "bye", "结束", "finish"],
            QueryIntent.GENERAL_QUESTION: ["吗", "呢", "？", "?", "如何", "怎么样"]
        }

    def _initialize_entity_patterns(self) -> Dict[str, str]:
        """Initialize entity extraction patterns."""
        return {
            "component": r'\b(?:轴承|齿轮|轴|联轴器|电机|泵|风机)\b',
            "fault": r'\b(?:内圈|外圈|滚动体|保持架|不对中|不平衡|松动|磨损)\b',
            "measurement": r'\b(?:mm/s|g|RPM|Hz|mm|μm)\b',
            "time": r'\b(?:小时|天|周|月|立即|马上)\b'
        }

    def _initialize_keyword_patterns(self) -> List[str]:
        """Initialize keyword patterns."""
        return [
            "故障", "诊断", "维修", "维护", "监测", "振动", "频率", "分析", "检查",
            "处理", "原因", "原理", "方法", "建议", "方案", "措施"
        ]

    def _initialize_urgency_indicators(self) -> Dict[str, List[str]]:
        """Initialize urgency indicators."""
        return {
            "urgent": ["紧急", "立即", "马上", "urgent", "immediate", "asap"],
            "high": ["严重", "重要", "关键", "severe", "important", "critical"],
            "medium": ["一般", "中等", "normal", "medium"],
            "low": ["轻微", "简单", "minor", "simple", "low"]
        }

    def _initialize_context_dependencies(self) -> Dict[QueryIntent, List[str]]:
        """Initialize context dependencies for each intent."""
        return {
            QueryIntent.FAULT_DIAGNOSIS: ["device_info", "measurement_data", "operating_conditions"],
            QueryIntent.CAUSE_ANALYSIS: ["fault_type", "history", "environmental_factors"],
            QueryIntent.SEVERITY_ASSESSMENT: ["measurement_values", "standards", "comparisons"],
            QueryIntent.MAINTENANCE_GUIDANCE: ["fault_severity", "available_resources", "time_constraints"],
            QueryIntent.TECHNICAL_EXPLANATION: ["technical_details", "background_info"],
            QueryIntent.MONITORING_ADVICE: ["current_status", "monitoring_capabilities"],
            QueryIntent.PREVENTION_STRATEGY: ["failure_history", "best_practices"]
        }