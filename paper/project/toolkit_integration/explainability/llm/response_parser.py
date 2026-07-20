"""
Response Parser for LLM-Enhanced Explainability

Parses and structures LLM responses into standardized formats for
mechanical fault diagnosis explanations.
"""

import re
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ResponseType(Enum):
    """Types of LLM responses."""
    COMPREHENSIVE = "comprehensive"
    CONVERSATIONAL = "conversational"
    FOCUSED = "focused"
    ERROR = "error"


@dataclass
class ParsedExplanation:
    """Structured explanation from LLM response."""
    title: str
    main_conclusion: str
    fault_mechanism: str
    technical_evidence: str
    maintenance_suggestions: str
    risk_assessment: str
    prevention_measures: str
    confidence_level: str
    additional_info: Dict[str, Any]


@dataclass
class ParsedMaintenance:
    """Structured maintenance recommendations."""
    priority_level: str
    immediate_actions: List[str]
    planned_actions: List[str]
    resource_requirements: Dict[str, Any]
    timeline_estimate: str
    risk_mitigation: List[str]
    verification_methods: List[str]


@dataclass
class ParsedRisk:
    """Structured risk assessment."""
    severity_level: str
    probability_level: str
    impact_assessment: str
    affected_components: List[str]
    failure_modes: List[str]
    monitoring_recommendations: List[str]
    emergency_actions: List[str]


class ResponseParser:
    """
    Parses LLM responses into structured, domain-specific formats.

    This class handles the extraction and organization of information from
    LLM-generated text, converting natural language explanations into
    structured data that can be used programmatically.
    """

    def __init__(self):
        """Initialize response parser with domain-specific patterns."""
        self.section_patterns = self._initialize_section_patterns()
        self.severity_patterns = self._initialize_severity_patterns()
        self.priority_patterns = self._initialize_priority_patterns()
        self.confidence_patterns = self._initialize_confidence_patterns()

    def parse_response(self, response: str, response_type: ResponseType = ResponseType.COMPREHENSIVE) -> Dict[str, Any]:
        """
        Parse LLM response into structured format.

        Args:
            response: Raw LLM response text
            response_type: Type of response to parse

        Returns:
            Structured dictionary with parsed content
        """
        if response_type == ResponseType.COMPREHENSIVE:
            return self._parse_comprehensive_response(response)
        elif response_type == ResponseType.CONVERSATIONAL:
            return self._parse_conversational_response(response)
        elif response_type == ResponseType.FOCUSED:
            return self._parse_focused_response(response)
        else:
            return self._parse_error_response(response)

    def parse_comprehensive_explanation(self, response: str) -> ParsedExplanation:
        """
        Parse comprehensive fault diagnosis explanation.

        Args:
            response: LLM response text

        Returns:
            Structured explanation object
        """
        sections = self._extract_sections(response)

        return ParsedExplanation(
            title=self._extract_title(response),
            main_conclusion=self._clean_text(sections.get("故障诊断结论", sections.get("diagnosis_conclusion", ""))),
            fault_mechanism=self._clean_text(sections.get("故障机理分析", sections.get("fault_mechanism", ""))),
            technical_evidence=self._clean_text(sections.get("技术证据支持", sections.get("technical_evidence", ""))),
            maintenance_suggestions=self._clean_text(sections.get("工程建议", sections.get("maintenance_suggestions", ""))),
            risk_assessment=self._clean_text(sections.get("风险评估", sections.get("risk_assessment", ""))),
            prevention_measures=self._clean_text(sections.get("预防措施", sections.get("prevention_measures", ""))),
            confidence_level=self._extract_confidence_level(response),
            additional_info=self._extract_additional_info(response)
        )

    def parse_maintenance_recommendations(self, response: str) -> ParsedMaintenance:
        """
        Parse maintenance recommendations from response.

        Args:
            response: LLM response text

        Returns:
            Structured maintenance object
        """
        sections = self._extract_sections(response)

        return ParsedMaintenance(
            priority_level=self._extract_priority_level(response),
            immediate_actions=self._extract_action_items(sections.get("immediate_actions", "")),
            planned_actions=self._extract_action_items(sections.get("planned_actions", "")),
            resource_requirements=self._extract_resource_requirements(sections.get("resources", "")),
            timeline_estimate=self._extract_timeline(response),
            risk_mitigation=self._extract_action_items(sections.get("risk_mitigation", "")),
            verification_methods=self._extract_action_items(sections.get("verification", ""))
        )

    def parse_risk_assessment(self, response: str) -> ParsedRisk:
        """
        Parse risk assessment from response.

        Args:
            response: LLM response text

        Returns:
            Structured risk object
        """
        sections = self._extract_sections(response)

        return ParsedRisk(
            severity_level=self._extract_severity_level(response),
            probability_level=self._extract_probability_level(response),
            impact_assessment=self._clean_text(sections.get("impact_assessment", "")),
            affected_components=self._extract_component_list(sections.get("affected_components", "")),
            failure_modes=self._extract_failure_modes(sections.get("failure_modes", "")),
            monitoring_recommendations=self._extract_action_items(sections.get("monitoring", "")),
            emergency_actions=self._extract_action_items(sections.get("emergency_actions", ""))
        )

    def parse_conversation_response(self, response: str) -> Dict[str, Any]:
        """
        Parse conversational response.

        Args:
            response: LLM response text

        Returns:
            Structured conversation response
        """
        return {
            "response_type": "conversational",
            "main_answer": self._extract_main_answer(response),
            "technical_explanation": self._extract_technical_explanation(response),
            "practical_advice": self._extract_practical_advice(response),
            "follow_up_questions": self._extract_follow_up_questions(response),
            "confidence_indicators": self._extract_confidence_indicators(response)
        }

    def _parse_comprehensive_response(self, response: str) -> Dict[str, Any]:
        """Parse comprehensive response format."""
        try:
            explanation = self.parse_comprehensive_explanation(response)
            return {
                "response_type": "comprehensive",
                "status": "success",
                "explanation": {
                    "title": explanation.title,
                    "main_conclusion": explanation.main_conclusion,
                    "fault_mechanism": explanation.fault_mechanism,
                    "technical_evidence": explanation.technical_evidence,
                    "maintenance_suggestions": explanation.maintenance_suggestions,
                    "risk_assessment": explanation.risk_assessment,
                    "prevention_measures": explanation.prevention_measures,
                    "confidence_level": explanation.confidence_level
                },
                "additional_info": explanation.additional_info
            }
        except Exception as e:
            logger.error(f"Failed to parse comprehensive response: {e}")
            return self._create_error_response(str(e), response)

    def _parse_conversational_response(self, response: str) -> Dict[str, Any]:
        """Parse conversational response format."""
        try:
            return self.parse_conversation_response(response)
        except Exception as e:
            logger.error(f"Failed to parse conversational response: {e}")
            return self._create_error_response(str(e), response)

    def _parse_focused_response(self, response: str) -> Dict[str, Any]:
        """Parse focused response format."""
        try:
            # Try to determine focus area
            focus_area = self._detect_focus_area(response)

            if focus_area == "maintenance":
                maintenance = self.parse_maintenance_recommendations(response)
                return {
                    "response_type": "focused",
                    "focus_area": "maintenance",
                    "status": "success",
                    "content": {
                        "priority_level": maintenance.priority_level,
                        "recommendations": {
                            "immediate": maintenance.immediate_actions,
                            "planned": maintenance.planned_actions
                        },
                        "resources": maintenance.resource_requirements,
                        "timeline": maintenance.timeline_estimate,
                        "risk_mitigation": maintenance.risk_mitigation,
                        "verification": maintenance.verification_methods
                    }
                }
            elif focus_area == "risk":
                risk = self.parse_risk_assessment(response)
                return {
                    "response_type": "focused",
                    "focus_area": "risk_assessment",
                    "status": "success",
                    "content": {
                        "severity_level": risk.severity_level,
                        "probability_level": risk.probability_level,
                        "impact_assessment": risk.impact_assessment,
                        "affected_components": risk.affected_components,
                        "failure_modes": risk.failure_modes,
                        "monitoring": risk.monitoring_recommendations,
                        "emergency_actions": risk.emergency_actions
                    }
                }
            else:
                # Generic focused response
                return {
                    "response_type": "focused",
                    "focus_area": focus_area,
                    "status": "success",
                    "content": {
                        "main_points": self._extract_main_points(response),
                        "detailed_explanation": self._clean_text(response)
                    }
                }

        except Exception as e:
            logger.error(f"Failed to parse focused response: {e}")
            return self._create_error_response(str(e), response)

    def _parse_error_response(self, response: str) -> Dict[str, Any]:
        """Parse error response."""
        return {
            "response_type": "error",
            "status": "error",
            "error_message": "Failed to parse LLM response",
            "raw_response": response[:500]  # Limit to first 500 chars
        }

    def _initialize_section_patterns(self) -> Dict[str, str]:
        """Initialize section header patterns."""
        return {
            "zh": {
                "fault_conclusion": r"[#]?[\s]*故障诊断结论[\s]*[:：]?\s*",
                "fault_mechanism": r"[#]?[\s]*故障机理分析[\s]*[:：]?\s*",
                "technical_evidence": r"[#]?[\s]*技术证据支持[\s]*[:：]?\s*",
                "maintenance_suggestions": r"[#]?[\s]*工程建议[\s]*[:：]?\s*",
                "risk_assessment": r"[#]?[\s]*风险评估[\s]*[:：]?\s*",
                "prevention_measures": r"[#]?[\s]*预防措施[\s]*[:：]?\s*"
            },
            "en": {
                "diagnosis_conclusion": r"[#]?[\s]*diagnosis conclusion[s]?[\s]*[:：]?\s*",
                "fault_mechanism": r"[#]?[\s]*fault mechanism[s]?[\s]*[:：]?\s*",
                "technical_evidence": r"[#]?[\s]*technical evidence[\s]*[:：]?\s*",
                "maintenance_suggestions": r"[#]?[\s]*maintenance suggestion[s]?[\s]*[:：]?\s*",
                "risk_assessment": r"[#]?[\s]*risk assessment[s]?[\s]*[:：]?\s*",
                "prevention_measures": r"[#]?[\s]*prevention measure[s]?[\s]*[:：]?\s*"
            }
        }

    def _initialize_severity_patterns(self) -> Dict[str, List[str]]:
        """Initialize severity level patterns."""
        return {
            "high": ["严重", "紧急", "高危", "critical", "severe", "high", "urgent"],
            "medium": ["中等", "一般", "moderate", "medium", "normal"],
            "low": ["轻微", "较低", "minor", "low", "slight"]
        }

    def _initialize_priority_patterns(self) -> Dict[str, List[str]]:
        """Initialize priority level patterns."""
        return {
            "immediate": ["立即", "紧急", "immediate", "urgent", "asap"],
            "high": ["高优先级", "尽快", "high priority", "soon"],
            "medium": ["中等优先级", "计划", "medium priority", "planned"],
            "low": ["低优先级", "后续", "low priority", "later"]
        }

    def _initialize_confidence_patterns(self) -> Dict[str, List[str]]:
        """Initialize confidence level patterns."""
        return {
            "high": ["很高", "确定", "high confidence", "certain", "definitive"],
            "medium": ["中等", "可能", "medium confidence", "likely", "probable"],
            "low": ["较低", "不确定", "low confidence", "uncertain", "possible"]
        }

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections based on headers."""
        sections = {}

        # Try Chinese patterns first
        for lang_patterns in [self.section_patterns["zh"], self.section_patterns["en"]]:
            for section_name, pattern in lang_patterns.items():
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    start = match.end()
                    # Find next section header
                    next_section_match = re.search(r"[#]?[\s]*[A-Za-z\u4e00-\u9fff]+[\s]*[:：]", text[start:])
                    if next_section_match:
                        end = start + next_section_match.start()
                        sections[section_name] = text[start:end].strip()
                    else:
                        sections[section_name] = text[start:].strip()
                    break

        return sections

    def _extract_title(self, text: str) -> str:
        """Extract title from response."""
        # Look for first line or first # header
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                return line[:100]  # Limit title length
            elif line.startswith('#'):
                return re.sub(r'^#+\s*', '', line)[:100]

        return "故障诊断分析"

    def _clean_text(self, text: str) -> str:
        """Clean and format text."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove markdown formatting
        text = re.sub(r'[#*_`\[\]]+', '', text)

        return text.strip()

    def _extract_confidence_level(self, text: str) -> str:
        """Extract confidence level from text."""
        text_lower = text.lower()

        for level, patterns in self.confidence_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return level

        return "medium"  # Default

    def _extract_priority_level(self, text: str) -> str:
        """Extract priority level from text."""
        text_lower = text.lower()

        for level, patterns in self.priority_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return level

        return "medium"  # Default

    def _extract_severity_level(self, text: str) -> str:
        """Extract severity level from text."""
        text_lower = text.lower()

        for level, patterns in self.severity_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return level

        return "medium"  # Default

    def _extract_probability_level(self, text: str) -> str:
        """Extract probability level from text."""
        text_lower = text.lower()

        if any(p in text_lower for p in ["很高", "high probability", "very likely"]):
            return "high"
        elif any(p in text_lower for p in ["很低", "low probability", "unlikely"]):
            return "low"
        else:
            return "medium"

    def _extract_action_items(self, text: str) -> List[str]:
        """Extract action items from text."""
        if not text:
            return []

        # Look for numbered lists, bullet points, or action verbs
        actions = []

        # Split by common separators
        separators = r'[\n;；•\-\d+\.\s]'
        items = re.split(separators, text)

        for item in items:
            item = item.strip()
            if len(item) > 10:  # Filter out very short items
                # Check if it starts with action verb or contains key phrases
                if any(verb in item.lower() for verb in
                      ["检查", "维修", "更换", "监测", "check", "repair", "replace", "monitor"]):
                    actions.append(item)

        return actions[:10]  # Limit to 10 actions

    def _extract_resource_requirements(self, text: str) -> Dict[str, Any]:
        """Extract resource requirements from text."""
        resources = {
            "personnel": [],
            "tools": [],
            "parts": [],
            "time_estimate": "",
            "cost_estimate": ""
        }

        if not text:
            return resources

        # Simple keyword-based extraction
        text_lower = text.lower()

        # Extract personnel requirements
        if any(word in text_lower for word in ["工程师", "技术员", "engineer", "technician"]):
            resources["personnel"] = ["技术工程师"]

        # Extract tools
        if any(word in text_lower for word in ["扳手", "检测仪", "wrench", "tester"]):
            resources["tools"] = ["检测工具"]

        # Extract parts
        if any(word in text_lower for word in ["轴承", "零件", "bearing", "parts"]):
            resources["parts"] = ["备件"]

        return resources

    def _extract_timeline(self, text: str) -> str:
        """Extract timeline estimate from text."""
        if not text:
            return ""

        # Look for time expressions
        time_patterns = [
            r'(\d+)\s*[天日月小时]',  # Chinese time
            r'(\d+)\s*(days?|hours?|weeks?)',  # English time
        ]

        for pattern in time_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(0)

        return "待定"

    def _extract_component_list(self, text: str) -> List[str]:
        """Extract component list from text."""
        if not text:
            return []

        # Look for component names
        component_patterns = [
            r'轴承[^\s，,。.]*',
            r'齿轮[^\s，,。.]*',
            r'轴[^\s，,。.]*',
            r'bearing[s]?',
            r'gear[s]?',
            r'shaft[s]?'
        ]

        components = []
        for pattern in component_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            components.extend(matches)

        return list(set(components))  # Remove duplicates

    def _extract_failure_modes(self, text: str) -> List[str]:
        """Extract failure modes from text."""
        if not text:
            return []

        failure_keywords = [
            "磨损", "疲劳", "腐蚀", "断裂", "松动",
            "wear", "fatigue", "corrosion", "fracture", "looseness"
        ]

        failure_modes = []
        text_lower = text.lower()

        for keyword in failure_keywords:
            if keyword in text_lower:
                failure_modes.append(keyword)

        return failure_modes

    def _detect_focus_area(self, text: str) -> str:
        """Detect the focus area of a response."""
        text_lower = text.lower()

        if any(word in text_lower for word in ["维修", "维护", "保养", "maintenance", "repair"]):
            return "maintenance"
        elif any(word in text_lower for word in ["风险", "危险", "severity", "risk", "danger"]):
            return "risk"
        elif any(word in text_lower for word in ["机理", "原理", "mechanism", "principle"]):
            return "mechanism"
        else:
            return "general"

    def _extract_main_points(self, text: str) -> List[str]:
        """Extract main points from text."""
        # Split text into sentences and filter important ones
        sentences = re.split(r'[.。!！?？]', text)

        main_points = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Filter out very short sentences
                # Check if sentence contains important keywords
                if any(keyword in sentence.lower() for keyword in
                      ["故障", "问题", "建议", "fault", "problem", "recommend"]):
                    main_points.append(sentence)

        return main_points[:5]  # Limit to 5 main points

    def _extract_main_answer(self, text: str) -> str:
        """Extract main answer from conversational response."""
        lines = text.split('\n')
        for line in lines:
            if line.strip() and not line.startswith('#'):
                return self._clean_text(line)
        return self._clean_text(text[:200])

    def _extract_technical_explanation(self, text: str) -> str:
        """Extract technical explanation part."""
        # Look for technical terms and extract surrounding context
        technical_keywords = ["频率", "振幅", "频谱", "frequency", "amplitude", "spectrum"]

        for keyword in technical_keywords:
            if keyword in text.lower():
                # Extract sentence containing the keyword
                sentences = re.split(r'[.。!！?？]', text)
                for sentence in sentences:
                    if keyword in sentence.lower():
                        return self._clean_text(sentence)

        return ""

    def _extract_practical_advice(self, text: str) -> List[str]:
        """Extract practical advice from response."""
        advice_patterns = [
            r"[应该|需要|建议|should|need to|recommend]\s*[^.。!！?？]*"
        ]

        advice_list = []
        for pattern in advice_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            advice_list.extend(matches)

        return [self._clean_text(advice) for advice in advice_list[:3]]

    def _extract_follow_up_questions(self, text: str) -> List[str]:
        """Extract follow-up questions from response."""
        question_patterns = [
            r"[?？]\s*$",
            r"[是否|会不会|is|are|do|does]\s*[^.。!！?？]*[?？]?"
        ]

        questions = []
        for pattern in question_patterns:
            matches = re.findall(pattern, text)
            questions.extend(matches)

        return [self._clean_text(q) for q in questions if q.strip()]

    def _extract_confidence_indicators(self, text: str) -> Dict[str, Any]:
        """Extract confidence indicators from response."""
        text_lower = text.lower()

        indicators = {
            "certainty_expressions": [],
            "uncertainty_expressions": [],
            "overall_confidence": "medium"
        }

        certainty_words = ["确定", "肯定", "certain", "definite", "clear"]
        uncertainty_words = ["可能", "或许", "maybe", "possibly", "uncertain"]

        for word in certainty_words:
            if word in text_lower:
                indicators["certainty_expressions"].append(word)

        for word in uncertainty_words:
            if word in text_lower:
                indicators["uncertainty_expressions"].append(word)

        # Determine overall confidence
        if len(indicators["certainty_expressions"]) > len(indicators["uncertainty_expressions"]):
            indicators["overall_confidence"] = "high"
        elif len(indicators["uncertainty_expressions"]) > len(indicators["certainty_expressions"]):
            indicators["overall_confidence"] = "low"

        return indicators

    def _extract_additional_info(self, text: str) -> Dict[str, Any]:
        """Extract additional structured information."""
        additional = {
            "keywords": self._extract_keywords(text),
            "entities": self._extract_entities(text),
            "sentiment": self._analyze_sentiment(text)
        }

        return additional

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction based on domain terms
        domain_keywords = [
            "轴承", "齿轮", "轴", "振动", "频率", "故障", "诊断",
            "bearing", "gear", "shaft", "vibration", "frequency", "fault", "diagnosis"
        ]

        found_keywords = []
        text_lower = text.lower()

        for keyword in domain_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)

        return list(set(found_keywords))

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        # Simple entity extraction (would use NLP library in production)
        entities = []

        # Look for capitalized words or specific patterns
        entity_patterns = [
            r'[A-Z][a-z]+\s+(?:设备|型号|model|type)',
            r'\b[A-Z]{2,}\b'  # Acronyms
        ]

        for pattern in entity_patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)

        return entities

    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of response."""
        positive_words = ["正常", "良好", "稳定", "normal", "good", "stable"]
        negative_words = ["异常", "严重", "危险", "abnormal", "severe", "dangerous"]

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def _create_error_response(self, error_message: str, raw_response: str) -> Dict[str, Any]:
        """Create error response structure."""
        return {
            "response_type": "error",
            "status": "error",
            "error_message": error_message,
            "raw_response": raw_response[:1000] if raw_response else "",
            "suggestions": [
                "Check if the LLM response is well-structured",
                "Verify the response contains expected sections",
                "Consider using a simpler response format"
            ]
        }