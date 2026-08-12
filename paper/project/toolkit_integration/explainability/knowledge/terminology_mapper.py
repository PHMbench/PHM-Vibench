"""
Terminology Mapper for Fault Diagnosis Domain

Provides mapping between technical terms, standard terminology, and
explanations to ensure consistent and accurate communication.
"""

from typing import Dict, List, Tuple, Optional, Set
import json
import re
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TermCategory(Enum):
    """Categories of terminology."""
    FAULT_TYPE = "fault_type"
    SYMPTOM = "symptom"
    MEASUREMENT = "measurement"
    FREQUENCY = "frequency"
    MAINTENANCE = "maintenance"
    SEVERITY = "severity"
    COMPONENT = "component"


@dataclass
class TermMapping:
    """Mapping between different term representations."""
    standard_term: str
    synonyms: List[str]
    abbreviations: List[str]
    english_equivalent: str
    category: TermCategory
    definition: str
    context_notes: str
    related_terms: List[str]


class TerminologyMapper:
    """
    Maps terminology between different forms and contexts.

    This class ensures consistent use of technical terms by providing
    mappings between synonyms, abbreviations, and standard terminology
    used in mechanical fault diagnosis.
    """

    def __init__(self):
        """Initialize the terminology mapper."""
        self.terminology_db = self._initialize_terminology()
        self.reverse_index = self._build_reverse_index()
        self.context_patterns = self._initialize_context_patterns()

    def get_standard_term(self, term: str) -> Optional[str]:
        """
        Get the standard term for a given input.

        Args:
            term: Input term (could be abbreviation, synonym, etc.)

        Returns:
            Standard term if found, None otherwise
        """
        term_lower = term.lower().strip()

        # Direct match
        if term_lower in self.reverse_index:
            return self.reverse_index[term_lower]

        # Partial match for multi-word terms
        for pattern, standard in self.reverse_index.items():
            if term_lower in pattern or pattern in term_lower:
                return standard

        return None

    def get_term_info(self, term: str) -> Optional[TermMapping]:
        """
        Get detailed information about a term.

        Args:
            term: Input term

        Returns:
            TermMapping if found, None otherwise
        """
        standard_term = self.get_standard_term(term)
        if not standard_term:
            return None

        return self.terminology_db.get(standard_term)

    def expand_abbreviations(self, text: str) -> str:
        """
        Expand abbreviations in text.

        Args:
            text: Input text containing abbreviations

        Returns:
            Text with expanded abbreviations
        """
        words = re.findall(r'\b\w+\b', text)
        expanded_text = text

        for word in words:
            term_info = self.get_term_info(word)
            if term_info and word in term_info.abbreviations:
                expanded_text = re.sub(
                    rf'\b{re.escape(word)}\b',
                    term_info.standard_term,
                    expanded_text
                )

        return expanded_text

    def normalize_terminology(self, text: str) -> str:
        """
        Normalize terminology in text to standard forms.

        Args:
            text: Input text with varied terminology

        Returns:
            Text with normalized terminology
        """
        # First expand abbreviations
        normalized = self.expand_abbreviations(text)

        # Then replace synonyms with standard terms
        for standard_term, mapping in self.terminology_db.items():
            for synonym in mapping.synonyms:
                # Use word boundaries to avoid partial replacements
                pattern = r'\b' + re.escape(synonym) + r'\b'
                normalized = re.sub(pattern, standard_term, normalized, flags=re.IGNORECASE)

        return normalized

    def extract_technical_terms(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract technical terms from text.

        Args:
            text: Input text

        Returns:
            List of extracted terms with their information
        """
        extracted_terms = []

        for standard_term, mapping in self.terminology_db.items():
            # Check for standard term
            if standard_term.lower() in text.lower():
                extracted_terms.append({
                    "term": standard_term,
                    "type": "standard",
                    "category": mapping.category.value,
                    "definition": mapping.definition
                })
                continue

            # Check for synonyms
            for synonym in mapping.synonyms:
                if re.search(r'\b' + re.escape(synonym) + r'\b', text, re.IGNORECASE):
                    extracted_terms.append({
                        "term": synonym,
                        "standard_form": standard_term,
                        "type": "synonym",
                        "category": mapping.category.value,
                        "definition": mapping.definition
                    })
                    break

            # Check for abbreviations
            for abbrev in mapping.abbreviations:
                if re.search(r'\b' + re.escape(abbrev) + r'\b', text, re.IGNORECASE):
                    extracted_terms.append({
                        "term": abbrev,
                        "standard_form": standard_term,
                        "type": "abbreviation",
                        "category": mapping.category.value,
                        "definition": mapping.definition
                    })
                    break

        return extracted_terms

    def suggest_alternatives(self, term: str, category: Optional[TermCategory] = None) -> List[str]:
        """
        Suggest alternative terms for given input.

        Args:
            term: Input term
            category: Optional category filter

        Returns:
            List of alternative terms
        """
        term_info = self.get_term_info(term)
        if not term_info:
            return []

        alternatives = []

        # Add synonyms
        alternatives.extend(term_info.synonyms)

        # Add abbreviations
        alternatives.extend(term_info.abbreviations)

        # Add English equivalent if different
        if term_info.english_equivalent != term_info.standard_term:
            alternatives.append(term_info.english_equivalent)

        # Filter by category if specified
        if category:
            alternatives = [alt for alt in alternatives
                           if self.get_term_info(alt) and self.get_term_info(alt).category == category]

        return alternatives

    def get_related_terms(self, term: str) -> List[str]:
        """
        Get terms related to the given term.

        Args:
            term: Input term

        Returns:
            List of related terms
        """
        term_info = self.get_term_info(term)
        if not term_info:
            return []

        return term_info.related_terms

    def validate_terminology(self, text: str) -> Dict[str, Any]:
        """
        Validate terminology usage in text.

        Args:
            text: Text to validate

        Returns:
            Validation results with suggestions
        """
        validation_result = {
            "total_terms": 0,
            "standard_terms": 0,
            "non_standard_terms": [],
            "suggestions": [],
            "confidence_score": 0.0
        }

        extracted_terms = self.extract_technical_terms(text)
        validation_result["total_terms"] = len(extracted_terms)

        for term_info in extracted_terms:
            if term_info["type"] == "standard":
                validation_result["standard_terms"] += 1
            else:
                validation_result["non_standard_terms"].append({
                    "term": term_info["term"],
                    "type": term_info["type"],
                    "standard_form": term_info.get("standard_form", ""),
                    "suggestion": f"建议使用标准术语: {term_info.get('standard_form', '')}"
                })

        # Calculate confidence score
        if validation_result["total_terms"] > 0:
            validation_result["confidence_score"] = (
                validation_result["standard_terms"] / validation_result["total_terms"]
            )

        return validation_result

    def create_glossary(self, category: Optional[TermCategory] = None) -> Dict[str, str]:
        """
        Create a glossary of terms.

        Args:
            category: Optional category filter

        Returns:
            Dictionary of terms and their definitions
        """
        glossary = {}

        for standard_term, mapping in self.terminology_db.items():
            if category and mapping.category != category:
                continue

            glossary[standard_term] = mapping.definition

        return glossary

    def _initialize_terminology(self) -> Dict[str, TermMapping]:
        """Initialize terminology database."""
        return {
            # Fault types
            "内圈故障": TermMapping(
                standard_term="内圈故障",
                synonyms=["内圈损坏", "内圈磨损", "内圈裂纹"],
                abbreviations=["IF", "inner_race"],
                english_equivalent="inner race fault",
                category=TermCategory.FAULT_TYPE,
                definition="轴承内圈表面的疲劳、磨损或裂纹故障",
                context_notes="通常表现为1x基频的高频谐波",
                related_terms=["外圈故障", "滚动体故障", "保持架故障"]
            ),

            "外圈故障": TermMapping(
                standard_term="外圈故障",
                synonyms=["外圈损坏", "外圈磨损", "外圈裂纹"],
                abbreviations=["OF", "outer_race"],
                english_equivalent="outer race fault",
                category=TermCategory.FAULT_TYPE,
                definition="轴承外圈表面的疲劳、磨损或裂纹故障",
                context_notes="通常在固定载荷区产生明显特征",
                related_terms=["内圈故障", "滚动体故障", "轴承座松动"]
            ),

            "滚动体故障": TermMapping(
                standard_term="滚动体故障",
                synonyms=["钢球故障", "滚子故障", "滚动体损坏"],
                abbreviations=["BF", "ball_defect"],
                english_equivalent="ball defect",
                category=TermCategory.FAULT_TYPE,
                definition="轴承滚动体表面的疲劳、剥落或裂纹故障",
                context_notes="通常表现为随机冲击信号",
                related_terms=["内圈故障", "外圈故障", "保持架故障"]
            ),

            "不对中": TermMapping(
                standard_term="不对中",
                synonyms=["不对中", "偏心", "轴向偏移"],
                abbreviations=["MISALIGN"],
                english_equivalent="misalignment",
                category=TermCategory.FAULT_TYPE,
                definition="旋转轴系中心线不重合的状态",
                context_notes="通常产生1x和2x频率特征",
                related_terms=["不平衡", "轴承故障", "联轴器故障"]
            ),

            "不平衡": TermMapping(
                standard_term="不平衡",
                synonyms=["转子不平衡", "质量不平衡", "动不平衡"],
                abbreviations=["IMBAL"],
                english_equivalent="imbalance",
                category=TermCategory.FAULT_TYPE,
                definition="转子质量分布不均匀导致的状态",
                context_notes="主要表现为1x基频特征",
                related_terms=["不对中", "弯曲", "松动"]
            ),

            # Measurements
            "振动加速度": TermMapping(
                standard_term="振动加速度",
                synonyms=["加速度", "振动加速度", "加速度值"],
                abbreviations=["ACC"],
                english_equivalent="vibration acceleration",
                category=TermCategory.MEASUREMENT,
                definition="振动速度的变化率，单位m/s²",
                context_notes="对高频故障敏感，常用于轴承故障检测",
                related_terms=["振动速度", "振动位移", "加速度包络"]
            ),

            "振动速度": TermMapping(
                standard_term="振动速度",
                synonyms=["速度", "振动速度", "振动烈度"],
                abbreviations=["VEL"],
                english_equivalent="vibration velocity",
                category=TermCategory.MEASUREMENT,
                definition="振动位移的变化率，单位mm/s",
                context_notes="国际通用振动评估标准，适用于10-1000Hz范围",
                related_terms=["振动加速度", "振动位移", "RMS值"]
            ),

            "振动位移": TermMapping(
                standard_term="振动位移",
                synonyms=["位移", "振动位移", "振幅"],
                abbreviations=["DISP"],
                english_equivalent="vibration displacement",
                category=TermCategory.MEASUREMENT,
                definition="振动幅值，单位μm或mm",
                context_notes="适用于低频振动测量，如不平衡、不对中",
                related_terms=["振动加速度", "振动速度", "峰峰值"]
            ),

            # Frequencies
            "转频": TermMapping(
                standard_term="转频",
                synonyms=["旋转频率", "基频", "1倍频"],
                abbreviations=["1X", "F1", "RPM"],
                english_equivalent="rotational frequency",
                category=TermCategory.FREQUENCY,
                definition="轴的旋转频率，单位Hz",
                context_notes="所有频率分析的基础参考频率",
                related_terms=["2倍频", "3倍频", "固有频率"]
            ),

            "固有频率": TermMapping(
                standard_term="固有频率",
                synonyms=["自然频率", "共振频率", "特征频率"],
                abbreviations=["NF", "FN"],
                english_equivalent="natural frequency",
                category=TermCategory.FREQUENCY,
                definition="系统固有的振动频率，与系统参数有关",
                context_notes="避免工作频率接近固有频率以防共振",
                related_terms=["转频", "临界转速", "共振"]
            ),

            "特征频率": TermMapping(
                standard_term="特征频率",
                synonyms=["故障频率", "诊断频率", "标识频率"],
                abbreviations=["CF", "FF"],
                english_equivalent="characteristic frequency",
                category=TermCategory.FREQUENCY,
                definition="特定故障类型在频域上的表现形式",
                context_notes="故障诊断的重要依据",
                related_terms=["故障模式", "频谱分析", "故障特征"]
            ),

            # Symptoms
            "冲击": TermMapping(
                standard_term="冲击",
                synonyms=["撞击", "脉冲", "瞬态振动"],
                abbreviations=["IMPACT"],
                english_equivalent="impact",
                category=TermCategory.SYMPTOM,
                definition="短时、高幅值的振动事件",
                context_notes="通常表明存在松动或故障",
                related_terms=["包络", "峭度", "脉冲指标"]
            ),

            "调制": TermMapping(
                standard_term="调制",
                synonyms=["频率调制", "幅值调制", "边带"],
                abbreviations=["MOD"],
                english_equivalent="modulation",
                category=TermCategory.SYMPTOM,
                definition="一个信号的特征随另一个信号变化的现象",
                context_notes="齿轮故障和不对中的典型特征",
                related_terms=["边带", "包络谱", "解调"]
            ),

            # Maintenance
            "预测性维护": TermMapping(
                standard_term="预测性维护",
                synonyms=["预见性维护", "状态维护", "智能维护"],
                abbreviations=["PdM", "PM"],
                english_equivalent="predictive maintenance",
                category=TermCategory.MAINTENANCE,
                definition="基于设备状态监测的维护策略",
                context_notes="通过状态监测实现故障早期预警",
                related_terms=["状态监测", "故障诊断", "预防性维护"]
            ),

            "状态监测": TermMapping(
                standard_term="状态监测",
                synonyms=["设备监测", "运行监测", "健康监测"],
                abbreviations=["CM", "CONDMON"],
                english_equivalent="condition monitoring",
                category=TermCategory.MAINTENANCE,
                definition="持续或定期监测设备运行状态的技术",
                context_notes="预测性维护的基础",
                related_terms=["预测性维护", "振动分析", "油液分析"]
            )
        }

    def _build_reverse_index(self) -> Dict[str, str]:
        """Build reverse lookup index."""
        reverse_index = {}

        for standard_term, mapping in self.terminology_db.items():
            # Add standard term
            reverse_index[standard_term.lower()] = standard_term

            # Add synonyms
            for synonym in mapping.synonyms:
                reverse_index[synonym.lower()] = standard_term

            # Add abbreviations
            for abbrev in mapping.abbreviations:
                reverse_index[abbrev.lower()] = standard_term

            # Add English equivalent
            reverse_index[mapping.english_equivalent.lower()] = standard_term

        return reverse_index

    def _initialize_context_patterns(self) -> Dict[str, List[str]]:
        """Initialize context-specific terminology patterns."""
        return {
            "formal_report": [
                "内圈故障", "外圈故障", "滚动体故障", "不对中", "不平衡",
                "振动加速度", "振动速度", "特征频率", "状态监测", "预测性维护"
            ],
            "technical_discussion": [
                "转频", "固有频率", "冲击", "调制", "包络",
                "频谱", "时域", "相位", "共振"
            ],
            "maintenance_plan": [
                "预防性维护", "纠正性维护", "紧急维修", "计划维修",
                "备件", "维修窗口", "停机时间"
            ]
        }

    def export_terminology(self, format: str = "json") -> str:
        """Export terminology database."""
        export_data = {
            "terminology": {
                term: {
                    "standard_term": mapping.standard_term,
                    "synonyms": mapping.synonyms,
                    "abbreviations": mapping.abbreviations,
                    "english_equivalent": mapping.english_equivalent,
                    "category": mapping.category.value,
                    "definition": mapping.definition,
                    "context_notes": mapping.context_notes,
                    "related_terms": mapping.related_terms
                }
                for term, mapping in self.terminology_db.items()
            },
            "context_patterns": self.context_patterns
        }

        if format == "json":
            return json.dumps(export_data, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def add_term(self, term_mapping: TermMapping) -> None:
        """
        Add a new term to the terminology database.

        Args:
            term_mapping: New term mapping to add
        """
        self.terminology_db[term_mapping.standard_term] = term_mapping
        self._update_reverse_index()

    def _update_reverse_index(self) -> None:
        """Update reverse index after adding terms."""
        self.reverse_index = self._build_reverse_index()