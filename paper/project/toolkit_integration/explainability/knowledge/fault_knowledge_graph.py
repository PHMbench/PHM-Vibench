"""
Fault Knowledge Graph for Enhanced Explanations

Provides domain knowledge about mechanical faults, their characteristics,
and relationships to enhance LLM-generated explanations with structured knowledge.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
import json
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FaultType(Enum):
    """Types of mechanical faults."""
    INNER_RACE_FAULT = "inner_race_fault"
    OUTER_RACE_FAULT = "outer_race_fault"
    BALL_DEFECT = "ball_defect"
    CAGE_DAMAGE = "cage_damage"
    MISALIGNMENT = "misalignment"
    IMBALANCE = "imbalance"
    LOOSENESS = "looseness"
    GEAR_TOOTH_FAULT = "gear_tooth_fault"
    SHAFT_BEND = "shaft_bend"
    BEARING_WEAR = "bearing_wear"


class SeverityLevel(Enum):
    """Fault severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FaultPattern:
    """Fault pattern with characteristic frequencies and symptoms."""
    fault_type: FaultType
    characteristic_frequencies: Dict[str, float]
    frequency_multiples: List[float]
    time_domain_features: Dict[str, str]
    frequency_domain_features: Dict[str, str]
    typical_amplitudes: Dict[str, float]
    symptoms: List[str]
    common_causes: List[str]


@dataclass
class MaintenanceAction:
    """Maintenance action recommendation."""
    action_id: str
    description: str
    priority: str
    required_tools: List[str]
    required_parts: List[str]
    time_estimate: str
    skill_level: str
    safety_precautions: List[str]


@dataclass
class FaultRelationship:
    """Relationship between faults."""
    source_fault: FaultType
    target_fault: FaultType
    relationship_type: str  # "causes", "accompanies", "progresses_to"
    probability: float
    time_frame: Optional[str]


class FaultKnowledgeGraph:
    """
    Knowledge graph for mechanical fault diagnosis.

    This class provides structured domain knowledge about mechanical faults,
    including characteristic frequencies, symptom patterns, maintenance actions,
    and fault relationships.
    """

    def __init__(self):
        """Initialize the fault knowledge graph."""
        self.fault_patterns = self._initialize_fault_patterns()
        self.maintenance_actions = self._initialize_maintenance_actions()
        self.fault_relationships = self._initialize_fault_relationships()
        self.component_frequencies = self._initialize_component_frequencies()
        self.diagnostic_rules = self._initialize_diagnostic_rules()

    def get_fault_pattern(self, fault_type: FaultType) -> Optional[FaultPattern]:
        """
        Get fault pattern for specified fault type.

        Args:
            fault_type: Type of fault

        Returns:
            Fault pattern if found, None otherwise
        """
        return self.fault_patterns.get(fault_type)

    def get_maintenance_actions(self, fault_type: FaultType, severity: SeverityLevel) -> List[MaintenanceAction]:
        """
        Get maintenance actions for specified fault and severity.

        Args:
            fault_type: Type of fault
            severity: Severity level

        Returns:
            List of maintenance actions
        """
        actions = []

        # Filter actions by fault type and severity
        for action in self.maintenance_actions:
            if fault_type.value in action.action_id:
                # Adjust priority based on severity
                if severity == SeverityLevel.CRITICAL:
                    if action.priority in ["medium", "high"]:
                        # Create high-priority version
                        high_priority_action = MaintenanceAction(
                            action_id=f"{action.action_id}_urgent",
                            description=f"[紧急] {action.description}",
                            priority="immediate",
                            required_tools=action.required_tools,
                            required_parts=action.required_parts,
                            time_estimate=action.time_estimate,
                            skill_level=action.skill_level,
                            safety_precautions=action.safety_precautions + ["需要立即停机检查"]
                        )
                        actions.append(high_priority_action)
                elif severity in [SeverityLevel.HIGH, SeverityLevel.MEDIUM]:
                    if action.priority in ["medium", "low"]:
                        # Upgrade priority
                        upgraded_action = MaintenanceAction(
                            action_id=f"{action.action_id}_priority",
                            description=action.description,
                            priority="high",
                            required_tools=action.required_tools,
                            required_parts=action.required_parts,
                            time_estimate=action.time_estimate,
                            skill_level=action.skill_level,
                            safety_precautions=action.safety_precautions
                        )
                        actions.append(upgraded_action)
                    else:
                        actions.append(action)

        return actions

    def get_related_faults(self, fault_type: FaultType) -> List[FaultRelationship]:
        """
        Get faults related to the specified fault.

        Args:
            fault_type: Type of fault

        Returns:
            List of fault relationships
        """
        return [
            rel for rel in self.fault_relationships
            if rel.source_fault == fault_type or rel.target_fault == fault_type
        ]

    def get_characteristic_frequencies(self,
                                     fault_type: FaultType,
                                     shaft_speed: float,
                                     gear_ratio: float = 1.0) -> Dict[str, float]:
        """
        Calculate characteristic frequencies for the specified fault.

        Args:
            fault_type: Type of fault
            shaft_speed: Shaft speed in RPM
            gear_ratio: Gear ratio (if applicable)

        Returns:
            Dictionary of characteristic frequencies in Hz
        """
        pattern = self.get_fault_pattern(fault_type)
        if not pattern:
            return {}

        shaft_freq = shaft_speed / 60.0  # Convert RPM to Hz

        frequencies = {}
        for freq_name, multiplier in pattern.characteristic_frequencies.items():
            if freq_name == "shaft":
                frequencies[f"shaft_frequency"] = shaft_freq
            elif freq_name == "gear_mesh":
                frequencies[f"gear_mesh_frequency"] = shaft_freq * gear_ratio
            else:
                frequencies[f"{freq_name}_frequency"] = shaft_freq * multiplier

        return frequencies

    def get_symptoms_by_severity(self, fault_type: FaultType, severity: SeverityLevel) -> List[str]:
        """
        Get symptoms filtered by severity level.

        Args:
            fault_type: Type of fault
            severity: Severity level

        Returns:
            List of symptoms appropriate for the severity level
        """
        pattern = self.get_fault_pattern(fault_type)
        if not pattern:
            return []

        all_symptoms = pattern.symptoms
        filtered_symptoms = []

        severity_keywords = {
            SeverityLevel.LOW: ["轻微", "初期", "small", "minor"],
            SeverityLevel.MEDIUM: ["明显", "中等", "noticeable", "moderate"],
            SeverityLevel.HIGH: ["严重", "显著", "severe", "significant"],
            SeverityLevel.CRITICAL: ["紧急", "危险", "critical", "urgent"]
        }

        keywords = severity_keywords.get(severity, [])

        for symptom in all_symptoms:
            if any(keyword in symptom.lower() for keyword in keywords):
                filtered_symptoms.append(symptom)

        # If no severity-specific symptoms found, return general ones
        if not filtered_symptoms:
            filtered_symptoms = all_symptoms[:3]  # Return first 3 general symptoms

        return filtered_symptoms

    def get_diagnostic_explanation(self,
                                 fault_type: FaultType,
                                 observed_frequencies: List[float],
                                 shaft_speed: float) -> Dict[str, Any]:
        """
        Generate diagnostic explanation based on observed frequencies.

        Args:
            fault_type: Suspected fault type
            observed_frequencies: List of observed peak frequencies
            shaft_speed: Shaft speed in RPM

        Returns:
            Diagnostic explanation with confidence and evidence
        """
        pattern = self.get_fault_pattern(fault_type)
        if not pattern:
            return {"error": "Unknown fault type"}

        expected_freqs = self.get_characteristic_frequencies(fault_type, shaft_speed)

        # Match observed frequencies with expected ones
        matches = self._match_frequencies(observed_frequencies, expected_freqs)

        # Calculate confidence based on frequency matches
        confidence = self._calculate_diagnostic_confidence(matches, pattern)

        return {
            "fault_type": fault_type.value,
            "confidence": confidence,
            "evidence": {
                "matched_frequencies": matches,
                "expected_frequencies": expected_freqs,
                "symptoms": pattern.symptoms,
                "causes": pattern.common_causes
            },
            "characteristics": {
                "time_domain": pattern.time_domain_features,
                "frequency_domain": pattern.frequency_domain_features,
                "typical_amplitudes": pattern.typical_amplitudes
            }
        }

    def get_maintenance_timeline(self,
                               fault_type: FaultType,
                               severity: SeverityLevel) -> Dict[str, Any]:
        """
        Get recommended maintenance timeline.

        Args:
            fault_type: Type of fault
            severity: Severity level

        Returns:
            Maintenance timeline with phases and durations
        """
        base_timeline = {
            FaultType.INNER_RACE_FAULT: {"immediate": 0, "planned": 7, "overhaul": 30},
            FaultType.OUTER_RACE_FAULT: {"immediate": 0, "planned": 14, "overhaul": 45},
            FaultType.BALL_DEFECT: {"immediate": 0, "planned": 3, "overhaul": 30},
            FaultType.MISALIGNMENT: {"immediate": 1, "planned": 7, "overhaul": 60},
            FaultType.IMBALANCE: {"immediate": 0, "planned": 3, "overhaul": 30},
            FaultType.LOOSENESS: {"immediate": 0, "planned": 1, "overhaul": 14}
        }

        timeline = base_timeline.get(fault_type, {"immediate": 0, "planned": 7, "overhaul": 30})

        # Adjust timeline based on severity
        if severity == SeverityLevel.CRITICAL:
            timeline["immediate"] = 0
            timeline["planned"] = max(1, timeline["planned"] // 2)
        elif severity == SeverityLevel.LOW:
            timeline["immediate"] = max(1, timeline["immediate"] + 7)
            timeline["planned"] = timeline["planned"] * 2

        return {
            "timeline_days": timeline,
            "phases": self._generate_maintenance_phases(timeline),
            "milestones": self._generate_maintenance_milestones(fault_type, timeline)
        }

    def _initialize_fault_patterns(self) -> Dict[FaultType, FaultPattern]:
        """Initialize fault patterns with domain knowledge."""
        return {
            FaultType.INNER_RACE_FAULT: FaultPattern(
                fault_type=FaultType.INNER_RACE_FAULT,
                characteristic_frequencies={
                    "bpfi": 3.05,  # Ball pass frequency inner race
                    "shaft": 1.0,
                    "harmonics": [2.0, 3.0, 4.0]
                },
                frequency_multiples=[1.0, 2.05, 3.05, 4.05],
                time_domain_features={
                    "impulsiveness": "高",
                    "periodicity": "明显",
                    "envelope_modulation": "强"
                },
                frequency_domain_features={
                    "harmonic_content": "丰富",
                    "sideband_pattern": "明显",
                    "frequency_multiples": "清晰"
                },
                typical_amplitudes={
                    "low_speed": 5.0,
                    "normal_speed": 10.0,
                    "high_speed": 20.0
                },
                symptoms=[
                    "内圈出现周期性冲击",
                    "载荷区信号增强明显",
                    "温度略有升高",
                    "噪声增大"
                ],
                common_causes=[
                    "润滑不良",
                    "安装不当",
                    "材料疲劳",
                    "过载运行"
                ]
            ),

            FaultType.OUTER_RACE_FAULT: FaultPattern(
                fault_type=FaultType.OUTER_RACE_FAULT,
                characteristic_frequencies={
                    "bpfo": 2.05,  # Ball pass frequency outer race
                    "shaft": 1.0,
                    "harmonics": [2.0, 3.0, 4.0]
                },
                frequency_multiples=[1.0, 2.05, 3.05, 4.05],
                time_domain_features={
                    "impulsiveness": "中等",
                    "periodicity": "稳定",
                    "envelope_modulation": "明显"
                },
                frequency_domain_features={
                    "harmonic_content": "清晰",
                    "sideband_pattern": "较弱",
                    "frequency_multiples": "规则"
                },
                typical_amplitudes={
                    "low_speed": 3.0,
                    "normal_speed": 8.0,
                    "high_speed": 15.0
                },
                symptoms=[
                    "外圈出现周期性冲击",
                    "固定载荷区信号特征",
                    "运行温度正常或略高",
                    "噪声周期性变化"
                ],
                common_causes=[
                    "轴承座松动",
                    "润滑不足",
                    "异物侵入",
                    "材料老化"
                ]
            ),

            FaultType.BALL_DEFECT: FaultPattern(
                fault_type=FaultType.BALL_DEFECT,
                characteristic_frequencies={
                    "bsf": 2.35,  # Ball spin frequency
                    "shaft": 1.0,
                    "cage": 0.4
                },
                frequency_multiples=[1.0, 2.35, 3.35],
                time_domain_features={
                    "impulsiveness": "随机性强",
                    "periodicity": "不稳定",
                    "envelope_modulation": "变化"
                },
                frequency_domain_features={
                    "harmonic_content": "复杂",
                    "sideband_pattern": "随机",
                    "frequency_multiples": "不规则"
                },
                typical_amplitudes={
                    "low_speed": 2.0,
                    "normal_speed": 6.0,
                    "high_speed": 12.0
                },
                symptoms=[
                    "随机冲击信号",
                    "振动幅值波动大",
                    "噪声不规则",
                    "可能伴随温度异常"
                ],
                common_causes=[
                    "钢球表面疲劳",
                    "润滑污染",
                    "安装损伤",
                    "过载冲击"
                ]
            ),

            FaultType.MISALIGNMENT: FaultPattern(
                fault_type=FaultType.MISALIGNMENT,
                characteristic_frequencies={
                    "shaft": 1.0,
                    "2x_shaft": 2.0,
                    "3x_shaft": 3.0
                },
                frequency_multiples=[1.0, 2.0, 3.0],
                time_domain_features={
                    "impulsiveness": "低",
                    "periodicity": "与转速同步",
                    "envelope_modulation": "弱"
                },
                frequency_domain_features={
                    "harmonic_content": "主要为基频和2倍频",
                    "sideband_pattern": "弱",
                    "frequency_multiples": "以2倍频为主"
                },
                typical_amplitudes={
                    "low_speed": 4.0,
                    "normal_speed": 8.0,
                    "high_speed": 16.0
                },
                symptoms=[
                    "轴向振动增大",
                    "径向振动2倍频突出",
                    "联轴器位置异响",
                    "轴承温度分布不均"
                ],
                common_causes=[
                    "安装找正不良",
                    "基础沉降",
                    "热变形",
                    "联轴器损坏"
                ]
            ),

            FaultType.IMBALANCE: FaultPattern(
                fault_type=FaultType.IMBALANCE,
                characteristic_frequencies={
                    "shaft": 1.0
                },
                frequency_multiples=[1.0],
                time_domain_features={
                    "impulsiveness": "低",
                    "periodicity": "与转速完全同步",
                    "envelope_modulation": "无"
                },
                frequency_domain_features={
                    "harmonic_content": "主要为基频",
                    "sideband_pattern": "无",
                    "frequency_multiples": "基频为主"
                },
                typical_amplitudes={
                    "low_speed": 3.0,
                    "normal_speed": 12.0,
                    "high_speed": 25.0
                },
                symptoms=[
                    "径向振动基频突出",
                    "相位稳定",
                    "随速度平方增加",
                    "启动停止过程明显"
                ],
                common_causes=[
                    "转子不平衡",
                    "叶片损坏脱落",
                    "结垢不均",
                    "制造误差"
                ]
            )
        }

    def _initialize_maintenance_actions(self) -> List[MaintenanceAction]:
        """Initialize maintenance action database."""
        return [
            MaintenanceAction(
                action_id="inner_race_replacement",
                description="更换内圈损坏的轴承",
                priority="high",
                required_tools=["轴承拉马", "加热器", "扭力扳手"],
                required_parts=["新轴承", "润滑脂"],
                time_estimate="4-6小时",
                skill_level="中级技工",
                safety_precautions=["断电挂牌", "使用个人防护装备", "确保设备完全停止"]
            ),

            MaintenanceAction(
                action_id="outer_race_replacement",
                description="更换外圈损坏的轴承",
                priority="high",
                required_tools=["轴承拉马", "清洗剂", "测量工具"],
                required_parts=["新轴承", "密封件"],
                time_estimate="3-5小时",
                skill_level="中级技工",
                safety_precautions=["断电挂牌", "检查轴承座状况", "清洁安装表面"]
            ),

            MaintenanceAction(
                action_id="alignment_correction",
                description="重新进行设备对中",
                priority="medium",
                required_tools=["激光对中仪", "垫片", "扳手套组"],
                required_parts=["垫片", "螺栓"],
                time_estimate="2-4小时",
                skill_level="高级技工",
                safety_precautions=["确保设备稳定", "检查基础状况", "记录调整数据"]
            ),

            MaintenanceAction(
                action_id="balancing_correction",
                description="进行转子动平衡校正",
                priority="medium",
                required_tools=["动平衡仪", "平衡块", "焊接设备"],
                required_parts=["平衡块", "焊条"],
                time_estimate="1-3小时",
                skill_level="高级技工",
                safety_precautions=["断电挂牌", "清除转子杂物", "检查旋转部件"]
            ),

            MaintenanceAction(
                action_id="bearing_lubrication",
                description="补充或更换润滑脂",
                priority="low",
                required_tools=["润滑枪", "清洁布", "刮刀"],
                required_parts=["润滑脂", "清洁剂"],
                time_estimate="30-60分钟",
                skill_level="初级技工",
                safety_precautions=["使用合适润滑脂", "避免过度润滑", "清洁润滑点"]
            )
        ]

    def _initialize_fault_relationships(self) -> List[FaultRelationship]:
        """Initialize fault relationship graph."""
        return [
            FaultRelationship(
                source_fault=FaultType.IMBALANCE,
                target_fault=FaultType.INNER_RACE_FAULT,
                relationship_type="causes",
                probability=0.7,
                time_frame="3-6个月"
            ),

            FaultRelationship(
                source_fault=FaultType.MISALIGNMENT,
                target_fault=FaultType.OUTER_RACE_FAULT,
                relationship_type="causes",
                probability=0.6,
                time_frame="2-4个月"
            ),

            FaultRelationship(
                source_fault=FaultType.LOOSENESS,
                target_fault=FaultType.MISALIGNMENT,
                relationship_type="causes",
                probability=0.8,
                time_frame="1-2个月"
            ),

            FaultRelationship(
                source_fault=FaultType.INNER_RACE_FAULT,
                target_fault=FaultType.BALL_DEFECT,
                relationship_type="progresses_to",
                probability=0.5,
                time_frame="1-3个月"
            )
        ]

    def _initialize_component_frequencies(self) -> Dict[str, Any]:
        """Initialize component frequency calculation parameters."""
        return {
            "bearing": {
                "geometric_factors": {
                    "deep_groove_ball": {"bpfi": 3.05, "bpfo": 2.05, "bsf": 2.35, "ftf": 0.4},
                    "angular_contact": {"bpfi": 4.05, "bpfo": 2.95, "bsf": 2.85, "ftf": 0.45},
                    "cylindrical_roller": {"bpfi": 3.5, "bpfo": 2.5, "bsf": 2.8, "ftf": 0.4}
                }
            },
            "gear": {
                "mesh_frequencies": {
                    "single_stage": 1.0,
                    "planetary": {"sun": 3.0, "ring": -1.5, "carrier": 0.5}
                }
            }
        }

    def _initialize_diagnostic_rules(self) -> List[Dict[str, Any]]:
        """Initialize diagnostic rule base."""
        return [
            {
                "rule_id": "harmonic_detection",
                "condition": "multiple harmonics present",
                "fault_types": [FaultType.INNER_RACE_FAULT, FaultType.OUTER_RACE_FAULT],
                "confidence_boost": 0.2
            },

            {
                "rule_id": "1x_dominant",
                "condition": "1x frequency dominant",
                "fault_types": [FaultType.IMBALANCE],
                "confidence_boost": 0.3
            },

            {
                "rule_id": "2x_dominant",
                "condition": "2x frequency dominant",
                "fault_types": [FaultType.MISALIGNMENT],
                "confidence_boost": 0.25
            },

            {
                "rule_id": "random_impulses",
                "condition": "random impulse pattern",
                "fault_types": [FaultType.BALL_DEFECT, FaultType.LOOSENESS],
                "confidence_boost": 0.15
            }
        ]

    def _match_frequencies(self,
                          observed: List[float],
                          expected: Dict[str, float],
                          tolerance: float = 0.1) -> List[Dict[str, Any]]:
        """Match observed frequencies with expected ones."""
        matches = []

        for obs_freq in observed:
            for exp_name, exp_freq in expected.items():
                if abs(obs_freq - exp_freq) / exp_freq <= tolerance:
                    matches.append({
                        "observed": obs_freq,
                        "expected": exp_freq,
                        "expected_name": exp_name,
                        "error_percent": abs(obs_freq - exp_freq) / exp_freq * 100
                    })

        return matches

    def _calculate_diagnostic_confidence(self,
                                       matches: List[Dict[str, Any]],
                                       pattern: FaultPattern) -> float:
        """Calculate diagnostic confidence based on frequency matches."""
        if not matches:
            return 0.1

        # Base confidence from frequency matches
        freq_confidence = min(len(matches) / 3.0, 1.0) * 0.7

        # Additional confidence from match accuracy
        avg_error = sum(match["error_percent"] for match in matches) / len(matches)
        accuracy_confidence = max(1.0 - avg_error / 10.0, 0.0) * 0.3

        return min(freq_confidence + accuracy_confidence, 1.0)

    def _generate_maintenance_phases(self, timeline: Dict[str, int]) -> List[Dict[str, Any]]:
        """Generate maintenance phases from timeline."""
        phases = []

        if timeline["immediate"] == 0:
            phases.append({
                "phase": "immediate",
                "duration_days": 0,
                "description": "立即处理",
                "actions": ["停机检查", "安全措施", "临时修复"]
            })

        if timeline["planned"] > 0:
            phases.append({
                "phase": "planned",
                "duration_days": timeline["planned"],
                "description": "计划维修",
                "actions": ["准备备件", "安排维修窗口", "详细检查"]
            })

        phases.append({
            "phase": "overhaul",
            "duration_days": timeline["overhaul"],
            "description": "全面检修",
            "actions": ["更换部件", "系统检查", "性能测试"]
        })

        return phases

    def _generate_maintenance_milestones(self,
                                       fault_type: FaultType,
                                       timeline: Dict[str, int]) -> List[Dict[str, Any]]:
        """Generate maintenance milestones."""
        return [
            {
                "milestone": "initial_diagnosis",
                "day": 0,
                "description": f"{fault_type.value} 初始诊断",
                "critical": True
            },
            {
                "milestone": "inspection_complete",
                "day": 1,
                "description": "现场检查完成",
                "critical": True
            },
            {
                "milestone": "parts_ordered",
                "day": 2,
                "description": "备件订购",
                "critical": False
            },
            {
                "milestone": "repair_complete",
                "day": timeline["planned"],
                "description": "维修完成",
                "critical": True
            },
            {
                "milestone": "testing_complete",
                "day": timeline["planned"] + 1,
                "description": "测试验证完成",
                "critical": True
            }
        ]

    def export_knowledge(self, format: str = "json") -> str:
        """Export knowledge graph data."""
        export_data = {
            "fault_patterns": {k.value: asdict(v) for k, v in self.fault_patterns.items()},
            "maintenance_actions": [asdict(action) for action in self.maintenance_actions],
            "fault_relationships": [asdict(rel) for rel in self.fault_relationships],
            "component_frequencies": self.component_frequencies,
            "diagnostic_rules": self.diagnostic_rules
        }

        if format == "json":
            return json.dumps(export_data, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def search_by_symptom(self, symptoms: List[str]) -> List[Tuple[FaultType, float]]:
        """
        Search for possible faults based on symptoms.

        Args:
            symptoms: List of observed symptoms

        Returns:
            List of (fault_type, confidence) tuples
        """
        symptom_matches = []

        for fault_type, pattern in self.fault_patterns.items():
            match_count = 0
            for symptom in symptoms:
                if any(symptom.lower() in pattern_symptom.lower()
                      for pattern_symptom in pattern.symptoms):
                    match_count += 1

            if match_count > 0:
                confidence = match_count / len(symptoms)
                symptom_matches.append((fault_type, confidence))

        # Sort by confidence
        symptom_matches.sort(key=lambda x: x[1], reverse=True)
        return symptom_matches