"""
Context Processor for Enhanced LLM Explanations

Provides context-aware processing and enhancement of explanations
by incorporating domain knowledge, operational context, and
historical information.
"""

from typing import Dict, Any, List, Optional, Tuple
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class OperationalContext:
    """Operational context information."""
    device_type: str
    operating_speed: float
    load_condition: str
    environmental_conditions: str
    maintenance_history: List[Dict[str, Any]]
    last_inspection: Optional[str]
    operating_hours: int


@dataclass
class HistoricalContext:
    """Historical diagnostic context."""
    previous_diagnoses: List[Dict[str, Any]]
    fault_trends: Dict[str, Any]
    maintenance_records: List[Dict[str, Any]]
    performance_degradation: Dict[str, float]
    failure_history: List[Dict[str, Any]]


@dataclass
class SystemContext:
    """System-level context information."""
    connected_equipment: List[str]
    process_impact: str
    criticality_level: str
    redundancy_info: Dict[str, Any]
    safety_constraints: List[str]


class ContextProcessor:
    """
    Processes and enhances explanations with contextual information.

    This class integrates various context sources to provide comprehensive
    background information for LLM-enhanced explanations.
    """

    def __init__(self):
        """Initialize the context processor."""
        self.device_database = self._initialize_device_database()
        self.fault_severity_matrix = self._initialize_severity_matrix()
        self.maintenance_guidelines = self._initialize_maintenance_guidelines()

    def process_diagnostic_context(self,
                                 device_info: Dict[str, Any],
                                 diagnostic_data: Dict[str, Any],
                                 historical_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process comprehensive diagnostic context.

        Args:
            device_info: Device information
            diagnostic_data: Current diagnostic results
            historical_data: Historical data (optional)

        Returns:
            Enhanced context dictionary
        """
        context = {
            "operational": self._process_operational_context(device_info),
            "diagnostic": self._process_diagnostic_context(diagnostic_data),
            "historical": self._process_historical_context(historical_data),
            "system": self._process_system_context(device_info),
            "enhancement": self._generate_context_enhancement(device_info, diagnostic_data)
        }

        return context

    def get_relevant_historical_patterns(self,
                                       current_diagnosis: Dict[str, Any],
                                       historical_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find relevant historical patterns for current diagnosis.

        Args:
            current_diagnosis: Current diagnosis results
            historical_data: Historical data

        Returns:
            List of relevant historical patterns
        """
        patterns = []

        if not historical_data:
            return patterns

        current_fault_type = current_diagnosis.get("fault_type", "")
        current_severity = current_diagnosis.get("severity", "")

        # Look for similar fault patterns
        for past_diagnosis in historical_data.get("previous_diagnoses", []):
            similarity_score = self._calculate_diagnosis_similarity(
                current_diagnosis, past_diagnosis
            )

            if similarity_score > 0.6:  # Threshold for similarity
                patterns.append({
                    "historical_case": past_diagnosis,
                    "similarity_score": similarity_score,
                    "relevance_reason": self._explain_relevance(current_diagnosis, past_diagnosis)
                })

        # Sort by similarity score
        patterns.sort(key=lambda x: x["similarity_score"], reverse=True)

        return patterns[:5]  # Return top 5 most relevant patterns

    def generate_maintenance_context(self,
                                   fault_type: str,
                                   severity: str,
                                   device_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate maintenance-specific context.

        Args:
            fault_type: Type of fault
            severity: Severity level
            device_info: Device information

        Returns:
            Maintenance context information
        """
        device_type = device_info.get("device_type", "")
        criticality = device_info.get("criticality_level", "medium")

        maintenance_context = {
            "urgency_level": self._calculate_maintenance_urgency(severity, criticality),
            "required_resources": self._get_required_resources(fault_type, device_type),
            "time_constraints": self._determine_time_constraints(device_info),
            "safety_considerations": self._get_safety_considerations(fault_type, device_info),
            "impact_assessment": self._assess_maintenance_impact(device_info),
            "recommended_actions": self._get_maintenance_recommendations(fault_type, severity)
        }

        return maintenance_context

    def enhance_explanation_with_context(self,
                                       base_explanation: str,
                                       context: Dict[str, Any]) -> Dict[str, str]:
        """
        Enhance base explanation with contextual information.

        Args:
            base_explanation: Original LLM explanation
            context: Processed context information

        Returns:
            Enhanced explanation with context sections
        """
        enhanced_sections = {
            "base_explanation": base_explanation,
            "operational_context": self._generate_operational_section(context.get("operational", {})),
            "historical_context": self._generate_historical_section(context.get("historical", {})),
            "system_context": self._generate_system_section(context.get("system", {})),
            "maintenance_context": self._generate_maintenance_section(context.get("enhancement", {}))
        }

        return enhanced_sections

    def detect_contextual_anomalies(self,
                                  diagnostic_data: Dict[str, Any],
                                  context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies based on contextual information.

        Args:
            diagnostic_data: Current diagnostic results
            context: Context information

        Returns:
            List of detected anomalies
        """
        anomalies = []

        # Check for speed-related anomalies
        operating_speed = context.get("operational", {}).get("operating_speed", 0)
        if operating_speed > 0:
            speed_anomalies = self._check_speed_anomalies(diagnostic_data, operating_speed)
            anomalies.extend(speed_anomalies)

        # Check for load-related anomalies
        load_condition = context.get("operational", {}).get("load_condition", "")
        if load_condition:
            load_anomalies = self._check_load_anomalies(diagnostic_data, load_condition)
            anomalies.extend(load_anomalies)

        # Check for historical trend anomalies
        historical_data = context.get("historical", {})
        if historical_data:
            trend_anomalies = self._check_trend_anomalies(diagnostic_data, historical_data)
            anomalies.extend(trend_anomalies)

        return anomalies

    def generate_contextual_recommendations(self,
                                           diagnosis: Dict[str, Any],
                                           context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on contextual information.

        Args:
            diagnosis: Diagnosis results
            context: Context information

        Returns:
            List of contextual recommendations
        """
        recommendations = []

        # Operational recommendations
        operational_recs = self._generate_operational_recommendations(diagnosis, context)
        recommendations.extend(operational_recs)

        # Maintenance recommendations
        maintenance_recs = self._generate_maintenance_recommendations_from_context(diagnosis, context)
        recommendations.extend(maintenance_recs)

        # Monitoring recommendations
        monitoring_recs = self._generate_monitoring_recommendations(diagnosis, context)
        recommendations.extend(monitoring_recs)

        return recommendations

    def _process_operational_context(self, device_info: Dict[str, Any]) -> OperationalContext:
        """Process operational context information."""
        return OperationalContext(
            device_type=device_info.get("device_type", ""),
            operating_speed=device_info.get("operating_speed", 0.0),
            load_condition=device_info.get("load_condition", "normal"),
            environmental_conditions=device_info.get("environmental_conditions", "normal"),
            maintenance_history=device_info.get("maintenance_history", []),
            last_inspection=device_info.get("last_inspection"),
            operating_hours=device_info.get("operating_hours", 0)
        )

    def _process_diagnostic_context(self, diagnostic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process diagnostic context information."""
        return {
            "fault_type": diagnostic_data.get("fault_type", ""),
            "confidence": diagnostic_data.get("confidence", 0.0),
            "severity": diagnostic_data.get("severity", ""),
            "key_indicators": diagnostic_data.get("key_indicators", []),
            "frequency_content": diagnostic_data.get("frequency_content", {}),
            "amplitude_levels": diagnostic_data.get("amplitude_levels", {})
        }

    def _process_historical_context(self, historical_data: Optional[Dict[str, Any]]) -> HistoricalContext:
        """Process historical context information."""
        if not historical_data:
            return HistoricalContext([], {}, {}, {}, [])

        return HistoricalContext(
            previous_diagnoses=historical_data.get("previous_diagnoses", []),
            fault_trends=historical_data.get("fault_trends", {}),
            maintenance_records=historical_data.get("maintenance_records", []),
            performance_degradation=historical_data.get("performance_degradation", {}),
            failure_history=historical_data.get("failure_history", [])
        )

    def _process_system_context(self, device_info: Dict[str, Any]) -> SystemContext:
        """Process system-level context information."""
        return SystemContext(
            connected_equipment=device_info.get("connected_equipment", []),
            process_impact=device_info.get("process_impact", "unknown"),
            criticality_level=device_info.get("criticality_level", "medium"),
            redundancy_info=device_info.get("redundancy_info", {}),
            safety_constraints=device_info.get("safety_constraints", [])
        )

    def _generate_context_enhancement(self,
                                     device_info: Dict[str, Any],
                                     diagnostic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate context enhancement information."""
        fault_type = diagnostic_data.get("fault_type", "")
        severity = diagnostic_data.get("severity", "")

        return {
            "fault_type_specific": self._get_fault_type_context(fault_type),
            "severity_adjusted": self._get_severity_context(severity),
            "device_specific": self._get_device_specific_context(device_info),
            "operational_impact": self._assess_operational_impact(diagnostic_data, device_info)
        }

    def _calculate_diagnosis_similarity(self,
                                      current: Dict[str, Any],
                                      historical: Dict[str, Any]) -> float:
        """Calculate similarity between current and historical diagnoses."""
        similarity = 0.0
        factors = 0

        # Fault type similarity
        if current.get("fault_type") == historical.get("fault_type"):
            similarity += 0.4
        factors += 1

        # Severity similarity
        if current.get("severity") == historical.get("severity"):
            similarity += 0.3
        factors += 1

        # Device type similarity
        if current.get("device_type") == historical.get("device_type"):
            similarity += 0.3
        factors += 1

        return similarity if factors > 0 else 0.0

    def _explain_relevance(self,
                         current: Dict[str, Any],
                         historical: Dict[str, Any]) -> str:
        """Explain why historical case is relevant."""
        reasons = []

        if current.get("fault_type") == historical.get("fault_type"):
            reasons.append("相同故障类型")

        if current.get("severity") == historical.get("severity"):
            reasons.append("相同严重程度")

        if current.get("device_type") == historical.get("device_type"):
            reasons.append("相同设备类型")

        return "；".join(reasons) if reasons else "部分特征匹配"

    def _calculate_maintenance_urgency(self, severity: str, criticality: str) -> str:
        """Calculate maintenance urgency level."""
        urgency_matrix = {
            ("critical", "critical"): "immediate",
            ("critical", "high"): "immediate",
            ("critical", "medium"): "urgent",
            ("high", "critical"): "immediate",
            ("high", "high"): "urgent",
            ("high", "medium"): "high",
            ("medium", "critical"): "urgent",
            ("medium", "high"): "high",
            ("medium", "medium"): "medium",
            ("low", "critical"): "high",
            ("low", "high"): "medium",
            ("low", "medium"): "low"
        }

        return urgency_matrix.get((severity.lower(), criticality.lower()), "medium")

    def _get_required_resources(self, fault_type: str, device_type: str) -> Dict[str, Any]:
        """Get required maintenance resources."""
        # This would typically query a resource database
        return {
            "personnel": ["maintenance_technician", "engineer"],
            "tools": ["vibration_analyzer", "bearing_puller", "torque_wrench"],
            "parts": ["spare_bearing", "lubricant", "seals"],
            "equipment": ["crane", "lift"]
        }

    def _determine_time_constraints(self, device_info: Dict[str, Any]) -> Dict[str, Any]:
        """Determine maintenance time constraints."""
        criticality = device_info.get("criticality_level", "medium")
        process_impact = device_info.get("process_impact", "unknown")

        constraints = {
            "min_downtime": "2 hours",
            "max_downtime": "24 hours",
            "preferred_window": "weekend",
            "lead_time": "1 week"
        }

        if criticality == "critical":
            constraints["max_downtime"] = "8 hours"
            constraints["lead_time"] = "48 hours"

        return constraints

    def _get_safety_considerations(self, fault_type: str, device_info: Dict[str, Any]) -> List[str]:
        """Get safety considerations for maintenance."""
        base_considerations = [
            "Lockout/Tagout procedures",
            "Personal Protective Equipment (PPE)",
            "Electrical safety checks",
            "Mechanical energy release"
        ]

        if fault_type in ["bearing_failure", "shaft_damage"]:
            base_considerations.extend([
                "Heavy lifting precautions",
                "Bearing handling safety",
                "Alignment tool safety"
            ])

        return base_considerations

    def _assess_maintenance_impact(self, device_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess impact of maintenance activities."""
        return {
            "production_impact": device_info.get("production_impact", "medium"),
            "cost_estimate": "5000-15000",
            "resource_utilization": "medium",
            "quality_impact": "low"
        }

    def _get_maintenance_recommendations(self, fault_type: str, severity: str) -> List[str]:
        """Get maintenance recommendations based on fault and severity."""
        recommendations = []

        if fault_type == "bearing_failure":
            recommendations.extend([
                "更换损坏轴承",
                "检查润滑系统",
                "验证安装精度"
            ])
        elif fault_type == "misalignment":
            recommendations.extend([
                "重新对中",
                "检查基础状态",
                "验证联轴器状态"
            ])

        if severity == "critical":
            recommendations.insert(0, "立即停机检查")

        return recommendations

    def _generate_operational_section(self, operational_context: OperationalContext) -> str:
        """Generate operational context section."""
        sections = []

        if operational_context.device_type:
            sections.append(f"设备类型: {operational_context.device_type}")

        if operational_context.operating_speed > 0:
            sections.append(f"运行转速: {operational_context.operating_speed} RPM")

        if operational_context.load_condition != "normal":
            sections.append(f"载荷条件: {operational_context.load_condition}")

        if operational_context.operating_hours > 0:
            sections.append(f"运行时间: {operational_context.operating_hours} 小时")

        return "\n".join(sections) if sections else "运行状态正常"

    def _generate_historical_section(self, historical_context: HistoricalContext) -> str:
        """Generate historical context section."""
        sections = []

        if historical_context.previous_diagnoses:
            sections.append(f"历史诊断记录: {len(historical_context.previous_diagnoses)} 次")

        if historical_context.failure_history:
            sections.append(f"故障历史: {len(historical_context.failure_history)} 次")

        if historical_context.maintenance_records:
            sections.append(f"维护记录: {len(historical_context.maintenance_records)} 次")

        return "\n".join(sections) if sections else "无相关历史记录"

    def _generate_system_section(self, system_context: SystemContext) -> str:
        """Generate system context section."""
        sections = []

        if system_context.connected_equipment:
            sections.append(f"关联设备: {', '.join(system_context.connected_equipment[:3])}")

        if system_context.criticality_level != "medium":
            sections.append(f"关键程度: {system_context.criticality_level}")

        if system_context.process_impact != "unknown":
            sections.append(f"工艺影响: {system_context.process_impact}")

        return "\n".join(sections) if sections else "系统信息有限"

    def _generate_maintenance_section(self, enhancement_context: Dict[str, Any]) -> str:
        """Generate maintenance context section."""
        sections = []

        if "fault_type_specific" in enhancement_context:
            sections.append("故障特性: " + str(enhancement_context["fault_type_specific"]))

        if "severity_adjusted" in enhancement_context:
            sections.append("严重程度评估: " + str(enhancement_context["severity_adjusted"]))

        if "operational_impact" in enhancement_context:
            impact = enhancement_context["operational_impact"]
            if impact:
                sections.append("运行影响: 需要关注设备运行状态变化")

        return "\n".join(sections) if sections else "维护建议基于标准流程"

    def _check_speed_anomalies(self, diagnostic_data: Dict[str, Any], operating_speed: float) -> List[Dict[str, Any]]:
        """Check for speed-related anomalies."""
        anomalies = []

        # Example: Check if frequency content matches operating speed
        frequency_content = diagnostic_data.get("frequency_content", {})
        if frequency_content:
            # This would involve actual frequency analysis
            pass

        return anomalies

    def _check_load_anomalies(self, diagnostic_data: Dict[str, Any], load_condition: str) -> List[Dict[str, Any]]:
        """Check for load-related anomalies."""
        anomalies = []

        if load_condition == "heavy_load":
            amplitude_levels = diagnostic_data.get("amplitude_levels", {})
            if amplitude_levels.get("overall", 0) < 5.0:  # Example threshold
                anomalies.append({
                    "type": "load_amplitude_mismatch",
                    "description": "重载条件下振动幅值偏低，可能存在传感器故障"
                })

        return anomalies

    def _check_trend_anomalies(self, diagnostic_data: Dict[str, Any], historical_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for trend anomalies in historical data."""
        anomalies = []

        fault_trends = historical_data.get("fault_trends", {})
        current_fault = diagnostic_data.get("fault_type", "")

        if current_fault in fault_trends:
            trend = fault_trends[current_fault]
            if trend.get("direction") == "increasing" and trend.get("rate", 0) > 0.5:
                anomalies.append({
                    "type": "rapid_degradation",
                    "description": f"{current_fault} 故障特征快速恶化",
                    "urgency": "high"
                })

        return anomalies

    def _generate_operational_recommendations(self, diagnosis: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate operational recommendations."""
        recommendations = []

        severity = diagnosis.get("severity", "")
        if severity in ["high", "critical"]:
            recommendations.append({
                "type": "operational",
                "priority": "immediate",
                "action": "降低设备载荷",
                "reason": "减少故障进一步恶化风险"
            })

        return recommendations

    def _generate_maintenance_recommendations_from_context(self, diagnosis: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate maintenance recommendations based on context."""
        recommendations = []

        fault_type = diagnosis.get("fault_type", "")
        operational_context = context.get("operational", {})

        if fault_type:
            recommendations.append({
                "type": "maintenance",
                "priority": "high",
                "action": f"计划{fault_type}相关维修",
                "reason": "基于当前诊断结果"
            })

        if operational_context.operating_hours > 10000:  # Example threshold
            recommendations.append({
                "type": "preventive",
                "priority": "medium",
                "action": "安排全面检查",
                "reason": "设备运行时间较长"
            })

        return recommendations

    def _generate_monitoring_recommendations(self, diagnosis: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate monitoring recommendations."""
        recommendations = []

        recommendations.append({
            "type": "monitoring",
            "priority": "medium",
            "action": "增加振动监测频率",
            "reason": "持续跟踪故障发展"
        })

        return recommendations

    def _initialize_device_database(self) -> Dict[str, Any]:
        """Initialize device database."""
        # This would typically load from a database or configuration file
        return {
            "motor": {
                "typical_speeds": [1500, 3000, 3600],
                "common_faults": ["bearing_failure", "misalignment", "imbalance"],
                "maintenance_intervals": {"inspection": "monthly", "lubrication": "quarterly"}
            },
            "pump": {
                "typical_speeds": [1800, 3600],
                "common_faults": ["bearing_failure", "cavitation", "misalignment"],
                "maintenance_intervals": {"inspection": "monthly", "seal_check": "quarterly"}
            }
        }

    def _initialize_severity_matrix(self) -> Dict[str, Any]:
        """Initialize fault severity matrix."""
        return {
            "amplitude_thresholds": {
                "low": 2.0,
                "medium": 5.0,
                "high": 10.0,
                "critical": 20.0
            },
            "urgency_levels": {
                "low": "routine",
                "medium": "scheduled",
                "high": "priority",
                "critical": "immediate"
            }
        }

    def _initialize_maintenance_guidelines(self) -> Dict[str, Any]:
        """Initialize maintenance guidelines."""
        return {
            "standard_procedures": {
                "bearing_replacement": "4-8 hours",
                "alignment_check": "2-4 hours",
                "lubrication": "1-2 hours"
            },
            "skill_requirements": {
                "basic": "technician",
                "advanced": "senior_technician",
                "specialized": "engineer"
            }
        }

    def _get_fault_type_context(self, fault_type: str) -> Dict[str, Any]:
        """Get fault type specific context."""
        return {
            "description": f"检测到 {fault_type} 类型故障",
            "common_causes": ["正常磨损", "润滑不足", "过载运行"],
            "typical_progression": "渐进发展，需要及时处理"
        }

    def _get_severity_context(self, severity: str) -> Dict[str, Any]:
        """Get severity specific context."""
        severity_info = {
            "low": {"urgency": "低", "impact": "轻微", "timeline": "1-2周"},
            "medium": {"urgency": "中等", "impact": "明显", "timeline": "1周"},
            "high": {"urgency": "高", "impact": "严重", "timeline": "48-72小时"},
            "critical": {"urgency": "紧急", "impact": "危险", "timeline": "立即"}
        }

        return severity_info.get(severity, severity_info["medium"])

    def _get_device_specific_context(self, device_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get device specific context."""
        device_type = device_info.get("device_type", "")
        return {
            "device_characteristics": f"{device_type} 设备特性",
            "operating_range": "正常工作范围",
            "special_considerations": "特殊考虑事项"
        }

    def _assess_operational_impact(self, diagnostic_data: Dict[str, Any], device_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess operational impact of the diagnosis."""
        return {
            "production_risk": "中等",
            "safety_implications": "需要监控",
            "quality_impact": "轻微",
            "efficiency_impact": "可能下降"
        }