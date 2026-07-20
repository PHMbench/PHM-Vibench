"""
Intermediate Representation for LLM Fault Diagnosis Explanation

This module defines the intermediate data structures that bridge
signal processing explanations and LLM-based natural language generation.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class FaultInfo:
    """Fault information from model prediction."""
    fault_type: str
    confidence: float
    probability_distribution: List[float] = field(default_factory=list)
    predicted_class: int = -1
    prediction_method: str = "Unknown"
    model_name: str = "Unknown"


@dataclass
class SignalAnalysis:
    """Signal analysis results."""
    statistics: Dict[str, float] = field(default_factory=dict)
    frequency_analysis: Dict[str, float] = field(default_factory=dict)
    signal_length: int = 0
    sampling_rate: int = 1024
    key_findings: List[str] = field(default_factory=list)


@dataclass
class TechnicalExplanation:
    """Technical explanation components."""
    signal_path: Optional[Dict[str, Any]] = None
    processing_stages: int = 0
    energy_analysis: Dict[str, Any] = field(default_factory=dict)
    important_features: List[Dict[str, Any]] = field(default_factory=list)
    frequency_components: List[Dict[str, Any]] = field(default_factory=list)
    path_signature: Optional[Dict[str, Any]] = None


@dataclass
class DeviceContext:
    """Device and operational context."""
    device_type: str = "Unknown"
    operating_speed: Optional[float] = None
    load_condition: str = "Unknown"
    environment: str = "Unknown"
    maintenance_history: str = "Unknown"
    installation_date: Optional[str] = None


@dataclass
class LLMIntermediateRepresentation:
    """
    Intermediate representation for LLM-based explanation generation.

    This structure bridges the gap between technical signal processing
    results and natural language explanation generation.
    """
    # Metadata
    explanation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Core diagnostic information
    fault_info: FaultInfo = field(default_factory=lambda: FaultInfo("Unknown", 0.0))
    signal_analysis: SignalAnalysis = field(default_factory=SignalAnalysis)
    technical_explanation: TechnicalExplanation = field(default_factory=TechnicalExplanation)

    # Context information
    device_context: DeviceContext = field(default_factory=DeviceContext)
    user_query: Optional[str] = None
    explanation_style: str = "standard"

    # Additional metadata
    signal_data_info: Dict[str, Any] = field(default_factory=dict)
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "explanation_id": self.explanation_id,
            "timestamp": self.timestamp,
            "fault_info": {
                "fault_type": self.fault_info.fault_type,
                "confidence": self.fault_info.confidence,
                "probability_distribution": self.fault_info.probability_distribution,
                "predicted_class": self.fault_info.predicted_class,
                "prediction_method": self.fault_info.prediction_method,
                "model_name": self.fault_info.model_name
            },
            "signal_analysis": {
                "statistics": self.signal_analysis.statistics,
                "frequency_analysis": self.signal_analysis.frequency_analysis,
                "signal_length": self.signal_analysis.signal_length,
                "sampling_rate": self.signal_analysis.sampling_rate,
                "key_findings": self.signal_analysis.key_findings
            },
            "technical_explanation": {
                "signal_path": self.technical_explanation.signal_path,
                "processing_stages": self.technical_explanation.processing_stages,
                "energy_analysis": self.technical_explanation.energy_analysis,
                "important_features": self.technical_explanation.important_features,
                "frequency_components": self.technical_explanation.frequency_components,
                "path_signature": self.technical_explanation.path_signature
            },
            "device_context": {
                "device_type": self.device_context.device_type,
                "operating_speed": self.device_context.operating_speed,
                "load_condition": self.device_context.load_condition,
                "environment": self.device_context.environment,
                "maintenance_history": self.device_context.maintenance_history,
                "installation_date": self.device_context.installation_date
            },
            "user_query": self.user_query,
            "explanation_style": self.explanation_style,
            "signal_data_info": self.signal_data_info,
            "model_metadata": self.model_metadata,
            "processing_time": self.processing_time
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMIntermediateRepresentation':
        """Create instance from dictionary."""
        ir = cls()

        # Basic fields
        ir.explanation_id = data.get("explanation_id", ir.explanation_id)
        ir.timestamp = data.get("timestamp", ir.timestamp)
        ir.user_query = data.get("user_query")
        ir.explanation_style = data.get("explanation_style", "standard")
        ir.processing_time = data.get("processing_time")

        # Fault info
        fault_data = data.get("fault_info", {})
        ir.fault_info = FaultInfo(
            fault_type=fault_data.get("fault_type", "Unknown"),
            confidence=fault_data.get("confidence", 0.0),
            probability_distribution=fault_data.get("probability_distribution", []),
            predicted_class=fault_data.get("predicted_class", -1),
            prediction_method=fault_data.get("prediction_method", "Unknown"),
            model_name=fault_data.get("model_name", "Unknown")
        )

        # Signal analysis
        signal_data = data.get("signal_analysis", {})
        ir.signal_analysis = SignalAnalysis(
            statistics=signal_data.get("statistics", {}),
            frequency_analysis=signal_data.get("frequency_analysis", {}),
            signal_length=signal_data.get("signal_length", 0),
            sampling_rate=signal_data.get("sampling_rate", 1024),
            key_findings=signal_data.get("key_findings", [])
        )

        # Technical explanation
        tech_data = data.get("technical_explanation", {})
        ir.technical_explanation = TechnicalExplanation(
            signal_path=tech_data.get("signal_path"),
            processing_stages=tech_data.get("processing_stages", 0),
            energy_analysis=tech_data.get("energy_analysis", {}),
            important_features=tech_data.get("important_features", []),
            frequency_components=tech_data.get("frequency_components", []),
            path_signature=tech_data.get("path_signature")
        )

        # Device context
        device_data = data.get("device_context", {})
        ir.device_context = DeviceContext(
            device_type=device_data.get("device_type", "Unknown"),
            operating_speed=device_data.get("operating_speed"),
            load_condition=device_data.get("load_condition", "Unknown"),
            environment=device_data.get("environment", "Unknown"),
            maintenance_history=device_data.get("maintenance_history", "Unknown"),
            installation_date=device_data.get("installation_date")
        )

        # Additional metadata
        ir.signal_data_info = data.get("signal_data_info", {})
        ir.model_metadata = data.get("model_metadata", {})

        return ir

    def get_summary(self) -> str:
        """Get a concise summary of the representation."""
        return (f"故障类型：{self.fault_info.fault_type} (置信度: {self.fault_info.confidence:.1%}), "
                f"设备类型：{self.device_context.device_type}, "
                f"关键发现：{len(self.signal_analysis.key_findings)} 项")


def create_mock_ir(fault_type: str = "内圈故障",
                   confidence: float = 0.85,
                   device_type: str = "滚动轴承") -> LLMIntermediateRepresentation:
    """
    Create a mock intermediate representation for testing.

    Args:
        fault_type: Type of fault
        confidence: Confidence level
        device_type: Type of device

    Returns:
        Mock intermediate representation
    """
    ir = LLMIntermediateRepresentation()

    # Set fault info
    ir.fault_info.fault_type = fault_type
    ir.fault_info.confidence = confidence
    ir.fault_info.probability_distribution = [0.05, 0.1, confidence, 0.05, 0.0]
    ir.fault_info.predicted_class = 2
    ir.fault_info.prediction_method = "TSPN"
    ir.fault_info.model_name = "Transparent Signal Processing Network"

    # Set signal analysis
    ir.signal_analysis.statistics = {
        "mean": 2.5,
        "std": 15.8,
        "rms": 16.0,
        "peak": 45.2,
        "crest_factor": 2.8,
        "skewness": 0.3,
        "kurtosis": 2.1
    }

    ir.signal_analysis.frequency_analysis = {
        "dominant_frequency": 157.5,
        "dominant_power": 12.8,
        "spectral_centroid": 89.3
    }

    ir.signal_analysis.signal_length = 4096
    ir.signal_analysis.sampling_rate = 1024

    ir.signal_analysis.key_findings = [
        f"检测到 {fault_type} 特征频率成分",
        "振动RMS值显著增高",
        "频域分析显示明显谐波",
        "时域波形存在周期性冲击"
    ]

    # Set technical explanation
    ir.technical_explanation.processing_stages = 4
    ir.technical_explanation.important_features = [
        {"feature": "RMS值", "value": 16.0, "threshold": 5.0, "significance": "高"},
        {"feature": "峰值因子", "value": 2.8, "threshold": 3.0, "significance": "中等"},
        {"feature": "频谱峰值", "value": 157.5, "threshold": 100.0, "significance": "高"}
    ]

    ir.technical_explanation.frequency_components = [
        {"frequency": 157.5, "amplitude": 12.8, "type": "故障频率"},
        {"frequency": 315.0, "amplitude": 6.2, "type": "谐波"},
        {"frequency": 472.5, "amplitude": 2.1, "type": "谐波"}
    ]

    # Set device context
    ir.device_context.device_type = device_type
    ir.device_context.operating_speed = 1800.0
    ir.device_context.load_condition = "正常载荷"
    ir.device_context.environment = "室内"
    ir.device_context.maintenance_history = "定期维护"

    return ir