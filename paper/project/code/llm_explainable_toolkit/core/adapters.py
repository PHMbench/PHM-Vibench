"""
Adapters for converting between different explanation formats.

This module provides adapters to convert between various explanation
formats and the LLM intermediate representation.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .intermediate_representation import (
    LLMIntermediateRepresentation,
    FaultInfo,
    SignalAnalysis,
    TechnicalExplanation,
    DeviceContext
)


class ExplanationToIRAdapter:
    """
    Adapter for converting various explanation formats to LLM intermediate representation.
    """

    @staticmethod
    def from_model_prediction(signal_data: Union[torch.Tensor, np.ndarray],
                             model_prediction: Dict[str, Any],
                             device_context: Optional[Dict[str, Any]] = None,
                             user_query: Optional[str] = None,
                             explanation_style: str = "standard") -> LLMIntermediateRepresentation:
        """
        Convert model prediction and signal data to intermediate representation.

        Args:
            signal_data: Input signal data
            model_prediction: Model prediction results
            device_context: Device context information
            user_query: User query
            explanation_style: Explanation style

        Returns:
            LLM intermediate representation
        """
        ir = LLMIntermediateRepresentation()
        ir.user_query = user_query
        ir.explanation_style = explanation_style

        # Convert fault info
        ir.fault_info = FaultInfo(
            fault_type=model_prediction.get("fault_type", "Unknown"),
            confidence=model_prediction.get("confidence", 0.0),
            probability_distribution=model_prediction.get("probabilities", []),
            predicted_class=model_prediction.get("predicted_class", -1),
            prediction_method=model_prediction.get("method", "Unknown"),
            model_name=model_prediction.get("model_name", "Unknown")
        )

        # Analyze signal
        ir.signal_analysis = ExplanationToIRAdapter._analyze_signal(signal_data)

        # Set device context
        if device_context:
            ir.device_context = DeviceContext(
                device_type=device_context.get("device_type", "Unknown"),
                operating_speed=device_context.get("operating_speed"),
                load_condition=device_context.get("load_condition", "Unknown"),
                environment=device_context.get("environment", "Unknown"),
                maintenance_history=device_context.get("maintenance_history", "Unknown"),
                installation_date=device_context.get("installation_date")
            )

        # Generate technical explanation
        ir.technical_explanation = ExplanationToIRAdapter._generate_technical_explanation(
            ir.fault_info, ir.signal_analysis
        )

        # Store signal data info
        if isinstance(signal_data, torch.Tensor):
            signal_np = signal_data.detach().cpu().numpy()
        else:
            signal_np = signal_data

        ir.signal_data_info = {
            "shape": signal_np.shape,
            "data_type": str(signal_np.dtype),
            "min_value": float(np.min(signal_np)),
            "max_value": float(np.max(signal_np))
        }

        # Store model metadata
        ir.model_metadata = {
            "prediction_timestamp": datetime.now().isoformat(),
            "prediction_details": {k: v for k, v in model_prediction.items()
                                if k not in ["fault_type", "confidence", "probabilities", "predicted_class"]}
        }

        return ir

    @staticmethod
    def from_explainable_toolkit(toolkit_output: Dict[str, Any],
                               user_query: Optional[str] = None,
                               explanation_style: str = "standard") -> LLMIntermediateRepresentation:
        """
        Convert Explainable_FD_Toolkit output to intermediate representation.

        Args:
            toolkit_output: Output from Explainable_FD_Toolkit
            user_query: User query
            explanation_style: Explanation style

        Returns:
            LLM intermediate representation
        """
        ir = LLMIntermediateRepresentation()
        ir.user_query = user_query
        ir.explanation_style = explanation_style

        # Extract fault information
        prediction = toolkit_output.get("prediction", {})
        ir.fault_info = FaultInfo(
            fault_type=prediction.get("fault_type", "Unknown"),
            confidence=prediction.get("confidence", 0.0),
            probability_distribution=prediction.get("probabilities", []),
            predicted_class=prediction.get("predicted_class", -1),
            prediction_method=prediction.get("method", "Explainable_FD_Toolkit"),
            model_name=prediction.get("model_name", "Unknown")
        )

        # Extract signal analysis
        signal_info = toolkit_output.get("signal_analysis", {})
        ir.signal_analysis = SignalAnalysis(
            statistics=signal_info.get("statistics", {}),
            frequency_analysis=signal_info.get("frequency_analysis", {}),
            signal_length=signal_info.get("signal_length", 0),
            sampling_rate=signal_info.get("sampling_rate", 1024),
            key_findings=signal_info.get("key_findings", [])
        )

        # Extract technical explanation
        explanation = toolkit_output.get("explanation", {})
        ir.technical_explanation = TechnicalExplanation(
            signal_path=explanation.get("signal_path"),
            processing_stages=explanation.get("processing_stages", 0),
            energy_analysis=explanation.get("energy_analysis", {}),
            important_features=explanation.get("important_features", []),
            frequency_components=explanation.get("frequency_components", []),
            path_signature=explanation.get("path_signature")
        )

        # Extract device context
        context = toolkit_output.get("context", {})
        ir.device_context = DeviceContext(
            device_type=context.get("device_type", "Unknown"),
            operating_speed=context.get("operating_speed"),
            load_condition=context.get("load_condition", "Unknown"),
            environment=context.get("environment", "Unknown"),
            maintenance_history=context.get("maintenance_history", "Unknown"),
            installation_date=context.get("installation_date")
        )

        # Store additional metadata
        ir.signal_data_info = toolkit_output.get("signal_data_info", {})
        ir.model_metadata = {
            "toolkit_version": toolkit_output.get("toolkit_version", "Unknown"),
            "explanation_method": toolkit_output.get("explanation_method", "Unknown"),
            "generation_timestamp": toolkit_output.get("timestamp", datetime.now().isoformat())
        }

        return ir

    @staticmethod
    def from_json(json_data: Dict[str, Any]) -> LLMIntermediateRepresentation:
        """
        Convert JSON data to intermediate representation.

        Args:
            json_data: JSON data containing explanation information

        Returns:
            LLM intermediate representation
        """
        return LLMIntermediateRepresentation.from_dict(json_data)

    @staticmethod
    def _analyze_signal(signal_data: Union[torch.Tensor, np.ndarray]) -> SignalAnalysis:
        """Analyze signal data and extract key features."""
        # Convert to numpy for analysis
        if isinstance(signal_data, torch.Tensor):
            signal_np = signal_data.detach().cpu().numpy()
        else:
            signal_np = signal_data

        # Flatten if needed
        if signal_np.ndim > 1:
            signal_np = signal_np.flatten()

        # Basic statistics
        stats = {
            "mean": float(np.mean(signal_np)),
            "std": float(np.std(signal_np)),
            "rms": float(np.sqrt(np.mean(signal_np ** 2))),
            "peak": float(np.max(np.abs(signal_np))),
            "crest_factor": float(np.max(np.abs(signal_np)) / (np.sqrt(np.mean(signal_np ** 2)) + 1e-8)),
            "skewness": float(ExplanationToIRAdapter._calculate_skewness(signal_np)),
            "kurtosis": float(ExplanationToIRAdapter._calculate_kurtosis(signal_np))
        }

        # Frequency analysis (simple FFT)
        fft_vals = np.fft.fft(signal_np)
        fft_freq = np.fft.fftfreq(len(signal_np), 1/1024.0)  # Assuming 1kHz sampling

        # Find dominant frequencies
        positive_freq_idx = fft_freq > 0
        positive_freq = fft_freq[positive_freq_idx]
        positive_fft = np.abs(fft_vals[positive_freq_idx])

        if len(positive_fft) > 0:
            dominant_freq_idx = np.argmax(positive_fft)
            dominant_freq = float(positive_freq[dominant_freq_idx])
            dominant_power = float(positive_fft[dominant_freq_idx])
        else:
            dominant_freq = 0.0
            dominant_power = 0.0

        freq_analysis = {
            "dominant_frequency": dominant_freq,
            "dominant_power": dominant_power,
            "spectral_centroid": float(np.sum(positive_freq * positive_fft) / (np.sum(positive_fft) + 1e-8))
        }

        # Generate key findings
        key_findings = ExplanationToIRAdapter._generate_key_findings(stats, freq_analysis)

        return SignalAnalysis(
            statistics=stats,
            frequency_analysis=freq_analysis,
            signal_length=len(signal_np),
            sampling_rate=1024,  # Assumed
            key_findings=key_findings
        )

    @staticmethod
    def _generate_technical_explanation(fault_info: FaultInfo,
                                      signal_analysis: SignalAnalysis) -> TechnicalExplanation:
        """Generate technical explanation components."""
        # Extract important features based on signal analysis
        important_features = []

        stats = signal_analysis.statistics
        if stats["rms"] > 10.0:
            important_features.append({
                "feature": "RMS值",
                "value": stats["rms"],
                "threshold": 10.0,
                "significance": "高"
            })

        if stats["crest_factor"] > 5.0:
            important_features.append({
                "feature": "峰值因子",
                "value": stats["crest_factor"],
                "threshold": 5.0,
                "significance": "高"
            })

        # Frequency components
        freq_components = []
        freq_analysis = signal_analysis.frequency_analysis
        if freq_analysis["dominant_frequency"] > 0:
            freq_components.append({
                "frequency": freq_analysis["dominant_frequency"],
                "amplitude": freq_analysis["dominant_power"],
                "type": "主频成分"
            })

        # Add harmonics for specific fault types
        fault_type = fault_info.fault_type.lower()
        if "内圈" in fault_type or "外圈" in fault_type:
            base_freq = freq_analysis["dominant_frequency"]
            if base_freq > 0:
                for i in range(2, 4):  # Add 2nd and 3rd harmonics
                    freq_components.append({
                        "frequency": base_freq * i,
                        "amplitude": freq_analysis["dominant_power"] / (i * 2),
                        "type": f"{i}次谐波"
                    })

        return TechnicalExplanation(
            processing_stages=4,  # Assumed number of processing stages
            important_features=important_features,
            frequency_components=freq_components,
            energy_analysis={
                "total_energy": float(np.sum([f["amplitude"]**2 for f in freq_components])),
                "dominant_energy_ratio": freq_analysis["dominant_power"] / (freq_analysis["dominant_power"] + 1e-8)
            }
        )

    @staticmethod
    def _generate_key_findings(stats: Dict[str, float],
                             freq_analysis: Dict[str, float]) -> List[str]:
        """Generate key findings from signal analysis."""
        findings = []

        # Statistical findings
        if stats["rms"] > 10.0:
            findings.append("振动RMS值较高，表明存在明显振动异常")

        if stats["crest_factor"] > 5.0:
            findings.append("峰值因子较高，可能存在冲击性故障")

        if stats["skewness"] > 1.0:
            findings.append("信号偏度较大，分布不对称")

        if stats["kurtosis"] > 3.0:
            findings.append("信号峰度较高，存在冲击成分")

        # Frequency findings
        if freq_analysis["dominant_frequency"] > 0:
            findings.append(f"检测到主频成分：{freq_analysis['dominant_frequency']:.1f} Hz")

        if freq_analysis["spectral_centroid"] > 200:
            findings.append("频谱重心较高，高频成分丰富")

        return findings

    @staticmethod
    def _calculate_skewness(signal: np.ndarray) -> float:
        """Calculate signal skewness."""
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return 0.0
        return np.mean(((signal - mean) / std) ** 3)

    @staticmethod
    def _calculate_kurtosis(signal: np.ndarray) -> float:
        """Calculate signal kurtosis."""
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return 0.0
        return np.mean(((signal - mean) / std) ** 4) - 3


class MockDataAdapter:
    """
    Adapter for creating mock data for testing and demonstration.
    """

    @staticmethod
    def create_comprehensive_example(fault_type: str = "内圈故障",
                                   confidence: float = 0.87,
                                   device_type: str = "滚动轴承") -> LLMIntermediateRepresentation:
        """
        Create a comprehensive mock example for demonstration.

        Args:
            fault_type: Type of fault
            confidence: Confidence level
            device_type: Type of device

        Returns:
            Comprehensive mock intermediate representation
        """
        # Create basic mock IR
        ir = ExplanationToIRAdapter.from_model_prediction(
            signal_data=np.random.randn(4096) * 10,  # Mock signal data
            model_prediction={
                "fault_type": fault_type,
                "confidence": confidence,
                "probabilities": [0.03, 0.05, confidence, 0.04, 0.01],
                "predicted_class": 2,
                "method": "TSPN",
                "model_name": "Transparent Signal Processing Network v1.0"
            },
            device_context={
                "device_type": device_type,
                "operating_speed": 1800.0,
                "load_condition": "正常载荷",
                "environment": "工业厂房",
                "maintenance_history": "最近一次维护：3个月前",
                "installation_date": "2022-01-15"
            },
            user_query="请详细解释这个故障的原因和维修建议",
            explanation_style="detailed"
        )

        # Enhance with more detailed analysis
        ir.technical_explanation.signal_path = {
            "input_shape": [1, 4096],
            "processing_steps": [
                {"layer": "FFT", "output_shape": [1, 4096], "description": "快速傅里叶变换"},
                {"layer": "WF", "output_shape": [1, 4096], "description": "小波滤波"},
                {"layer": "Feature Extraction", "output_shape": [1, 52], "description": "统计特征提取"},
                {"layer": "Classification", "output_shape": [1, 5], "description": "故障分类"}
            ]
        }

        ir.technical_explanation.path_signature = {
            "energy_concentration": [0.15, 0.23, 0.87, 0.92, 0.78],
            "frequency_bands": [(0, 50), (50, 200), (200, 500), (500, 1000), (1000, 2000)],
            "key_transformations": ["FFT->频域分析", "WF->噪声滤除", "统计特征提取"]
        }

        # Add processing time
        ir.processing_time = 0.142

        return ir