#!/usr/bin/env python3
"""
API Bridge between LLM Explainable Toolkit and Explainable_FD_Toolkit

This module provides the integration layer for connecting the LLM toolkit
with real model explanations from the Explainable_FD_Toolkit.
"""

import sys
import os
import json
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# Add main repository paths
project_root = Path(__file__).parent.parent.parent.parent
main_repo_root = project_root.parent
sys.path.insert(0, str(main_repo_root))
sys.path.insert(0, str(project_root / "code"))

from llm_explainable_toolkit.core.intermediate_representation import (
    LLMIntermediateRepresentation,
    FaultInfo,
    SignalAnalysis,
    TechnicalExplanation,
    DeviceContext
)


class ExplainableToolkitBridge:
    """
    Bridge class for integrating with Explainable_FD_Toolkit.

    This class handles:
    - Loading model explanations from toolkit
    - Converting explanations to LLM intermediate representation
    - Managing different explanation formats
    """

    def __init__(self, toolkit_path: Optional[str] = None):
        """
        Initialize the bridge.

        Args:
            toolkit_path: Path to Explainable_FD_Toolkit
        """
        self.toolkit_path = Path(toolkit_path) if toolkit_path else main_repo_root / "Paper" / "Explainable_FD_Toolkit"
        self.explanations_cache = {}

    def load_explanation_from_file(self,
                                 explanation_path: Union[str, Path],
                                 format: str = "auto") -> Dict[str, Any]:
        """
        Load explanation from file.

        Args:
            explanation_path: Path to explanation file
            format: File format ("json", "pkl", "auto")

        Returns:
            Explanation dictionary
        """
        explanation_path = Path(explanation_path)

        if not explanation_path.exists():
            raise FileNotFoundError(f"Explanation file not found: {explanation_path}")

        # Auto-detect format if needed
        if format == "auto":
            if explanation_path.suffix.lower() == ".json":
                format = "json"
            elif explanation_path.suffix.lower() == ".pkl":
                format = "pkl"
            else:
                raise ValueError(f"Cannot detect format for file: {explanation_path}")

        # Load explanation
        if format == "json":
            with open(explanation_path, 'r', encoding='utf-8') as f:
                explanation = json.load(f)
        elif format == "pkl":
            with open(explanation_path, 'rb') as f:
                explanation = pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return explanation

    def convert_toolkit_explanation_to_ir(self,
                                        explanation: Dict[str, Any],
                                        signal_data: Optional[np.ndarray] = None,
                                        model_prediction: Optional[Dict[str, Any]] = None,
                                        device_context: Optional[Dict[str, Any]] = None) -> LLMIntermediateRepresentation:
        """
        Convert toolkit explanation to LLM intermediate representation.

        Args:
            explanation: Explanation from toolkit
            signal_data: Original signal data
            model_prediction: Model prediction results
            device_context: Device context information

        Returns:
            LLM intermediate representation
        """
        # Create fault info
        fault_info = FaultInfo(
            fault_type=explanation.get('fault_type', 'unknown'),
            confidence=explanation.get('confidence', 0.0),
            severity=explanation.get('severity', 'medium'),
            description=explanation.get('description', '')
        )

        # Create signal analysis
        signal_analysis = self._extract_signal_analysis(explanation, signal_data)

        # Create technical explanation
        technical_explanation = self._extract_technical_explanation(explanation)

        # Create device context
        device_context_obj = self._create_device_context(device_context, explanation)

        # Create intermediate representation
        ir = LLMIntermediateRepresentation(
            fault_info=fault_info,
            signal_analysis=signal_analysis,
            technical_explanation=technical_explanation,
            device_context=device_context_obj,
            metadata={
                'source': 'explainable_fd_toolkit',
                'explanation_method': explanation.get('method', 'unknown'),
                'model_type': explanation.get('model_type', 'unknown'),
                'extraction_time': datetime.now().isoformat()
            }
        )

        return ir

    def _extract_signal_analysis(self,
                               explanation: Dict[str, Any],
                               signal_data: Optional[np.ndarray] = None) -> SignalAnalysis:
        """Extract signal analysis information from explanation."""

        # Get signal statistics
        statistics = explanation.get('signal_statistics', {})

        # Compute statistics from signal data if available
        if signal_data is not None and not statistics:
            statistics = {
                'mean': float(np.mean(signal_data)),
                'std': float(np.std(signal_data)),
                'rms': float(np.sqrt(np.mean(signal_data**2))),
                'max': float(np.max(signal_data)),
                'min': float(np.min(signal_data)),
                'energy': float(np.sum(signal_data**2))
            }

        # Get frequency analysis
        frequency_analysis = explanation.get('frequency_analysis', {})

        # Get key findings
        key_findings = explanation.get('key_findings', [])

        # Add automatic findings if signal data available
        if signal_data is not None:
            if len(key_findings) == 0:
                key_findings = self._generate_key_findings(signal_data, statistics)

        return SignalAnalysis(
            signal_length=len(signal_data) if signal_data is not None else explanation.get('signal_length', 0),
            sampling_rate=explanation.get('sampling_rate', 1024.0),
            statistics=statistics,
            frequency_analysis=frequency_analysis,
            key_findings=key_findings
        )

    def _extract_technical_explanation(self, explanation: Dict[str, Any]) -> TechnicalExplanation:
        """Extract technical explanation from toolkit explanation."""

        # Get processing path
        signal_path = explanation.get('signal_path', {})

        # Get important features
        important_features = explanation.get('important_features', [])

        # Get attention weights
        attention_weights = explanation.get('attention_weights', {})

        # Get layer contributions
        layer_contributions = explanation.get('layer_contributions', {})

        return TechnicalExplanation(
            signal_path=signal_path,
            important_features=important_features,
            attention_weights=attention_weights,
            layer_contributions=layer_contributions,
            processing_steps=explanation.get('processing_steps', [])
        )

    def _create_device_context(self,
                             device_context: Optional[Dict[str, Any]],
                             explanation: Dict[str, Any]) -> DeviceContext:
        """Create device context from available information."""

        # Use provided device context or extract from explanation
        if device_context is None:
            device_context = explanation.get('device_context', {})

        return DeviceContext(
            device_type=device_context.get('device_type', 'unknown'),
            operating_conditions=device_context.get('operating_conditions', {}),
            maintenance_history=device_context.get('maintenance_history', ''),
            specifications=device_context.get('specifications', {})
        )

    def _generate_key_findings(self,
                             signal_data: np.ndarray,
                             statistics: Dict[str, float]) -> List[str]:
        """Generate key findings from signal data and statistics."""

        findings = []

        if statistics:
            rms = statistics.get('rms', 0)
            peak_factor = statistics.get('peak_factor', 0)

            # High energy indication
            if rms > 5.0:
                findings.append(f"信号能量较高，RMS值为{rms:.2f}")

            # Peak factor indication
            if peak_factor > 4.0:
                findings.append(f"信号峰值因子较高({peak_factor:.2f})，可能存在冲击")
            elif peak_factor < 3.0:
                findings.append(f"信号峰值因子较低({peak_factor:.2f})，较为平稳")

        # Check for periodicity
        if len(signal_data) > 100:
            # Simple autocorrelation check
            signal_centered = signal_data - np.mean(signal_data)
            autocorr = np.correlate(signal_centered, signal_centered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # Find peaks in autocorrelation
            if len(autocorr) > 20:
                peaks = []
                for i in range(20, len(autocorr)//2):
                    if autocorr[i] > np.max(autocorr[max(0, i-10):i]) * 1.2:
                        peaks.append(i)

                if peaks:
                    findings.append(f"信号呈现周期性特征，检测到{len(peaks)}个显著周期")

        if not findings:
            findings.append("信号特征正常，未检测到明显异常")

        return findings

    def generate_mock_tspn_explanation(self,
                                     signal_data: np.ndarray,
                                     fault_type: str = "内圈故障",
                                     confidence: float = 0.85) -> Dict[str, Any]:
        """
        Generate mock TSPN explanation for testing.

        This simulates the output from TSPN_explainable.py
        """
        # Compute basic signal statistics
        statistics = {
            'mean': float(np.mean(signal_data)),
            'std': float(np.std(signal_data)),
            'rms': float(np.sqrt(np.mean(signal_data**2))),
            'max': float(np.max(signal_data)),
            'min': float(np.min(signal_data)),
            'energy': float(np.sum(signal_data**2))
        }

        # Frequency analysis
        fft_vals = np.fft.fft(signal_data)
        fft_freq = np.fft.fftfreq(len(signal_data), 1/1024.0)
        pos_mask = fft_freq > 0
        pos_freq = fft_freq[pos_mask]
        pos_fft = np.abs(fft_vals[pos_mask])

        frequency_analysis = {
            'dominant_frequency': float(pos_freq[np.argmax(pos_fft)]),
            'spectral_centroid': float(np.sum(pos_freq * pos_fft) / (np.sum(pos_fft) + 1e-8)),
            'total_power': float(np.sum(pos_fft))
        }

        # Generate TSPN-specific explanation
        explanation = {
            'fault_type': fault_type,
            'confidence': confidence,
            'severity': 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low',
            'description': f'TSPN模型检测到{fault_type}，置信度为{confidence:.1%}',
            'method': 'TSPN_explainable',
            'model_type': 'Transparent Signal Processing Network',
            'signal_statistics': statistics,
            'frequency_analysis': frequency_analysis,
            'signal_length': len(signal_data),
            'sampling_rate': 1024.0,

            # TSPN-specific processing path
            'signal_path': {
                'processing_steps': [
                    {
                        'layer': 'Input',
                        'output_shape': [1, len(signal_data)],
                        'description': '原始振动信号输入'
                    },
                    {
                        'layer': 'SignalProcessing_Layer1',
                        'modules': ['FFT', 'WF', 'HT', 'I'],
                        'output_shape': [1, len(signal_data)],
                        'description': '第一层信号处理（FFT + 小波滤波 + 希尔伯特变换 + 恒等）'
                    },
                    {
                        'layer': 'FeatureExtractor',
                        'output_shape': [1, 52],
                        'description': '多域特征提取（13个统计特征 × 4个处理路径）'
                    },
                    {
                        'layer': 'Attention_Mechanism',
                        'output_shape': [1, 52],
                        'description': '注意力机制加权'
                    },
                    {
                        'layer': 'Classifier',
                        'output_shape': [1, 10],
                        'description': '故障分类输出'
                    }
                ]
            },

            # Important features (mock)
            'important_features': [
                {
                    'feature': 'RMS值',
                    'value': statistics['rms'],
                    'significance': 0.92,
                    'domain': 'time_domain'
                },
                {
                    'feature': '峰值因子',
                    'value': statistics['max'] / (statistics['rms'] + 1e-8),
                    'significance': 0.87,
                    'domain': 'time_domain'
                },
                {
                    'feature': '主频幅值',
                    'value': frequency_analysis['dominant_frequency'],
                    'significance': 0.95,
                    'domain': 'frequency_domain'
                }
            ],

            # Layer contributions (mock)
            'layer_contributions': {
                'layer1_fft': 0.35,
                'layer1_wf': 0.28,
                'layer1_ht': 0.22,
                'layer1_identity': 0.15,
                'feature_extractor': 0.78,
                'attention': 0.65,
                'classifier': 1.0
            },

            # Key findings
            'key_findings': self._generate_key_findings(signal_data, statistics)
        }

        return explanation

    def save_explanation_batch(self,
                             explanations: List[Dict[str, Any]],
                             output_dir: Union[str, Path],
                             format: str = "json") -> List[Path]:
        """
        Save a batch of explanations to files.

        Args:
            explanations: List of explanation dictionaries
            output_dir: Output directory
            format: Output format ("json" or "pkl")

        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = []

        for i, explanation in enumerate(explanations):
            filename = f"explanation_{i+1:03d}_{timestamp}.{format}"
            filepath = output_dir / filename

            if format == "json":
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(explanation, f, ensure_ascii=False, indent=2)
            elif format == "pkl":
                with open(filepath, 'wb') as f:
                    pickle.dump(explanation, f)

            saved_files.append(filepath)

        return saved_files


def create_demo_signal_data(fault_type: str = "inner_race") -> np.ndarray:
    """
    Create demonstration signal data with specific fault characteristics.

    Args:
        fault_type: Type of fault to simulate

    Returns:
        Simulated signal data
    """
    t = np.linspace(0, 4, 4096)  # 4 seconds at 1024 Hz

    # Base signal with some noise
    signal = np.random.randn(4096) * 0.5

    if fault_type == "inner_race":
        # Inner race fault characteristic frequencies
        # Add harmonics at specific frequencies
        signal += 2.0 * np.sin(2 * np.pi * 50 * t)  # 50 Hz fundamental
        signal += 1.5 * np.sin(2 * np.pi * 150 * t)  # 3rd harmonic
        signal += 1.0 * np.sin(2 * np.pi * 250 * t)  # 5th harmonic

        # Add some impact components
        impact_times = np.random.choice(len(t), size=10, replace=False)
        for impact_time in impact_times:
            if impact_time < len(t) - 100:
                signal[impact_time:impact_time+100] += 3.0 * np.exp(-0.05 * np.arange(100))

    elif fault_type == "misalignment":
        # Misalignment shows strong 1x and 2x running speed components
        signal += 2.5 * np.sin(2 * np.pi * 30 * t)  # 1x
        signal += 1.8 * np.sin(2 * np.pi * 60 * t)  # 2x
        signal += 0.8 * np.sin(2 * np.pi * 90 * t)  # 3x

    elif fault_type == "outer_race":
        # Outer race fault
        signal += 1.8 * np.sin(2 * np.pi * 40 * t)
        signal += 1.2 * np.sin(2 * np.pi * 120 * t)

    else:  # normal
        # Just some background noise and minor components
        signal += 0.3 * np.sin(2 * np.pi * 25 * t)
        signal += 0.2 * np.sin(2 * np.pi * 50 * t)

    return signal


# Example usage and testing functions
def test_bridge_integration():
    """Test the bridge integration with mock data."""

    print("🔗 测试Explainable_FD_Toolkit API桥接...")

    # Initialize bridge
    bridge = ExplainableToolkitBridge()

    # Create test signal data
    signal_data = create_demo_signal_data("inner_race")

    # Generate mock TSPN explanation
    explanation = bridge.generate_mock_tspn_explanation(
        signal_data=signal_data,
        fault_type="内圈故障",
        confidence=0.89
    )

    # Convert to intermediate representation
    ir = bridge.convert_toolkit_explanation_to_ir(
        explanation=explanation,
        signal_data=signal_data,
        device_context={
            "device_type": "滚动轴承",
            "operating_speed": 1800.0,
            "load_condition": "正常载荷"
        }
    )

    print(f"✅ 成功转换解释结果")
    print(f"   故障类型: {ir.fault_info.fault_type}")
    print(f"   置信度: {ir.fault_info.confidence:.1%}")
    print(f"   关键发现: {len(ir.signal_analysis.key_findings)} 项")
    print(f"   重要特征: {len(ir.technical_explanation.important_features)} 项")

    return ir


if __name__ == "__main__":
    # Run test
    test_bridge_integration()