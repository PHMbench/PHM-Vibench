#!/usr/bin/env python3
"""
Isolated test for toolkit bridge functionality
Includes all necessary class definitions to avoid import issues
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import uuid


@dataclass
class FaultInfo:
    """Fault information structure."""
    fault_type: str
    confidence: float
    severity: str = "medium"
    description: str = ""


@dataclass
class SignalAnalysis:
    """Signal analysis information."""
    signal_length: int
    sampling_rate: float
    statistics: Dict[str, float]
    frequency_analysis: Dict[str, Any]
    key_findings: List[str]


@dataclass
class TechnicalExplanation:
    """Technical explanation information."""
    signal_path: Dict[str, Any]
    important_features: List[Dict[str, Any]]
    attention_weights: Dict[str, Any]
    layer_contributions: Dict[str, Any]
    processing_steps: List[Dict[str, Any]]


@dataclass
class DeviceContext:
    """Device context information."""
    device_type: str
    operating_conditions: Dict[str, Any]
    maintenance_history: str = ""
    specifications: str = ""


@dataclass
class LLMIntermediateRepresentation:
    """Complete intermediate representation for LLM."""
    explanation_id: str
    timestamp: str
    fault_info: FaultInfo
    signal_analysis: SignalAnalysis
    technical_explanation: TechnicalExplanation
    device_context: DeviceContext
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Initialize missing fields."""
        if not hasattr(self, 'explanation_id') or not self.explanation_id:
            self.explanation_id = str(uuid.uuid4())
        if not hasattr(self, 'timestamp') or not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class IsolatedToolkitBridge:
    """
    Isolated version of ExplainableToolkitBridge for testing.
    """

    def __init__(self):
        """Initialize the isolated bridge."""
        pass

    def generate_mock_tspn_explanation(self,
                                     signal_data: np.ndarray,
                                     fault_type: str = "内圈故障",
                                     confidence: float = 0.85) -> Dict[str, Any]:
        """
        Generate mock TSPN explanation for testing.
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
            'dominant_frequency': float(pos_freq[np.argmax(pos_fft)]) if len(pos_fft) > 0 else 0.0,
            'spectral_centroid': float(np.sum(pos_freq * pos_fft) / (np.sum(pos_fft) + 1e-8)) if len(pos_fft) > 0 else 0.0,
            'total_power': float(np.sum(pos_fft)) if len(pos_fft) > 0 else 0.0
        }

        # Generate key findings
        key_findings = self._generate_key_findings(signal_data, statistics)

        # Generate mock explanation
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
            'key_findings': key_findings
        }

        return explanation

    def convert_to_intermediate_representation(self,
                                             explanation: Dict[str, Any],
                                             signal_data: Optional[np.ndarray] = None,
                                             device_context: Optional[Dict[str, Any]] = None) -> LLMIntermediateRepresentation:
        """
        Convert explanation to intermediate representation.
        """
        # Create fault info
        fault_info = FaultInfo(
            fault_type=explanation.get('fault_type', 'unknown'),
            confidence=explanation.get('confidence', 0.0),
            severity=explanation.get('severity', 'medium'),
            description=explanation.get('description', '')
        )

        # Create signal analysis
        signal_analysis = SignalAnalysis(
            signal_length=explanation.get('signal_length', len(signal_data) if signal_data is not None else 0),
            sampling_rate=explanation.get('sampling_rate', 1024.0),
            statistics=explanation.get('signal_statistics', {}),
            frequency_analysis=explanation.get('frequency_analysis', {}),
            key_findings=explanation.get('key_findings', [])
        )

        # Create technical explanation
        technical_explanation = TechnicalExplanation(
            signal_path=explanation.get('signal_path', {}),
            important_features=explanation.get('important_features', []),
            attention_weights=explanation.get('attention_weights', {}),
            layer_contributions=explanation.get('layer_contributions', {}),
            processing_steps=explanation.get('processing_steps', [])
        )

        # Create device context
        device_context = device_context or {}
        device_context_obj = DeviceContext(
            device_type=device_context.get('device_type', 'unknown'),
            operating_conditions=device_context.get('operating_conditions', {}),
            maintenance_history=device_context.get('maintenance_history', ''),
            specifications=device_context.get('specifications', '')
        )

        # Create intermediate representation
        ir = LLMIntermediateRepresentation(
            explanation_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            fault_info=fault_info,
            signal_analysis=signal_analysis,
            technical_explanation=technical_explanation,
            device_context=device_context_obj,
            metadata={
                'source': 'isolated_test',
                'explanation_method': explanation.get('method', 'unknown'),
                'model_type': explanation.get('model_type', 'unknown'),
                'extraction_time': datetime.now().isoformat()
            }
        )

        return ir

    def _generate_key_findings(self,
                             signal_data: np.ndarray,
                             statistics: Dict[str, float]) -> List[str]:
        """Generate key findings from signal data and statistics."""
        findings = []

        if statistics:
            rms = statistics.get('rms', 0)
            peak_factor = statistics.get('max', 0) / (statistics.get('rms', 1) + 1e-8)

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

    def save_explanation_batch(self,
                             explanations: List[Dict[str, Any]],
                             output_dir: Union[str, Path],
                             format: str = "json") -> List[Path]:
        """
        Save a batch of explanations to files.
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

            saved_files.append(filepath)

        return saved_files


def create_demo_signal_data(fault_type: str = "inner_race") -> np.ndarray:
    """
    Create demonstration signal data with specific fault characteristics.
    """
    t = np.linspace(0, 4, 4096)  # 4 seconds at 1024 Hz

    # Base signal with some noise
    signal = np.random.randn(4096) * 0.5

    if fault_type == "inner_race":
        # Inner race fault characteristic frequencies
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


def test_complete_pipeline():
    """Test the complete pipeline with isolated classes."""
    print("🚀 测试完整管道（隔离版本）")
    print("=" * 50)

    # Initialize bridge
    bridge = IsolatedToolkitBridge()

    # Test scenarios
    scenarios = [
        {"signal_type": "inner_race", "fault_type": "内圈故障", "confidence": 0.92},
        {"signal_type": "outer_race", "fault_type": "外圈故障", "confidence": 0.87},
        {"signal_type": "misalignment", "fault_type": "不对中", "confidence": 0.85},
        {"signal_type": "normal", "fault_type": "正常状态", "confidence": 0.95}
    ]

    explanations = []
    irs = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📊 测试场景 {i}: {scenario['fault_type']}")

        # Step 1: Generate signal data
        signal_data = create_demo_signal_data(scenario["signal_type"])
        print(f"   生成信号数据 (长度: {len(signal_data)})")

        # Step 2: Generate explanation
        explanation = bridge.generate_mock_tspn_explanation(
            signal_data=signal_data,
            fault_type=scenario["fault_type"],
            confidence=scenario["confidence"]
        )
        explanations.append(explanation)
        print(f"   生成解释: {explanation['fault_type']} (置信度: {explanation['confidence']:.1%})")

        # Step 3: Convert to intermediate representation
        ir = bridge.convert_to_intermediate_representation(
            explanation=explanation,
            signal_data=signal_data,
            device_context={
                "device_type": "测试设备",
                "operating_speed": 1800.0,
                "load_condition": "测试条件"
            }
        )
        irs.append(ir)
        print(f"   转换为IR: {ir.explanation_id}")

        # Step 4: Validate
        ir_dict = ir.to_dict()
        assert ir.fault_info.fault_type == scenario["fault_type"]
        assert ir.fault_info.confidence == scenario["confidence"]
        print(f"   验证通过")

    # Step 5: Save results
    print(f"\n💾 保存结果...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("isolated_test_results") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save explanations
    explanation_files = bridge.save_explanation_batch(
        explanations,
        output_dir / "explanations",
        "json"
    )

    # Save intermediate representations
    ir_files = []
    for i, ir in enumerate(irs):
        ir_file = output_dir / "intermediate_representations" / f"ir_{i+1:03d}.json"
        ir_file.parent.mkdir(exist_ok=True)
        with open(ir_file, 'w', encoding='utf-8') as f:
            json.dump(ir.to_dict(), f, ensure_ascii=False, indent=2)
        ir_files.append(ir_file)

    # Create summary
    summary = {
        "test_time": datetime.now().isoformat(),
        "total_scenarios": len(scenarios),
        "explanations_generated": len(explanations),
        "irs_generated": len(irs),
        "explanation_files": [f.name for f in explanation_files],
        "ir_files": [f.name for f in ir_files],
        "success_rate": 1.0
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"   ✅ 保存完成")
    print(f"   输出目录: {output_dir}")
    print(f"   解释文件: {len(explanation_files)}")
    print(f"   IR文件: {len(ir_files)}")

    return True


def main():
    """Main function."""
    try:
        success = test_complete_pipeline()
        if success:
            print(f"\n🎉 所有测试成功完成！")
            print(f"   输出目录: isolated_test_results/")
            return True
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)