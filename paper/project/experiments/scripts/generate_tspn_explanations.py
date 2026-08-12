#!/usr/bin/env python3
"""
Generate Real TSPN Model Explanations

This script loads a trained TSPN model and generates real explanations
using the Explainable_FD_Toolkit integration.
"""

import sys
import os
import argparse
import json
import pickle
import numpy as np
import torch
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add paths
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
code_dir = project_root / "code"
main_repo_root = project_root.parent

sys.path.insert(0, str(code_dir))
sys.path.insert(0, str(main_repo_root))

from llm_explainable_toolkit.core.toolkit_bridge import (
    ExplainableToolkitBridge,
    create_demo_signal_data
)

# Import TSPN and related modules
try:
    from model.TSPN_explainable import ExplainableTSPN
    from model.TSPN import load_model
except ImportError:
    print("⚠️  无法导入TSPN_explainable，将使用模拟数据")
    ExplainableTSPN = None


class TSPNExplanationGenerator:
    """
    Generate explanations from trained TSPN models.
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 device: str = "cpu"):
        """
        Initialize the explanation generator.

        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to model configuration
            device: Device to run model on
        """
        self.model_path = model_path
        self.config_path = config_path
        self.device = torch.device(device)
        self.model = None
        self.bridge = ExplainableToolkitBridge()
        self.config = {}

        # Load model and configuration if provided
        if model_path and config_path:
            self.load_model()

    def load_model(self):
        """Load trained TSPN model with explainability features."""
        if self.model_path is None or self.config_path is None:
            raise ValueError("Both model_path and config_path must be provided")

        try:
            # Load configuration
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)

            # Initialize explainable TSPN model
            if ExplainableTSPN is not None:
                self.model = ExplainableTSPN(self.config)

                # Load model weights
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)

                self.model.to(self.device)
                self.model.eval()

                print(f"✅ 成功加载TSPN模型: {self.model_path}")
                print(f"   模型配置: {self.config_path}")
                print(f"   运行设备: {self.device}")

            else:
                print("⚠️  ExplainableTSPN不可用，将使用模拟解释")
                self.model = None

        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            print("   将使用模拟解释模式")
            self.model = None

    def generate_explanation_for_signal(self,
                                      signal_data: np.ndarray,
                                      fault_type: Optional[str] = None,
                                      use_real_model: bool = True) -> Dict[str, Any]:
        """
        Generate explanation for a single signal.

        Args:
            signal_data: Input signal data
            fault_type: Expected fault type (for mock explanations)
            use_real_model: Whether to use real model or mock

        Returns:
            Explanation dictionary
        """
        if use_real_model and self.model is not None:
            return self._generate_real_explanation(signal_data)
        else:
            # Generate mock explanation
            fault_type = fault_type or "内圈故障"
            confidence = np.random.uniform(0.75, 0.95)

            return self.bridge.generate_mock_tspn_explanation(
                signal_data=signal_data,
                fault_type=fault_type,
                confidence=confidence
            )

    def _generate_real_explanation(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Generate real explanation using TSPN model."""
        try:
            # Convert to tensor
            if len(signal_data.shape) == 1:
                signal_tensor = torch.from_numpy(signal_data).float().unsqueeze(0).unsqueeze(-1)
            else:
                signal_tensor = torch.from_numpy(signal_data).float()

            signal_tensor = signal_tensor.to(self.device)

            with torch.no_grad():
                # Get model prediction
                outputs = self.model(signal_tensor)
                probabilities = torch.softmax(outputs, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0, predicted_class].item()

                # Get model explanation
                explanation_data = self.model.explain(signal_tensor, method='comprehensive')

            # Map class to fault type (this should be adapted based on actual dataset)
            fault_type_map = {
                0: "正常",
                1: "内圈故障",
                2: "外圈故障",
                3: "滚动体故障",
                4: "保持架故障",
                5: "不对中",
                6: "不平衡",
                7: "摩擦",
                8: "松动",
                9: "复合故障"
            }

            fault_type = fault_type_map.get(predicted_class, f"未知故障{predicted_class}")

            # Convert to standard explanation format
            explanation = {
                'fault_type': fault_type,
                'confidence': confidence,
                'severity': 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low',
                'description': f'TSPN模型检测到{fault_type}，置信度为{confidence:.1%}',
                'method': 'TSPN_explainable',
                'model_type': 'Transparent Signal Processing Network',
                'predicted_class': predicted_class,
                'probabilities': probabilities[0].cpu().numpy().tolist(),

                # Extract signal path from model
                'signal_path': explanation_data.get('signal_path', {}),

                # Extract important features
                'important_features': explanation_data.get('important_features', []),

                # Extract layer contributions
                'layer_contributions': explanation_data.get('layer_contributions', {}),

                # Signal statistics
                'signal_statistics': {
                    'mean': float(np.mean(signal_data)),
                    'std': float(np.std(signal_data)),
                    'rms': float(np.sqrt(np.mean(signal_data**2))),
                    'max': float(np.max(signal_data)),
                    'min': float(np.min(signal_data)),
                    'energy': float(np.sum(signal_data**2))
                },

                # Frequency analysis
                'frequency_analysis': self._analyze_frequency(signal_data),

                # Additional explanation data
                'signal_length': len(signal_data),
                'sampling_rate': self.config.get('data', {}).get('sampling_rate', 1024.0),

                # Key findings
                'key_findings': self._extract_key_findings(explanation_data, signal_data)
            }

            return explanation

        except Exception as e:
            print(f"❌ 生成真实解释失败: {e}")
            print("   回退到模拟解释")
            # Fallback to mock explanation
            return self.bridge.generate_mock_tspn_explanation(
                signal_data=signal_data,
                fault_type="未知故障",
                confidence=0.5
            )

    def _analyze_frequency(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Analyze frequency content of signal."""
        # Compute FFT
        fft_vals = np.fft.fft(signal_data)
        fft_freq = np.fft.fftfreq(len(signal_data), 1/1024.0)

        # Only positive frequencies
        pos_mask = fft_freq > 0
        pos_freq = fft_freq[pos_mask]
        pos_fft = np.abs(fft_vals[pos_mask])

        if len(pos_fft) > 0:
            dominant_freq_idx = np.argmax(pos_fft)
            dominant_freq = pos_freq[dominant_freq_idx]
            dominant_power = pos_fft[dominant_freq_idx]
        else:
            dominant_freq = 0.0
            dominant_power = 0.0

        spectral_centroid = np.sum(pos_freq * pos_fft) / (np.sum(pos_fft) + 1e-8)

        return {
            'dominant_frequency': float(dominant_freq),
            'dominant_power': float(dominant_power),
            'spectral_centroid': float(spectral_centroid),
            'total_power': float(np.sum(pos_fft))
        }

    def _extract_key_findings(self,
                            explanation_data: Dict[str, Any],
                            signal_data: np.ndarray) -> List[str]:
        """Extract key findings from model explanation."""
        findings = []

        # Findings from explanation data
        if 'key_findings' in explanation_data:
            findings.extend(explanation_data['key_findings'])

        # Findings from signal analysis
        rms = np.sqrt(np.mean(signal_data**2))
        peak_factor = np.max(np.abs(signal_data)) / (rms + 1e-8)

        if rms > 5.0:
            findings.append(f"信号能量较高，RMS值为{rms:.2f}")

        if peak_factor > 4.0:
            findings.append(f"信号峰值因子较高({peak_factor:.2f})，可能存在冲击")

        # Findings from layer contributions
        layer_contributions = explanation_data.get('layer_contributions', {})
        if layer_contributions:
            top_layer = max(layer_contributions.items(), key=lambda x: x[1])
            findings.append(f"最重要的处理层为{top_layer[0]}，贡献度为{top_layer[1]:.2f}")

        return findings

    def generate_explanation_batch(self,
                                 signal_types: List[str],
                                 num_samples_per_type: int = 3,
                                 output_dir: str = "generated_explanations") -> List[Dict[str, Any]]:
        """
        Generate a batch of explanations for different signal types.

        Args:
            signal_types: List of signal types to generate
            num_samples_per_type: Number of samples per type
            output_dir: Output directory for saving explanations

        Returns:
            List of generated explanations
        """
        print(f"🔄 生成解释批次: {signal_types}")
        print(f"   每种类型样本数: {num_samples_per_type}")

        explanations = []

        for signal_type in signal_types:
            print(f"\n📊 处理信号类型: {signal_type}")

            for i in range(num_samples_per_type):
                # Generate signal data
                signal_data = create_demo_signal_data(signal_type)

                # Generate explanation
                explanation = self.generate_explanation_for_signal(
                    signal_data=signal_data,
                    fault_type=self._map_signal_to_fault(signal_type),
                    use_real_model=(self.model is not None)
                )

                # Add metadata
                explanation['metadata'] = {
                    'signal_type': signal_type,
                    'sample_index': i + 1,
                    'generation_time': datetime.now().isoformat(),
                    'model_used': self.model is not None
                }

                explanations.append(explanation)

                print(f"   样本 {i+1}: {explanation['fault_type']} "
                      f"(置信度: {explanation['confidence']:.1%})")

        # Save explanations
        if explanations:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            saved_files = self.bridge.save_explanation_batch(
                explanations,
                output_path,
                format="json"
            )

            print(f"\n💾 已保存 {len(saved_files)} 个解释文件到: {output_path}")

        return explanations

    def _map_signal_to_fault(self, signal_type: str) -> str:
        """Map signal type to fault type name."""
        fault_mapping = {
            "inner_race": "内圈故障",
            "outer_race": "外圈故障",
            "misalignment": "不对中",
            "normal": "正常状态"
        }
        return fault_mapping.get(signal_type, "未知故障")


def main():
    """Main function to generate TSPN explanations."""

    parser = argparse.ArgumentParser(description="Generate TSPN Model Explanations")
    parser.add_argument("--model_path", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--config_path", type=str, help="Path to model configuration")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run model on")
    parser.add_argument("--output_dir", type=str, default="generated_explanations",
                       help="Output directory for explanations")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of samples per signal type")
    parser.add_argument("--signal_types", nargs="+",
                       default=["inner_race", "outer_race", "misalignment", "normal"],
                       help="Signal types to generate")

    args = parser.parse_args()

    print("🚀 TSPN模型解释生成器")
    print("=" * 50)

    # Initialize generator
    generator = TSPNExplanationGenerator(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device
    )

    # Generate explanations
    explanations = generator.generate_explanation_batch(
        signal_types=args.signal_types,
        num_samples_per_type=args.num_samples,
        output_dir=args.output_dir
    )

    print(f"\n✅ 完成！共生成 {len(explanations)} 个解释")
    print(f"   输出目录: {args.output_dir}")

    # Summary statistics
    fault_counts = {}
    confidence_sum = 0.0
    for exp in explanations:
        fault_type = exp['fault_type']
        fault_counts[fault_type] = fault_counts.get(fault_type, 0) + 1
        confidence_sum += exp['confidence']

    print(f"\n📊 生成统计:")
    for fault_type, count in fault_counts.items():
        print(f"   {fault_type}: {count} 个")

    if explanations:
        avg_confidence = confidence_sum / len(explanations)
        print(f"   平均置信度: {avg_confidence:.1%}")


if __name__ == "__main__":
    main()