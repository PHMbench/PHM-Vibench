"""
Signal Encoder for LLM Explanation

This module converts technical signal processing explanations and features
into natural language descriptions that LLMs can understand and process.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Union
import scipy.signal as signal


class SignalEncoder:
    """
    Encodes signal processing results and technical features into natural language
    descriptions for LLM-based explanations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the signal encoder.

        Args:
            config: Configuration dictionary for encoding parameters
        """
        self.config = config or {}

        # Configuration parameters
        self.max_length = self.config.get('max_length', 1000)
        self.include_technical_details = self.config.get('include_technical_details', True)
        self.include_physical_interpretation = self.config.get('include_physical_interpretation', True)
        self.language = self.config.get('language', 'zh')  # 'zh' or 'en'

        # Technical terminology mappings
        self.technical_terms = self._initialize_terminology()

    def encode_explanation(self, explanation: Any) -> str:
        """
        Encode an explanation object into natural language description.

        Args:
            explanation: Explanation object or dictionary containing explanation data

        Returns:
            Natural language description string
        """
        encoded_parts = []

        # Encode different types of explanations
        if hasattr(explanation, 'get_data'):
            data = explanation.get_data()
            encoded_parts.append(self._encode_header(explanation))

        # Signal path encoding
        if hasattr(explanation, 'get_data') and 'signal_path' in explanation.get_data():
            signal_path = explanation.get_data()['signal_path']
            signal_path_text = self._encode_signal_path(signal_path)
            encoded_parts.append(signal_path_text)

        # Importance scores encoding
        if hasattr(explanation, 'get_data') and 'importance_scores' in explanation.get_data():
            importance_scores = explanation.get_data()['importance_scores']
            importance_text = self._encode_importance_scores(importance_scores)
            encoded_parts.append(importance_text)

        # Physical analysis encoding
        if hasattr(explanation, 'get_data') and 'physical_analysis' in explanation.get_data():
            physical_analysis = explanation.get_data()['physical_analysis']
            physical_text = self._encode_physical_analysis(physical_analysis)
            encoded_parts.append(physical_text)

        # Operator importance encoding
        if hasattr(explanation, 'get_data') and 'operator_importance' in explanation.get_data():
            operator_importance = explanation.get_data()['operator_importance']
            operator_text = self._encode_operator_importance(operator_importance)
            encoded_parts.append(operator_text)

        # Attention maps encoding
        if hasattr(explanation, 'get_data') and 'attention_maps' in explanation.get_data():
            attention_maps = explanation.get_data()['attention_maps']
            attention_text = self._encode_attention_maps(attention_maps)
            encoded_parts.append(attention_text)

        # Join all encoded parts
        if encoded_parts:
            encoded_text = "\n\n".join(encoded_parts)
            # Truncate if too long
            if len(encoded_text) > self.max_length:
                encoded_text = encoded_text[:self.max_length] + "..."
        else:
            encoded_text = "没有检测到可编码的解释信息"

        return encoded_text

    def encode_technical_summary(self, technical_summary: Dict[str, Any]) -> str:
        """
        Encode technical summary for LLM processing.

        Args:
            technical_summary: Dictionary containing technical analysis results

        Returns:
            Natural language description of technical findings
        """
        encoded_parts = []

        # Signal characteristics
        if 'signal_characteristics' in technical_summary:
            signal_chars = technical_summary['signal_characteristics']
            chars_text = self._encode_signal_characteristics(signal_chars)
            encoded_parts.append(f"信号特征分析：{chars_text}")

        # Prediction confidence
        if 'prediction_confidence' in technical_summary:
            confidence = technical_summary['prediction_confidence']
            conf_text = self._encode_prediction_confidence(confidence)
            encoded_parts.append(f"预测置信度：{conf_text}")

        # Key features
        if 'key_features' in technical_summary:
            features = technical_summary['key_features']
            features_text = self._encode_key_features(features)
            encoded_parts.append(f"关键特征：{features_text}")

        # Anomaly indicators
        if 'anomaly_indicators' in technical_summary:
            anomalies = technical_summary['anomaly_indicators']
            anomalies_text = self._encode_anomaly_indicators(anomalies)
            encoded_parts.append(f"异常指标：{anomalies_text}")

        return "\n".join(encoded_parts) if encoded_parts else "未发现异常指标"

    def encode_context_information(self, context: Dict[str, Any]) -> str:
        """
        Encode diagnostic context information for LLM processing.

        Args:
            context: Dictionary containing diagnostic context

        Returns:
            Natural language description of diagnostic context
        """
        context_parts = []

        # Device information
        if 'model_name' in context:
            context_parts.append(f"诊断模型：{context['model_name']}")

        # Input statistics
        if 'input_statistics' in context:
            stats = context['input_statistics']
            stats_text = self._encode_input_statistics(stats)
            context_parts.append(f"输入信号统计：{stats_text}")

        # Device parameters
        if 'device_parameters' in context:
            device_params = context['device_parameters']
            device_text = self._encode_device_parameters(device_params)
            context_parts.append(f"设备参数：{device_text}")

        # Operating conditions
        if 'operating_conditions' in context:
            conditions = context['operating_conditions']
            conditions_text = self._encode_operating_conditions(conditions)
            context_parts.append(f"运行工况：{conditions_text}")

        return "\n".join(context_parts) if context_parts else "无可用上下文信息"

    def encode_signal_features(self, signal_data: torch.Tensor, features: Dict[str, Any]) -> str:
        """
        Encode signal features and characteristics into natural language.

        Args:
            signal_data: Input signal tensor
            features: Dictionary containing extracted features

        Returns:
            Natural language description of signal features
        """
        feature_parts = []

        # Basic signal description
        signal_description = self._describe_signal_basic(signal_data)
        feature_parts.append(f"信号基本信息：{signal_description}")

        # Time domain features
        if 'time_domain' in features:
            time_features = features['time_domain']
            time_text = self._encode_time_domain_features(time_features)
            feature_parts.append(f"时域特征：{time_text}")

        # Frequency domain features
        if 'frequency_domain' in features:
            freq_features = features['frequency_domain']
            freq_text = self._encode_frequency_domain_features(freq_features)
            feature_parts.append(f"频域特征：{freq_text}")

        # Statistical features
        if 'statistical' in features:
            stat_features = features['statistical']
            stat_text = self._encode_statistical_features(stat_features)
            feature_parts.append(f"统计特征：{stat_text}")

        return "\n".join(feature_parts)

    def _encode_header(self, explanation: Any) -> str:
        """Encode explanation header information."""
        method = explanation.get_meta('method', 'unknown')
        model_name = explanation.get_meta('model_name', 'unknown')

        if self.language == 'zh':
            return f"诊断方法：{method}，使用模型：{model_name}"
        else:
            return f"Diagnosis method: {method}, Model: {model_name}"

    def _encode_signal_path(self, signal_path: List[Dict[str, Any]]) -> str:
        """Encode signal processing path into natural language."""
        if self.language == 'zh':
            path_description = "信号处理路径：\n"
        else:
            path_description = "Signal processing path:\n"

        descriptions = []

        for i, step in enumerate(signal_path):
            # Skip input layer
            if step.get('operator_type') == 'raw_signal':
                continue

            layer_name = step.get('layer_name', f'layer_{i}')
            operator_type = step.get('operator_type', 'unknown_operator')

            description = f"{i+1}. {layer_name}层：经过{operator_type}处理"

            # Add energy change information if available
            if 'input_stats' in step and 'output_stats' in step:
                input_energy = step['input_stats'].get('energy', 0)
                output_energy = step['output_stats'].get('energy', 0)
                if input_energy > 0:
                    energy_change = (output_energy - input_energy) / input_energy
                    if self.language == 'zh':
                        if energy_change > 0.2:
                            description += f"，能量显著增加{energy_change:.1%}"
                        elif energy_change < -0.2:
                            description += f"，能量显著减少{energy_change:.1%}"
                    else:
                        if energy_change > 0.2:
                            description += f", energy significantly increased by {energy_change:.1%}"
                        elif energy_change < -0.2:
                            description += f", energy significantly decreased by {energy_change:.1%}"

            descriptions.append(description)

        return path_description + "\n".join(descriptions)

    def _encode_importance_scores(self, importance_scores: Dict[str, Any]) -> str:
        """Encode importance scores into natural language."""
        if not importance_scores:
            return "无重要性分数信息"

        if self.language == 'zh':
            description = "模块重要性分析：\n"
        else:
            description = "Module importance analysis:\n"

        # Sort by importance
        sorted_items = []
        for name, scores in importance_scores.items():
            if isinstance(scores, dict):
                # Use combined_score if available, otherwise use first value
                score = scores.get('combined_score', list(scores.values())[0])
            else:
                score = scores
            sorted_items.append((name, float(score)))

        sorted_items.sort(key=lambda x: x[1], reverse=True)

        descriptions = []
        for i, (name, score) in enumerate(sorted_items[:5]):  # Top 5
            if self.language == 'zh':
                descriptions.append(f"{i+1}. {name}：重要性分数{score:.3f}")
            else:
                descriptions.append(f"{i+1}. {name}: importance score {score:.3f}")

        return description + "\n".join(descriptions)

    def _encode_physical_analysis(self, physical_analysis: Dict[str, Any]) -> str:
        """Encode physical analysis into natural language."""
        if not physical_analysis:
            return "无物理分析信息"

        if self.language == 'zh':
            description = "物理分析结果：\n"
        else:
            description = "Physical analysis results:\n"

        descriptions = []

        # Energy flow analysis
        if 'energy_flow' in physical_analysis:
            energy_flow = physical_analysis['energy_flow']
            max_change = 0
            max_change_layer = ""

            for flow_info in energy_flow:
                change = abs(flow_info.get('energy_change_ratio', 0))
                if change > max_change:
                    max_change = change
                    max_change_layer = flow_info.get('layer_name', 'unknown')

            if max_change > 0:
                if self.language == 'zh':
                    descriptions.append(f"最大能量变换：{max_change_layer}层，变化率{max_change:.1%}")
                else:
                    descriptions.append(f"Maximum energy transformation: {max_change_layer} layer, change ratio {max_change:.1%}")

        # Frequency evolution
        if 'frequency_evolution' in physical_analysis:
            freq_evo = physical_analysis['frequency_evolution']
            if freq_evo:
                dominant_shifts = []
                for evo in freq_evo:
                    shift = abs(evo.get('frequency_shift', 0))
                    if shift > 10:  # Significant frequency shift
                        dominant_shifts.append((evo.get('layer_name', 'unknown'), shift))

                if dominant_shifts:
                    if self.language == 'zh':
                        descriptions.append(f"显著频率偏移：{len(dominant_shifts)}个层级")
                    else:
                        descriptions.append(f"Significant frequency shifts: {len(dominant_shifts)} layers")

        # Dominant transformations
        if 'dominant_transformations' in physical_analysis:
            transformations = physical_analysis['dominant_transformations']
            if transformations:
                trans_types = [t[0] for t in transformations[:3]]  # Top 3
                if self.language == 'zh':
                    descriptions.append(f"主导变换类型：{', '.join(trans_types)}")
                else:
                    descriptions.append(f"Dominant transformation types: {', '.join(trans_types)}")

        return description + "\n".join(descriptions)

    def _encode_operator_importance(self, operator_importance: Dict[str, Any]) -> str:
        """Encode operator importance into natural language."""
        if not operator_importance:
            return "无算子重要性信息"

        if self.language == 'zh':
            description = "算子重要性分析：\n"
        else:
            description = "Operator importance analysis:\n"

        descriptions = []

        for operator_name, scores in operator_importance.items():
            if isinstance(scores, dict):
                # Use specific importance metrics
                if 'physical_meaning' in scores:
                    meaning = scores['physical_meaning']
                    if 'importance_score' in scores:
                        importance = scores['importance_score']
                        descriptions.append(f"{operator_name} ({meaning}): 重要性{importance:.3f}")
                    else:
                        descriptions.append(f"{operator_name} ({meaning}): 中等重要性")
                else:
                    # Use the first available metric
                    first_metric = list(scores.keys())[0]
                    first_value = list(scores.values())[0]
                    descriptions.append(f"{operator_name}: {first_metric}{first_value:.3f}")
            else:
                descriptions.append(f"{operator_name}: 重要性{scores:.3f}")

        return description + "\n".join(descriptions)

    def _encode_attention_maps(self, attention_maps: Dict[str, torch.Tensor]) -> str:
        """Encode attention maps into natural language."""
        if not attention_maps:
            return "无注意力图信息"

        if self.language == 'zh':
            description = "注意力权重分析：\n"
        else:
            description = "Attention weight analysis:\n"

        descriptions = []

        for layer_name, attention_weights in attention_maps.items():
            if isinstance(attention_weights, torch.Tensor):
                # Convert to numpy for analysis
                attention_np = attention_weights.detach().cpu().numpy()

                # Find maximum attention
                max_attention = np.max(attention_np)
                mean_attention = np.mean(attention_np)

                # Find location of max attention
                max_idx = np.unravel_index(np.argmax(attention_np, axis=None), attention_np.shape)

                if self.language == 'zh':
                    descriptions.append(
                        f"{layer_name}层：最大注意力{max_attention:.3f}，"
                        f"平均注意力{mean_attention:.3f}，"
                        f"最大注意力位置{max_idx}"
                    )
                else:
                    descriptions.append(
                        f"{layer_name} layer: max attention {max_attention:.3f}, "
                        f"mean attention {mean_attention:.3f}, "
                        f"max attention location {max_idx}"
                    )

        return description + "\n".join(descriptions)

    def _encode_signal_characteristics(self, signal_chars: Dict[str, Any]) -> str:
        """Encode signal characteristics into natural language."""
        if self.language == 'zh':
            description_parts = []
        else:
            description_parts = []

        # Dominant frequency
        dominant_freq = signal_chars.get('dominant_frequency', 0)
        if dominant_freq > 0:
            if self.language == 'zh':
                description_parts.append(f"主导频率{dominant_freq:.1f}Hz")
            else:
                description_parts.append(f"Dominant frequency: {dominant_freq:.1f} Hz")

        # Spectral centroid
        spectral_centroid = signal_chars.get('spectral_centroid', 0)
        if spectral_centroid > 0:
            if self.language == 'zh':
                description_parts.append(f"频谱中心{spectral_centroid:.1f}Hz")
            else:
                description_parts.append(f"Spectral centroid: {spectral_centroid:.1f} Hz")

        # Frequency content type
        freq_type = signal_chars.get('frequency_content', 'unknown')
        signal_type = signal_chars.get('signal_type', 'unknown')

        if self.language == 'zh':
            if freq_type == 'low_frequency':
                description_parts.append("主要为低频成分")
            elif freq_type == 'high_frequency':
                description_parts.append("主要为高频成分")
            elif freq_type == 'mixed':
                description_parts.append("包含多种频率成分")

            if signal_type == 'impulsive':
                description_parts.append("呈现冲击性特征")
            elif signal_type == 'periodic':
                description_parts.append("呈现周期性特征")
            elif signal_type == 'random':
                description_parts.append("呈现随机噪声特征")
        else:
            if freq_type == 'low_frequency':
                description_parts.append("Mainly low-frequency content")
            elif freq_type == 'high_frequency':
                description_parts.append("Mainly high-frequency content")
            elif freq_type == 'mixed':
                description_parts.append("Mixed frequency content")

            if signal_type == 'impulsive':
                description_parts.append("Shows impulsive characteristics")
            elif signal_type == 'periodic':
                description_parts.append("Shows periodic characteristics")
            elif signal_type == 'random':
                description_parts.append("Shows random noise characteristics")

        return "，".join(description_parts)

    def _encode_prediction_confidence(self, confidence: Dict[str, Any]) -> str:
        """Encode prediction confidence into natural language."""
        conf_value = confidence.get('confidence', 0)
        pred_class = confidence.get('predicted_class', -1)
        entropy = confidence.get('entropy', 0)

        if self.language == 'zh':
            confidence_level = "高" if conf_value > 0.8 else ("中" if conf_value > 0.5 else "低")
            return f"预测置信度{conf_value:.3f}（{confidence_level}），预测类别{pred_class}，信息熵{entropy:.3f}"
        else:
            confidence_level = "high" if conf_value > 0.8 else ("medium" if conf_value > 0.5 else "low")
            return f"Prediction confidence: {conf_value:.3f} ({confidence_level}), predicted class: {pred_class}, entropy: {entropy:.3f}"

    def _encode_key_features(self, features: List[Dict[str, Any]]) -> str:
        """Encode key features into natural language."""
        if not features:
            return "无关键特征信息"

        if self.language == 'zh':
            description_parts = []
        else:
            description_parts = []

        for i, feature in enumerate(features[:5]):  # Top 5 features
            feature_type = feature.get('type', 'unknown')
            feature_value = feature.get('value', 'unknown')
            importance = feature.get('importance', 0)

            if self.language == 'zh':
                description_parts.append(f"{i+1}. {feature_type}：{feature_value}（重要性：{importance:.2f}）")
            else:
                description_parts.append(f"{i+1}. {feature_type}: {feature_value} (importance: {importance:.2f})")

        return "；".join(description_parts)

    def _encode_anomaly_indicators(self, anomalies: Dict[str, Any]) -> str:
        """Encode anomaly indicators into natural language."""
        if not anomalies:
            return "无异常指标"

        if self.language == 'zh':
            descriptions = []
        else:
            descriptions = []

        for anomaly_type, info in anomalies.items():
            if info.get('detected', False):
                severity = info.get('severity', 'unknown')
                value = info.get('value', 'unknown')

                if self.language == 'zh':
                    descriptions.append(f"{anomaly_type}检测到异常，严重程度：{severity}，数值：{value}")
                else:
                    descriptions.append(f"{anomaly_type}: anomaly detected, severity: {severity}, value: {value}")

        return "；".join(descriptions)

    def _encode_input_statistics(self, stats: Dict[str, float]) -> str:
        """Encode input statistics into natural language."""
        if self.language == 'zh':
            stats_desc = []
            if 'rms' in stats:
                stats_desc.append(f"RMS值{stats['rms']:.3f}")
            if 'crest_factor' in stats:
                stats_desc.append(f"波峰因子{stats['crest_factor']:.3f}")
            if 'skewness' in stats:
                stats_desc.append(f"偏度{stats['skewness']:.3f}")
            return "，".join(stats_desc)
        else:
            stats_desc = []
            if 'rms' in stats:
                stats_desc.append(f"RMS: {stats['rms']:.3f}")
            if 'crest_factor' in stats:
                stats_desc.append(f"Crest factor: {stats['crest_factor']:.3f}")
            if 'skewness' in stats:
                stats_desc.append(f"Skewness: {stats['skewness']:.3f}")
            return ", ".join(stats_desc)

    def _encode_device_parameters(self, device_params: Dict[str, Any]) -> str:
        """Encode device parameters into natural language."""
        if self.language == 'zh':
            params_desc = []
            if device_params.get('device_type'):
                params_desc.append(f"设备类型：{device_params['device_type']}")
            if device_params.get('rated_speed'):
                params_desc.append(f"额定转速：{device_params['rated_speed']} RPM")
            if device_params.get('load_condition'):
                params_desc.append(f"负载状况：{device_params['load_condition']}")
            return "，".join(params_desc)
        else:
            params_desc = []
            if device_params.get('device_type'):
                params_desc.append(f"Device type: {device_params['device_type']}")
            if device_params.get('rated_speed'):
                params_desc.append(f"Rated speed: {device_params['rated_speed']} RPM")
            if device_params.get('load_condition'):
                params_desc.append(f"Load condition: {device_params['load_condition']}")
            return ", ".join(params_desc)

    def _encode_operating_conditions(self, conditions: Dict[str, Any]) -> str:
        """Encode operating conditions into natural language."""
        if self.language == 'zh':
            conditions_desc = []
            if conditions.get('timestamp'):
                conditions_desc.append(f"时间：{conditions['timestamp']}")
            if conditions.get('temperature'):
                conditions_desc.append(f"温度：{conditions['temperature']}")
            if conditions.get('noise_level'):
                conditions_desc.append(f"噪声水平：{conditions['noise_level']}")
            return "，".join(conditions_desc)
        else:
            conditions_desc = []
            if conditions.get('timestamp'):
                conditions_desc.append(f"Time: {conditions['timestamp']}")
            if conditions.get('temperature'):
                conditions_desc.append(f"Temperature: {conditions['temperature']}")
            if conditions.get('noise_level'):
                conditions_desc.append(f"Noise level: {conditions['noise_level']}")
            return ", ".join(conditions_desc)

    def _describe_signal_basic(self, signal_data: torch.Tensor) -> str:
        """Provide basic description of signal."""
        if len(signal_data.shape) == 3:
            batch, length, channels = signal_data.shape
        elif len(signal_data.shape) == 2:
            batch, length = signal_data.shape
            channels = 1
        else:
            batch = signal_data.shape[0]
            length = signal_data.shape[1]
            channels = 1

        if self.language == 'zh':
            return f"振动信号：{batch}个样本，长度{length}，{channels}个通道"
        else:
            return f"Vibration signal: {batch} samples, length {length}, {channels} channels"

    def _encode_time_domain_features(self, time_features: Dict[str, Any]) -> str:
        """Encode time domain features."""
        if self.language == 'zh':
            return f"时域特征：{time_features.get('description', '无描述')}"
        else:
            return f"Time domain features: {time_features.get('description', 'No description')}"

    def _encode_frequency_domain_features(self, freq_features: Dict[str, Any]) -> str:
        """Encode frequency domain features."""
        if self.language == 'zh':
            return f"频域特征：{freq_features.get('description', '无描述')}"
        else:
            return f"Frequency domain features: {freq_features.get('description', 'No description')}"

    def _encode_statistical_features(self, stat_features: Dict[str, Any]) -> str:
        """Encode statistical features."""
        if self.language == 'zh':
            return f"统计特征：{stat_features.get('description', '无描述')}"
        else:
            return f"Statistical features: {stat_features.get('description', 'No description')}"

    def _initialize_terminology(self) -> Dict[str, Dict[str, str]]:
        """Initialize technical terminology mappings."""
        if self.language == 'zh':
            return {
                'operators': {
                    'FFT': '快速傅里叶变换',
                    'WF': '小波滤波',
                    'HT': '希尔伯特变换',
                    'I': '恒等变换',
                    'LNO': '拉普拉斯神经算子'
                },
                'faults': {
                    'inner_race': '内圈故障',
                    'outer_race': '外圈故障',
                    'ball_defect': '滚动体故障',
                    'cage_damage': '保持架故障'
                },
                'features': {
                    'rms': '均方根值',
                    'peak': '峰值',
                    'crest_factor': '波峰因子',
                    'skewness': '偏度',
                    'kurtosis': '峭度'
                }
            }
        else:
            return {
                'operators': {
                    'FFT': 'Fast Fourier Transform',
                    'WF': 'Wavelet Filter',
                    'HT': 'Hilbert Transform',
                    'I': 'Identity',
                    'LNO': 'Laplacian Neural Operator'
                },
                'faults': {
                    'inner_race': 'inner race fault',
                    'outer_race': 'outer race fault',
                    'ball_defect': 'ball defect',
                    'cage_damage': 'cage damage'
                },
                'features': {
                    'rms': 'Root Mean Square',
                    'peak': 'Peak value',
                    'crest_factor': 'Crest factor',
                    'skewness': 'Skewness',
                    'kurtosis': 'Kurtosis'
                }
            }