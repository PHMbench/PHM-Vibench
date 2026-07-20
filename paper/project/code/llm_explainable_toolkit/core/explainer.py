"""
LLM-Enhanced Explainer Core Module

Provides the main explanation generation functionality, integrating
signal processing analysis with natural language generation.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

try:
    from llm.llm_explainer import LLMExplainer as ExternalLLMExplainer
except ImportError:
    ExternalLLMExplainer = None

try:
    from knowledge.fault_knowledge_graph import FaultKnowledgeGraph
    from knowledge.terminology_mapper import TerminologyMapper
except ImportError:

    class FaultKnowledgeGraph:
        """Minimal local knowledge graph used when the legacy package is absent."""

        def __init__(self):
            self.fault_patterns = {
                "正常": {
                    "description": "设备运行状态正常",
                    "causes": ["无明显故障"],
                    "maintenance_steps": ["保持常规监测"],
                },
                "内圈故障": {
                    "description": "轴承内圈可能存在局部损伤或磨损",
                    "causes": ["材料疲劳", "润滑不足", "冲击载荷", "安装偏差"],
                    "maintenance_steps": [
                        "复核振动频谱和包络特征",
                        "检查轴承润滑和配合状态",
                        "安排停机复检或更换计划",
                    ],
                },
                "外圈故障": {
                    "description": "轴承外圈可能存在点蚀、裂纹或局部磨损",
                    "causes": ["载荷集中", "污染颗粒", "润滑退化"],
                    "maintenance_steps": [
                        "检查轴承座和载荷路径",
                        "复核外圈故障特征频率",
                        "制定维修窗口",
                    ],
                },
                "不平衡": {
                    "description": "转子质量分布不均可能导致 1x 转频振动增强",
                    "causes": ["积灰", "叶轮磨损", "装配偏心"],
                    "maintenance_steps": ["检查转子清洁度", "执行动平衡校正"],
                },
                "不对中": {
                    "description": "联轴器或轴系不对中可能导致 1x/2x 转频异常",
                    "causes": ["安装误差", "基础松动", "热变形"],
                    "maintenance_steps": ["检查联轴器", "复测轴线同心度"],
                },
            }

    class TerminologyMapper:
        """Small terminology mapper compatible with the legacy interface."""

        def map_term(self, term: str) -> str:
            aliases = {
                "inner race": "内圈故障",
                "outer race": "外圈故障",
                "unbalance": "不平衡",
                "misalignment": "不对中",
            }
            return aliases.get(term.lower(), term)


class LocalTemplateExplainer:
    """Local deterministic explainer used when the legacy LLM package is absent."""

    def __init__(self, model=None, config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.config = config or {}

    def explain_with_llm(
        self,
        signal_data: torch.Tensor,
        user_query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        fault_info = context.get("fault_info", {})
        fault_type = fault_info.get("fault_type", "未知故障")
        confidence = fault_info.get("confidence", 0.0)
        query = user_query or "general explanation"
        response = (
            f"本地模板解释：诊断结果为 {fault_type}，置信度 {confidence:.1%}。"
            f" 针对问题“{query}”，建议结合振动统计特征、频域峰值和维护记录复核。"
        )
        return {
            "llm_enhanced_explanation": {"response": response},
            "technical_summary": {
                "backend": "local_template",
                "signal_shape": list(signal_data.shape),
                "fault_type": fault_type,
                "confidence": confidence,
            },
        }

    def generate_conversation_response(
        self,
        last_diagnosis: Dict[str, Any],
        user_message: str,
        context: Dict[str, Any],
    ) -> str:
        diagnosis = last_diagnosis or context.get("diagnostic_context", {})
        fault_type = diagnosis.get("fault_type", "未知故障")
        confidence = diagnosis.get("confidence", 0.0)
        if "why" in user_message.lower() or "原因" in user_message:
            return (
                f"{fault_type} 的常见原因包括润滑不足、磨损、冲击载荷或安装偏差；"
                f"当前诊断置信度为 {confidence:.1%}。"
            )
        if "repair" in user_message.lower() or "维修" in user_message:
            return f"针对 {fault_type}，建议先复核信号证据，再安排检查、备件和维修窗口。"
        return f"当前诊断关注 {fault_type}，可继续询问原因、严重程度、维修或监测建议。"


class LLMEnhancedExplainer:
    """
    Main explainer class for LLM-enhanced fault diagnosis explanations.

    This class integrates signal processing analysis with LLM-based natural
    language generation to provide comprehensive, understandable explanations.
    """

    def __init__(self,
                 llm_config: Optional[Dict[str, Any]] = None,
                 model=None):
        """
        Initialize the LLM-enhanced explainer.

        Args:
            llm_config: Configuration for LLM providers
            model: Optional pre-trained diagnosis model
        """
        self.llm_config = llm_config or {}
        self.model = model

        # Initialize components
        self._initialize_components()

        # Explanation history for context
        self.explanation_history = []

    def _initialize_components(self):
        """Initialize core components."""
        try:
            explainer_cls = ExternalLLMExplainer or LocalTemplateExplainer
            self.llm_explainer = explainer_cls(self.model, self.llm_config)
            self._llm_available = True
        except Exception as e:
            print(f"Warning: LLM initialization failed: {e}")
            self.llm_explainer = None
            self._llm_available = False

        # Knowledge components
        self.knowledge_graph = FaultKnowledgeGraph()
        self.terminology_mapper = TerminologyMapper()

    def explain(self,
                signal_data: Union[torch.Tensor, np.ndarray],
                fault_prediction: Dict[str, Any],
                user_query: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None,
                style: str = "standard") -> Dict[str, Any]:
        """
        Generate comprehensive explanation for fault diagnosis.

        Args:
            signal_data: Input signal data
            fault_prediction: Model prediction results
            user_query: Optional user query
            context: Additional context information
            style: Explanation style ("standard", "detailed", "simple", "expert")

        Returns:
            Complete explanation with multiple components
        """
        # Validate inputs
        if not isinstance(signal_data, (torch.Tensor, np.ndarray)):
            raise ValueError("Signal data must be torch.Tensor or numpy array")

        # Convert to tensor if needed
        if isinstance(signal_data, np.ndarray):
            signal_data = torch.tensor(signal_data, dtype=torch.float32)

        # Add batch dimension if needed
        if signal_data.dim() == 2:
            signal_data = signal_data.unsqueeze(0)

        # Generate explanation components
        explanation = {
            "timestamp": datetime.now().isoformat(),
            "fault_info": self._extract_fault_info(fault_prediction),
            "signal_analysis": self._analyze_signal(signal_data),
            "technical_explanation": self._generate_technical_explanation(
                signal_data, fault_prediction
            ),
            "natural_language_explanation": None,
            "knowledge_enhanced_insights": None,
            "recommendations": None,
            "metadata": self._generate_metadata(signal_data, fault_prediction, style)
        }

        # Generate LLM-enhanced explanation if available
        if self._llm_available and self.llm_explainer:
            try:
                llm_result = self._generate_llm_explanation(
                    signal_data, fault_prediction, user_query, context, style
                )
                explanation["natural_language_explanation"] = llm_result.get("response")
                explanation["knowledge_enhanced_insights"] = llm_result.get("insights")
            except Exception as e:
                print(f"Warning: LLM explanation failed: {e}")
                # Fallback to rule-based explanation
                explanation["natural_language_explanation"] = self._generate_rule_based_explanation(
                    fault_prediction, user_query, style
                )

        # Generate recommendations
        explanation["recommendations"] = self._generate_recommendations(
            fault_prediction, signal_data
        )

        # Store in history
        self.explanation_history.append({
            "timestamp": explanation["timestamp"],
            "fault_type": fault_prediction.get("fault_type"),
            "confidence": fault_prediction.get("confidence", 0.0),
            "user_query": user_query
        })

        return explanation

    def explain_conversation(self,
                            session_id: str,
                            user_message: str,
                            conversation_context: Dict[str, Any]) -> str:
        """
        Generate response for conversational interaction.

        Args:
            session_id: Conversation session identifier
            user_message: User's message
            conversation_context: Conversation context

        Returns:
            Generated response
        """
        if not self._llm_available:
            return self._generate_fallback_response(user_message, conversation_context)

        try:
            # Prepare context for LLM
            context = {
                "session_info": conversation_context.get("session_info", {}),
                "diagnostic_context": conversation_context.get("diagnostic_context", {}),
                "conversation_history": conversation_context.get("history", []),
                "device_info": conversation_context.get("device_info", {})
            }

            # Generate response
            response = self.llm_explainer.generate_conversation_response(
                conversation_context.get("last_diagnosis", {}),
                user_message,
                context
            )

            return response

        except Exception as e:
            print(f"Warning: Conversation response failed: {e}")
            return self._generate_fallback_response(user_message, conversation_context)

    def get_explanation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent explanation history.

        Args:
            limit: Maximum number of items to return

        Returns:
            List of explanation history entries
        """
        return self.explanation_history[-limit:]

    def _extract_fault_info(self, fault_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fault information from prediction."""
        return {
            "fault_type": fault_prediction.get("fault_type", "Unknown"),
            "confidence": fault_prediction.get("confidence", 0.0),
            "probability_distribution": fault_prediction.get("probabilities", []),
            "prediction_method": fault_prediction.get("method", "Neural Network"),
            "model_name": fault_prediction.get("model_name", "Unknown")
        }

    def _analyze_signal(self, signal_data: torch.Tensor) -> Dict[str, Any]:
        """Analyze signal characteristics."""
        # Convert to numpy for analysis
        signal_np = signal_data.detach().cpu().numpy().flatten()

        # Basic statistics
        stats = {
            "mean": float(np.mean(signal_np)),
            "std": float(np.std(signal_np)),
            "rms": float(np.sqrt(np.mean(signal_np ** 2))),
            "peak": float(np.max(np.abs(signal_np))),
            "crest_factor": float(np.max(np.abs(signal_np)) / (np.sqrt(np.mean(signal_np ** 2)) + 1e-8)),
            "skewness": float(self._calculate_skewness(signal_np)),
            "kurtosis": float(self._calculate_kurtosis(signal_np))
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

        return {
            "statistics": stats,
            "frequency_analysis": freq_analysis,
            "signal_length": len(signal_np),
            "sampling_rate": 1024  # Assumed
        }

    def _generate_technical_explanation(self,
                                       signal_data: torch.Tensor,
                                       fault_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Generate technical explanation using signal path analysis."""
        try:
            # Try to use model's signal path analysis if available
            if self.model and hasattr(self.model, 'get_signal_path'):
                signal_path = self.model.get_signal_path(signal_data)
                return {
                    "signal_path": signal_path,
                    "processing_stages": len(signal_path.get("data", {}).get("signal_path", [])),
                    "energy_analysis": signal_path.get("data", {}).get("physical_analysis", {})
                }
            else:
                # Fallback technical analysis
                return {
                    "signal_analysis_complete": True,
                    "processing_method": "Statistical and frequency analysis",
                    "key_findings": self._extract_key_findings(signal_data, fault_prediction)
                }
        except Exception as e:
            print(f"Warning: Technical analysis failed: {e}")
            return {"technical_analysis_failed": True, "error": str(e)}

    def _generate_llm_explanation(self,
                                 signal_data: torch.Tensor,
                                 fault_prediction: Dict[str, Any],
                                 user_query: Optional[str],
                                 context: Optional[Dict[str, Any]],
                                 style: str) -> Dict[str, Any]:
        """Generate LLM-enhanced explanation."""
        if not self.llm_explainer:
            return {}

        try:
            # Generate technical summary
            technical_summary = {
                "fault_info": self._extract_fault_info(fault_prediction),
                "signal_analysis": self._analyze_signal(signal_data),
                "technical_explanation": self._generate_technical_explanation(signal_data, fault_prediction)
            }

            # Generate LLM explanation
            llm_result = self.llm_explainer.explain_with_llm(
                signal_data,
                user_query=user_query,
                context={
                    **(context or {}),
                    "fault_info": self._extract_fault_info(fault_prediction),
                    "technical_summary": technical_summary,
                }
            )

            return {
                "response": llm_result.get("llm_enhanced_explanation", {}).get("response"),
                "insights": llm_result.get("technical_summary", {}),
                "style": style,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"Warning: LLM explanation generation failed: {e}")
            return {}

    def _generate_rule_based_explanation(self,
                                       fault_prediction: Dict[str, Any],
                                       user_query: Optional[str],
                                       style: str) -> str:
        """Generate rule-based explanation as fallback."""
        fault_type = fault_prediction.get("fault_type", "Unknown")
        confidence = fault_prediction.get("confidence", 0.0)

        base_explanation = f"检测到设备存在 **{fault_type}** 故障，诊断置信度为 {confidence:.1%}。"

        if user_query:
            if "原因" in user_query or "why" in user_query.lower():
                return base_explanation + " 可能的原因包括正常磨损、润滑不良、过载运行或安装不当。建议结合设备运行历史进行进一步分析。"
            elif "维修" in user_query or "repair" in user_query.lower():
                return base_explanation + " 建议的维修措施包括：详细检查设备状态、准备必要备件、制定维修计划并安排合适的维修窗口。"
            elif "严重" in user_query or "severity" in user_query.lower():
                severity = "高" if confidence > 0.8 else "中等" if confidence > 0.6 else "低"
                return base_explanation + f" 根据诊断置信度评估，故障严重程度为{severity}，建议采取相应的维修措施。"
            else:
                return base_explanation + " 如需更详细的分析或维修建议，请提出具体问题。"
        else:
            return base_explanation + " 系统可以提供详细的故障机理分析、维修建议和风险评估。请告诉我您希望了解哪个方面。"

    def _generate_recommendations(self,
                                  fault_prediction: Dict[str, Any],
                                  signal_data: torch.Tensor) -> List[Dict[str, Any]]:
        """Generate maintenance and monitoring recommendations."""
        fault_type = fault_prediction.get("fault_type", "Unknown")
        confidence = fault_prediction.get("confidence", 0.0)

        recommendations = []

        # Urgency-based recommendations
        if confidence > 0.8:
            recommendations.append({
                "category": "urgent_action",
                "priority": "high",
                "action": "立即停机检查，安排紧急维修",
                "reason": "高置信度故障检测，需要立即处理"
            })

        # Technical recommendations
        if fault_type != "Unknown":
            recommendations.append({
                "category": "technical_investigation",
                "priority": "medium",
                "action": f"详细检查{fault_type}相关部件",
                "reason": "确认故障具体位置和严重程度"
            })

        # Monitoring recommendations
        recommendations.append({
            "category": "monitoring",
            "priority": "medium",
            "action": "增加振动监测频率，跟踪故障发展趋势",
            "reason": "持续监控设备状态变化"
        })

        return recommendations

    def _generate_metadata(self,
                          signal_data: torch.Tensor,
                          fault_prediction: Dict[str, Any],
                          style: str) -> Dict[str, Any]:
        """Generate explanation metadata."""
        return {
            "signal_shape": list(signal_data.shape),
            "explanation_style": style,
            "llm_available": self._llm_available,
            "generation_time": datetime.now().isoformat(),
            "toolkit_version": "1.0.0"
        }

    def _extract_key_findings(self,
                             signal_data: torch.Tensor,
                             fault_prediction: Dict[str, Any]) -> List[str]:
        """Extract key findings from signal analysis."""
        findings = []

        signal_stats = self._analyze_signal(signal_data)
        stats = signal_stats["statistics"]
        freq_analysis = signal_stats["frequency_analysis"]

        # Statistical findings
        if stats["rms"] > 10.0:
            findings.append("振动RMS值较高，表明存在明显振动异常")

        if stats["crest_factor"] > 5.0:
            findings.append("峰值因子较高，可能存在冲击性故障")

        # Frequency findings
        if freq_analysis["dominant_frequency"] > 0:
            findings.append(f"检测到主频成分：{freq_analysis['dominant_frequency']:.1f} Hz")

        # Fault-specific findings
        fault_type = fault_prediction.get("fault_type", "")
        if "内圈" in fault_type:
            findings.append("内圈故障特征：可能出现高频谐波成分")
        elif "不对中" in fault_type:
            findings.append("不对中特征：1x和2x转速频率成分明显")
        elif "不平衡" in fault_type:
            findings.append("不平衡特征：1x转速频率占主导地位")

        return findings

    def _generate_fallback_response(self,
                                  user_message: str,
                                  context: Dict[str, Any]) -> str:
        """Generate fallback response when LLM is unavailable."""
        fault_type = context.get("diagnostic_context", {}).get("fault_type", "未知故障")
        confidence = context.get("diagnostic_context", {}).get("confidence", 0.0)

        if "原因" in user_message or "why" in user_message.lower():
            return f"关于 **{fault_type}** 的原因分析：常见原因包括正常磨损、润滑不足、过载运行或安装不当。建议结合设备运行历史和维护记录进行具体分析。"
        elif "维修" in user_message or "repair" in user_message.lower():
            return f"针对 **{fault_type}** 的维修建议：1) 详细检查故障部件 2) 评估损坏程度 3) 准备必要备件 4) 制定维修计划 5) 验证维修效果。"
        elif "严重" in user_message or "severity" in user_message.lower():
            severity_text = "高" if confidence > 0.8 else "中等" if confidence > 0.6 else "低"
            return f"当前诊断置信度为 {confidence:.1%}，评估 **{fault_type}** 的严重程度为{severity_text}。"
        else:
            return f"关于您的设备问题（**{fault_type}**），我可以提供故障原因分析、维修建议、严重程度评估和预防措施等方面的帮助。请告诉我您希望了解哪个具体方面。"

    def _calculate_skewness(self, signal: np.ndarray) -> float:
        """Calculate signal skewness."""
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return 0.0
        return np.mean(((signal - mean) / std) ** 3)

    def _calculate_kurtosis(self, signal: np.ndarray) -> float:
        """Calculate signal kurtosis."""
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return 0.0
        return np.mean(((signal - mean) / std) ** 4) - 3

    def get_component_info(self) -> Dict[str, Any]:
        """Get information about explainer components."""
        return {
            "llm_available": self._llm_available,
            "knowledge_graph_available": True,
            "terminology_mapper_available": True,
            "explanation_history_size": len(self.explanation_history),
            "supported_fault_types": list(self.knowledge_graph.fault_patterns.keys()),
            "llm_config": self.llm_config
        }

    def reset_history(self):
        """Reset explanation history."""
        self.explanation_history = []
