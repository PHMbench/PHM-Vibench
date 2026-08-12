"""
Unified Explainer - LLM Enhanced Version

This module provides an enhanced version of the UnifiedExplainer that integrates
Large Language Model capabilities for natural language explanations and interactive
diagnostic conversations.
"""

from typing import Dict, Any, Optional, Union, List
import torch
from .unified_explainer import UnifiedExplainer
from .explanation import Explanation
from .base_explainer import BaseExplainer


class UnifiedExplainerLLMEnhanced(UnifiedExplainer):
    """
    Enhanced Unified Explainer with LLM integration.

    This class extends the original UnifiedExplainer with additional capabilities:
    - LLM-enhanced explanations
    - Interactive diagnostic conversations
    - Knowledge-based enhancement
    - Natural language report generation
    """

    def __init__(self,
                 model: torch.nn.Module,
                 config: Optional[Dict[str, Any]] = None,
                 method: str = 'auto'):
        """
        Initialize the LLM-enhanced unified explainer.

        Args:
            model: The model to explain
            config: Configuration dictionary
            method: Explanation method to use
        """
        # Initialize parent class
        super().__init__(model, config, method)

        # LLM configuration
        self.llm_enabled = self.config.get('llm_enabled', False)
        self.llm_config = self.config.get('llm_config', {})

        # Initialize LLM components if enabled
        self._llm_explainer = None
        self._conversation_engine = None

        if self.llm_enabled:
            self._initialize_llm_components()

    def _initialize_llm_components(self):
        """Initialize LLM-related components."""
        from ..llm.llm_explainer import LLMExplainer

        self._llm_explainer = LLMExplainer(self.model, self.llm_config)

        # Initialize conversation engine
        if self.config.get('enable_conversation', True):
            from ..conversation.conversation_engine import ConversationEngine
            self._conversation_engine = ConversationEngine(self, self.llm_config)

    def explain(self,
                input_data: torch.Tensor,
                target_class: Optional[int] = None,
                **kwargs) -> Explanation:
        """
        Generate explanation with optional LLM enhancement.

        Args:
            input_data: Input tensor [batch_size, sequence_length, channels]
            target_class: Target class for explanation (for multi-class)
            **kwargs: Additional arguments including:
                - llm_enhanced: bool - Enable LLM enhancement
                - user_query: str - User query for targeted explanation
                - conversation_mode: bool - Enable conversation mode

        Returns:
            Explanation object containing explanation results
        """
        # Extract LLM-specific parameters
        llm_enhanced = kwargs.pop('llm_enhanced', False)
        user_query = kwargs.pop('user_query', None)
        conversation_mode = kwargs.pop('conversation_mode', False)

        # Generate traditional explanation first
        traditional_explanation = super().explain(input_data, target_class, **kwargs)

        # Skip LLM processing if not enabled
        if not self.llm_enabled or not llm_enhanced:
            return traditional_explanation

        # Handle different LLM modes
        if conversation_mode and self._conversation_engine:
            # Use conversation engine for interactive mode
            return self._handle_conversation_mode(
                traditional_explanation,
                user_query,
                input_data,
                target_class,
                **kwargs
            )

        # Standard LLM enhancement
        return self._enhance_with_llm(
            traditional_explanation,
            input_data,
            user_query,
            target_class,
            **kwargs
        )

    def _enhance_with_llm(self,
                      traditional_explanation: Explanation,
                      input_data: torch.Tensor,
                      user_query: Optional[str],
                      target_class: Optional[int],
                      **kwargs) -> Explanation:
        """Enhance explanation using LLM."""
        # Check if model supports LLM explainability
        if hasattr(self.model, 'get_diagnosis_context'):
            context = self.model.get_diagnosis_context(input_data)
        else:
            context = {}

        # Generate technical summary
        technical_summary = self.model.generate_technical_summary(input_data, traditional_explanation)

        # Generate LLM-enhanced explanation
        if self._llm_explainer is not None:
            llm_result = self._llm_explainer.explain_with_llm(
                traditional_explanation,
                technical_summary,
                user_query,
                context
            )

            # Create enhanced explanation object
            enhanced_explanation = self._create_enhanced_explanation(
                traditional_explanation,
                llm_result,
                technical_summary,
                context
            )

            return enhanced_explanation
        else:
            # Fallback to traditional explanation
            self.logger.warning("LLM explainer not available, returning traditional explanation")
            return traditional_explanation

    def _handle_conversation_mode(self,
                               traditional_explanation: Explanation,
                               user_query: str,
                               input_data: torch.Tensor,
                               target_class: Optional[int],
                               **kwargs) -> Explanation:
        """Handle conversation mode explanation."""
        if self._conversation_engine is None:
            raise ValueError("Conversation engine not initialized")

        # Start or continue conversation
        session = self._conversation_engine.get_or_create_session(input_data)
        response = session.query(user_query, traditional_explanation)

        # Create explanation with conversation response
        enhanced_explanation = self._create_conversation_explanation(
            traditional_explanation,
            response
        )

        return enhanced_explanation

    def _create_enhanced_explanation(self,
                                         traditional_explanation: Explanation,
                                         llm_result: Dict[str, Any],
                                         technical_summary: Dict[str, Any],
                                         context: Dict[str, Any]) -> Explanation:
        """Create enhanced explanation combining traditional and LLM results."""
        # Merge traditional explanation data
        enhanced_data = traditional_explanation.data.copy()
        enhanced_meta = traditional_explanation.meta.copy()

        # Add LLM enhancements
        if llm_result:
            enhanced_data['llm_explanation'] = llm_result.get('response_text', '')
            enhanced_data['llm_findings'] = llm_result.get('key_findings', [])
            enhanced_data['recommendations'] = llm_result.get('recommendations', [])
            enhanced_data['llm_confidence'] = llm_result.get('confidence_assessment', 0.0)

            # Update metadata
            enhanced_meta.update({
                'method': f"{enhanced_meta.get('method', 'unknown')}_llm_enhanced",
                'llm_model': llm_result.get('model_used', 'unknown'),
                'explanation_enhanced': True,
                'response_time': llm_result.get('response_time', 0),
                'tokens_used': llm_result.get('tokens_used', 0)
            })

        # Add technical summary
        if technical_summary:
            enhanced_data['technical_summary'] = technical_summary

        # Add context information
        if context:
            enhanced_data['context'] = context

        return Explanation(enhanced_data, enhanced_meta)

    def _create_conversation_explanation(self,
                                         traditional_explanation: Explanation,
                                         conversation_response: Dict[str, Any]) -> Explanation:
        """Create explanation for conversation mode."""
        enhanced_data = traditional_explanation.data.copy()
        enhanced_meta = traditional_explanation.meta.copy()

        # Add conversation information
        enhanced_data['conversation_response'] = {
            'response': conversation_response.get('response_text', ''),
            'query_time': conversation_response.get('response_time', 0),
            'conversation_id': conversation_response.get('conversation_id', 'unknown')
        }

        # Update metadata
        enhanced_meta.update({
            'method': f"{enhanced_meta.get('method', 'unknown')}_conversation",
            'conversation_mode': True,
            'explanation_enhanced': True
        })

        return Explanation(enhanced_data, enhanced_meta)

    def explain_with_llm_only(self,
                            input_data: torch.Tensor,
                            user_query: Optional[str] = None,
                            **kwargs) -> Dict[str, Any]:
        """
        Generate explanation using only LLM without traditional preprocessing.

        Args:
            input_data: Input tensor
            user_query: Optional user query
            **kwargs: Additional parameters

        Returns:
            Dictionary containing LLM-only explanation results
        """
        if not self.llm_enabled:
            raise ValueError("LLM is not enabled. Set llm_enabled=True in config.")

        if not hasattr(self, 'get_diagnosis_context'):
            raise ValueError("Model does not support LLM explainability")

        # Get model context and summary
        context = self.get_diagnosis_context(input_data)
        prediction = self._get_model_predictions(input_data)
        technical_summary = self.generate_technical_summary(input_data, prediction)

        # Generate explanation
        return self._llm_explainer.explain_with_llm(
            None,  # No traditional explanation
            technical_summary,
            user_query,
            context,
            **kwargs
        )

    def start_conversation(self, input_data: torch.Tensor, context: Optional[Dict[str, Any]] = None):
        """
        Start an interactive diagnostic conversation.

        Args:
            input_data: Input tensor for diagnosis
            context: Optional context information

        Returns:
            ConversationSession object
        """
        if not self.llm_enabled:
            raise ValueError("LLM is not enabled. Set llm_enabled=True in config.")

        if not self.enable_conversation:
            raise ValueError("Conversation is not enabled. Set enable_conversation=True in config.")

        from ..conversation.conversation_engine import ConversationEngine

        if self._conversation_engine is None:
            self._conversation_engine = ConversationEngine(self, self.llm_config)

        return self._conversation_engine.start_session(input_data, context)

    def generate_maintenance_suggestions(self,
                                          explanation: Optional[Explanation] = None,
                                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate maintenance suggestions based on diagnosis.

        Args:
            explanation: Explanation object (optional)
            context: Context information (optional)

        Returns:
            Dictionary containing maintenance suggestions
        """
        if not self.llm_enabled:
            raise ValueError("LLM is not enabled. Set llm_enabled=True in config.")

        if hasattr(self.model, 'generate_maintenance_suggestions'):
            return self.model.generate_maintenance_suggestions(explanation, context)
        else:
            # Fallback method
            return self._fallback_maintenance_suggestions(explanation, context)

    def _fallback_maintenance_suggestions(self,
                                        explanation: Optional[Explanation] = None,
                                        context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fallback maintenance suggestions method."""
        suggestions = {
            'status': 'fallback',
            'recommended_actions': [
                '定期检查设备运行状态',
                '关注异常振动水平',
                '维护设备润滑状态',
                '记录历史维修记录'
            ],
            'maintenance_priority': 'medium',
            'estimated_downtime': 'unknown'
        }

        # Add information from explanation if available
        if explanation:
            if hasattr(explanation, 'get_data'):
                data = explanation.get_data()

                # Check for anomaly indicators
                if 'anomaly_indicators' in data:
                    anomalies = data['anomaly_indicators']
                    for anomaly_type, anomaly_info in anomalies.items():
                        if anomaly_info.get('detected', False):
                            if anomaly_info.get('severity') == 'high':
                                suggestions['maintenance_priority'] = 'urgent'
                                suggestions['urgent_actions'] = [f"检查{anomaly_type}问题"]
                            elif anomaly_info.get('severity') == 'medium':
                                suggestions['moderate_actions'] = [f"监控{anomaly_type}变化"]
                # Check prediction confidence
                if hasattr(explanation, 'get_meta'):
                    meta = explanation.get_meta()
                    if meta.get('max_prob', 1.0) < 0.7:
                        suggestions['recommended_actions'].append('进行更详细的诊断分析')

        return suggestions

    def create_interactive_report(self,
                                 explanation: Explanation,
                                 user_preferences: Optional[Dict[str, Any]] = None) -> str:
        """
        Create an interactive report based on the explanation.

        Args:
            explanation: Explanation object
            user_preferences: User preference configuration

        Returns:
            Formatted report string
        """
        if not self.llm_enabled:
            return "LLM is not enabled. Cannot generate interactive report."

        try:
            # Get comprehensive context
            context = self._get_comprehensive_context(explanation)

            # Generate technical summary if not available
            if not hasattr(explanation, 'technical_summary'):
                prediction = self._get_model_predictions(explanation.get_data('original_signal'))
                context['technical_summary'] = self.generate_technical_summary(prediction)

            # Build report prompt
            if not hasattr(self, 'prompt_manager'):
                from ..llm.prompt_manager import PromptManager
                self.prompt_manager = PromptManager(self.config)

            prompt = self.prompt_manager.build_report_prompt(
                explanation,
                context,
                user_preferences
            )

            # Generate LLM response
            llm_response = self.llm_interface.generate_response(prompt)

            # Format the report
            report = self._format_comprehensive_report(
                llm_response, explanation, context
            )

            return report

        except Exception as e:
            return f"Error generating interactive report: {str(e)}"

    def _get_comprehensive_context(self, explanation: Explanation) -> Dict[str, Any]:
        """Get comprehensive context for report generation."""
        # Get model explainability info
        if hasattr(self.model, 'get_model_explainability_info'):
            model_info = self.model.get_model_explainability_info()
        else:
            model_info = {'model_type': 'unknown'}

        # Get explanation data and meta
        explanation_data = explanation.data if explanation else {}
        explanation_meta = explanation.meta if explanation else {}

        # Build comprehensive context
        context = {
            'model_info': model_info,
            'explanation_data': explanation_data,
            'explanation_meta': explanation_meta,
            'current_time': self._get_current_timestamp()
        }

        # Add traditional explanation methods if available
        traditional_methods = []
        if 'signal_path' in explanation_data:
            traditional_methods.append('signal_path')
        if 'importance_scores' in explanation_data:
            traditional_methods.append('importance_scores')
        if 'attention_maps' in explanation_data:
            traditional_methods.append('attention_maps')

        if traditional_methods:
            context['traditional_methods'] = traditional_methods

        return context

    def _format_comprehensive_report(self,
                                      llm_response: Dict[str, Any],
                                      explanation: Explanation,
                                      context: Dict[str, Any]) -> str:
        """Format comprehensive diagnostic report."""
        report_parts = []

        # Header
        report_parts.append("=" * 60)
        report_parts.append("智能故障诊断分析报告")
        report_parts.append("=" * 60)
        report_parts.append(f"生成时间: {self._get_current_timestamp()}")
        report_parts.append(f"分析模型: {context.get('model_info', {}).get('model_name', 'unknown')}")
        report_parts.append("")

        # Analysis Summary
        if llm_response.get('analysis_summary'):
            report_parts.append("分析摘要:")
            report_parts.append("-" * 30)
            report_parts.append(llm_response['analysis_summary'])
            report_parts.append("")

        # Detailed Findings
        if llm_response.get('key_findings'):
            report_parts.append("关键发现:")
            report_parts.append("-" * 30)
            for i, finding in enumerate(llm_response['key_findings'], 1):
                report_parts.append(f"{i}. {finding}")
            report_parts.append("")

        # Diagnostic Results
        if llm_response.get('diagnostic_results'):
            report_parts.append("诊断结果:")
            report_parts.append("-" * 30)
            results = llm_response['diagnostic_results']
            if isinstance(results, dict):
                for key, value in results.items():
                    report_parts.append(f"{key}: {value}")
            report_parts.append("")

        # Maintenance Recommendations
        if llm_response.get('maintenance_recommendations'):
            report_parts.append("维护建议:")
            report_parts.append("-" * 30)
            recommendations = llm_response['maintenance_recommendations']
            if isinstance(recommendations, list):
                for i, rec in enumerate(recommendations, 1):
                    report_parts.append(f"{i}. {rec}")
            report_parts.append("")

        # Risk Assessment
        if llm_response.get('risk_assessment'):
            report_parts.append("风险评估:")
            report_parts.append("-" * 30)
            report_parts.append(llm_response['risk_assessment'])
            report_parts.append("")

        # Traditional Methods Summary
        if context.get('traditional_methods'):
            report_parts.append("传统分析方法:")
            report_parts.append("-" * 30)
            for method in context['traditional_methods']:
                if explanation.get_data(method):
                    report_parts.append(f"- {method}: {len(explanation.get_data(method))} 个分析结果")
            report_parts.append("")

        # Footer
        report_parts.append("=" * 60)
        report_parts.append("报告生成完成")
        report_parts.append("此报告基于AI分析，请结合实际情况进行最终决策")

        return "\n".join(report_parts)

    def _get_current_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def get_llm_explainability_info(self) -> Dict[str, Any]:
        """Get LLM explainability information."""
        info = {
            'llm_enabled': self.llm_enabled,
            'conversation_enabled': self.enable_conversation,
            'knowledge_enhancement': self.enable_knowledge_enhancement,
            'available_llm_models': ['gpt-4', 'gpt-3.5-turbo', 'claude-3-sonnet']
        }

        # Add LLM explainer info if available
        if self._llm_explainer:
            info['llm_explainer'] = str(type(self._llm_explainer))
            info['llm_model'] = self._llm_explainer.llm_interface.get_current_model() if hasattr(self._llm_explainer, 'llm_interface') else 'unknown'
            info['prompt_template'] = self._llm_explainer.prompt_manager.template_name if hasattr(self._llm_explainer, 'prompt_manager') else 'default'

        # Add conversation engine info if available
        if self._conversation_engine:
            info['conversation_engine'] = str(type(self._conversation_engine))
            info['active_sessions'] = len(self._conversation_engine.active_sessions)

        return info

    def __repr__(self) -> str:
        """String representation of the LLM-enhanced explainer."""
        llm_status = "LLM_ENABLED" if self.llm_enabled else "LLM_DISABLED"
        return f"UnifiedExplainerLLMEnhanced(model={type(self.model).__name__}, method='{self.method}', {llm_status})"

    @property
    def llm_enabled(self) -> bool:
        """Check if LLM is enabled."""
        return self.llm_enabled

    @property
    def conversation_available(self) -> bool:
        """Check if conversation feature is available."""
        return self.llm_enabled and self.enable_conversation

    @property
    def knowledge_enhancement_available(self) -> bool:
        """Check if knowledge enhancement is available."""
        return self.llm_enabled and self.enable_knowledge_enhancement