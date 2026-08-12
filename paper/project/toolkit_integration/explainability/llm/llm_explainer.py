"""
LLM Explainer Core Implementation

This module provides the core LLM explainer that integrates with Large Language Models
to generate natural language explanations and enable interactive diagnostic conversations.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from ..core.base_explainer import BaseExplainer
from ..core.explanation import Explanation
from .prompt_manager import PromptManager
from .llm_interface import LLMInterface
from .response_parser import ResponseParser


class LLMExplainer(BaseExplainer):
    """
    LLM Enhanced Explainer for fault diagnosis.

    This explainer integrates with Large Language Models to provide
    natural language explanations and interactive diagnostic capabilities.
    """

    def __init__(self,
                 model,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM explainer.

        Args:
            model: The model to explain
            config: Configuration dictionary for LLM settings
        """
        super().__init__(model, config)

        # Initialize components
        self.prompt_manager = PromptManager(self.config)
        self.llm_interface = LLMInterface(self.config)
        self.response_parser = ResponseParser(self.config)

        # Configuration for LLM integration
        self.enable_conversation = self.config.get('enable_conversation', True)
        self.enable_knowledge_enhancement = self.config.get('enable_knowledge_enhancement', True)
        self.fallback_on_error = self.config.get('fallback_on_error', True)

        # Logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def explain_with_llm(self,
                       traditional_explanation: Optional[Explanation],
                       technical_summary: Optional[Dict[str, Any]],
                       user_query: Optional[str] = None,
                       context: Optional[Dict[str, Any]] = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Generate LLM-enhanced explanation.

        Args:
            traditional_explanation: Traditional explanation object
            technical_summary: Technical analysis summary
            user_query: Optional user query for targeted explanation
            context: Additional context information
            **kwargs: Additional parameters

        Returns:
            Dictionary containing LLM-enhanced explanation results
        """
        try:
            # Encode traditional explanation into text
            encoded_explanation = ""
            if traditional_explanation is not None:
                from .signal_encoder import SignalEncoder
                encoder = SignalEncoder(self.config)
                encoded_explanation = encoder.encode_explanation(traditional_explanation)

            # Encode technical summary
            encoded_summary = ""
            if technical_summary is not None:
                encoder = SignalEncoder(self.config)
                encoded_summary = encoder.encode_technical_summary(technical_summary)

            # Encode context information
            encoded_context = ""
            if context is not None:
                encoder = SignalEncoder(self.config)
                encoded_context = encoder.encode_context_information(context)

            # Build prompt
            prompt = self.prompt_manager.build_prompt(
                encoded_explanation,
                encoded_summary,
                user_query,
                encoded_context,
                **kwargs
            )

            # Generate LLM response
            self.logger.info("Generating LLM response...")
            llm_response = self.llm_interface.generate_response(prompt)

            # Parse response
            structured_response = self.response_parser.parse(llm_response)

            # Create enhanced explanation
            enhanced_explanation = self._create_llm_enhanced_explanation(
                traditional_explanation,
                structured_response,
                technical_summary,
                context
            )

            return {
                'enhanced_explanation': enhanced_explanation,
                'llm_response': structured_response,
                'prompt': prompt,
                'response_time': structured_response.get('response_time', 0),
                'model_used': structured_response.get('model_used', 'unknown'),
                'tokens_used': structured_response.get('tokens_used', 0)
            }

        except Exception as e:
            self.logger.error(f"Error generating LLM explanation: {e}")
            if self.fallback_on_error:
                return self._create_fallback_response(traditional_explanation, technical_summary, e)
            else:
                raise

    def explain_with_knowledge_enhancement(self,
                                             traditional_explanation: Explanation,
                                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate explanation enhanced with domain knowledge.

        Args:
            traditional_explanation: Traditional explanation object
            context: Additional context information

        Returns:
            Dictionary containing knowledge-enhanced explanation
        """
        if not self.enable_knowledge_enhancement:
            return {'error': 'Knowledge enhancement not enabled'}

        try:
            # Get domain knowledge
            from ..knowledge.fault_knowledge import FaultKnowledgeGraph
            knowledge_graph = FaultKnowledgeGraph()

            # Analyze explanation to identify potential faults
            fault_analysis = self._analyze_explanation_for_faults(traditional_explanation)

            # Get knowledge enhancement
            if fault_analysis['potential_fault']:
                fault_type = fault_analysis['potential_fault']
                features = fault_analysis['detected_features']
                knowledge_info = knowledge_graph.get_fault_explanation(fault_type, features)

                if knowledge_info:
                    # Create knowledge-enhanced explanation
                    enhanced_explanation = self._enhance_with_knowledge(
                        traditional_explanation,
                        knowledge_info,
                        context
                    )

                    return {
                        'enhanced_explanation': enhanced_explanation,
                        'knowledge_info': knowledge_info,
                        'fault_type': fault_type,
                        'features': features
                    }

            return {'status': 'no_potential_fault_identified'}

        except Exception as e:
            self.logger.error(f"Error in knowledge enhancement: {e}")
            return {'error': str(e)}

    def start_conversation(self, input_data, context: Optional[Dict[str, Any]] = None):
        """
        Start an interactive conversation for diagnosis.

        Args:
            input_data: Input tensor for diagnosis
            context: Initial context information

        Returns:
            Conversation session object
        """
        if not self.enable_conversation:
            raise ValueError("Conversation not enabled")

        from .conversation.conversation_engine import ConversationEngine

        if not hasattr(self, '_conversation_engine') or self._conversation_engine is None:
            self._conversation_engine = ConversationEngine(self, self.config)

        return self._conversation_engine.start_session(input_data, context)

    def explain_conversational_query(self,
                                     input_data: torch.Tensor,
                                     query: str,
                                     conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Handle a conversational query about the diagnosis.

        Args:
            input_data: Input tensor
            query: User query
            conversation_history: Previous conversation history

        Returns:
            Response dictionary
        """
        if not self.enable_conversation:
            raise ValueError("Conversation not enabled")

        try:
            # Generate explanation for context
            traditional_explanation = None
            if hasattr(self, 'get_signal_path'):
                try:
                    traditional_explanation = self.get_signal_path(input_data)
                except Exception as e:
                    self.logger.warning(f"Could not get signal path: {e}")

            technical_summary = self.generate_technical_summary(input_data, traditional_explanation)
            context = self.get_diagnosis_context(input_data)

            # Build conversation-specific prompt
            prompt = self.prompt_manager.build_conversation_prompt(
                traditional_explanation,
                technical_summary,
                query,
                conversation_history,
                context
            )

            # Generate LLM response
            llm_response = self.llm_interface.generate_response(prompt)
            structured_response = self.response_parser.parse(llm_response)

            return {
                'response': structured_response,
                'traditional_explanation': traditional_explanation,
                'technical_summary': technical_summary,
                'prompt': prompt
            }

        except Exception as e:
            self.logger.error(f"Error handling conversational query: {e}")
            return {'error': str(e), 'fallback': 'Unable to process query'}

    def generate_maintenance_suggestions(self,
                                          explanation: Explanation,
                                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate maintenance suggestions based on diagnosis.

        Args:
            explanation: Explanation object
            context: Context information

        Returns:
            Dictionary containing maintenance suggestions
        """
        try:
            # Analyze explanation for maintenance recommendations
            maintenance_analysis = self._analyze_for_maintenance(explanation, context)

            if maintenance_analysis:
                # Build maintenance-specific prompt
                prompt = self.prompt_manager.build_maintenance_prompt(
                    explanation,
                    maintenance_analysis,
                    context
                )

                # Generate LLM response
                llm_response = self.llm_interface.generate_response(prompt)
                structured_response = self.response_parser.parse(llm_response)

                return {
                    'suggestions': structured_response,
                    'analysis': maintenance_analysis,
                    'prompt': prompt
                }

            return {'status': 'no_maintenance_recommendations'}

        except Exception as e:
            self.logger.error(f"Error generating maintenance suggestions: {e}")
            return {'error': str(e)}

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
        try:
            # Generate comprehensive context
            context = self.get_diagnosis_context(
                explanation.get_data('original_signal')
                if explanation.get_data('original_signal') is not None else None
            )

            # Build report prompt
            prompt = self.prompt_manager.build_report_prompt(
                explanation,
                context,
                user_preferences
            )

            # Generate LLM response
            llm_response = self.llm_interface.generate_response(prompt)
            structured_response = self.response_parser.parse(llm_response)

            # Format the report
            report = self._format_report(structured_response, explanation, context)

            return report

        except Exception as e:
            self.logger.error(f"Error creating interactive report: {e}")
            return f"Error generating report: {str(e)}"

    def _create_llm_enhanced_explanation(self,
                                         traditional_explanation: Optional[Explanation],
                                         structured_response: Dict[str, Any],
                                         technical_summary: Optional[Dict[str, Any]],
                                         context: Optional[Dict[str, Any]]) -> Explanation:
        """
        Create an enhanced explanation object combining traditional and LLM results.
        """
        # Merge data from traditional explanation
        enhanced_data = {}
        enhanced_meta = {}

        if traditional_explanation is not None:
            enhanced_data.update(traditional_explanation.data)
            enhanced_meta.update(traditional_explanation.meta)

        # Add LLM enhancements
        if structured_response.get('explanation_text'):
            enhanced_data['llm_explanation'] = structured_response['explanation_text']

        if structured_response.get('key_findings'):
            enhanced_data['llm_findings'] = structured_response['key_findings']

        if structured_response.get('recommendations'):
            enhanced_data['recommendations'] = structured_response['recommendations']

        if structured_response.get('confidence_assessment'):
            enhanced_data['llm_confidence'] = structured_response['confidence_assessment']

        # Add technical summary
        if technical_summary:
            enhanced_data['technical_summary'] = technical_summary

        # Add context information
        if context:
            enhanced_data['context'] = context

        # Update metadata
        enhanced_meta.update({
            'method': 'llm_enhanced',
            'llm_model': structured_response.get('model_used', 'unknown'),
            'response_time': structured_response.get('response_time', 0),
            'tokens_used': structured_response.get('tokens_used', 0),
            'explanation_generated_at': datetime.now().isoformat()
        })

        return Explanation(enhanced_data, enhanced_meta)

    def _enhance_with_knowledge(self,
                                 traditional_explanation: Explanation,
                                 knowledge_info: Dict[str, Any],
                                 context: Optional[Dict[str, Any]]) -> Explanation:
        """Enhance explanation with domain knowledge."""
        # Enhanced data with knowledge
        enhanced_data = traditional_explanation.data.copy()
        enhanced_meta = traditional_explanation.meta.copy()

        # Add knowledge information
        enhanced_data['knowledge_enhancement'] = {
            'fault_type': knowledge_info['fault_type'],
            'characteristic_frequencies': knowledge_info['characteristic_frequencies'],
            'symptoms': knowledge_info['symptoms'],
            'possible_causes': knowledge_info['possible_causes'],
            'recommended_actions': knowledge_info['recommended_actions']
        }

        # Update metadata
        enhanced_meta.update({
            'method': f"{enhanced_meta.get('method', 'unknown')}_with_knowledge",
            'knowledge_applied': True,
            'explanation_enhanced_at': datetime.now().isoformat()
        })

        return Explanation(enhanced_data, enhanced_meta)

    def _analyze_explanation_for_faults(self, explanation: Explanation) -> Dict[str, Any]:
        """Analyze explanation to identify potential faults."""
        analysis = {
            'potential_fault': None,
            'confidence': 0.0,
            'detected_features': {},
            'evidence': []
        }

        # Look for fault indicators in explanation data
        if explanation.data.get('signal_path'):
            signal_path = explanation.data['signal_path']

            # Analyze energy changes for anomaly
            energy_anomalies = self._detect_energy_anomalies(signal_path)
            if energy_anomalies:
                analysis['detected_features']['energy_anomalies'] = energy_anomalies
                analysis['evidence'].extend(energy_anomalies)

        if explanation.data.get('importance_scores'):
            importance_scores = explanation.data['importance_scores']

            # Look for unusually high importance scores
            high_importance = self._detect_high_importance_scores(importance_scores)
            if high_importance:
                analysis['detected_features']['high_importance'] = high_importance
                analysis['evidence'].extend(high_importance)

        # Determine potential fault type based on detected features
        if analysis['evidence']:
            potential_fault = self._infer_fault_type(analysis['detected_features'])
            analysis['potential_fault'] = potential_fault
            analysis['confidence'] = min(0.9, len(analysis['evidence']) * 0.2)

        return analysis

    def _detect_energy_anomalies(self, signal_path: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect energy anomalies in signal path."""
        anomalies = []

        for step in signal_path:
            if 'input_stats' in step and 'output_stats' in step:
                input_energy = step['input_stats'].get('energy', 0)
                output_energy = step['output_stats'].get('energy', 0)

                if input_energy > 0:
                    energy_ratio = output_energy / input_energy
                    if energy_ratio > 2.0 or energy_ratio < 0.5:  # Significant change
                        anomalies.append({
                            'layer': step.get('layer_name', 'unknown'),
                            'energy_ratio': energy_ratio,
                            'input_energy': input_energy,
                            'output_energy': output_energy,
                            'anomaly_type': 'high_amplification' if energy_ratio > 1 else 'high_attenuation'
                        })

        return anomalies

    def _detect_high_importance_scores(self, importance_scores: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect unusually high importance scores."""
        high_importance = []

        for name, scores in importance_scores.items():
            if isinstance(scores, dict):
                # Use the highest available score
                score = max(abs(v) for v in scores.values() if isinstance(v, (int, float)))
            else:
                score = abs(scores)

            # Consider scores above 0.7 as high importance
            if score > 0.7:
                high_importance.append({
                    'component': name,
                    'importance_score': score,
                    'category': 'very_high' if score > 0.9 else 'high'
                })

        return high_importance

    def _infer_fault_type(self, detected_features: Dict[str, Any]) -> str:
        """Infer fault type from detected features."""
        # Simple inference based on feature types
        if 'energy_anomalies' in detected_features:
            anomalies = detected_features['energy_anomalies']
            if any(a['anomaly_type'] == 'high_amplification' for a in anomalies):
                return 'impact_fault'
            elif any(a['anomaly_type'] == 'high_attenuation' for a in anomalies):
                return 'fault_degradation'

        if 'high_importance' in detected_features:
            # Look at which components have high importance
            high_importance = detected_features['high_importance']
            for comp in high_importance:
                comp_name = comp['component'].lower()
                if 'filter' in comp_name:
                    return 'filtering_related_fault'
                elif 'transform' in comp_name:
                    return 'transformation_related_fault'

        return 'unknown_fault_type'

    def _analyze_for_maintenance(self, explanation: Explanation, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze explanation for maintenance recommendations."""
        analysis = {
            'maintenance_priority': 'medium',
            'recommended_actions': [],
            'urgency_level': 'low',
            'estimated_downtime': 'unknown'
        }

        # Check for indicators requiring immediate attention
        if explanation.data.get('anomaly_indicators'):
            anomalies = explanation.data['anomaly_indicators']
            high_severity = any(annomaly.get('severity') == 'high' for anomaly in anomalies.values())

            if high_severity:
                analysis['maintenance_priority'] = 'high'
                analysis['urgency_level'] = 'urgent'
                analysis['estimated_downtime'] = '1-2 days'
            else:
                analysis['maintenance_priority'] = 'medium'
                analysis['urgency_level'] = 'moderate'
                analysis['estimated_downtime'] = '3-7 days'

        # Check prediction confidence
        if hasattr(self, '_get_model_predictions'):
            try:
                prediction = self._get_model_predictions(
                    explanation.get_data('original_signal')
                )
                if hasattr(prediction, 'get_meta'):
                    # Assuming prediction has confidence info
                    if prediction.get_meta('max_prob') < 0.7:
                        analysis['maintenance_priority'] = 'high'
                        analysis['recommended_actions'].append('further_diagnosis_recommended')
            except Exception:
                pass  # Skip if unable to get prediction

        return analysis

    def _format_report(self, structured_response: Dict[str, Any], explanation: Explanation, context: Dict[str, Any]) -> str:
        """Format the response into a readable report."""
        report_parts = []

        # Header
        report_parts.append("=" * 50)
        report_parts.append("故障诊断报告")
        report_parts.append("=" * 50)
        report_parts.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_parts.append("")

        # Model information
        if explanation and explanation.get_meta:
            method = explanation.get_meta('method', 'unknown')
            model = explanation.get_meta('model_name', 'unknown')
            report_parts.append(f"分析方法: {method}")
            report_parts.append(f"使用模型: {model}")
            report_parts.append("")

        # LLM explanation
        if structured_response.get('explanation_text'):
            report_parts.append("LLM增强分析:")
            report_parts.append("-" * 20)
            report_parts.append(structured_response['explanation_text'])
            report_parts.append("")

        # Key findings
        if structured_response.get('key_findings'):
            report_parts.append("关键发现:")
            report_parts.append("-" * 20)
            for i, finding in enumerate(structured_response['key_findings'], 1):
                report_parts.append(f"{i}. {finding}")
            report_parts.append("")

        # Recommendations
        if structured_response.get('recommendations'):
            report_parts.append("维护建议:")
            report_parts.append("-" * 20)
            for i, rec in enumerate(structured_response['recommendations'], 1):
                report_parts.append(f"{i}. {rec}")
            report_parts.append("")

        # Footer
        report_parts.append("=" * 50)
        report_parts.append("报告生成完成")

        return "\n".join(report_parts)

    def _create_fallback_response(self, traditional_explanation: Optional[Explanation], technical_summary: Optional[Dict[str, Any]], error: Exception) -> Dict[str, Any]:
        """Create fallback response when LLM fails."""
        fallback_data = {
            'status': 'error',
            'error_message': str(error),
            'fallback_mode': True
        }

        if traditional_explanation:
            fallback_data['traditional_explanation'] = {
                'method': traditional_explanation.get_meta('method'),
                'metrics': traditional_explanation.get_metrics()
            }

        if technical_summary:
            fallback_data['technical_summary'] = {
                'signal_characteristics': technical_summary.get('signal_characteristics'),
                'prediction_confidence': technical_summary.get('prediction_confidence')
            }

        return fallback_data

    def _setup_logging(self):
        """Setup logging for the explainer."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def generate_technical_summary(self, input_data: torch.Tensor, explanation: Optional[Explanation] = None) -> Dict[str, Any]:
        """Generate technical summary for LLM processing."""
        # This is already implemented in the base class, but we'll override it here
        # to provide more comprehensive information
        return super().generate_technical_summary(input_data, explanation)

    def __repr__(self) -> str:
        """String representation of the LLM explainer."""
        return f"LLMExplainer(model={type(self.model).__name__}, config_enabled={self.config is not None})"
