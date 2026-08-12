"""
Diagnostic System - Main Interface

Provides the high-level interface for the LLM-enhanced fault diagnosis
system, integrating all components into a cohesive workflow.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import uuid
import json
from pathlib import Path

from .explainer import LLMEnhancedExplainer
from ..interactive_interface.conversation_agent import ConversationAgent


class DiagnosticSystem:
    """
    Main diagnostic system interface.

    This class provides a high-level API for performing LLM-enhanced
    fault diagnosis with explanations and conversations.
    """

    def __init__(self,
                 llm_config: Optional[Dict[str, Any]] = None,
                 knowledge_config: Optional[Dict[str, Any]] = None,
                 model=None):
        """
        Initialize the diagnostic system.

        Args:
            llm_config: Configuration for LLM providers
            knowledge_config: Configuration for knowledge base
            model: Optional pre-trained diagnosis model
        """
        self.llm_config = llm_config or {}
        self.knowledge_config = knowledge_config or {}
        self.model = model

        # Initialize components
        self.explainer = LLMEnhancedExplainer(llm_config, model)
        self.conversation_agent = ConversationAgent(self.explainer)

        # Session management
        self.active_sessions = {}
        self.diagnostic_history = []

        # System configuration
        self.config = {
            "default_explanation_style": "standard",
            "max_history_size": 100,
            "auto_save_sessions": True
        }

    def diagnose(self,
                 signal_data: Union[torch.Tensor, np.ndarray],
                 model_prediction: Optional[Dict[str, Any]] = None,
                 user_query: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None,
                 style: str = "standard") -> Dict[str, Any]:
        """
        Perform comprehensive fault diagnosis with explanation.

        Args:
            signal_data: Input vibration signal data
            model_prediction: Optional model prediction results
            user_query: Optional user query
            context: Additional context information
            style: Explanation style

        Returns:
            Complete diagnostic results with explanations
        """
        # Generate model prediction if not provided
        if model_prediction is None and self.model is not None:
            model_prediction = self._predict_with_model(signal_data)
        elif model_prediction is None:
            model_prediction = self._create_dummy_prediction(signal_data)

        # Generate explanation
        explanation = self.explainer.explain(
            signal_data, model_prediction, user_query, context, style
        )

        # Create diagnostic result
        diagnostic_result = {
            "session_id": None,  # Not associated with a session
            "timestamp": datetime.now().isoformat(),
            "signal_info": self._analyze_signal_info(signal_data),
            "model_prediction": model_prediction,
            "explanation": explanation,
            "system_info": self._get_system_info()
        }

        # Store in history
        self.diagnostic_history.append(diagnostic_result)

        # Limit history size
        if len(self.diagnostic_history) > self.config["max_history_size"]:
            self.diagnostic_history = self.diagnostic_history[-self.config["max_history_size"]:]

        return diagnostic_result

    def start_conversation(self,
                          signal_data: Union[torch.Tensor, np.ndarray],
                          device_info: Optional[Dict[str, Any]] = None,
                          session_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Start an interactive diagnostic conversation.

        Args:
            signal_data: Input signal data for initial diagnosis
            device_info: Device information
            session_config: Session configuration

        Returns:
            Session information and initial greeting
        """
        # Generate initial diagnosis
        initial_diagnosis = self.diagnose(signal_data)

        # Create session
        session_id = str(uuid.uuid4())
        session = {
            "session_id": session_id,
            "start_time": datetime.now(),
            "signal_data": signal_data,
            "device_info": device_info or {},
            "initial_diagnosis": initial_diagnosis,
            "conversation_history": [],
            "config": session_config or {},
            "status": "active"
        }

        self.active_sessions[session_id] = session

        # Generate greeting
        greeting = self.conversation_agent.generate_greeting(session)

        # Add greeting to conversation history
        session["conversation_history"].append({
            "speaker": "assistant",
            "message": greeting,
            "timestamp": datetime.now(),
            "message_type": "greeting"
        })

        return {
            "session_id": session_id,
            "greeting": greeting,
            "initial_diagnosis": initial_diagnosis,
            "session_info": {
                "start_time": session["start_time"].isoformat(),
                "device_info": session["device_info"]
            }
        }

    def continue_conversation(self,
                             session_id: str,
                             user_message: str) -> Dict[str, Any]:
        """
        Continue an ongoing conversation.

        Args:
            session_id: Session identifier
            user_message: User's message

        Returns:
            Response and session information
        """
        # Validate session
        if session_id not in self.active_sessions:
            return {
                "error": "Session not found",
                "message": "对话会话已过期或不存在",
                "session_id": session_id
            }

        session = self.active_sessions[session_id]

        # Add user message to history
        session["conversation_history"].append({
            "speaker": "user",
            "message": user_message,
            "timestamp": datetime.now(),
            "message_type": "query"
        })

        # Generate response
        try:
            response = self.conversation_agent.process_message(
                user_message,
                session
            )
        except Exception as e:
            print(f"Warning: Conversation processing failed: {e}")
            response = self._generate_fallback_response(user_message, session)

        # Add response to history
        session["conversation_history"].append({
            "speaker": "assistant",
            "message": response,
            "timestamp": datetime.now(),
            "message_type": "response"
        })

        # Update session
        session["last_activity"] = datetime.now()

        # Auto-save if enabled
        if self.config["auto_save_sessions"]:
            self._save_session(session_id)

        return {
            "session_id": session_id,
            "response": response,
            "session_info": {
                "conversation_length": len(session["conversation_history"]),
                "duration": (datetime.now() - session["start_time"]).total_seconds(),
                "status": session["status"]
            }
        }

    def end_conversation(self, session_id: str) -> Dict[str, Any]:
        """
        End a conversation session.

        Args:
            session_id: Session identifier

        Returns:
            Session summary and conclusion
        """
        if session_id not in self.active_sessions:
            return {
                "error": "Session not found",
                "message": "对话会话不存在",
                "session_id": session_id
            }

        session = self.active_sessions[session_id]

        # Generate conclusion
        conclusion = self.conversation_agent.generate_conclusion(session)

        # Update session status
        session["status"] = "ended"
        session["end_time"] = datetime.now()
        session["conclusion"] = conclusion

        # Calculate session statistics
        duration = (session["end_time"] - session["start_time"]).total_seconds()
        num_messages = len(session["conversation_history"])

        session_summary = {
            "session_id": session_id,
            "duration_seconds": duration,
            "num_messages": num_messages,
            "initial_diagnosis": session["initial_diagnosis"],
            "conversation_summary": self._summarize_conversation(session["conversation_history"]),
            "conclusion": conclusion,
            "session_info": {
                "device_info": session["device_info"],
                "start_time": session["start_time"].isoformat(),
                "end_time": session["end_time"].isoformat()
            }
        }

        # Remove from active sessions but keep for history
        del self.active_sessions[session_id]

        # Save final session data
        if self.config["auto_save_sessions"]:
            self._save_session_final(session_id, session_summary)

        return session_summary

    def batch_diagnose(self,
                      signal_data_list: List[Union[torch.Tensor, np.ndarray]],
                      batch_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform batch diagnosis on multiple signals.

        Args:
            signal_data_list: List of signal data
            batch_config: Batch processing configuration

        Returns:
            List of diagnostic results
        """
        batch_config = batch_config or {}
        results = []

        for i, signal_data in enumerate(signal_data_list):
            try:
                result = self.diagnose(
                    signal_data,
                    context=batch_config.get("context", {}),
                    style=batch_config.get("style", "standard")
                )
                result["batch_index"] = i
                results.append(result)
            except Exception as e:
                print(f"Warning: Batch diagnosis failed for item {i}: {e}")
                # Add error result
                results.append({
                    "batch_index": i,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

        return results

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an active session.

        Args:
            session_id: Session identifier

        Returns:
            Session information or None if not found
        """
        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]

        return {
            "session_id": session_id,
            "start_time": session["start_time"].isoformat(),
            "duration_seconds": (datetime.now() - session["start_time"]).total_seconds(),
            "num_messages": len(session["conversation_history"]),
            "status": session["status"],
            "device_info": session["device_info"],
            "last_activity": session.get("last_activity", session["start_time"]).isoformat()
        }

    def get_active_sessions(self) -> List[str]:
        """
        Get list of active session IDs.

        Returns:
            List of active session IDs
        """
        return list(self.active_sessions.keys())

    def get_diagnostic_history(self,
                              limit: int = 50,
                              fault_type: Optional[str] = None,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get diagnostic history with filtering options.

        Args:
            limit: Maximum number of items to return
            fault_type: Filter by fault type
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            Filtered diagnostic history
        """
        history = self.diagnostic_history

        # Apply filters
        if fault_type:
            history = [
                item for item in history
                if item.get("model_prediction", {}).get("fault_type") == fault_type
            ]

        if start_time:
            history = [
                item for item in history
                if datetime.fromisoformat(item["timestamp"]) >= start_time
            ]

        if end_time:
            history = [
                item for item in history
                if datetime.fromisoformat(item["timestamp"]) <= end_time
            ]

        # Sort by timestamp (newest first) and limit
        history.sort(key=lambda x: x["timestamp"], reverse=True)

        return history[:limit]

    def export_data(self,
                   output_path: str,
                   include_sessions: bool = True,
                   include_history: bool = True) -> None:
        """
        Export system data to file.

        Args:
            output_path: Output file path
            include_sessions: Whether to include session data
            include_history: Whether to include diagnostic history
        """
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "config": self.config
        }

        if include_sessions:
            export_data["sessions"] = self._prepare_sessions_for_export()

        if include_history:
            export_data["diagnostic_history"] = self.diagnostic_history

        # Convert datetime objects to strings for JSON serialization
        export_data = self._prepare_for_json(export_data)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

    def _predict_with_model(self, signal_data: Union[torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """Predict fault using the loaded model."""
        if self.model is None:
            return self._create_dummy_prediction(signal_data)

        try:
            with torch.no_grad():
                if isinstance(signal_data, np.ndarray):
                    signal_data = torch.tensor(signal_data, dtype=torch.float32)

                # Add batch dimension if needed
                if signal_data.dim() == 2:
                    signal_data = signal_data.unsqueeze(0)

                prediction = self.model(signal_data)

                # Convert to probabilities
                if prediction.dim() > 1:
                    probabilities = torch.softmax(prediction, dim=-1)
                    confidence, predicted_class = torch.max(probabilities, dim=-1)

                    return {
                        "fault_type": self._get_class_name(predicted_class.item()),
                        "confidence": confidence.item(),
                        "probabilities": probabilities.tolist()[0],
                        "predicted_class": predicted_class.item()
                    }
                else:
                    return {
                        "fault_type": "unknown",
                        "confidence": 0.0,
                        "probabilities": [],
                        "predicted_class": -1
                    }
        except Exception as e:
            print(f"Warning: Model prediction failed: {e}")
            return self._create_dummy_prediction(signal_data)

    def _create_dummy_prediction(self, signal_data: Union[torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """Create a dummy prediction for testing."""
        return {
            "fault_type": "内圈故障",
            "confidence": 0.85,
            "probabilities": [0.1, 0.05, 0.85, 0.0, 0.0],
            "predicted_class": 2,
            "method": "mock_prediction"
        }

    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID."""
        class_names = [
            "正常", "内圈故障", "外圈故障", "滚动体故障",
            "保持架故障", "不对中", "不平衡", "松动",
            "齿轮故障", "其他故障"
        ]
        return class_names[class_id] if class_id < len(class_names) else "未知"

    def _analyze_signal_info(self, signal_data: Union[torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """Analyze basic signal information."""
        if isinstance(signal_data, torch.Tensor):
            signal_np = signal_data.detach().cpu().numpy()
        else:
            signal_np = signal_data

        return {
            "shape": signal_np.shape,
            "data_type": str(signal_np.dtype),
            "min_value": float(np.min(signal_np)),
            "max_value": float(np.max(signal_np)),
            "mean_value": float(np.mean(signal_np)),
            "std_value": float(np.std(signal_np))
        }

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "toolkit_version": "1.0.0",
            "components": {
                "explainer": self.explainer.get_component_info(),
                "conversation_agent": self.conversation_agent.get_info()
            },
            "active_sessions": len(self.active_sessions),
            "diagnostic_history_size": len(self.diagnostic_history),
            "config": self.config
        }

    def _summarize_conversation(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize conversation."""
        if not conversation_history:
            return {"total_messages": 0, "user_messages": 0, "assistant_messages": 0}

        user_messages = [msg for msg in conversation_history if msg["speaker"] == "user"]
        assistant_messages = [msg for msg in conversation_history if msg["speaker"] == "assistant"]
        first_timestamp = self._coerce_datetime(conversation_history[0]["timestamp"])
        last_timestamp = self._coerce_datetime(conversation_history[-1]["timestamp"])

        return {
            "total_messages": len(conversation_history),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "first_message_time": conversation_history[0]["timestamp"],
            "last_message_time": conversation_history[-1]["timestamp"],
            "conversation_duration": (last_timestamp - first_timestamp).total_seconds()
        }

    def _coerce_datetime(self, value: Any) -> datetime:
        """Normalize supported timestamp representations."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        raise TypeError(f"Unsupported timestamp type: {type(value).__name__}")

    def _generate_fallback_response(self, user_message: str, session: Dict[str, Any]) -> str:
        """Generate fallback response when conversation agent fails."""
        fault_type = session["initial_diagnosis"]["model_prediction"]["fault_type"]
        confidence = session["initial_diagnosis"]["model_prediction"]["confidence"]

        return f"抱歉，我暂时无法提供详细的对话。根据初始诊断，检测到 **{fault_type}** 故障，置信度为 {confidence:.1%}。建议联系技术支持获取进一步帮助。"

    def _save_session(self, session_id: str) -> None:
        """Save session data to file."""
        if not hasattr(self, '_session_save_path'):
            self._session_save_path = Path("./sessions")
            self._session_save_path.mkdir(exist_ok=True)

        session_file = self._session_save_path / f"{session_id}.json"
        session_data = self.active_sessions[session_id]
        prepared_data = self._prepare_for_json(session_data)

        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(prepared_data, f, ensure_ascii=False, indent=2)

    def _save_session_final(self, session_id: str, session_summary: Dict[str, Any]) -> None:
        """Save final session data."""
        if not hasattr(self, '_session_save_path'):
            self._session_save_path = Path("./sessions")
            self._session_save_path.mkdir(exist_ok=True)

        session_file = self._session_save_path / f"{session_id}_final.json"
        prepared_data = self._prepare_for_json(session_summary)

        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(prepared_data, f, ensure_ascii=False, indent=2)

    def _prepare_sessions_for_export(self) -> Dict[str, Any]:
        """Prepare sessions data for export."""
        sessions_data = {}

        for session_id, session in self.active_sessions.items():
            sessions_data[session_id] = {
                "session_id": session_id,
                "start_time": session["start_time"].isoformat(),
                "device_info": session["device_info"],
                "initial_diagnosis": session["initial_diagnosis"],
                "conversation_length": len(session["conversation_history"]),
                "status": session["status"]
            }

        return sessions_data

    def _prepare_for_json(self, data: Any) -> Any:
        """Prepare data for JSON serialization."""
        if isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, dict):
            return {key: self._prepare_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy().tolist()
            return data.tolist()
        else:
            return data
