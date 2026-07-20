"""
Conversation Engine for Interactive Diagnostic Dialogues

Provides an interactive conversation interface that allows engineers to
engage in multi-turn dialogues with the AI system for fault diagnosis,
explanation, and maintenance guidance.
"""

from typing import Dict, Any, List, Optional, Callable
import json
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DialogueState(Enum):
    """States of the conversation."""
    GREETING = "greeting"
    INITIAL_DIAGNOSIS = "initial_diagnosis"
    DETAILED_ANALYSIS = "detailed_analysis"
    MAINTENANCE_PLANNING = "maintenance_planning"
    FOLLOW_UP = "follow_up"
    CONCLUSION = "conclusion"


class QueryType(Enum):
    """Types of user queries."""
    FAULT_IDENTIFICATION = "fault_identification"
    CAUSE_ANALYSIS = "cause_analysis"
    SEVERITY_ASSESSMENT = "severity_assessment"
    MAINTENANCE_RECOMMENDATION = "maintenance_recommendation"
    TECHNICAL_EXPLANATION = "technical_explanation"
    GENERAL_QUESTION = "general_question"


@dataclass
class ConversationTurn:
    """Single turn in the conversation."""
    turn_id: str
    timestamp: datetime
    speaker: str  # "user" or "assistant"
    content: str
    query_type: Optional[QueryType]
    intent: Optional[str]
    entities: List[str]
    context_snapshot: Dict[str, Any]
    response_metadata: Dict[str, Any]


@dataclass
class ConversationSession:
    """Complete conversation session."""
    session_id: str
    start_time: datetime
    device_info: Dict[str, Any]
    initial_diagnosis: Dict[str, Any]
    conversation_state: DialogueState
    turns: List[ConversationTurn]
    session_metadata: Dict[str, Any]


class ConversationEngine:
    """
    Manages interactive diagnostic conversations.

    This engine handles multi-turn dialogues, maintains context,
    processes user queries, and generates appropriate responses.
    """

    def __init__(self, model, llm_config: Dict[str, Any]):
        """
        Initialize conversation engine.

        Args:
            model: The fault diagnosis model
            llm_config: LLM configuration
        """
        self.model = model
        self.llm_config = llm_config
        self.active_sessions = {}
        self.query_processor = self._initialize_query_processor()
        self.response_generator = self._initialize_response_generator()
        self.state_transitions = self._initialize_state_transitions()
        self.context_manager = self._initialize_context_manager()

    def start_session(self, input_data, device_info: Optional[Dict[str, Any]] = None) -> ConversationSession:
        """
        Start a new conversation session.

        Args:
            input_data: Input data for initial diagnosis
            device_info: Device information

        Returns:
            New conversation session
        """
        session_id = str(uuid.uuid4())
        start_time = datetime.now()

        # Generate initial diagnosis
        initial_diagnosis = self._generate_initial_diagnosis(input_data)

        # Create session
        session = ConversationSession(
            session_id=session_id,
            start_time=start_time,
            device_info=device_info or {},
            initial_diagnosis=initial_diagnosis,
            conversation_state=DialogueState.GREETING,
            turns=[],
            session_metadata={
                "model_name": type(self.model).__name__,
                "input_data_shape": input_data.shape if hasattr(input_data, 'shape') else str(type(input_data)),
                "llm_config": self.llm_config
            }
        )

        self.active_sessions[session_id] = session

        # Generate greeting message
        greeting = self._generate_greeting(session)
        greeting_turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            speaker="assistant",
            content=greeting,
            query_type=None,
            intent="greeting",
            entities=[],
            context_snapshot=self._capture_context_snapshot(session),
            response_metadata={"type": "greeting"}
        )
        session.turns.append(greeting_turn)

        logger.info(f"Started conversation session {session_id}")
        return session

    def process_user_query(self,
                          session_id: str,
                          user_query: str,
                          additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process user query and generate response.

        Args:
            session_id: Session identifier
            user_query: User's question or request
            additional_context: Additional context information

        Returns:
            Assistant's response
        """
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Process user query
        processed_query = self.query_processor.process_query(
            user_query, session.conversation_state
        )

        # Create user turn
        user_turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            speaker="user",
            content=user_query,
            query_type=processed_query["query_type"],
            intent=processed_query["intent"],
            entities=processed_query["entities"],
            context_snapshot=self._capture_context_snapshot(session),
            response_metadata={}
        )
        session.turns.append(user_turn)

        # Update conversation state if needed
        new_state = self._determine_next_state(session, processed_query)
        if new_state != session.conversation_state:
            session.conversation_state = new_state

        # Generate response
        response = self._generate_response(session, processed_query, additional_context)

        # Create assistant turn
        assistant_turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            speaker="assistant",
            content=response,
            query_type=processed_query["query_type"],
            intent="response",
            entities=[],
            context_snapshot=self._capture_context_snapshot(session),
            response_metadata={
                "state": session.conversation_state.value,
                "query_type": processed_query["query_type"].value if processed_query["query_type"] else None
            }
        )
        session.turns.append(assistant_turn)

        logger.info(f"Processed query in session {session_id}, state: {session.conversation_state.value}")
        return response

    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get summary of conversation session.

        Args:
            session_id: Session identifier

        Returns:
            Conversation summary
        """
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        summary = {
            "session_id": session_id,
            "duration": datetime.now() - session.start_time,
            "num_turns": len(session.turns),
            "current_state": session.conversation_state.value,
            "initial_diagnosis": session.initial_diagnosis,
            "device_info": session.device_info,
            "key_topics": self._extract_key_topics(session),
            "recommendations": self._extract_recommendations(session),
            "follow_up_actions": self._extract_follow_up_actions(session)
        }

        return summary

    def end_session(self, session_id: str) -> Dict[str, Any]:
        """
        End conversation session and provide summary.

        Args:
            session_id: Session identifier

        Returns:
            Session summary and final recommendations
        """
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Generate concluding message
        conclusion = self._generate_conclusion(session)

        # Create final assistant turn
        conclusion_turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            speaker="assistant",
            content=conclusion,
            query_type=None,
            intent="conclusion",
            entities=[],
            context_snapshot=self._capture_context_snapshot(session),
            response_metadata={"type": "conclusion"}
        )
        session.turns.append(conclusion_turn)

        # Get summary
        summary = self.get_conversation_summary(session_id)

        # Remove from active sessions
        del self.active_sessions[session_id]

        logger.info(f"Ended conversation session {session_id}")
        return {
            "conclusion": conclusion,
            "summary": summary
        }

    def save_session(self, session_id: str, filename: str) -> None:
        """
        Save conversation session to file.

        Args:
            session_id: Session identifier
            filename: Output filename
        """
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        session_data = {
            "session_id": session.session_id,
            "start_time": session.start_time.isoformat(),
            "device_info": session.device_info,
            "initial_diagnosis": session.initial_diagnosis,
            "conversation_state": session.conversation_state.value,
            "turns": [self._serialize_turn(turn) for turn in session.turns],
            "session_metadata": session.session_metadata
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Saved session {session_id} to {filename}")

    def load_session(self, filename: str) -> str:
        """
        Load conversation session from file.

        Args:
            filename: Input filename

        Returns:
            Session identifier
        """
        with open(filename, 'r', encoding='utf-8') as f:
            session_data = json.load(f)

        # Reconstruct session
        session = ConversationSession(
            session_id=session_data["session_id"],
            start_time=datetime.fromisoformat(session_data["start_time"]),
            device_info=session_data["device_info"],
            initial_diagnosis=session_data["initial_diagnosis"],
            conversation_state=DialogueState(session_data["conversation_state"]),
            turns=[self._deserialize_turn(turn) for turn in session_data["turns"]],
            session_metadata=session_data["session_metadata"]
        )

        self.active_sessions[session.session_id] = session
        logger.info(f"Loaded session {session.session_id} from {filename}")

        return session.session_id

    def _generate_initial_diagnosis(self, input_data) -> Dict[str, Any]:
        """Generate initial diagnosis from input data."""
        try:
            # Get basic diagnosis from model
            if hasattr(self.model, 'predict'):
                prediction = self.model.predict(input_data)
                if hasattr(self.model, 'explain'):
                    explanation = self.model.explain(input_data)
                else:
                    explanation = {}
            else:
                prediction = {"fault_type": "unknown", "confidence": 0.0}
                explanation = {}

            return {
                "prediction": prediction,
                "explanation": explanation,
                "timestamp": datetime.now().isoformat(),
                "status": "initial"
            }

        except Exception as e:
            logger.error(f"Failed to generate initial diagnosis: {e}")
            return {
                "fault_type": "error",
                "confidence": 0.0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _generate_greeting(self, session: ConversationSession) -> str:
        """Generate greeting message."""
        fault_type = session.initial_diagnosis.get("fault_type", "未知")
        confidence = session.initial_diagnosis.get("confidence", 0.0)

        greeting = f"""您好！我是机械故障诊断专家助手。

根据您的设备数据，我初步检测到可能存在 **{fault_type}** 类型的故障，诊断置信度为 {confidence:.1%}。

我可以为您提供以下帮助：
• 详细的故障机理分析
• 维修建议和方案制定
• 风险评估和紧急程度判断
• 后续监测和预防措施

请问您希望了解哪方面的信息？或者您有其他相关问题需要讨论吗？"""

        return greeting

    def _determine_next_state(self, session: ConversationSession, processed_query: Dict[str, Any]) -> DialogueState:
        """Determine next conversation state based on query."""
        current_state = session.conversation_state
        query_type = processed_query.get("query_type")
        intent = processed_query.get("intent", "")

        # State transition logic
        if current_state == DialogueState.GREETING:
            if query_type in [QueryType.FAULT_IDENTIFICATION, QueryType.TECHNICAL_EXPLANATION]:
                return DialogueState.DETAILED_ANALYSIS
            elif query_type == QueryType.MAINTENANCE_RECOMMENDATION:
                return DialogueState.MAINTENANCE_PLANNING
            else:
                return DialogueState.INITIAL_DIAGNOSIS

        elif current_state == DialogueState.INITIAL_DIAGNOSIS:
            if "维护" in intent or "维修" in intent:
                return DialogueState.MAINTENANCE_PLANNING
            else:
                return DialogueState.DETAILED_ANALYSIS

        elif current_state == DialogueState.DETAILED_ANALYSIS:
            if "维护" in intent or "维修" in intent:
                return DialogueState.MAINTENANCE_PLANNING
            elif "结束" in intent or "完成" in intent:
                return DialogueState.CONCLUSION
            else:
                return DialogueState.FOLLOW_UP

        elif current_state == DialogueState.MAINTENANCE_PLANNING:
            if "结束" in intent or "完成" in intent:
                return DialogueState.CONCLUSION
            else:
                return DialogueState.FOLLOW_UP

        elif current_state == DialogueState.FOLLOW_UP:
            if "结束" in intent or "完成" in intent:
                return DialogueState.CONCLUSION
            elif "维护" in intent or "维修" in intent:
                return DialogueState.MAINTENANCE_PLANNING
            else:
                return DialogueState.DETAILED_ANALYSIS

        return current_state

    def _generate_response(self,
                          session: ConversationSession,
                          processed_query: Dict[str, Any],
                          additional_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response to user query."""
        try:
            # Use LLM to generate response
            if hasattr(self.model, 'explain_with_llm'):
                # Create input data from context
                context_data = {
                    "session_info": {
                        "session_id": session.session_id,
                        "state": session.conversation_state.value,
                        "num_turns": len(session.turns)
                    },
                    "initial_diagnosis": session.initial_diagnosis,
                    "device_info": session.device_info,
                    "conversation_history": self._format_conversation_history(session),
                    "current_query": processed_query,
                    "additional_context": additional_context or {}
                }

                # Mock input data for LLM explanation
                import torch
                mock_input = torch.randn(1, 1024, 1)  # Create mock input

                llm_response = self.model.explain_with_llm(
                    mock_input,
                    user_query=processed_query["original_query"],
                    context=context_data
                )

                if llm_response and "llm_enhanced_explanation" in llm_response:
                    return llm_response["llm_enhanced_explanation"].get("response", "")

            # Fallback to rule-based response
            return self._generate_rule_based_response(session, processed_query)

        except Exception as e:
            logger.error(f"Failed to generate LLM response: {e}")
            return self._generate_fallback_response(processed_query)

    def _format_conversation_history(self, session: ConversationSession) -> List[Dict[str, str]]:
        """Format conversation history for LLM context."""
        history = []
        for turn in session.turns[-5:]:  # Keep last 5 turns
            history.append({
                "speaker": turn.speaker,
                "content": turn.content,
                "timestamp": turn.timestamp.strftime("%H:%M:%S")
            })
        return history

    def _generate_rule_based_response(self,
                                    session: ConversationSession,
                                    processed_query: Dict[str, Any]) -> str:
        """Generate rule-based response as fallback."""
        query_type = processed_query.get("query_type")
        fault_type = session.initial_diagnosis.get("fault_type", "未知故障")

        responses = {
            QueryType.FAULT_IDENTIFICATION: f"根据分析，设备主要问题是 **{fault_type}**。这是一种常见的机械故障，通常需要及时处理以避免进一步损坏。",
            QueryType.CAUSE_ANALYSIS: f"**{fault_type}** 的常见原因包括：1) 正常磨损 2) 润滑不足 3) 过载运行 4) 安装不当。建议结合设备运行历史进行具体分析。",
            QueryType.SEVERITY_ASSESSMENT: f"基于当前诊断结果，**{fault_type}** 的严重程度需要进一步评估。建议关注振动幅值变化趋势，并安排必要的检查。",
            QueryType.MAINTENANCE_RECOMMENDATION: f"对于 **{fault_type}**，建议的维护措施包括：1) 详细检查设备状态 2) 准备必要的备件 3) 制定维修计划 4) 安排合适的维修窗口。",
            QueryType.TECHNICAL_EXPLANATION: f"**{fault_type}** 是一种机械故障类型。其技术特征包括振动信号的变化、频率成分的异常等。需要结合频谱分析进行深入诊断。"
        }

        if query_type and query_type in responses:
            base_response = responses[query_type]
        else:
            base_response = f"关于您的设备问题（**{fault_type}**），我建议从以下几个方面进行分析：故障机理、影响因素、维修方案和预防措施。"

        # Add follow-up question
        base_response += "\n\n您希望了解更具体的哪个方面？"

        return base_response

    def _generate_fallback_response(self, processed_query: Dict[str, Any]) -> str:
        """Generate fallback response when all else fails."""
        return "抱歉，我现在无法提供详细的回答。这可能是因为技术问题或信息不足。建议您稍后重试或联系技术支持团队。"

    def _generate_conclusion(self, session: ConversationSession) -> str:
        """Generate concluding message."""
        duration = datetime.now() - session.start_time
        fault_type = session.initial_diagnosis.get("fault_type", "未知故障")

        conclusion = f"""感谢您的咨询！我们的对话持续了 {duration.total_seconds():.0f} 秒。

## 讨论要点总结
• 主要问题：**{fault_type}**
• 对话轮次：{len(session.turns)} 次
• 当前状态：已详细分析

## 建议后续行动
1. 根据讨论结果制定具体的维修计划
2. 加强设备状态监测
3. 定期进行预防性维护
4. 建立故障诊断记录档案

如果您还需要进一步的帮助，请随时开启新的对话。祝您工作顺利！"""

        return conclusion

    def _capture_context_snapshot(self, session: ConversationSession) -> Dict[str, Any]:
        """Capture current context snapshot."""
        return {
            "session_state": session.conversation_state.value,
            "turn_count": len(session.turns),
            "elapsed_time": (datetime.now() - session.start_time).total_seconds(),
            "device_type": session.device_info.get("device_type", "unknown"),
            "initial_fault": session.initial_diagnosis.get("fault_type", "unknown")
        }

    def _extract_key_topics(self, session: ConversationSession) -> List[str]:
        """Extract key topics from conversation."""
        topics = []
        for turn in session.turns:
            if turn.speaker == "user":
                content = turn.content.lower()
                if "故障" in content:
                    topics.append("故障分析")
                if "维修" in content or "维护" in content:
                    topics.append("维护方案")
                if "原因" in content:
                    topics.append("原因分析")
                if "严重" in content:
                    topics.append("严重程度评估")

        return list(set(topics))  # Remove duplicates

    def _extract_recommendations(self, session: ConversationSession) -> List[str]:
        """Extract recommendations from conversation."""
        recommendations = []
        for turn in session.turns:
            if turn.speaker == "assistant":
                content = turn.content
                if "建议" in content:
                    # Simple extraction - could be enhanced with NLP
                    lines = content.split('\n')
                    for line in lines:
                        if "建议" in line and len(line.strip()) > 10:
                            recommendations.append(line.strip())

        return recommendations[:5]  # Return top 5 recommendations

    def _extract_follow_up_actions(self, session: ConversationSession) -> List[str]:
        """Extract follow-up actions from conversation."""
        actions = ["定期监测设备振动状态", "建立故障诊断记录", "制定预防性维护计划"]

        # Extract specific actions from conversation
        for turn in session.turns:
            if turn.speaker == "assistant":
                content = turn.content.lower()
                if "检查" in content:
                    actions.append("详细检查设备状态")
                if "监测" in content:
                    actions.append("加强状态监测")

        return list(set(actions))  # Remove duplicates

    def _serialize_turn(self, turn: ConversationTurn) -> Dict[str, Any]:
        """Serialize conversation turn for storage."""
        return {
            "turn_id": turn.turn_id,
            "timestamp": turn.timestamp.isoformat(),
            "speaker": turn.speaker,
            "content": turn.content,
            "query_type": turn.query_type.value if turn.query_type else None,
            "intent": turn.intent,
            "entities": turn.entities,
            "context_snapshot": turn.context_snapshot,
            "response_metadata": turn.response_metadata
        }

    def _deserialize_turn(self, turn_data: Dict[str, Any]) -> ConversationTurn:
        """Deserialize conversation turn from storage."""
        return ConversationTurn(
            turn_id=turn_data["turn_id"],
            timestamp=datetime.fromisoformat(turn_data["timestamp"]),
            speaker=turn_data["speaker"],
            content=turn_data["content"],
            query_type=QueryType(turn_data["query_type"]) if turn_data["query_type"] else None,
            intent=turn_data["intent"],
            entities=turn_data["entities"],
            context_snapshot=turn_data["context_snapshot"],
            response_metadata=turn_data["response_metadata"]
        )

    def _initialize_query_processor(self) -> Any:
        """Initialize query processor."""
        # This would typically import and initialize the actual query processor
        class MockQueryProcessor:
            def process_query(self, query: str, state: DialogueState) -> Dict[str, Any]:
                # Simple mock processing
                query_lower = query.lower()
                if "故障" in query_lower or "问题" in query_lower:
                    query_type = QueryType.FAULT_IDENTIFICATION
                elif "原因" in query_lower:
                    query_type = QueryType.CAUSE_ANALYSIS
                elif "严重" in query_lower or "程度" in query_lower:
                    query_type = QueryType.SEVERITY_ASSESSMENT
                elif "维修" in query_lower or "维护" in query_lower:
                    query_type = QueryType.MAINTENANCE_RECOMMENDATION
                elif "解释" in query_lower or "技术" in query_lower:
                    query_type = QueryType.TECHNICAL_EXPLANATION
                else:
                    query_type = QueryType.GENERAL_QUESTION

                return {
                    "original_query": query,
                    "query_type": query_type,
                    "intent": "user_inquiry",
                    "entities": [],
                    "processed_text": query
                }

        return MockQueryProcessor()

    def _initialize_response_generator(self) -> Any:
        """Initialize response generator."""
        # Mock implementation
        class MockResponseGenerator:
            def generate(self, context: Dict[str, Any]) -> str:
                return "基于当前分析，建议进行详细检查。"

        return MockResponseGenerator()

    def _initialize_state_transitions(self) -> Dict[DialogueState, Dict[str, DialogueState]]:
        """Initialize state transition rules."""
        return {}

    def _initialize_context_manager(self) -> Any:
        """Initialize context manager."""
        # Mock implementation
        class MockContextManager:
            def update_context(self, session: ConversationSession, turn: ConversationTurn):
                pass

        return MockContextManager()