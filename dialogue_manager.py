"""
智能对话管理系统
实现多轮对话、上下文维护、对话策略、对话状态跟踪
"""
import json
import time
import uuid
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class DialogueState(Enum):
    """对话状态"""
    GREETING = "greeting"
    INFORMATION_GATHERING = "information_gathering"
    PROBLEM_SOLVING = "problem_solving"
    CLARIFICATION = "clarification"
    CONCLUSION = "conclusion"
    FAREWELL = "farewell"
    ERROR_RECOVERY = "error_recovery"

class DialogueIntent(Enum):
    """对话意图"""
    QUESTION = "question"
    REQUEST = "request"
    CONFIRMATION = "confirmation"
    OBJECTION = "objection"
    APPRECIATION = "appreciation"
    COMPLAINT = "complaint"
    SMALL_TALK = "small_talk"

@dataclass
class DialogueTurn:
    """对话轮次"""
    turn_id: str
    user_message: str
    assistant_response: str
    timestamp: float
    dialogue_state: DialogueState
    detected_intent: DialogueIntent
    context_updates: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    entities_extracted: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DialogueContext:
    """对话上下文"""
    session_id: str
    user_profile: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[DialogueTurn] = field(default_factory=list)
    current_state: DialogueState = DialogueState.GREETING
    active_topics: List[str] = field(default_factory=list)
    unresolved_questions: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    context_memory: Dict[str, Any] = field(default_factory=dict)
    session_start_time: float = field(default_factory=time.time)
    last_activity_time: float = field(default_factory=time.time)

class IntentClassifier:
    """意图分类器"""
    
    def __init__(self):
        self.intent_patterns = self._initialize_intent_patterns()
        self.confidence_threshold = 0.6
    
    def _initialize_intent_patterns(self) -> Dict[DialogueIntent, List[str]]:
        """初始化意图模式"""
        return {
            DialogueIntent.QUESTION: [
                "什么", "如何", "为什么", "哪里", "何时", "谁", "怎么",
                "what", "how", "why", "where", "when", "who", "which"
            ],
            DialogueIntent.REQUEST: [
                "请", "帮我", "可以", "能否", "希望", "想要",
                "please", "help", "can you", "could you", "would you"
            ],
            DialogueIntent.CONFIRMATION: [
                "是的", "对", "正确", "没错", "确实", "同意",
                "yes", "correct", "right", "agree", "exactly"
            ],
            DialogueIntent.OBJECTION: [
                "不是", "错误", "不对", "不同意", "反对",
                "no", "wrong", "incorrect", "disagree", "object"
            ],
            DialogueIntent.APPRECIATION: [
                "谢谢", "感谢", "太好了", "棒", "优秀", "完美",
                "thank", "thanks", "great", "excellent", "perfect", "awesome"
            ],
            DialogueIntent.COMPLAINT: [
                "问题", "错误", "不满", "糟糕", "失望", "不好",
                "problem", "issue", "wrong", "bad", "terrible", "disappointed"
            ],
            DialogueIntent.SMALL_TALK: [
                "你好", "再见", "天气", "今天", "怎么样",
                "hello", "hi", "goodbye", "weather", "how are you"
            ]
        }
    
    def classify_intent(self, message: str) -> Tuple[DialogueIntent, float]:
        """分类意图"""
        message_lower = message.lower()
        intent_scores = defaultdict(float)
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in message_lower:
                    intent_scores[intent] += 1.0
        
        if not intent_scores:
            return DialogueIntent.QUESTION, 0.3  # 默认为问题
        
        # 计算置信度
        total_matches = sum(intent_scores.values())
        best_intent = max(intent_scores.keys(), key=lambda x: intent_scores[x])
        confidence = intent_scores[best_intent] / max(total_matches, 1.0)
        
        return best_intent, min(confidence, 1.0)

class EntityExtractor:
    """实体提取器"""
    
    def __init__(self):
        self.entity_patterns = self._initialize_entity_patterns()
    
    def _initialize_entity_patterns(self) -> Dict[str, List[str]]:
        """初始化实体模式"""
        return {
            "time": ["今天", "明天", "昨天", "现在", "稍后", "上午", "下午", "晚上"],
            "tech": ["Python", "Java", "JavaScript", "AI", "机器学习", "深度学习", "算法"],
            "action": ["创建", "删除", "修改", "分析", "处理", "优化", "改进"],
            "domain": ["技术", "教育", "商业", "健康", "娱乐", "体育", "科学"]
        }
    
    def extract_entities(self, message: str) -> Dict[str, List[str]]:
        """提取实体"""
        message_lower = message.lower()
        extracted = defaultdict(list)
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                if pattern.lower() in message_lower:
                    extracted[entity_type].append(pattern)
        
        return dict(extracted)

class DialogueStateTracker:
    """对话状态跟踪器"""
    
    def __init__(self):
        self.state_transitions = self._initialize_state_transitions()
        self.state_duration_limits = {
            DialogueState.GREETING: 60,  # 1分钟
            DialogueState.INFORMATION_GATHERING: 300,  # 5分钟
            DialogueState.PROBLEM_SOLVING: 600,  # 10分钟
            DialogueState.CLARIFICATION: 120,  # 2分钟
            DialogueState.CONCLUSION: 60,  # 1分钟
            DialogueState.ERROR_RECOVERY: 180  # 3分钟
        }
    
    def _initialize_state_transitions(self) -> Dict[DialogueState, List[DialogueState]]:
        """初始化状态转换规则"""
        return {
            DialogueState.GREETING: [
                DialogueState.INFORMATION_GATHERING,
                DialogueState.PROBLEM_SOLVING,
                DialogueState.SMALL_TALK
            ],
            DialogueState.INFORMATION_GATHERING: [
                DialogueState.PROBLEM_SOLVING,
                DialogueState.CLARIFICATION,
                DialogueState.CONCLUSION
            ],
            DialogueState.PROBLEM_SOLVING: [
                DialogueState.CLARIFICATION,
                DialogueState.CONCLUSION,
                DialogueState.INFORMATION_GATHERING
            ],
            DialogueState.CLARIFICATION: [
                DialogueState.PROBLEM_SOLVING,
                DialogueState.INFORMATION_GATHERING,
                DialogueState.CONCLUSION
            ],
            DialogueState.CONCLUSION: [
                DialogueState.FAREWELL,
                DialogueState.INFORMATION_GATHERING
            ],
            DialogueState.ERROR_RECOVERY: [
                DialogueState.INFORMATION_GATHERING,
                DialogueState.PROBLEM_SOLVING,
                DialogueState.CLARIFICATION
            ]
        }
    
    def predict_next_state(
        self, 
        current_state: DialogueState, 
        intent: DialogueIntent, 
        context: DialogueContext
    ) -> DialogueState:
        """预测下一个状态"""
        possible_states = self.state_transitions.get(current_state, [current_state])
        
        # 基于意图和上下文选择下一个状态
        if intent == DialogueIntent.QUESTION:
            if current_state == DialogueState.GREETING:
                return DialogueState.INFORMATION_GATHERING
            elif len(context.unresolved_questions) > 2:
                return DialogueState.CLARIFICATION
        
        elif intent == DialogueIntent.REQUEST:
            return DialogueState.PROBLEM_SOLVING
        
        elif intent == DialogueIntent.CONFIRMATION:
            if current_state == DialogueState.CLARIFICATION:
                return DialogueState.PROBLEM_SOLVING
            elif current_state == DialogueState.PROBLEM_SOLVING:
                return DialogueState.CONCLUSION
        
        elif intent == DialogueIntent.APPRECIATION:
            return DialogueState.CONCLUSION
        
        elif intent == DialogueIntent.COMPLAINT:
            return DialogueState.ERROR_RECOVERY
        
        # 检查状态持续时间
        state_duration = time.time() - context.last_activity_time
        if state_duration > self.state_duration_limits.get(current_state, 300):
            if current_state != DialogueState.CONCLUSION:
                return DialogueState.CONCLUSION
        
        return current_state

class DialogueStrategy:
    """对话策略"""
    
    def __init__(self):
        self.response_templates = self._initialize_response_templates()
        self.adaptive_strategies = {}
    
    def _initialize_response_templates(self) -> Dict[DialogueState, Dict[str, str]]:
        """初始化响应模板"""
        return {
            DialogueState.GREETING: {
                "default": "你好！我是智能助手，很高兴为您服务。有什么可以帮助您的吗？",
                "returning_user": "欢迎回来！根据您之前的互动，我记得您对{topic}很感兴趣。"
            },
            DialogueState.INFORMATION_GATHERING: {
                "default": "我需要了解更多信息来更好地帮助您。",
                "clarifying": "让我确认一下我的理解是否正确...",
                "probing": "您能详细说明一下{entity}吗？"
            },
            DialogueState.PROBLEM_SOLVING: {
                "default": "基于您提供的信息，我建议...",
                "step_by_step": "让我们一步步来解决这个问题：",
                "alternative": "除了之前的方案，您还可以考虑..."
            },
            DialogueState.CLARIFICATION: {
                "default": "为了确保我正确理解您的需求，",
                "confusion": "我对{point}有些困惑，您能解释一下吗？",
                "confirmation": "您的意思是{interpretation}，对吗？"
            },
            DialogueState.CONCLUSION: {
                "default": "希望我的回答对您有帮助。",
                "summary": "总结一下，我们讨论了{topics}。",
                "next_steps": "接下来您可以..."
            },
            DialogueState.ERROR_RECOVERY: {
                "default": "抱歉出现了问题，让我重新理解您的需求。",
                "misunderstanding": "看起来我误解了您的意思，",
                "technical_error": "遇到了技术问题，正在尝试修复..."
            }
        }
    
    def select_response_strategy(
        self, 
        state: DialogueState, 
        intent: DialogueIntent, 
        context: DialogueContext
    ) -> str:
        """选择响应策略"""
        templates = self.response_templates.get(state, {})
        
        # 根据上下文选择合适的模板
        if state == DialogueState.GREETING:
            if len(context.conversation_history) > 0:
                return templates.get("returning_user", templates["default"])
            return templates["default"]
        
        elif state == DialogueState.INFORMATION_GATHERING:
            if len(context.unresolved_questions) > 1:
                return templates.get("clarifying", templates["default"])
            return templates["default"]
        
        elif state == DialogueState.PROBLEM_SOLVING:
            if len(context.conversation_history) > 3:
                return templates.get("step_by_step", templates["default"])
            return templates["default"]
        
        return templates.get("default", "我正在思考如何最好地回应您。")

class DialogueManager:
    """对话管理器主类"""
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.state_tracker = DialogueStateTracker()
        self.dialogue_strategy = DialogueStrategy()
        
        self.active_sessions = {}
        self.session_timeout = 1800  # 30分钟
        self.max_turns_per_session = 50
        
    def start_session(self, user_id: str = None) -> str:
        """开始新会话"""
        session_id = str(uuid.uuid4())
        
        context = DialogueContext(
            session_id=session_id,
            user_profile={"user_id": user_id} if user_id else {}
        )
        
        self.active_sessions[session_id] = context
        logger.info(f"Started new dialogue session: {session_id}")
        
        return session_id
    
    async def process_message(
        self, 
        session_id: str, 
        user_message: str,
        additional_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """处理用户消息"""
        if session_id not in self.active_sessions:
            session_id = self.start_session()
        
        context = self.active_sessions[session_id]
        
        # 检查会话超时
        if self._is_session_expired(context):
            self._cleanup_session(session_id)
            session_id = self.start_session()
            context = self.active_sessions[session_id]
        
        # 分析用户消息
        intent, intent_confidence = self.intent_classifier.classify_intent(user_message)
        entities = self.entity_extractor.extract_entities(user_message)
        
        # 更新对话状态
        new_state = self.state_tracker.predict_next_state(
            context.current_state, intent, context
        )
        
        # 选择响应策略
        response_template = self.dialogue_strategy.select_response_strategy(
            new_state, intent, context
        )
        
        # 更新上下文
        context.current_state = new_state
        context.last_activity_time = time.time()
        
        # 更新活跃主题
        if entities:
            for entity_type, entity_list in entities.items():
                context.active_topics.extend(entity_list)
                context.active_topics = list(set(context.active_topics))[-10:]  # 保持最近10个主题
        
        # 创建对话轮次记录
        turn = DialogueTurn(
            turn_id=f"turn_{len(context.conversation_history) + 1}",
            user_message=user_message,
            assistant_response=response_template,
            timestamp=time.time(),
            dialogue_state=new_state,
            detected_intent=intent,
            context_updates=additional_context or {},
            confidence_scores={"intent": intent_confidence},
            entities_extracted=entities
        )
        
        context.conversation_history.append(turn)
        
        # 限制历史长度
        if len(context.conversation_history) > self.max_turns_per_session:
            context.conversation_history = context.conversation_history[-self.max_turns_per_session:]
        
        return {
            "session_id": session_id,
            "response_template": response_template,
            "dialogue_state": new_state.value,
            "detected_intent": intent.value,
            "intent_confidence": intent_confidence,
            "extracted_entities": entities,
            "active_topics": context.active_topics,
            "conversation_turn": len(context.conversation_history),
            "context_summary": self._generate_context_summary(context)
        }
    
    def _is_session_expired(self, context: DialogueContext) -> bool:
        """检查会话是否过期"""
        return (time.time() - context.last_activity_time) > self.session_timeout
    
    def _cleanup_session(self, session_id: str):
        """清理过期会话"""
        if session_id in self.active_sessions:
            logger.info(f"Cleaning up expired session: {session_id}")
            del self.active_sessions[session_id]
    
    def _generate_context_summary(self, context: DialogueContext) -> Dict[str, Any]:
        """生成上下文摘要"""
        recent_states = [turn.dialogue_state.value for turn in context.conversation_history[-5:]]
        recent_intents = [turn.detected_intent.value for turn in context.conversation_history[-5:]]
        
        return {
            "session_duration": time.time() - context.session_start_time,
            "total_turns": len(context.conversation_history),
            "recent_states": recent_states,
            "recent_intents": recent_intents,
            "dominant_intent": max(set(recent_intents), key=recent_intents.count) if recent_intents else None,
            "conversation_flow_score": self._calculate_flow_score(context)
        }
    
    def _calculate_flow_score(self, context: DialogueContext) -> float:
        """计算对话流畅度评分"""
        if len(context.conversation_history) < 2:
            return 1.0
        
        # 基于状态转换的合理性评分
        valid_transitions = 0
        total_transitions = len(context.conversation_history) - 1
        
        for i in range(total_transitions):
            current_state = context.conversation_history[i].dialogue_state
            next_state = context.conversation_history[i + 1].dialogue_state
            
            valid_next_states = self.state_tracker.state_transitions.get(current_state, [])
            if next_state in valid_next_states or next_state == current_state:
                valid_transitions += 1
        
        return valid_transitions / max(total_transitions, 1)
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息"""
        if session_id not in self.active_sessions:
            return None
        
        context = self.active_sessions[session_id]
        return {
            "session_id": session_id,
            "current_state": context.current_state.value,
            "active_topics": context.active_topics,
            "conversation_turns": len(context.conversation_history),
            "session_duration": time.time() - context.session_start_time,
            "last_activity": context.last_activity_time,
            "user_profile": context.user_profile
        }
    
    def get_all_active_sessions(self) -> List[str]:
        """获取所有活跃会话"""
        # 清理过期会话
        expired_sessions = [
            session_id for session_id, context in self.active_sessions.items()
            if self._is_session_expired(context)
        ]
        
        for session_id in expired_sessions:
            self._cleanup_session(session_id)
        
        return list(self.active_sessions.keys())
    
    def end_session(self, session_id: str) -> bool:
        """结束会话"""
        if session_id in self.active_sessions:
            context = self.active_sessions[session_id]
            context.current_state = DialogueState.FAREWELL
            
            # 可以在这里保存会话数据到长期存储
            
            del self.active_sessions[session_id]
            logger.info(f"Ended dialogue session: {session_id}")
            return True
        
        return False