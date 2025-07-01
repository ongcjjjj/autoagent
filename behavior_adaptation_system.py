"""
动态行为适应系统
实现行为模式学习、动态策略调整、个性化交互、行为预测
"""
import json
import time
import math
import random
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class BehaviorType(Enum):
    """行为类型"""
    COMMUNICATION = "communication"
    LEARNING = "learning" 
    PROBLEM_SOLVING = "problem_solving"
    DECISION_MAKING = "decision_making"
    EMOTIONAL_RESPONSE = "emotional_response"
    ADAPTATION = "adaptation"
    EXPLORATION = "exploration"
    SOCIAL_INTERACTION = "social_interaction"

class AdaptationStrategy(Enum):
    """适应策略"""
    CONSERVATIVE = "conservative"  # 保守型
    AGGRESSIVE = "aggressive"     # 激进型
    BALANCED = "balanced"         # 平衡型
    EXPLORATORY = "exploratory"   # 探索型
    REACTIVE = "reactive"         # 反应型

@dataclass
class BehaviorPattern:
    """行为模式"""
    pattern_id: str
    behavior_type: BehaviorType
    triggers: List[str]
    actions: List[str]
    success_rate: float = 0.0
    usage_count: int = 0
    last_used: float = 0.0
    confidence: float = 0.5
    context_conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BehaviorEvent:
    """行为事件"""
    event_id: str
    behavior_type: BehaviorType
    trigger: str
    action_taken: str
    outcome: str
    success: bool
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    feedback_score: float = 0.5
    learning_value: float = 0.0

@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    interaction_patterns: Dict[str, int] = field(default_factory=dict)
    communication_style: str = "neutral"
    expertise_level: str = "intermediate"
    response_time_preference: float = 2.0  # 秒
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)
    personality_traits: Dict[str, float] = field(default_factory=dict)
    last_interaction: float = 0.0
    total_interactions: int = 0

@dataclass
class AdaptationState:
    """适应状态"""
    current_strategy: AdaptationStrategy
    confidence_level: float = 0.5
    exploration_rate: float = 0.2
    learning_rate: float = 0.1
    adaptation_speed: float = 0.5
    stress_level: float = 0.0
    performance_trend: List[float] = field(default_factory=list)
    last_adaptation: float = 0.0

class BehaviorLearner:
    """行为学习器"""
    
    def __init__(self):
        self.learned_patterns = {}
        self.pattern_effectiveness = defaultdict(list)
        self.context_correlations = defaultdict(dict)
        self.learning_history = deque(maxlen=1000)
        
    def learn_from_event(self, event: BehaviorEvent) -> Dict[str, Any]:
        """从行为事件中学习"""
        learning_result = {
            "pattern_updates": 0,
            "new_patterns": 0,
            "confidence_changes": 0
        }
        
        # 更新现有模式
        pattern_key = f"{event.behavior_type.value}_{event.trigger}"
        
        if pattern_key in self.learned_patterns:
            pattern = self.learned_patterns[pattern_key]
            
            # 更新成功率
            old_success_rate = pattern.success_rate
            total_uses = pattern.usage_count + 1
            new_success_rate = (pattern.success_rate * pattern.usage_count + (1.0 if event.success else 0.0)) / total_uses
            
            pattern.success_rate = new_success_rate
            pattern.usage_count = total_uses
            pattern.last_used = event.timestamp
            
            # 更新置信度
            confidence_change = abs(new_success_rate - old_success_rate)
            if confidence_change > 0.1:
                pattern.confidence = min(1.0, pattern.confidence + 0.1)
                learning_result["confidence_changes"] += 1
            
            learning_result["pattern_updates"] += 1
            
        else:
            # 创建新模式
            new_pattern = BehaviorPattern(
                pattern_id=pattern_key,
                behavior_type=event.behavior_type,
                triggers=[event.trigger],
                actions=[event.action_taken],
                success_rate=1.0 if event.success else 0.0,
                usage_count=1,
                last_used=event.timestamp,
                confidence=0.5,
                context_conditions=self._extract_context_conditions(event.context)
            )
            
            self.learned_patterns[pattern_key] = new_pattern
            learning_result["new_patterns"] += 1
        
        # 记录效果
        self.pattern_effectiveness[pattern_key].append({
            "success": event.success,
            "feedback_score": event.feedback_score,
            "timestamp": event.timestamp,
            "context": event.context
        })
        
        # 学习上下文相关性
        self._learn_context_correlations(event)
        
        # 记录学习历史
        self.learning_history.append({
            "event_id": event.event_id,
            "learning_result": learning_result,
            "timestamp": time.time()
        })
        
        return learning_result
    
    def _extract_context_conditions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """提取上下文条件"""
        conditions = {}
        
        for key, value in context.items():
            if isinstance(value, (int, float)):
                # 数值型条件：提取范围
                conditions[f"{key}_range"] = {"min": value * 0.8, "max": value * 1.2}
            elif isinstance(value, str):
                # 字符串型条件：直接匹配
                conditions[key] = value
            elif isinstance(value, bool):
                # 布尔型条件
                conditions[key] = value
        
        return conditions
    
    def _learn_context_correlations(self, event: BehaviorEvent):
        """学习上下文相关性"""
        for context_key, context_value in event.context.items():
            if isinstance(context_value, (int, float)):
                behavior_key = f"{event.behavior_type.value}_{event.action_taken}"
                
                if behavior_key not in self.context_correlations[context_key]:
                    self.context_correlations[context_key][behavior_key] = []
                
                self.context_correlations[context_key][behavior_key].append({
                    "value": context_value,
                    "success": event.success,
                    "feedback": event.feedback_score
                })
    
    def predict_pattern_success(self, pattern: BehaviorPattern, context: Dict[str, Any]) -> float:
        """预测模式成功概率"""
        base_probability = pattern.success_rate
        
        # 基于上下文调整
        context_adjustment = 0.0
        adjustment_count = 0
        
        for condition_key, condition_value in pattern.context_conditions.items():
            if condition_key.endswith("_range") and condition_key[:-6] in context:
                actual_value = context[condition_key[:-6]]
                if isinstance(actual_value, (int, float)) and isinstance(condition_value, dict):
                    min_val = condition_value.get("min", 0)
                    max_val = condition_value.get("max", 1)
                    
                    if min_val <= actual_value <= max_val:
                        context_adjustment += 0.1
                    else:
                        context_adjustment -= 0.1
                    
                    adjustment_count += 1
            elif condition_key in context:
                if context[condition_key] == condition_value:
                    context_adjustment += 0.1
                else:
                    context_adjustment -= 0.1
                
                adjustment_count += 1
        
        # 平均上下文调整
        if adjustment_count > 0:
            context_adjustment /= adjustment_count
        
        # 考虑模式置信度
        confidence_factor = pattern.confidence
        
        # 考虑使用频率（更频繁使用的模式更可靠）
        frequency_factor = min(1.0, pattern.usage_count / 10.0)
        
        final_probability = base_probability + context_adjustment * confidence_factor * frequency_factor
        
        return max(0.0, min(1.0, final_probability))
    
    def get_best_pattern(self, behavior_type: BehaviorType, trigger: str, context: Dict[str, Any]) -> Optional[BehaviorPattern]:
        """获取最佳行为模式"""
        candidates = []
        
        for pattern in self.learned_patterns.values():
            if pattern.behavior_type == behavior_type and trigger in pattern.triggers:
                predicted_success = self.predict_pattern_success(pattern, context)
                candidates.append((pattern, predicted_success))
        
        if not candidates:
            return None
        
        # 按预测成功率排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[0][0]

class PersonalityEngine:
    """个性化引擎"""
    
    def __init__(self):
        self.user_profiles = {}
        self.personality_models = {}
        self.interaction_patterns = defaultdict(list)
        self.adaptation_rules = {}
        
        self._initialize_personality_models()
    
    def _initialize_personality_models(self):
        """初始化个性化模型"""
        self.personality_models = {
            "communication_style": {
                "formal": {"precision": 0.9, "warmth": 0.3, "directness": 0.8},
                "casual": {"precision": 0.6, "warmth": 0.8, "directness": 0.5},
                "friendly": {"precision": 0.7, "warmth": 0.9, "directness": 0.6},
                "professional": {"precision": 0.9, "warmth": 0.5, "directness": 0.9}
            },
            "expertise_level": {
                "beginner": {"detail_level": 0.9, "explanation_depth": 0.8, "examples": 0.9},
                "intermediate": {"detail_level": 0.6, "explanation_depth": 0.6, "examples": 0.5},
                "expert": {"detail_level": 0.3, "explanation_depth": 0.4, "examples": 0.2}
            }
        }
    
    def update_user_profile(self, user_id: str, interaction_data: Dict[str, Any]) -> UserProfile:
        """更新用户画像"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
        
        profile = self.user_profiles[user_id]
        
        # 更新交互统计
        profile.total_interactions += 1
        profile.last_interaction = time.time()
        
        # 更新偏好
        self._update_preferences(profile, interaction_data)
        
        # 更新交互模式
        self._update_interaction_patterns(profile, interaction_data)
        
        # 推断沟通风格
        profile.communication_style = self._infer_communication_style(profile)
        
        # 推断专业水平
        profile.expertise_level = self._infer_expertise_level(profile)
        
        # 更新个性特征
        self._update_personality_traits(profile, interaction_data)
        
        return profile
    
    def _update_preferences(self, profile: UserProfile, interaction_data: Dict[str, Any]):
        """更新用户偏好"""
        # 响应长度偏好
        if "response_length" in interaction_data:
            length = interaction_data["response_length"]
            if "response_length" not in profile.preferences:
                profile.preferences["response_length"] = length
            else:
                # 移动平均
                profile.preferences["response_length"] = (
                    profile.preferences["response_length"] * 0.8 + length * 0.2
                )
        
        # 详细程度偏好
        if "detail_preference" in interaction_data:
            detail = interaction_data["detail_preference"]
            profile.preferences["detail_level"] = detail
        
        # 响应时间偏好
        if "response_time" in interaction_data:
            response_time = interaction_data["response_time"]
            profile.response_time_preference = (
                profile.response_time_preference * 0.7 + response_time * 0.3
            )
        
        # 主题偏好
        if "topics" in interaction_data:
            for topic in interaction_data["topics"]:
                if "topic_interests" not in profile.preferences:
                    profile.preferences["topic_interests"] = defaultdict(int)
                profile.preferences["topic_interests"][topic] += 1
    
    def _update_interaction_patterns(self, profile: UserProfile, interaction_data: Dict[str, Any]):
        """更新交互模式"""
        # 问题类型模式
        if "question_type" in interaction_data:
            question_type = interaction_data["question_type"]
            profile.interaction_patterns[f"question_{question_type}"] = (
                profile.interaction_patterns.get(f"question_{question_type}", 0) + 1
            )
        
        # 会话长度模式
        if "session_length" in interaction_data:
            length = interaction_data["session_length"]
            if "avg_session_length" not in profile.interaction_patterns:
                profile.interaction_patterns["avg_session_length"] = length
            else:
                profile.interaction_patterns["avg_session_length"] = (
                    profile.interaction_patterns["avg_session_length"] * 0.8 + length * 0.2
                )
        
        # 反馈模式
        if "feedback_given" in interaction_data:
            if interaction_data["feedback_given"]:
                profile.interaction_patterns["feedback_frequency"] = (
                    profile.interaction_patterns.get("feedback_frequency", 0) + 1
                )
    
    def _infer_communication_style(self, profile: UserProfile) -> str:
        """推断沟通风格"""
        # 基于交互模式推断
        formal_indicators = 0
        casual_indicators = 0
        
        # 基于问题类型
        if profile.interaction_patterns.get("question_factual", 0) > profile.interaction_patterns.get("question_conversational", 0):
            formal_indicators += 1
        else:
            casual_indicators += 1
        
        # 基于会话长度
        avg_length = profile.interaction_patterns.get("avg_session_length", 5)
        if avg_length > 10:
            formal_indicators += 1
        else:
            casual_indicators += 1
        
        # 基于反馈频率
        feedback_freq = profile.interaction_patterns.get("feedback_frequency", 0)
        if feedback_freq > profile.total_interactions * 0.3:
            formal_indicators += 1
        else:
            casual_indicators += 1
        
        if formal_indicators > casual_indicators:
            return "formal"
        else:
            return "casual"
    
    def _infer_expertise_level(self, profile: UserProfile) -> str:
        """推断专业水平"""
        # 基于问题复杂度和主题深度
        if "topic_interests" in profile.preferences:
            topics = profile.preferences["topic_interests"]
            technical_topics = sum(count for topic, count in topics.items() 
                                 if any(keyword in topic.lower() for keyword in ["技术", "算法", "系统", "架构"]))
            
            total_topics = sum(topics.values())
            
            if total_topics > 0:
                technical_ratio = technical_topics / total_topics
                
                if technical_ratio > 0.6:
                    return "expert"
                elif technical_ratio > 0.3:
                    return "intermediate"
                else:
                    return "beginner"
        
        return "intermediate"
    
    def _update_personality_traits(self, profile: UserProfile, interaction_data: Dict[str, Any]):
        """更新个性特征"""
        # 外向性
        if "social_interaction" in interaction_data:
            social_score = interaction_data["social_interaction"]
            profile.personality_traits["extraversion"] = (
                profile.personality_traits.get("extraversion", 0.5) * 0.8 + social_score * 0.2
            )
        
        # 开放性
        if "exploration_behavior" in interaction_data:
            exploration_score = interaction_data["exploration_behavior"]
            profile.personality_traits["openness"] = (
                profile.personality_traits.get("openness", 0.5) * 0.8 + exploration_score * 0.2
            )
        
        # 耐心
        if "patience_level" in interaction_data:
            patience_score = interaction_data["patience_level"]
            profile.personality_traits["patience"] = (
                profile.personality_traits.get("patience", 0.5) * 0.8 + patience_score * 0.2
            )
    
    def generate_personalized_response_style(self, user_id: str) -> Dict[str, Any]:
        """生成个性化响应风格"""
        if user_id not in self.user_profiles:
            return self._get_default_response_style()
        
        profile = self.user_profiles[user_id]
        
        # 基于沟通风格
        comm_style = self.personality_models["communication_style"].get(
            profile.communication_style, 
            self.personality_models["communication_style"]["casual"]
        )
        
        # 基于专业水平
        expertise_style = self.personality_models["expertise_level"].get(
            profile.expertise_level,
            self.personality_models["expertise_level"]["intermediate"]
        )
        
        # 合并风格参数
        response_style = {
            "precision": comm_style["precision"],
            "warmth": comm_style["warmth"],
            "directness": comm_style["directness"],
            "detail_level": expertise_style["detail_level"],
            "explanation_depth": expertise_style["explanation_depth"],
            "examples": expertise_style["examples"],
            "response_length": profile.preferences.get("response_length", 100),
            "preferred_response_time": profile.response_time_preference
        }
        
        # 基于个性特征调整
        if "extraversion" in profile.personality_traits:
            response_style["warmth"] *= (0.5 + profile.personality_traits["extraversion"] * 0.5)
        
        if "openness" in profile.personality_traits:
            response_style["examples"] *= (0.5 + profile.personality_traits["openness"] * 0.5)
        
        return response_style
    
    def _get_default_response_style(self) -> Dict[str, Any]:
        """获取默认响应风格"""
        return {
            "precision": 0.7,
            "warmth": 0.6,
            "directness": 0.6,
            "detail_level": 0.6,
            "explanation_depth": 0.6,
            "examples": 0.5,
            "response_length": 100,
            "preferred_response_time": 2.0
        }

class AdaptationController:
    """适应控制器"""
    
    def __init__(self):
        self.adaptation_state = AdaptationState(current_strategy=AdaptationStrategy.BALANCED)
        self.strategy_performance = defaultdict(list)
        self.adaptation_history = deque(maxlen=100)
        self.environment_factors = {}
        
    def update_adaptation_state(self, performance_metrics: Dict[str, float], environment_context: Dict[str, Any]):
        """更新适应状态"""
        current_time = time.time()
        
        # 更新性能趋势
        overall_performance = sum(performance_metrics.values()) / len(performance_metrics)
        self.adaptation_state.performance_trend.append(overall_performance)
        
        # 限制趋势历史长度
        if len(self.adaptation_state.performance_trend) > 20:
            self.adaptation_state.performance_trend = self.adaptation_state.performance_trend[-20:]
        
        # 更新环境因素
        self.environment_factors.update(environment_context)
        
        # 评估当前策略效果
        strategy_effectiveness = self._evaluate_strategy_effectiveness()
        self.strategy_performance[self.adaptation_state.current_strategy.value].append(strategy_effectiveness)
        
        # 更新压力水平
        self._update_stress_level(performance_metrics)
        
        # 更新置信度
        self._update_confidence_level(performance_metrics)
        
        # 决定是否需要适应
        if self._should_adapt():
            new_strategy = self._select_adaptation_strategy()
            if new_strategy != self.adaptation_state.current_strategy:
                self._execute_adaptation(new_strategy)
        
        # 记录适应历史
        self.adaptation_history.append({
            "timestamp": current_time,
            "strategy": self.adaptation_state.current_strategy.value,
            "performance": overall_performance,
            "stress_level": self.adaptation_state.stress_level,
            "confidence": self.adaptation_state.confidence_level
        })
    
    def _evaluate_strategy_effectiveness(self) -> float:
        """评估策略有效性"""
        if len(self.adaptation_state.performance_trend) < 2:
            return 0.5
        
        # 计算性能改善
        recent_performance = sum(self.adaptation_state.performance_trend[-5:]) / min(5, len(self.adaptation_state.performance_trend))
        older_performance = sum(self.adaptation_state.performance_trend[-10:-5]) / min(5, len(self.adaptation_state.performance_trend) - 5)
        
        if older_performance == 0:
            return recent_performance
        
        improvement = (recent_performance - older_performance) / older_performance
        
        # 归一化到0-1范围
        effectiveness = 0.5 + improvement * 0.5
        return max(0.0, min(1.0, effectiveness))
    
    def _update_stress_level(self, performance_metrics: Dict[str, float]):
        """更新压力水平"""
        # 基于性能下降和变化率计算压力
        if len(self.adaptation_state.performance_trend) >= 3:
            recent_trend = self.adaptation_state.performance_trend[-3:]
            
            # 计算性能方差（不稳定性）
            mean_perf = sum(recent_trend) / len(recent_trend)
            variance = sum((x - mean_perf) ** 2 for x in recent_trend) / len(recent_trend)
            
            # 计算下降趋势
            declining_trend = 0
            for i in range(1, len(recent_trend)):
                if recent_trend[i] < recent_trend[i-1]:
                    declining_trend += 1
            
            decline_factor = declining_trend / (len(recent_trend) - 1)
            
            stress = variance * 2 + decline_factor * 0.5
            
            # 平滑更新
            self.adaptation_state.stress_level = (
                self.adaptation_state.stress_level * 0.7 + stress * 0.3
            )
        
        # 限制在0-1范围内
        self.adaptation_state.stress_level = max(0.0, min(1.0, self.adaptation_state.stress_level))
    
    def _update_confidence_level(self, performance_metrics: Dict[str, float]):
        """更新置信度水平"""
        current_performance = sum(performance_metrics.values()) / len(performance_metrics)
        
        # 基于当前性能和历史表现更新置信度
        if len(self.adaptation_state.performance_trend) > 0:
            historical_avg = sum(self.adaptation_state.performance_trend) / len(self.adaptation_state.performance_trend)
            
            if current_performance >= historical_avg:
                confidence_delta = 0.1
            else:
                confidence_delta = -0.1
            
            self.adaptation_state.confidence_level = max(0.1, min(1.0, 
                self.adaptation_state.confidence_level + confidence_delta * 0.1
            ))
    
    def _should_adapt(self) -> bool:
        """判断是否应该适应"""
        current_time = time.time()
        
        # 时间间隔检查
        if current_time - self.adaptation_state.last_adaptation < 300:  # 5分钟内不重复适应
            return False
        
        # 压力水平检查
        if self.adaptation_state.stress_level > 0.7:
            return True
        
        # 置信度检查
        if self.adaptation_state.confidence_level < 0.3:
            return True
        
        # 性能下降检查
        if len(self.adaptation_state.performance_trend) >= 5:
            recent_avg = sum(self.adaptation_state.performance_trend[-3:]) / 3
            older_avg = sum(self.adaptation_state.performance_trend[-6:-3]) / 3
            
            if recent_avg < older_avg * 0.8:  # 性能下降20%以上
                return True
        
        return False
    
    def _select_adaptation_strategy(self) -> AdaptationStrategy:
        """选择适应策略"""
        current_performance = self.adaptation_state.performance_trend[-1] if self.adaptation_state.performance_trend else 0.5
        
        # 基于当前状态选择策略
        if self.adaptation_state.stress_level > 0.8:
            # 高压力状态：选择保守策略
            return AdaptationStrategy.CONSERVATIVE
        
        elif self.adaptation_state.confidence_level < 0.3:
            # 低置信度：选择探索策略
            return AdaptationStrategy.EXPLORATORY
        
        elif current_performance < 0.3:
            # 低性能：选择激进策略
            return AdaptationStrategy.AGGRESSIVE
        
        elif current_performance > 0.8:
            # 高性能：选择平衡策略
            return AdaptationStrategy.BALANCED
        
        else:
            # 中等状态：选择反应策略
            return AdaptationStrategy.REACTIVE
    
    def _execute_adaptation(self, new_strategy: AdaptationStrategy):
        """执行适应"""
        old_strategy = self.adaptation_state.current_strategy
        self.adaptation_state.current_strategy = new_strategy
        self.adaptation_state.last_adaptation = time.time()
        
        # 根据策略调整参数
        if new_strategy == AdaptationStrategy.CONSERVATIVE:
            self.adaptation_state.exploration_rate = max(0.05, self.adaptation_state.exploration_rate * 0.5)
            self.adaptation_state.learning_rate = max(0.01, self.adaptation_state.learning_rate * 0.8)
            self.adaptation_state.adaptation_speed = max(0.1, self.adaptation_state.adaptation_speed * 0.7)
        
        elif new_strategy == AdaptationStrategy.AGGRESSIVE:
            self.adaptation_state.exploration_rate = min(0.5, self.adaptation_state.exploration_rate * 1.5)
            self.adaptation_state.learning_rate = min(0.3, self.adaptation_state.learning_rate * 1.5)
            self.adaptation_state.adaptation_speed = min(1.0, self.adaptation_state.adaptation_speed * 1.3)
        
        elif new_strategy == AdaptationStrategy.EXPLORATORY:
            self.adaptation_state.exploration_rate = min(0.8, self.adaptation_state.exploration_rate * 2.0)
            self.adaptation_state.learning_rate = min(0.25, self.adaptation_state.learning_rate * 1.2)
        
        elif new_strategy == AdaptationStrategy.BALANCED:
            self.adaptation_state.exploration_rate = 0.2
            self.adaptation_state.learning_rate = 0.1
            self.adaptation_state.adaptation_speed = 0.5
        
        elif new_strategy == AdaptationStrategy.REACTIVE:
            self.adaptation_state.adaptation_speed = min(1.0, self.adaptation_state.adaptation_speed * 1.5)
        
        logger.info(f"Adaptation executed: {old_strategy.value} -> {new_strategy.value}")

class BehaviorPredictor:
    """行为预测器"""
    
    def __init__(self):
        self.prediction_models = {}
        self.behavior_sequences = defaultdict(list)
        self.context_patterns = defaultdict(list)
        self.prediction_accuracy = defaultdict(list)
        
    def record_behavior_sequence(self, user_id: str, behavior_event: BehaviorEvent):
        """记录行为序列"""
        self.behavior_sequences[user_id].append({
            "behavior_type": behavior_event.behavior_type.value,
            "trigger": behavior_event.trigger,
            "action": behavior_event.action_taken,
            "timestamp": behavior_event.timestamp,
            "context": behavior_event.context
        })
        
        # 限制序列长度
        if len(self.behavior_sequences[user_id]) > 50:
            self.behavior_sequences[user_id] = self.behavior_sequences[user_id][-50:]
    
    def predict_next_behavior(self, user_id: str, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """预测下一个行为"""
        if user_id not in self.behavior_sequences or len(self.behavior_sequences[user_id]) < 3:
            return {"prediction": None, "confidence": 0.0}
        
        user_sequence = self.behavior_sequences[user_id]
        
        # 查找相似的行为模式
        similar_patterns = self._find_similar_patterns(user_sequence, current_context)
        
        if not similar_patterns:
            return {"prediction": None, "confidence": 0.0}
        
        # 预测最可能的下一个行为
        behavior_counts = defaultdict(int)
        total_patterns = len(similar_patterns)
        
        for pattern in similar_patterns:
            behavior_counts[pattern["next_behavior"]] += 1
        
        # 选择最频繁的行为
        most_likely_behavior = max(behavior_counts.items(), key=lambda x: x[1])
        confidence = most_likely_behavior[1] / total_patterns
        
        return {
            "prediction": most_likely_behavior[0],
            "confidence": confidence,
            "alternatives": [
                {"behavior": behavior, "probability": count / total_patterns}
                for behavior, count in sorted(behavior_counts.items(), key=lambda x: x[1], reverse=True)
            ]
        }
    
    def _find_similar_patterns(self, user_sequence: List[Dict[str, Any]], current_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """查找相似的行为模式"""
        patterns = []
        
        # 分析最近的行为序列
        recent_behaviors = user_sequence[-5:]  # 最近5个行为
        
        for i in range(len(user_sequence) - 5):
            sequence_slice = user_sequence[i:i+5]
            
            # 计算序列相似度
            similarity = self._calculate_sequence_similarity(recent_behaviors[:-1], sequence_slice[:-1])
            
            # 计算上下文相似度
            context_similarity = self._calculate_context_similarity(current_context, sequence_slice[-1]["context"])
            
            overall_similarity = similarity * 0.7 + context_similarity * 0.3
            
            if overall_similarity > 0.6:  # 相似度阈值
                patterns.append({
                    "similarity": overall_similarity,
                    "next_behavior": sequence_slice[-1]["behavior_type"],
                    "context": sequence_slice[-1]["context"]
                })
        
        return sorted(patterns, key=lambda x: x["similarity"], reverse=True)[:10]
    
    def _calculate_sequence_similarity(self, seq1: List[Dict[str, Any]], seq2: List[Dict[str, Any]]) -> float:
        """计算序列相似度"""
        if len(seq1) != len(seq2):
            return 0.0
        
        matches = 0
        for b1, b2 in zip(seq1, seq2):
            if b1["behavior_type"] == b2["behavior_type"]:
                matches += 1
        
        return matches / len(seq1)
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """计算上下文相似度"""
        if not context1 or not context2:
            return 0.0
        
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        similarity_sum = 0.0
        for key in common_keys:
            val1, val2 = context1[key], context2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # 数值相似度
                max_val = max(abs(val1), abs(val2), 1)
                similarity_sum += 1.0 - abs(val1 - val2) / max_val
            elif val1 == val2:
                # 完全匹配
                similarity_sum += 1.0
            else:
                # 不匹配
                similarity_sum += 0.0
        
        return similarity_sum / len(common_keys)

class BehaviorAdaptationSystem:
    """行为适应系统主类"""
    
    def __init__(self):
        self.behavior_learner = BehaviorLearner()
        self.personality_engine = PersonalityEngine()
        self.adaptation_controller = AdaptationController()
        self.behavior_predictor = BehaviorPredictor()
        
        self.event_history = deque(maxlen=1000)
        self.active_behaviors = {}
        self.system_performance = defaultdict(float)
        
    def record_behavior_event(self, event: BehaviorEvent, user_id: str = "default") -> Dict[str, Any]:
        """记录行为事件"""
        # 学习行为模式
        learning_result = self.behavior_learner.learn_from_event(event)
        
        # 更新用户画像
        interaction_data = {
            "behavior_type": event.behavior_type.value,
            "success": event.success,
            "feedback_score": event.feedback_score,
            "response_time": event.context.get("response_time", 2.0),
            "question_type": event.context.get("question_type", "general")
        }
        user_profile = self.personality_engine.update_user_profile(user_id, interaction_data)
        
        # 记录行为序列
        self.behavior_predictor.record_behavior_sequence(user_id, event)
        
        # 更新系统性能
        self._update_system_performance(event)
        
        # 记录事件历史
        self.event_history.append({
            "event": event,
            "user_id": user_id,
            "learning_result": learning_result,
            "timestamp": time.time()
        })
        
        return {
            "learning_result": learning_result,
            "user_profile_updated": True,
            "system_performance": dict(self.system_performance)
        }
    
    def get_adaptive_behavior_recommendation(
        self, 
        behavior_type: BehaviorType, 
        trigger: str, 
        context: Dict[str, Any], 
        user_id: str = "default"
    ) -> Dict[str, Any]:
        """获取自适应行为推荐"""
        
        # 获取最佳行为模式
        best_pattern = self.behavior_learner.get_best_pattern(behavior_type, trigger, context)
        
        # 获取个性化响应风格
        response_style = self.personality_engine.generate_personalized_response_style(user_id)
        
        # 预测下一个行为
        behavior_prediction = self.behavior_predictor.predict_next_behavior(user_id, context)
        
        # 基于适应状态调整推荐
        adaptation_adjustments = self._get_adaptation_adjustments()
        
        recommendation = {
            "recommended_pattern": {
                "pattern_id": best_pattern.pattern_id if best_pattern else None,
                "actions": best_pattern.actions if best_pattern else ["default_action"],
                "predicted_success": self.behavior_learner.predict_pattern_success(best_pattern, context) if best_pattern else 0.5,
                "confidence": best_pattern.confidence if best_pattern else 0.5
            },
            "personalized_style": response_style,
            "behavior_prediction": behavior_prediction,
            "adaptation_state": {
                "current_strategy": self.adaptation_controller.adaptation_state.current_strategy.value,
                "exploration_rate": self.adaptation_controller.adaptation_state.exploration_rate,
                "confidence_level": self.adaptation_controller.adaptation_state.confidence_level
            },
            "adjustments": adaptation_adjustments
        }
        
        return recommendation
    
    def _update_system_performance(self, event: BehaviorEvent):
        """更新系统性能"""
        behavior_type = event.behavior_type.value
        
        # 更新行为类型的成功率
        current_rate = self.system_performance[f"{behavior_type}_success_rate"]
        self.system_performance[f"{behavior_type}_success_rate"] = (
            current_rate * 0.9 + (1.0 if event.success else 0.0) * 0.1
        )
        
        # 更新反馈分数
        current_feedback = self.system_performance[f"{behavior_type}_feedback_score"]
        self.system_performance[f"{behavior_type}_feedback_score"] = (
            current_feedback * 0.9 + event.feedback_score * 0.1
        )
        
        # 更新整体性能
        overall_performance = (
            sum(self.system_performance[key] for key in self.system_performance if "success_rate" in key) /
            max(len([key for key in self.system_performance if "success_rate" in key]), 1)
        )
        self.system_performance["overall_performance"] = overall_performance
    
    def _get_adaptation_adjustments(self) -> Dict[str, Any]:
        """获取适应性调整"""
        state = self.adaptation_controller.adaptation_state
        
        adjustments = {
            "exploration_boost": state.exploration_rate > 0.3,
            "confidence_penalty": state.confidence_level < 0.5,
            "stress_mitigation": state.stress_level > 0.6,
            "learning_acceleration": state.learning_rate > 0.15,
            "adaptation_recommendations": []
        }
        
        # 生成具体的适应建议
        if state.stress_level > 0.7:
            adjustments["adaptation_recommendations"].append("减少复杂任务，专注基础功能")
        
        if state.confidence_level < 0.3:
            adjustments["adaptation_recommendations"].append("增加确认步骤，提供更多解释")
        
        if state.exploration_rate > 0.5:
            adjustments["adaptation_recommendations"].append("尝试新的解决方案和方法")
        
        return adjustments
    
    def trigger_adaptation_cycle(self):
        """触发适应周期"""
        # 更新适应状态
        self.adaptation_controller.update_adaptation_state(
            performance_metrics=dict(self.system_performance),
            environment_context={"timestamp": time.time()}
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "learned_patterns": len(self.behavior_learner.learned_patterns),
            "user_profiles": len(self.personality_engine.user_profiles),
            "adaptation_strategy": self.adaptation_controller.adaptation_state.current_strategy.value,
            "system_performance": dict(self.system_performance),
            "event_history_size": len(self.event_history),
            "adaptation_state": {
                "confidence_level": self.adaptation_controller.adaptation_state.confidence_level,
                "stress_level": self.adaptation_controller.adaptation_state.stress_level,
                "exploration_rate": self.adaptation_controller.adaptation_state.exploration_rate,
                "learning_rate": self.adaptation_controller.adaptation_state.learning_rate
            }
        }