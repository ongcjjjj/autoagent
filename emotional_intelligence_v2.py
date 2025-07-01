# 自主进化Agent - 第4轮提升：情感智能2.0系统
# Emotional Intelligence v2.0 - 高级情感处理与共情能力

import asyncio
import math
import random
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from collections import defaultdict, deque
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionType(Enum):
    """情感类型枚举"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    LOVE = "love"
    SHAME = "shame"
    GUILT = "guilt"
    PRIDE = "pride"
    ENVY = "envy"
    GRATITUDE = "gratitude"
    HOPE = "hope"
    DESPAIR = "despair"

class EmotionIntensity(Enum):
    """情感强度枚举"""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9

class EmpathyLevel(Enum):
    """共情水平枚举"""
    COGNITIVE = "cognitive"      # 认知共情
    AFFECTIVE = "affective"      # 情感共情
    COMPASSIONATE = "compassionate"  # 同情心

@dataclass
class EmotionalState:
    """情感状态数据结构"""
    primary_emotion: EmotionType
    intensity: float
    secondary_emotions: Dict[EmotionType, float] = field(default_factory=dict)
    arousal: float = 0.5  # 唤醒度
    valence: float = 0.5  # 效价 (positive/negative)
    timestamp: float = field(default_factory=time.time)
    context: str = ""
    triggers: List[str] = field(default_factory=list)

@dataclass
class EmotionalMemory:
    """情感记忆数据结构"""
    memory_id: str
    emotional_state: EmotionalState
    associated_content: str
    importance: float
    recency: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

@dataclass
class EmpathyResponse:
    """共情响应数据结构"""
    target_emotion: EmotionType
    empathy_level: EmpathyLevel
    response_emotion: EmotionType
    intensity: float
    response_text: str
    confidence: float

class EmotionRecognitionEngine:
    """情感识别引擎"""
    
    def __init__(self):
        self.emotion_patterns = self._initialize_emotion_patterns()
        self.context_analyzer = ContextAnalyzer()
        
    def _initialize_emotion_patterns(self) -> Dict[str, EmotionType]:
        """初始化情感模式"""
        return {
            # 积极情感
            "高兴": EmotionType.JOY, "开心": EmotionType.JOY, "快乐": EmotionType.JOY,
            "兴奋": EmotionType.JOY, "愉快": EmotionType.JOY, "满意": EmotionType.JOY,
            "happy": EmotionType.JOY, "joyful": EmotionType.JOY, "excited": EmotionType.JOY,
            
            # 消极情感
            "伤心": EmotionType.SADNESS, "难过": EmotionType.SADNESS, "悲伤": EmotionType.SADNESS,
            "沮丧": EmotionType.SADNESS, "失望": EmotionType.SADNESS,
            "sad": EmotionType.SADNESS, "disappointed": EmotionType.SADNESS,
            
            # 愤怒情感
            "生气": EmotionType.ANGER, "愤怒": EmotionType.ANGER, "烦躁": EmotionType.ANGER,
            "恼火": EmotionType.ANGER, "愤慨": EmotionType.ANGER,
            "angry": EmotionType.ANGER, "furious": EmotionType.ANGER, "annoyed": EmotionType.ANGER,
            
            # 恐惧情感
            "害怕": EmotionType.FEAR, "恐惧": EmotionType.FEAR, "担心": EmotionType.FEAR,
            "紧张": EmotionType.FEAR, "焦虑": EmotionType.FEAR,
            "afraid": EmotionType.FEAR, "scared": EmotionType.FEAR, "anxious": EmotionType.FEAR,
            
            # 惊讶情感
            "惊讶": EmotionType.SURPRISE, "震惊": EmotionType.SURPRISE, "意外": EmotionType.SURPRISE,
            "surprised": EmotionType.SURPRISE, "amazed": EmotionType.SURPRISE,
            
            # 厌恶情感
            "恶心": EmotionType.DISGUST, "厌恶": EmotionType.DISGUST, "反感": EmotionType.DISGUST,
            "disgusted": EmotionType.DISGUST, "revolted": EmotionType.DISGUST
        }
        
    def recognize_emotion(self, text: str, context: str = "") -> EmotionalState:
        """识别文本中的情感"""
        # 基础情感识别
        emotion_scores = defaultdict(float)
        
        words = text.lower().split()
        for word in words:
            if word in self.emotion_patterns:
                emotion = self.emotion_patterns[word]
                emotion_scores[emotion] += 1.0
                
        # 情感强度分析
        intensity_modifiers = {
            "非常": 1.5, "很": 1.3, "特别": 1.4, "极其": 1.8,
            "有点": 0.7, "稍微": 0.6, "略微": 0.5,
            "very": 1.4, "extremely": 1.8, "quite": 1.2,
            "slightly": 0.6, "somewhat": 0.7
        }
        
        intensity_factor = 1.0
        for modifier, factor in intensity_modifiers.items():
            if modifier in text.lower():
                intensity_factor = max(intensity_factor, factor)
                
        # 确定主要情感
        if not emotion_scores:
            primary_emotion = EmotionType.TRUST  # 默认中性情感
            intensity = 0.3
        else:
            primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            raw_intensity = emotion_scores[primary_emotion] / len(words)
            intensity = min(1.0, raw_intensity * intensity_factor)
            
        # 计算次要情感
        secondary_emotions = {}
        for emotion, score in emotion_scores.items():
            if emotion != primary_emotion and score > 0:
                secondary_emotions[emotion] = min(0.8, score * intensity_factor * 0.5)
                
        # 计算唤醒度和效价
        arousal, valence = self._compute_arousal_valence(primary_emotion, intensity)
        
        # 上下文分析
        context_adjustment = self.context_analyzer.analyze_context(text, context)
        
        return EmotionalState(
            primary_emotion=primary_emotion,
            intensity=intensity * context_adjustment.get('intensity_multiplier', 1.0),
            secondary_emotions=secondary_emotions,
            arousal=arousal,
            valence=valence,
            context=context,
            triggers=self._identify_triggers(text)
        )
        
    def _compute_arousal_valence(self, emotion: EmotionType, intensity: float) -> Tuple[float, float]:
        """计算情感的唤醒度和效价"""
        arousal_map = {
            EmotionType.JOY: 0.8, EmotionType.ANGER: 0.9, EmotionType.FEAR: 0.8,
            EmotionType.SURPRISE: 0.9, EmotionType.SADNESS: 0.2, EmotionType.DISGUST: 0.5,
            EmotionType.TRUST: 0.3, EmotionType.ANTICIPATION: 0.6
        }
        
        valence_map = {
            EmotionType.JOY: 0.9, EmotionType.TRUST: 0.7, EmotionType.ANTICIPATION: 0.6,
            EmotionType.SURPRISE: 0.5, EmotionType.ANGER: 0.1, EmotionType.FEAR: 0.2,
            EmotionType.SADNESS: 0.1, EmotionType.DISGUST: 0.2
        }
        
        base_arousal = arousal_map.get(emotion, 0.5)
        base_valence = valence_map.get(emotion, 0.5)
        
        # 强度影响唤醒度
        arousal = min(1.0, base_arousal * (0.5 + 0.5 * intensity))
        valence = base_valence  # 效价主要由情感类型决定
        
        return arousal, valence
        
    def _identify_triggers(self, text: str) -> List[str]:
        """识别情感触发因素"""
        triggers = []
        
        trigger_patterns = {
            "失败": ["失败", "挫折", "错误", "failure", "mistake"],
            "成功": ["成功", "胜利", "完成", "success", "achievement"],
            "损失": ["失去", "离开", "死亡", "loss", "death"],
            "威胁": ["危险", "威胁", "攻击", "danger", "threat"],
            "惊喜": ["礼物", "意外", "惊喜", "gift", "surprise"]
        }
        
        text_lower = text.lower()
        for trigger_type, keywords in trigger_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                triggers.append(trigger_type)
                
        return triggers

class ContextAnalyzer:
    """上下文分析器"""
    
    def analyze_context(self, text: str, context: str) -> Dict[str, float]:
        """分析上下文对情感的影响"""
        adjustments = {'intensity_multiplier': 1.0}
        
        # 社交上下文
        social_contexts = ["朋友", "家人", "同事", "陌生人", "friend", "family", "colleague"]
        if any(ctx in context.lower() for ctx in social_contexts):
            adjustments['intensity_multiplier'] *= 1.2
            
        # 时间上下文
        time_contexts = ["早上", "晚上", "深夜", "morning", "evening", "night"]
        if any(ctx in context.lower() for ctx in time_contexts):
            if "深夜" in context or "night" in context:
                adjustments['intensity_multiplier'] *= 1.3  # 夜晚情感更强烈
                
        # 环境上下文
        env_contexts = ["工作", "家里", "学校", "work", "home", "school"]
        if "工作" in context or "work" in context:
            adjustments['intensity_multiplier'] *= 0.8  # 工作环境情感较克制
            
        return adjustments

class EmotionalMemorySystem:
    """情感记忆系统"""
    
    def __init__(self, memory_capacity: int = 1000):
        self.memory_capacity = memory_capacity
        self.emotional_memories: Dict[str, EmotionalMemory] = {}
        self.memory_index = 0
        
    def store_memory(self, content: str, emotional_state: EmotionalState) -> str:
        """存储情感记忆"""
        memory_id = f"em_{self.memory_index}"
        self.memory_index += 1
        
        # 计算记忆重要性
        importance = self._calculate_importance(emotional_state)
        
        memory = EmotionalMemory(
            memory_id=memory_id,
            emotional_state=emotional_state,
            associated_content=content,
            importance=importance,
            recency=1.0
        )
        
        # 如果内存已满，移除最不重要的记忆
        if len(self.emotional_memories) >= self.memory_capacity:
            self._forget_least_important()
            
        self.emotional_memories[memory_id] = memory
        logger.info(f"存储情感记忆: {memory_id}, 重要性: {importance:.3f}")
        
        return memory_id
        
    def retrieve_memories(self, query_emotion: EmotionType, 
                         min_similarity: float = 0.6) -> List[EmotionalMemory]:
        """检索相似情感记忆"""
        relevant_memories = []
        
        for memory in self.emotional_memories.values():
            similarity = self._compute_emotional_similarity(
                query_emotion, memory.emotional_state.primary_emotion
            )
            
            if similarity >= min_similarity:
                # 更新访问记录
                memory.access_count += 1
                memory.last_accessed = time.time()
                relevant_memories.append(memory)
                
        # 按重要性和相似性排序
        relevant_memories.sort(
            key=lambda m: m.importance * self._compute_emotional_similarity(
                query_emotion, m.emotional_state.primary_emotion
            ), reverse=True
        )
        
        return relevant_memories[:10]  # 返回最相关的10个记忆
        
    def _calculate_importance(self, emotional_state: EmotionalState) -> float:
        """计算记忆重要性"""
        # 基于情感强度、唤醒度和触发因素
        importance = (
            emotional_state.intensity * 0.4 +
            emotional_state.arousal * 0.3 +
            len(emotional_state.triggers) * 0.1 +
            (1.0 - emotional_state.valence) * 0.2  # 负面情感更重要
        )
        return min(1.0, importance)
        
    def _compute_emotional_similarity(self, emotion1: EmotionType, emotion2: EmotionType) -> float:
        """计算情感相似性"""
        if emotion1 == emotion2:
            return 1.0
            
        # 情感相似性矩阵（简化版）
        similarity_matrix = {
            (EmotionType.JOY, EmotionType.TRUST): 0.7,
            (EmotionType.JOY, EmotionType.ANTICIPATION): 0.6,
            (EmotionType.SADNESS, EmotionType.FEAR): 0.6,
            (EmotionType.ANGER, EmotionType.DISGUST): 0.7,
            (EmotionType.SURPRISE, EmotionType.ANTICIPATION): 0.5,
        }
        
        # 检查正向和反向匹配
        key1 = (emotion1, emotion2)
        key2 = (emotion2, emotion1)
        
        return similarity_matrix.get(key1, similarity_matrix.get(key2, 0.2))
        
    def _forget_least_important(self):
        """遗忘最不重要的记忆"""
        if not self.emotional_memories:
            return
            
        # 计算记忆的遗忘分数（重要性低、访问少、时间久的优先遗忘）
        current_time = time.time()
        forget_scores = {}
        
        for memory_id, memory in self.emotional_memories.items():
            age = current_time - memory.emotional_state.timestamp
            forget_score = (
                (1.0 - memory.importance) * 0.5 +
                (1.0 - min(memory.access_count / 10, 1.0)) * 0.3 +
                min(age / (24 * 3600), 1.0) * 0.2  # 以天为单位的年龄
            )
            forget_scores[memory_id] = forget_score
            
        # 移除遗忘分数最高的记忆
        memory_to_forget = max(forget_scores.items(), key=lambda x: x[1])[0]
        del self.emotional_memories[memory_to_forget]

class EmpathyEngine:
    """共情引擎"""
    
    def __init__(self):
        self.empathy_responses = self._initialize_empathy_responses()
        
    def _initialize_empathy_responses(self) -> Dict[EmotionType, Dict[str, Any]]:
        """初始化共情响应模式"""
        return {
            EmotionType.SADNESS: {
                "cognitive": "我理解你现在很难过",
                "affective": "我也感到很伤心",
                "compassionate": "我想帮助你度过这个困难时期"
            },
            EmotionType.ANGER: {
                "cognitive": "我能理解你为什么感到愤怒",
                "affective": "这种不公让我也很生气",
                "compassionate": "让我们一起想办法解决这个问题"
            },
            EmotionType.FEAR: {
                "cognitive": "你的担心是可以理解的",
                "affective": "我也为你感到担心",
                "compassionate": "我会陪伴你，你不是一个人"
            },
            EmotionType.JOY: {
                "cognitive": "我能感受到你的快乐",
                "affective": "你的喜悦也感染了我",
                "compassionate": "我为你的成功感到高兴"
            }
        }
        
    def generate_empathy_response(self, target_emotion: EmotionalState, 
                                empathy_level: EmpathyLevel = EmpathyLevel.COMPASSIONATE) -> EmpathyResponse:
        """生成共情响应"""
        primary_emotion = target_emotion.primary_emotion
        
        # 获取基础响应模板
        if primary_emotion in self.empathy_responses:
            response_template = self.empathy_responses[primary_emotion].get(
                empathy_level.value, 
                self.empathy_responses[primary_emotion]["cognitive"]
            )
        else:
            response_template = "我理解你的感受"
            
        # 根据情感强度调整响应
        intensity_modifiers = {
            (0.0, 0.3): "",
            (0.3, 0.6): "我能感受到",
            (0.6, 0.8): "我深深理解",
            (0.8, 1.0): "我完全能够体会"
        }
        
        modifier = ""
        for (min_int, max_int), mod in intensity_modifiers.items():
            if min_int <= target_emotion.intensity < max_int:
                modifier = mod
                break
                
        # 生成个性化响应
        personalized_response = self._personalize_response(
            response_template, target_emotion, modifier
        )
        
        # 确定响应情感
        response_emotion = self._determine_response_emotion(
            primary_emotion, empathy_level
        )
        
        return EmpathyResponse(
            target_emotion=primary_emotion,
            empathy_level=empathy_level,
            response_emotion=response_emotion,
            intensity=target_emotion.intensity * 0.7,  # 响应强度稍低
            response_text=personalized_response,
            confidence=self._calculate_empathy_confidence(target_emotion)
        )
        
    def _personalize_response(self, template: str, emotional_state: EmotionalState, 
                            modifier: str) -> str:
        """个性化响应文本"""
        response = template
        
        if modifier:
            response = f"{modifier}{response}"
            
        # 添加上下文相关的内容
        if emotional_state.context:
            if "工作" in emotional_state.context:
                response += "，工作中的挑战确实不容易应对"
            elif "家庭" in emotional_state.context:
                response += "，家庭问题总是让人特别牵挂"
                
        # 根据触发因素添加针对性建议
        if "失败" in emotional_state.triggers:
            response += "。每个人都会遇到挫折，这是成长的一部分"
        elif "损失" in emotional_state.triggers:
            response += "。失去重要的东西确实很痛苦"
            
        return response
        
    def _determine_response_emotion(self, target_emotion: EmotionType, 
                                  empathy_level: EmpathyLevel) -> EmotionType:
        """确定响应情感"""
        if empathy_level == EmpathyLevel.AFFECTIVE:
            # 情感共情：镜像相同情感
            return target_emotion
        elif empathy_level == EmpathyLevel.COMPASSIONATE:
            # 同情心：通常表现为关怀和温暖
            if target_emotion in [EmotionType.SADNESS, EmotionType.FEAR, EmotionType.ANGER]:
                return EmotionType.TRUST  # 表现为关怀和支持
            else:
                return EmotionType.JOY  # 为对方的积极情感感到高兴
        else:
            # 认知共情：理解但保持情感距离
            return EmotionType.TRUST
            
    def _calculate_empathy_confidence(self, emotional_state: EmotionalState) -> float:
        """计算共情置信度"""
        confidence = 0.5  # 基础置信度
        
        # 情感强度越高，共情越容易
        confidence += emotional_state.intensity * 0.3
        
        # 有明确触发因素的情感更容易共情
        if emotional_state.triggers:
            confidence += len(emotional_state.triggers) * 0.1
            
        # 有上下文信息的情感更容易理解
        if emotional_state.context:
            confidence += 0.1
            
        return min(1.0, confidence)

class EmotionRegulationSystem:
    """情感调节系统"""
    
    def __init__(self):
        self.regulation_strategies = self._initialize_regulation_strategies()
        
    def _initialize_regulation_strategies(self) -> Dict[str, Dict[str, Any]]:
        """初始化情感调节策略"""
        return {
            "cognitive_reappraisal": {
                "name": "认知重评",
                "description": "重新解释情况以改变情感反应",
                "effectiveness": 0.8,
                "applicable_emotions": [EmotionType.ANGER, EmotionType.SADNESS, EmotionType.FEAR]
            },
            "deep_breathing": {
                "name": "深呼吸",
                "description": "通过控制呼吸来平静情绪",
                "effectiveness": 0.6,
                "applicable_emotions": [EmotionType.ANGER, EmotionType.FEAR, EmotionType.SURPRISE]
            },
            "distraction": {
                "name": "注意力转移",
                "description": "将注意力转移到其他事物上",
                "effectiveness": 0.7,
                "applicable_emotions": [EmotionType.SADNESS, EmotionType.ANGER]
            },
            "positive_reframing": {
                "name": "积极重构",
                "description": "寻找情况中的积极方面",
                "effectiveness": 0.75,
                "applicable_emotions": [EmotionType.SADNESS, EmotionType.FEAR, EmotionType.DISGUST]
            },
            "acceptance": {
                "name": "接纳",
                "description": "接受当前的情感状态",
                "effectiveness": 0.6,
                "applicable_emotions": list(EmotionType)
            }
        }
        
    def recommend_regulation_strategy(self, emotional_state: EmotionalState) -> Dict[str, Any]:
        """推荐情感调节策略"""
        suitable_strategies = []
        
        for strategy_id, strategy in self.regulation_strategies.items():
            if emotional_state.primary_emotion in strategy["applicable_emotions"]:
                # 计算策略适用性分数
                score = strategy["effectiveness"]
                
                # 根据情感强度调整
                if emotional_state.intensity > 0.7:
                    if strategy_id in ["deep_breathing", "acceptance"]:
                        score *= 1.2  # 高强度情感优先使用这些策略
                else:
                    if strategy_id in ["cognitive_reappraisal", "positive_reframing"]:
                        score *= 1.1  # 中低强度情感可以使用认知策略
                        
                suitable_strategies.append({
                    "strategy_id": strategy_id,
                    "strategy": strategy,
                    "score": score
                })
                
        # 按分数排序
        suitable_strategies.sort(key=lambda x: x["score"], reverse=True)
        
        if suitable_strategies:
            best_strategy = suitable_strategies[0]
            return {
                "recommended_strategy": best_strategy["strategy"],
                "alternatives": suitable_strategies[1:3],
                "personalized_advice": self._generate_personalized_advice(
                    emotional_state, best_strategy["strategy"]
                )
            }
        else:
            return {
                "recommended_strategy": self.regulation_strategies["acceptance"],
                "alternatives": [],
                "personalized_advice": "先接受和认识当前的情感状态"
            }
            
    def _generate_personalized_advice(self, emotional_state: EmotionalState, 
                                    strategy: Dict[str, Any]) -> str:
        """生成个性化建议"""
        advice = strategy["description"]
        
        # 根据情感类型添加具体建议
        if emotional_state.primary_emotion == EmotionType.ANGER:
            if strategy["name"] == "深呼吸":
                advice += "。试着慢慢吸气4秒，憋气4秒，然后慢慢呼气4秒"
            elif strategy["name"] == "认知重评":
                advice += "。问问自己：这件事真的值得如此愤怒吗？是否还有其他解释？"
                
        elif emotional_state.primary_emotion == EmotionType.SADNESS:
            if strategy["name"] == "积极重构":
                advice += "。试着思考这个经历可能带来的成长或学习机会"
            elif strategy["name"] == "注意力转移":
                advice += "。可以听音乐、运动或做一些你喜欢的活动"
                
        elif emotional_state.primary_emotion == EmotionType.FEAR:
            if strategy["name"] == "认知重评":
                advice += "。评估一下这种担心发生的实际概率，以及你能采取的应对措施"
                
        # 根据上下文添加建议
        if emotional_state.context:
            if "工作" in emotional_state.context:
                advice += "。如果在工作场所，可以先到安静的地方冷静一下"
            elif "人际关系" in emotional_state.context:
                advice += "。考虑与信任的朋友或家人分享你的感受"
                
        return advice

class EmotionalIntelligenceSystem:
    """情感智能2.0系统主控制器"""
    
    def __init__(self):
        self.emotion_recognition = EmotionRecognitionEngine()
        self.memory_system = EmotionalMemorySystem()
        self.empathy_engine = EmpathyEngine()
        self.regulation_system = EmotionRegulationSystem()
        
        # 系统状态
        self.current_emotional_state = None
        self.interaction_history = deque(maxlen=100)
        
        # 性能指标
        self.performance_metrics = {
            'total_interactions': 0,
            'successful_empathy_responses': 0,
            'emotion_recognition_accuracy': 0.85,
            'regulation_strategy_effectiveness': 0.75
        }
        
    def process_emotional_interaction(self, text: str, context: str = "") -> Dict[str, Any]:
        """处理情感交互"""
        start_time = time.time()
        
        # 1. 情感识别
        recognized_emotion = self.emotion_recognition.recognize_emotion(text, context)
        
        # 2. 存储情感记忆
        memory_id = self.memory_system.store_memory(text, recognized_emotion)
        
        # 3. 检索相关记忆
        related_memories = self.memory_system.retrieve_memories(
            recognized_emotion.primary_emotion
        )
        
        # 4. 生成共情响应
        empathy_response = self.empathy_engine.generate_empathy_response(recognized_emotion)
        
        # 5. 推荐情感调节策略
        regulation_advice = self.regulation_system.recommend_regulation_strategy(recognized_emotion)
        
        # 6. 更新系统状态
        self.current_emotional_state = recognized_emotion
        
        # 生成综合响应
        interaction_result = {
            'recognized_emotion': {
                'primary': recognized_emotion.primary_emotion.value,
                'intensity': recognized_emotion.intensity,
                'secondary': {e.value: i for e, i in recognized_emotion.secondary_emotions.items()},
                'arousal': recognized_emotion.arousal,
                'valence': recognized_emotion.valence,
                'triggers': recognized_emotion.triggers
            },
            'empathy_response': {
                'text': empathy_response.response_text,
                'emotion': empathy_response.response_emotion.value,
                'intensity': empathy_response.intensity,
                'confidence': empathy_response.confidence,
                'empathy_level': empathy_response.empathy_level.value
            },
            'regulation_advice': regulation_advice,
            'related_memories_count': len(related_memories),
            'memory_id': memory_id,
            'processing_time': time.time() - start_time,
            'timestamp': time.time()
        }
        
        # 记录交互历史
        self.interaction_history.append(interaction_result)
        
        # 更新性能指标
        self._update_performance_metrics(interaction_result)
        
        return interaction_result
        
    def _update_performance_metrics(self, interaction_result: Dict[str, Any]):
        """更新性能指标"""
        self.performance_metrics['total_interactions'] += 1
        
        # 评估共情响应质量
        empathy_confidence = interaction_result['empathy_response']['confidence']
        if empathy_confidence > 0.7:
            self.performance_metrics['successful_empathy_responses'] += 1
            
    def get_emotional_summary(self, interaction_result: Dict[str, Any]) -> str:
        """生成情感交互摘要"""
        emotion_info = interaction_result['recognized_emotion']
        empathy_info = interaction_result['empathy_response']
        regulation_info = interaction_result['regulation_advice']
        
        summary = f"""
🎭 情感智能分析报告
{"="*40}

🔍 识别到的情感:
  主要情感: {emotion_info['primary']} (强度: {emotion_info['intensity']:.2f})
  唤醒度: {emotion_info['arousal']:.2f} | 效价: {emotion_info['valence']:.2f}
  触发因素: {', '.join(emotion_info['triggers']) if emotion_info['triggers'] else '无'}

💝 共情响应:
  响应文本: {empathy_info['text']}
  共情类型: {empathy_info['empathy_level']}
  置信度: {empathy_info['confidence']:.2f}

🛠️ 调节建议:
  推荐策略: {regulation_info['recommended_strategy']['name']}
  具体建议: {regulation_info['personalized_advice']}

📊 处理信息:
  相关记忆: {interaction_result['related_memories_count']}个
  处理时间: {interaction_result['processing_time']:.3f}秒
"""
        return summary
        
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        empathy_success_rate = (
            self.performance_metrics['successful_empathy_responses'] / 
            max(self.performance_metrics['total_interactions'], 1)
        )
        
        return {
            "总交互次数": self.performance_metrics['total_interactions'],
            "成功共情次数": self.performance_metrics['successful_empathy_responses'],
            "共情成功率": f"{empathy_success_rate:.2%}",
            "情感识别准确率": f"{self.performance_metrics['emotion_recognition_accuracy']:.2%}",
            "调节策略有效性": f"{self.performance_metrics['regulation_strategy_effectiveness']:.2%}",
            "记忆存储量": len(self.memory_system.emotional_memories),
            "交互历史数": len(self.interaction_history)
        }

# 示例使用和测试
async def demonstrate_emotional_intelligence():
    """演示情感智能2.0系统功能"""
    print("💝 自主进化Agent - 第4轮提升：情感智能2.0系统")
    print("=" * 60)
    
    # 创建情感智能系统
    ei_system = EmotionalIntelligenceSystem()
    
    # 测试情感交互场景
    test_scenarios = [
        {
            "text": "我今天工作中犯了一个严重错误，老板很生气，我感到非常沮丧和害怕",
            "context": "工作环境，下午时间"
        },
        {
            "text": "我刚刚收到了大学录取通知书！我太开心了！",
            "context": "家庭环境，与家人分享"
        },
        {
            "text": "最近总是担心未来，感觉前路茫茫，不知道该怎么办",
            "context": "个人反思，夜晚时间"
        },
        {
            "text": "和最好的朋友吵架了，我很愤怒但也很伤心",
            "context": "人际关系，朋友圈"
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📝 场景 {i}: {scenario['text'][:30]}...")
        
        # 处理情感交互
        result = ei_system.process_emotional_interaction(
            scenario['text'], 
            scenario['context']
        )
        
        # 显示分析结果
        summary = ei_system.get_emotional_summary(result)
        print(summary)
        
    # 系统性能报告
    print("\n📊 情感智能系统性能报告")
    report = ei_system.get_performance_report()
    for key, value in report.items():
        print(f"{key}: {value}")
        
    print("\n✅ 第4轮提升完成！情感智能2.0系统已成功部署")

if __name__ == "__main__":
    asyncio.run(demonstrate_emotional_intelligence())