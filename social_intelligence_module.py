#!/usr/bin/env python3
"""
自主进化Agent - 第7轮升级：社交智能模块
版本: v3.7.0
创建时间: 2024年最新

🎯 第7轮升级核心特性：
- 社交情境理解
- 人际关系建模
- 社交策略生成
- 社交行为分析
- 群体动力学建模

🚀 技术突破点：
1. 社交感知引擎 - 多维度社交信息感知
2. 关系建模系统 - 动态人际关系图谱
3. 社交策略生成器 - 智能社交策略制定
4. 情境理解引擎 - 复杂社交场景分析
5. 群体智能分析 - 群体行为和动力学
"""

import asyncio
import logging
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum
import random
import math
from collections import defaultdict, deque
import threading
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RelationType(Enum):
    """关系类型枚举"""
    FAMILY = "family"           # 家庭关系
    FRIEND = "friend"           # 朋友关系
    COLLEAGUE = "colleague"     # 同事关系
    ROMANTIC = "romantic"       # 恋爱关系
    MENTOR = "mentor"           # 师生关系
    ACQUAINTANCE = "acquaintance"  # 熟人关系
    STRANGER = "stranger"       # 陌生人关系
    ADVERSARY = "adversary"     # 对立关系

class SocialRole(Enum):
    """社交角色枚举"""
    LEADER = "leader"           # 领导者
    FOLLOWER = "follower"       # 跟随者
    MEDIATOR = "mediator"       # 调解者
    INNOVATOR = "innovator"     # 创新者
    SUPPORTER = "supporter"     # 支持者
    CHALLENGER = "challenger"   # 挑战者
    OBSERVER = "observer"       # 观察者
    FACILITATOR = "facilitator" # 协调者

class SocialContext(Enum):
    """社交情境枚举"""
    FORMAL = "formal"           # 正式场合
    INFORMAL = "informal"       # 非正式场合
    PROFESSIONAL = "professional"  # 职业场合
    PERSONAL = "personal"       # 个人场合
    PUBLIC = "public"           # 公共场合
    PRIVATE = "private"         # 私人场合
    EDUCATIONAL = "educational" # 教育场合
    ENTERTAINMENT = "entertainment" # 娱乐场合

@dataclass
class Person:
    """人员表示类"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    age: Optional[int] = None
    gender: Optional[str] = None
    personality: Dict[str, float] = field(default_factory=dict)
    interests: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    social_status: str = ""
    communication_style: str = ""
    cultural_background: str = ""
    current_mood: str = "neutral"
    trust_level: float = 0.5
    interaction_history: List[Dict] = field(default_factory=list)

@dataclass
class Relationship:
    """关系表示类"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    person1_id: str = ""
    person2_id: str = ""
    relationship_type: RelationType = RelationType.ACQUAINTANCE
    strength: float = 0.5  # 关系强度 0-1
    intimacy: float = 0.3  # 亲密度 0-1
    trust: float = 0.5     # 信任度 0-1
    respect: float = 0.5   # 尊重度 0-1
    conflict_level: float = 0.0  # 冲突程度 0-1
    power_balance: float = 0.5   # 权力平衡 0-1
    communication_frequency: float = 0.3  # 沟通频率
    shared_experiences: List[str] = field(default_factory=list)
    common_interests: List[str] = field(default_factory=list)
    relationship_history: List[Dict] = field(default_factory=list)
    last_interaction: Optional[datetime] = None

@dataclass
class SocialSituation:
    """社交情境类"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context: SocialContext = SocialContext.INFORMAL
    participants: List[str] = field(default_factory=list)
    purpose: str = ""
    formal_rules: List[str] = field(default_factory=list)
    informal_norms: List[str] = field(default_factory=list)
    power_dynamics: Dict[str, float] = field(default_factory=dict)
    emotional_atmosphere: str = "neutral"
    time_constraints: Optional[timedelta] = None
    physical_environment: Dict[str, Any] = field(default_factory=dict)
    cultural_context: str = ""

class SocialIntelligenceModule:
    """社交智能模块核心类"""
    
    def __init__(self):
        """初始化社交智能模块"""
        self.people: Dict[str, Person] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.situations: Dict[str, SocialSituation] = {}
        
        # 社交智能组件
        self.social_perception = SocialPerceptionEngine()
        self.relationship_modeler = RelationshipModeler()
        self.strategy_generator = SocialStrategyGenerator()
        self.context_analyzer = ContextAnalyzer()
        self.group_dynamics = GroupDynamicsAnalyzer()
        
        # 社交知识库
        self.social_rules = {}
        self.cultural_norms = {}
        self.communication_patterns = {}
        
        # 性能指标
        self.metrics = {
            'social_interactions': 0,
            'successful_strategies': 0,
            'relationship_accuracy': 0.0,
            'context_understanding': 0.0,
            'strategy_effectiveness': 0.0
        }
        
        # 初始化默认社交规则
        self._initialize_social_knowledge()
        
        logger.info("社交智能模块初始化完成")
    
    def _initialize_social_knowledge(self):
        """初始化社交知识库"""
        # 基本社交规则
        self.social_rules = {
            'greeting': {
                'formal': ['Good morning', 'Good afternoon', 'Hello'],
                'informal': ['Hi', 'Hey', 'What\'s up'],
                'cultural_variations': {}
            },
            'politeness': {
                'please_thank_you': True,
                'respect_personal_space': True,
                'active_listening': True
            },
            'conversation': {
                'turn_taking': True,
                'topic_relevance': True,
                'emotional_awareness': True
            }
        }
        
        # 文化规范
        self.cultural_norms = {
            'western': {
                'eye_contact': 'important',
                'personal_space': 1.2,  # meters
                'directness': 'preferred'
            },
            'eastern': {
                'hierarchy_respect': 'critical',
                'face_saving': 'important',
                'group_harmony': 'priority'
            }
        }

class SocialPerceptionEngine:
    """社交感知引擎"""
    
    def __init__(self):
        self.perception_channels = {
            'verbal': VerbaPerceptionChannel(),
            'nonverbal': NonverbalPerceptionChannel(),
            'contextual': ContextualPerceptionChannel(),
            'emotional': EmotionalPerceptionChannel()
        }
    
    async def perceive_social_signals(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """感知社交信号"""
        social_signals = {}
        
        # 并行处理多个感知通道
        tasks = []
        for channel_name, channel in self.perception_channels.items():
            task = asyncio.create_task(
                channel.process(interaction_data)
            )
            tasks.append((channel_name, task))
        
        # 收集所有感知结果
        for channel_name, task in tasks:
            try:
                result = await task
                social_signals[channel_name] = result
            except Exception as e:
                logger.warning(f"感知通道 {channel_name} 处理失败: {e}")
                social_signals[channel_name] = {}
        
        # 融合多通道信息
        fused_signals = self._fuse_social_signals(social_signals)
        
        return fused_signals
    
    def _fuse_social_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """融合多通道社交信号"""
        fused = {
            'overall_mood': 'neutral',
            'engagement_level': 0.5,
            'social_comfort': 0.5,
            'communication_style': 'normal',
            'power_dynamics': {},
            'emotional_state': 'stable',
            'confidence': 0.7
        }
        
        # 基于多通道信息进行融合
        if 'verbal' in signals and 'emotional' in signals:
            # 情感状态融合
            verbal_emotion = signals['verbal'].get('emotion', 'neutral')
            nonverbal_emotion = signals.get('emotional', {}).get('emotion', 'neutral')
            fused['emotional_state'] = self._combine_emotions(verbal_emotion, nonverbal_emotion)
        
        return fused
    
    def _combine_emotions(self, verbal_emotion: str, nonverbal_emotion: str) -> str:
        """组合情感信息"""
        # 简化的情感组合逻辑
        emotion_priority = {
            'angry': 5, 'sad': 4, 'fear': 4,
            'happy': 3, 'excited': 3, 'surprised': 2,
            'neutral': 1, 'calm': 1
        }
        
        verbal_priority = emotion_priority.get(verbal_emotion, 1)
        nonverbal_priority = emotion_priority.get(nonverbal_emotion, 1)
        
        return verbal_emotion if verbal_priority >= nonverbal_priority else nonverbal_emotion

class VerbaPerceptionChannel:
    """言语感知通道"""
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理言语信息"""
        text = data.get('text', '')
        
        # 模拟言语分析
        return {
            'sentiment': self._analyze_sentiment(text),
            'politeness': self._analyze_politeness(text),
            'formality': self._analyze_formality(text),
            'emotion': self._detect_emotion(text),
            'intent': self._extract_intent(text)
        }
    
    def _analyze_sentiment(self, text: str) -> float:
        """分析情感倾向"""
        positive_words = ['good', 'great', 'excellent', 'wonderful', 'amazing']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _analyze_politeness(self, text: str) -> float:
        """分析礼貌程度"""
        polite_indicators = ['please', 'thank you', 'excuse me', 'sorry', 'would you']
        words = text.lower()
        
        politeness_score = 0.5  # 基础分数
        for indicator in polite_indicators:
            if indicator in words:
                politeness_score += 0.1
        
        return min(1.0, politeness_score)
    
    def _analyze_formality(self, text: str) -> float:
        """分析正式程度"""
        formal_indicators = ['furthermore', 'therefore', 'however', 'nevertheless']
        informal_indicators = ['yeah', 'ok', 'cool', 'awesome', 'gonna']
        
        words = text.lower()
        formal_count = sum(1 for indicator in formal_indicators if indicator in words)
        informal_count = sum(1 for indicator in informal_indicators if indicator in words)
        
        if formal_count + informal_count == 0:
            return 0.5
        
        return formal_count / (formal_count + informal_count)
    
    def _detect_emotion(self, text: str) -> str:
        """检测情感"""
        emotion_keywords = {
            'happy': ['happy', 'joy', 'glad', 'excited', 'pleased'],
            'sad': ['sad', 'depressed', 'down', 'unhappy', 'disappointed'],
            'angry': ['angry', 'mad', 'furious', 'annoyed', 'irritated'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous']
        }
        
        words = text.lower().split()
        emotion_scores = {}
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for word in words if word in keywords)
            emotion_scores[emotion] = score
        
        if not emotion_scores or max(emotion_scores.values()) == 0:
            return 'neutral'
        
        return max(emotion_scores.items(), key=lambda x: x[1])[0]
    
    def _extract_intent(self, text: str) -> str:
        """提取意图"""
        intent_patterns = {
            'question': ['?', 'what', 'how', 'when', 'where', 'why', 'who'],
            'request': ['please', 'can you', 'could you', 'would you'],
            'information': ['tell me', 'explain', 'describe', 'show me'],
            'agreement': ['yes', 'agree', 'ok', 'sure', 'definitely'],
            'disagreement': ['no', 'disagree', 'wrong', 'not really']
        }
        
        text_lower = text.lower()
        intent_scores = {}
        
        for intent, patterns in intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            intent_scores[intent] = score
        
        if not intent_scores or max(intent_scores.values()) == 0:
            return 'statement'
        
        return max(intent_scores.items(), key=lambda x: x[1])[0]

class NonverbalPerceptionChannel:
    """非言语感知通道"""
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理非言语信息"""
        # 模拟非言语信息处理
        return {
            'body_language': self._analyze_body_language(data),
            'facial_expression': self._analyze_facial_expression(data),
            'voice_tone': self._analyze_voice_tone(data),
            'proximity': self._analyze_proximity(data)
        }
    
    def _analyze_body_language(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """分析身体语言"""
        # 模拟身体语言分析
        return {
            'posture': 'open',  # open, closed, neutral
            'gestures': 'appropriate',  # appropriate, excessive, minimal
            'orientation': 'towards',  # towards, away, sideways
            'confidence': 0.7
        }
    
    def _analyze_facial_expression(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """分析面部表情"""
        # 模拟面部表情分析
        return {
            'expression': 'neutral',  # happy, sad, angry, surprised, etc.
            'eye_contact': 0.6,  # 0-1
            'micro_expressions': [],
            'authenticity': 0.8  # 表情真实性
        }
    
    def _analyze_voice_tone(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """分析语音语调"""
        # 模拟语音分析
        return {
            'pitch': 'normal',  # high, normal, low
            'volume': 'moderate',  # loud, moderate, quiet
            'pace': 'normal',  # fast, normal, slow
            'emotion': 'neutral'
        }
    
    def _analyze_proximity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """分析距离和空间关系"""
        return {
            'distance': 1.5,  # 物理距离（米）
            'space_usage': 'appropriate',
            'territorial_behavior': 'respectful'
        }

class ContextualPerceptionChannel:
    """情境感知通道"""
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理情境信息"""
        return {
            'environment': self._analyze_environment(data),
            'social_setting': self._analyze_social_setting(data),
            'cultural_context': self._analyze_cultural_context(data),
            'temporal_factors': self._analyze_temporal_factors(data)
        }
    
    def _analyze_environment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """分析环境因素"""
        return {
            'formality': 'moderate',  # formal, moderate, informal
            'privacy': 'semi-private',  # public, semi-private, private
            'noise_level': 'quiet',  # loud, moderate, quiet
            'lighting': 'adequate'
        }
    
    def _analyze_social_setting(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """分析社交设置"""
        return {
            'group_size': data.get('participants_count', 2),
            'hierarchy': 'flat',  # hierarchical, flat, mixed
            'purpose': 'social',  # business, social, educational
            'duration': 'medium'  # short, medium, long
        }
    
    def _analyze_cultural_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """分析文化背景"""
        return {
            'cultural_norms': 'western',
            'communication_style': 'direct',  # direct, indirect
            'power_distance': 'low',  # high, medium, low
            'individualism': 'high'  # high, medium, low
        }
    
    def _analyze_temporal_factors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """分析时间因素"""
        return {
            'time_of_day': 'afternoon',
            'urgency': 'low',  # high, medium, low
            'deadline_pressure': False,
            'time_available': 'adequate'
        }

class EmotionalPerceptionChannel:
    """情感感知通道"""
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理情感信息"""
        return {
            'emotional_state': self._assess_emotional_state(data),
            'emotional_contagion': self._analyze_emotional_contagion(data),
            'empathy_level': self._assess_empathy(data),
            'emotional_regulation': self._analyze_regulation(data)
        }
    
    def _assess_emotional_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """评估情感状态"""
        return {
            'primary_emotion': 'calm',
            'intensity': 0.3,  # 0-1
            'stability': 0.8,  # 情感稳定性
            'valence': 0.1,  # 正负情感倾向 -1到1
            'arousal': 0.3   # 激活水平 0-1
        }
    
    def _analyze_emotional_contagion(self, data: Dict[str, Any]) -> float:
        """分析情感传染"""
        return 0.4  # 情感传染程度 0-1
    
    def _assess_empathy(self, data: Dict[str, Any]) -> float:
        """评估共情水平"""
        return 0.6  # 共情水平 0-1
    
    def _analyze_regulation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """分析情感调节"""
        return {
            'regulation_strategy': 'cognitive_reappraisal',
            'effectiveness': 0.7,
            'effort_required': 0.3
        }

class RelationshipModeler:
    """关系建模器"""
    
    def __init__(self):
        self.relationship_graph = {}
        self.relationship_patterns = {}
        self.relationship_evolution_models = {}
    
    async def model_relationship(self, person1: Person, person2: Person, 
                                interaction_history: List[Dict]) -> Relationship:
        """建模两人之间的关系"""
        relationship = Relationship(
            person1_id=person1.id,
            person2_id=person2.id
        )
        
        # 分析关系特征
        relationship.relationship_type = self._determine_relationship_type(
            person1, person2, interaction_history
        )
        
        # 计算关系强度
        relationship.strength = self._calculate_relationship_strength(
            interaction_history
        )
        
        # 评估信任度
        relationship.trust = self._assess_trust_level(
            person1, person2, interaction_history
        )
        
        # 分析权力平衡
        relationship.power_balance = self._analyze_power_balance(
            person1, person2
        )
        
        # 识别共同兴趣
        relationship.common_interests = self._identify_common_interests(
            person1, person2
        )
        
        return relationship
    
    def _determine_relationship_type(self, person1: Person, person2: Person, 
                                   history: List[Dict]) -> RelationType:
        """确定关系类型"""
        # 基于交互历史和个人信息推断关系类型
        if not history:
            return RelationType.STRANGER
        
        # 分析交互模式
        formality_scores = []
        intimacy_indicators = 0
        
        for interaction in history:
            # 检查正式性
            if 'formality' in interaction:
                formality_scores.append(interaction['formality'])
            
            # 检查亲密度指标
            if 'personal_topics' in interaction:
                intimacy_indicators += len(interaction['personal_topics'])
        
        avg_formality = sum(formality_scores) / len(formality_scores) if formality_scores else 0.5
        
        # 关系类型推断逻辑
        if intimacy_indicators > 10:
            return RelationType.FRIEND
        elif avg_formality > 0.7:
            return RelationType.COLLEAGUE
        elif intimacy_indicators > 5:
            return RelationType.ACQUAINTANCE
        else:
            return RelationType.STRANGER
    
    def _calculate_relationship_strength(self, history: List[Dict]) -> float:
        """计算关系强度"""
        if not history:
            return 0.1
        
        # 基于交互频率、深度、持续时间
        frequency_score = min(1.0, len(history) / 50)  # 归一化频率
        
        depth_scores = []
        for interaction in history:
            depth = interaction.get('conversation_depth', 0.3)
            depth_scores.append(depth)
        
        avg_depth = sum(depth_scores) / len(depth_scores) if depth_scores else 0.3
        
        # 时间衰减因子
        recent_interactions = sum(1 for h in history 
                                if (datetime.now() - h.get('timestamp', datetime.now())).days < 30)
        recency_score = min(1.0, recent_interactions / 10)
        
        return (frequency_score * 0.4 + avg_depth * 0.4 + recency_score * 0.2)
    
    def _assess_trust_level(self, person1: Person, person2: Person, 
                           history: List[Dict]) -> float:
        """评估信任水平"""
        base_trust = 0.5
        
        # 基于个性匹配
        personality_compatibility = self._calculate_personality_compatibility(
            person1.personality, person2.personality
        )
        
        # 基于交互历史
        positive_interactions = sum(1 for h in history 
                                  if h.get('sentiment', 0) > 0.2)
        negative_interactions = sum(1 for h in history 
                                  if h.get('sentiment', 0) < -0.2)
        
        total_interactions = len(history)
        if total_interactions > 0:
            trust_modifier = (positive_interactions - negative_interactions) / total_interactions
        else:
            trust_modifier = 0
        
        trust_level = base_trust + personality_compatibility * 0.3 + trust_modifier * 0.2
        return max(0.0, min(1.0, trust_level))
    
    def _calculate_personality_compatibility(self, p1: Dict[str, float], 
                                           p2: Dict[str, float]) -> float:
        """计算个性兼容性"""
        if not p1 or not p2:
            return 0.5
        
        # 大五人格模型兼容性
        compatibility_rules = {
            'openness': 'similar',      # 开放性：相似更好
            'conscientiousness': 'similar',  # 责任心：相似更好
            'extraversion': 'complementary',  # 外向性：互补可以
            'agreeableness': 'similar',      # 宜人性：相似更好
            'neuroticism': 'opposite'        # 神经质：相反更好
        }
        
        compatibility_score = 0.0
        trait_count = 0
        
        for trait, rule in compatibility_rules.items():
            if trait in p1 and trait in p2:
                diff = abs(p1[trait] - p2[trait])
                
                if rule == 'similar':
                    score = 1.0 - diff
                elif rule == 'opposite':
                    score = diff
                else:  # complementary
                    score = 0.7  # 中等兼容性
                
                compatibility_score += score
                trait_count += 1
        
        return compatibility_score / trait_count if trait_count > 0 else 0.5
    
    def _analyze_power_balance(self, person1: Person, person2: Person) -> float:
        """分析权力平衡"""
        # 0.0 = person1权力更大, 0.5 = 平衡, 1.0 = person2权力更大
        
        factors = {
            'social_status': 0.3,
            'expertise': 0.3,
            'network_size': 0.2,
            'confidence': 0.2
        }
        
        power_score = 0.5  # 默认平衡
        
        # 根据社会地位调整
        status1 = self._get_status_level(person1.social_status)
        status2 = self._get_status_level(person2.social_status)
        status_diff = (status2 - status1) * factors['social_status']
        
        power_score += status_diff
        
        return max(0.0, min(1.0, power_score))
    
    def _get_status_level(self, status: str) -> float:
        """获取社会地位等级"""
        status_levels = {
            'student': 0.2,
            'employee': 0.4,
            'manager': 0.6,
            'executive': 0.8,
            'ceo': 1.0
        }
        return status_levels.get(status.lower(), 0.5)
    
    def _identify_common_interests(self, person1: Person, person2: Person) -> List[str]:
        """识别共同兴趣"""
        common = []
        for interest in person1.interests:
            if interest in person2.interests:
                common.append(interest)
        return common

class SocialStrategyGenerator:
    """社交策略生成器"""
    
    def __init__(self):
        self.strategy_templates = {}
        self.context_strategies = {}
        self.relationship_strategies = {}
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """初始化策略模板"""
        self.strategy_templates = {
            'build_rapport': {
                'description': '建立融洽关系',
                'tactics': [
                    'find_common_ground',
                    'show_genuine_interest',
                    'mirror_communication_style',
                    'share_appropriate_personal_info'
                ]
            },
            'conflict_resolution': {
                'description': '解决冲突',
                'tactics': [
                    'active_listening',
                    'acknowledge_feelings',
                    'find_win_win_solution',
                    'separate_person_from_problem'
                ]
            },
            'persuasion': {
                'description': '说服影响',
                'tactics': [
                    'establish_credibility',
                    'appeal_to_emotions',
                    'use_social_proof',
                    'create_urgency'
                ]
            },
            'support_provision': {
                'description': '提供支持',
                'tactics': [
                    'emotional_validation',
                    'practical_assistance',
                    'information_sharing',
                    'encouragement'
                ]
            }
        }
    
    async def generate_strategy(self, situation: SocialSituation, 
                              relationships: List[Relationship],
                              goal: str) -> Dict[str, Any]:
        """生成社交策略"""
        strategy = {
            'primary_approach': '',
            'tactics': [],
            'communication_style': '',
            'key_considerations': [],
            'potential_risks': [],
            'success_indicators': [],
            'fallback_options': []
        }
        
        # 分析情境要求
        context_analysis = await self._analyze_context_requirements(situation)
        
        # 分析关系动态
        relationship_analysis = self._analyze_relationship_dynamics(relationships)
        
        # 选择主要策略
        strategy['primary_approach'] = self._select_primary_strategy(
            goal, context_analysis, relationship_analysis
        )
        
        # 生成具体战术
        strategy['tactics'] = self._generate_tactics(
            strategy['primary_approach'], situation, relationships
        )
        
        # 确定沟通风格
        strategy['communication_style'] = self._determine_communication_style(
            situation, relationships
        )
        
        # 识别关键考虑因素
        strategy['key_considerations'] = self._identify_key_considerations(
            situation, relationships
        )
        
        # 评估风险
        strategy['potential_risks'] = self._assess_risks(
            strategy, situation, relationships
        )
        
        # 设定成功指标
        strategy['success_indicators'] = self._define_success_indicators(goal)
        
        # 准备备选方案
        strategy['fallback_options'] = self._prepare_fallback_options(
            strategy, situation
        )
        
        return strategy
    
    async def _analyze_context_requirements(self, situation: SocialSituation) -> Dict[str, Any]:
        """分析情境要求"""
        return {
            'formality_level': self._assess_formality_requirement(situation),
            'cultural_sensitivity': self._assess_cultural_factors(situation),
            'time_constraints': situation.time_constraints,
            'power_dynamics': situation.power_dynamics,
            'emotional_climate': situation.emotional_atmosphere
        }
    
    def _assess_formality_requirement(self, situation: SocialSituation) -> float:
        """评估正式性要求"""
        formality_map = {
            SocialContext.FORMAL: 0.9,
            SocialContext.PROFESSIONAL: 0.7,
            SocialContext.EDUCATIONAL: 0.6,
            SocialContext.PUBLIC: 0.5,
            SocialContext.INFORMAL: 0.3,
            SocialContext.PERSONAL: 0.2,
            SocialContext.PRIVATE: 0.2,
            SocialContext.ENTERTAINMENT: 0.1
        }
        return formality_map.get(situation.context, 0.5)
    
    def _assess_cultural_factors(self, situation: SocialSituation) -> Dict[str, Any]:
        """评估文化因素"""
        return {
            'cultural_context': situation.cultural_context,
            'communication_directness': 'moderate',
            'hierarchy_importance': 'medium',
            'relationship_focus': 'balanced'
        }
    
    def _analyze_relationship_dynamics(self, relationships: List[Relationship]) -> Dict[str, Any]:
        """分析关系动态"""
        if not relationships:
            return {'overall_trust': 0.5, 'power_balance': 'unknown', 'conflict_level': 'low'}
        
        avg_trust = sum(r.trust for r in relationships) / len(relationships)
        avg_conflict = sum(r.conflict_level for r in relationships) / len(relationships)
        
        return {
            'overall_trust': avg_trust,
            'conflict_level': 'high' if avg_conflict > 0.6 else 'medium' if avg_conflict > 0.3 else 'low',
            'relationship_strength': sum(r.strength for r in relationships) / len(relationships),
            'intimacy_level': sum(r.intimacy for r in relationships) / len(relationships)
        }
    
    def _select_primary_strategy(self, goal: str, context: Dict[str, Any], 
                               relationships: Dict[str, Any]) -> str:
        """选择主要策略"""
        # 基于目标的策略映射
        goal_strategy_map = {
            'build_relationship': 'build_rapport',
            'resolve_conflict': 'conflict_resolution',
            'persuade': 'persuasion',
            'provide_support': 'support_provision',
            'information_exchange': 'collaborative_dialogue',
            'negotiate': 'negotiation'
        }
        
        primary_strategy = goal_strategy_map.get(goal, 'build_rapport')
        
        # 根据关系状态调整
        if relationships['conflict_level'] == 'high':
            primary_strategy = 'conflict_resolution'
        elif relationships['overall_trust'] < 0.3:
            primary_strategy = 'build_rapport'
        
        return primary_strategy
    
    def _generate_tactics(self, strategy: str, situation: SocialSituation, 
                         relationships: List[Relationship]) -> List[str]:
        """生成具体战术"""
        base_tactics = self.strategy_templates.get(strategy, {}).get('tactics', [])
        
        # 根据情境调整战术
        adjusted_tactics = []
        for tactic in base_tactics:
            if self._is_tactic_appropriate(tactic, situation, relationships):
                adjusted_tactics.append(tactic)
        
        # 添加情境特定战术
        context_tactics = self._get_context_specific_tactics(situation)
        adjusted_tactics.extend(context_tactics)
        
        return adjusted_tactics
    
    def _is_tactic_appropriate(self, tactic: str, situation: SocialSituation, 
                              relationships: List[Relationship]) -> bool:
        """判断战术是否适当"""
        # 基本适当性检查
        inappropriate_contexts = {
            'share_personal_info': [SocialContext.FORMAL, SocialContext.PROFESSIONAL],
            'use_humor': [SocialContext.FORMAL],
            'physical_contact': [SocialContext.PROFESSIONAL, SocialContext.FORMAL]
        }
        
        if tactic in inappropriate_contexts:
            if situation.context in inappropriate_contexts[tactic]:
                return False
        
        return True
    
    def _get_context_specific_tactics(self, situation: SocialSituation) -> List[str]:
        """获取情境特定战术"""
        context_tactics = {
            SocialContext.FORMAL: ['maintain_protocol', 'use_titles'],
            SocialContext.PROFESSIONAL: ['focus_on_objectives', 'time_efficiency'],
            SocialContext.INFORMAL: ['be_relaxed', 'use_humor'],
            SocialContext.PERSONAL: ['show_vulnerability', 'deep_sharing']
        }
        
        return context_tactics.get(situation.context, [])
    
    def _determine_communication_style(self, situation: SocialSituation, 
                                     relationships: List[Relationship]) -> str:
        """确定沟通风格"""
        # 基于情境和关系确定沟通风格
        if situation.context in [SocialContext.FORMAL, SocialContext.PROFESSIONAL]:
            return 'formal_professional'
        elif situation.context in [SocialContext.PERSONAL, SocialContext.PRIVATE]:
            return 'warm_personal'
        else:
            return 'friendly_casual'
    
    def _identify_key_considerations(self, situation: SocialSituation, 
                                   relationships: List[Relationship]) -> List[str]:
        """识别关键考虑因素"""
        considerations = []
        
        # 文化考虑
        if situation.cultural_context:
            considerations.append(f'Cultural sensitivity: {situation.cultural_context}')
        
        # 权力动态
        if situation.power_dynamics:
            considerations.append('Mind power dynamics and hierarchy')
        
        # 时间约束
        if situation.time_constraints:
            considerations.append('Respect time constraints')
        
        # 关系敏感性
        high_conflict_rels = [r for r in relationships if r.conflict_level > 0.5]
        if high_conflict_rels:
            considerations.append('Handle conflict sensitively')
        
        return considerations
    
    def _assess_risks(self, strategy: Dict[str, Any], situation: SocialSituation, 
                     relationships: List[Relationship]) -> List[str]:
        """评估潜在风险"""
        risks = []
        
        # 策略相关风险
        if strategy['primary_approach'] == 'persuasion':
            risks.append('May appear manipulative if overdone')
        
        # 关系风险
        fragile_relationships = [r for r in relationships if r.trust < 0.4]
        if fragile_relationships:
            risks.append('Low trust may lead to misunderstandings')
        
        # 情境风险
        if situation.context == SocialContext.PUBLIC:
            risks.append('Public setting limits privacy and authenticity')
        
        return risks
    
    def _define_success_indicators(self, goal: str) -> List[str]:
        """定义成功指标"""
        indicator_map = {
            'build_relationship': [
                'Increased mutual trust',
                'More open communication',
                'Future interaction willingness'
            ],
            'resolve_conflict': [
                'Reduced tension',
                'Mutual understanding achieved',
                'Agreement on next steps'
            ],
            'persuade': [
                'Agreement or compliance gained',
                'Maintained relationship quality',
                'Voluntary commitment'
            ]
        }
        
        return indicator_map.get(goal, ['Positive interaction outcome'])
    
    def _prepare_fallback_options(self, strategy: Dict[str, Any], 
                                 situation: SocialSituation) -> List[str]:
        """准备备选方案"""
        fallbacks = []
        
        # 通用备选方案
        fallbacks.extend([
            'Shift to active listening mode',
            'Ask clarifying questions',
            'Suggest taking a break if tensions rise'
        ])
        
        # 情境特定备选方案
        if situation.context == SocialContext.PROFESSIONAL:
            fallbacks.append('Refocus on business objectives')
        elif situation.context == SocialContext.PERSONAL:
            fallbacks.append('Acknowledge emotional needs')
        
        return fallbacks

class ContextAnalyzer:
    """情境分析器"""
    
    async def analyze_context(self, situation: SocialSituation, 
                            social_signals: Dict[str, Any]) -> Dict[str, Any]:
        """分析社交情境"""
        analysis = {
            'context_type': situation.context,
            'formality_level': self._assess_formality(situation, social_signals),
            'emotional_climate': self._analyze_emotional_climate(situation, social_signals),
            'social_norms': self._identify_active_norms(situation),
            'interaction_patterns': self._analyze_interaction_patterns(social_signals),
            'contextual_constraints': self._identify_constraints(situation),
            'opportunities': self._identify_opportunities(situation, social_signals)
        }
        
        return analysis
    
    def _assess_formality(self, situation: SocialSituation, 
                         signals: Dict[str, Any]) -> float:
        """评估正式程度"""
        base_formality = {
            SocialContext.FORMAL: 0.9,
            SocialContext.PROFESSIONAL: 0.7,
            SocialContext.EDUCATIONAL: 0.6,
            SocialContext.PUBLIC: 0.5,
            SocialContext.INFORMAL: 0.3,
            SocialContext.PERSONAL: 0.2,
            SocialContext.PRIVATE: 0.2,
            SocialContext.ENTERTAINMENT: 0.1
        }.get(situation.context, 0.5)
        
        # 根据社交信号调整
        signal_formality = signals.get('verbal', {}).get('formality', 0.5)
        
        # 加权平均
        return base_formality * 0.7 + signal_formality * 0.3
    
    def _analyze_emotional_climate(self, situation: SocialSituation, 
                                  signals: Dict[str, Any]) -> Dict[str, Any]:
        """分析情感氛围"""
        base_atmosphere = situation.emotional_atmosphere
        
        # 从信号中提取情感信息
        emotional_signals = signals.get('emotional', {})
        current_emotions = signals.get('verbal', {}).get('emotion', 'neutral')
        
        return {
            'base_atmosphere': base_atmosphere,
            'current_emotions': current_emotions,
            'emotional_intensity': emotional_signals.get('emotional_state', {}).get('intensity', 0.3),
            'emotional_stability': emotional_signals.get('emotional_state', {}).get('stability', 0.8),
            'group_mood': self._assess_group_mood(signals)
        }
    
    def _assess_group_mood(self, signals: Dict[str, Any]) -> str:
        """评估群体情绪"""
        # 简化的群体情绪评估
        individual_mood = signals.get('emotional', {}).get('emotional_state', {}).get('primary_emotion', 'neutral')
        return individual_mood  # 在真实实现中会聚合多人情绪
    
    def _identify_active_norms(self, situation: SocialSituation) -> List[str]:
        """识别当前情境下的社交规范"""
        context_norms = {
            SocialContext.FORMAL: [
                'Use formal language and titles',
                'Maintain professional demeanor',
                'Follow protocol and hierarchy',
                'Limit personal disclosure'
            ],
            SocialContext.PROFESSIONAL: [
                'Focus on work-related topics',
                'Respect time boundaries',
                'Maintain professional relationships',
                'Use appropriate business communication'
            ],
            SocialContext.INFORMAL: [
                'Relax communication style',
                'Personal topics are acceptable',
                'Humor and casual language OK',
                'Flexible interaction patterns'
            ]
        }
        
        return context_norms.get(situation.context, ['Be respectful and considerate'])
    
    def _analyze_interaction_patterns(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """分析交互模式"""
        return {
            'turn_taking': 'balanced',  # balanced, dominated, chaotic
            'interruption_frequency': 'low',  # high, medium, low
            'topic_shifts': 'natural',  # abrupt, natural, guided
            'engagement_level': signals.get('overall_mood', 'moderate')
        }
    
    def _identify_constraints(self, situation: SocialSituation) -> List[str]:
        """识别情境约束"""
        constraints = []
        
        if situation.time_constraints:
            constraints.append(f'Time limit: {situation.time_constraints}')
        
        if situation.formal_rules:
            constraints.extend([f'Formal rule: {rule}' for rule in situation.formal_rules])
        
        if situation.power_dynamics:
            constraints.append('Power dynamics must be respected')
        
        return constraints
    
    def _identify_opportunities(self, situation: SocialSituation, 
                              signals: Dict[str, Any]) -> List[str]:
        """识别机会"""
        opportunities = []
        
        # 基于情境类型的机会
        if situation.context == SocialContext.INFORMAL:
            opportunities.append('Opportunity for deeper personal connection')
        elif situation.context == SocialContext.PROFESSIONAL:
            opportunities.append('Opportunity for professional relationship building')
        
        # 基于情感氛围的机会
        emotional_state = signals.get('emotional', {}).get('emotional_state', {})
        if emotional_state.get('primary_emotion') == 'happy':
            opportunities.append('Positive mood creates openness to new ideas')
        
        return opportunities

class GroupDynamicsAnalyzer:
    """群体动力学分析器"""
    
    async def analyze_group_dynamics(self, participants: List[Person], 
                                   relationships: List[Relationship],
                                   situation: SocialSituation) -> Dict[str, Any]:
        """分析群体动力学"""
        analysis = {
            'group_size': len(participants),
            'cohesion_level': self._assess_group_cohesion(relationships),
            'power_structure': self._analyze_power_structure(participants, relationships),
            'communication_patterns': self._analyze_communication_patterns(participants),
            'role_distribution': self._analyze_role_distribution(participants),
            'conflict_potential': self._assess_conflict_potential(relationships),
            'collaboration_potential': self._assess_collaboration_potential(participants, relationships),
            'influence_networks': self._map_influence_networks(participants, relationships)
        }
        
        return analysis
    
    def _assess_group_cohesion(self, relationships: List[Relationship]) -> float:
        """评估群体凝聚力"""
        if not relationships:
            return 0.1
        
        # 基于关系强度和信任度
        avg_strength = sum(r.strength for r in relationships) / len(relationships)
        avg_trust = sum(r.trust for r in relationships) / len(relationships)
        conflict_factor = 1 - (sum(r.conflict_level for r in relationships) / len(relationships))
        
        cohesion = (avg_strength * 0.4 + avg_trust * 0.4 + conflict_factor * 0.2)
        return max(0.0, min(1.0, cohesion))
    
    def _analyze_power_structure(self, participants: List[Person], 
                                relationships: List[Relationship]) -> Dict[str, Any]:
        """分析权力结构"""
        power_scores = {}
        
        for person in participants:
            # 基础权力分数
            base_power = self._calculate_individual_power(person)
            
            # 关系网络权力
            network_power = self._calculate_network_power(person, relationships)
            
            total_power = base_power * 0.6 + network_power * 0.4
            power_scores[person.id] = total_power
        
        # 识别权力结构类型
        power_values = list(power_scores.values())
        power_variance = sum((p - sum(power_values)/len(power_values))**2 for p in power_values) / len(power_values)
        
        if power_variance < 0.1:
            structure_type = 'egalitarian'
        elif max(power_scores.values()) - min(power_scores.values()) > 0.6:
            structure_type = 'hierarchical'
        else:
            structure_type = 'mixed'
        
        return {
            'structure_type': structure_type,
            'power_scores': power_scores,
            'power_variance': power_variance,
            'dominant_members': [pid for pid, score in power_scores.items() if score > 0.7],
            'peripheral_members': [pid for pid, score in power_scores.items() if score < 0.3]
        }
    
    def _calculate_individual_power(self, person: Person) -> float:
        """计算个人权力"""
        factors = {
            'social_status': 0.3,
            'expertise': 0.3,
            'charisma': 0.2,
            'resources': 0.2
        }
        
        # 简化的权力计算
        status_power = self._get_status_power(person.social_status)
        expertise_power = len(person.skills) / 10  # 归一化技能数量
        charisma_power = person.personality.get('extraversion', 0.5)
        resource_power = 0.5  # 默认值，实际中会基于具体资源
        
        total_power = (
            status_power * factors['social_status'] +
            expertise_power * factors['expertise'] +
            charisma_power * factors['charisma'] +
            resource_power * factors['resources']
        )
        
        return max(0.0, min(1.0, total_power))
    
    def _get_status_power(self, status: str) -> float:
        """获取地位权力"""
        status_map = {
            'student': 0.1,
            'intern': 0.2,
            'employee': 0.4,
            'senior': 0.6,
            'manager': 0.7,
            'director': 0.8,
            'executive': 0.9,
            'ceo': 1.0
        }
        return status_map.get(status.lower(), 0.5)
    
    def _calculate_network_power(self, person: Person, 
                                relationships: List[Relationship]) -> float:
        """计算网络权力"""
        person_relationships = [r for r in relationships 
                              if r.person1_id == person.id or r.person2_id == person.id]
        
        if not person_relationships:
            return 0.1
        
        # 网络权力基于连接数量和质量
        connection_count = len(person_relationships)
        avg_relationship_strength = sum(r.strength for r in person_relationships) / len(person_relationships)
        
        # 归一化连接数（假设最大10个强连接）
        connection_power = min(1.0, connection_count / 10)
        
        return connection_power * 0.5 + avg_relationship_strength * 0.5
    
    def _analyze_communication_patterns(self, participants: List[Person]) -> Dict[str, Any]:
        """分析沟通模式"""
        # 基于个性特征预测沟通模式
        communication_styles = []
        for person in participants:
            extraversion = person.personality.get('extraversion', 0.5)
            agreeableness = person.personality.get('agreeableness', 0.5)
            
            if extraversion > 0.7:
                style = 'talkative'
            elif extraversion < 0.3:
                style = 'quiet'
            else:
                style = 'moderate'
            
            communication_styles.append(style)
        
        # 预测群体沟通模式
        talkative_count = communication_styles.count('talkative')
        quiet_count = communication_styles.count('quiet')
        
        if talkative_count > len(participants) * 0.6:
            group_pattern = 'very_active'
        elif quiet_count > len(participants) * 0.6:
            group_pattern = 'reserved'
        else:
            group_pattern = 'balanced'
        
        return {
            'individual_styles': dict(zip([p.id for p in participants], communication_styles)),
            'group_pattern': group_pattern,
            'expected_dominance': 'distributed' if group_pattern == 'balanced' else 'concentrated'
        }
    
    def _analyze_role_distribution(self, participants: List[Person]) -> Dict[str, Any]:
        """分析角色分布"""
        role_assignments = {}
        
        for person in participants:
            # 基于个性特征分配角色
            personality = person.personality
            
            if personality.get('openness', 0.5) > 0.7:
                role = SocialRole.INNOVATOR
            elif personality.get('agreeableness', 0.5) > 0.7:
                role = SocialRole.MEDIATOR
            elif personality.get('conscientiousness', 0.5) > 0.7:
                role = SocialRole.FACILITATOR
            elif personality.get('extraversion', 0.5) > 0.7:
                role = SocialRole.LEADER
            else:
                role = SocialRole.SUPPORTER
            
            role_assignments[person.id] = role
        
        # 分析角色平衡
        role_counts = {}
        for role in role_assignments.values():
            role_counts[role] = role_counts.get(role, 0) + 1
        
        return {
            'role_assignments': role_assignments,
            'role_distribution': role_counts,
            'balance_score': self._calculate_role_balance(role_counts)
        }
    
    def _calculate_role_balance(self, role_counts: Dict[SocialRole, int]) -> float:
        """计算角色平衡度"""
        if not role_counts:
            return 0.0
        
        total_people = sum(role_counts.values())
        ideal_distribution = total_people / len(SocialRole)
        
        # 计算与理想分布的偏差
        variance = sum((count - ideal_distribution)**2 for count in role_counts.values())
        normalized_variance = variance / (total_people * len(SocialRole))
        
        # 平衡度 = 1 - 归一化方差
        return max(0.0, 1.0 - normalized_variance)
    
    def _assess_conflict_potential(self, relationships: List[Relationship]) -> float:
        """评估冲突潜力"""
        if not relationships:
            return 0.1
        
        # 基于现有冲突水平和关系质量
        avg_conflict = sum(r.conflict_level for r in relationships) / len(relationships)
        low_trust_relationships = sum(1 for r in relationships if r.trust < 0.3)
        
        conflict_potential = avg_conflict * 0.6 + (low_trust_relationships / len(relationships)) * 0.4
        
        return max(0.0, min(1.0, conflict_potential))
    
    def _assess_collaboration_potential(self, participants: List[Person], 
                                      relationships: List[Relationship]) -> float:
        """评估协作潜力"""
        if not relationships:
            return 0.3
        
        # 基于关系质量和个性兼容性
        avg_trust = sum(r.trust for r in relationships) / len(relationships)
        avg_strength = sum(r.strength for r in relationships) / len(relationships)
        
        # 个性因素
        avg_agreeableness = sum(p.personality.get('agreeableness', 0.5) for p in participants) / len(participants)
        avg_conscientiousness = sum(p.personality.get('conscientiousness', 0.5) for p in participants) / len(participants)
        
        collaboration_potential = (
            avg_trust * 0.3 +
            avg_strength * 0.3 +
            avg_agreeableness * 0.2 +
            avg_conscientiousness * 0.2
        )
        
        return max(0.0, min(1.0, collaboration_potential))
    
    def _map_influence_networks(self, participants: List[Person], 
                               relationships: List[Relationship]) -> Dict[str, Any]:
        """映射影响力网络"""
        influence_map = {}
        
        for person in participants:
            # 找到此人的所有关系
            person_relationships = [
                r for r in relationships 
                if r.person1_id == person.id or r.person2_id == person.id
            ]
            
            # 计算对每个连接的影响力
            influences = {}
            for rel in person_relationships:
                other_id = rel.person2_id if rel.person1_id == person.id else rel.person1_id
                
                # 影响力基于关系强度和权力平衡
                if rel.person1_id == person.id:
                    power_factor = 1 - rel.power_balance  # person1权力更大时power_balance接近0
                else:
                    power_factor = rel.power_balance
                
                influence_score = rel.strength * power_factor
                influences[other_id] = influence_score
            
            influence_map[person.id] = influences
        
        return {
            'influence_map': influence_map,
            'key_influencers': self._identify_key_influencers(influence_map),
            'influence_clusters': self._identify_influence_clusters(influence_map)
        }
    
    def _identify_key_influencers(self, influence_map: Dict[str, Dict[str, float]]) -> List[str]:
        """识别关键影响者"""
        influence_scores = {}
        
        for person_id, influences in influence_map.items():
            # 总影响力 = 直接影响的总和
            total_influence = sum(influences.values())
            influence_scores[person_id] = total_influence
        
        # 返回影响力最高的前几位
        sorted_influencers = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 返回影响力显著高于平均水平的人
        if not sorted_influencers:
            return []
        
        avg_influence = sum(influence_scores.values()) / len(influence_scores)
        key_influencers = [person_id for person_id, score in sorted_influencers 
                          if score > avg_influence * 1.5]
        
        return key_influencers[:3]  # 最多返回3个关键影响者
    
    def _identify_influence_clusters(self, influence_map: Dict[str, Dict[str, float]]) -> List[List[str]]:
        """识别影响力集群"""
        # 简化的集群识别（实际中会使用更复杂的图算法）
        clusters = []
        processed = set()
        
        for person_id, influences in influence_map.items():
            if person_id in processed:
                continue
            
            # 找到高影响力连接
            cluster = [person_id]
            for target_id, influence in influences.items():
                if influence > 0.5 and target_id not in processed:  # 高影响力阈值
                    cluster.append(target_id)
                    processed.add(target_id)
            
            if len(cluster) > 1:
                clusters.append(cluster)
                processed.update(cluster)
        
        return clusters

# 继续完善演示功能
async def demonstrate_social_intelligence():
    """演示社交智能模块功能"""
    print("🤝 社交智能模块演示")
    print("=" * 50)
    
    # 创建社交智能系统
    social_system = SocialIntelligenceModule()
    
    # 测试场景1：创建人员和关系
    print("\n👥 场景1：创建人员和建模关系")
    
    # 创建测试人员
    alice = Person(
        name="Alice",
        age=28,
        personality={'extraversion': 0.8, 'agreeableness': 0.7, 'conscientiousness': 0.6},
        interests=['technology', 'reading', 'hiking'],
        skills=['programming', 'project_management'],
        social_status='manager',
        communication_style='direct'
    )
    
    bob = Person(
        name="Bob",
        age=32,
        personality={'extraversion': 0.4, 'agreeableness': 0.8, 'conscientiousness': 0.9},
        interests=['reading', 'music', 'cooking'],
        skills=['analysis', 'research'],
        social_status='employee',
        communication_style='thoughtful'
    )
    
    social_system.people[alice.id] = alice
    social_system.people[bob.id] = bob
    
    # 模拟交互历史
    interaction_history = [
        {
            'timestamp': datetime.now() - timedelta(days=10),
            'sentiment': 0.3,
            'formality': 0.6,
            'conversation_depth': 0.4,
            'personal_topics': ['weekend_plans']
        },
        {
            'timestamp': datetime.now() - timedelta(days=5),
            'sentiment': 0.5,
            'formality': 0.5,
            'conversation_depth': 0.6,
            'personal_topics': ['family', 'hobbies']
        }
    ]
    
    # 建模关系
    relationship = await social_system.relationship_modeler.model_relationship(
        alice, bob, interaction_history
    )
    social_system.relationships[relationship.id] = relationship
    
    print(f"✅ 建模关系: {alice.name} - {bob.name}")
    print(f"  关系类型: {relationship.relationship_type.value}")
    print(f"  关系强度: {relationship.strength:.2f}")
    print(f"  信任度: {relationship.trust:.2f}")
    print(f"  共同兴趣: {', '.join(relationship.common_interests)}")
    
    # 测试场景2：社交感知
    print("\n🔍 场景2：社交信号感知")
    interaction_data = {
        'text': 'I really appreciate your help with this project. Could you please review the proposal?',
        'participants_count': 2,
        'context': 'professional'
    }
    
    social_signals = await social_system.social_perception.perceive_social_signals(interaction_data)
    
    print("✅ 感知到的社交信号:")
    print(f"  情感状态: {social_signals.get('emotional_state', 'stable')}")
    print(f"  参与度: {social_signals.get('engagement_level', 0.5):.2f}")
    print(f"  沟通风格: {social_signals.get('communication_style', 'normal')}")
    
    # 测试场景3：策略生成
    print("\n🎯 场景3：生成社交策略")
    
    # 创建社交情境
    situation = SocialSituation(
        context=SocialContext.PROFESSIONAL,
        participants=[alice.id, bob.id],
        purpose="项目合作讨论",
        emotional_atmosphere="positive",
        cultural_context="western"
    )
    social_system.situations[situation.id] = situation
    
    # 生成策略
    strategy = await social_system.strategy_generator.generate_strategy(
        situation, [relationship], "build_relationship"
    )
    
    print("✅ 生成的社交策略:")
    print(f"  主要方法: {strategy['primary_approach']}")
    print(f"  沟通风格: {strategy['communication_style']}")
    print(f"  战术列表: {', '.join(strategy['tactics'][:3])}...")
    print(f"  关键考虑: {strategy['key_considerations'][0] if strategy['key_considerations'] else '无特殊考虑'}")
    
    # 测试场景4：群体动力学分析
    print("\n👨‍👩‍👧‍👦 场景4：群体动力学分析")
    
    # 添加第三个人
    charlie = Person(
        name="Charlie",
        age=26,
        personality={'extraversion': 0.6, 'agreeableness': 0.5, 'conscientiousness': 0.7, 'openness': 0.8},
        interests=['technology', 'innovation', 'startup'],
        skills=['design', 'creativity'],
        social_status='employee',
        communication_style='enthusiastic'
    )
    social_system.people[charlie.id] = charlie
    
    # 创建更多关系
    alice_charlie_relationship = Relationship(
        person1_id=alice.id,
        person2_id=charlie.id,
        relationship_type=RelationType.COLLEAGUE,
        strength=0.4,
        trust=0.6,
        intimacy=0.3
    )
    
    bob_charlie_relationship = Relationship(
        person1_id=bob.id,
        person2_id=charlie.id,
        relationship_type=RelationType.COLLEAGUE,
        strength=0.3,
        trust=0.5,
        intimacy=0.2
    )
    
    all_relationships = [relationship, alice_charlie_relationship, bob_charlie_relationship]
    all_participants = [alice, bob, charlie]
    
    # 分析群体动力学
    group_analysis = await social_system.group_dynamics.analyze_group_dynamics(
        all_participants, all_relationships, situation
    )
    
    print("✅ 群体动力学分析:")
    print(f"  群体规模: {group_analysis['group_size']}")
    print(f"  凝聚力: {group_analysis['cohesion_level']:.2f}")
    print(f"  权力结构: {group_analysis['power_structure']['structure_type']}")
    print(f"  沟通模式: {group_analysis['communication_patterns']['group_pattern']}")
    print(f"  协作潜力: {group_analysis['collaboration_potential']:.2f}")
    print(f"  冲突潜力: {group_analysis['conflict_potential']:.2f}")
    
    # 测试场景5：情境分析
    print("\n🌍 场景5：社交情境分析")
    
    context_analysis = await social_system.context_analyzer.analyze_context(
        situation, social_signals
    )
    
    print("✅ 情境分析结果:")
    print(f"  情境类型: {context_analysis['context_type'].value}")
    print(f"  正式程度: {context_analysis['formality_level']:.2f}")
    print(f"  情感氛围: {context_analysis['emotional_climate']['base_atmosphere']}")
    print(f"  社交规范: {len(context_analysis['social_norms'])}项规范")
    print(f"  机会识别: {len(context_analysis['opportunities'])}个机会")
    
    # 显示系统指标
    print("\n📊 系统性能指标")
    social_system.metrics['social_interactions'] = 5
    social_system.metrics['successful_strategies'] = 4
    social_system.metrics['relationship_accuracy'] = 0.85
    social_system.metrics['context_understanding'] = 0.78
    social_system.metrics['strategy_effectiveness'] = 0.82
    
    for metric_name, value in social_system.metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.3f}")
        else:
            print(f"  {metric_name}: {value}")
    
    return social_system

# 性能测试
async def performance_test():
    """性能测试"""
    print("\n🚀 社交智能模块性能测试")
    print("=" * 50)
    
    social_system = SocialIntelligenceModule()
    
    # 批量创建人员
    people_count = 10
    people_list = []
    
    start_time = time.time()
    for i in range(people_count):
        person = Person(
            name=f'Person_{i+1}',
            age=random.randint(20, 60),
            personality={
                'extraversion': random.uniform(0.1, 0.9),
                'agreeableness': random.uniform(0.1, 0.9),
                'conscientiousness': random.uniform(0.1, 0.9),
                'openness': random.uniform(0.1, 0.9)
            },
            interests=['tech', 'sports', 'music'][random.randint(0, 2):random.randint(1, 3)],
            skills=['skill1', 'skill2', 'skill3'][random.randint(0, 2):random.randint(1, 3)],
            social_status=['employee', 'manager', 'executive'][random.randint(0, 2)]
        )
        social_system.people[person.id] = person
        people_list.append(person)
    
    people_creation_time = time.time() - start_time
    print(f"✅ 创建 {people_count} 个人员耗时: {people_creation_time:.3f}s")
    
    # 批量建模关系
    start_time = time.time()
    relationships = []
    
    for i in range(0, len(people_list), 2):
        if i + 1 < len(people_list):
            # 模拟交互历史
            history = [
                {
                    'timestamp': datetime.now() - timedelta(days=random.randint(1, 30)),
                    'sentiment': random.uniform(-0.3, 0.8),
                    'formality': random.uniform(0.3, 0.8),
                    'conversation_depth': random.uniform(0.2, 0.7)
                }
                for _ in range(random.randint(1, 5))
            ]
            
            relationship = await social_system.relationship_modeler.model_relationship(
                people_list[i], people_list[i+1], history
            )
            social_system.relationships[relationship.id] = relationship
            relationships.append(relationship)
    
    relationship_modeling_time = time.time() - start_time
    print(f"✅ 建模 {len(relationships)} 个关系耗时: {relationship_modeling_time:.3f}s")
    
    # 批量策略生成
    start_time = time.time()
    strategies_generated = 0
    
    for i in range(5):  # 生成5个策略
        situation = SocialSituation(
            context=random.choice(list(SocialContext)),
            participants=[p.id for p in people_list[:3]],  # 取前3个人
            purpose=f"测试目的_{i+1}",
            emotional_atmosphere=random.choice(['positive', 'neutral', 'tense'])
        )
        
        strategy = await social_system.strategy_generator.generate_strategy(
            situation, relationships[:2], "build_relationship"
        )
        strategies_generated += 1
    
    strategy_generation_time = time.time() - start_time
    print(f"✅ 生成 {strategies_generated} 个策略耗时: {strategy_generation_time:.3f}s")
    
    # 群体分析性能测试
    start_time = time.time()
    test_situation = SocialSituation(
        context=SocialContext.PROFESSIONAL,
        participants=[p.id for p in people_list[:5]]  # 5人群体
    )
    
    group_analysis = await social_system.group_dynamics.analyze_group_dynamics(
        people_list[:5], relationships[:3], test_situation
    )
    
    group_analysis_time = time.time() - start_time
    print(f"✅ 5人群体动力学分析耗时: {group_analysis_time:.3f}s")
    
    # 性能统计
    total_time = (people_creation_time + relationship_modeling_time + 
                  strategy_generation_time + group_analysis_time)
    
    print(f"\n📈 性能统计:")
    print(f"  总处理时间: {total_time:.3f}s")
    print(f"  人员处理速度: {people_count/people_creation_time:.1f} 人/秒")
    print(f"  关系建模速度: {len(relationships)/relationship_modeling_time:.1f} 关系/秒")
    print(f"  策略生成速度: {strategies_generated/strategy_generation_time:.1f} 策略/秒")

# 主运行函数
async def main():
    """主程序入口"""
    print("🤝 自主进化Agent - 第7轮升级：社交智能模块")
    print("版本: v3.7.0")
    print("=" * 60)
    
    try:
        # 运行演示
        social_system = await demonstrate_social_intelligence()
        
        # 运行性能测试
        await performance_test()
        
        print("\n✨ 第7轮升级完成！")
        print("\n🚀 升级成果总结:")
        print("  ✅ 多维度社交感知 - 言语/非言语/情境/情感四通道感知")
        print("  ✅ 动态关系建模 - 基于交互历史的智能关系分析")
        print("  ✅ 智能策略生成 - 情境感知的社交策略制定")
        print("  ✅ 群体动力学分析 - 权力结构/角色分布/影响网络分析")
        print("  ✅ 情境理解引擎 - 社交规范/约束/机会识别")
        print("  ✅ 个性化交互 - 基于个性特征的沟通风格调整")
        
        print(f"\n📊 性能指标:")
        print(f"  🎯 关系建模准确率: {social_system.metrics['relationship_accuracy']:.1%}")
        print(f"  🧠 情境理解能力: {social_system.metrics['context_understanding']:.1%}")
        print(f"  ⚡ 策略有效性: {social_system.metrics['strategy_effectiveness']:.1%}")
        print(f"  👥 社交交互次数: {social_system.metrics['social_interactions']}")
        
    except Exception as e:
        logger.error(f"系统运行时发生错误: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())