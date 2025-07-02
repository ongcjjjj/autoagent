"""
自主进化Agent系统
基于最新研究实现的自我进化、自我优化AI Agent框架

主要特性：
1. ReAct架构 - 推理与行动循环
2. 多Agent协作 - 角色专业化
3. 自我评估与改进
4. 训练无关评估系统
5. 记忆与学习机制
"""

import asyncio
import json
import logging
import uuid
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # 如果没有numpy，使用内置的math模块
    import math
    import random
    
    # 创建numpy的替代实现
    class NumpyAlternative:
        @staticmethod
        def random():
            return random.random()
        
        @staticmethod
        def uniform(low, high):
            return random.uniform(low, high)
        
        @staticmethod
        def normal(mu, sigma):
            return random.gauss(mu, sigma)
        
        @staticmethod
        def mean(arr):
            return sum(arr) / len(arr) if arr else 0
        
        @staticmethod
        def log(x):
            return math.log(max(x, 1e-8))
        
        @staticmethod
        def exp(x):
            return math.exp(min(x, 700))  # 防止溢出
    
    np = NumpyAlternative()

from datetime import datetime
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
import json


# 系统级配置常量
MAX_MEMORY_SIZE: int = 100
PERFORMANCE_HISTORY_SIZE: int = 50
SUCCESS_PATTERN_LIMIT: int = 20
OPTIMIZATION_THRESHOLD: float = 0.1
COLLABORATION_TIMEOUT: float = 30.0

# 评估权重配置
EVALUATION_WEIGHTS = {
    'trainability': 0.15,
    'generalization': 0.15,
    'expressiveness': 0.10,
    'creativity_score': 0.15,
    'adaptation_rate': 0.10,
    'collaboration_efficiency': 0.15,
    'error_recovery_rate': 0.10,
    'knowledge_retention': 0.05,
    'innovation_index': 0.05
}

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent角色定义"""
    RESEARCHER = "researcher"          # 研究者 - 信息收集与分析
    EXECUTOR = "executor"             # 执行者 - 任务执行
    CRITIC = "critic"                 # 评判者 - 性能评估
    COORDINATOR = "coordinator"       # 协调者 - 任务分配与协调
    OPTIMIZER = "optimizer"           # 优化者 - 自我改进
    MEMORY_MANAGER = "memory_manager" # 记忆管理者 - 知识存储与检索
    ARCHITECT = "architect"           # 架构师 - 系统设计与重构
    MONITOR = "monitor"              # 监控者 - 系统监控与预警
    LEARNER = "learner"              # 学习者 - 模式识别与知识提取


class ActionType(Enum):
    """行动类型"""
    THINK = "think"
    OBSERVE = "observe"
    EXECUTE = "execute"
    COMMUNICATE = "communicate"
    LEARN = "learn"
    SELF_MODIFY = "self_modify"


@dataclass
class AgentAction:
    """Agent行动数据结构"""
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    action_type: ActionType = ActionType.THINK
    content: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMemory:
    """Agent记忆数据结构"""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Any = None
    memory_type: str = "general"
    importance: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    success_rate: float = 0.0


@dataclass
class EvaluationMetrics:
    """评估指标"""
    trainability: float = 0.0      # 可训练性 (ST)
    generalization: float = 0.0    # 泛化能力 (SG)
    expressiveness: float = 0.0    # 表达能力 (SE)
    composite_score: float = 0.0   # 综合得分
    execution_time: float = 0.0    # 执行时间
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    # 新增的高级评估指标
    creativity_score: float = 0.0      # 创造性得分
    adaptation_rate: float = 0.0       # 适应速度
    collaboration_efficiency: float = 0.0  # 协作效率
    error_recovery_rate: float = 0.0   # 错误恢复率
    knowledge_retention: float = 0.0   # 知识保持率
    innovation_index: float = 0.0      # 创新指数
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'trainability': self.trainability,
            'generalization': self.generalization,
            'expressiveness': self.expressiveness,
            'composite_score': self.composite_score,
            'execution_time': self.execution_time,
            'resource_usage': self.resource_usage,
            'creativity_score': self.creativity_score,
            'adaptation_rate': self.adaptation_rate,
            'collaboration_efficiency': self.collaboration_efficiency,
            'error_recovery_rate': self.error_recovery_rate,
            'knowledge_retention': self.knowledge_retention,
            'innovation_index': self.innovation_index
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationMetrics':
        """从字典创建实例"""
        return cls(
            trainability=data.get('trainability', 0.0),
            generalization=data.get('generalization', 0.0),
            expressiveness=data.get('expressiveness', 0.0),
            composite_score=data.get('composite_score', 0.0),
            execution_time=data.get('execution_time', 0.0),
            resource_usage=data.get('resource_usage', {}),
            creativity_score=data.get('creativity_score', 0.0),
            adaptation_rate=data.get('adaptation_rate', 0.0),
            collaboration_efficiency=data.get('collaboration_efficiency', 0.0),
            error_recovery_rate=data.get('error_recovery_rate', 0.0),
            knowledge_retention=data.get('knowledge_retention', 0.0),
            innovation_index=data.get('innovation_index', 0.0)
        )


class CommunicationProtocol:
    """Agent间通信协议"""
    
    def __init__(self):
        self.message_queue: Dict[str, List[Dict]] = {}
        self.subscribers: Dict[str, List[str]] = {}
        self.lock = threading.Lock()
    
    def subscribe(self, topic: str, agent_id: str):
        """订阅主题"""
        with self.lock:
            if topic not in self.subscribers:
                self.subscribers[topic] = []
            if agent_id not in self.subscribers[topic]:
                self.subscribers[topic].append(agent_id)
    
    def publish(self, topic: str, message: Dict, sender_id: str):
        """发布消息"""
        with self.lock:
            if topic in self.subscribers:
                for agent_id in self.subscribers[topic]:
                    if agent_id != sender_id:  # 不发送给自己
                        if agent_id not in self.message_queue:
                            self.message_queue[agent_id] = []
                        self.message_queue[agent_id].append({
                            'topic': topic,
                            'message': message,
                            'sender': sender_id,
                            'timestamp': datetime.now()
                        })
    
    def get_messages(self, agent_id: str) -> List[Dict]:
        """获取消息"""
        with self.lock:
            messages = self.message_queue.get(agent_id, [])
            self.message_queue[agent_id] = []  # 清空已读消息
            return messages


class AdvancedEvaluator:
    """高级评估器 - 基于最新研究的多指标评估系统"""
    
    @staticmethod
    def calculate_trainability(gradients: List[float], dataset_size: int = 1000, 
                             lipschitz_constant: float = 1.0, batch_size: int = 32) -> float:
        """计算可训练性指标 (ST) - 改进版本"""
        if not gradients:
            return 0.0
        
        # 使用梯度统计信息进行更精确的计算
        gradient_norm_squared = sum(g**2 for g in gradients)
        gradient_variance = sum((g - np.mean(gradients))**2 for g in gradients) / len(gradients)
        
        # 结合梯度范数和方差
        st = (dataset_size / (lipschitz_constant * batch_size)) * gradient_norm_squared
        st_adjusted = st * (1 + gradient_variance)  # 方差越大，训练能力越强
        
        return min(st_adjusted / 1000.0, 1.0)  # 归一化到[0,1]
    
    @staticmethod
    def calculate_generalization(original_output: List[float], 
                               noisy_output: List[float]) -> float:
        """计算泛化能力指标 (SG)"""
        if len(original_output) != len(noisy_output):
            return 0.0
        
        differences = [(o - n)**2 for o, n in zip(original_output, noisy_output)]
        sg = sum(differences)
        return max(0.0, 1.0 - sg / len(differences))  # 转换为[0,1]范围，值越高越好
    
    @staticmethod
    def calculate_expressiveness(complexity_score: float, 
                               variance_stats: List[float]) -> float:
        """计算表达能力指标 (SE)"""
        if not variance_stats:
            return complexity_score
        
        variance_term = sum(np.log(max(v, 1e-8)) for v in variance_stats) / len(variance_stats)
        se = complexity_score + variance_term
        return max(0.0, min(se / 10.0, 1.0))  # 归一化
    
    @staticmethod
    def calculate_composite_score(st: float, sg: float, se: float, epsilon: float = 1e-8) -> float:
        """计算综合得分 - 使用对数-指数变换"""
        metrics = [st, sg, se]
        ranks = [sorted(metrics, reverse=True).index(m) + 1 for m in metrics]
        
        composite = sum(np.exp(-np.log(rank + epsilon)) for rank in ranks)
        return composite / 3.0  # 归一化
    
    @staticmethod
    def calculate_creativity_score(action_patterns: List[Dict], innovation_threshold: float = 0.3) -> float:
        """计算创造性得分 - 基于行动模式的新颖性"""
        if len(action_patterns) < 2:
            return 0.5
        
        # 计算行动模式的多样性
        pattern_types = set()
        for pattern in action_patterns:
            pattern_signature = str(sorted(pattern.items()))
            pattern_types.add(pattern_signature)
        
        diversity_ratio = len(pattern_types) / len(action_patterns)
        
        # 计算创新程度 - 新模式的比例
        recent_patterns = action_patterns[-5:]  # 最近5个模式
        new_patterns = 0
        for pattern in recent_patterns:
            pattern_sig = str(sorted(pattern.items()))
            if action_patterns[:-5].count({'signature': pattern_sig}) == 0:
                new_patterns += 1
        
        innovation_rate = new_patterns / len(recent_patterns) if recent_patterns else 0
        
        # 综合创造性得分
        creativity = (diversity_ratio * 0.6 + innovation_rate * 0.4)
        return min(creativity, 1.0)
    
    @staticmethod
    def calculate_adaptation_rate(performance_history: List[float], window_size: int = 5) -> float:
        """计算适应速度 - 基于性能改进的速度"""
        if len(performance_history) < window_size:
            return 0.5
        
        # 计算滑动窗口内的改进趋势
        recent_perf = performance_history[-window_size:]
        earlier_perf = performance_history[-window_size*2:-window_size] if len(performance_history) >= window_size*2 else recent_perf
        
        recent_avg = sum(recent_perf) / len(recent_perf)
        earlier_avg = sum(earlier_perf) / len(earlier_perf)
        
        improvement_rate = (recent_avg - earlier_avg) / max(earlier_avg, 0.001)
        
        # 转换为0-1范围
        adaptation_rate = 0.5 + improvement_rate * 0.5
        return max(0.0, min(adaptation_rate, 1.0))
    
    @staticmethod
    def calculate_collaboration_efficiency(communication_data: List[Dict]) -> float:
        """计算协作效率 - 基于通信效果和响应时间"""
        if not communication_data:
            return 0.5
        
        total_messages = len(communication_data)
        successful_interactions = sum(1 for msg in communication_data 
                                    if msg.get('response_received', False))
        
        success_rate = successful_interactions / total_messages
        
        # 计算平均响应时间
        response_times = [msg.get('response_time', 1.0) for msg in communication_data 
                         if msg.get('response_time')]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 1.0
        
        # 响应时间越短，效率越高
        time_efficiency = 1.0 / (1.0 + avg_response_time)
        
        collaboration_efficiency = (success_rate * 0.7 + time_efficiency * 0.3)
        return min(collaboration_efficiency, 1.0)
    
    @staticmethod
    def calculate_error_recovery_rate(error_history: List[Dict]) -> float:
        """计算错误恢复率 - 系统从错误中恢复的能力"""
        if not error_history:
            return 1.0  # 没有错误记录，假设恢复能力良好
        
        total_errors = len(error_history)
        recovered_errors = sum(1 for error in error_history 
                              if error.get('recovered', False))
        
        recovery_rate = recovered_errors / total_errors
        
        # 考虑恢复时间
        recovery_times = [error.get('recovery_time', 0) for error in error_history 
                         if error.get('recovered', False)]
        
        if recovery_times:
            avg_recovery_time = sum(recovery_times) / len(recovery_times)
            time_penalty = min(avg_recovery_time / 10.0, 0.3)  # 最多减少30%
            recovery_rate = recovery_rate * (1 - time_penalty)
        
        return min(recovery_rate, 1.0)


class TrainingFreeEvaluator:
    """训练无关评估器 - 无需训练即可评估模型性能"""
    
    def evaluate_trainability(self, model_params: Dict) -> float:
        """评估模型可训练性"""
        # 基于参数数量、梯度统计等评估
        param_count = model_params.get('param_count', 1000)
        gradient_norm = model_params.get('gradient_norm', 1.0)
        learning_rate = model_params.get('learning_rate', 0.01)
        
        # 简化的可训练性评估
        # 参数数量适中，梯度范数稳定，学习率合理时可训练性高
        param_factor = min(1.0, np.log(param_count) / 10.0)
        gradient_factor = 1.0 / (1.0 + gradient_norm)
        lr_factor = min(1.0, learning_rate * 10)
        
        trainability = (param_factor * 0.4 + gradient_factor * 0.4 + lr_factor * 0.2)
        return max(0.1, min(trainability, 1.0))
    
    def evaluate_generalization(self, model_complexity: float) -> float:
        """评估泛化能力"""
        # 基于模型复杂度评估泛化能力
        # 复杂度适中时泛化能力最好
        if model_complexity < 0.3:
            return 0.6 + model_complexity  # 复杂度太低，表达能力不足
        elif model_complexity > 0.8:
            return 1.4 - model_complexity  # 复杂度太高，容易过拟合
        else:
            return 0.8 + 0.2 * (1 - abs(model_complexity - 0.55) / 0.25)
    
    def evaluate_expressiveness(self, architecture_info: Dict) -> float:
        """评估表达能力"""
        # 基于架构信息评估表达能力
        layer_count = architecture_info.get('layer_count', 3)
        parameter_diversity = architecture_info.get('parameter_diversity', 0.5)
        activation_types = architecture_info.get('activation_types', 1)
        
        # 层数、参数多样性、激活函数类型影响表达能力
        layer_factor = min(1.0, layer_count / 10.0)
        diversity_factor = parameter_diversity
        activation_factor = min(1.0, activation_types / 5.0)
        
        expressiveness = (layer_factor * 0.4 + diversity_factor * 0.4 + activation_factor * 0.2)
        return max(0.1, min(expressiveness, 1.0))


class BaseAgent(ABC):
    """基础Agent抽象类"""
    
    def __init__(self, agent_id: str, role: AgentRole, 
                 communication: CommunicationProtocol):
        self.agent_id = agent_id
        self.role = role
        self.communication = communication
        self.memory: List[AgentMemory] = []
        self.action_history: List[AgentAction] = []
        self.performance_metrics: List[EvaluationMetrics] = []
        self.is_active = True
        self.learning_rate = 0.1
        self.temperature = 0.7  # 创造性参数
        
        # 新增的进化参数
        self.adaptation_speed = 0.1      # 适应速度
        self.exploration_rate = 0.3      # 探索率
        self.confidence_threshold = 0.7  # 置信度阈值
        self.error_tolerance = 0.2       # 错误容忍度
        self.collaboration_preference = 0.8  # 协作偏好
        
        # 高级记忆和学习机制
        self.knowledge_graph = {}        # 知识图谱
        self.pattern_library = []        # 模式库
        self.error_history = []          # 错误历史
        self.success_patterns = []       # 成功模式
        self.communication_history = []  # 通信历史
        
        # 自我优化机制
        self.optimization_counter = 0
        self.last_optimization_time = datetime.now()
        self.optimization_interval = 10  # 每10次行动进行一次优化
        
        # 订阅相关主题
        self.communication.subscribe("global_broadcast", self.agent_id)
        self.communication.subscribe(f"agent_{self.agent_id}", self.agent_id)
        self.communication.subscribe(f"role_{self.role.value}", self.agent_id)
    
    @abstractmethod
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """思考 - 生成行动计划"""
        pass
    
    @abstractmethod
    async def act(self, plan: Dict[str, Any]) -> AgentAction:
        """行动 - 执行计划"""
        pass
    
    @abstractmethod
    async def observe(self, action_result: AgentAction) -> Dict[str, Any]:
        """观察 - 分析行动结果"""
        pass
    
    async def react_cycle(self, initial_context: Dict[str, Any]) -> List[AgentAction]:
        """ReAct循环 - 思考-行动-观察"""
        actions = []
        context = initial_context.copy()
        max_iterations = 10
        
        for iteration in range(max_iterations):
            try:
                # 检查消息
                messages = self.communication.get_messages(self.agent_id)
                if messages:
                    context['messages'] = messages
                
                # 思考
                plan = await self.think(context)
                if not plan or plan.get('stop', False):
                    break
                
                # 行动
                action = await self.act(plan)
                actions.append(action)
                
                # 观察
                observation = await self.observe(action)
                context.update(observation)
                
                # 学习
                await self.learn_from_action(action, observation)
                
                # 如果任务完成则退出
                if observation.get('task_completed', False):
                    break
                    
            except Exception as e:
                logger.error(f"Agent {self.agent_id} ReAct cycle error: {e}")
                break
        
        return actions
    
    async def learn_from_action(self, action: AgentAction, observation: Dict[str, Any]):
        """从行动中学习"""
        # 评估行动效果
        success_score = observation.get('success_score', 0.5)
        
        # 更新记忆
        memory = AgentMemory(
            content={
                'action': action,
                'observation': observation,
                'success_score': success_score
            },
            memory_type='action_result',
            importance=success_score,
            success_rate=success_score
        )
        self.memory.append(memory)
        
        # 保持记忆数量限制
        if len(self.memory) > MAX_MEMORY_SIZE:
            # 移除重要性最低的记忆
            self.memory.sort(key=lambda m: m.importance)
            self.memory = self.memory[20:]
    
    def get_relevant_memories(self, context: Dict[str, Any], limit: int = 5) -> List[AgentMemory]:
        """获取相关记忆"""
        # 简单的相关性匹配 - 实际实现可以使用更复杂的相似度计算
        relevant_memories = []
        
        for memory in self.memory:
            if memory.memory_type == 'action_result':
                memory.access_count += 1
                relevant_memories.append(memory)
        
        # 按重要性和成功率排序
        relevant_memories.sort(
            key=lambda m: m.importance * m.success_rate, 
            reverse=True
        )
        
        return relevant_memories[:limit]
    
    def add_memory(self, content: Any, importance: float = 0.5) -> None:
        """添加记忆到Agent的记忆系统"""
        memory = AgentMemory(
            content=content,
            importance=importance,
            memory_type="manual_add",
            timestamp=datetime.now()
        )
        self.memory.append(memory)
        
        # 保持记忆数量限制
        if len(self.memory) > MAX_MEMORY_SIZE:
            self.memory.sort(key=lambda m: m.importance)
            self.memory = self.memory[20:]  # 保留重要性最高的记忆
    
    def learn_from_success(self, action: AgentAction) -> None:
        """从成功行动中学习"""
        if action.success and action.metadata.get('output_score', 0) > 0.7:
            success_pattern = {
                'action_type': action.action_type,
                'content_signature': str(hash(str(action.content))),
                'parameters': {
                    'temperature': self.temperature,
                    'learning_rate': self.learning_rate,
                    'exploration_rate': self.exploration_rate,
                    'adaptation_speed': self.adaptation_speed
                },
                'success_score': action.metadata.get('output_score', 0),
                'timestamp': datetime.now()
            }
            self.success_patterns.append(success_pattern)
            
            # 保持成功模式数量限制
            if len(self.success_patterns) > SUCCESS_PATTERN_LIMIT:
                self.success_patterns = self.success_patterns[-15:]
    
    def record_performance(self, metrics: Any) -> None:
        """记录性能指标"""
        performance_record = {
            'timestamp': datetime.now(),
            'metrics': metrics,
            'agent_state': {
                'memory_size': len(self.memory),
                'action_count': len(self.action_history),
                'optimization_count': self.optimization_counter,
                'success_patterns': len(self.success_patterns)
            }
        }
        
        # 如果metrics是EvaluationMetrics对象，添加到performance_metrics列表
        if isinstance(metrics, EvaluationMetrics):
            self.performance_metrics.append(metrics)
        
        # 保持性能历史记录数量限制
        if len(self.performance_metrics) > PERFORMANCE_HISTORY_SIZE:
            self.performance_metrics = self.performance_metrics[-30:]
    
    async def self_evaluate(self) -> EvaluationMetrics:
        """高级自我评估 - 全面评估Agent性能"""
        if not self.action_history:
            return EvaluationMetrics()
        
        # 计算各项指标
        recent_actions = self.action_history[-10:]  # 最近10个行动
        
        # 模拟梯度计算
        gradients = [np.random.normal(0, 1) for _ in range(len(recent_actions))]
        
        # 模拟输出对比
        original_output = [a.metadata.get('output_score', 0.5) for a in recent_actions]
        noisy_output = [o + np.random.normal(0, 0.1) for o in original_output]
        
        # 计算基础指标
        evaluator = AdvancedEvaluator()
        st = evaluator.calculate_trainability(gradients)
        sg = evaluator.calculate_generalization(original_output, noisy_output)
        se = evaluator.calculate_expressiveness(
            np.mean(original_output), 
            [abs(g) for g in gradients]
        )
        composite = evaluator.calculate_composite_score(st, sg, se)
        
        # 计算高级指标
        action_patterns = [{'type': a.action_type.value, 'success': a.success, 
                           'duration': a.metadata.get('execution_time', 0)} 
                          for a in recent_actions]
        creativity = evaluator.calculate_creativity_score(action_patterns)
        
        performance_history = [m.composite_score for m in self.performance_metrics[-10:]]
        adaptation_rate = evaluator.calculate_adaptation_rate(performance_history)
        
        collaboration_efficiency = evaluator.calculate_collaboration_efficiency(
            self.communication_history[-20:])
        
        error_recovery = evaluator.calculate_error_recovery_rate(self.error_history)
        
        # 计算知识保持率
        knowledge_retention = self.calculate_knowledge_retention()
        
        # 计算创新指数
        innovation_index = self.calculate_innovation_index()
        
        metrics = EvaluationMetrics(
            trainability=st,
            generalization=sg,
            expressiveness=se,
            composite_score=composite,
            execution_time=sum(a.metadata.get('execution_time', 0) for a in recent_actions),
            resource_usage={'memory': len(self.memory), 'actions': len(self.action_history)},
            creativity_score=creativity,
            adaptation_rate=adaptation_rate,
            collaboration_efficiency=collaboration_efficiency,
            error_recovery_rate=error_recovery,
            knowledge_retention=knowledge_retention,
            innovation_index=innovation_index
        )
        
        self.performance_metrics.append(metrics)
        return metrics
    
    def calculate_knowledge_retention(self) -> float:
        """计算知识保持率 - 基于记忆访问频率和成功率"""
        if not self.memory:
            return 0.5
        
        # 计算高价值记忆的保持率
        high_value_memories = [m for m in self.memory if m.importance > 0.7]
        if not high_value_memories:
            return 0.3
        
        # 基于访问频率和成功率计算保持率
        total_retention = 0
        for memory in high_value_memories:
            # 访问频率越高，成功率越高，保持率越高
            access_factor = min(memory.access_count / 10.0, 1.0)
            success_factor = memory.success_rate
            retention = (access_factor * 0.4 + success_factor * 0.6)
            total_retention += retention
        
        return total_retention / len(high_value_memories)
    
    def calculate_innovation_index(self) -> float:
        """计算创新指数 - 基于新模式的发现和应用"""
        if len(self.action_history) < 5:
            return 0.5
        
        # 分析最近的行动模式
        recent_actions = self.action_history[-10:]
        action_types = [a.action_type.value for a in recent_actions]
        
        # 计算行动类型的多样性
        unique_types = set(action_types)
        diversity_score = len(unique_types) / len(action_types)
        
        # 计算新模式的出现频率
        pattern_signatures = []
        for i in range(len(recent_actions) - 2):
            pattern = tuple(action_types[i:i+3])  # 3个连续行动的模式
            pattern_signatures.append(pattern)
        
        unique_patterns = set(pattern_signatures)
        pattern_diversity = len(unique_patterns) / max(len(pattern_signatures), 1)
        
        # 检查是否有突破性的成功
        breakthrough_actions = [a for a in recent_actions 
                               if a.metadata.get('output_score', 0) > 0.9]
        breakthrough_rate = len(breakthrough_actions) / len(recent_actions)
        
        # 综合创新指数
        innovation_index = (diversity_score * 0.4 + 
                          pattern_diversity * 0.4 + 
                          breakthrough_rate * 0.2)
        
        return min(innovation_index, 1.0)
    
    async def self_improve(self, metrics: EvaluationMetrics):
        """智能自我改进 - 基于多维度指标的全面优化"""
        old_params = {
            'temperature': self.temperature,
            'learning_rate': self.learning_rate,
            'exploration_rate': self.exploration_rate,
            'adaptation_speed': self.adaptation_speed,
            'confidence_threshold': self.confidence_threshold
        }
        
        # 多维度自适应调整
        if metrics.composite_score < 0.3:
            # 性能较差，增加探索和适应性
            self.temperature = min(1.0, self.temperature + 0.1)
            self.learning_rate = min(0.5, self.learning_rate + 0.05)
            self.exploration_rate = min(0.8, self.exploration_rate + 0.1)
            self.adaptation_speed = min(0.5, self.adaptation_speed + 0.05)
            
        elif metrics.composite_score > 0.7:
            # 性能良好，减少探索，增加利用
            self.temperature = max(0.1, self.temperature - 0.05)
            self.learning_rate = max(0.01, self.learning_rate - 0.01)
            self.exploration_rate = max(0.1, self.exploration_rate - 0.05)
        
        # 基于具体指标进行细化调整
        if metrics.creativity_score < 0.4:
            # 创造性不足，增加创新性
            self.temperature = min(1.0, self.temperature + 0.05)
            self.exploration_rate = min(0.8, self.exploration_rate + 0.1)
        
        if metrics.adaptation_rate < 0.3:
            # 适应速度慢，提高适应参数
            self.adaptation_speed = min(0.5, self.adaptation_speed + 0.1)
            self.learning_rate = min(0.3, self.learning_rate + 0.02)
        
        if metrics.collaboration_efficiency < 0.5:
            # 协作效率低，调整协作偏好
            self.collaboration_preference = min(1.0, self.collaboration_preference + 0.1)
        
        if metrics.error_recovery_rate < 0.6:
            # 错误恢复能力差，降低错误容忍度
            self.error_tolerance = max(0.05, self.error_tolerance - 0.05)
        
        # 更新优化计数器
        self.optimization_counter += 1
        self.last_optimization_time = datetime.now()
        
        # 学习成功模式
        if metrics.composite_score > 0.7:
            success_pattern = {
                'timestamp': datetime.now(),
                'parameters': old_params.copy(),
                'metrics': metrics,
                'context': {
                    'recent_actions': len(self.action_history[-5:]),
                    'memory_size': len(self.memory)
                }
            }
            self.success_patterns.append(success_pattern)
            
            # 保持成功模式数量限制
            if len(self.success_patterns) > SUCCESS_PATTERN_LIMIT:
                self.success_patterns = self.success_patterns[-15:]
        
        # 记录改进行动
        improvement_action = AgentAction(
            agent_id=self.agent_id,
            action_type=ActionType.SELF_MODIFY,
            content={
                'old_parameters': old_params,
                'new_parameters': {
                    'temperature': self.temperature,
                    'learning_rate': self.learning_rate,
                    'exploration_rate': self.exploration_rate,
                    'adaptation_speed': self.adaptation_speed,
                    'confidence_threshold': self.confidence_threshold
                },
                'metrics': metrics,
                'optimization_cycle': self.optimization_counter
            },
            metadata={
                'improvement_type': 'multi_dimensional_optimization',
                'improvement_score': metrics.composite_score
            }
        )
        self.action_history.append(improvement_action)
        
        # 应用成功模式（如果性能持续下降）
        if len(self.performance_metrics) >= 3:
            recent_scores = [m.composite_score for m in self.performance_metrics[-3:]]
            if all(recent_scores[i] > recent_scores[i+1] for i in range(len(recent_scores)-1)):
                # 性能连续下降，尝试应用历史成功模式
                await self.apply_successful_pattern()
        
        logger.info(f"Agent {self.agent_id} self-improved (cycle {self.optimization_counter}): "
                   f"score={metrics.composite_score:.3f}, "
                   f"temp={self.temperature:.2f}, "
                   f"lr={self.learning_rate:.3f}, "
                   f"exploration={self.exploration_rate:.2f}")
    
    async def apply_successful_pattern(self):
        """应用历史成功模式"""
        if not self.success_patterns:
            return
        
        # 选择最佳成功模式（基于当时的性能指标）
        best_pattern = max(self.success_patterns, 
                          key=lambda p: p['metrics'].composite_score)
        
        # 应用成功模式的参数（带有一定随机性避免过拟合）
        noise_factor = 0.1
        params = best_pattern['parameters']
        
        self.temperature = max(0.1, min(1.0, 
            params['temperature'] + np.uniform(-noise_factor, noise_factor)))
        self.learning_rate = max(0.01, min(0.5,
            params['learning_rate'] + np.uniform(-noise_factor/10, noise_factor/10)))
        self.exploration_rate = max(0.1, min(0.8,
            params['exploration_rate'] + np.uniform(-noise_factor, noise_factor)))
        
        logger.info(f"Agent {self.agent_id} applied successful pattern from "
                   f"{best_pattern['timestamp']} with score {best_pattern['metrics'].composite_score:.3f}")


class ResearcherAgent(BaseAgent):
    """研究者Agent - 负责信息收集与分析"""
    
    def __init__(self, agent_id: str, communication: CommunicationProtocol):
        super().__init__(agent_id, AgentRole.RESEARCHER, communication)
        self.knowledge_base = {}
    
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """研究者思考过程"""
        goal = context.get('goal', 'general_research')
        
        # 检查是否有相关知识
        relevant_memories = self.get_relevant_memories(context)
        
        plan = {
            'action_type': 'research',
            'target': goal,
            'methods': ['web_search', 'knowledge_lookup', 'analysis'],
            'use_memories': len(relevant_memories) > 0
        }
        
        return plan
    
    async def act(self, plan: Dict[str, Any]) -> AgentAction:
        """执行研究任务"""
        start_time = time.time()
        
        # 模拟研究过程
        research_result = {
            'findings': f"Research findings for {plan['target']}",
            'confidence': np.random.uniform(0.6, 0.9),
            'sources': ['source1', 'source2', 'source3']
        }
        
        # 广播研究结果
        self.communication.publish(
            'research_results',
            research_result,
            self.agent_id
        )
        
        action = AgentAction(
            agent_id=self.agent_id,
            action_type=ActionType.EXECUTE,
            content=research_result,
            metadata={
                'execution_time': time.time() - start_time,
                'output_score': research_result['confidence']
            }
        )
        
        self.action_history.append(action)
        return action
    
    async def observe(self, action_result: AgentAction) -> Dict[str, Any]:
        """观察研究结果"""
        success_score = action_result.content.get('confidence', 0.5)
        
        observation = {
            'success_score': success_score,
            'task_completed': success_score > 0.8,
            'next_action': 'refine_research' if success_score < 0.7 else 'complete'
        }
        
        return observation


class ExecutorAgent(BaseAgent):
    """执行者Agent - 负责任务执行"""
    
    def __init__(self, agent_id: str, communication: CommunicationProtocol):
        super().__init__(agent_id, AgentRole.EXECUTOR, communication)
        self.execution_capabilities = ['code_generation', 'task_automation', 'system_interaction']
    
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行者思考过程"""
        task = context.get('task', 'general_execution')
        
        plan = {
            'action_type': 'execute',
            'task': task,
            'approach': 'step_by_step',
            'estimated_difficulty': np.random.uniform(0.3, 0.8)
        }
        
        return plan
    
    async def act(self, plan: Dict[str, Any]) -> AgentAction:
        """执行任务"""
        start_time = time.time()
        
        # 模拟任务执行
        difficulty = plan['estimated_difficulty']
        success_probability = max(0.1, 1.0 - difficulty)
        
        execution_result = {
            'task': plan['task'],
            'success': np.random.random() < success_probability,
            'output': f"Execution result for {plan['task']}",
            'efficiency': np.random.uniform(0.5, 1.0)
        }
        
        action = AgentAction(
            agent_id=self.agent_id,
            action_type=ActionType.EXECUTE,
            content=execution_result,
            success=execution_result['success'],
            metadata={
                'execution_time': time.time() - start_time,
                'output_score': execution_result['efficiency']
            }
        )
        
        self.action_history.append(action)
        return action
    
    async def observe(self, action_result: AgentAction) -> Dict[str, Any]:
        """观察执行结果"""
        success_score = 1.0 if action_result.success else 0.2
        
        observation = {
            'success_score': success_score,
            'task_completed': action_result.success,
            'next_action': 'complete' if action_result.success else 'retry'
        }
        
        return observation


class CriticAgent(BaseAgent):
    """评判者Agent - 负责性能评估和质量控制"""
    
    def __init__(self, agent_id: str, communication: CommunicationProtocol):
        super().__init__(agent_id, AgentRole.CRITIC, communication)
        self.evaluation_criteria = ['accuracy', 'efficiency', 'completeness', 'creativity']
    
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """评判者思考过程"""
        target_actions = context.get('actions_to_evaluate', [])
        
        plan = {
            'action_type': 'evaluate',
            'targets': target_actions,
            'criteria': self.evaluation_criteria,
            'evaluation_depth': 'comprehensive'
        }
        
        return plan
    
    async def act(self, plan: Dict[str, Any]) -> AgentAction:
        """执行评估"""
        start_time = time.time()
        
        evaluations = []
        for target in plan['targets']:
            evaluation = {
                'target': target,
                'scores': {criterion: np.random.uniform(0.3, 1.0) 
                          for criterion in self.evaluation_criteria},
                'overall_score': np.random.uniform(0.4, 0.9),
                'recommendations': [f"Improve {criterion}" 
                                  for criterion in self.evaluation_criteria[:2]]
            }
            evaluations.append(evaluation)
        
        # 广播评估结果
        self.communication.publish(
            'evaluation_results',
            {'evaluations': evaluations},
            self.agent_id
        )
        
        action = AgentAction(
            agent_id=self.agent_id,
            action_type=ActionType.EXECUTE,
            content={'evaluations': evaluations},
            metadata={
                'execution_time': time.time() - start_time,
                'output_score': np.mean([e['overall_score'] for e in evaluations])
            }
        )
        
        self.action_history.append(action)
        return action
    
    async def observe(self, action_result: AgentAction) -> Dict[str, Any]:
        """观察评估结果"""
        evaluations = action_result.content.get('evaluations', [])
        average_score = np.mean([e['overall_score'] for e in evaluations]) if evaluations else 0.5
        
        observation = {
            'success_score': average_score,
            'task_completed': True,
            'insights': f"Evaluated {len(evaluations)} items with average score {average_score:.2f}"
        }
        
        return observation


class ArchitectAgent(BaseAgent):
    """架构师Agent - 负责系统架构设计和自我重构"""
    
    def __init__(self, agent_id: str, communication: CommunicationProtocol):
        super().__init__(agent_id, AgentRole.ARCHITECT, communication)
        self.system_blueprints = []
        self.architecture_patterns = {
            'hierarchical': {'efficiency': 0.8, 'scalability': 0.6},
            'mesh': {'efficiency': 0.6, 'scalability': 0.9},
            'pipeline': {'efficiency': 0.9, 'scalability': 0.5},
            'hybrid': {'efficiency': 0.7, 'scalability': 0.8}
        }
        self.optimization_history = []
    
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """架构师思考过程 - 分析系统状态并规划改进"""
        system_metrics = context.get('system_metrics', {})
        agent_count = context.get('agent_count', 0)
        
        # 分析当前系统性能
        current_efficiency = system_metrics.get('composite_score', 0.5)
        bottlenecks = self.identify_bottlenecks(system_metrics)
        
        plan = {
            'action_type': 'architect_analysis',
            'current_efficiency': current_efficiency,
            'bottlenecks': bottlenecks,
            'recommended_pattern': self.recommend_architecture_pattern(system_metrics),
            'optimization_priority': 'high' if current_efficiency < 0.4 else 'medium'
        }
        
        return plan
    
    def identify_bottlenecks(self, metrics: Dict) -> List[str]:
        """识别系统瓶颈"""
        bottlenecks = []
        
        if metrics.get('collaboration_efficiency', 0.5) < 0.4:
            bottlenecks.append('communication_overhead')
        
        if metrics.get('adaptation_rate', 0.5) < 0.3:
            bottlenecks.append('learning_inefficiency')
        
        if metrics.get('error_recovery_rate', 0.5) < 0.5:
            bottlenecks.append('error_handling')
        
        return bottlenecks
    
    def recommend_architecture_pattern(self, metrics: Dict) -> str:
        """推荐架构模式"""
        agent_count = metrics.get('agent_count', 4)
        
        if agent_count <= 3:
            return 'hierarchical'
        elif agent_count <= 6:
            return 'hybrid'
        else:
            return 'mesh'
    
    async def act(self, plan: Dict[str, Any]) -> AgentAction:
        """执行架构分析和优化建议"""
        start_time = time.time()
        
        # 生成架构优化方案
        optimization_proposal = {
            'current_analysis': plan,
            'proposed_changes': self.generate_optimization_proposals(plan),
            'expected_improvement': np.random.uniform(0.1, 0.3),
            'implementation_complexity': np.random.uniform(0.3, 0.8)
        }
        
        # 广播架构建议
        self.communication.publish(
            'architecture_proposal',
            optimization_proposal,
            self.agent_id
        )
        
        action = AgentAction(
            agent_id=self.agent_id,
            action_type=ActionType.EXECUTE,
            content=optimization_proposal,
            metadata={
                'execution_time': time.time() - start_time,
                'output_score': optimization_proposal['expected_improvement']
            }
        )
        
        self.action_history.append(action)
        return action
    
    def generate_optimization_proposals(self, plan: Dict) -> List[Dict]:
        """生成优化建议"""
        proposals = []
        
        for bottleneck in plan.get('bottlenecks', []):
            if bottleneck == 'communication_overhead':
                proposals.append({
                    'type': 'communication_optimization',
                    'description': 'Implement message batching and priority queues',
                    'expected_impact': 0.2
                })
            elif bottleneck == 'learning_inefficiency':
                proposals.append({
                    'type': 'learning_enhancement',
                    'description': 'Add meta-learning capabilities and knowledge distillation',
                    'expected_impact': 0.25
                })
            elif bottleneck == 'error_handling':
                proposals.append({
                    'type': 'resilience_improvement',
                    'description': 'Implement circuit breakers and graceful degradation',
                    'expected_impact': 0.15
                })
        
        return proposals
    
    async def observe(self, action_result: AgentAction) -> Dict[str, Any]:
        """观察架构优化结果"""
        proposal = action_result.content
        success_score = proposal.get('expected_improvement', 0.1)
        
        observation = {
            'success_score': success_score,
            'task_completed': True,
            'architectural_insights': f"Proposed {len(proposal.get('proposed_changes', []))} optimizations"
        }
        
        return observation


class CoordinatorAgent(BaseAgent):
    """协调者Agent - 负责任务分配和协调"""
    
    def __init__(self, agent_id: str, communication: CommunicationProtocol):
        super().__init__(agent_id, AgentRole.COORDINATOR, communication)
        self.managed_agents = []
        self.task_queue = []
    
    def register_agent(self, agent: BaseAgent):
        """注册被管理的Agent"""
        self.managed_agents.append(agent)
    
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """协调者思考过程"""
        goal = context.get('goal', 'coordinate_tasks')
        available_agents = [agent for agent in self.managed_agents if agent.is_active]
        
        plan = {
            'action_type': 'coordinate',
            'goal': goal,
            'available_agents': len(available_agents),
            'coordination_strategy': 'balanced_allocation'
        }
        
        return plan
    
    async def act(self, plan: Dict[str, Any]) -> AgentAction:
        """执行协调任务"""
        start_time = time.time()
        
        # 分配任务给不同角色的Agent
        task_assignments = {
            'research_tasks': [agent for agent in self.managed_agents 
                             if agent.role == AgentRole.RESEARCHER],
            'execution_tasks': [agent for agent in self.managed_agents 
                              if agent.role == AgentRole.EXECUTOR],
            'evaluation_tasks': [agent for agent in self.managed_agents 
                               if agent.role == AgentRole.CRITIC]
        }
        
        coordination_result = {
            'assignments': task_assignments,
            'coordination_efficiency': np.random.uniform(0.6, 0.9)
        }
        
        # 广播协调结果
        self.communication.publish(
            'coordination_update',
            coordination_result,
            self.agent_id
        )
        
        action = AgentAction(
            agent_id=self.agent_id,
            action_type=ActionType.COMMUNICATE,
            content=coordination_result,
            metadata={
                'execution_time': time.time() - start_time,
                'output_score': coordination_result['coordination_efficiency']
            }
        )
        
        self.action_history.append(action)
        return action
    
    async def observe(self, action_result: AgentAction) -> Dict[str, Any]:
        """观察协调结果"""
        efficiency = action_result.content.get('coordination_efficiency', 0.5)
        
        observation = {
            'success_score': efficiency,
            'task_completed': efficiency > 0.7,
            'next_action': 'monitor_progress'
        }
        
        return observation


class AutonomousEvolutionarySystem:
    """自主进化系统 - 管理多个Agent的协作和进化"""
    
    def __init__(self):
        self.communication = CommunicationProtocol()
        self.agents: Dict[str, BaseAgent] = {}
        self.system_metrics: List[Dict[str, Any]] = []
        self.evolution_cycles = 0
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    def add_agent(self, agent: BaseAgent):
        """添加Agent到系统"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Added agent {agent.agent_id} with role {agent.role.value}")
    
    def create_standard_team(self) -> Dict[str, BaseAgent]:
        """创建标准团队 - 包含所有核心角色"""
        team = {}
        
        # 创建各种角色的Agent
        researcher = ResearcherAgent("researcher_001", self.communication)
        executor = ExecutorAgent("executor_001", self.communication)
        critic = CriticAgent("critic_001", self.communication)
        coordinator = CoordinatorAgent("coordinator_001", self.communication)
        architect = ArchitectAgent("architect_001", self.communication)
        
        # 协调者注册其他Agent
        coordinator.register_agent(researcher)
        coordinator.register_agent(executor)
        coordinator.register_agent(critic)
        coordinator.register_agent(architect)
        
        team = {
            'researcher': researcher,
            'executor': executor,
            'critic': critic,
            'coordinator': coordinator,
            'architect': architect
        }
        
        # 添加到系统
        for agent in team.values():
            self.add_agent(agent)
        
        return team
    
    async def run_collaborative_task(self, goal: str, max_cycles: int = 5) -> Dict[str, Any]:
        """运行协作任务"""
        logger.info(f"Starting collaborative task: {goal}")
        
        context = {
            'goal': goal,
            'cycle': 0,
            'system_state': 'active'
        }
        
        all_actions = []
        cycle_results = []
        
        for cycle in range(max_cycles):
            context['cycle'] = cycle
            logger.info(f"Starting cycle {cycle + 1}/{max_cycles}")
            
            # 并行执行多个Agent的ReAct循环
            cycle_actions = []
            tasks = []
            
            for agent in self.agents.values():
                if agent.is_active:
                    task = asyncio.create_task(agent.react_cycle(context.copy()))
                    tasks.append((agent.agent_id, task))
            
            # 等待所有Agent完成
            for agent_id, task in tasks:
                try:
                    actions = await task
                    cycle_actions.extend(actions)
                    logger.info(f"Agent {agent_id} completed with {len(actions)} actions")
                except Exception as e:
                    logger.error(f"Agent {agent_id} failed: {e}")
            
            all_actions.extend(cycle_actions)
            
            # 评估系统性能
            system_metrics = await self.evaluate_system_performance()
            cycle_results.append({
                'cycle': cycle,
                'actions_count': len(cycle_actions),
                'metrics': system_metrics
            })
            
            # 系统自我进化
            await self.evolve_system(system_metrics)
            
            # 检查是否达到目标
            if system_metrics.composite_score > 0.8:
                logger.info(f"Goal achieved in cycle {cycle + 1}")
                break
        
        final_result = {
            'goal': goal,
            'total_cycles': len(cycle_results),
            'total_actions': len(all_actions),
            'final_metrics': cycle_results[-1]['metrics'] if cycle_results else None,
            'cycle_results': cycle_results,
            'evolution_cycles': self.evolution_cycles
        }
        
        logger.info(f"Task completed: {len(all_actions)} actions in {len(cycle_results)} cycles")
        return final_result
    
    async def evaluate_system_performance(self) -> EvaluationMetrics:
        """评估系统整体性能"""
        agent_metrics = []
        
        # 收集所有Agent的评估结果
        for agent in self.agents.values():
            try:
                metrics = await agent.self_evaluate()
                agent_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Failed to evaluate agent {agent.agent_id}: {e}")
        
        if not agent_metrics:
            return EvaluationMetrics()
        
        # 计算系统级指标
        system_metrics = EvaluationMetrics(
            trainability=np.mean([m.trainability for m in agent_metrics]),
            generalization=np.mean([m.generalization for m in agent_metrics]),
            expressiveness=np.mean([m.expressiveness for m in agent_metrics]),
            composite_score=np.mean([m.composite_score for m in agent_metrics]),
            execution_time=sum([m.execution_time for m in agent_metrics]),
            resource_usage={
                'total_memory': sum([m.resource_usage.get('memory', 0) for m in agent_metrics]),
                'total_actions': sum([m.resource_usage.get('actions', 0) for m in agent_metrics])
            }
        )
        
        self.system_metrics.append({
            'timestamp': datetime.now(),
            'metrics': system_metrics,
            'agent_count': len(self.agents)
        })
        
        return system_metrics
    
    async def evolve_system(self, metrics: EvaluationMetrics):
        """系统进化"""
        self.evolution_cycles += 1
        
        # 基于性能指标进化系统
        if metrics.composite_score < 0.4:
            # 性能较差，尝试添加新Agent或调整现有Agent
            await self.adapt_system_low_performance()
        elif metrics.composite_score > 0.8:
            # 性能良好，优化效率
            await self.optimize_system_high_performance()
        
        # 让所有Agent进行自我改进
        for agent in self.agents.values():
            try:
                await agent.self_improve(metrics)
            except Exception as e:
                logger.error(f"Failed to improve agent {agent.agent_id}: {e}")
        
        logger.info(f"System evolution cycle {self.evolution_cycles} completed")
    
    async def adapt_system_low_performance(self):
        """低性能时的系统适应"""
        # 增加探索性，可能添加新的Agent类型
        for agent in self.agents.values():
            agent.temperature = min(1.0, agent.temperature + 0.1)
        
        logger.info("Adapted system for low performance - increased exploration")
    
    async def optimize_system_high_performance(self):
        """高性能时的系统优化"""
        # 减少探索，增加利用
        for agent in self.agents.values():
            agent.temperature = max(0.1, agent.temperature - 0.05)
        
        logger.info("Optimized system for high performance - reduced exploration")
    
    def save_system_state(self, filepath: str):
        """保存系统状态"""
        state = {
            'agents': {aid: {
                'role': agent.role.value,
                'memory': agent.memory,
                'action_history': agent.action_history,
                'performance_metrics': agent.performance_metrics,
                'learning_rate': agent.learning_rate,
                'temperature': agent.temperature
            } for aid, agent in self.agents.items()},
            'system_metrics': self.system_metrics,
            'evolution_cycles': self.evolution_cycles
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"System state saved to {filepath}")
    
    def load_system_state(self, filepath: str):
        """加载系统状态"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.system_metrics = state['system_metrics']
            self.evolution_cycles = state['evolution_cycles']
            
            # 重建Agent
            for aid, agent_data in state['agents'].items():
                if aid in self.agents:
                    agent = self.agents[aid]
                    agent.memory = agent_data['memory']
                    agent.action_history = agent_data['action_history']
                    agent.performance_metrics = agent_data['performance_metrics']
                    agent.learning_rate = agent_data['learning_rate']
                    agent.temperature = agent_data['temperature']
            
            logger.info(f"System state loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load system state: {e}")


# 使用示例和测试
async def demo_autonomous_evolutionary_system():
    """演示自主进化系统"""
    print("🚀 启动自主进化Agent系统演示...")
    
    # 创建系统
    system = AutonomousEvolutionarySystem()
    
    # 创建标准团队
    team = system.create_standard_team()
    
    print(f"✅ 创建了包含 {len(team)} 个Agent的团队：")
    for role, agent in team.items():
        print(f"   - {role}: {agent.agent_id}")
    
    # 运行协作任务
    goals = [
        "研究并实现一个新的机器学习算法",
        "优化现有系统的性能",
        "设计一个自动化测试框架"
    ]
    
    for i, goal in enumerate(goals, 1):
        print(f"\n🎯 任务 {i}: {goal}")
        result = await system.run_collaborative_task(goal, max_cycles=3)
        
        print(f"📊 任务结果:")
        print(f"   - 总循环数: {result['total_cycles']}")
        print(f"   - 总行动数: {result['total_actions']}")
        print(f"   - 进化循环数: {result['evolution_cycles']}")
        
        if result['final_metrics']:
            metrics = result['final_metrics']
            print(f"   - 最终性能: {metrics.composite_score:.3f}")
            print(f"   - 可训练性: {metrics.trainability:.3f}")
            print(f"   - 泛化能力: {metrics.generalization:.3f}")
            print(f"   - 表达能力: {metrics.expressiveness:.3f}")
            print(f"   - 创造性得分: {metrics.creativity_score:.3f}")
            print(f"   - 适应速度: {metrics.adaptation_rate:.3f}")
            print(f"   - 协作效率: {metrics.collaboration_efficiency:.3f}")
            print(f"   - 错误恢复率: {metrics.error_recovery_rate:.3f}")
            print(f"   - 知识保持率: {metrics.knowledge_retention:.3f}")
            print(f"   - 创新指数: {metrics.innovation_index:.3f}")
    
    # 保存系统状态
    system.save_system_state("autonomous_system_state.pkl")
    print("\n💾 系统状态已保存")
    
    # 显示系统进化历史
    print(f"\n📈 系统进化历史 ({len(system.system_metrics)} 个评估点):")
    for i, record in enumerate(system.system_metrics[-5:], 1):  # 显示最近5个评估点
        metrics = record['metrics']
        print(f"   评估 {i}: 综合得分 {metrics.composite_score:.3f}")
    
    # 显示个体Agent的进化信息
    print(f"\n🧠 个体Agent进化报告:")
    for role, agent in team.items():
        print(f"\n   {role.upper()} Agent ({agent.agent_id}):")
        print(f"      - 优化次数: {agent.optimization_counter}")
        print(f"      - 当前温度: {agent.temperature:.3f}")
        print(f"      - 学习率: {agent.learning_rate:.4f}")
        print(f"      - 探索率: {agent.exploration_rate:.3f}")
        print(f"      - 记忆条目: {len(agent.memory)}")
        print(f"      - 行动历史: {len(agent.action_history)}")
        print(f"      - 成功模式: {len(agent.success_patterns)}")
        
        if agent.performance_metrics:
            latest_metrics = agent.performance_metrics[-1]
            print(f"      - 最新综合得分: {latest_metrics.composite_score:.3f}")
            print(f"      - 创新指数: {latest_metrics.innovation_index:.3f}")
    
    # 显示系统架构优化建议
    if 'architect' in team:
        architect = team['architect']
        if architect.action_history:
            latest_action = architect.action_history[-1]
            if latest_action.content and 'proposed_changes' in latest_action.content:
                proposals = latest_action.content['proposed_changes']
                print(f"\n🏗️ 最新架构优化建议 ({len(proposals)} 项):")
                for i, proposal in enumerate(proposals, 1):
                    print(f"   {i}. {proposal.get('type', 'unknown')}: "
                         f"{proposal.get('description', 'no description')}")
                    print(f"      预期影响: {proposal.get('expected_impact', 0):.1%}")
    
    print("\n🎉 自主进化Agent系统演示完成！")
    print("🔬 系统已展示以下核心能力：")
    print("   ✅ ReAct循环 - 思考-行动-观察")
    print("   ✅ 多Agent协作 - 角色专业化分工")
    print("   ✅ 自我评估 - 多维度性能指标")
    print("   ✅ 自主进化 - 参数自适应调整")
    print("   ✅ 记忆学习 - 成功模式识别")
    print("   ✅ 架构优化 - 系统瓶颈分析")
    print("   ✅ 错误恢复 - 历史模式应用")
    print("   ✅ 知识保持 - 长期记忆管理")


if __name__ == "__main__":
    # 运行演示
    asyncio.run(demo_autonomous_evolutionary_system())