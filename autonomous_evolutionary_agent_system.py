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
import numpy as np
from datetime import datetime
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


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


class TrainingFreeEvaluator:
    """训练无关评估器 - 基于最新研究的多指标评估"""
    
    @staticmethod
    def calculate_trainability(gradients: List[float], dataset_size: int = 1000, 
                             lipschitz_constant: float = 1.0, batch_size: int = 32) -> float:
        """计算可训练性指标 (ST)"""
        if not gradients:
            return 0.0
        
        gradient_norm_squared = sum(g**2 for g in gradients)
        st = (dataset_size / (lipschitz_constant * batch_size)) * gradient_norm_squared
        return min(st / 1000.0, 1.0)  # 归一化到[0,1]
    
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
        if len(self.memory) > 1000:
            # 移除重要性最低的记忆
            self.memory.sort(key=lambda m: m.importance)
            self.memory = self.memory[100:]
    
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
    
    async def self_evaluate(self) -> EvaluationMetrics:
        """自我评估"""
        if not self.action_history:
            return EvaluationMetrics()
        
        # 计算各项指标
        recent_actions = self.action_history[-10:]  # 最近10个行动
        
        # 模拟梯度计算
        gradients = [np.random.normal(0, 1) for _ in range(len(recent_actions))]
        
        # 模拟输出对比
        original_output = [a.metadata.get('output_score', 0.5) for a in recent_actions]
        noisy_output = [o + np.random.normal(0, 0.1) for o in original_output]
        
        # 计算指标
        evaluator = TrainingFreeEvaluator()
        st = evaluator.calculate_trainability(gradients)
        sg = evaluator.calculate_generalization(original_output, noisy_output)
        se = evaluator.calculate_expressiveness(
            np.mean(original_output), 
            [abs(g) for g in gradients]
        )
        composite = evaluator.calculate_composite_score(st, sg, se)
        
        metrics = EvaluationMetrics(
            trainability=st,
            generalization=sg,
            expressiveness=se,
            composite_score=composite,
            execution_time=sum(a.metadata.get('execution_time', 0) for a in recent_actions),
            resource_usage={'memory': len(self.memory), 'actions': len(self.action_history)}
        )
        
        self.performance_metrics.append(metrics)
        return metrics
    
    async def self_improve(self, metrics: EvaluationMetrics):
        """自我改进"""
        # 基于评估结果调整参数
        if metrics.composite_score < 0.3:
            # 性能较差，增加探索性
            self.temperature = min(1.0, self.temperature + 0.1)
            self.learning_rate = min(0.5, self.learning_rate + 0.05)
        elif metrics.composite_score > 0.7:
            # 性能良好，减少探索，增加利用
            self.temperature = max(0.1, self.temperature - 0.05)
            self.learning_rate = max(0.01, self.learning_rate - 0.01)
        
        # 记录改进行动
        improvement_action = AgentAction(
            agent_id=self.agent_id,
            action_type=ActionType.SELF_MODIFY,
            content={
                'old_temperature': self.temperature,
                'old_learning_rate': self.learning_rate,
                'metrics': metrics
            },
            metadata={'improvement_type': 'parameter_adjustment'}
        )
        self.action_history.append(improvement_action)
        
        logger.info(f"Agent {self.agent_id} self-improved: temp={self.temperature:.2f}, lr={self.learning_rate:.3f}")


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
        """创建标准团队"""
        team = {}
        
        # 创建各种角色的Agent
        researcher = ResearcherAgent("researcher_001", self.communication)
        executor = ExecutorAgent("executor_001", self.communication)
        critic = CriticAgent("critic_001", self.communication)
        coordinator = CoordinatorAgent("coordinator_001", self.communication)
        
        # 协调者注册其他Agent
        coordinator.register_agent(researcher)
        coordinator.register_agent(executor)
        coordinator.register_agent(critic)
        
        team = {
            'researcher': researcher,
            'executor': executor,
            'critic': critic,
            'coordinator': coordinator
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
    
    # 保存系统状态
    system.save_system_state("autonomous_system_state.pkl")
    print("\n💾 系统状态已保存")
    
    # 显示系统进化历史
    print(f"\n📈 系统进化历史 ({len(system.system_metrics)} 个评估点):")
    for i, record in enumerate(system.system_metrics[-5:], 1):  # 显示最近5个评估点
        metrics = record['metrics']
        print(f"   评估 {i}: 综合得分 {metrics.composite_score:.3f}")
    
    print("\n🎉 演示完成！")


if __name__ == "__main__":
    # 运行演示
    asyncio.run(demo_autonomous_evolutionary_system())